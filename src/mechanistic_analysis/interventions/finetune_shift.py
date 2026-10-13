"""What fine-tuning changes where the answer starts, and whether that change is one direction.

The fine-tuned model is the untuned model with a LoRA adapter, so both are run
on the same prompts in one process (the adapter is switched off for the untuned
passes). Two prompts are used, neither of which states a norm: ``base`` (no
country) and ``country_only``.

**Phase 1, observation.** For every record and both prompts, under both models:
the decision margin (``scoring.py``) and the state of the last prompt token
after each block in ``--layers``. The *shift* of a record at a layer is
``state(tuned) - state(untuned)``.

**Phase 2, intervention.** If fine-tuning mostly moves that state along one
direction that does not depend on the record, then adding the mean shift to the
untuned model should reproduce the tuned model's margin, and taking it away
from the tuned model should undo it. For each layer, with the mean shift
estimated on the other half of the records (two folds, so a record never
contributes to the vector it is steered with):

* ``steer``    untuned model, the mean shift added after that block;
* ``country``  untuned model, the mean shift of the record's own country added
  (the overall mean where the other fold has fewer than ``--min-country``
  records of it). If this does no better than ``steer``, the shift carries
  nothing specific to the country;
* ``random``   untuned model, a random direction of the same length added;
* ``ablate``   tuned model, the mean shift subtracted.

Vectors are added at the last prompt token and every later position. The
margin is a fast proxy; what the models write is in the judged answers.

PYTHONPATH=src python -m mechanistic_analysis.interventions.finetune_shift \
  --model /path/to/model --model-slug llama-3.1-8b --adapter runs/<name>/best \
  --records output/finetune/mechanism/test_all.json --sample 3000 \
  --output-root output/finetune/mechanism/<name>/finetune_shift
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from typing import Any, Iterator

import numpy as np

from mechanistic_analysis.activations.backend import require_torch
from mechanistic_analysis.activations.cache import sha256_file
from mechanistic_analysis.interventions.common import (
    add_model_arguments, encode_prompt, load, sample_records, thinking_flag,
)
from mechanistic_analysis.interventions.erasure import _first_tensor
from mechanistic_analysis.interventions.scoring import BatchScorer

PROMPTS = ("base", "country_only")
ARMS = ("steer", "country", "random", "ablate")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--adapter", required=True, help="The LoRA adapter of the fine-tuned model.")
    parser.add_argument("--layers", type=int, nargs="+",
                        help="Blocks to read and intervene after; eight evenly spaced when omitted.")
    parser.add_argument("--min-country", type=int, default=3)
    return parser.parse_args(argv)


def attach(loaded: Any, adapter: str) -> tuple[Any, Any]:
    """The model with the adapter on, and a context manager that switches it off."""

    from mechanistic_analysis.activations.backend import attach_adapter, require_adapter_weights

    model = attach_adapter(loaded.model, adapter)
    print(f"adapter matrices loaded: {require_adapter_weights(model, adapter)}", flush=True)
    model.eval()
    return model, model.disable_adapter


@contextlib.contextmanager
def nothing() -> Iterator[None]:
    yield


@contextlib.contextmanager
def captured(decoder_layers: Any, layers: list[int], position: Any) -> Iterator[dict[int, Any]]:
    """Inside the block, ``found[layer]`` is that block's output at each row's ``position``."""

    found: dict[int, Any] = {}
    handles = []
    for layer in layers:
        def hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
            state = _first_tensor(output)
            rows = require_torch().arange(state.shape[0], device=state.device)
            found[layer] = state[rows, position.to(state.device)].float().cpu()
        handles.append(decoder_layers[layer].register_forward_hook(hook))
    try:
        yield found
    finally:
        for handle in handles:
            handle.remove()


@contextlib.contextmanager
def added(decoder_layers: Any, layer: int, start: Any, vectors: Any) -> Iterator[None]:
    """Inside the block, ``vectors[row]`` is added to block ``layer``'s output
    at every position from ``start[row]`` on."""

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        state = _first_tensor(output)
        torch = require_torch()
        after = torch.arange(state.shape[1], device=state.device)[None, :] >= start.to(state.device)[:, None]
        state += after[:, :, None] * vectors.to(device=state.device, dtype=state.dtype)[:, None, :]

    handle = decoder_layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def mean_shifts(shift: np.ndarray, country: np.ndarray, fold: np.ndarray, minimum: int
                ) -> tuple[np.ndarray, np.ndarray]:
    """Per record, the mean shift of the *other* fold: overall, and of the
    record's country (the overall one where that country has under ``minimum``
    records there). ``shift`` is ``(records, layers, width)``."""

    overall, by_country = np.zeros_like(shift), np.zeros_like(shift)
    for held in np.unique(fold):
        inside, outside = fold == held, fold != held
        mean = shift[outside].mean(axis=0)
        overall[inside] = mean
        for code in np.unique(country[inside]):
            rows = inside & (country == code)
            source = outside & (country == code)
            by_country[rows] = shift[source].mean(axis=0) if source.sum() >= minimum else mean
    return overall, by_country


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    started = time.time()
    records = sample_records(args.records, args.sample, args.seed)
    loaded = load(args)
    model, untuned = attach(loaded, args.adapter)
    thinking = thinking_flag(args)
    tokenizer = loaded.tokenizer
    n_layers = len(loaded.decoder_layers)
    layers = sorted(set(args.layers)) if args.layers else list(range(max(1, n_layers // 8), n_layers,
                                                                     max(1, n_layers // 8)))
    count = len(records)
    country = np.array([str(record["norm_id"]).split("__")[0] for record in records])
    encoded = {name: [encode_prompt(loaded, record, name, thinking)[0] for record in records]
               for name in PROMPTS}
    answers = [[tokenizer(str(record["options"][key]), add_special_tokens=False)["input_ids"]
                for key in ("A", "B")] for record in records]
    order = np.argsort([len(ids) for ids in encoded["country_only"]], kind="stable")
    batches = [order[start:start + args.batch_size] for start in range(0, count, args.batch_size)]
    print(f"model={args.model_slug} adapter={args.adapter} sample={count} layers={layers}", flush=True)

    def scorer_for(name: str, rows: np.ndarray) -> BatchScorer:
        return BatchScorer(model, tokenizer.pad_token_id, [encoded[name][row] for row in rows],
                           [answers[row] for row in rows])

    # ---- phase 1: margins and states under both models
    scores = {(who, name): torch.zeros(count, 2) for who in ("untuned", "tuned") for name in PROMPTS}
    # float32: some models' states (Gemma) exceed the range of float16
    states = {key: np.zeros((count, len(layers), loaded.hidden_size), dtype=np.float32) for key in scores}
    for number, rows in enumerate(batches):
        for name in PROMPTS:
            scorer = scorer_for(name, rows)
            for who, switch in (("untuned", untuned), ("tuned", nothing)):
                with switch(), captured(loaded.decoder_layers, layers, scorer.position) as found:
                    scores[(who, name)][rows] = scorer.score()
                for slot, layer in enumerate(layers):
                    states[(who, name)][rows, slot] = found[layer][0::2].numpy()
        if number % 10 == 9 or number == len(batches) - 1:
            print(f"observed {min((number + 1) * args.batch_size, count)}/{count}  "
                  f"{time.time() - started:.0f}s", flush=True)

    # ---- phase 2: add or remove the mean shift
    fold = np.random.default_rng(args.seed).permutation(count) % 2
    vectors: dict[str, dict[str, np.ndarray]] = {}
    for name in PROMPTS:
        shift = states[("tuned", name)].astype(np.float32) - states[("untuned", name)].astype(np.float32)
        overall, by_country = mean_shifts(shift, country, fold, args.min_country)
        noise = np.random.default_rng(args.seed + 1).standard_normal(overall.shape[1:]).astype(np.float32)
        noise *= (np.linalg.norm(overall, axis=2).mean(axis=0)
                  / np.linalg.norm(noise, axis=1))[:, None]
        vectors[name] = {"steer": overall, "country": by_country, "ablate": -overall,
                         "random": np.broadcast_to(noise, overall.shape)}
        del shift
    changed = {(arm, name): torch.zeros(len(layers), count, 2) for arm in ARMS for name in PROMPTS}
    for number, rows in enumerate(batches):
        for name in PROMPTS:
            scorer = scorer_for(name, rows)
            for arm in ARMS:
                switch = nothing if arm == "ablate" else untuned
                for slot, layer in enumerate(layers):
                    push = torch.from_numpy(np.ascontiguousarray(vectors[name][arm][rows, slot])
                                            ).repeat_interleave(2, dim=0)
                    with switch(), added(loaded.decoder_layers, layer, scorer.position, push):
                        changed[(arm, name)][slot, rows] = scorer.score()
        if number % 10 == 9 or number == len(batches) - 1:
            print(f"intervened {min((number + 1) * args.batch_size, count)}/{count}  "
                  f"{time.time() - started:.0f}s", flush=True)

    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "finetune_shift.npz",
             norm_ids=np.asarray([str(record["norm_id"]) for record in records]),
             layers=np.asarray(layers), fold=fold,
             **{f"scores__{who}__{name}": value.numpy() for (who, name), value in scores.items()},
             **{f"states__{who}__{name}": value for (who, name), value in states.items()},
             **{f"{arm}__{name}": value.numpy() for (arm, name), value in changed.items()})
    margin = lambda value: (value[..., 0] - value[..., 1]).mean(dim=-1)  # noqa: E731
    summary: dict[str, Any] = {}
    for name in PROMPTS:
        low, high = float(margin(scores[("untuned", name)])), float(margin(scores[("tuned", name)]))
        summary[name] = {"untuned": low, "tuned": high,
                         **{arm: {str(layer): float(margin(changed[(arm, name)][slot]))
                                  for slot, layer in enumerate(layers)} for arm in ARMS}}
    (output / "metrics.json").write_text(json.dumps({
        "analysis": "finetune_shift", "model_slug": args.model_slug, "adapter": str(args.adapter),
        "dataset_sha256": sha256_file(args.records), "sample": count,
        "sample_rule": f"uniform without replacement over the records file, seed {args.seed}",
        "layers": layers, "prompts": list(PROMPTS),
        "endpoint": "summed log P(option A) - summed log P(option B); option A is the reference",
        "vectors": "mean of state(tuned) - state(untuned) at the last prompt token, from the other of "
                   "two folds; added from that token on",
        "mean_margin": summary, "seconds": round(time.time() - started, 1)}, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    print(f"saved {output}  {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
