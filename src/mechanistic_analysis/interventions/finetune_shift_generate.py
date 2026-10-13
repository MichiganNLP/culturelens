"""Answers written with the fine-tuning shift added or removed.

``finetune_shift`` measures the margin, a proxy. This module asks the question
on what the models write. The mean shift (state of the tuned model minus state
of the untuned one at the last prompt token, after one block) is estimated on
the records *outside* the generation sample, from ``finetune_shift.npz``, and
then, for the no-country and the country-only prompt:

* ``untuned``, ``tuned``         the two models as they are;
* ``steer-<layer>``              untuned model, mean shift added after that block;
* ``random-<layer>``             untuned model, a random direction of the same length added;
* ``ablate-<layer>``             tuned model, mean shift subtracted.

A vector is added at the last prompt token and at every generated token. Each
arm writes ``<arm>.<prompt>.jsonl`` in the format the judge reads.

PYTHONPATH=src python -m mechanistic_analysis.interventions.finetune_shift_generate \
  --model /path/to/model --model-slug llama-3.1-8b --adapter runs/<name>/best \
  --shift output/finetune/mechanism/<name>/finetune_shift/llama-3.1-8b/finetune_shift.npz \
  --records output/finetune/mechanism/test_all.json --sample 500 --steer-layers 12 20 \
  --output-root output/finetune/mechanism/<name>/shift_answers
"""

from __future__ import annotations

import argparse
import contextlib
import json
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from mechanistic_analysis.activations.backend import require_torch
from mechanistic_analysis.interventions import finetune_shift
from mechanistic_analysis.interventions.common import (
    add_model_arguments, encode_prompt, load, sample_records, thinking_flag,
)
from mechanistic_analysis.interventions.erasure import _first_tensor
from mechanistic_analysis.interventions.finetune_shift import PROMPTS, nothing
from mechanistic_analysis.interventions.generate import left_padded


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--shift", type=Path, required=True, help="finetune_shift.npz of the same adapter.")
    parser.add_argument("--steer-layers", type=int, nargs="+", required=True)
    parser.add_argument("--random-layers", type=int, nargs="*", default=None,
                        help="Layers that also get the random-direction arm; the first steer layer when omitted.")
    parser.add_argument("--prompts", nargs="+", default=list(PROMPTS),
                        help="Prompt conditions to answer. Others than the two default ones "
                             "(for example the norm conditions) need --arms untuned and/or tuned, "
                             "because the shift is measured on the default two only.")
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Run only these arms (e.g. tuned); all of them when omitted.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    return parser.parse_args(argv)


@contextlib.contextmanager
def pushed(decoder_layers: Any, layer: int, start: Any, width: int, vector: Any) -> Iterator[None]:
    """Inside the block, ``vector`` is added to block ``layer``'s output: from
    ``start[row]`` on in the pass that holds the whole prompt of ``width``
    tokens, and everywhere in the later passes, which hold only new tokens."""

    torch = require_torch()

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        state = _first_tensor(output)
        extra = vector.to(device=state.device, dtype=state.dtype)
        if state.shape[1] >= width:
            chosen = torch.arange(state.shape[1], device=state.device)[None, :] >= start.to(state.device)[:, None]
            state += chosen[:, :, None] * extra[None, None, :]
        else:
            state += extra[None, None, :]

    handle = decoder_layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def shift_vectors(path: Path, exclude: set[str], layers: list[int], seed: int
                  ) -> dict[str, dict[int, dict[str, np.ndarray]]]:
    """``prompt -> layer -> {"mean", "random"}`` from the records not in ``exclude``."""

    saved = np.load(path)
    keep = np.array([str(key) not in exclude for key in saved["norm_ids"]])
    if keep.sum() < 10:
        raise ValueError("Too few records outside the generation sample to estimate the shift")
    known = [int(layer) for layer in saved["layers"]]
    out: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    for name in PROMPTS:
        out[name] = {}
        for layer in layers:
            slot = known.index(layer)
            mean = (saved[f"states__tuned__{name}"][keep, slot].astype(np.float32)
                    - saved[f"states__untuned__{name}"][keep, slot].astype(np.float32)).mean(axis=0)
            noise = np.random.default_rng(seed + layer).standard_normal(mean.shape).astype(np.float32)
            out[name][layer] = {"mean": mean, "random": noise * np.linalg.norm(mean) / np.linalg.norm(noise)}
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    started = time.time()
    records = sample_records(args.records, args.sample, args.seed)
    plain = bool(args.arms) and set(args.arms) <= {"untuned", "tuned"}      # no vector is needed
    if not plain and set(args.prompts) - set(PROMPTS):
        raise ValueError(f"The shift is defined for {list(PROMPTS)}; pass --arms untuned tuned for other prompts")
    vectors = {} if plain else shift_vectors(
        args.shift, {str(record["norm_id"]) for record in records}, args.steer_layers, args.seed)
    loaded = load(args)
    model, untuned = finetune_shift.attach(loaded, args.adapter)
    thinking = thinking_flag(args)
    tokenizer = loaded.tokenizer
    device = model.get_input_embeddings().weight.device
    randoms = args.steer_layers[:1] if args.random_layers is None else args.random_layers
    # arm -> (adapter off?, layer or None, which vector, sign)
    arms: dict[str, tuple[bool, int | None, str, float]] = {
        "untuned": (True, None, "", 0.0), "tuned": (False, None, "", 0.0)}
    for layer in args.steer_layers:
        arms[f"steer-{layer}"] = (True, layer, "mean", 1.0)
        arms[f"ablate-{layer}"] = (False, layer, "mean", -1.0)
    for layer in randoms:
        arms[f"random-{layer}"] = (True, layer, "random", 1.0)
    if args.arms:
        unknown = sorted(set(args.arms) - set(arms))
        if unknown:
            raise ValueError(f"Unknown arms {unknown}; expected some of {list(arms)}")
        arms = {name: arms[name] for name in args.arms}
    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model_slug} sample={len(records)} arms={list(arms)} prompts={args.prompts}", flush=True)

    counts: dict[str, int] = {}
    for arm, (off, layer, which, sign) in arms.items():
        for name in args.prompts:
            path = output / f"{arm}.{name}.jsonl"
            done = ({json.loads(line)["norm_id"] for line in path.read_text(encoding="utf-8").splitlines()
                     if line.strip()} if path.exists() else set())
            # a prompt that states the value is defined only for records that have one
            asked = [record for record in records
                     if "value" not in name or record["scenario"].get("value_evidence")]
            pending = [record for record in asked if str(record["norm_id"]) not in done]
            encoded = [encode_prompt(loaded, record, name, thinking)[0] for record in pending]
            order = np.argsort([len(ids) for ids in encoded], kind="stable")
            for begin in range(0, len(order), args.batch_size):
                rows = order[begin:begin + args.batch_size]
                ids, mask, pads = left_padded([encoded[row] for row in rows], tokenizer.pad_token_id)
                call = dict(input_ids=ids.to(device), attention_mask=mask.to(device),
                            max_new_tokens=args.max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.pad_token_id)
                start = torch.tensor([ids.shape[1] - 1] * len(rows))      # rows are flush right
                edit = (pushed(loaded.decoder_layers, layer, start, ids.shape[1],
                               torch.from_numpy(sign * vectors[name][layer][which]))
                        if layer is not None else nothing())
                with torch.no_grad(), (untuned() if off else nothing()), edit:
                    generated = model.generate(**call)
                with path.open("a", encoding="utf-8") as handle:
                    for position, row in enumerate(rows):
                        record = pending[row]
                        iso3, category = str(record["norm_id"]).split("__")[:2]
                        handle.write(json.dumps({
                            "norm_id": record["norm_id"], "iso3": iso3, "category": category,
                            "condition": name, "arm": arm,
                            "response": tokenizer.decode(generated[position, ids.shape[1]:],
                                                         skip_special_tokens=True).strip(),
                        }, ensure_ascii=False) + "\n")
            counts[f"{arm}.{name}"] = sum(1 for line in path.read_text(encoding="utf-8").splitlines()
                                          if line.strip()) if path.exists() else 0
            print(f"arm {arm} {name}: {counts[f'{arm}.{name}']}/{len(asked)} answers  "
                  f"{time.time() - started:.0f}s", flush=True)
    (output / "metrics.json").write_text(json.dumps({
        "analysis": "answers_with_the_finetuning_shift", "model_slug": args.model_slug,
        "adapter": str(args.adapter), "sample": len(records),
        "sample_rule": f"uniform without replacement over the records file, seed {args.seed}",
        "vectors": "mean of state(tuned) - state(untuned) at the last prompt token over the records "
                   "outside the sample; added at that token and at every generated token",
        "arms": {arm: {"adapter": "off" if off else "on", "layer": layer, "vector": which or None,
                       "sign": sign} for arm, (off, layer, which, sign) in arms.items()},
        "answers": counts, "decoding": {"sampling": False, "max_new_tokens": args.max_new_tokens},
        "seconds": round(time.time() - started, 1)}, indent=2))
    print(f"saved {output}  {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
