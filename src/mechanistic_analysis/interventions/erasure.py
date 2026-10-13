"""Erasing the claim's representation after a given depth.

After decoder block ``layer``, the residual state at every position of the
claim is replaced by a neutral vector: the mean state, at that layer, of the
same prompt's situation text. From that depth on, nothing in the claim's own
positions distinguishes this claim from filler, whatever mechanism later
layers use to move information between positions. Anything a later position
copied from the claim *before* that depth is untouched.

So the effect of erasing after block L measures how much of the claim's
influence on the answer still has to be read from the claim's positions above
L. Erasing after the first block removes nearly all of it; once the claim has
been read into later positions, erasing does nothing. Unlike attention
knockout this needs no attention mask, so it applies to models that mix
tokens with recurrent blocks.

Run as a module, it sweeps the depth and records the decision margin:

PYTHONPATH=src python -m mechanistic_analysis.interventions.erasure \
  --model /path/to/model --model-slug qwen-3.5-27b \
  --records data/processed/culturelens/records.json \
  --output-root output/mechanistic_analysis/erasure
"""

from __future__ import annotations

import argparse
import contextlib
import time
from typing import Any, Iterator

import numpy as np

from mechanistic_analysis.activations.backend import _first_tensor, require_torch
from mechanistic_analysis.activations.cache import sha256_file, write_manifest
from mechanistic_analysis.interventions.common import (
    add_model_arguments, encode_prompt, load, sample_records, thinking_flag,
)
from mechanistic_analysis.interventions.scoring import BatchScorer, margin

TARGETS = ("country_opposite_norm", "country_norm")


@contextlib.contextmanager
def erased(decoder_layers: Any, layer: int, span: Any, neutral: Any) -> Iterator[None]:
    """Inside the block, block ``layer``'s output at the positions marked in
    ``span`` (rows, positions) becomes the mean of that row's output over the
    positions marked in ``neutral``. Passes that do not contain the prompt's
    positions (cached decoding steps) are left alone."""

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        state = _first_tensor(output)
        if state.shape[1] < span.shape[1]:
            return
        marked = span.to(state.device)
        weights = neutral.to(device=state.device, dtype=state.dtype)
        filler = (state[:, :weights.shape[1]] * weights[:, :, None]).sum(dim=1) / weights.sum(
            dim=1, keepdim=True).clamp_min(1)
        view = state[:, :marked.shape[1]]
        view[marked] = filler[:, None, :].expand(-1, marked.shape[1], -1)[marked]

    handle = decoder_layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


def cut_layers(n_layers: int, count: int) -> list[int]:
    """About ``count`` evenly spaced blocks, always including the first."""

    step = max(1, n_layers // count)
    return list(range(0, n_layers, step))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--cuts", type=int, default=16, help="Number of depths to try.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    started = time.time()
    records = sample_records(args.records, args.sample, args.seed)
    loaded = load(args)
    thinking = thinking_flag(args)
    tokenizer = loaded.tokenizer
    cuts = cut_layers(len(loaded.decoder_layers), args.cuts)
    encoded = {name: [encode_prompt(loaded, record, name, thinking) for record in records]
               for name in TARGETS}
    answers = [[tokenizer(str(record["options"][key]), add_special_tokens=False)["input_ids"]
                for key in ("A", "B")] for record in records]
    order = np.argsort([len(item[0]) for item in encoded[TARGETS[0]]], kind="stable")
    batches = [order[start:start + args.batch_size] for start in range(0, len(order), args.batch_size)]
    batches = [batches[i] for i in np.random.default_rng(args.seed).permutation(len(batches))]
    baseline = {name: torch.zeros(len(records), 2) for name in TARGETS}
    cut = {name: torch.zeros(len(cuts), len(records), 2) for name in TARGETS}
    completed = np.zeros(len(records), dtype=bool)
    print(f"model={args.model_slug} sample={len(records)} cuts={cuts}", flush=True)
    for number, rows in enumerate(batches):
        if args.max_seconds and number and time.time() - started > args.max_seconds:
            print(f"time limit reached after {int(completed.sum())} records", flush=True)
            break
        for name in TARGETS:
            scorer = BatchScorer(loaded.model, tokenizer.pad_token_id,
                                 [encoded[name][row][0] for row in rows],
                                 [answers[row] for row in rows])
            baseline[name][rows] = scorer.score()
            span = torch.zeros(len(rows) * 2, scorer.width, dtype=torch.bool)
            neutral = torch.zeros(len(rows) * 2, scorer.width, dtype=torch.bool)
            for position, row in enumerate(rows):
                spans = encoded[name][row][2]
                span[2 * position:2 * position + 2, spans["claim"][0]:spans["claim"][1]] = True
                neutral[2 * position:2 * position + 2,
                        spans["situation"][0]:spans["situation"][1]] = True
            for slot, layer in enumerate(cuts):
                with erased(loaded.decoder_layers, layer, span, neutral):
                    cut[name][slot, rows] = scorer.score()
        completed[rows] = True
        print(f"{int(completed.sum())}/{len(records)} records  {time.time() - started:.0f}s", flush=True)

    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "erasure.npz",
             norm_ids=np.asarray([str(record["norm_id"]) for record in records]),
             completed=completed, cuts=np.asarray(cuts),
             **{f"baseline__{name}": values.numpy() for name, values in baseline.items()},
             **{f"erase__{name}": values.numpy() for name, values in cut.items()})
    done = torch.as_tensor(completed)
    write_manifest({
        "analysis": "claim_erasure_by_depth",
        "model_slug": args.model_slug,
        "dataset_sha256": sha256_file(args.records),
        "sample": len(records), "completed": int(completed.sum()),
        "sample_rule": f"uniform without replacement over all records, seed {args.seed}",
        "cuts": cuts,
        "intervention": "after the given block, the claim's positions take the mean state of "
                        "the same prompt's situation text at that block",
        "endpoint": "summed log P(option A) - summed log P(option B); option A is the reference",
        "mean_change": {name: {str(layer): float(
            (margin(cut[name][slot]) - margin(baseline[name]))[done].mean())
            for slot, layer in enumerate(cuts)} for name in TARGETS},
        "seconds": round(time.time() - started, 1),
    }, output / "metrics.json")
    print(f"saved {output}  {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
