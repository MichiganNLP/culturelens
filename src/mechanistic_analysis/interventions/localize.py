"""At which depth does the answer read the claim? (attention knockout, by layer window)

For a window of decoder layers, every position after the user's text is
stopped from attending to one part of the prompt (``attention.py``), and the
decision margin (``scoring.py``) is measured again:

    claim     the stated norm (true or false)
    country   the country line
    control   as many tokens as the claim, from the start of the situation

Blocking a part in every layer shows how much the answer depends on reading it;
blocking it in one window shows where that reading happens. Prompts scored:
the false norm under its country and the true norm under its country.

The margin is a proxy. What the model then writes is tested by ``generate.py``.

PYTHONPATH=src python -m mechanistic_analysis.interventions.localize \
  --model /path/to/model --model-slug llama-3.1-8b \
  --records data/processed/culturelens/records.json \
  --output-root output/mechanistic_analysis/localize
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np

from mechanistic_analysis.activations.backend import require_torch
from mechanistic_analysis.activations.cache import sha256_file, write_manifest
from mechanistic_analysis.interventions.attention import blocked, maskable_layers
from mechanistic_analysis.interventions.common import (
    add_model_arguments, encode_prompt, load, sample_records, thinking_flag,
)
from mechanistic_analysis.interventions.scoring import BatchScorer, margin

TARGETS = {"country_opposite_norm": ("claim", "country", "control"),
           "country_norm": ("claim", "country")}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--window", type=int, default=4, help="Layers per window.")
    return parser.parse_args(argv)


def windows(n_layers: int, width: int) -> list[tuple[int, int]]:
    """Consecutive windows of ``width`` layers, then one covering every layer."""

    return [(start, min(start + width, n_layers)) for start in range(0, n_layers, width)] + [
        (0, n_layers)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    started = time.time()
    records = sample_records(args.records, args.sample, args.seed)
    loaded = load(args)
    thinking = thinking_flag(args)
    tokenizer = loaded.tokenizer
    n_layers = len(loaded.decoder_layers)
    maskable = maskable_layers(loaded.decoder_layers)
    if not maskable:
        raise RuntimeError("No decoder layer exposes a maskable self-attention module")
    spans_of_layers = windows(n_layers, args.window)

    encoded = {name: [encode_prompt(loaded, record, name, thinking) for record in records]
               for name in TARGETS}
    answers = [[tokenizer(str(record["options"][key]), add_special_tokens=False)["input_ids"]
                for key in ("A", "B")] for record in records]
    order = np.argsort([len(item[0]) for item in encoded["country_opposite_norm"]], kind="stable")
    batches = [order[start:start + args.batch_size]
               for start in range(0, len(order), args.batch_size)]
    generator = np.random.default_rng(args.seed)
    batches = [batches[index] for index in generator.permutation(len(batches))]
    baseline = {name: torch.zeros(len(records), 2) for name in TARGETS}
    knocked = {(name, span): torch.zeros(len(spans_of_layers), len(records), 2)
               for name, listed in TARGETS.items() for span in listed}
    completed = np.zeros(len(records), dtype=bool)
    print(f"model={args.model_slug} sample={len(records)} layers={n_layers} "
          f"maskable={len(maskable)} windows={spans_of_layers}", flush=True)
    for number, rows in enumerate(batches):
        if args.max_seconds and number and time.time() - started > args.max_seconds:
            print(f"time limit reached after {int(completed.sum())} records", flush=True)
            break
        for name, listed in TARGETS.items():
            scorer = BatchScorer(loaded.model, tokenizer.pad_token_id,
                                 [encoded[name][row][0] for row in rows],
                                 [answers[row] for row in rows])
            baseline[name][rows] = scorer.score()
            query = torch.tensor([encoded[name][row][1] for row in rows]).repeat_interleave(2)
            for span in listed:
                keys = torch.zeros(len(rows) * 2, scorer.width, dtype=torch.bool)
                for position, row in enumerate(rows):
                    first, last = encoded[name][row][2][span]
                    keys[2 * position:2 * position + 2, first:last] = True
                for slot, (low, high) in enumerate(spans_of_layers):
                    with blocked(loaded.decoder_layers, list(range(low, high)), query, keys):
                        knocked[(name, span)][slot, rows] = scorer.score()
        completed[rows] = True
        print(f"{int(completed.sum())}/{len(records)} records  {time.time() - started:.0f}s",
              flush=True)

    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "localize.npz",
             norm_ids=np.asarray([str(record["norm_id"]) for record in records]),
             completed=completed, windows=np.asarray(spans_of_layers),
             **{f"baseline__{name}": values.numpy() for name, values in baseline.items()},
             **{f"knock__{name}__{span}": values.numpy() for (name, span), values in knocked.items()})
    done = torch.as_tensor(completed)
    write_manifest({
        "analysis": "attention_knockout_localisation",
        "model_slug": args.model_slug,
        "dataset_sha256": sha256_file(args.records),
        "sample": len(records),
        "completed": int(completed.sum()),
        "sample_rule": f"uniform without replacement over all records, seed {args.seed}",
        "windows": spans_of_layers,
        "maskable_layers": maskable,
        "queries": "every position after the user's text: chat-template tail and answer tokens",
        "endpoint": "summed log P(option A) - summed log P(option B); option A is the reference",
        "baseline_mean_margin": {name: float(margin(values)[done].mean())
                                 for name, values in baseline.items()},
        "mean_change": {f"{name}__{span}": {
            f"{low}-{high}": float((margin(values[slot]) - margin(baseline[name]))[done].mean())
            for slot, (low, high) in enumerate(spans_of_layers)}
            for (name, span), values in knocked.items()},
        "seconds": round(time.time() - started, 1),
    }, output / "metrics.json")
    print(f"saved {output}  {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
