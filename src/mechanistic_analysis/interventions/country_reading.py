"""Where is the country read when no norm is stated?

On the ``country_only`` prompt, the country line is cut off from the answer in
two ways, by depth, and the decision margin is measured (``scoring.py``):

* attention knockout by layer window: positions after the user's text cannot
  attend to the country line in those layers (models whose layers can be
  masked). A span of the situation with as many tokens is the control;
* erasure by depth: after a block, the country line's positions take the mean
  state of the situation text (every model).

The reference is the margin on the untouched ``country_only`` prompt, and the
``base`` prompt gives the margin with no country at all, so each change can be
read as a share of what the country contributes. No norm appears in any prompt.

PYTHONPATH=src python -m mechanistic_analysis.interventions.country_reading \
  --model /path/to/model --model-slug llama-3.1-8b --window 4 \
  --records data/processed/culturelens/records.json \
  --output-root output/mechanistic_analysis/country_reading
"""

from __future__ import annotations

import argparse
import time

import numpy as np

from mechanistic_analysis.activations.backend import require_torch
from mechanistic_analysis.activations.cache import sha256_file, write_manifest
from mechanistic_analysis.interventions.attention import blocked, maskable_layers
from mechanistic_analysis.interventions.common import (
    add_model_arguments, encode_prompt, load, sample_records, thinking_flag,
)
from mechanistic_analysis.interventions.erasure import cut_layers, erased
from mechanistic_analysis.interventions.localize import windows
from mechanistic_analysis.interventions.scoring import BatchScorer, margin

TARGET, BARE = "country_only", "base"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--window", type=int, default=4, help="Layers per knockout window.")
    parser.add_argument("--cuts", type=int, default=16, help="Number of erasure depths.")
    return parser.parse_args(argv)


def control_span(spans: dict[str, tuple[int, int]]) -> tuple[int, int]:
    """As many tokens as the country line, from the start of the situation."""

    length = spans["country"][1] - spans["country"][0]
    first, last = spans["situation"]
    return first, min(first + length, last)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    started = time.time()
    records = sample_records(args.records, args.sample, args.seed)
    loaded = load(args)
    thinking = thinking_flag(args)
    tokenizer = loaded.tokenizer
    n_layers = len(loaded.decoder_layers)
    maskable = len(maskable_layers(loaded.decoder_layers)) == n_layers
    spans_of_layers = windows(n_layers, args.window) if maskable else []
    cuts = cut_layers(n_layers, args.cuts)
    encoded = {name: [encode_prompt(loaded, record, name, thinking) for record in records]
               for name in (TARGET, BARE)}
    answers = [[tokenizer(str(record["options"][key]), add_special_tokens=False)["input_ids"]
                for key in ("A", "B")] for record in records]
    order = np.argsort([len(item[0]) for item in encoded[TARGET]], kind="stable")
    batches = [order[start:start + args.batch_size] for start in range(0, len(order), args.batch_size)]
    batches = [batches[i] for i in np.random.default_rng(args.seed).permutation(len(batches))]
    count = len(records)
    untouched, bare = torch.zeros(count, 2), torch.zeros(count, 2)
    knocked = {span: torch.zeros(len(spans_of_layers), count, 2) for span in ("country", "control")}
    cut = torch.zeros(len(cuts), count, 2)
    completed = np.zeros(count, dtype=bool)
    print(f"model={args.model_slug} sample={count} maskable={maskable} "
          f"windows={spans_of_layers} cuts={cuts}", flush=True)
    for number, rows in enumerate(batches):
        if args.max_seconds and number and time.time() - started > args.max_seconds:
            print(f"time limit reached after {int(completed.sum())} records", flush=True)
            break
        bare[rows] = BatchScorer(loaded.model, tokenizer.pad_token_id,
                                 [encoded[BARE][row][0] for row in rows],
                                 [answers[row] for row in rows]).score()
        scorer = BatchScorer(loaded.model, tokenizer.pad_token_id,
                             [encoded[TARGET][row][0] for row in rows],
                             [answers[row] for row in rows])
        untouched[rows] = scorer.score()
        query = torch.tensor([encoded[TARGET][row][1] for row in rows]).repeat_interleave(2)
        keys = {span: torch.zeros(len(rows) * 2, scorer.width, dtype=torch.bool)
                for span in ("country", "control", "situation")}
        for position, row in enumerate(rows):
            spans = encoded[TARGET][row][2]
            marks = {"country": spans["country"], "control": control_span(spans),
                     "situation": spans["situation"]}
            for span, (first, last) in marks.items():
                keys[span][2 * position:2 * position + 2, first:last] = True
        for span in knocked:
            for slot, (low, high) in enumerate(spans_of_layers):
                with blocked(loaded.decoder_layers, list(range(low, high)), query, keys[span]):
                    knocked[span][slot, rows] = scorer.score()
        for slot, layer in enumerate(cuts):
            with erased(loaded.decoder_layers, layer, keys["country"], keys["situation"]):
                cut[slot, rows] = scorer.score()
        completed[rows] = True
        print(f"{int(completed.sum())}/{count} records  {time.time() - started:.0f}s", flush=True)

    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "country_reading.npz",
             norm_ids=np.asarray([str(record["norm_id"]) for record in records]),
             completed=completed, windows=np.asarray(spans_of_layers), cuts=np.asarray(cuts),
             untouched=untouched.numpy(), bare=bare.numpy(), erase=cut.numpy(),
             **{f"knock__{span}": values.numpy() for span, values in knocked.items()})
    done = torch.as_tensor(completed)
    reference = margin(untouched)[done]
    write_manifest({
        "analysis": "country_reading_by_depth",
        "model_slug": args.model_slug, "lora": getattr(args, "lora", None),
        "dataset_sha256": sha256_file(args.records),
        "sample": count, "completed": int(completed.sum()),
        "sample_rule": f"uniform without replacement over the records file, seed {args.seed}",
        "endpoint": "summed log P(option A) - summed log P(option B); option A is the reference",
        "mean_margin": {"country_only": float(reference.mean()), "base": float(margin(bare)[done].mean())},
        "knockout_change": {span: {f"{low}-{high}": float((margin(values[slot])[done] - reference).mean())
                                   for slot, (low, high) in enumerate(spans_of_layers)}
                            for span, values in knocked.items()},
        "erasure_change": {str(layer): float((margin(cut[slot])[done] - reference).mean())
                           for slot, layer in enumerate(cuts)},
        "seconds": round(time.time() - started, 1),
    }, output / "metrics.json")
    print(f"saved {output}  {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
