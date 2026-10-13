"""Does the state register that a stated norm does not fit the country?

A label-free test, run on a model directly (optionally with a LoRA adapter
merged in), so a fine-tuned model can be compared with the model it came from
on the same records.

For each sampled record the last prompt token's state is taken under six
prompts: no country or the record's country, crossed with no norm, the true
norm, or the contradictory norm. The part of the state under country + norm
that neither piece of evidence explains alone is

    interaction = state(country + norm) - state(norm) - state(country) + state(bare)

A model that only added the two pieces of evidence would have none. Per layer:

* ``auroc``: how well one direction separates contradictory-norm interactions
  from true-norm ones. The direction is the difference of the two class means,
  fitted on four folds of records and scored on the fifth. 0.5 means the state
  does not tell a norm that fits the country from one that does not;
* ``paired``: the share of held-out records whose contradictory-norm
  interaction lies further along that direction than its own true-norm one;
* ``size_true``, ``size_false``: root-mean-square size of each interaction,
  relative to the country effect ``state(country) - state(bare)``.

No judged answer and no behavioural label is used.

PYTHONPATH=src python -m mechanistic_analysis.conflict \
  --model /path/to/model --model-slug llama-3.1-8b [--lora runs/<name>/best] \
  --records data/processed/culturelens/splits/test_all.json \
  --output-root output/finetune/mechanism/<name>/conflict
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from mechanistic_analysis.activations.backend import extract_batch, require_torch
from mechanistic_analysis.activations.cache import sha256_file
from mechanistic_analysis.activations.conditions import build_user_message
from mechanistic_analysis.activations.extract import render_prompt
from mechanistic_analysis.interventions.common import (
    add_model_arguments, load, sample_records, thinking_flag,
)
from mechanistic_analysis.stats import auroc

CONDITIONS = ("base", "country_only", "norm_only", "opposite_norm_only", "country_norm",
              "country_opposite_norm")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.set_defaults(sample=1000)
    parser.add_argument("--folds", type=int, default=5)
    return parser.parse_args(argv)


def interactions(states: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """True-norm and contradictory-norm interactions and the country effect,
    each ``(records, layers, width)``."""

    bare, country = states["base"], states["country_only"]
    true = states["country_norm"] - states["norm_only"] - country + bare
    false = states["country_opposite_norm"] - states["opposite_norm_only"] - country + bare
    return true, false, country - bare


def separability(true: np.ndarray, false: np.ndarray, folds: int, seed: int) -> tuple[float, float]:
    """Held-out AUROC and paired accuracy of the mean-difference direction for
    one layer; ``true`` and ``false`` are ``(records, width)``. A record's two
    interactions are always in the same fold."""

    count = len(true)
    fold = np.random.default_rng(seed).permutation(count) % folds
    score_true, score_false = np.zeros(count), np.zeros(count)
    for held in range(folds):
        fit, test = fold != held, fold == held
        direction = false[fit].mean(axis=0) - true[fit].mean(axis=0)
        direction /= max(float(np.linalg.norm(direction)), 1e-12)
        score_true[test], score_false[test] = true[test] @ direction, false[test] @ direction
    scores = np.concatenate([score_true, score_false])
    labels = np.concatenate([np.zeros(count, dtype=bool), np.ones(count, dtype=bool)])
    return float(auroc(scores, labels)), float((score_false > score_true).mean())


def summarize(states: dict[str, np.ndarray], folds: int, seed: int) -> list[dict[str, Any]]:
    true, false, country = interactions(states)
    rows = []
    for layer in range(true.shape[1]):
        scale = float(np.sqrt((country[:, layer] ** 2).sum(axis=1).mean())) or 1.0
        area, paired = separability(true[:, layer], false[:, layer], folds, seed)
        rows.append({"layer": layer, "auroc": area, "paired": paired,
                     "size_true": float(np.sqrt((true[:, layer] ** 2).sum(axis=1).mean())) / scale,
                     "size_false": float(np.sqrt((false[:, layer] ** 2).sum(axis=1).mean())) / scale})
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    require_torch()
    started = time.time()
    records = sample_records(args.records, args.sample, args.seed)
    loaded = load(args)
    thinking = thinking_flag(args)
    states: dict[str, np.ndarray] = {}
    for condition in CONDITIONS:
        prompts = [render_prompt(loaded, build_user_message(record, condition, include_options=False)[0],
                                 constrained=False, thinking=thinking) for record in records]
        parts = []
        for start in range(0, len(prompts), args.batch_size):
            batch, _, _ = extract_batch(loaded, prompts[start:start + args.batch_size],
                                        storage_dtype="float32", max_length=8192)
            parts.append(batch.numpy().astype(np.float64))
        states[condition] = np.concatenate(parts)
        print(f"{condition}: {states[condition].shape}  {time.time() - started:.0f}s", flush=True)

    rows = summarize(states, args.folds, args.seed)
    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps({
        "analysis": "country_norm_conflict_separability",
        "model_slug": args.model_slug, "lora": args.lora,
        "dataset_sha256": sha256_file(args.records), "records": len(records),
        "sample_rule": f"uniform without replacement over the records file, seed {args.seed}",
        "folds": args.folds, "layers": rows, "seconds": round(time.time() - started, 1)}, indent=2))
    best = max(rows, key=lambda row: row["auroc"])
    print(f"best layer {best['layer']}: auroc {best['auroc']:.3f}, paired {best['paired']:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
