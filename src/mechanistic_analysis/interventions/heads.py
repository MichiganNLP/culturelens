"""Which attention heads carry the claim into the answer?

Inside the claim window (``mechanistic_analysis.CLAIM_WINDOW``), each head is
stopped, alone, from letting the answer stream attend to the false claim, and
the decision margin is measured. Heads are ranked by that effect; the top k
are then blocked together for growing k, next to k heads drawn at random from
the same window.

The selection used downstream is rule-based: the smallest k (among the sizes
tried) whose joint effect reaches half of the whole window's. It is made on
the development sample (seed 42). ``generate.py`` tests the selected heads,
and a random set of the same size, on the confirmation sample, with judged
answers.

PYTHONPATH=src python -m mechanistic_analysis.interventions.heads \
  --model /path/to/model --model-slug llama-3.1-8b --seed 42 --sample 200 \
  --records data/processed/culturelens/records.json \
  --output-root output/mechanistic_analysis/heads
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from mechanistic_analysis import CLAIM_WINDOW, DEVELOPMENT_SEED
from mechanistic_analysis.activations.backend import require_torch
from mechanistic_analysis.activations.cache import sha256_file, write_manifest
from mechanistic_analysis.interventions.attention import attention_module, blocked, head_count
from mechanistic_analysis.interventions.common import (
    add_model_arguments, encode_prompt, load, sample_records, thinking_flag,
)
from mechanistic_analysis.interventions.scoring import BatchScorer, margin

TARGET = "country_opposite_norm"
SIZES = (1, 2, 4, 8, 16, 32, 64)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.set_defaults(seed=DEVELOPMENT_SEED, sample=200)
    parser.add_argument("--share", type=float, default=0.5,
                        help="Share of the window's effect the selected heads must reach.")
    return parser.parse_args(argv)


def grouped(heads: list[tuple[int, int]]) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for layer, head in heads:
        out.setdefault(layer, []).append(head)
    return out


def select(effect_by_size: dict[int, float], window_effect: float, share: float) -> int:
    """The smallest size whose joint effect reaches ``share`` of the window's;
    the largest size tried if none does."""

    for size in sorted(effect_by_size):
        if effect_by_size[size] >= share * window_effect:
            return size
    return max(effect_by_size)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    started = time.time()
    if args.model_slug not in CLAIM_WINDOW:
        raise ValueError(f"No claim window is registered for {args.model_slug}")
    low, high = CLAIM_WINDOW[args.model_slug]
    records = sample_records(args.records, args.sample, args.seed)
    loaded = load(args)
    thinking = thinking_flag(args)
    tokenizer = loaded.tokenizer
    candidates = [(layer, head) for layer in range(low, high)
                  for head in range(head_count(attention_module(loaded.decoder_layers[layer])))]
    encoded = [encode_prompt(loaded, record, TARGET, thinking) for record in records]
    answers = [[tokenizer(str(record["options"][key]), add_special_tokens=False)["input_ids"]
                for key in ("A", "B")] for record in records]
    order = np.argsort([len(item[0]) for item in encoded], kind="stable")
    batches = [order[start:start + args.batch_size] for start in range(0, len(order), args.batch_size)]
    print(f"model={args.model_slug} sample={len(records)} window={low}-{high} "
          f"heads={len(candidates)}", flush=True)

    def sweep(sets: list[dict[int, list[int]] | None]) -> Any:
        """Margins (sets, records) with each head set blocked; None blocks the whole window."""

        out = torch.zeros(len(sets), len(records))
        plain = torch.zeros(len(records))
        for rows in batches:
            scorer = BatchScorer(loaded.model, tokenizer.pad_token_id,
                                 [encoded[row][0] for row in rows], [answers[row] for row in rows])
            plain[rows] = margin(scorer.score())
            query = torch.tensor([encoded[row][1] for row in rows]).repeat_interleave(2)
            keys = torch.zeros(len(rows) * 2, scorer.width, dtype=torch.bool)
            for position, row in enumerate(rows):
                first, last = encoded[row][2]["claim"]
                keys[2 * position:2 * position + 2, first:last] = True
            for slot, heads in enumerate(sets):
                with blocked(loaded.decoder_layers, list(range(low, high)), query, keys, heads=heads):
                    out[slot, rows] = margin(scorer.score())
        return out, plain

    single, plain = sweep([{layer: [head]} for layer, head in candidates] + [None])
    effect = (single - plain).mean(dim=1)
    window_effect = float(effect[-1])
    ranked = [candidates[i] for i in torch.argsort(effect[:-1], descending=True).tolist()]
    print(f"window effect {window_effect:+.2f}; best single head {ranked[0]} "
          f"{float(effect[:-1].max()):+.2f}  {time.time() - started:.0f}s", flush=True)

    sizes = [size for size in SIZES if size < len(candidates)]
    generator = np.random.default_rng(args.seed)
    shuffled = [candidates[i] for i in generator.permutation(len(candidates))]
    joint, _ = sweep([grouped(ranked[:size]) for size in sizes]
                     + [grouped(shuffled[:size]) for size in sizes])
    joint_effect = (joint - plain).mean(dim=1).tolist()
    top = dict(zip(sizes, joint_effect[:len(sizes)]))
    random = dict(zip(sizes, joint_effect[len(sizes):]))
    chosen = select(top, window_effect, args.share)
    others = [head for head in shuffled if head not in ranked[:chosen]][:chosen]

    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "heads.npz", candidates=np.asarray(candidates),
             single_effect=effect[:-1].numpy(), ranked=np.asarray(ranked),
             sizes=np.asarray(sizes), top_effect=np.asarray(list(top.values())),
             random_effect=np.asarray(list(random.values())))
    selection = {
        "model_slug": args.model_slug, "window": [low, high], "size": chosen,
        "heads": {str(layer): heads for layer, heads in grouped(ranked[:chosen]).items()},
        "random_heads": {str(layer): heads for layer, heads in grouped(others).items()},
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=1), encoding="utf-8")
    write_manifest({
        "analysis": "claim_reading_heads",
        "model_slug": args.model_slug,
        "dataset_sha256": sha256_file(args.records),
        "sample": len(records),
        "sample_rule": f"uniform without replacement over all records, seed {args.seed} "
                       "(the development sample)",
        "window": [low, high], "heads_in_window": len(candidates),
        "endpoint": "mean change in margin when the answer stream cannot attend to the false "
                    "claim through the given heads",
        "window_effect": window_effect,
        "single_head_effects_ranked": [
            {"layer": layer, "head": head, "effect": float(effect[candidates.index((layer, head))])}
            for layer, head in ranked[:20]],
        "joint_effect_top": {str(k): v for k, v in top.items()},
        "joint_effect_random": {str(k): v for k, v in random.items()},
        "selection_rule": f"smallest size reaching {args.share:.0%} of the window effect",
        "selection": selection,
        "seconds": round(time.time() - started, 1),
    }, output / "metrics.json")
    print(f"saved {output}  selected {chosen} heads: top {top}  random {random}  "
          f"{time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
