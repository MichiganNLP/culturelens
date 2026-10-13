"""How do the country and the claim combine in the model's state?

Each record is cached under a 2 x 3 design: no country or its own country,
crossed with no claim, the documented norm, or its opposite. Writing ``b`` for
the bare prompt, the state under country + claim splits exactly into

    b + country effect + claim effect + interaction

with country effect ``country_only - b``, claim effect ``norm_only - b`` (or
``opposite_norm_only - b``), and the interaction what is left. A model that
only added the two pieces of evidence would have no interaction; whatever it
does to check a claim against a country is in it.

Per layer:

* ``country``, ``claim_*``, ``interaction_*``: root-mean-square size of each
  part, and ``additive_share_*``, the share of the whole shift the additive
  parts explain;
* ``interaction_cosine``: mean cosine between a record's true-claim and
  false-claim interactions. Near 1 the model treats the two alike; a drop is
  where it starts to tell them apart given the country;
* ``country_given_false_*``: size (relative to the plain country effect) and
  direction (cosine with it) of what the country adds once the false claim is
  present.

Per record, at every ``--features-step``-th layer, saved for ``report.py``:

    conflict   projection of the record's false-claim interaction on the
               direction separating false- from true-claim interactions.
               That direction is fitted on the other folds and uses no
               behavioural label.

PYTHONPATH=src python -m mechanistic_analysis.interaction \
  --activations-root <caches> --model-slug llama-3.1-8b \
  --records data/processed/culturelens/records.json \
  --output-root output/mechanistic_analysis/interaction
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Any

import numpy as np

from mechanistic_analysis.activations.backend import require_torch
from mechanistic_analysis.activations.cache import write_manifest
from mechanistic_analysis.activations.extract import read_records
from mechanistic_analysis.common import (
    assign_folds, claim_groups, load_conditions, resolve_device,
)

CONDITIONS = ("base", "country_only", "norm_only", "opposite_norm_only", "country_norm",
              "country_opposite_norm")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations-root", type=Path, required=True)
    parser.add_argument("--variant", default="unconstrained")
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--features-step", type=int, default=2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def rms(x: Any) -> float:
    return float(x.norm(dim=1).square().mean().sqrt())


def layer_statistics(s: dict[str, Any]) -> tuple[dict[str, float], dict[str, Any]]:
    """``s`` maps each condition to (records, features) at one layer."""

    torch = require_torch()
    cosine = torch.nn.functional.cosine_similarity
    b = s["base"]
    country = s["country_only"] - b
    parts: dict[str, Any] = {}
    row: dict[str, float] = {"state_norm": rms(b), "country": rms(country)}
    for tag, alone, both in (("true", "norm_only", "country_norm"),
                             ("false", "opposite_norm_only", "country_opposite_norm")):
        claim = s[alone] - b
        total = s[both] - b
        parts[tag] = total - country - claim
        row[f"claim_{tag}"] = rms(claim)
        row[f"interaction_{tag}"] = rms(parts[tag])
        row[f"additive_share_{tag}"] = 1.0 - rms(parts[tag]) ** 2 / rms(total) ** 2
    given = s["country_opposite_norm"] - s["opposite_norm_only"]
    row["country_given_false_ratio"] = rms(given) / rms(country)
    row["country_given_false_cosine"] = float(cosine(given, country, dim=1).mean())
    row["interaction_cosine"] = float(cosine(parts["true"], parts["false"], dim=1).mean())
    return row, parts


def conflict_feature(parts: dict[str, Any], folds: Any, n_folds: int) -> Any:
    torch = require_torch()
    gap = parts["false"] - parts["true"]
    out = torch.empty(len(folds), device=gap.device)
    for fold in range(n_folds):
        test = folds == fold
        direction = torch.nn.functional.normalize(gap[~test].mean(dim=0), dim=0)
        out[test] = parts["false"][test] @ direction
    return out


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    device = resolve_device(args.device)
    started = time.time()
    caches = load_conditions(args.activations_root, args.variant, args.model_slug, CONDITIONS)
    first = caches["base"]
    norm_ids = [str(value) for value in first["norm_ids"]]
    by_id = {str(record["norm_id"]): record for record in read_records(args.records)}
    groups = claim_groups([by_id[n] for n in norm_ids])
    folds = torch.tensor(assign_folds(groups, args.folds, args.seed), device=device)
    layers = list(first["layers"])
    keep = sorted(set(layers[::max(1, args.features_step)]) | {layers[-1]})
    print(f"model={args.model_slug} records={len(norm_ids)} layers={len(layers)} device={device}",
          flush=True)

    rows: list[dict[str, Any]] = []
    conflict: list[np.ndarray] = []
    for position, layer in enumerate(layers):
        states = {name: payload["assistant_start"][:, position].to(device, torch.float32)
                  for name, payload in caches.items()}
        row, parts = layer_statistics(states)
        rows.append({"layer": layer, **row})
        if layer in keep:
            conflict.append(conflict_feature(parts, folds, args.folds).cpu().numpy().astype(np.float32))
    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    with (output / "layers.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez(output / "features.npz", norm_ids=np.asarray(norm_ids), layers=np.asarray(keep),
             groups=np.asarray(groups), conflict=np.stack(conflict))
    write_manifest({
        "analysis": "country_claim_interaction",
        "model_slug": args.model_slug,
        "dataset_sha256": first["dataset_sha256"],
        "records": len(norm_ids),
        "definitions": {
            "country": "country_only - base",
            "claim": "norm_only - base (true) or opposite_norm_only - base (false)",
            "interaction": "country+claim state - base - country - claim",
            "additive_share": "1 - |interaction|^2 / |country+claim state - base|^2",
            "conflict": "false-claim interaction projected on mean(false - true interaction), "
                        "direction fitted on the other folds; no behavioural label",
        },
        "feature_layers": keep,
        "layers": rows,
        "seconds": round(time.time() - started, 1),
    }, output / "metrics.json")
    print(f"saved {output}  {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
