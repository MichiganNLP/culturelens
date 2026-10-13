"""What each model did on the open-ended prompts, as each judge saw it.

One table per evaluated model. Columns are ``<judge>__<condition>``: 1 when
that judge labelled the answer ``aligned`` (the strict endpoint), 0 otherwise,
-1 where there is no such answer or no judgment. ``both__<condition>`` is 1
only when every judge said aligned, and -1 if any judgment is missing.

PYTHONPATH=src python -m mechanistic_analysis.outcomes \
  --records data/processed/culturelens/records.json \
  --judgments-root output/behavioral_evaluation/unconstrained/judgments \
  --model-slug llama-3.1-8b --output output/mechanistic_analysis/outcomes/llama-3.1-8b.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mechanistic_analysis import JUDGES

CONDITIONS = ("base", "country_only", "norm_only", "country_norm", "country_opposite_norm")


def build(norm_ids: list[str], judgments: dict[str, Path]) -> tuple[list[str], np.ndarray]:
    row = {norm_id: index for index, norm_id in enumerate(norm_ids)}
    names: list[str] = []
    columns: list[np.ndarray] = []
    per_judge: dict[str, dict[str, np.ndarray]] = {}
    for judge, path in judgments.items():
        per_judge[judge] = {c: np.full(len(norm_ids), -1, dtype=np.int8) for c in CONDITIONS}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                index = row.get(str(item["norm_id"]))
                if index is not None and item["condition"] in per_judge[judge]:
                    per_judge[judge][item["condition"]][index] = item["alignment"] == "aligned"
        for condition in CONDITIONS:
            names.append(f"{judge}__{condition}")
            columns.append(per_judge[judge][condition])
    for condition in CONDITIONS:
        stacked = np.stack([per_judge[judge][condition] for judge in judgments])
        both = np.where((stacked < 0).any(axis=0), -1, stacked.min(axis=0)).astype(np.int8)
        names.append(f"both__{condition}")
        columns.append(both)
    return names, np.stack(columns, axis=1)


def load(path: Path, norm_ids: list[str]) -> dict[str, np.ndarray]:
    """Outcome columns aligned to ``norm_ids`` (which may be a subset, in any order)."""

    saved = np.load(path)
    row = {str(value): index for index, value in enumerate(saved["norm_ids"])}
    missing = [norm_id for norm_id in norm_ids if str(norm_id) not in row]
    if missing:
        raise ValueError(f"{path} lacks {len(missing)} of the requested records")
    take = np.array([row[str(norm_id)] for norm_id in norm_ids])
    return {str(name): saved["outcomes"][take, column]
            for column, name in enumerate(saved["names"])}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--judges", nargs="+", default=list(JUDGES))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    norm_ids = [str(record["norm_id"]) for record in
                json.loads(args.records.read_text(encoding="utf-8"))]
    files = {judge: args.judgments_root / args.model_slug / f"{judge}.jsonl" for judge in args.judges}
    absent = [str(path) for path in files.values() if not path.is_file()]
    if absent:
        raise FileNotFoundError(f"No judgments at {absent}")
    names, table = build(norm_ids, files)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, norm_ids=np.asarray(norm_ids), names=np.asarray(names), outcomes=table)
    for name, values in zip(names, table.T):
        known = values >= 0
        print(f"{name:40s} {values[known].mean():.3f}  n={int(known.sum())}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"ERROR: {error}")
