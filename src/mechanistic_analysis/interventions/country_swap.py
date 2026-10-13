"""Records with the country line of a different, distant country.

A control for what a model does with the country it is told. Each record keeps
its situation, question, norms and options, and its ``norm_id`` (so a judge
still scores answers against the record's own documented norm), but its
``country_context`` names another country. Comparing a prompt with the right
country, with this wrong one, and with none separates what the model knows
about *that* country from the effect of naming *a* country.

The other country is drawn, per record and from a seed, among the ``--farthest``
countries most distant from the record's own. Distances are the cosine
distances between country vectors of an untuned model
(``mechanistic_analysis.country_space``, ``rdm_shift`` at one layer), so a
record is not handed a culturally close neighbour whose norms may coincide.

PYTHONPATH=src python -m mechanistic_analysis.interventions.country_swap \
  --records output/finetune/mechanism/test_all.json \
  --all-records data/processed/culturelens/records.json \
  --geometry output/mechanistic_analysis/country_space/llama-3.1-8b/geometry.npz \
  --output output/finetune/mechanism/test_all_swapped.json
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any


def iso3(record: dict[str, Any]) -> str:
    return str(record["norm_id"]).split("__")[0]


def country_lines(records: list[dict[str, Any]]) -> dict[str, str]:
    """``iso3 -> country_context``; a country's records must agree on its line."""

    lines: dict[str, str] = {}
    for record in records:
        code, line = iso3(record), record["scenario"]["country_context"]
        if lines.setdefault(code, line) != line:
            raise ValueError(f"{code} has more than one country line")
    return lines


def swap(records: list[dict[str, Any]], lines: dict[str, str], countries: list[str], distance: Any,
         farthest: int, seed: int) -> list[dict[str, Any]]:
    position = {code: index for index, code in enumerate(countries)}
    out = []
    for record in records:
        own = iso3(record)
        ranked = sorted((code for code in lines if code != own and code in position),
                        key=lambda code: -float(distance[position[own]][position[code]]))[:farthest]
        if not ranked:
            raise ValueError(f"No other country to give {record['norm_id']}")
        other = random.Random(f"{seed}:{record['norm_id']}").choice(ranked)
        changed = copy.deepcopy(record)
        changed["scenario"]["country_context"] = lines[other]
        changed["swapped_iso3"] = other
        out.append(changed)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--all-records", type=Path, required=True,
                        help="Every record of the benchmark, for each country's line.")
    parser.add_argument("--geometry", type=Path, required=True)
    parser.add_argument("--geometry-layer", type=int, default=16)
    parser.add_argument("--farthest", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import numpy as np

    geometry = np.load(args.geometry)
    layer = [int(value) for value in geometry["layers"]].index(args.geometry_layer)
    records = json.loads(args.records.read_text(encoding="utf-8"))
    swapped = swap(records, country_lines(json.loads(args.all_records.read_text(encoding="utf-8"))),
                   [str(code) for code in geometry["countries"]], geometry["rdm_shift"][layer],
                   args.farthest, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(swapped, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {len(swapped)} records to {args.output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
