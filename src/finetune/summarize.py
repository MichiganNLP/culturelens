"""One table per evaluation: constrained accuracy and judged alignment.

    python -m finetune.summarize --eval-root .../eval --judge-slug gemma-4-31b

Reads ``<root>/<model>/<split>/constrained.jsonl`` and
``<root>/judgments/<model>-<split>/<judge>.jsonl``. For every model, split and
condition it reports:

* ``accuracy``   constrained A/B exact match -- no model graded it;
* ``aligned``    share the judge labelled aligned (strict);
* ``lenient``    aligned + half of partially aligned, the same weighting the
                 in-loop critic reward uses;
* ``coverage``   rows present over rows expected, so a gap cannot pass as a score.

A later row for the same ``(norm_id, condition)`` replaces an earlier one, which
is how the generator's answer-only recovery is recorded.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

SPLITS = ("test_seen", "test_unseen")
LENIENT = {"aligned": 1.0, "partially_aligned": 0.5, "misaligned": 0.0}


def _latest(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[(str(row["norm_id"]), str(row["condition"]))] = row
    return rows


def summarize(root: Path, judge: str) -> dict[str, Any]:
    models = sorted(p.name for p in root.iterdir()
                    if p.is_dir() and p.name != "judgments" and any((p / s).is_dir() for s in SPLITS))
    table: dict[str, Any] = {}
    for model in models:
        table[model] = {}
        for split in SPLITS:
            constrained = _latest(root / model / split / "constrained.jsonl")
            open_rows = _latest(root / model / split / "unconstrained.jsonl")
            judged = _latest(root / "judgments" / f"{model}-{split}" / f"{judge}.jsonl")
            by_condition: dict[str, dict[str, list[Any]]] = defaultdict(lambda: defaultdict(list))
            for (_, condition), row in constrained.items():
                by_condition[condition]["correct"].append(bool(row.get("correct")))
            for key in open_rows:
                by_condition[key[1]]["open"].append(key)
            for (_, condition), row in judged.items():
                by_condition[condition]["alignment"].append(row["alignment"])
            split_table = {}
            for condition, values in sorted(by_condition.items()):
                correct, labels = values["correct"], values["alignment"]
                split_table[condition] = {
                    "accuracy": sum(correct) / len(correct) if correct else None,
                    "constrained_n": len(correct),
                    "aligned": labels.count("aligned") / len(labels) if labels else None,
                    "lenient": sum(LENIENT[x] for x in labels) / len(labels) if labels else None,
                    "judged_n": len(labels),
                    "coverage": len(labels) / len(values["open"]) if values["open"] else None,
                }
            table[model][split] = split_table
    return {"judge": judge, "models": table}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--judge-slug", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    report = summarize(args.eval_root, args.judge_slug)
    args.output.write_text(json.dumps(report, indent=2))
    for model, splits in report["models"].items():
        for split, conditions in splits.items():
            for condition, m in conditions.items():
                fmt = lambda v: "  -  " if v is None else f"{100 * v:5.1f}"
                print(f"{model:40s} {split:12s} {condition:22s} acc {fmt(m['accuracy'])}  "
                      f"aligned {fmt(m['aligned'])}  lenient {fmt(m['lenient'])}  n={m['judged_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
