"""The fine-tuning tables of the paper, recomputed from the judged answers.

Prints, for each model and method, the percentage of unconstrained answers
judged aligned on seen and unseen countries under four evidence conditions
(the main table: mean of the two judges; ``--by-judge`` prints each judge),
and the request table: how often each Llama adapter does what a person in the
scenario asked, with its alignment on the conditions without the country.

PYTHONPATH=src python -m finetune.tables [--by-judge]
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

JUDGES = ("gemma-4-31b", "qwen-3.6-27b")
MODELS = (("Llama", "llama-3.1-8b", "collab"), ("Gemma", "gemma-3-12b", "collab-gemma"), ("Qwen", "qwen-3.5-27b", "collab-qwen"))
CONDITIONS = ("country_only", "country_value", "country_norm", "country_opposite_norm")
NO_COUNTRY = ("base", "norm_only", "opposite_norm_only")
ANSWER_SET = {"base": "full", "country_only": "full", "country_value": "value", "country_norm": "norms",
              "country_opposite_norm": "norms", "norm_only": "norms", "opposite_norm_only": "norms"}
REQUEST = "country_opposite_norm"   # request prompts replace the contradictory norm of this condition


def rows(path: Path, condition: str | None = None) -> dict[str, dict]:
    found: dict[str, dict] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if condition is None or row.get("condition") == condition:
                        found[row["norm_id"]] = row
    return found


def aligned(root: Path, slug: str, run: str | None, condition: str, judge: str, ids: set[str]) -> float | None:
    """Percentage judged aligned; ``run`` None is the untuned model's benchmark answers."""

    if run is None:
        judgments = root / "behavioral_evaluation" / "unconstrained" / "judgments"
        found = rows(judgments / slug / f"{judge}.jsonl", condition)
        if not found:  # the contradictory norm without the country was judged as a separate set
            found = rows(judgments / f"{slug}.{condition}" / f"{judge}.jsonl", condition)
    else:
        name = f"{ANSWER_SET[condition]}-{run}.tuned.{condition}"
        found = rows(root / "finetune" / "mechanism" / run / "judgments" / name / f"{judge}.jsonl")
    values = [found[i]["alignment"] == "aligned" for i in ids if i in found]
    return 100 * sum(values) / len(values) if values else None


def cell(values: list[float | None], by_judge: bool) -> str:
    if any(value is None for value in values):
        return "   --"
    return "/".join(f"{value:5.1f}" for value in values) if by_judge else f"{sum(values) / len(values):5.1f}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--splits", type=Path, default=Path("data/processed/culturelens/splits"))
    parser.add_argument("--by-judge", action="store_true", help="Print Gemma judge / Qwen judge instead of their mean.")
    args = parser.parse_args(argv)
    split = {name: {row["norm_id"] for row in json.loads((args.splits / f"{name}.json").read_text(encoding="utf-8"))}
             for name in ("test_seen", "test_unseen")}
    everything = split["test_seen"] | split["test_unseen"]

    print("Aligned (%): seen Country, + Value, + True norm, + Contra. norm | unseen, same order")
    for model, slug, tag in MODELS:
        for label, run in (("Untuned", None), ("SFT", f"{tag}-sft-country"), ("SFT + CF", f"{tag}-sft-cf")):
            line = f"{model:<6}{label:<10}"
            for name in ("test_seen", "test_unseen"):
                line += " ".join(cell([aligned(args.output_root, slug, run, condition, judge, split[name])
                                       for judge in JUDGES], args.by_judge) for condition in CONDITIONS) + " | "
            print(line)

    print("\nLlama with request examples: no country (None, True, Contra.), then does what was asked (%)")
    mechanism = args.output_root / "finetune" / "mechanism"
    for label, run, who in (("Untuned", None, "untuned"), ("SFT", "collab-sft-country", "tuned"),
                            ("SFT + CF", "collab-sft-cf", "tuned"), ("SFT + CF + Request", "sft-cf-request", "tuned")):
        line = f"{label:<20}" + " ".join(cell([aligned(args.output_root, "llama-3.1-8b", run, condition, judge, everything)
                                               for judge in JUDGES], args.by_judge) for condition in NO_COUNTRY)
        holder = run or "collab-sft-cf"   # the untuned request answers were generated in the SFT + CF run
        asked: list[float | None] = []
        for coder in JUDGES:
            coded = rows(mechanism / holder / "request_stance" / f"request-{holder}.{who}.{REQUEST}" / f"{coder}.jsonl")
            share = collections.Counter(row["stance"] for row in coded.values())
            asked.append(100 * (share["follows"] + share["follows_with_caveat"]) / len(coded) if coded else None)
        print(f"{line} | {cell(asked, args.by_judge)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
