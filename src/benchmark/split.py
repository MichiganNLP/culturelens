"""Create deterministic train, validation, seen-test, and unseen-test splits.

The splitter samples records independently rather than grouping semantically.
Every seen country contributes exactly ``--test-per-country`` records to the
seen-country test. Countries listed in ``--unseen-countries`` contribute all
their records to the unseen-country test and never enter training or validation.

Validation contains records only from high-resource seen countries. Unseen
countries remain absent from both training and model selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "data/processed/culturelens/records.json"
DEFAULT_OUTPUT_DIR = ROOT / "data/processed/culturelens/splits"
DEFAULT_UNSEEN_COUNTRIES = (
    "ARG",
    "DEU",
    "EGY",
    "ETH",
    "IDN",
    "IND",
    "JPN",
    "RUS",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-per-country", type=int, default=25)
    parser.add_argument("--validation-per-country", type=int, default=10)
    parser.add_argument(
        "--validation-min-country-items",
        type=int,
        default=100,
        help="Seen countries at or above this total size contribute validation records.",
    )
    parser.add_argument(
        "--unseen-countries",
        default=",".join(DEFAULT_UNSEEN_COUNTRIES),
        help="Comma-separated ISO3 codes held out from training.",
    )
    return parser.parse_args(argv)


def read_json(path: Path) -> list[dict[str, Any]]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list) or not all(isinstance(row, dict) for row in value):
        raise ValueError(f"{path} must contain a JSON array of objects")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def record_id(row: dict[str, Any]) -> str:
    value = row.get("norm_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Every record must have a non-empty norm_id")
    return value.strip()


def country_code(row: dict[str, Any]) -> str:
    identifier = record_id(row)
    code, separator, _ = identifier.partition("__")
    if not separator or len(code) != 3 or not code.isalpha():
        raise ValueError(f"Cannot recover ISO3 country code from norm_id={identifier!r}")
    return code.upper()


def rank(seed: int, identifier: str) -> bytes:
    """Return a stable pseudorandom rank independent of input ordering."""
    return hashlib.sha256(f"{seed}:{identifier}".encode()).digest()


def sorted_sample(rows: list[dict[str, Any]], seed: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (rank(seed, record_id(row)), record_id(row)))


def parse_country_list(value: str) -> frozenset[str]:
    countries = frozenset(part.strip().upper() for part in value.split(",") if part.strip())
    if not countries or any(len(country) != 3 for country in countries):
        raise ValueError("--unseen-countries must be a non-empty comma-separated ISO3 list")
    return countries


def country_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(country_code(row) for row in rows).items()))


def ensure_disjoint(splits: dict[str, list[dict[str, Any]]]) -> None:
    owner: dict[str, str] = {}
    for split_name, rows in splits.items():
        for row in rows:
            identifier = record_id(row)
            if identifier in owner:
                raise RuntimeError(
                    f"Record {identifier} appears in both {owner[identifier]} and {split_name}"
                )
            owner[identifier] = split_name


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.test_per_country < 1:
        raise ValueError("--test-per-country must be positive")
    if args.validation_per_country < 0:
        raise ValueError("Validation sample size cannot be negative")
    if args.validation_min_country_items < 1:
        raise ValueError("--validation-min-country-items must be positive")

    input_path = args.input.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    rows = read_json(input_path)
    identifiers = [record_id(row) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        duplicates = [key for key, count in Counter(identifiers).items() if count > 1]
        raise ValueError(f"Input contains duplicate norm_id values: {duplicates[:5]}")

    by_country: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_country[country_code(row)].append(row)
    unseen = parse_country_list(args.unseen_countries)
    missing_unseen = unseen - set(by_country)
    if missing_unseen:
        raise ValueError(f"Unseen countries absent from input: {sorted(missing_unseen)}")

    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    test_seen: list[dict[str, Any]] = []
    test_unseen: list[dict[str, Any]] = []
    high_resource_seen: list[str] = []

    for country in sorted(by_country):
        country_rows = sorted_sample(by_country[country], args.seed)
        if len(country_rows) < args.test_per_country:
            raise ValueError(
                f"{country} has {len(country_rows)} records, fewer than "
                f"--test-per-country={args.test_per_country}"
            )

        if country in unseen:
            test_unseen.extend(country_rows)
            continue

        test_rows = country_rows[: args.test_per_country]
        remaining = country_rows[args.test_per_country :]
        test_seen.extend(test_rows)
        if len(country_rows) >= args.validation_min_country_items:
            high_resource_seen.append(country)
            validation_size = args.validation_per_country
            if len(remaining) < validation_size:
                raise ValueError(f"{country} has insufficient records for validation")
            validation.extend(remaining[:validation_size])
            remaining = remaining[validation_size:]
        train.extend(remaining)

    splits = {
        "train": train,
        "validation": validation,
        "test_seen": test_seen,
        "test_unseen": test_unseen,
    }
    ensure_disjoint(splits)

    train_countries = set(country_counts(train))
    if train_countries & unseen:
        raise RuntimeError("An unseen country leaked into training")
    if set(country_counts(test_seen)) & unseen:
        raise RuntimeError("An unseen country leaked into the seen-country test")
    if set(country_counts(test_unseen)) != set(unseen):
        raise RuntimeError("The unseen-country test does not cover the requested countries")
    seen_test_counts = country_counts(test_seen)
    if any(count != args.test_per_country for count in seen_test_counts.values()):
        raise RuntimeError(
            f"test_seen is not balanced at {args.test_per_country} per country"
        )

    for split_name, split_rows in splits.items():
        write_json(output_dir / f"{split_name}.json", split_rows)

    assigned = sum(len(split_rows) for split_rows in splits.values())
    summary = {
        "input": str(input_path),
        "seed": args.seed,
        "total_input_records": len(rows),
        "total_countries": len(by_country),
        "unseen_countries": sorted(unseen),
        "high_resource_seen_countries": high_resource_seen,
        "policy": {
            "test_per_country": args.test_per_country,
            "validation_per_high_resource_seen_country": args.validation_per_country,
            "validation_min_country_items": args.validation_min_country_items,
            "semantic_grouping": False,
            "unseen_test_uses_all_country_records": True,
            "unseen_test_fully_blind_to_model_selection": True,
        },
        "splits": {
            name: {
                "records": len(split_rows),
                "countries": len(country_counts(split_rows)),
                "records_by_country": country_counts(split_rows),
            }
            for name, split_rows in splits.items()
        },
        "assigned_records": assigned,
        "accounted_records": assigned,
    }
    if summary["accounted_records"] != len(rows):
        raise RuntimeError("Split accounting does not equal the input size")
    write_json(output_dir / "summary.json", summary)

    print(f"Input: {len(rows)} records across {len(by_country)} countries")
    for name, split_rows in splits.items():
        print(f"{name}: {len(split_rows)} records, {len(country_counts(split_rows))} countries")
    print(f"Output: {output_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        raise SystemExit(f"ERROR: {exc}")
