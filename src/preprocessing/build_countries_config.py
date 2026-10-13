"""Build the benchmark country universe from the WVS–Cultural Atlas overlap."""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


ISO3_PATTERN = re.compile(r"[A-Z]{3}")
DEFAULT_IWM = Path("data/processed/references/wvs/iwm_coords.csv")
DEFAULT_ATLAS = Path("data/raw/atlas/atlas_feb_2026.csv")
DEFAULT_OUTPUT = Path("configs/countries_wvs.yaml")
DEFAULT_AUDIT = Path("data/processed/references/country_universe.audit.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iwm-path", type=Path, default=DEFAULT_IWM)
    parser.add_argument("--atlas-path", type=Path, default=DEFAULT_ATLAS)
    parser.add_argument("--output-path", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--audit-path", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_iso3(values: pd.Series, source: str) -> pd.Series:
    codes = values.astype("string").str.strip().str.upper()
    if codes.isna().any() or (codes == "").any():
        raise ValueError(f"{source} contains blank ISO3 values.")
    duplicates = sorted(codes[codes.duplicated()].unique().tolist())
    malformed = sorted(codes[~codes.str.fullmatch(r"[A-Z]{3}")].unique().tolist())
    if duplicates or malformed:
        raise ValueError(
            f"{source} ISO3 validation failed; duplicates={duplicates}, malformed={malformed}"
        )
    return codes


def load_iwm_coords(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"iso3", "secular", "selfexp"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
    frame["iso3"] = validate_iso3(frame["iso3"], str(path))
    for column in ("secular", "selfexp"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(frame[column]).all():
            raise ValueError(f"{path} column {column!r} contains non-finite values.")
    return frame


def load_atlas(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, keep_default_na=False)
    if "iso3" not in frame.columns:
        raise ValueError(f"Cultural Atlas CSV {path} is missing the iso3 column.")
    frame["iso3"] = validate_iso3(frame["iso3"], str(path))
    category_columns = [column for column in frame.columns if column != "iso3"]
    if not category_columns:
        raise ValueError("Cultural Atlas CSV has no category columns.")
    empty = ~frame[category_columns].apply(
        lambda row: any(isinstance(value, str) and value.strip() for value in row), axis=1
    )
    if empty.any():
        raise ValueError(f"Atlas profiles contain no text: {frame.loc[empty, 'iso3'].tolist()}")
    malformed = [
        f"{row.iso3}/{column}"
        for row in frame.itertuples(index=False)
        for column in category_columns
        if (text := getattr(row, column)).strip() and not text.lstrip().startswith("# ")
    ]
    if malformed:
        raise ValueError(
            "Atlas category cells must begin with a '# Section' heading: "
            f"{malformed[:20]}"
        )
    return frame


def quadrant_counts(benchmark: list[str], iwm: pd.DataFrame) -> dict[str, int]:
    secular_median = float(iwm["secular"].median())
    selfexp_median = float(iwm["selfexp"].median())
    selected = iwm.set_index("iso3").loc[benchmark]
    vertical = np.where(selected["secular"] >= secular_median, "Secular", "Traditional")
    horizontal = np.where(
        selected["selfexp"] >= selfexp_median, "Self-expression", "Survival"
    )
    labels = [f"{left}/{right}" for left, right in zip(vertical, horizontal)]
    return dict(sorted(pd.Series(labels).value_counts().astype(int).items()))


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    for path in (args.iwm_path, args.atlas_path):
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
    existing = [str(path) for path in (args.output_path, args.audit_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "Generated outputs already exist; pass --overwrite to replace them: "
            + ", ".join(existing)
        )

    iwm = load_iwm_coords(args.iwm_path)
    atlas = load_atlas(args.atlas_path)
    wvs_iso3 = set(iwm["iso3"])
    atlas_iso3 = set(atlas["iso3"])
    benchmark = sorted(wvs_iso3 & atlas_iso3)
    if not benchmark:
        raise ValueError("The WVS and Cultural Atlas inputs have no overlapping countries.")
    if len(benchmark) != len(set(benchmark)) or any(
        not ISO3_PATTERN.fullmatch(code) for code in benchmark
    ):
        raise ValueError("Derived benchmark country list failed validation.")

    payload = yaml.safe_dump({"countries": benchmark}, sort_keys=False)
    audit = {
        "wvs_country_count": len(wvs_iso3),
        "atlas_country_count": len(atlas_iso3),
        "intersection_count": len(benchmark),
        "countries": benchmark,
        "wvs_only": sorted(wvs_iso3 - atlas_iso3),
        "atlas_only": sorted(atlas_iso3 - wvs_iso3),
        "iwm_quadrants": quadrant_counts(benchmark, iwm),
    }
    atomic_text(args.output_path, payload)
    atomic_text(args.audit_path, json.dumps(audit, indent=2, sort_keys=True) + "\n")
    print(f"Created {args.output_path} with {len(benchmark)} WVS–Atlas countries.")
    print(f"Audit: {args.audit_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
