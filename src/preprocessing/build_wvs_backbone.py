"""Build the country-level WVS Wave 7 reference geometry.

All generated artifacts are machine-readable and are written beneath
``data/processed/references/wvs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import pdist, squareform


DEFAULT_INPUT = Path("data/raw/wvs/cross_national_wave_7.csv")
DEFAULT_DATA_OUTPUT = Path("data/processed/references/wvs")

COUNTRY_COLUMN = "B_COUNTRY_ALPHA"
SECULAR_COLUMN = "SACSECVAL"
SELF_EXPRESSION_COLUMN = "RESEMAVAL"

# WVS Wave 7 codebook labels. These 33 questionnaire items plus the two
# precomputed Inglehart-Welzel axes produce a 35-dimensional country vector.
VALUE_ITEMS: dict[str, str] = {
    "Q1": "Importance of family in life — family attachment and traditional social ties.",
    "Q5": "Importance of work in life — work centrality.",
    "Q6": "Importance of religion in life — personal religiosity.",
    "Q8": "Child quality: independence — autonomy-oriented socialization.",
    "Q11": "Child quality: imagination — autonomy and creativity.",
    "Q12": "Child quality: tolerance and respect — prosocial socialization.",
    "Q15": "Child quality: religious faith — religious socialization.",
    "Q17": "Child quality: obedience — authority-oriented socialization.",
    "Q19": "Would not want neighbors of a different race — racial tolerance.",
    "Q21": "Would not want immigrant/foreign-worker neighbors — migrant tolerance.",
    "Q23": "Would not want neighbors of a different religion — religious tolerance.",
    "Q29": "Men make better political leaders — gender-role traditionalism.",
    "Q30": "University is more important for boys — gender equality in education.",
    "Q33": "Men should have priority for scarce jobs — gender equality in work.",
    "Q45": "Greater respect for authority is good/bad — authority orientation.",
    "Q48": "Free choice and control over life — individual agency.",
    "Q57": "Most people can be trusted — generalized social trust.",
    "Q61": "Trust in people met for the first time — out-group trust.",
    "Q62": "Trust in people of another religion — religious out-group trust.",
    "Q63": "Trust in people of another nationality — national out-group trust.",
    "Q64": "Confidence in churches — confidence in religious institutions.",
    "Q106": "Income equality versus incentives — distributive economic values.",
    "Q184": "Justifiability of abortion — reproductive-choice values.",
    "Q185": "Justifiability of divorce — family and lifestyle choice.",
    "Q188": "Justifiability of euthanasia — individual choice in ethical matters.",
    "Q209": "Petition participation — non-institutional political voice.",
    "Q211": "Peaceful demonstration participation — civic action.",
    "Q235": "Strong leader without parliament/elections — authoritarian preference.",
    "Q238": "Having a democratic political system — democratic preference.",
    "Q239": "Government by religious law — religious-authority preference.",
    "Q246": "Civil rights protect against state oppression — liberal-democratic values.",
    "Q249": "Women have the same rights as men — gender equality.",
    "Q254": "Pride in nationality — nationalism and national attachment.",
}

EXPECTED_CLOSE_PAIRS = [
    ("USA", "CAN"),
    ("AUS", "NZL"),
    ("DEU", "NLD"),
    ("ARG", "URY"),
    ("CZE", "SVK"),
]
EXPECTED_FAR_PAIRS = [
    ("NLD", "EGY"),
    ("DEU", "EGY"),
    ("AUS", "PAK"),
    ("NZL", "JOR"),
    ("CAN", "IRQ"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--data-output-dir", type=Path, default=DEFAULT_DATA_OUTPUT)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing generated artifacts. Existing files are protected by default.",
    )
    return parser.parse_args()


def require_columns(frame: pd.DataFrame, required: set[str], source: str) -> None:
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source} is missing required columns: {sorted(missing)}")


def normalize_country_codes(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame[COUNTRY_COLUMN] = frame[COUNTRY_COLUMN].astype("string").str.strip().str.upper()
    frame = frame[frame[COUNTRY_COLUMN].notna() & (frame[COUNTRY_COLUMN] != "")]
    valid = frame[COUNTRY_COLUMN].str.fullmatch(r"[A-Z]{3}")
    malformed = sorted(frame.loc[~valid.fillna(False), COUNTRY_COLUMN].unique().tolist())
    if malformed:
        raise ValueError(f"WVS country codes are not three-letter ISO codes: {malformed}")
    return frame


def _ensure_targets(paths: list[Path], overwrite: bool) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Generated outputs already exist; pass --overwrite to replace them: "
            + ", ".join(existing)
        )
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)


def _atomic_writer(path: Path, mode: str = "wb"):
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
    )
    return descriptor, Path(temporary_name), mode


def _atomic_bytes(path: Path, writer) -> None:
    descriptor, temporary, mode = _atomic_writer(path)
    try:
        with os.fdopen(descriptor, mode) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_npy(path: Path, array: np.ndarray) -> None:
    _atomic_bytes(path, lambda handle: np.save(handle, array))


def save_csv(path: Path, frame: pd.DataFrame, **kwargs: object) -> None:
    descriptor, temporary, _ = _atomic_writer(path, "w")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            frame.to_csv(handle, **kwargs)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_parquet(path: Path, frame: pd.DataFrame) -> None:
    descriptor, temporary, _ = _atomic_writer(path)
    os.close(descriptor)
    try:
        frame.to_parquet(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def save_json(path: Path, payload: dict[str, object]) -> None:
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    _atomic_bytes(path, lambda handle: handle.write(encoded))


def pair_rank(
    pair: tuple[str, str], pair_distances: list[tuple[float, str, str]]
) -> tuple[float, int, float] | None:
    target = frozenset(pair)
    ordered = sorted(pair_distances)
    for rank, (distance, left, right) in enumerate(ordered, start=1):
        if frozenset((left, right)) == target:
            percentile = 100 * (rank - 1) / max(1, len(ordered) - 1)
            return distance, rank, percentile
    return None


def build_audit(
    countries: list[str], pair_distances: list[tuple[float, str, str]]
) -> dict[str, object]:
    close_results = [
        (pair, pair_rank(pair, pair_distances))
        for pair in EXPECTED_CLOSE_PAIRS
        if pair[0] in countries and pair[1] in countries
    ]
    far_results = [
        (pair, pair_rank(pair, pair_distances))
        for pair in EXPECTED_FAR_PAIRS
        if pair[0] in countries and pair[1] in countries
    ]
    close_ok = sum(result is not None and result[2] <= 25 for _, result in close_results)
    far_ok = sum(result is not None and result[2] >= 75 for _, result in far_results)

    face_checks: list[dict[str, object]] = []
    for expectation, results in (("near", close_results), ("far", far_results)):
        for (left, right), result in results:
            assert result is not None
            distance, rank, percentile = result
            face_checks.append(
                {
                    "expectation": expectation,
                    "left": left,
                    "right": right,
                    "distance": distance,
                    "rank": rank,
                    "percentile": percentile,
                }
            )
    passed = close_ok >= 3 and far_ok >= 3
    return {
        "countries": len(countries),
        "dimensions": len(VALUE_ITEMS) + 2,
        "questionnaire_items": len(VALUE_ITEMS),
        "aggregation": "unweighted_country_mean",
        "standardization": "country_level_zscore_ddof_0",
        "distance": "cosine",
        "linkage": "ward_euclidean",
        "face_validity_passed": passed,
        "expected_near_in_closest_quartile": close_ok,
        "expected_far_in_farthest_quartile": far_ok,
        "face_checks": face_checks,
    }


def main() -> int:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"WVS input file not found: {args.input}")

    data_paths = {
        "vectors": args.data_output_dir / "country_vectors.parquet",
        "coords": args.data_output_dir / "iwm_coords.csv",
        "distance_npy": args.data_output_dir / "distance_matrix.npy",
        "distance_csv": args.data_output_dir / "distance_matrix.csv",
        "linkage": args.data_output_dir / "linkage.npy",
        "audit": args.data_output_dir / "audit.json",
    }
    _ensure_targets(list(data_paths.values()), args.overwrite)

    frame = pd.read_csv(args.input, low_memory=False)
    selected_columns = [SECULAR_COLUMN, SELF_EXPRESSION_COLUMN, *VALUE_ITEMS]
    require_columns(frame, {COUNTRY_COLUMN, *selected_columns}, str(args.input))
    frame = normalize_country_codes(frame)

    values = frame[selected_columns].apply(pd.to_numeric, errors="coerce")
    negative_count = int(values.lt(0).sum().sum())
    values = values.mask(values.lt(0))
    cleaned = pd.concat([frame[[COUNTRY_COLUMN]], values], axis=1)

    country_means = cleaned.groupby(COUNTRY_COLUMN, sort=True)[selected_columns].mean()
    country_means.index.name = "iso3"
    if country_means.isna().any().any():
        failures = {
            column: country_means.index[country_means[column].isna()].tolist()
            for column in country_means.columns[country_means.isna().any()]
        }
        raise ValueError(f"Selected items lack valid responses for countries: {failures}")

    standard_deviations = country_means.std(axis=0, ddof=0)
    zero_variance = standard_deviations[standard_deviations == 0].index.tolist()
    if zero_variance:
        raise ValueError(f"Cannot z-score zero-variance columns: {zero_variance}")
    country_vectors = (country_means - country_means.mean(axis=0)) / standard_deviations
    country_vectors.index.name = "iso3"

    coords = country_means[[SECULAR_COLUMN, SELF_EXPRESSION_COLUMN]].rename(
        columns={SECULAR_COLUMN: "secular", SELF_EXPRESSION_COLUMN: "selfexp"}
    )
    vector_array = country_vectors.to_numpy(dtype=float)
    labels = country_vectors.index.tolist()
    distances = squareform(pdist(vector_array, metric="cosine"))
    np.fill_diagonal(distances, 0.0)
    linkage_matrix = linkage(vector_array, method="ward", metric="euclidean")

    pair_distances = [
        (float(distances[i, j]), labels[i], labels[j])
        for i in range(len(labels))
        for j in range(i + 1, len(labels))
    ]
    audit = build_audit(labels, pair_distances)
    audit.update(
        {
            "input": str(args.input),
            "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
            "input_rows": len(frame),
            "negative_missing_codes_removed": negative_count,
            "country_codes": labels,
        }
    )

    save_parquet(data_paths["vectors"], country_vectors)
    save_csv(data_paths["coords"], coords.reset_index(), index=False)
    save_npy(data_paths["distance_npy"], distances)
    save_csv(
        data_paths["distance_csv"],
        pd.DataFrame(distances, index=labels, columns=labels),
        index_label="iso3",
    )
    save_npy(data_paths["linkage"], linkage_matrix)
    save_json(data_paths["audit"], audit)

    print(f"Built a {len(labels)}-country, {country_vectors.shape[1]}-dimensional WVS backbone.")
    print(f"Numeric outputs: {args.data_output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
