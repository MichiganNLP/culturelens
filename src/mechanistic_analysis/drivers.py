"""What drives resisting a false norm, from the judged answers alone.

The benchmark is a factorial design over what the prompt states (country,
value, true norm, false norm), with every record answered under each
condition. This module reads the judged open-ended answers and reports

1. the effect of each cue on alignment, paired on records (the average
   treatment effect of adding that cue), and how cues interact;
2. what predicts resisting a false norm across records: a logistic regression
   on what the model answers without the false norm and on the record's
   category;
3. whether resistance is a property of the record or of the model: agreement
   between models on which records are resisted;
4. the same by country: which countries' norms survive a false claim, and
   whether that follows what the model knows from the country alone;
5. how the answers treat the false claim (behavioral_evaluation/stance.py),
   crossed with the alignment label, on the full benchmark and under the
   mechanistic interventions;
6. what the framing of the false claim does (hearsay, a guide, the user's
   doubt, order), on the confirmation sample.

No model is run. Intervals are 95% bootstrap intervals over records.

PYTHONPATH=src python -m mechanistic_analysis.drivers --draws 2000
"""

from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np

from mechanistic_analysis import JUDGES, REPORT_LAYER, RESIST
from mechanistic_analysis.stats import bootstrap, paired_difference, rate_with_interval

MODELS = {"llama-3.1-8b": "Llama-3.1-8B", "gemma-3-12b": "Gemma-3-12B", "qwen-3.5-27b": "Qwen-3.5-27B"}
CLAIM_ONLY = "opposite_norm_only"
# name, condition with the cue, condition without it, which records
EFFECTS = (
    ("Country, nothing else stated", "country_only", "base", "all"),
    ("Country, value stated", "country_value", "value_only", "value"),
    ("Country, true norm stated", "country_norm", "norm_only", "all"),
    ("Country, false norm stated", RESIST, CLAIM_ONLY, "all"),
    ("Value, nothing else stated", "value_only", "base", "value"),
    ("Value, country stated", "country_value", "country_only", "value"),
    ("Value, country and true norm stated", "country_value_norm", "country_norm", "value"),
    ("True norm, nothing else stated", "norm_only", "base", "all"),
    ("True norm, country stated", "country_norm", "country_only", "all"),
    ("False norm, nothing else stated", CLAIM_ONLY, "base", "all"),
    ("False norm, country stated", RESIST, "country_only", "all"),
)
STANCES = ("follows", "follows_with_caveat", "challenges", "sets_aside", "no_clear_recommendation")
STANCE_LABELS = {"follows": "Follows the claim", "follows_with_caveat": "Follows it, with doubt",
                 "challenges": "Challenges the claim", "sets_aside": "Sets the claim aside",
                 "no_clear_recommendation": "No clear recommendation"}
STANCE_ARMS = {
    "false": "Country + false norm, untouched",
    "false-noclaim-win": "… claim blocked in the claim window",
    "false-noclaim-heads": "… claim blocked for the selected heads",
    "false-noclaim": "… claim blocked in every layer",
    "false-erase-before": "… claim erased before the band",
    "false-erase-first": "… claim erased after the first block",
    "claim-only": "False norm, no country",
    "country": "Country only (claim never shown)",
}
FRAMINGS = {
    "country_opposite_norm_guide": "Attributed to a guide to local customs",
    "opposite_norm_country": "Stated before the country line",
    "country_opposite_norm_hearsay": "Attributed to hearsay (\"Someone told me this\")",
    "country_opposite_norm_doubt": "Followed by the user's doubt (\"I am not sure this is right\")",
}
BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"


# ------------------------------------------------------------------- inputs --


def read_labels(path: Path) -> dict[str, dict[str, str]]:
    """``condition -> norm_id -> alignment label`` from one judgment file."""

    found: dict[str, dict[str, str]] = {}
    if not path.is_file():
        return found
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                found.setdefault(str(row["condition"]), {})[str(row["norm_id"])] = str(row["alignment"])
    return found


def load_model(judgments: Path, slug: str, judge: str, norm_ids: list[str]) -> dict[str, np.ndarray]:
    """Per condition, 1 aligned / 0 not / -1 no judged answer, in ``norm_ids`` order;
    ``label:<condition>`` keeps the three-way label."""

    labels = read_labels(judgments / slug / f"{judge}.jsonl")
    labels.update(read_labels(judgments / f"{slug}.{CLAIM_ONLY}" / f"{judge}.jsonl"))
    out: dict[str, np.ndarray] = {}
    for condition, found in labels.items():
        out[condition] = np.array([-1 if k not in found else int(found[k] == "aligned")
                                   for k in norm_ids], dtype=np.int8)
        out[f"label:{condition}"] = np.array([found.get(k, "") for k in norm_ids])
    return out


# ----------------------------------------------------------------- analyses --


def cue_effects(table: dict[str, np.ndarray], has_value: np.ndarray, draws: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, with_cue, without, scope in EFFECTS:
        if with_cue not in table or without not in table:
            continue
        keep = (table[with_cue] >= 0) & (table[without] >= 0)
        if scope == "value":
            keep &= has_value
        if keep.sum() >= 50:
            out[name] = {**paired_difference(table[with_cue][keep], table[without][keep], draws=draws),
                         "with": float(table[with_cue][keep].mean()),
                         "without": float(table[without][keep].mean())}
    # Does the country add more once the true norm is stated, or less?
    needed = ("base", "country_only", "norm_only", "country_norm")
    if all(c in table for c in needed):
        keep = np.all([table[c] >= 0 for c in needed], axis=0)
        t = {c: table[c][keep].astype(np.float64) for c in needed}
        out["Country × true norm (interaction)"] = paired_difference(
            t["country_norm"] - t["norm_only"], t["country_only"] - t["base"], draws=draws)
    return out


def outcome_shares(table: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out = {}
    for condition in ("country_only", RESIST, CLAIM_ONLY):
        labels = table.get(f"label:{condition}")
        if labels is not None and (labels != "").any():
            judged = labels[labels != ""]
            out[condition] = {name: float((judged == name).mean())
                              for name in ("aligned", "partially_aligned", "misaligned")}
    return out


def by_group(table: dict[str, np.ndarray], group: np.ndarray, draws: int) -> dict[str, Any]:
    """Per group: resisting, the country-only rate, and the drop the false norm causes."""

    out = {}
    keep = (table[RESIST] >= 0) & (table["country_only"] >= 0)
    for name in sorted(set(group.tolist())):
        rows = keep & (group == name)
        if rows.sum() >= 20:
            out[name] = {"n": int(rows.sum()),
                         "resist": rate_with_interval(table[RESIST][rows], draws=draws),
                         "country_only": float(table["country_only"][rows].mean()),
                         "resist_given_right": float(table[RESIST][rows & (table["country_only"] == 1)].mean())
                         if (rows & (table["country_only"] == 1)).sum() >= 10 else float("nan")}
    return out


def fit_logistic(x: np.ndarray, y: np.ndarray, steps: int = 60) -> np.ndarray:
    """Logistic regression by Newton's method with a negligible ridge (for separation);
    the first column of ``x`` must be the intercept."""

    beta = np.zeros(x.shape[1])
    ridge = 1e-4 * np.eye(x.shape[1])
    ridge[0, 0] = 0.0
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -30, 30)))
        step = np.linalg.solve((x * (p * (1 - p))[:, None]).T @ x + ridge, x.T @ (y - p) - ridge @ beta)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def resisting_model(table: dict[str, np.ndarray], categories: np.ndarray, has_value: np.ndarray,
                    countries: np.ndarray, draws: int) -> dict[str, Any] | None:
    """Odds ratios for resisting: what the model answers without the false norm,
    whether the record has a value, and its category (against the largest)."""

    needed = ("base", "country_only", "norm_only", "country_norm", RESIST)
    if not all(c in table for c in needed):
        return None
    keep = np.all([table[c] >= 0 for c in needed], axis=0)
    names = ["Right with no country and no norm", "Right with the country alone",
             "Right with the true norm alone", "Right with country + true norm", "Record has a value"]
    columns = [table["base"], table["country_only"], table["norm_only"], table["country_norm"], has_value]
    kinds, counts = np.unique(categories[keep], return_counts=True)
    reference = kinds[counts.argmax()]
    for kind in kinds:
        if kind != reference:
            names.append(f"Category: {kind} (vs {reference})")
            columns.append(categories == kind)
    x = np.column_stack([np.ones(keep.sum())] + [np.asarray(c)[keep].astype(np.float64) for c in columns])
    y = table[RESIST][keep].astype(np.float64)
    beta = fit_logistic(x, y)
    drawn = []
    generator = np.random.default_rng(0)
    # Records of one country are drawn together.
    groups: dict[str, list[int]] = {}
    for index, name in enumerate(countries[keep].tolist()):
        groups.setdefault(name, []).append(index)
    keys = list(groups)
    for _ in range(min(draws, 500)):
        rows = np.concatenate([groups[keys[k]] for k in generator.integers(0, len(keys), len(keys))])
        drawn.append(fit_logistic(x[rows], y[rows]))
    low, high = np.percentile(np.array(drawn), [2.5, 97.5], axis=0)
    return {"records": int(keep.sum()), "reference_category": str(reference),
            "odds_ratio": {name: {"or": float(np.exp(beta[i + 1])), "low": float(np.exp(low[i + 1])),
                                  "high": float(np.exp(high[i + 1]))} for i, name in enumerate(names)}}


def kappa(a: np.ndarray, b: np.ndarray) -> float:
    observed = float((a == b).mean())
    expected = float(a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean()))
    return (observed - expected) / (1 - expected) if expected < 1 else float("nan")


def cross_model(tables: dict[str, dict[str, np.ndarray]], draws: int) -> dict[str, Any] | None:
    """Do models resist on the same records?"""

    slugs = [s for s in tables if RESIST in tables[s]]
    if len(slugs) < 2:
        return None
    keep = np.all([tables[s][RESIST] >= 0 for s in slugs], axis=0)
    resist = {s: tables[s][RESIST][keep].astype(np.int64) for s in slugs}
    pairs = {}
    for a, b in combinations(slugs, 2):
        low, high = bootstrap(lambda rows, a=a, b=b: kappa(resist[a][rows], resist[b][rows]),
                              int(keep.sum()), draws=min(draws, 1000))
        pairs[f"{a} / {b}"] = {"kappa": kappa(resist[a], resist[b]), "low": low, "high": high}
    count = np.sum([resist[s] for s in slugs], axis=0)
    rates = [resist[s].mean() for s in slugs]
    # Distribution of the number of resisting models if models were independent.
    independent = np.zeros(len(slugs) + 1)
    independent[0] = 1.0
    for rate in rates:
        independent = np.concatenate([independent[:1] * (1 - rate),
                                      independent[1:] * (1 - rate) + independent[:-1] * rate])
    return {"records": int(keep.sum()), "models": slugs, "pairwise_kappa": pairs,
            "resisted_by": {str(k): float((count == k).mean()) for k in range(len(slugs) + 1)},
            "if_independent": {str(k): float(independent[k]) for k in range(len(slugs) + 1)}}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda v: np.argsort(np.argsort(v)).astype(np.float64)  # noqa: E731
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def external(references: Path | None, names: list[str]) -> dict[str, dict[str, float | str]]:
    """Per-country covariates that do not come from any model: world region, the two
    World Values Survey axes, and WVS distance from the United States."""

    out: dict[str, dict[str, float | str]] = {}
    if references is None:
        return out
    geography = references / "geography" / "countries.csv"
    if geography.is_file():
        with geography.open(encoding="utf-8") as handle:
            out["region"] = {row["iso3"]: row["region"] for row in csv.DictReader(handle)
                             if row["iso3"] in names}
    axes = references / "wvs" / "iwm_coords.csv"
    if axes.is_file():
        with axes.open(encoding="utf-8") as handle:
            rows = [row for row in csv.DictReader(handle) if row["iso3"] in names]
        out["secular"] = {row["iso3"]: float(row["secular"]) for row in rows}
        out["selfexp"] = {row["iso3"]: float(row["selfexp"]) for row in rows}
    distances = references / "wvs" / "distance_matrix.csv"
    if distances.is_file():
        with distances.open(encoding="utf-8") as handle:
            out["distance_from_usa"] = {row["iso3"]: float(row["USA"]) for row in csv.DictReader(handle)
                                        if row["iso3"] in names and row["iso3"] != "USA" and "USA" in row}
    return out


def conflict_by_country(root: Path, slug: str, norm_ids: list[str], countries: np.ndarray,
                        names: list[str]) -> np.ndarray | None:
    """Mean of the label-free conflict feature (interaction.py) per country, at the report layer."""

    path = root / "interaction" / slug / "features.npz"
    if not path.is_file() or slug not in REPORT_LAYER:
        return None
    saved = np.load(path)
    layers = saved["layers"].tolist()
    if REPORT_LAYER[slug] not in layers or [str(x) for x in saved["norm_ids"]] != norm_ids:
        return None
    feature = saved["conflict"][layers.index(REPORT_LAYER[slug])]
    return np.array([feature[countries == c].mean() for c in names])


def country_level(tables: dict[str, dict[str, np.ndarray]], countries: np.ndarray,
                  draws: int, root: Path | None = None, norm_ids: list[str] | None = None,
                  references: Path | None = None) -> dict[str, Any]:
    names = sorted(set(countries.tolist()))
    out: dict[str, Any] = {"countries": names, "models": {}}
    covariates = external(references, names)
    profile = {}
    for slug, table in tables.items():
        if RESIST not in table or "country_only" not in table:
            continue
        keep = (table[RESIST] >= 0) & (table["country_only"] >= 0)
        resist = np.array([table[RESIST][keep & (countries == c)].mean() for c in names])
        known = np.array([table["country_only"][keep & (countries == c)].mean() for c in names])
        right = keep & (table["country_only"] == 1)
        held = np.array([table[RESIST][right & (countries == c)].mean() for c in names])
        profile[slug] = resist
        low, high = bootstrap(lambda rows: spearman(resist[rows], known[rows]), len(names),
                              draws=min(draws, 1000))
        order = np.argsort(resist)
        out["models"][slug] = {
            "resist": resist.tolist(), "country_only": known.tolist(), "resist_given_right": held.tolist(),
            "range": [float(resist.min()), float(resist.max())],
            "spearman_resist_vs_country_only": {"rho": spearman(resist, known), "low": low, "high": high},
            "spearman_held_vs_country_only": spearman(held, known),
            "lowest": [(names[i], float(resist[i])) for i in order[:5]],
            "highest": [(names[i], float(resist[i])) for i in order[::-1][:5]]}
        entry = out["models"][slug]
        for name in ("selfexp", "secular", "distance_from_usa"):
            if name in covariates:
                rows = np.array([i for i, c in enumerate(names) if c in covariates[name]])
                x = np.array([covariates[name][names[i]] for i in rows])
                low, high = bootstrap(lambda pick, x=x, rows=rows: spearman(x[pick], resist[rows][pick]),
                                      len(rows), draws=min(draws, 1000))
                entry[f"spearman_{name}"] = {"rho": spearman(x, resist[rows]), "low": low, "high": high,
                                             "countries": int(len(rows)),
                                             "rho_country_only": spearman(x, known[rows])}
        if "region" in covariates:
            region = np.array([covariates["region"].get(c, "") for c in countries.tolist()])
            entry["by_region"] = {
                name: {"countries": int(sum(covariates["region"].get(c) == name for c in names)),
                       "resist": float(table[RESIST][keep & (region == name)].mean()),
                       "country_only": float(table["country_only"][keep & (region == name)].mean()),
                       "resist_given_right": float(table[RESIST][right & (region == name)].mean())}
                for name in sorted(set(region.tolist())) if name and (keep & (region == name)).sum() >= 50}
        conflict = conflict_by_country(root, slug, norm_ids, countries, names) if root and norm_ids else None
        if conflict is not None:
            low, high = bootstrap(lambda rows: spearman(conflict[rows], resist[rows]), len(names),
                                  draws=min(draws, 1000))
            out["models"][slug]["spearman_conflict_vs_resist"] = {
                "rho": spearman(conflict, resist), "low": low, "high": high}
            out["models"][slug]["spearman_conflict_vs_country_only"] = spearman(conflict, known)
    out["between_models"] = {f"{a} / {b}": spearman(profile[a], profile[b])
                             for a, b in combinations(profile, 2)}
    return out


def read_stance(path: Path) -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    found[str(row["norm_id"])] = row
    return found


def stance_summary(coded: dict[str, dict[str, Any]], aligned: dict[str, int] | None,
                   draws: int) -> dict[str, Any] | None:
    if len(coded) < 20:
        return None
    keys = sorted(coded)
    stance = np.array([coded[k]["stance"] for k in keys])
    out: dict[str, Any] = {"n": len(keys),
                           "share": {s: rate_with_interval(stance == s, draws=draws) for s in STANCES},
                           "restates_claim": float(np.mean([coded[k]["restates_claim"] for k in keys])),
                           "attributes_to_country": float(np.mean([coded[k]["attributes_to_country"]
                                                                   for k in keys]))}
    if aligned:
        shared = [i for i, k in enumerate(keys) if k in aligned]
        y = np.array([aligned[keys[i]] for i in shared])
        s = stance[shared]
        out["aligned_given_stance"] = {name: float(y[s == name].mean()) for name in STANCES
                                       if (s == name).sum() >= 10}
        out["stance_given_aligned"] = {name: float((s[y == 1] == name).mean()) for name in STANCES}
        out["stance_given_not_aligned"] = {name: float((s[y == 0] == name).mean()) for name in STANCES}
    return out


def stances(root: Path, behavioural: Path, slug: str, table: dict[str, np.ndarray],
            norm_ids: list[str], draws: int) -> dict[str, Any]:
    coder = JUDGES[0]
    out: dict[str, Any] = {}
    full = read_stance(behavioural / "stance" / slug / f"{coder}.jsonl")
    aligned = {k: int(v) for k, v in zip(norm_ids, table.get(RESIST, []), strict=False) if v >= 0}
    summary = stance_summary(full, aligned, draws)
    if summary:
        out["benchmark"] = summary
    for arm in STANCE_ARMS:
        coded = read_stance(root / "stance" / f"{slug}.{arm}" / f"{coder}.jsonl")
        judged = {}
        path = root / "judgments" / f"{slug}.{arm}" / f"{coder}.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                judged = {str(r["norm_id"]): int(r["alignment"] == "aligned")
                          for r in map(json.loads, filter(str.strip, handle))}
        summary = stance_summary(coded, judged, draws)
        if summary:
            out.setdefault("arms", {})[arm] = summary
    return out


def framing(behavioural: Path, slug: str, table: dict[str, np.ndarray], norm_ids: list[str],
            draws: int, judge: str = JUDGES[0]) -> dict[str, Any] | None:
    """Resisting under each framing of the false claim, against the bare statement
    on the same records; ``table`` must hold the same judge's benchmark labels."""

    coder = JUDGES[0]
    found = read_labels(behavioural / "framing" / "judgments" / slug / f"{judge}.jsonl")
    if not found or RESIST not in table:
        return None
    bare = {k: int(v) for k, v in zip(norm_ids, table[RESIST], strict=True) if v >= 0}
    coded: dict[str, dict[str, Any]] = {}
    path = behavioural / "framing" / "stance" / slug / f"{coder}.jsonl"
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for row in map(json.loads, filter(str.strip, handle)):
                coded.setdefault(str(row["condition"]), {})[str(row["norm_id"])] = row
    out: dict[str, Any] = {}
    for condition in FRAMINGS:
        labels = {k: int(v == "aligned") for k, v in found.get(condition, {}).items() if k in bare}
        if len(labels) < 20:
            continue
        keys = sorted(labels)
        framed, plain = np.array([labels[k] for k in keys]), np.array([bare[k] for k in keys])
        out[condition] = {"rate": rate_with_interval(framed, draws=draws), "bare": float(plain.mean()),
                          "effect": paired_difference(framed, plain, draws=draws),
                          "stance": stance_summary(coded.get(condition, {}), labels, draws)}
    return out or None


# ------------------------------------------------------------------- output --


def build(records_path: Path, behavioural: Path, root: Path, draws: int,
          references: Path | None = None) -> dict[str, Any]:
    records = json.loads(records_path.read_text(encoding="utf-8"))
    norm_ids = [str(r["norm_id"]) for r in records]
    countries = np.array([k.split("__")[0] for k in norm_ids])
    categories = np.array([k.split("__")[1] for k in norm_ids])
    has_value = np.array([bool(r.get("values")) for r in records])
    report: dict[str, Any] = {"judges": list(JUDGES), "bootstrap_draws": draws, "models": {}}
    tables = {}
    for slug in MODELS:
        table = load_model(behavioural / "judgments", slug, JUDGES[0], norm_ids)
        if RESIST not in table:
            continue
        tables[slug] = table
        second = load_model(behavioural / "judgments", slug, JUDGES[1], norm_ids)
        report["models"][slug] = {
            "effects": cue_effects(table, has_value, draws),
            "effects_second_judge": cue_effects(second, has_value, draws) if second else {},
            "outcome_shares": outcome_shares(table),
            "by_category": by_group(table, categories, draws),
            "resisting_model": resisting_model(table, categories, has_value, countries, draws),
            "stance": stances(root, behavioural, slug, table, norm_ids, draws),
            "framing": framing(behavioural, slug, table, norm_ids, draws),
            "framing_second_judge": framing(behavioural, slug, second, norm_ids, draws, JUDGES[1]),
        }
    if tables:
        report["cross_model"] = cross_model(tables, draws)
        report["countries"] = country_level(tables, countries, draws, root, norm_ids, references)
    return report


def _pct(value: float) -> str:
    return "–" if value is None or np.isnan(value) else f"{100 * value:.1f}%"


def _pp(entry: dict[str, float] | None) -> str:
    if not entry:
        return "–"
    p = "p<0.001" if entry["p"] < 0.001 else f"p={entry['p']:.3g}"
    return f"{100 * entry['difference']:+.1f} [{100 * entry['low']:+.1f}, {100 * entry['high']:+.1f}], {p}"


def markdown(report: dict[str, Any]) -> str:
    present = list(report["models"])
    if not present:
        return "# Behavioural drivers: no judged answers found\n"
    get = lambda slug, *path: _dig(report["models"][slug], path)  # noqa: E731
    head = "| | " + " | ".join(MODELS[s] for s in present) + " |\n|---|" + "---|" * len(present)
    lines = ["# What drives resisting a false norm: tables", "",
             f"Judged open-ended answers, full benchmark. Primary judge {JUDGES[0]}; intervals are "
             "95% bootstrap intervals over records.", "",
             "## 1. Effect of each cue on alignment (percentage points, paired on records)", "", head]
    names = [name for name, *_ in EFFECTS] + ["Country × true norm (interaction)"]
    for name in names:
        if any(get(s, "effects", name) for s in present):
            lines.append(f"| {name} | " + " | ".join(_pp(get(s, "effects", name)) for s in present) + " |")
    lines.extend(["", f"Second judge ({JUDGES[1]}):", "", head])
    for name in names:
        if any(get(s, "effects_second_judge", name) for s in present):
            lines.append(f"| {name} | " + " | ".join(
                _pp(get(s, "effects_second_judge", name)) for s in present) + " |")

    lines.extend(["", "## 2. How answers are labelled (aligned / partially aligned / misaligned)", "", head])
    for condition, label in (("country_only", "Country only"), (RESIST, "Country + false norm"),
                             (CLAIM_ONLY, "False norm, no country")):
        if any(get(s, "outcome_shares", condition) for s in present):
            cell = lambda e: "–" if not e else " / ".join(  # noqa: E731
                _pct(e[k]) for k in ("aligned", "partially_aligned", "misaligned"))
            lines.append(f"| {label} | " + " | ".join(cell(get(s, "outcome_shares", condition))
                                                       for s in present) + " |")

    lines.extend(["", "## 3. What predicts resisting (logistic regression, odds ratios; "
                      "intervals from resampling countries)", "", head])
    rows = list(get(present[0], "resisting_model", "odds_ratio") or {})
    for name in rows:
        cell = lambda e: "–" if not e else f"{e['or']:.2f} [{e['low']:.2f}, {e['high']:.2f}]"  # noqa: E731
        lines.append(f"| {name} | " + " | ".join(cell(get(s, "resisting_model", "odds_ratio", name))
                                                 for s in present) + " |")

    lines.extend(["", "## 4. By category: resisting, and resisting among records right with the country alone",
                  "", head])
    kinds = sorted({k for s in present for k in get(s, "by_category") or {}},
                   key=lambda k: -(get(present[0], "by_category", k, "n") or 0))
    for kind in kinds:
        cell = lambda e: "–" if not e else (  # noqa: E731
            f"{_pct(e['resist']['rate'])} / {_pct(e['resist_given_right'])} (n={e['n']})")
        lines.append(f"| {kind} | " + " | ".join(cell(get(s, "by_category", kind)) for s in present) + " |")

    cross = report.get("cross_model")
    if cross:
        lines.extend(["", "## 5. Do models resist on the same records?", "",
                      "| Pair | Cohen's κ |", "|---|---|"])
        lines.extend(f"| {' / '.join(MODELS[s] for s in pair.split(' / '))} | {e['kappa']:.2f} "
                     f"[{e['low']:.2f}, {e['high']:.2f}] |" for pair, e in cross["pairwise_kappa"].items())
        lines.extend(["", "| Resisted by | " + " | ".join(f"{k} models" for k in cross["resisted_by"]) + " |",
                      "|---|" + "---|" * len(cross["resisted_by"]),
                      "| Observed | " + " | ".join(_pct(v) for v in cross["resisted_by"].values()) + " |",
                      "| If models were independent | " + " | ".join(
                          _pct(v) for v in cross["if_independent"].values()) + " |"])

    nations = report.get("countries")
    if nations and nations["models"]:
        lines.extend(["", f"## 6. By country ({len(nations['countries'])} countries)", "", head])
        m = lambda s: nations["models"].get(s)  # noqa: E731
        lines.append("| Resisting, lowest to highest country | " + " | ".join(
            f"{_pct(m(s)['range'][0])} – {_pct(m(s)['range'][1])}" if m(s) else "–" for s in present) + " |")
        lines.append("| Spearman ρ across countries: resisting vs country-only accuracy | " + " | ".join(
            (lambda e: f"{e['rho']:+.2f} [{e['low']:+.2f}, {e['high']:+.2f}]")(
                m(s)["spearman_resist_vs_country_only"]) if m(s) else "–" for s in present) + " |")
        lines.append("| … resisting among records right with the country alone vs country-only accuracy | "
                     + " | ".join(f"{m(s)['spearman_held_vs_country_only']:+.2f}" if m(s) else "–"
                                  for s in present) + " |")
        if any(m(s) and "spearman_conflict_vs_resist" in m(s) for s in present):
            lines.append("| Spearman ρ across countries: internal conflict signal vs resisting | " + " | ".join(
                (lambda e: f"{e['rho']:+.2f} [{e['low']:+.2f}, {e['high']:+.2f}]")(
                    m(s)["spearman_conflict_vs_resist"]) if m(s) and "spearman_conflict_vs_resist" in m(s)
                else "–" for s in present) + " |")
            lines.append("| … internal conflict signal vs country-only accuracy | " + " | ".join(
                f"{m(s)['spearman_conflict_vs_country_only']:+.2f}"
                if m(s) and "spearman_conflict_vs_country_only" in m(s) else "–" for s in present) + " |")
        for name, label in (("selfexp", "WVS self-expression axis"), ("secular", "WVS secular-rational axis"),
                            ("distance_from_usa", "WVS distance from the United States")):
            if any(m(s) and f"spearman_{name}" in m(s) for s in present):
                cell = lambda e: (f"{e['rho']:+.2f} [{e['low']:+.2f}, {e['high']:+.2f}] "  # noqa: E731
                                  f"(country-only accuracy {e['rho_country_only']:+.2f}; {e['countries']} countries)")
                lines.append(f"| Spearman ρ: resisting vs {label} | " + " | ".join(
                    cell(m(s)[f"spearman_{name}"]) if m(s) and f"spearman_{name}" in m(s) else "–"
                    for s in present) + " |")
        lines.append("| Lowest five | " + " | ".join(
            ", ".join(f"{c} {_pct(v)}" for c, v in m(s)["lowest"]) if m(s) else "–" for s in present) + " |")
        lines.append("| Highest five | " + " | ".join(
            ", ".join(f"{c} {_pct(v)}" for c, v in m(s)["highest"]) if m(s) else "–" for s in present) + " |")
        regions = sorted({r for s in present if m(s) for r in m(s).get("by_region", {})})
        if regions:
            lines.extend(["", "By world region: resisting / resisting among records right with the country "
                              "alone / country-only accuracy", "", head])
            for name in regions:
                cell = lambda e: "–" if not e else (  # noqa: E731
                    f"{_pct(e['resist'])} / {_pct(e['resist_given_right'])} / {_pct(e['country_only'])}")
                count = next(m(s)["by_region"][name]["countries"] for s in present
                             if m(s) and name in m(s).get("by_region", {}))
                lines.append(f"| {name} ({count} countries) | " + " | ".join(
                    cell(m(s).get("by_region", {}).get(name) if m(s) else None) for s in present) + " |")
        lines.extend(["", "Agreement between models on the country profile (Spearman ρ): " + "; ".join(
            f"{' / '.join(MODELS[s] for s in pair.split(' / '))} {rho:+.2f}"
            for pair, rho in nations["between_models"].items()) + "."])

    coded = [s for s in present if get(s, "stance", "benchmark")]
    if coded:
        sub = "| | " + " | ".join(MODELS[s] for s in coded) + " |\n|---|" + "---|" * len(coded)
        lines.extend(["", "## 7. How answers to country + false norm treat the claim (full benchmark)", "", sub])
        for name in STANCES:
            lines.append(f"| {STANCE_LABELS[name]} | " + " | ".join(
                (lambda e: f"{_pct(e['rate'])} [{_pct(e['low'])}, {_pct(e['high'])}]")(
                    get(s, "stance", "benchmark", "share", name)) for s in coded) + " |")
        lines.append("| Restates the claim as fact | " + " | ".join(
            _pct(get(s, "stance", "benchmark", "restates_claim")) for s in coded) + " |")
        lines.append("| Presents its advice as the country's custom | " + " | ".join(
            _pct(get(s, "stance", "benchmark", "attributes_to_country")) for s in coded) + " |")
        for title, key in (("Among answers judged aligned (resisting)", "stance_given_aligned"),
                           ("Among answers not judged aligned", "stance_given_not_aligned"),
                           ("Share judged aligned, by stance", "aligned_given_stance")):
            lines.extend(["", f"{title}:", "", sub])
            lines.extend(f"| {STANCE_LABELS[name]} | " + " | ".join(
                _pct((get(s, "stance", "benchmark", key) or {}).get(name, float("nan")))
                for s in coded) + " |" for name in STANCES)
    armed = [s for s in present if get(s, "stance", "arms")]
    for slug in armed:
        arms = get(slug, "stance", "arms")
        lines.extend(["", f"## 8. Stance under the mechanistic interventions: {MODELS[slug]}", "",
                      "| Prompt and intervention | n | " + " | ".join(STANCE_LABELS[s] for s in STANCES)
                      + " | Restates claim | Names the country's custom |", "|---|---|" + "---|" * 7])
        lines.extend(f"| {STANCE_ARMS[arm]} | {e['n']} | " + " | ".join(
            _pct(e["share"][s]["rate"]) for s in STANCES)
            + f" | {_pct(e['restates_claim'])} | {_pct(e['attributes_to_country'])} |"
            for arm, e in arms.items())
    framed = [s for s in present if get(s, "framing")]
    if framed:
        sub = "| | " + " | ".join(MODELS[s] for s in framed) + " |\n|---|" + "---|" * len(framed)
        lines.extend(["", "## 9. Framing of the false claim, country stated: resisting, and change "
                          "against the bare statement on the same records", "", sub])
        lines.append("| Stated as is (benchmark) | " + " | ".join(
            (lambda e: _pct(next(iter(e.values()))["bare"]) + f" (n={next(iter(e.values()))['rate']['n']})")(
                get(s, "framing")) for s in framed) + " |")
        for condition, label in FRAMINGS.items():
            cell = lambda e: "–" if not e else f"{_pct(e['rate']['rate'])}; {_pp(e['effect'])}"  # noqa: E731
            lines.append(f"| {label} | " + " | ".join(cell(get(s, "framing", condition)) for s in framed) + " |")
        again = [s for s in framed if get(s, "framing_second_judge")]
        if again:
            lines.extend(["", f"Second judge ({JUDGES[1]}):", "",
                          "| | " + " | ".join(MODELS[s] for s in again) + " |\n|---|" + "---|" * len(again)])
            lines.append("| Stated as is (benchmark) | " + " | ".join(
                _pct(next(iter(get(s, "framing_second_judge").values()))["bare"]) for s in again) + " |")
            for condition, label in FRAMINGS.items():
                cell = lambda e: "–" if not e else f"{_pct(e['rate']['rate'])}; {_pp(e['effect'])}"  # noqa: E731
                lines.append(f"| {label} | " + " | ".join(
                    cell(get(s, "framing_second_judge", condition)) for s in again) + " |")
        if any(get(s, "framing", c, "stance") for s in framed for c in FRAMINGS):
            lines.extend(["", "Share of answers that challenge the claim / follow it with doubt / follow it:",
                          "", sub])
            for condition, label in FRAMINGS.items():
                cell = lambda e: "–" if not e else " / ".join(  # noqa: E731
                    _pct(e["share"][k]["rate"]) for k in ("challenges", "follows_with_caveat", "follows"))
                lines.append(f"| {label} | " + " | ".join(
                    cell(get(s, "framing", condition, "stance")) for s in framed) + " |")
    return "\n".join(lines) + "\n"


def _dig(value: Any, path: tuple) -> Any:
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value


def figure(report: dict[str, Any], output: Path) -> list[str]:
    try:
        import matplotlib
    except ImportError:
        return []
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = list(report["models"])
    names = [name for name, *_ in EFFECTS if all(
        name in report["models"][s]["effects"] for s in present)]
    if not present or not names:
        return []
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, ax = plt.subplots(figsize=(8.6, 0.42 * len(names) + 1.6))
    colours = dict(zip(MODELS, (BLUE, ORANGE, AQUA), strict=True))
    offsets = np.linspace(0.22, -0.22, len(present))
    for offset, slug in zip(offsets, present, strict=True):
        effects = [report["models"][slug]["effects"][name] for name in names]
        x = np.array([100 * e["difference"] for e in effects])
        y = np.arange(len(names))[::-1] + offset
        ax.hlines(y, [100 * e["low"] for e in effects], [100 * e["high"] for e in effects],
                  color=colours[slug], lw=1.4)
        ax.plot(x, y, "o", color=colours[slug], ms=5, label=MODELS[slug])
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(np.arange(len(names))[::-1], [f"{name}" for name in names])
    ax.xaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_xlabel("change in answers judged aligned when the cue is added (percentage points)")
    ax.set_title("What each cue does to the answer", loc="left", fontsize=11, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    written = []
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"cue_effects.{suffix}", dpi=200)
        written.append(f"cue_effects.{suffix}")
    plt.close(fig)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--records", type=Path, default=Path("data/processed/culturelens/records.json"))
    parser.add_argument("--behavioural", type=Path, default=Path("output/behavioral_evaluation/unconstrained"))
    parser.add_argument("--root", type=Path, default=Path("output/mechanistic_analysis"))
    parser.add_argument("--references", type=Path, default=Path("data/processed/references"),
                        help="Country covariates: geography/countries.csv and wvs/*.csv.")
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args(argv)
    report = build(args.records, args.behavioural, args.root, args.draws, args.references)
    output = args.root / "report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "drivers.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (output / "drivers.md").write_text(markdown(report), encoding="utf-8")
    written = figure(report, output)
    print(f"saved {output}: drivers.json, drivers.md" + "".join(f", {name}" for name in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
