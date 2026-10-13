"""Tables and figures of the mechanistic study, from the saved outputs.

Reads, under ``--root`` (default ``output/mechanistic_analysis``):

    outcomes/<model>.npz                      outcomes.py
    readout/<model>/scores.npz                readout.py
    interaction/<model>/{layers.csv,features.npz}   interaction.py
    localize/<model>/localize.npz             interventions/localize.py
    answers/<model>/<arm>.jsonl               interventions/generate.py
    judgments/<model>.<arm>/<judge>.jsonl     the judges, on those answers

and writes ``report/tables.json``, ``report/tables.md`` and, with matplotlib
installed, ``report/decision_depth.{png,pdf}`` and ``report/answers.{png,pdf}``.
Anything not yet produced is left out of the report, not guessed.

Conventions: the outcome is *resisting*, the answer to the country + false-norm
prompt judged ``aligned``. AUROCs rank scores within country, so differences
between countries cannot produce them. Single-layer numbers use the layer fixed
in ``mechanistic_analysis.REPORT_LAYER``. Intervals are 95% bootstrap intervals
(records sharing a claim resampled together for the cached-state analyses).

PYTHONPATH=src python -m mechanistic_analysis.report
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from mechanistic_analysis import (
    CLAIM_WINDOW, CONTROL_WINDOW, ERASE_LAYERS, JUDGES, REPORT_LAYER, RESIST,
)
from mechanistic_analysis import outcomes as outcome_table
from mechanistic_analysis.common import country_index
from mechanistic_analysis.stats import (
    auroc, auroc_with_interval, bootstrap, paired_difference, rate_with_interval, within_ranks,
)

MODELS = {"llama-3.1-8b": "Llama-3.1-8B", "gemma-3-12b": "Gemma-3-12B", "qwen-3.5-27b": "Qwen-3.5-27B"}
LABELLERS = (*JUDGES, "both")
READOUT_STATES = {
    f"text:{RESIST}": "Words of the prompt (no model)",
    "base": "State: no country, no claim",
    "country_only": "State: country only",
    "opposite_norm_only": "State: false norm, no country",
    RESIST: "State: country + false norm",
}
ARM_LABELS = {
    "false": "Country + false norm",
    "false-noclaim": "… claim blocked, every layer",
    "false-noclaim-win": "… claim blocked, claim window",
    "false-noclaim-late": "… claim blocked, later window",
    "false-nocountry": "… country line blocked",
    "false-nocontrol": "… situation text blocked (control)",
    "false-noclaim-heads": "… claim blocked, selected heads only",
    "false-noclaim-randheads": "… claim blocked, as many random heads",
    "false-erase-first": "… claim erased after the first block",
    "false-erase-before": "… claim erased before the reading band",
    "false-erase-after": "… claim erased after the reading band",
    "claim-only": "False norm, no country in prompt",
    "country": "Country only",
    "true": "Country + true norm",
    "true-noclaim": "… claim blocked, every layer",
}
# after arm, before arm, what the difference shows
CONTRASTS = (
    ("false-noclaim", "false", "blocking the false claim, every layer"),
    ("false-noclaim-win", "false", "blocking it in the claim window only"),
    ("false-noclaim-late", "false", "blocking it in a later window (control)"),
    ("false-noclaim-heads", "false", "blocking it for the selected heads only"),
    ("false-noclaim-randheads", "false", "blocking it for as many random heads (control)"),
    ("false-erase-first", "false", "erasing the claim after the first block"),
    ("false-erase-before", "false", "erasing it before the reading band"),
    ("false-erase-after", "false", "erasing it after the reading band (control)"),
    ("false-nocontrol", "false", "blocking situation text (control)"),
    ("false-nocountry", "false", "blocking the country line"),
    ("false", "claim-only", "naming the country, false norm stated"),
    ("true-noclaim", "true", "blocking the true claim, every layer"),
    ("false-noclaim", "country", "claim blocked vs never stated"),
)
BLUE, LIGHT_BLUE, ORANGE, AQUA, INK, MUTED, GRID = (
    "#2a78d6", "#86b6ef", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df")
LIGHT_ORANGE = "#f4b294"


# ------------------------------------------------------------------ pieces --


def behaviour(root: Path, slug: str) -> dict[str, Any] | None:
    path = root / "outcomes" / f"{slug}.npz"
    if not path.is_file():
        return None
    saved = np.load(path)
    table = {str(name): saved["outcomes"][:, i] for i, name in enumerate(saved["names"])}
    out: dict[str, Any] = {}
    for judge in LABELLERS:
        rates = {}
        for condition in outcome_table.CONDITIONS:
            values = table[f"{judge}__{condition}"]
            rates[condition] = float(values[values >= 0].mean())
        known = table[f"{judge}__country_only"] == 1
        resist = table[f"{judge}__{RESIST}"]
        rates["resist_given_right_with_country"] = float(resist[known & (resist >= 0)].mean())
        rates["resist_given_wrong_with_country"] = float(
            resist[(table[f"{judge}__country_only"] == 0) & (resist >= 0)].mean())
        # Resisting by what the model answers without the false claim.
        base = table[f"{judge}__base"]
        with_country = table[f"{judge}__country_only"]
        for default in (0, 1):
            for cued in (0, 1):
                cell = (base == default) & (with_country == cued) & (resist >= 0)
                rates[f"resist__default_{default}__country_{cued}"] = {
                    "rate": float(resist[cell].mean()), "n": int(cell.sum())}
        out[judge] = rates
    a, b = (table[f"{judge}__{RESIST}"] for judge in JUDGES)
    both = (a >= 0) & (b >= 0)
    agree = float((a[both] == b[both]).mean())
    pa, pb = a[both].mean(), b[both].mean()
    chance = pa * pb + (1 - pa) * (1 - pb)
    out["judge_agreement_on_resisting"] = {"agreement": agree, "kappa": float((agree - chance) / (1 - chance))}
    return out


def readout(root: Path, slug: str, draws: int) -> dict[str, Any] | None:
    path = root / "readout" / slug / "scores.npz"
    table_path = root / "outcomes" / f"{slug}.npz"
    if not path.is_file() or not table_path.is_file():
        return None
    saved = np.load(path)
    norm_ids = [str(value) for value in saved["norm_ids"]]
    table = outcome_table.load(table_path, norm_ids)
    names = [str(value) for value in saved["outcomes"]]
    layers = saved["layers"].tolist()
    country = np.asarray(country_index(norm_ids))
    groups = saved["groups"]
    at = layers.index(REPORT_LAYER[slug]) if REPORT_LAYER[slug] in layers else int(
        np.argmin(np.abs(np.asarray(layers) - REPORT_LAYER[slug])))
    out: dict[str, Any] = {"layer": layers[at], "layers": layers, "by_judge": {}, "curves": {}}
    for judge in LABELLERS:
        name = f"{judge}__{RESIST}"
        if name not in names:
            continue
        labels = table[name]
        keep = labels >= 0
        y = labels[keep] == 1
        rows: dict[str, Any] = {}
        ranked: dict[str, np.ndarray] = {}
        for state in READOUT_STATES:
            key = f"score__{state}"
            if key not in saved.files:
                continue
            scores = saved[key].astype(np.float64)[:, names.index(name)]
            layer_scores = scores[0 if scores.shape[0] == 1 else at][keep]
            rows[state] = auroc_with_interval(layer_scores, y, country[keep],
                                              groups=groups[keep], draws=draws)
            ranked[state] = within_ranks(layer_scores, country[keep])
            if judge == JUDGES[0] and scores.shape[0] > 1:
                out["curves"][state] = [auroc(within_ranks(scores[i][keep], country[keep]), y)
                                        for i in range(len(layers))]
            elif judge == JUDGES[0]:
                out["curves"][state] = [rows[state]["auroc"]]
        if RESIST in ranked and "opposite_norm_only" in ranked:
            gain = lambda r: auroc(ranked[RESIST][r], y[r]) - auroc(ranked["opposite_norm_only"][r], y[r])  # noqa: E731
            low, high = bootstrap(gain, len(y), draws=draws, groups=groups[keep])
            rows["country_adds"] = {"difference": float(gain(np.arange(len(y)))), "low": low, "high": high}
        # Among records the model answers correctly with the country alone.
        known = table[f"{judge}__country_only"][keep] == 1
        if RESIST in ranked and known.sum() > 50:
            scores = saved[f"score__{RESIST}"].astype(np.float64)[at, names.index(name)][keep]
            rows["among_right_with_country"] = auroc_with_interval(
                scores[known], y[known], country[keep][known], groups=groups[keep][known], draws=draws)
        out["by_judge"][judge] = rows
    return out


def interaction(root: Path, slug: str, draws: int) -> dict[str, Any] | None:
    folder = root / "interaction" / slug
    table_path = root / "outcomes" / f"{slug}.npz"
    if not (folder / "features.npz").is_file() or not table_path.is_file():
        return None
    with (folder / "layers.csv").open(encoding="utf-8") as handle:
        rows = [{key: float(value) for key, value in row.items()} for row in csv.DictReader(handle)]
    layer = REPORT_LAYER[slug]
    at_layer = next(row for row in rows if int(row["layer"]) == layer)
    early = rows[: max(1, len(rows) // 4)]
    saved = np.load(folder / "features.npz")
    norm_ids = [str(value) for value in saved["norm_ids"]]
    table = outcome_table.load(table_path, norm_ids)
    country = np.asarray(country_index(norm_ids))
    layers = saved["layers"].tolist()
    at = layers.index(layer) if layer in layers else int(np.argmin(np.abs(np.asarray(layers) - layer)))
    out: dict[str, Any] = {
        "layer": layer,
        "interaction_cosine_early": max(row["interaction_cosine"] for row in early),
        "interaction_cosine_at_layer": at_layer["interaction_cosine"],
        "additive_share_false": at_layer["additive_share_false"],
        "country_over_claim": at_layer["country"] / at_layer["claim_false"],
        "cosine_curve": {"layers": [int(row["layer"]) for row in rows],
                         "values": [row["interaction_cosine"] for row in rows]},
        "conflict": {}, "conflict_curve": {"layers": layers, "values": []},
    }
    for judge in LABELLERS:
        labels = table[f"{judge}__{RESIST}"]
        keep = labels >= 0
        out["conflict"][judge] = auroc_with_interval(
            saved["conflict"][at].astype(np.float64)[keep], labels[keep] == 1, country[keep],
            groups=saved["groups"][keep], draws=draws)
        if judge == JUDGES[0]:
            out["conflict_curve"]["values"] = [
                auroc(within_ranks(saved["conflict"][i].astype(np.float64)[keep], country[keep]),
                      labels[keep] == 1) for i in range(len(layers))]
    return out


def localisation(root: Path, slug: str, draws: int) -> dict[str, Any] | None:
    path = root / "localize" / slug / "localize.npz"
    if not path.is_file():
        return None
    saved = np.load(path)
    done = saved["completed"]
    spans = [tuple(window) for window in saved["windows"].tolist()]
    margin = lambda values: (values[..., 0] - values[..., 1])  # noqa: E731
    out: dict[str, Any] = {"records": int(done.sum()), "windows": [f"{a}-{b}" for a, b in spans],
                           "baseline": {}, "change": {}}
    for key in saved.files:
        if key.startswith("baseline__"):
            values = margin(saved[key])[done]
            out["baseline"][key[10:]] = {"mean_margin": float(values.mean()),
                                         "share_positive": float((values > 0).mean())}
    for key in saved.files:
        if not key.startswith("knock__"):
            continue
        _, target, span = key.split("__")
        plain = margin(saved[f"baseline__{target}"])[done]
        knocked = margin(saved[key])[:, done]
        out["change"][f"{target}__{span}"] = {
            f"{a}-{b}": {**paired_difference(knocked[i], plain, draws=draws),
                         "share_positive": float((knocked[i] > 0).mean())}
            for i, (a, b) in enumerate(spans)}
    return out


def read_arm(root: Path, slug: str, arm: str, judge: str) -> dict[str, int]:
    """``norm_id -> 1/0`` for one arm's answers under one judge; ``both`` needs every judge."""

    if judge == "both":
        parts = [read_arm(root, slug, arm, each) for each in JUDGES]
        return {k: int(all(part[k] for part in parts)) for k in parts[0] if all(k in p for p in parts)}
    path = root / "judgments" / f"{slug}.{arm}" / f"{judge}.jsonl"
    if not path.is_file():
        return {}
    labels = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                labels[str(row["norm_id"])] = int(row["alignment"] == "aligned")
    return labels


def answers(root: Path, slug: str, draws: int) -> dict[str, Any] | None:
    folder = root / "answers" / slug
    if not folder.is_dir():
        return None
    arms = [arm for arm in ARM_LABELS if (folder / f"{arm}.jsonl").is_file()]
    out: dict[str, Any] = {"arms": arms, "by_judge": {}}
    for judge in LABELLERS:
        labels = {arm: read_arm(root, slug, arm, judge) for arm in arms}
        labels = {arm: found for arm, found in labels.items() if found}
        if not labels:
            continue
        rates = {arm: rate_with_interval(np.array(list(found.values())), draws=draws)
                 for arm, found in labels.items()}
        contrasts = {}
        for after, before, meaning in CONTRASTS:
            if after in labels and before in labels:
                shared = sorted(set(labels[after]) & set(labels[before]))
                if len(shared) >= 20:
                    contrasts[f"{after} - {before}"] = {
                        "meaning": meaning, **paired_difference(
                            np.array([labels[after][k] for k in shared]),
                            np.array([labels[before][k] for k in shared]), draws=draws)}
        # The same arms among records the model answers correctly with the country alone
        # (same engine, same records), and among those it does not.
        strata = {}
        if "country" in labels:
            for name, wanted in (("right_with_country", 1), ("wrong_with_country", 0)):
                keep = {k for k, value in labels["country"].items() if value == wanted}
                strata[name] = {arm: rate_with_interval(
                    np.array([value for k, value in found.items() if k in keep]), draws=draws)
                    for arm, found in labels.items()
                    if arm != "country" and sum(k in keep for k in found) >= 20}
        out["by_judge"][judge] = {"rate": rates, "contrast": contrasts, "strata": strata}
    return out if out["by_judge"] else None


def erasure(root: Path, slug: str, draws: int) -> dict[str, Any] | None:
    path = root / "erasure" / slug / "erasure.npz"
    if not path.is_file():
        return None
    saved = np.load(path)
    done = saved["completed"]
    cuts = saved["cuts"].tolist()
    margin = lambda values: (values[..., 0] - values[..., 1])  # noqa: E731
    out: dict[str, Any] = {"records": int(done.sum()), "cuts": cuts, "change": {}}
    for target in ("country_opposite_norm", "country_norm"):
        plain = margin(saved[f"baseline__{target}"])[done]
        cut = margin(saved[f"erase__{target}"])[:, done]
        out["change"][target] = [paired_difference(cut[i], plain, draws=draws) for i in range(len(cuts))]
    return out


def heads(root: Path, slug: str) -> dict[str, Any] | None:
    path = root / "heads" / slug / "metrics.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text(encoding="utf-8"))
    return {key: saved[key] for key in (
        "sample", "window", "heads_in_window", "window_effect", "single_head_effects_ranked",
        "joint_effect_top", "joint_effect_random", "selection")}


def claim_only_benchmark(root: Path, slug: str, draws: int) -> dict[str, Any] | None:
    """Full benchmark: the false norm with and without the country line, judged by the
    primary judge (``scripts/behavioral_evaluation/claim_only.sbatch``)."""

    judgments = root.parent / "behavioral_evaluation" / "unconstrained" / "judgments"
    with_country = judgments / slug / f"{JUDGES[0]}.jsonl"
    without = judgments / f"{slug}.opposite_norm_only" / f"{JUDGES[0]}.jsonl"
    if not with_country.is_file() or not without.is_file():
        return None

    def read(path: Path, condition: str) -> dict[str, dict[str, int]]:
        found: dict[str, dict[str, int]] = {}
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    found.setdefault(str(row["condition"]), {})[str(row["norm_id"])] = int(
                        row["alignment"] == "aligned")
        return found.get(condition, {})

    named, alone = read(with_country, RESIST), read(without, "opposite_norm_only")
    known = read(with_country, "country_only")
    shared = sorted(set(named) & set(alone))
    if len(shared) < 100:
        return None
    a, b = np.array([named[k] for k in shared]), np.array([alone[k] for k in shared])
    right = np.array([known.get(k, 0) for k in shared]).astype(bool)
    return {"records": len(shared),
            "with_country": rate_with_interval(a, draws=draws),
            "without_country": rate_with_interval(b, draws=draws),
            "effect_of_country": paired_difference(a, b, draws=draws),
            "effect_among_right_with_country": paired_difference(a[right], b[right], draws=draws),
            "effect_among_wrong_with_country": paired_difference(a[~right], b[~right], draws=draws)}


def margin_validity(root: Path, slug: str, draws: int) -> dict[str, Any] | None:
    """How well the decision margin tracks the judged answer, record by record: the
    margin on a prompt (untouched or under an intervention) against whether the
    answer generated under the same intervention is judged aligned."""

    margin = lambda values: (values[..., 0] - values[..., 1])  # noqa: E731
    pairs: list[tuple[str, list[str], np.ndarray]] = []
    path = root / "erasure" / slug / "erasure.npz"
    if path.is_file():
        saved = np.load(path)
        ids = [str(x) for x in saved["norm_ids"]]
        cuts = saved["cuts"].tolist()
        pairs.append(("false", ids, margin(saved[f"baseline__{RESIST}"])))
        erased = margin(saved[f"erase__{RESIST}"])
        for arm, layer in zip(("false-erase-first", "false-erase-before", "false-erase-after"),
                              ERASE_LAYERS.get(slug, ())):
            if layer in cuts:
                pairs.append((arm, ids, erased[cuts.index(layer)]))
    path = root / "localize" / slug / "localize.npz"
    if path.is_file() and slug in CLAIM_WINDOW:
        saved = np.load(path)
        ids = [str(x) for x in saved["norm_ids"]]
        windows = [tuple(int(v) for v in w) for w in saved["windows"]]
        knocked = margin(saved[f"knock__{RESIST}__claim"])
        for arm, window in (("false-noclaim-win", tuple(CLAIM_WINDOW[slug])),
                            ("false-noclaim-late", tuple(CONTROL_WINDOW[slug])),
                            ("false-noclaim", windows[-1])):
            if window in windows:
                pairs.append((arm, ids, knocked[windows.index(window)]))
    out = {}
    for arm, ids, values in pairs:
        labels = read_arm(root, slug, arm, JUDGES[0])
        keep = [i for i, k in enumerate(ids) if k in labels]
        y = np.array([labels[ids[i]] for i in keep])
        if len(keep) >= 20 and 0 < y.sum() < len(y):
            out[arm] = {**auroc_with_interval(values[keep], y, draws=draws),
                        "positive_margin": float((values[keep] > 0).mean()), "aligned": float(y.mean())}
    return out or None


# ------------------------------------------------------------------ tables --


def _pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def _ci(entry: dict[str, float], key: str = "auroc", digits: int = 3) -> str:
    return f"{entry[key]:.{digits}f} [{entry['low']:.{digits}f}, {entry['high']:.{digits}f}]"


def markdown(report: dict[str, Any]) -> str:
    present = [slug for slug in MODELS if slug in report["models"]]
    head = "| | " + " | ".join(MODELS[slug] for slug in present) + " |\n|---|" + "---|" * len(present)
    lines: list[str] = ["# Mechanistic study: tables", "",
                        f"Primary judge: {JUDGES[0]}. Second judge: {JUDGES[1]}. "
                        "Intervals are 95% bootstrap intervals.", ""]
    if not present:
        return "\n".join(lines + ["No outputs found."]) + "\n"

    def table(title: str, rows: list[tuple[str, list[str]]]) -> None:
        lines.extend([f"## {title}", "", head])
        lines.extend(f"| {label} | " + " | ".join(cells) + " |" for label, cells in rows)
        lines.append("")

    get = lambda slug, *path: _dig(report["models"][slug], path)  # noqa: E731

    if all(get(slug, "behaviour") for slug in present):
        names = {"base": "No country, no norm", "country_only": "Country only",
                 "norm_only": "True norm, no country", "country_norm": "Country + true norm",
                 RESIST: "Country + false norm (resisting)",
                 "resist_given_right_with_country": "Resisting, among records right with the country alone"}
        table("1. Behaviour: share of open-ended answers judged aligned (primary / second judge)", [
            (label, [f"{_pct(get(s, 'behaviour', JUDGES[0], key))} / {_pct(get(s, 'behaviour', JUDGES[1], key))}"
                     for s in present]) for key, label in names.items()] + [
            *[(f"Resisting when the answer without a country is {'right' if d else 'wrong'} and with the "
               f"country alone is {'right' if c else 'wrong'}",
               [f"{_pct(get(s, 'behaviour', JUDGES[0], f'resist__default_{d}__country_{c}', 'rate'))} "
                f"(n={get(s, 'behaviour', JUDGES[0], f'resist__default_{d}__country_{c}', 'n')})" for s in present])
              for d in (0, 1) for c in (0, 1)],
            ("Judges agree on resisting (agreement, κ)", [
                f"{_pct(get(s, 'behaviour', 'judge_agreement_on_resisting', 'agreement'))}, "
                f"{get(s, 'behaviour', 'judge_agreement_on_resisting', 'kappa'):.2f}" for s in present])])

    for judge in LABELLERS:
        if not all(get(slug, "readout", "by_judge", judge) for slug in present):
            continue
        rows = [(label, [_ci(get(s, "readout", "by_judge", judge, state)) if get(s, "readout", "by_judge", judge, state) else "–"
                         for s in present]) for state, label in READOUT_STATES.items()]
        rows.append(("Gain from the country (country + false norm minus false norm alone)", [
            _ci(get(s, "readout", "by_judge", judge, "country_adds"), "difference") for s in present]))
        rows.append(("Country + false norm state, among records right with the country alone", [
            _ci(get(s, "readout", "by_judge", judge, "among_right_with_country")) for s in present]))
        rows.append(("Layer", [str(get(s, "readout", "layer")) for s in present]))
        table(f"2. Readout: AUROC for resisting, labels from {judge}", rows)

    if all(get(slug, "interaction") for slug in present):
        rows = [("True- vs false-claim interaction, cosine: early layers → report layer", [
            f"{get(s, 'interaction', 'interaction_cosine_early'):.2f} → {get(s, 'interaction', 'interaction_cosine_at_layer'):.2f}"
            for s in present]),
            ("Share of the country + false-norm shift that is additive", [
                _pct(get(s, "interaction", "additive_share_false")) for s in present]),
            ("Size of the country effect relative to the false-claim effect", [
                f"{get(s, 'interaction', 'country_over_claim'):.2f}" for s in present])]
        rows += [(f"Conflict feature → resisting, AUROC ({judge})", [
            _ci(get(s, "interaction", "conflict", judge)) for s in present]) for judge in LABELLERS]
        table("3. Country × claim interaction", rows)

    local = [slug for slug in present if get(slug, "localisation")]
    if local:
        lines.extend(["## 4. Where the claim is read: change in margin (nats) when the answer cannot attend to a span", ""])
        for slug in local:
            data = get(slug, "localisation")
            lines.extend([f"**{MODELS[slug]}** ({data['records']} records; baseline margin "
                          + ", ".join(f"{k} {v['mean_margin']:+.1f}" for k, v in data["baseline"].items()) + ")", "",
                          "| Blocked | " + " | ".join(data["windows"][:-1]) + " | every layer |",
                          "|---|" + "---|" * len(data["windows"])])
            for key, row in data["change"].items():
                target, span = key.split("__")
                label = f"{span}, {'false' if target == RESIST else 'true'}-norm prompt"
                lines.append(f"| {label} | " + " | ".join(f"{row[w]['difference']:+.2f}" for w in data["windows"]) + " |")
            lines.append("")

    for judge in LABELLERS:
        have = [slug for slug in present if get(slug, "answers", "by_judge", judge)]
        if not have:
            continue
        sub = "| | " + " | ".join(MODELS[s] for s in have) + " |\n|---|" + "---|" * len(have)
        lines.extend([f"## 5. Answers under knockout: share judged aligned ({judge})", "", sub])
        for arm, label in ARM_LABELS.items():
            cells = []
            for s in have:
                entry = get(s, "answers", "by_judge", judge, "rate", arm)
                cells.append(f"{_pct(entry['rate'])} [{_pct(entry['low'])}, {_pct(entry['high'])}] (n={entry['n']})" if entry else "–")
            if any(cell != "–" for cell in cells):
                lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.extend(["", f"Paired differences in percentage points ({judge}):", "", sub])
        for after, before, meaning in CONTRASTS:
            cells = []
            for s in have:
                entry = get(s, "answers", "by_judge", judge, "contrast", f"{after} - {before}")
                cells.append(f"{100 * entry['difference']:+.1f} [{100 * entry['low']:+.1f}, {100 * entry['high']:+.1f}], p={entry['p']:.3g}"
                             if entry else "–")
            if any(cell != "–" for cell in cells):
                lines.append(f"| Effect of {meaning} | " + " | ".join(cells) + " |")
        strata_rows = []
        for stratum, label in (("right_with_country", "right with the country alone"),
                               ("wrong_with_country", "wrong with the country alone")):
            for arm in ("false", "false-noclaim-win", "false-noclaim", "claim-only"):
                cells = []
                for s in have:
                    entry = get(s, "answers", "by_judge", judge, "strata", stratum, arm)
                    cells.append(f"{_pct(entry['rate'])} [{_pct(entry['low'])}, {_pct(entry['high'])}] (n={entry['n']})"
                                 if entry else "–")
                if any(cell != "–" for cell in cells):
                    strata_rows.append(f"| Records {label}: {ARM_LABELS[arm].lstrip('… ')} | " + " | ".join(cells) + " |")
        if strata_rows:
            lines.extend(["", f"By what the model answers with the country alone ({judge}):", "", sub] + strata_rows)
        lines.append("")

    erased = [slug for slug in present if get(slug, "erasure")]
    if erased:
        lines.extend(["## 6. Erasing the claim's positions after a given block: change in margin (nats), false-norm prompt", ""])
        for slug in erased:
            data = get(slug, "erasure")
            rows = data["change"]["country_opposite_norm"]
            lines.extend([f"**{MODELS[slug]}** ({data['records']} records)", "",
                          "| After block | " + " | ".join(str(c) for c in data["cuts"]) + " |",
                          "|---|" + "---|" * len(data["cuts"]),
                          "| False-norm prompt | " + " | ".join(f"{r['difference']:+.2f}" for r in rows) + " |",
                          "| True-norm prompt | " + " | ".join(
                              f"{r['difference']:+.2f}" for r in data["change"]["country_norm"]) + " |", ""])

    scanned = [slug for slug in present if get(slug, "heads")]
    if scanned:
        lines.extend(["## 7. Heads that carry the claim (development sample, margin)", ""])
        for slug in scanned:
            data = get(slug, "heads")
            top = ", ".join(f"L{h['layer']}.H{h['head']} ({h['effect']:+.2f})"
                            for h in data["single_head_effects_ranked"][:8])
            lines.extend([
                f"**{MODELS[slug]}**: window {data['window'][0]}–{data['window'][1]}, "
                f"{data['heads_in_window']} heads, whole-window effect {data['window_effect']:+.2f} nats "
                f"({data['sample']} records).", "",
                f"Largest single heads: {top}.", "",
                "| Heads blocked together | " + " | ".join(data["joint_effect_top"]) + " |",
                "|---|" + "---|" * len(data["joint_effect_top"]),
                "| Top-ranked | " + " | ".join(f"{v:+.2f}" for v in data["joint_effect_top"].values()) + " |",
                "| Random | " + " | ".join(f"{v:+.2f}" for v in data["joint_effect_random"].values()) + " |", "",
                f"Selected: {data['selection']['size']} heads ({data['selection']['heads']}).", ""])

    full = [slug for slug in present if get(slug, "claim_only_benchmark")]
    if full:
        sub = "| | " + " | ".join(MODELS[s] for s in full) + " |\n|---|" + "---|" * len(full)
        cell = lambda e: f"{100 * e['difference']:+.1f} [{100 * e['low']:+.1f}, {100 * e['high']:+.1f}], p={e['p']:.3g} (n={e['n']})"  # noqa: E731
        rate = lambda e: f"{_pct(e['rate'])} [{_pct(e['low'])}, {_pct(e['high'])}]"  # noqa: E731
        lines.extend([f"## 8. Full benchmark: resisting a false norm with and without the country line ({JUDGES[0]})", "", sub,
                      "| False norm, country named | " + " | ".join(rate(get(s, "claim_only_benchmark", "with_country")) for s in full) + " |",
                      "| False norm, no country | " + " | ".join(rate(get(s, "claim_only_benchmark", "without_country")) for s in full) + " |",
                      "| Effect of naming the country (points) | " + " | ".join(cell(get(s, "claim_only_benchmark", "effect_of_country")) for s in full) + " |",
                      "| … among records right with the country alone | " + " | ".join(cell(get(s, "claim_only_benchmark", "effect_among_right_with_country")) for s in full) + " |",
                      "| … among records wrong with the country alone | " + " | ".join(cell(get(s, "claim_only_benchmark", "effect_among_wrong_with_country")) for s in full) + " |", ""])
    checked = [slug for slug in present if get(slug, "margin_validity")]
    if checked:
        arms = [arm for arm in ARM_LABELS if any(arm in get(s, "margin_validity") for s in checked)]
        entry = lambda s, arm: get(s, "margin_validity").get(arm)  # noqa: E731
        show = lambda e: "–" if e is None else (  # noqa: E731
            f"{_ci(e)}; margin > 0 in {_pct(e['positive_margin'])}, aligned {_pct(e['aligned'])}")
        lines.extend([f"## 9. Does the margin track the judged answer? AUROC of the margin for the answer being judged aligned ({JUDGES[0]})", "",
                      "| Prompt and intervention | " + " | ".join(MODELS[s] for s in checked) + " |",
                      "|---|" + "---|" * len(checked)])
        lines.extend("| " + ARM_LABELS[arm] + " | " + " | ".join(show(entry(s, arm)) for s in checked) + " |"
                     for arm in arms)
        lines.append("")
    return "\n".join(lines) + "\n"


def _dig(value: Any, path: tuple) -> Any:
    for key in path:
        if not isinstance(value, dict) or key not in value or value[key] is None:
            return None
        value = value[key]
    return value


# ----------------------------------------------------------------- figures --


def figures(report: dict[str, Any], output: Path) -> list[str]:
    try:
        import matplotlib
    except ImportError:
        return []
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    written: list[str] = []
    present = [slug for slug in MODELS if _dig(report["models"], (slug, "readout"))]
    if present:
        fig, axes = plt.subplots(3, len(present), figsize=(4.3 * len(present), 7.4), squeeze=False,
                                 gridspec_kw={"height_ratios": [3, 1.6, 1.6], "hspace": 0.32, "wspace": 0.24})
        for column, slug in enumerate(present):
            data = report["models"][slug]
            read, top, middle, bottom = data["readout"], axes[0, column], axes[1, column], axes[2, column]
            layers = read["layers"]
            for state, colour, label in ((RESIST, BLUE, "State: country + false norm"),
                                         ("opposite_norm_only", LIGHT_BLUE, "State: false norm, no country")):
                if state in read["curves"]:
                    top.plot(layers, read["curves"][state], color=colour, lw=2, label=label)
            inter = data.get("interaction")
            if inter:
                top.plot(inter["conflict_curve"]["layers"], inter["conflict_curve"]["values"],
                         color=AQUA, lw=2, label="Conflict feature (no behaviour labels)")
            text = read["curves"].get(f"text:{RESIST}")
            if text:
                top.axhline(text[0], color=MUTED, lw=1, ls=(0, (2, 2)), label="Words of the prompt only")
            top.axhline(0.5, color=GRID, lw=1)
            top.axvline(read["layer"], color=GRID, lw=1)
            top.set_ylim(0.45, 0.97)
            top.set_title(MODELS[slug], loc="left", fontsize=11, fontweight="bold")
            if column == 0:
                top.set_ylabel("AUROC for resisting\nthe false norm")
                top.legend(frameon=False, fontsize=7.5, loc="upper left")
            if inter:
                middle.plot(inter["cosine_curve"]["layers"], inter["cosine_curve"]["values"], color=ORANGE, lw=2)
            middle.set_ylim(0.5, 1.02)
            if column == 0:
                middle.set_ylabel("True- vs false-claim\ninteraction (cosine)")
            local = data.get("localisation")
            if local:
                row = local["change"][f"{RESIST}__claim"]
                total = row[local["windows"][-1]]["difference"]
                for window in local["windows"][:-1]:
                    low, high = map(int, window.split("-"))
                    bottom.bar((low + high) / 2 - 0.5, row[window]["difference"] / total,
                               width=(high - low) - 0.5, color=BLUE)
                bottom.set_ylim(0, 0.6)
            else:
                bottom.text(0.5, 0.5, "attention cannot be blocked\nin most layers of this model",
                            transform=bottom.transAxes, ha="center", va="center", color=MUTED, fontsize=8)
                bottom.set_yticks([])
            if column == 0:
                bottom.set_ylabel("Share of the claim's effect\nremoved by blocking it here")
            bottom.set_xlabel("layer")
            for axis in (top, middle, bottom):
                axis.set_xlim(0, layers[-1])
                axis.grid(axis="y", color=GRID, lw=0.6)
                axis.set_axisbelow(True)
        fig.suptitle("Where the decision to follow or resist a false norm forms",
                     x=0.07, ha="left", fontsize=12)
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"decision_depth.{suffix}", dpi=180, bbox_inches="tight")
            written.append(f"decision_depth.{suffix}")
        plt.close(fig)

    swept = [slug for slug in MODELS if _dig(report["models"], (slug, "erasure"))]
    if swept:
        fig, axes = plt.subplots(1, len(swept), figsize=(4.3 * len(swept), 3.3), squeeze=False, sharey=True)
        for column, slug in enumerate(swept):
            data, axis = report["models"][slug]["erasure"], axes[0, column]
            for target, colour, label in (("country_opposite_norm", BLUE, "False norm in the prompt"),
                                          ("country_norm", ORANGE, "True norm in the prompt")):
                rows = data["change"][target]
                first = rows[0]["difference"]
                axis.plot(data["cuts"], [row["difference"] / first for row in rows], color=colour,
                          lw=2, marker="o", ms=3.5, label=label)
            axis.axhline(0, color=GRID, lw=1)
            axis.set_ylim(-0.08, 1.12)
            axis.set_title(MODELS[slug], loc="left", fontsize=11, fontweight="bold")
            axis.set_xlabel("claim erased after block")
            axis.grid(axis="y", color=GRID, lw=0.6)
            axis.set_axisbelow(True)
            if column == 0:
                axis.set_ylabel("Share of the claim's effect on the\nanswer still removed by erasing it")
                axis.legend(frameon=False, fontsize=8, loc="lower left")
        fig.suptitle("By which depth has the answer already taken the claim from its positions?",
                     x=0.02, ha="left", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"erasure_depth.{suffix}", dpi=180, bbox_inches="tight")
            written.append(f"erasure_depth.{suffix}")
        plt.close(fig)

    judged = [slug for slug in MODELS if _dig(report["models"], (slug, "answers", "by_judge", JUDGES[0]))]
    if judged:
        fig, axes = plt.subplots(1, len(judged), figsize=(4.4 * len(judged), 5.6), squeeze=False, sharex=True)
        every = {slug: report["models"][slug]["answers"]["by_judge"][JUDGES[0]]["rate"] for slug in judged}
        # The same rows in every panel; an arm a model did not run is left empty.
        arms = [arm for arm in ARM_LABELS if any(arm in rates for rates in every.values())][::-1]
        for column, slug in enumerate(judged):
            rates = every[slug]
            axis = axes[0, column]
            for position, arm in enumerate(arms):
                if arm not in rates:
                    continue
                entry = rates[arm]
                colour = BLUE if ("-no" in arm or "-erase" in arm) else MUTED
                axis.barh(position, entry["rate"], height=0.62, color=colour)
                axis.plot([entry["low"], entry["high"]], [position, position], color=INK, lw=1)
                axis.text(min(entry["high"] + 0.02, 0.98), position, f"{100 * entry['rate']:.0f}%",
                          va="center", fontsize=8, color=INK)
            axis.set_ylim(-0.6, len(arms) - 0.4)
            axis.set_yticks(range(len(arms)))
            axis.set_yticklabels([ARM_LABELS[arm] for arm in arms] if column == 0 else [""] * len(arms), fontsize=8)
            axis.set_xlim(0, 1.08)
            axis.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            axis.set_xticklabels(["0", "25%", "50%", "75%", "100%"])
            axis.grid(axis="x", color=GRID, lw=0.6)
            axis.set_axisbelow(True)
            axis.set_title(MODELS[slug], loc="left", fontsize=11, fontweight="bold")
            axis.set_xlabel("answers judged aligned")
        from matplotlib.patches import Patch
        fig.suptitle("What the model writes when its answer cannot read part of the prompt",
                     x=0.02, ha="left", fontsize=11)
        fig.legend(handles=[Patch(color=MUTED, label="Untouched prompt"),
                            Patch(color=BLUE, label="One part of the prompt blocked or erased")],
                   frameon=False, fontsize=8.5, ncol=2, loc="upper left", bbox_to_anchor=(0.02, 0.95))
        fig.tight_layout(rect=(0, 0, 1, 0.9))
        for suffix in ("png", "pdf"):
            fig.savefig(output / f"answers.{suffix}", dpi=180, bbox_inches="tight")
            written.append(f"answers.{suffix}")
        plt.close(fig)
    return written


# -------------------------------------------------------------------- main --


def readout_figure(report: dict[str, Any], output: Path) -> list[str]:
    """One panel per model: how well resisting is read from the last prompt token,
    by layer, for the full prompt and for prompts that withhold evidence."""

    try:
        import matplotlib
    except ImportError:
        return []
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [slug for slug in MODELS if _dig(report["models"], (slug, "readout", "curves"))]
    if not present:
        return []
    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    series = ((RESIST, style.BLUE, "-", "Country + norm"),
              ("opposite_norm_only", style.VERMILLION, "-", "Norm only"),
              ("country_only", style.DARK_PURPLE, "-", "Country only"),
              ("base", style.MAGENTA, (0, (3, 1.5)), "No evidence"))
    fig, axes = plt.subplots(1, len(present), figsize=(2.6 * len(present), 2.3), sharey=True, squeeze=False)
    for axis, slug in zip(axes[0], present, strict=True):
        read = report["models"][slug]["readout"]
        for state, colour, line, label in series:
            if state in read["curves"]:
                axis.plot(read["layers"], read["curves"][state], color=colour, ls=line, lw=1.6, label=label,
                          zorder=4 if state == "base" else 3)
        conflict = _dig(report["models"], (slug, "interaction", "conflict_curve"))
        if conflict:
            axis.plot(conflict["layers"], conflict["values"], color=style.GREEN, lw=2.0, ls=(0, (1, 1.1)),
                      label="Conflict signal")
        text = read["curves"].get(f"text:{RESIST}")
        if text:
            axis.axhline(text[0], color=style.BLACK, lw=1, ls=(0, (4, 2)), label="Prompt words")
        axis.axvline(read["layer"], color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=2)
        axis.text(read["layer"], 0.448, f" Layer {read['layer']}", ha="left", va="bottom", fontsize=9.5,
                  color="#404040")
        axis.set_ylim(0.44, 0.96)
        axis.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
        axis.set_xlim(read["layers"][0], read["layers"][-1])
        axis.set_title(MODELS[slug])
        axis.set_xlabel("Layer")
        style.grid(axis)
    axes[0][0].set_ylabel("AUROC for resistance")
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda at: labels[at] == "Conflict signal")  # the signal last
    handles, labels = [handles[at] for at in order], [labels[at] for at in order]
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.6, columnspacing=1.1, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.885), w_pad=0.5, pad=0.3)
    written = []
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"readout_depth.{suffix}", dpi=300, bbox_inches="tight")
        written.append(f"readout_depth.{suffix}")
    plt.close(fig)
    return written


def erasure_figure(report: dict[str, Any], output: Path) -> list[str]:
    """One panel per model: the share of the stated norm's effect on the decision
    margin that erasing the norm's positions after a given layer still removes."""

    try:
        import matplotlib
    except ImportError:
        return []
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    present = [slug for slug in MODELS if _dig(report["models"], (slug, "erasure", "change"))]
    if not present:
        return []
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": MUTED,
                         "xtick.color": MUTED, "ytick.color": MUTED, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, len(present), figsize=(2.35 * len(present), 2.25), sharey=True, squeeze=False)
    for axis, slug in zip(axes[0], present, strict=True):
        data = report["models"][slug]["erasure"]
        if slug in CLAIM_WINDOW:      # the layers blocked in the knockout band (end-exclusive constant)
            low, high = CLAIM_WINDOW[slug]
            axis.axvspan(low - 0.5, high - 0.5, color=GRID, lw=0, zorder=0, label="Knockout band")
        for target, colour, label in ((RESIST, BLUE, "Contradictory norm"),
                                      ("country_norm", ORANGE, "True norm")):
            values = [entry["difference"] for entry in data["change"][target]]
            axis.plot(data["cuts"], [value / values[0] for value in values], color=colour, lw=1.8,
                      marker="o", ms=2.5, label=label)
        axis.axhline(0, color=MUTED, lw=0.8, zorder=0)
        axis.set_ylim(-0.05, 1.08)
        axis.set_title(MODELS[slug], loc="left", fontsize=9, fontweight="bold")
        axis.set_xlabel("norm erased after layer")
        axis.yaxis.grid(True, color=GRID, lw=0.6)
        axis.set_axisbelow(True)
    axes[0][0].set_ylabel("share of the norm's effect\nstill removed")
    handles, labels = axes[0][0].get_legend_handles_labels()
    order = sorted(range(len(labels)), key=lambda i: labels[i] == "Knockout band")
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=3,
               frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.02), handlelength=1.6, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    written = []
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"erasure_by_layer.{suffix}", dpi=300, bbox_inches="tight")
        written.append(f"erasure_by_layer.{suffix}")
    plt.close(fig)
    return written


def build(root: Path, draws: int) -> dict[str, Any]:
    report: dict[str, Any] = {"judges": list(JUDGES), "report_layer": REPORT_LAYER,
                              "erase_layers": ERASE_LAYERS,
                              "claim_window": CLAIM_WINDOW, "control_window": CONTROL_WINDOW,
                              "bootstrap_draws": draws, "models": {}}
    for slug in MODELS:
        parts = {"behaviour": behaviour(root, slug), "readout": readout(root, slug, draws),
                 "interaction": interaction(root, slug, draws),
                 "localisation": localisation(root, slug, draws), "answers": answers(root, slug, draws),
                 "erasure": erasure(root, slug, draws), "heads": heads(root, slug),
                 "claim_only_benchmark": claim_only_benchmark(root, slug, draws),
                 "margin_validity": margin_validity(root, slug, draws)}
        if any(value is not None for value in parts.values()):
            report["models"][slug] = parts
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=Path("output/mechanistic_analysis"))
    parser.add_argument("--draws", type=int, default=1000)
    args = parser.parse_args(argv)
    report = build(args.root, args.draws)
    output = args.root / "report"
    output.mkdir(parents=True, exist_ok=True)
    (output / "tables.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    (output / "tables.md").write_text(markdown(report), encoding="utf-8")
    written = figures(report, output) + readout_figure(report, output) + erasure_figure(report, output)
    print(f"saved {output}: tables.json, tables.md" + "".join(f", {name}" for name in written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
