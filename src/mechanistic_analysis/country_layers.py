"""Appendix figure and table for the country-representation analyses, by layer.

For every saved layer and each model, the Spearman correlation over countries of

* the size of the shift from naming the country with the gain in aligned answers;
* model distances between countries with World Values Survey distances, plain
  and controlling for geography (great-circle distance, region, subregion);
* the distinctness of a country (mean distance to the others, on the shift)
  with its resistance rate under a contradictory norm;

and the similarity of the distances under a stated norm to those without one.
Prints the values at the reporting layer with 95% bootstrap intervals over
countries (2,000 draws) and writes ``country_layers.{pdf,png}``.

PYTHONPATH=src python -m mechanistic_analysis.country_layers --root . --output paper/figures/mechanistic_analysis/country_representation
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mechanistic_analysis import JUDGES, REPORT_LAYER

MODELS = {"llama-3.1-8b": "Llama-3.1-8B", "gemma-3-12b": "Gemma-3-12B", "qwen-3.5-27b": "Qwen-3.5-27B"}
ANGLOPHONE = {"USA", "CAN", "GBR", "AUS", "NZL"}


def rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values)); ranks[order] = np.arange(len(values))
    for value in np.unique(values):                      # average ranks over ties
        same = values == value
        if same.sum() > 1:
            ranks[same] = ranks[same].mean()
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def partial(a: np.ndarray, b: np.ndarray, controls: list[np.ndarray]) -> float:
    design = np.column_stack([np.ones(len(a))] + [rank(c) for c in controls])
    left = lambda y: rank(y) - design @ np.linalg.lstsq(design, rank(y), rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(left(a), left(b))[0, 1])


def interval(a: np.ndarray, b: np.ndarray, draws: int = 2000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    found = []
    for _ in range(draws):
        rows = rng.integers(0, len(a), len(a))
        found.append(spearman(a[rows], b[rows]))
    return float(np.percentile(found, 2.5)), float(np.percentile(found, 97.5))


def aligned(judgments: Path, slug: str) -> dict[str, dict[str, float]]:
    per_judge = []
    for judge in JUDGES:
        seen: dict[tuple[str, str], list[float]] = defaultdict(list)
        with (judgments / slug / f"{judge}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    seen[(row["condition"], str(row["norm_id"]).split("__")[0])].append(
                        float(row["alignment"] == "aligned"))
        per_judge.append({key: 100 * float(np.mean(values)) for key, values in seen.items()})
    out: dict[str, dict[str, float]] = defaultdict(dict)
    for key in per_judge[0]:
        out[key[0]][key[1]] = float(np.mean([each[key] for each in per_judge]))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    root = args.root
    table = list(csv.reader((root / "data/processed/references/wvs/distance_matrix.csv").open()))
    survey_codes = table[0][1:]
    survey = np.array([[float(x) for x in row[1:]] for row in table[1:]])
    geography = {row["iso3"]: row for row in csv.DictReader(
        (root / "data/processed/references/geography/countries.csv").open())}

    style.apply(plt)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.4), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        geometry = np.load(root / f"output/mechanistic_analysis/country_space/{slug}/geometry.npz")
        codes = [str(code) for code in geometry["countries"]]
        layers = [int(layer) for layer in geometry["layers"]]
        scores = aligned(root / "output/behavioral_evaluation/unconstrained/judgments", slug)
        gain = np.array([scores["country_only"][c] - scores["base"][c] for c in codes])
        resist = np.array([scores["country_opposite_norm"][c] for c in codes])
        other = np.array([c not in ANGLOPHONE for c in codes])

        both = [c for c in codes if c in survey_codes]
        at = [codes.index(c) for c in both]
        upper = np.triu_indices(len(both), 1)
        wanted = survey[np.ix_([survey_codes.index(c) for c in both], [survey_codes.index(c) for c in both])][upper]
        lat = np.radians([float(geography[c]["latitude"]) for c in both])
        lon = np.radians([float(geography[c]["longitude"]) for c in both])
        i, j = upper
        far = 2 * 6371 * np.arcsin(np.sqrt(np.sin((lat[i] - lat[j]) / 2) ** 2
                                           + np.cos(lat[i]) * np.cos(lat[j]) * np.sin((lon[i] - lon[j]) / 2) ** 2))
        region = np.array([geography[c]["region"] for c in both])
        sub = np.array([geography[c]["subregion"] for c in both])
        controls = [far, (region[i] != region[j]).astype(float), (sub[i] != sub[j]).astype(float)]
        every = np.triu_indices(len(codes), 1)

        curves = {key: [] for key in ("shift", "survey", "survey_controlled", "distinct", "true", "contra")}
        for index in range(len(layers)):
            size = geometry["shift_size"][index]
            distances = geometry["rdm_country_only"][index]
            distinct = (geometry["rdm_shift"][index].sum(axis=1)) / (len(codes) - 1)
            model = distances[np.ix_(at, at)][upper]
            curves["shift"].append(spearman(size, gain))
            curves["survey"].append(spearman(model, wanted))
            curves["survey_controlled"].append(partial(model, wanted, controls))
            curves["distinct"].append(spearman(distinct, resist))
            curves["true"].append(spearman(distances[every], geometry["rdm_country_norm"][index][every]))
            curves["contra"].append(spearman(distances[every], geometry["rdm_country_opposite_norm"][index][every]))
        fixed = layers.index(REPORT_LAYER[slug])
        size = geometry["shift_size"][fixed]
        distinct = geometry["rdm_shift"][fixed].sum(axis=1) / (len(codes) - 1)
        order = np.argsort(np.argsort(size)) + 1
        print(f"\n{name}, layer {REPORT_LAYER[slug]}")
        print("  shift rank of USA CAN GBR AUS NZL:", [int(order[codes.index(c)]) for c in ("USA", "CAN", "GBR", "AUS", "NZL")],
              "| Anglophone / other mean shift %.2f" % (size[~other].mean() / size[other].mean()))
        for label, a, b in (("shift vs gain", size, gain), ("shift vs gain, no Anglophone", size[other], gain[other]),
                            ("distinctness vs resistance", distinct, resist),
                            ("distinctness vs resistance, no Anglophone", distinct[other], resist[other])):
            low, high = interval(a, b)
            print(f"  {label:44s} {spearman(a, b):.2f} [{low:.2f}, {high:.2f}]")
        print(f"  survey {curves['survey'][fixed]:.2f}, controlled {curves['survey_controlled'][fixed]:.2f} | "
              f"similarity true {curves['true'][fixed]:.2f}, contradictory {curves['contra'][fixed]:.2f} | "
              f"median over layers: shift {np.median(curves['shift']):.2f}, survey {np.median(curves['survey']):.2f}, "
              f"distinct {np.median(curves['distinct']):.2f}")

        depth = np.array(layers)
        axis.plot(depth, curves["shift"], color=style.BLUE, lw=1.6, label="Shift size and gain")
        axis.plot(depth, curves["survey"], color=style.VERMILLION, lw=1.6, label="Distances and survey")
        axis.plot(depth, curves["survey_controlled"], color=style.VERMILLION, lw=1.6, ls=(0, (3, 1.5)),
                  label="Controlling for geography")
        axis.plot(depth, curves["distinct"], color=style.GREEN, lw=1.6, label="Distinctness and resistance")
        axis.axhline(0, color=style.BLACK, lw=0.6, zorder=1)
        axis.axvline(REPORT_LAYER[slug], color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=2)
        axis.set_xlim(depth[0], depth[-1])
        axis.set_ylim(-0.25, 0.85)
        axis.set_title(name)
        axis.set_xlabel("Layer")
        style.grid(axis)
    axes[0].set_ylabel("Spearman correlation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.6, columnspacing=1.1, handletextpad=0.4, fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"country_layers.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
