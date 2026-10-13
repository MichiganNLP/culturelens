"""Appendix material for the agreement with the World Values Survey.

Per model: the Spearman correlation between model and survey distances over
pairs of countries, and the partial correlation after geography (great-circle
distance between centroids, different region, different subregion) is
regressed out of both, at every layer the geometry was saved at. Prints the
values at the reporting layer with permutation p-values and how well geography
alone tracks the survey, and draws the two curves by layer.

PYTHONPATH=src python -m mechanistic_analysis.wvs_figure \
  --geometry-root output/mechanistic_analysis/country_space \
  --references data/processed/references \
  --output paper/figures/mechanistic_analysis/country_representation
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from mechanistic_analysis import REPORT_LAYER
from mechanistic_analysis.country_figure import MODELS


def ranks(values: np.ndarray) -> np.ndarray:
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts - 1) / 2)[inverse]


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.corrcoef(ranks(a), ranks(b))[0, 1])


def partial(a: np.ndarray, b: np.ndarray, controls: list[np.ndarray]) -> float:
    design = np.column_stack([np.ones(len(a))] + [ranks(control) for control in controls])
    left = lambda y: ranks(y) - design @ np.linalg.lstsq(design, ranks(y), rcond=None)[0]  # noqa: E731
    return float(np.corrcoef(left(a), left(b))[0, 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--references", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--permutations", type=int, default=2000)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    with (args.references / "wvs" / "distance_matrix.csv").open(encoding="utf-8") as handle:
        table = list(csv.reader(handle))
    surveyed = table[0][1:]
    survey = np.array([[float(cell) for cell in row[1:]] for row in table[1:]])
    with (args.references / "geography" / "countries.csv").open(encoding="utf-8") as handle:
        place = {row["iso3"]: row for row in csv.DictReader(handle)}

    style.apply(plt)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.2), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        geometry = np.load(args.geometry_root / slug / "geometry.npz")
        countries = [str(code) for code in geometry["countries"]]
        kept = [code for code in countries if code in surveyed]
        rows = [countries.index(code) for code in kept]
        upper = np.triu_indices(len(kept), 1)
        w = survey[np.ix_([surveyed.index(code) for code in kept], [surveyed.index(code) for code in kept])][upper]
        latitude = np.radians([float(place[code]["latitude"]) for code in kept])
        longitude = np.radians([float(place[code]["longitude"]) for code in kept])
        i, j = upper
        far = 2 * 6371 * np.arcsin(np.sqrt(np.sin((latitude[i] - latitude[j]) / 2) ** 2 + np.cos(latitude[i])
                                           * np.cos(latitude[j]) * np.sin((longitude[i] - longitude[j]) / 2) ** 2))
        region = np.array([place[code]["region"] for code in kept])
        subregion = np.array([place[code]["subregion"] for code in kept])
        controls = [far, (region[i] != region[j]).astype(float), (subregion[i] != subregion[j]).astype(float)]
        layers = [int(layer) for layer in geometry["layers"]]
        distances = geometry["rdm_country_only"][:, rows][:, :, rows].astype(float)
        plain = [spearman(distances[at][upper], w) for at in range(len(layers))]
        cleaned = [partial(distances[at][upper], w, controls) for at in range(len(layers))]
        axis.plot(layers, plain, color=style.BLUE, lw=1.6, label="Correlation with the survey")
        axis.plot(layers, cleaned, color=style.VERMILLION, lw=1.6, ls=(0, (4, 1.5)), label="After removing geography")
        axis.axvline(REPORT_LAYER[slug], color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
        axis.axhline(0, color=style.BLACK, lw=0.6, zorder=1)
        axis.set_xlim(layers[0], layers[-1])
        axis.set_ylim(-0.1, 0.7)
        axis.set_title(name)
        axis.set_xlabel("Layer")
        style.grid(axis)
        at = layers.index(REPORT_LAYER[slug])
        rng = np.random.default_rng(0)
        beat_plain = beat_cleaned = 0
        for _ in range(args.permutations):
            order = rng.permutation(len(kept))
            shuffled = distances[at][np.ix_(order, order)][upper]
            beat_plain += abs(spearman(shuffled, w)) >= abs(plain[at])
            beat_cleaned += abs(partial(shuffled, w, controls)) >= abs(cleaned[at])
        best = int(np.argmax(plain))
        print(f"{name}: countries {len(kept)} pairs {len(w)} | layer {layers[at]}: {plain[at]:.2f} "
              f"(p {(beat_plain + 1) / (args.permutations + 1):.4f}), partial {cleaned[at]:.2f} "
              f"(p {(beat_cleaned + 1) / (args.permutations + 1):.4f}) | best layer {layers[best]}: "
              f"{plain[best]:.2f}, partial {cleaned[best]:.2f} | positive at "
              f"{sum(value > 0 for value in plain)}/{len(plain)} layers | survey vs geographic distance "
              f"{spearman(w, far):.2f}, vs different region {spearman(w, controls[1]):.2f}", flush=True)
    axes[0].set_ylabel("Spearman correlation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"wvs_layers.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
