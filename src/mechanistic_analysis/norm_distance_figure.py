"""Appendix material for the country distances under a stated norm.

Per model and layer: the Spearman correlation, over all pairs of countries,
between the distances with the country alone and the distances with a true or
a contradictory norm also stated. Prints the values at the reporting layer
with bootstrap intervals over countries and draws the two curves by layer.

PYTHONPATH=src python -m mechanistic_analysis.norm_distance_figure \
  --geometry-root output/mechanistic_analysis/country_space \
  --output paper/figures/mechanistic_analysis/country_representation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mechanistic_analysis import REPORT_LAYER
from mechanistic_analysis.country_figure import MODELS
from mechanistic_analysis.wvs_figure import spearman

CONDITIONS = (("country_norm", "True norm"), ("country_opposite_norm", "Contradictory norm"))


def interval(a: np.ndarray, b: np.ndarray, draws: int = 2000, seed: int = 0) -> tuple[float, float]:
    """Resample countries; pairs of a country with itself are dropped."""

    rng = np.random.default_rng(seed)
    n = len(a)
    upper = np.triu_indices(n, 1)
    values = []
    for _ in range(draws):
        rows = rng.integers(0, n, n)
        keep = rows[upper[0]] != rows[upper[1]]
        sub = np.ix_(rows, rows)
        values.append(spearman(a[sub][upper][keep], b[sub][upper][keep]))
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    colours = {"country_norm": style.BLUE, "country_opposite_norm": style.VERMILLION}
    lines = {"country_norm": "-", "country_opposite_norm": (0, (4, 1.5))}
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.2), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        geometry = np.load(args.geometry_root / slug / "geometry.npz")
        layers = [int(layer) for layer in geometry["layers"]]
        alone = geometry["rdm_country_only"].astype(float)
        upper = np.triu_indices(alone.shape[1], 1)
        at = layers.index(REPORT_LAYER[slug])
        report = name + ":"
        for key, label in CONDITIONS:
            stated = geometry[f"rdm_{key}"].astype(float)
            curve = [spearman(alone[index][upper], stated[index][upper]) for index in range(len(layers))]
            axis.plot(layers, curve, color=colours[key], ls=lines[key], lw=1.6, label=label)
            low, high = interval(alone[at], stated[at])
            report += (f" | {label} layer {layers[at]}: {curve[at]:.2f} [{low:.2f}, {high:.2f}], "
                       f"min {min(curve):.2f} at layer {layers[int(np.argmin(curve))]}, last layer {curve[-1]:.2f}")
        true = geometry["rdm_country_norm"].astype(float)[at][upper]
        contra = geometry["rdm_country_opposite_norm"].astype(float)[at][upper]
        report += f" | true vs contradictory {spearman(true, contra):.2f}"
        print(report, flush=True)
        axis.axvline(REPORT_LAYER[slug], color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
        axis.set_xlim(layers[0], layers[-1])
        axis.set_ylim(0.0, 1.03)
        axis.set_title(name)
        axis.set_xlabel("Layer")
        style.grid(axis)
    axes[0].set_ylabel("Similarity to\ncountry-only distances")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"norm_distance_layers.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
