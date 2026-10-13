"""Appendix material for country distinctness and resistance.

Distinctness of a country is its mean cosine distance to the other countries,
on the mean shift from naming it. Per model and layer: its Spearman correlation
with the country's resistance rate (percentage aligned under a contradictory
norm, mean of the two judges), and the same after country-only alignment is
regressed out of both. Prints the values at the reporting layer with bootstrap
intervals over countries and draws the curves by layer.

PYTHONPATH=src python -m mechanistic_analysis.distinctness_figure \
  --geometry-root output/mechanistic_analysis/country_space \
  --judgments-root output/behavioral_evaluation/unconstrained/judgments \
  --output paper/figures/mechanistic_analysis/country_representation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from behavioral_evaluation.country_resistance_figure import rates
from mechanistic_analysis import REPORT_LAYER
from mechanistic_analysis.country_figure import ANGLOPHONE, MODELS
from mechanistic_analysis.wvs_figure import partial, spearman


def interval(function, size: int, draws: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = [function(rng.integers(0, size, size)) for _ in range(draws)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.2), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        geometry = np.load(args.geometry_root / slug / "geometry.npz")
        countries = [str(code) for code in geometry["countries"]]
        layers = [int(layer) for layer in geometry["layers"]]
        found = rates(args.judgments_root, slug)
        resist = np.array([found["country_opposite_norm"][code] for code in countries])
        known = np.array([found["country_only"][code] for code in countries])
        marked = np.array([code in ANGLOPHONE for code in countries])
        distance = geometry["rdm_shift"].astype(float)
        size = len(countries)
        distinct = distance.sum(axis=2) / (size - 1)                       # layers x countries
        plain = [spearman(distinct[at], resist) for at in range(len(layers))]
        cleaned = [partial(distinct[at], resist, [known]) for at in range(len(layers))]
        axis.plot(layers, plain, color=style.BLUE, lw=1.6, label="Correlation with resistance")
        axis.plot(layers, cleaned, color=style.VERMILLION, lw=1.6, ls=(0, (4, 1.5)),
                  label="After removing country-only alignment")
        axis.axvline(REPORT_LAYER[slug], color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
        axis.axhline(0, color=style.BLACK, lw=0.6, zorder=1)
        axis.set_xlim(layers[0], layers[-1])
        axis.set_ylim(-0.5, 0.85)
        axis.set_title(name)
        axis.set_xlabel("Layer")
        style.grid(axis)
        at = layers.index(REPORT_LAYER[slug])
        d = distinct[at]
        a = interval(lambda rows: spearman(d[rows], resist[rows]), size)
        rest = np.where(~marked)[0]
        b = interval(lambda rows: spearman(d[rest][rows], resist[rest][rows]), len(rest))
        c = interval(lambda rows: partial(d[rows], resist[rows], [known[rows]]), size)
        print(f"{name}: layer {layers[at]} | all {plain[at]:.2f} [{a[0]:.2f}, {a[1]:.2f}] | without Anglophone "
              f"{spearman(d[~marked], resist[~marked]):.2f} [{b[0]:.2f}, {b[1]:.2f}] | controlling country-only "
              f"{cleaned[at]:.2f} [{c[0]:.2f}, {c[1]:.2f}] | distinctness vs country-only {spearman(d, known):.2f} | "
              f"layers: median {np.median(plain):.2f}, positive {sum(v > 0 for v in plain)}/{len(plain)}, max "
              f"{max(plain):.2f} at {layers[int(np.argmax(plain))]}; partial median {np.median(cleaned):.2f}", flush=True)
    axes[0].set_ylabel("Spearman correlation")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"distinctness_layers.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
