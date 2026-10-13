"""Appendix material for the shift from naming the country.

Prints the numbers of the summary table (ranks, the Anglophone ratio, the
correlation between a country's shift size and its behavioural gain with
bootstrap intervals over countries) and draws that correlation at every layer
the geometry was saved at, one line per model.

PYTHONPATH=src python -m mechanistic_analysis.country_shift_layers \
  --geometry-root output/mechanistic_analysis/country_space \
  --judgments-root output/behavioral_evaluation/unconstrained/judgments \
  --output paper/figures/mechanistic_analysis/country_representation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from mechanistic_analysis import REPORT_LAYER
from mechanistic_analysis.country_figure import ANGLOPHONE, MODELS, country_gain, spearman


def interval(a: np.ndarray, b: np.ndarray, draws: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        rows = rng.integers(0, len(a), len(a))
        values.append(spearman(a[rows], b[rows]))
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
        gains = country_gain(args.judgments_root, slug)
        gain = np.array([gains[code] for code in countries])
        marked = np.array([code in ANGLOPHONE for code in countries])
        every = [spearman(geometry["shift_size"][at], gain) for at in range(len(layers))]
        without = [spearman(geometry["shift_size"][at][~marked], gain[~marked]) for at in range(len(layers))]
        axis.plot(layers, every, color=style.BLUE, lw=1.6, label="All 75 countries")
        axis.plot(layers, without, color=style.VERMILLION, lw=1.6, ls=(0, (4, 1.5)), label="Without Anglophone")
        axis.axvline(REPORT_LAYER[slug], color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
        axis.axhline(0, color=style.BLACK, lw=0.6, zorder=1)
        axis.set_xlim(layers[0], layers[-1])
        axis.set_ylim(-0.5, 0.85)
        axis.set_title(name)
        axis.set_xlabel("Layer")
        style.grid(axis)
        at = layers.index(REPORT_LAYER[slug])
        shift = geometry["shift_size"][at]
        rank = np.argsort(np.argsort(shift)) + 1
        low, high = interval(shift, gain)
        low2, high2 = interval(shift[~marked], gain[~marked])
        print(f"{name}: ranks " + ", ".join(f"{code} {int(rank[countries.index(code)])}" for code in ANGLOPHONE)
              + f" | Anglophone/others {shift[marked].mean() / shift[~marked].mean():.2f}"
              + f" | US/others {shift[countries.index('USA')] / shift[~marked].mean():.2f}"
              + f" | rho {spearman(shift, gain):.2f} [{low:.2f}, {high:.2f}]"
              + f" | without {spearman(shift[~marked], gain[~marked]):.2f} [{low2:.2f}, {high2:.2f}]"
              + f" | positive at {sum(value > 0 for value in every)}/{len(every)} layers, "
              + f"median {np.median(every):.2f}", flush=True)
    axes[0].set_ylabel("Correlation with\nbehavioral gain")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"country_shift_layers.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
