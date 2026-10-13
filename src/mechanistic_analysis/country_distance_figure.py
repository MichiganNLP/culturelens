"""Appendix material for the distances between countries.

Draws a two-dimensional map of the 75 countries per model (classical
multidimensional scaling of the cosine distances between country vectors under
country_only at the reporting layer), coloured by world region, and prints the
agreement between models and how far distances follow regions.

PYTHONPATH=src python -m mechanistic_analysis.country_distance_figure \
  --geometry-root output/mechanistic_analysis/country_space \
  --geography data/processed/references/geography/countries.csv \
  --output paper/figures/mechanistic_analysis/country_representation
"""

from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path

import numpy as np

from mechanistic_analysis import REPORT_LAYER
from mechanistic_analysis.country_figure import MODELS, spearman

LABELLED = ("USA", "CHN", "JPN", "IND", "SAU", "EGY", "BRA", "DEU", "RUS", "ETH", "FJI", "MEX")


def scaled(distance: np.ndarray) -> np.ndarray:
    """Classical multidimensional scaling to two dimensions."""

    n = len(distance)
    centre = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * centre @ (distance ** 2) @ centre
    values, vectors = np.linalg.eigh(gram)
    order = np.argsort(values)[::-1][:2]
    return vectors[:, order] * np.sqrt(np.maximum(values[order], 0))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--geography", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    from mechanistic_analysis import figure_style as style

    with args.geography.open(encoding="utf-8") as handle:
        region = {row["iso3"]: row["region"] for row in csv.DictReader(handle)}
    colours = {"Africa": style.VERMILLION, "Americas": style.GREEN, "Asia": style.BLUE,
               "Europe": style.DARK_PURPLE, "Oceania": style.MAGENTA}
    style.apply(plt)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.8))
    matrices = {}
    labels = []
    for axis, (slug, name) in zip(axes, MODELS.items()):
        geometry = np.load(args.geometry_root / slug / "geometry.npz")
        countries = [str(code) for code in geometry["countries"]]
        distance = geometry["rdm_country_only"][list(geometry["layers"]).index(REPORT_LAYER[slug])].astype(float)
        matrices[name] = distance
        points = scaled(distance)
        groups = np.array([region.get(code, "Asia") for code in countries])
        for label, colour in colours.items():
            rows = groups == label
            axis.scatter(points[rows, 0], points[rows, 1], s=16, color=colour, alpha=0.8, linewidths=0, zorder=2)
        labels.append((axis, countries, points, groups))
        axis.margins(0.14)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(name)
        upper = np.triu_indices(len(countries), 1)
        same = groups[upper[0]] == groups[upper[1]]
        print(f"{name}: mean distance within region {distance[upper][same].mean():.3f}, between "
              f"{distance[upper][~same].mean():.3f}, ratio {distance[upper][same].mean() / distance[upper][~same].mean():.2f}, "
              f"rho with different-region {spearman(distance[upper], (~same).astype(float)):.2f}", flush=True)
    upper = np.triu_indices(75, 1)
    rng = np.random.default_rng(0)
    for a, b in combinations(matrices, 2):
        draws = []
        for _ in range(2000):
            rows = rng.integers(0, 75, 75)
            sub = np.ix_(rows, rows)
            keep = np.triu_indices(75, 1)
            x, y = matrices[a][sub][keep], matrices[b][sub][keep]
            mask = rows[keep[0]] != rows[keep[1]]
            draws.append(spearman(x[mask], y[mask]))
        print(f"{a} vs {b}: rho {spearman(matrices[a][upper], matrices[b][upper]):.2f} "
              f"[{np.percentile(draws, 2.5):.2f}, {np.percentile(draws, 97.5):.2f}]  pairs {len(upper[0])}", flush=True)
    fig.legend(handles=[Line2D([], [], ls="", marker="o", markersize=5, color=colour, label=label)
                        for label, colour in colours.items()],
               loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 1.03), handletextpad=0.1,
               columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=0.6, pad=0.3)
    # Labels go where they cover the fewest dots and no other label: for each one, the
    # offset (of eight around its dot) furthest from every other dot and placed label.
    from matplotlib import patheffects
    fig.canvas.draw()
    halo = [patheffects.withStroke(linewidth=2.4, foreground="white")]
    offsets = [(dx, dy) for dx in (-15, 0, 15) for dy in (-10, 0, 10) if (dx, dy) != (0, 0)]
    for axis, countries, points, groups in labels:
        pixels = axis.transData.transform(points)
        placed: list[np.ndarray] = []
        for code in LABELLED:
            if code not in countries:
                continue
            at = countries.index(code)
            others = np.vstack([np.delete(pixels, at, axis=0), *placed]) if placed else np.delete(pixels, at, axis=0)
            scale = fig.dpi / 72
            best = max(offsets, key=lambda off: np.min(np.linalg.norm(
                others - (pixels[at] + np.array(off) * scale * 1.3), axis=1)))
            centre = pixels[at] + np.array(best) * scale * 1.3
            placed.append(centre[None, :])
            axis.annotate(code, points[at], xytext=best, textcoords="offset points", fontsize=7.5,
                          ha="center", va="center", color=colours[groups[at]], zorder=6, path_effects=halo,
                          arrowprops={"arrowstyle": "-", "color": colours[groups[at]], "lw": 0.5,
                                      "shrinkA": 0, "shrinkB": 1.5})
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"country_map.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
