"""Figure: how far naming a country moves the state, against how much naming it helps.

One panel per model and one point per country. The horizontal axis is the
country's ``shift_size`` at the reporting layer (``country_space.py``); the
vertical axis is the gain in answers judged aligned from the ``base`` to the
``country_only`` prompt, averaged over the two judges. The five Anglophone
countries are marked and the United States is labelled.

PYTHONPATH=src python -m mechanistic_analysis.country_figure \
  --geometry-root output/mechanistic_analysis/country_space \
  --judgments-root output/behavioral_evaluation/unconstrained/judgments \
  --output output/mechanistic_analysis/report
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from mechanistic_analysis import JUDGES, REPORT_LAYER

MODELS = {"llama-3.1-8b": "Llama-3.1-8B", "gemma-3-12b": "Gemma-3-12B", "qwen-3.5-27b": "Qwen-3.5-27B"}
ANGLOPHONE = ("USA", "CAN", "GBR", "AUS", "NZL")
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--geometry-root", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def country_gain(judgments: Path, slug: str) -> dict[str, float]:
    """Per country, aligned(country_only) - aligned(base) in points, mean of the judges."""

    per_judge = []
    for judge in JUDGES:
        totals: dict[tuple[str, str], list[float]] = defaultdict(list)
        with (judgments / slug / f"{judge}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row["condition"] in ("base", "country_only"):
                        totals[(str(row["norm_id"]).split("__")[0], row["condition"])].append(
                            float(row["alignment"] == "aligned"))
        per_judge.append({code: 100 * (np.mean(totals[(code, "country_only")]) - np.mean(totals[(code, "base")]))
                          for code in {key[0] for key in totals}})
    return {code: float(np.mean([each[code] for each in per_judge])) for code in per_judge[0]}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda values: np.argsort(np.argsort(values)).astype(float)  # noqa: E731
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.4), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        geometry = np.load(args.geometry_root / slug / "geometry.npz")
        countries = [str(code) for code in geometry["countries"]]
        layer = list(geometry["layers"]).index(REPORT_LAYER[slug])
        shift = geometry["shift_size"][layer]
        gains = country_gain(args.judgments_root, slug)
        gain = np.array([gains[code] for code in countries])
        marked = np.array([code in ANGLOPHONE for code in countries])
        slope, intercept = np.polyfit(shift, gain, 1)
        ends = np.array([shift.min(), shift.max()])
        axis.plot(ends, slope * ends + intercept, color=style.BLACK, lw=1, zorder=1)
        axis.scatter(shift[~marked], gain[~marked], s=14, facecolor=style.BLUE, alpha=0.7, linewidths=0, zorder=2,
                     label="Other countries")
        axis.scatter(shift[marked], gain[marked], s=30, marker="^", facecolor=style.VERMILLION,
                     edgecolor=style.BLACK, linewidths=0.4, zorder=3, label="Anglophone countries")
        at = countries.index("USA")
        axis.annotate("US", (shift[at], gain[at]), xytext=(5, 5), textcoords="offset points",
                      fontsize=10, color=style.VERMILLION)
        style.grid(axis)
        axis.set_title(name)
        axis.set_xlabel("Shift from naming the country")
        print(slug, "layer", REPORT_LAYER[slug], "rho", round(spearman(shift, gain), 3),
              "without Anglophone", round(spearman(shift[~marked], gain[~marked]), 3),
              "US rank", int(np.argsort(np.argsort(shift))[countries.index("USA")]) + 1, flush=True)
    axes[0].set_ylabel("Gain in aligned answers\nfrom naming the country")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handletextpad=0.3, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"country_shift.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
