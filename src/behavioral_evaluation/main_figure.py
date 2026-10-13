"""Figure: the headline behavioural result, one panel per model.

Unconstrained alignment under five conditions, averaged over the two judges.
The values are those of the paper's behavioural results table (the mean of its
Gemma-judge and Qwen-judge entries), copied here so the figure and the table
agree.

PYTHONPATH=src python -m behavioral_evaluation.main_figure --output paper/figures/behavioral_evaluation
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = ("Llama-3.1-8B", "Gemma-3-12B", "Qwen-3.5-27B")
# label, kind, (Gemma judge, Qwen judge) per model
ROWS = (
    ("No evidence", "none", ((43.5, 43.9), (41.4, 39.4), (51.9, 51.8))),
    ("Country", "helps", ((57.9, 58.6), (64.1, 62.3), (75.2, 73.8))),
    ("Country + value", "helps", ((75.0, 76.8), (78.9, 76.7), (88.9, 88.7))),
    ("Country + true norm", "helps", ((93.1, 92.0), (94.0, 92.1), (95.9, 95.0))),
    ("Country + contradictory norm", "harms", ((17.9, 17.4), (17.1, 15.3), (44.0, 41.8))),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    colours = {"none": "#B4B4B4", "helps": style.BLUE, "harms": style.VERMILLION}
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.1), sharey=True)
    places = list(range(len(ROWS)))[::-1]
    for number, (axis, name) in enumerate(zip(axes, MODELS)):
        values = [sum(row[2][number]) / 2 for row in ROWS]
        for place, (_, kind, _), value in zip(places, ROWS, values):
            axis.barh(place, value, height=0.72, color=colours[kind], zorder=2)
            axis.text(value + 1.5, place, f"{value:.1f}", va="center_baseline", fontsize=9, zorder=5,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6})
        axis.axvline(values[1], color=style.BLACK, lw=1.0, ls=(0, (4, 2)), zorder=3)   # what the country alone gives
        axis.set_xlim(0, 112)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_title(name)
        axis.set_xlabel("Aligned answers (%)")
        axis.grid(axis="x", color=style.GRID, lw=0.5, ls=(0, (2, 2)))
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
        print(name, [round(value, 1) for value in values], "drop from country", round(values[4] - values[1], 1))
    axes[0].set_yticks(places)
    axes[0].set_yticklabels([row[0] for row in ROWS])
    fig.tight_layout(w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"main_results.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
