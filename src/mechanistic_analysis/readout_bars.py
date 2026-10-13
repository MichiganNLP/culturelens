"""Figure: the decision-readout table as bars, one panel per model.

The values are those of the paper's readout table (AUROC for resisting at the
reporting layer, primary judge), copied here so the figure and the table can
be compared side by side.

PYTHONPATH=src python -m mechanistic_analysis.readout_bars --output paper/figures/mechanistic_analysis/decision_readout
"""

from __future__ import annotations

import argparse
from pathlib import Path

BLUE, LIGHT_BLUE, ORANGE, AQUA, INK, MUTED, GRID = (
    "#2a78d6", "#86b6ef", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df")
MODELS = ("Llama-3.1-8B", "Gemma-3-12B", "Qwen-3.5-27B")
ROWS = (  # label, colour, values per model
    ("Prompt words (no model)", "#b9b8b3", (0.652, 0.683, 0.683)),
    ("Base", "#f4b294", (0.623, 0.666, 0.691)),
    ("Country only", ORANGE, (0.622, 0.680, 0.710)),
    ("Contradictory norm only", LIGHT_BLUE, (0.747, 0.790, 0.844)),
    ("Country + contradictory norm", BLUE, (0.791, 0.841, 0.917)),
    ("Conflict signal (no labels)", AQUA, (0.575, 0.575, 0.731)),
)
GAIN = (0.045, 0.051, 0.073)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": INK, "text.color": INK,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.0, 2.0), sharey=True)
    places = list(range(len(ROWS)))[::-1]
    for number, (axis, name) in enumerate(zip(axes, MODELS)):
        for place, (label, colour, values) in zip(places, ROWS):
            value = values[number]
            axis.barh(place, value - 0.5, left=0.5, height=0.68, color=colour)
            note = f"{value:.3f}".lstrip("0")
            if label.startswith("Country +"):
                note += f"  (+{GAIN[number]:.3f})".replace("+0.", "+.")
            axis.text(value + 0.008, place, note, va="center", fontsize=6.5, color=INK)
        axis.set_xlim(0.5, 1.0)
        axis.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9])
        axis.set_xticklabels([".5", ".6", ".7", ".8", ".9"])
        axis.grid(axis="x", color=GRID, lw=0.5)
        axis.set_axisbelow(True)
        axis.set_title(name, fontsize=8, fontweight="bold", loc="left")
        axis.set_xlabel("AUROC for resisting")
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(places)
    axes[0].set_yticklabels([row[0] for row in ROWS])
    fig.subplots_adjust(left=0.25, right=0.995, top=0.88, bottom=0.2, wspace=0.12)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"readout_bars.{suffix}", dpi=220, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
