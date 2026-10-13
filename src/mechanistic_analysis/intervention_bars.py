"""Figure: the norm-localization table as bars, one panel per model.

The values are those of the paper's intervention table (percentage of answers
to the country + contradictory-norm prompt judged aligned), copied here so the
figure and the table can be compared side by side.

PYTHONPATH=src python -m mechanistic_analysis.intervention_bars --output paper/figures/mechanistic_analysis/norm_localization
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = ("Llama-3.1-8B", "Gemma-3-12B", "Qwen-3.5-27B")
ROWS = (  # label, kind, values per model (None where the intervention does not apply)
    ("Untouched", "none", (18.0, 14.2, 47.0)),
    ("Knockout, all layers", "knockout", (58.0, 66.5, None)),
    ("Knockout, mid-depth band", "knockout", (47.0, 59.0, None)),
    ("Knockout, later band", "knockout", (19.5, 16.5, None)),
    ("Knockout, eight heads", "knockout", (46.5, 23.5, None)),
    ("Erasure, before the band", "erasure", (61.0, 68.2, 76.5)),
    ("Erasure, after the band", "erasure", (26.2, 21.8, 54.5)),
)
NEVER_STATED = (58.8, 68.0, None)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    colours = {"none": "#B4B4B4", "knockout": style.BLUE, "erasure": style.VERMILLION}
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.25), sharey=True)
    places = list(range(len(ROWS)))[::-1]
    for number, (axis, name) in enumerate(zip(axes, MODELS)):
        for place, (_, kind, values) in zip(places, ROWS):
            value = values[number]
            if value is None:
                axis.text(2, place, "not applicable", va="center", fontsize=8.5, color=style.GREY, style="italic")
                continue
            axis.barh(place, value, height=0.7, color=colours[kind], zorder=2)
            axis.text(value + 1.5, place, f"{value:.1f}", va="center_baseline", fontsize=9, zorder=5,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
        if NEVER_STATED[number] is not None:
            axis.axvline(NEVER_STATED[number], color=style.BLACK, lw=1.1, ls=(0, (4, 2)), zorder=3)
        axis.set_xlim(0, 92)
        axis.set_xticks([0, 20, 40, 60, 80])
        axis.set_title(name)
        axis.set_xlabel("Aligned answers (%)")
        axis.grid(axis="x", color=style.GRID, lw=0.5, ls=(0, (2, 2)))
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(places)
    axes[0].set_yticklabels([row[0] for row in ROWS])
    fig.legend(handles=[Patch(color=colours["none"], label="Untouched"),
                        Patch(color=colours["knockout"], label="Attention knockout"),
                        Patch(color=colours["erasure"], label="Norm erasure"),
                        Line2D([], [], color=style.BLACK, lw=1.1, ls=(0, (4, 2)), label="Norm never stated")],
               loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.6, columnspacing=1.4, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"interventions.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
