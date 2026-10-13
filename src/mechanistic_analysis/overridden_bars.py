"""Figure: the overridden-not-lost table as grouped bars, one panel per model.

The values are those of the paper's table (percentage of country +
contradictory-norm answers judged aligned), copied here so the figure and the
table can be compared side by side.

PYTHONPATH=src python -m mechanistic_analysis.overridden_bars --output paper/figures/mechanistic_analysis/overridden_not_lost
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = ("Llama-3.1-8B", "Gemma-3-12B", "Qwen-3.5-27B")
GROUPS = ("Correct", "Incorrect")
# Norm erasure, which applies to all three models (attention knockout does not
# apply to Qwen). Records are split by the model's country-only answer: the one
# generated in the intervention run for Llama and Gemma, the benchmark's for
# Qwen, whose run has no country-only arm. Primary judge.
ARMS = (  # label, values[model][group]
    ("Untouched", ((25.5, 7.3), (17.6, 7.0), (53.3, 27.1))),
    ("Norm erased after the band", ((35.7, 12.7), (27.9, 8.6), (61.8, 31.2))),
    ("Norm erased before the band", ((84.7, 27.3), (86.0, 30.5), (93.4, 22.9))),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    colours = ("#B4B4B4", style.SKY, style.BLUE)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.25), sharey=True)
    width = 0.26
    for number, (axis, name) in enumerate(zip(axes, MODELS)):
        for slot, ((_, values), colour) in enumerate(zip(ARMS, colours)):
            for group in range(len(GROUPS)):
                x = group + (slot - 1) * width
                value = values[number][group]
                axis.bar(x, value, width=width * 0.92, color=colour, zorder=2)
                axis.text(x, value + 1.5, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
        axis.set_xticks(range(len(GROUPS)))
        axis.set_xticklabels(GROUPS)
        axis.set_xlabel("Answer from the country alone")
        axis.set_ylim(0, 108)
        axis.set_yticks([0, 25, 50, 75, 100])
        axis.set_title(name)
        axis.grid(axis="y", color=style.GRID, lw=0.5, ls=(0, (2, 2)))
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", length=0)
    axes[0].set_ylabel("Aligned answers (%)")
    fig.legend(handles=[Patch(color=colour, label=label) for (label, _), colour in zip(ARMS, colours)],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.4, handletextpad=0.4, columnspacing=1.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"overridden.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
