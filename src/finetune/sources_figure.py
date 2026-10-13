"""Figure: where the effects of SFT + CF come from, one panel per model.

Untuned and fine-tuned models side by side on five quantities: aligned answers
from the country alone and under a contradictory norm, each with the right
country and with a culturally distant one in its place, and the share of
answers that do what a person in the situation asked. Values are two-judge
(or two-coder) means from the country-swap and request experiments; untuned
models on 500 records, fine-tuned models on all 2,797 test records, requests
on 500.

PYTHONPATH=src python -m finetune.sources_figure --output paper/figures/finetuning
"""

from __future__ import annotations

import argparse
from pathlib import Path

MODELS = ("Llama-3.1-8B", "Gemma-3-12B", "Qwen-3.5-27B")
# label, (untuned per model), (SFT + CF per model)
ROWS = (
    ("Country, right", (60.5, 68.6, 75.8), (69.3, 71.1, 75.9)),
    ("Country, wrong", (29.4, 15.7, 25.1), (52.7, 55.0, 56.5)),
    ("Contra. norm, right country", (18.8, 12.9, 41.8), (85.6, 87.7, 92.8)),
    ("Contra. norm, wrong country", (8.3, 3.4, 21.7), (76.9, 76.8, 82.0)),
    ("Request honored", (74.9, 83.7, 84.1), (25.6, 36.0, 21.0)),
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
    colours = ("#B4B4B4", style.BLUE)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.9), sharey=True)
    places = list(range(len(ROWS)))[::-1]
    for number, (axis, name) in enumerate(zip(axes, MODELS)):
        for place, (_, untuned, tuned) in zip(places, ROWS):
            for offset, value, colour in ((0.2, untuned[number], colours[0]), (-0.2, tuned[number], colours[1])):
                axis.barh(place + offset, value, height=0.36, color=colour, zorder=2)
                axis.text(value + 1.5, place + offset, f"{value:.1f}", va="center_baseline", fontsize=8)
        axis.axhline(0.5, color=style.BLACK, lw=0.6, zorder=1)        # the request row measures something else
        axis.set_xlim(0, 112)
        axis.set_ylim(-0.6, len(ROWS) - 0.4)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_title(name)
        axis.set_xlabel("Answers (%)")
        axis.grid(axis="x", color=style.GRID, lw=0.5, ls=(0, (2, 2)))
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(places)
    axes[0].set_yticklabels([row[0] for row in ROWS])
    fig.legend(handles=[Patch(color=colours[0], label="Untuned"), Patch(color=colours[1], label="SFT + CF")],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.4, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.925), w_pad=1.2, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"sources.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
