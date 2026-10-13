"""Figures for the behavioural appendix, drawn from the paper's own tables.

The numbers are read from the LaTeX tables, so a figure and its table cannot
disagree.

* ``label_shares``: the share of responses judged aligned, partially aligned
  and misaligned under each condition, as stacked bars.
* ``anglophone_gap``: alignment for the five Anglophone countries minus
  alignment for the other 70, for open-ended answers and for the A/B choice.

PYTHONPATH=src python -m behavioral_evaluation.appendix_figures \
  --tables paper/tables/behavioral_evaluation --output paper/figures/behavioral_evaluation
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

MODELS = ("Llama-3.1-8B", "Gemma-3-12B", "Qwen-3.5-27B")
SHORT = {"Contradictory Norm Only": "Contra. Norm Only", "Country + Contradictory Norm": "Country + Contra. Norm",
         "Base": "No Evidence"}


def rows(path: Path) -> list[tuple[str, list[float]]]:
    """Every table row that is a label followed only by numbers."""

    found = []
    for line in path.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.replace("\\\\", "").split("&")]
        if len(cells) < 2:
            continue
        try:
            values = [float(cell.replace("$", "")) for cell in cells[1:]]
        except ValueError:
            continue
        found.append((re.sub(r"\\quad\s*", "", cells[0]), values))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    args.output.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- label shares
    table = rows(args.tables / "label_shares.tex")
    labels = [SHORT.get(name, name) for name, _ in table]
    places = list(range(len(table)))[::-1]
    colours = (style.BLUE, style.YELLOW, style.VERMILLION)
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.4), sharey=True)
    for number, (axis, model) in enumerate(zip(axes, MODELS)):
        for place, (_, values) in zip(places, table):
            left = 0.0
            for value, colour in zip(values[3 * number:3 * number + 3], colours):
                axis.barh(place, value, left=left, height=0.72, color=colour, zorder=2)
                if value >= 9:
                    axis.text(left + value / 2, place, f"{value:.0f}", ha="center", va="center_baseline", fontsize=9,
                              color="white" if colour != style.YELLOW else style.BLACK)
                left += value
        axis.set_xlim(0, 100)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_title(model)
        axis.set_xlabel("Responses (%)")
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(places)
    axes[0].set_yticklabels(labels)
    fig.legend(handles=[Patch(color=colour, label=name) for colour, name in
                        zip(colours, ("Aligned", "Partially aligned", "Misaligned"))],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.4, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"label_shares.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # -------------------------------------------------------- Anglophone gap
    table = rows(args.tables / "anglophone_split.tex")
    split = next(index for index in range(1, len(table)) if table[index][0] == table[0][0])
    blocks = {"Unconstrained": dict(table[:split]), "Constrained": dict(table[split:])}
    names = [name for name, _ in table[:split]]
    places = list(range(len(names)))[::-1]
    colours = {"Unconstrained": style.BLUE, "Constrained": style.VERMILLION}
    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.9), sharey=True)
    for number, (axis, model) in enumerate(zip(axes, MODELS)):
        for offset, (kind, block) in zip((0.2, -0.2), blocks.items()):
            for place, name in zip(places, names):
                if name not in block:
                    continue
                other, anglophone = block[name][2 * number:2 * number + 2]
                gap = anglophone - other
                axis.barh(place + offset, gap, height=0.36, color=colours[kind], zorder=2)
                axis.text(max(gap, 0) + 0.7, place + offset, f"{gap:.1f}", va="center_baseline", fontsize=8)
        axis.axvline(0, color=style.BLACK, lw=0.7, zorder=3)
        axis.set_xlim(-4.5, 32)
        axis.set_ylim(-0.6, len(names) - 0.4)
        axis.set_xticks([0, 10, 20, 30])
        axis.set_title(model)
        axis.set_xlabel("Anglophone gap (points)")
        axis.grid(axis="x", color=style.GRID, lw=0.5, ls=(0, (2, 2)))
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(places)
    axes[0].set_yticklabels([SHORT.get(name, name) for name in names])
    fig.legend(handles=[Patch(color=colours[kind], label=f"{kind} answers") for kind in blocks],
               loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.4, handletextpad=0.4, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.925), w_pad=1.6, pad=0.3)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"anglophone_gap.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
