"""Appendix figure for the presentation of the contradictory norm.

Per model: the percentage of answers judged aligned when the same
contradictory norm is attributed to a guide, stated as is, attributed to
hearsay, placed before the country, or followed by the user's doubt. The
dashed line is the country-only rate, where no norm is stated (for Qwen, whose
run has no country-only arm, the full-benchmark rate).

PYTHONPATH=src python -m mechanistic_analysis.presentation_figure \
  --drivers output/mechanistic_analysis/report/drivers.json \
  --output paper/figures/mechanistic_analysis/compliance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanistic_analysis.country_figure import MODELS

ROWS = (("Attributed to a guide", "country_opposite_norm_guide"), ("Stated plainly", None),
        ("Attributed to hearsay", "country_opposite_norm_hearsay"),
        ("Placed before the country", "opposite_norm_country"),
        ("Followed by user's doubt", "country_opposite_norm_doubt"))
COUNTRY_ONLY = {"llama-3.1-8b": 58.8, "gemma-3-12b": 68.0, "qwen-3.5-27b": 75.2}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drivers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    from mechanistic_analysis import figure_style as style

    saved = json.loads(args.drivers.read_text(encoding="utf-8"))["models"]
    style.apply(plt)
    places = list(range(len(ROWS)))[::-1]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.0), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        framing = saved[slug]["framing"]
        plain = 100 * next(iter(framing.values()))["bare"]
        for place, (label, key) in zip(places, ROWS):
            value = plain if key is None else 100 * framing[key]["rate"]["rate"]
            axis.barh(place, value, height=0.7, color="#B4B4B4" if key is None else style.BLUE, zorder=2)
            axis.text(value + 1.5, place, f"{value:.1f}", va="center_baseline", fontsize=9, zorder=5,
                      bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.8})
            effect = "" if key is None else (f"{100 * framing[key]['effect']['difference']:+.1f} "
                                             f"[{100 * framing[key]['effect']['low']:+.1f}, "
                                             f"{100 * framing[key]['effect']['high']:+.1f}]")
            print(name, label, f"{value:.1f}", effect, flush=True)
        axis.axvline(COUNTRY_ONLY[slug], color=style.BLACK, lw=1.1, ls=(0, (4, 2)), zorder=3)
        axis.set_xlim(0, 110)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_title(name)
        axis.set_xlabel("Answers that resist (%)")
        axis.grid(axis="x", color=style.GRID, lw=0.5, ls=(0, (2, 2)))
        axis.set_axisbelow(True)
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(places)
    axes[0].set_yticklabels([label for label, _ in ROWS])
    fig.legend(handles=[Patch(color="#B4B4B4", label="Benchmark presentation"),
                        Patch(color=style.BLUE, label="Other presentations"),
                        Line2D([], [], color=style.BLACK, lw=1.1, ls=(0, (4, 2)), label="Norm never stated")],
               loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.04),
               handlelength=1.6, columnspacing=1.4, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"presentation.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
