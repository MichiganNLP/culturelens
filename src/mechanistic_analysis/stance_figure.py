"""Appendix figure and numbers for the stance of the answers.

Figure, per model: the share of answers with each stance toward the
contradictory norm, on the intervention sample, untouched and with the norm
blocked or erased, and for the prompts without the country and without the
norm. Prints the full-benchmark shares for the table.

PYTHONPATH=src python -m mechanistic_analysis.stance_figure \
  --drivers output/mechanistic_analysis/report/drivers.json \
  --output paper/figures/mechanistic_analysis/compliance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanistic_analysis.country_figure import MODELS

ROWS = (("Untouched", "false"), ("Blocked, norm band", "false-noclaim-win"), ("Blocked, all layers", "false-noclaim"),
        ("Erased, before band", "false-erase-before"), ("Norm without country", "claim-only"),
        ("Country only", "country"))


def shares(found: dict) -> tuple[float, float, float, float]:
    rate = lambda name: 100 * found["share"][name]["rate"]  # noqa: E731
    return (rate("follows"), rate("follows_with_caveat"), rate("challenges") + rate("sets_aside"),
            rate("no_clear_recommendation"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drivers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from mechanistic_analysis import figure_style as style

    saved = json.loads(args.drivers.read_text(encoding="utf-8"))["models"]
    style.apply(plt)
    parts = (("Follows the norm", style.VERMILLION), ("Follows it with doubt", style.YELLOW),
             ("Recommends a different behavior", style.BLUE), ("No clear recommendation", "#B4B4B4"))
    places = list(range(len(ROWS)))[::-1]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.2), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        stance = saved[slug]["stance"]
        whole = stance["benchmark"]
        print(name, "benchmark", [round(value, 1) for value in shares(whole)],
              "restates", round(100 * whole["restates_claim"], 1),
              "as custom", round(100 * whole["attributes_to_country"], 1),
              "aligned given", {k: round(100 * v, 1) for k, v in whole["aligned_given_stance"].items()},
              "given not aligned", {k: round(100 * v, 1) for k, v in whole["stance_given_not_aligned"].items()},
              flush=True)
        for place, (label, arm) in zip(places, ROWS):
            if arm not in stance["arms"]:
                axis.text(50, place, "not applicable", va="center", ha="center", fontsize=8.5, color=style.GREY,
                          style="italic")
                continue
            values = shares(stance["arms"][arm])
            left = 0.0
            for value, (_, colour) in zip(values, parts):
                axis.barh(place, value, left=left, height=0.72, color=colour, zorder=2)
                if value >= 9:
                    axis.text(left + value / 2, place, f"{value:.0f}", va="center_baseline", ha="center", fontsize=8.5,
                              color="white" if colour != style.YELLOW else style.BLACK, zorder=5)
                left += value
            print(name, label, [round(value, 1) for value in values], "restates",
                  round(100 * stance["arms"][arm]["restates_claim"], 1), flush=True)
        axis.set_xlim(0, 100)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_title(name)
        axis.set_xlabel("Answers (%)")
        axis.tick_params(axis="y", length=0)
    axes[0].set_yticks(places)
    axes[0].set_yticklabels([label for label, _ in ROWS])
    fig.legend(handles=[Patch(color=colour, label=label) for label, colour in parts], loc="upper center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 1.04), handlelength=1.2, columnspacing=1.1, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"stance.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
