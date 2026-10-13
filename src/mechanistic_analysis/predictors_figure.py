"""Appendix figure for the predictors of resistance.

Per model: the odds ratio of each predictor in the logistic regression for
resisting a contradictory norm, with its 95% bootstrap interval, on a
logarithmic axis.

PYTHONPATH=src python -m mechanistic_analysis.predictors_figure \
  --drivers output/mechanistic_analysis/report/drivers.json \
  --output paper/figures/mechanistic_analysis/compliance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanistic_analysis.country_figure import MODELS

ROWS = (  # label in the figure, key in the saved regression, group
    ("Aligned, no evidence", "Right with no country and no norm", 0),
    ("Aligned, country only", "Right with the country alone", 0),
    ("Aligned, norm only", "Right with the true norm alone", 0),
    ("Aligned, country + norm", "Right with country + true norm", 0),
    ("Record has a value", "Record has a value", 1),
    ("Do's and don'ts", "Category: dos_and_donts (vs etiquette)", 2),
    ("Business culture", "Category: business_culture (vs etiquette)", 2),
    ("Greetings", "Category: greetings (vs etiquette)", 2),
    ("Communication", "Category: communication (vs etiquette)", 2),
    ("Other considerations", "Category: other_considerations (vs etiquette)", 2),
    ("Core concepts", "Category: core_concepts (vs etiquette)", 2),
    ("Religion", "Category: religion (vs etiquette)", 2),
    ("Family", "Category: family (vs etiquette)", 2),
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--drivers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, NullLocator

    from mechanistic_analysis import figure_style as style

    saved = json.loads(args.drivers.read_text(encoding="utf-8"))["models"]
    style.apply(plt)
    colours = (style.BLUE, style.GREEN, style.VERMILLION)
    places = list(range(len(ROWS)))[::-1]
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 3.3), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        ratios = saved[slug]["resisting_model"]["odds_ratio"]
        for place, (label, key, group) in zip(places, ROWS):
            found = ratios[key]
            axis.plot([found["low"], found["high"]], [place, place], color=colours[group], lw=1.4, zorder=2)
            axis.plot(found["or"], place, "o", color=colours[group], ms=4, zorder=3)
            print(name, label, f"{found['or']:.2f} [{found['low']:.2f}, {found['high']:.2f}]", flush=True)
        for edge in (8.5, 7.5):
            axis.axhline(edge, color=style.GRID, lw=0.8, zorder=1)
        axis.axvline(1, color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
        axis.set_xscale("log")
        axis.set_xlim(0.4, 5)
        axis.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
        axis.xaxis.set_minor_locator(NullLocator())
        axis.set_xticklabels(["0.5", "1", "2", "4"])
        axis.set_ylim(-0.6, len(ROWS) - 0.4)
        axis.set_yticks(places, [label for label, _, _ in ROWS])
        axis.tick_params(axis="y", length=0, right=False)
        axis.set_title(name)
        axis.set_xlabel("Odds ratio")
        axis.grid(True, axis="x", color=style.GRID, lw=0.5, ls=(0, (2, 2)))
        axis.set_axisbelow(True)
    fig.tight_layout(w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"resistance_predictors.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
