"""Appendix figure for the attention knockout: where the norm is read.

Per model: the mean change in the decision margin on the country + contradictory
norm prompt when the answer cannot attend to a span, for each window of layers.
The window fixed beforehand as the norm band is shaded.

PYTHONPATH=src python -m mechanistic_analysis.knockout_figure \
  --localize-root output/mechanistic_analysis/localize \
  --output paper/figures/mechanistic_analysis/norm_localization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanistic_analysis import CLAIM_WINDOW
from mechanistic_analysis.country_figure import MODELS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--localize-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    spans = (("claim", "Norm", style.BLUE, "-", "o"), ("country", "Country context", style.VERMILLION, (0, (4, 1.5)), "s"),
             ("control", "Control span", style.GREEN, (0, (1, 1.2)), "^"))
    models = [slug for slug in MODELS if slug in CLAIM_WINDOW]
    fig, axes = plt.subplots(len(models), 1, figsize=(3.75, 4.1))
    for axis, slug in zip(axes, models):
        saved = json.loads((args.localize_root / slug / "metrics.json").read_text(encoding="utf-8"))
        windows = [tuple(window) for window in saved["windows"][:-1]]
        places = list(range(len(windows)))
        band = windows.index(tuple(CLAIM_WINDOW[slug]))
        axis.axvspan(band - 0.5, band + 0.5, color="#e6e6e6", zorder=0)
        for span, label, colour, dash, marker in spans:
            change = saved["mean_change"][f"country_opposite_norm__{span}"]
            values = [change[f"{start}-{stop}"] for start, stop in windows]
            axis.plot(places, values, color=colour, lw=1.6, ls=dash, marker=marker, ms=3.5, label=label)
            print(MODELS[slug], label, [round(value, 2) for value in values], "all layers",
                  round(change[f"0-{windows[-1][1]}"], 2), flush=True)
        axis.axhline(0, color=style.BLACK, lw=0.6, zorder=1)
        axis.set_xlim(-0.5, len(windows) - 0.5)
        axis.set_xticks(places, [f"{start}–{stop - 1}" for start, stop in windows])
        axis.set_title(MODELS[slug])
        axis.set_ylabel("Change in margin")
        axis.tick_params(axis="x", labelsize=8.5)
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
        style.grid(axis)
    axes[-1].set_xlabel("Layers blocked")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02),
               handlelength=1.5, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"knockout_windows.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
