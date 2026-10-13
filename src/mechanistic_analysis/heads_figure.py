"""Appendix figure for the head-level knockout.

Per model: the mean change in the decision margin when the answer cannot
attend to the norm through the k highest ranked heads of the norm band, or
through k random heads of the band, against the effect of the whole band.

PYTHONPATH=src python -m mechanistic_analysis.heads_figure \
  --heads-root output/mechanistic_analysis/heads \
  --output paper/figures/mechanistic_analysis/norm_localization
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanistic_analysis import CLAIM_WINDOW
from mechanistic_analysis.country_figure import MODELS

SIZES = (1, 2, 4, 8, 16, 32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--heads-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.ticker  # noqa: F401

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    models = [slug for slug in MODELS if slug in CLAIM_WINDOW]
    fig, axes = plt.subplots(len(models), 1, figsize=(3.75, 4.1))
    places = list(range(len(SIZES)))
    for axis, slug in zip(axes, models):
        saved = json.loads((args.heads_root / slug / "metrics.json").read_text(encoding="utf-8"))
        top = [saved["joint_effect_top"][str(size)] for size in SIZES]
        random = [saved["joint_effect_random"][str(size)] for size in SIZES]
        axis.axhline(saved["window_effect"], color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=1, label="Whole band")
        axis.plot(places, top, color=style.BLUE, lw=1.6, marker="o", ms=3.5, label="Top heads")
        axis.plot(places, random, color=style.VERMILLION, lw=1.6, ls=(0, (4, 1.5)), marker="s", ms=3.5,
                  label="Random heads")
        axis.axhline(0, color=style.BLACK, lw=0.6, zorder=1)
        axis.set_xlim(-0.3, len(SIZES) - 0.7)
        axis.set_xticks(places, [str(size) for size in SIZES])
        axis.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(nbins=4, integer=True))
        axis.set_title(MODELS[slug])
        axis.set_ylabel("Change in margin")
        style.grid(axis)
        print(MODELS[slug], "band", round(saved["window_effect"], 2), "top", [round(v, 2) for v in top],
              "random", [round(v, 2) for v in random], "selected", saved["selection"]["size"], flush=True)
    axes[-1].set_xlabel("Heads blocked")
    handles, labels = axes[0].get_legend_handles_labels()
    order = (1, 2, 0)
    fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.02), handlelength=1.5, columnspacing=1.0, handletextpad=0.4, labelspacing=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.95), h_pad=0.6, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"heads_joint.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
