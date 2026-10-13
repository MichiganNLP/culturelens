"""Appendix figure for norm erasure by depth.

Per model: the share of the stated norm's effect on the decision margin that
erasing the norm's positions after a given layer still removes, for the
contradictory and the true norm. The knockout band is shaded where one exists.

PYTHONPATH=src python -m mechanistic_analysis.erasure_figure \
  --erasure-root output/mechanistic_analysis/erasure \
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
    parser.add_argument("--erasure-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    targets = (("country_opposite_norm", "Contradictory norm", style.BLUE, "-", "o"),
               ("country_norm", "True norm", style.VERMILLION, (0, (4, 1.5)), "s"))
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.2), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        saved = json.loads((args.erasure_root / slug / "metrics.json").read_text(encoding="utf-8"))
        cuts = saved["cuts"]
        if slug in CLAIM_WINDOW:  # end-exclusive constant
            low, high = CLAIM_WINDOW[slug]
            axis.axvspan(low, high - 1, color="#e6e6e6", zorder=0)
        for target, label, colour, dash, marker in targets:
            change = [saved["mean_change"][target][str(cut)] for cut in cuts]
            share = [value / change[0] for value in change]
            axis.plot(cuts, share, color=colour, lw=1.6, ls=dash, marker=marker, ms=3, label=label)
            print(name, label, f"total {change[0]:+.1f}", dict(zip(cuts, [round(value, 2) for value in share])), flush=True)
        axis.axhline(0, color=style.BLACK, lw=0.6, zorder=1)
        axis.set_xlim(cuts[0], cuts[-1])
        axis.set_ylim(-0.08, 1.08)
        axis.set_title(name)
        axis.set_xlabel("Norm erased after layer")
        style.grid(axis)
    axes[0].set_ylabel("Share of effect")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(color="#e6e6e6"))
    labels.append("Norm band")
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"erasure_layers.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
