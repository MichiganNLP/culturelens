"""Appendix figure for the internal changes after fine-tuning.

Per model: the change in the decision margin when a contradictory norm is
erased after a given layer, for the untuned model, SFT and SFT + CF, on the
same test records. Prints the numbers of the accompanying table.

PYTHONPATH=src python -m finetune.route_figure \
  --root output/finetune/mechanism --output paper/figures/finetuning
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanistic_analysis import CLAIM_WINDOW
from mechanistic_analysis.country_figure import MODELS

RUNS = {  # model slug -> (untuned, SFT, SFT + CF) folders
    "llama-3.1-8b": ("base", "collab-sft-country", "collab-sft-cf"),
    "gemma-3-12b": ("base-gemma-3-12b", "collab-gemma-sft-country", "collab-gemma-sft-cf"),
    "qwen-3.5-27b": ("base-qwen-3.5-27b", "collab-qwen-sft-country", "collab-qwen-sft-cf"),
}
LABELS = ("Untuned", "SFT", "SFT + CF")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    looks = ((style.GREY, (0, (1, 1.2)), "^"), (style.VERMILLION, (0, (4, 1.5)), "s"), (style.BLUE, "-", "o"))
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.2))
    for axis, (slug, name) in zip(axes, MODELS.items()):
        if slug in CLAIM_WINDOW:  # end-exclusive constant
            low, high = CLAIM_WINDOW[slug]
            axis.axvspan(low, high - 1, color="#e6e6e6", zorder=0)
        for run, label, (colour, dash, marker) in zip(RUNS[slug], LABELS, looks):
            saved = json.loads((args.root / run / "erasure" / slug / "metrics.json").read_text(encoding="utf-8"))
            cuts = saved["cuts"]
            contra = [saved["mean_change"]["country_opposite_norm"][str(cut)] for cut in cuts]
            true = saved["mean_change"]["country_norm"][str(cuts[0])]
            axis.plot(cuts, contra, color=colour, lw=1.6, ls=dash, marker=marker, ms=3, label=label)
            line = f"{name} {label}: erase contra {contra[0]:.1f} true {true:.1f} ratio {contra[0] / abs(true):.2f}"
            path = args.root / run / "localize" / slug / "metrics.json"
            if path.is_file():
                change = json.loads(path.read_text(encoding="utf-8"))["mean_change"]["country_opposite_norm__claim"]
                low, high = CLAIM_WINDOW[slug]
                whole = max((int(key.split("-")[1]), value) for key, value in change.items() if key.startswith("0-"))[1]
                line += f" | band {change[f'{low}-{high}']:.1f} all layers {whole:.1f} share {change[f'{low}-{high}'] / whole:.0%}"
            print(line, flush=True)
        axis.axhline(0, color=style.BLACK, lw=0.6, zorder=1)
        axis.set_xlim(cuts[0], cuts[-1])
        axis.set_ylim(bottom=-0.8)
        axis.set_title(name)
        axis.set_xlabel("Norm erased after layer")
        style.grid(axis)
    axes[0].set_ylabel("Change in margin")
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Patch(color="#e6e6e6"))
    labels.append("Norm band")
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.8, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"route_erasure.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
