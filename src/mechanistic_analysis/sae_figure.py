"""Appendix figure for the sparse autoencoder features.

Per model and layer: the held-out, within-country AUROC for resistance under a
contradictory norm from the single most predictive feature, the five most
predictive, all active features, and the full state.

PYTHONPATH=src python -m mechanistic_analysis.sae_figure \
  --analysis-root output/mechanistic_analysis/sae/analysis/unconstrained \
  --output paper/figures/mechanistic_analysis/decision_readout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from mechanistic_analysis import REPORT_LAYER
from mechanistic_analysis.country_figure import MODELS


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.2), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        saved = json.loads((args.analysis_root / slug / "analysis.json").read_text(encoding="utf-8"))["layers"]
        layers = sorted(int(layer) for layer in saved)
        probes = [saved[str(layer)]["probes"] for layer in layers]
        curves = (
            ("Full state", [probe["dense_state"]["auroc"] for probe in probes], style.DARK_PURPLE, "-"),
            ("All active features", [probe["all_features"]["auroc"] for probe in probes], style.BLUE, "-"),
            ("Five features", [probe["top_k"]["5"]["auroc"] for probe in probes], style.VERMILLION, (0, (4, 1.5))),
            ("One feature", [probe["top_k"]["1"]["auroc"] for probe in probes], style.GREEN, (0, (1, 1.2))),
        )
        for label, values, colour, dash in curves:
            axis.plot(layers, values, color=colour, lw=1.6, ls=dash, marker="o", ms=2.4, label=label)
        axis.axvline(REPORT_LAYER[slug], color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
        axis.axhline(0.5, color=style.BLACK, lw=0.6, zorder=1)
        axis.set_xlim(layers[0], layers[-1])
        axis.set_ylim(0.45, 0.95)
        axis.set_title(name)
        axis.set_xlabel("Layer")
        style.grid(axis)
        print(name, {label: [round(value, 3) for value in values] for label, values, _, _ in curves}, flush=True)
    axes[0].set_ylabel("AUROC for resistance")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"sae_layers.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
