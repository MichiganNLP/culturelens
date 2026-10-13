"""Appendix figure for the conflict signal: how alike the model's true-norm and
contradictory-norm interactions are, by layer.

Reads the report tables (``report.py``): per model, the mean cosine between a
record's true-norm and contradictory-norm interaction at every layer, and the
AUROC of the conflict signal for resisting. Prints the numbers of the summary
table.

PYTHONPATH=src python -m mechanistic_analysis.interaction_figure \
  --tables output/mechanistic_analysis/report/tables.json \
  --output paper/figures/mechanistic_analysis/decision_readout
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MODELS = {"llama-3.1-8b": "Llama-3.1-8B", "gemma-3-12b": "Gemma-3-12B", "qwen-3.5-27b": "Qwen-3.5-27B"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tables", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    report = json.loads(args.tables.read_text(encoding="utf-8"))["models"]
    style.apply(plt)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.2), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        found = report[slug]["interaction"]
        cosine, signal = found["cosine_curve"], found["conflict_curve"]
        axis.plot(cosine["layers"], cosine["values"], color=style.BLUE, lw=1.6,
                  label="Similarity of the two interactions")
        axis.plot(signal["layers"], signal["values"], color=style.GREEN, lw=2.0, ls=(0, (1, 1.1)),
                  label="Conflict signal, AUROC for resistance")
        axis.axvline(found["layer"], color="#404040", lw=1.2, ls=(0, (5, 2.5)), zorder=1)
        axis.axhline(0.5, color=style.BLACK, lw=0.6, zorder=1)
        axis.set_xlim(cosine["layers"][0], cosine["layers"][-1])
        axis.set_ylim(0.4, 1.02)
        axis.set_title(name)
        axis.set_xlabel("Layer")
        style.grid(axis)
        values = signal["values"]
        best = max(range(len(values)), key=values.__getitem__)
        low = min(range(len(cosine["values"])), key=cosine["values"].__getitem__)
        print(f"{name}: layer {found['layer']} | cosine early {found['interaction_cosine_early']:.2f} at layer "
              f"{found['interaction_cosine_at_layer']:.2f}, lowest {cosine['values'][low]:.2f} at layer "
              f"{cosine['layers'][low]}, last {cosine['values'][-1]:.2f} | additive share "
              f"{found['additive_share_false']:.2f} | conflict " + json.dumps(found["conflict"])
              + f" | conflict highest {values[best]:.3f} at layer {signal['layers'][best]}, first {values[0]:.3f}",
              flush=True)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handlelength=1.8, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"interaction_layers.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
