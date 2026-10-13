"""Appendix figure for the probe transfer between true and contradictory norms.

Per model: the within-country AUROC for following the stated norm when the
probe is fitted on the state of one condition (row) and scored on held-out
records of another (column), at the reporting layer. Prints the means over
transfers within and across the two kinds of norm, for the state and for the
bag-of-words probe.

PYTHONPATH=src python -m mechanistic_analysis.transfer_figure \
  --root output/mechanistic_analysis/country_space \
  --output paper/figures/mechanistic_analysis/compliance
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mechanistic_analysis import REPORT_LAYER
from mechanistic_analysis.country_figure import MODELS

CONDITIONS = {"norm_only": "True", "country_norm": "Country + true",
              "opposite_norm_only": "Contra.", "country_opposite_norm": "Country + contra."}
TRUE = (0, 1)
JUDGE = "gemma-4-31b"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    names = list(CONDITIONS)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.5), sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        saved = json.loads((args.root / slug / "metrics.json").read_text(encoding="utf-8"))["follow_probe"][JUDGE]
        tables = {"state": saved["state_transfer"][str(REPORT_LAYER[slug])], "words": saved["text_transfer"]}
        grids = {kind: np.array([[table[a][b] for b in names] for a in names]) for kind, table in tables.items()}
        image = axis.imshow(grids["state"], cmap="Blues", vmin=0.4, vmax=1.0, aspect="auto")
        for row in range(len(names)):
            for column in range(len(names)):
                value = grids["state"][row, column]
                axis.text(column, row, f"{value:.2f}".lstrip("0"), ha="center", va="center", fontsize=10,
                          color="white" if value > 0.78 else style.BLACK)
        axis.axhline(1.5, color="white", lw=2)
        axis.axvline(1.5, color="white", lw=2)
        axis.set_xticks(range(len(names)), ["T", "C+T", "CN", "C+CN"])
        axis.set_yticks(range(len(names)), ["T", "C+T", "CN", "C+CN"])
        axis.tick_params(length=0, top=False, right=False)
        axis.set_title(name)
        axis.set_xlabel("Scored on")
        same = [(a, b) for a in range(4) for b in range(4) if a != b and (a in TRUE) == (b in TRUE)]
        forward = [(a, b) for a in TRUE for b in (2, 3)]
        back = [(a, b) for a in (2, 3) for b in TRUE]
        for kind, grid in grids.items():
            print(name, kind, "diagonal", round(float(np.diag(grid).mean()), 3),
                  "same kind", round(float(np.mean([grid[cell] for cell in same])), 3),
                  "true to contradictory", round(float(np.mean([grid[cell] for cell in forward])), 3),
                  "contradictory to true", round(float(np.mean([grid[cell] for cell in back])), 3),
                  "lowest", round(float(grid.min()), 3), flush=True)
    axes[0].set_ylabel("Fitted on")
    fig.tight_layout(w_pad=0.6, pad=0.3)
    bar = fig.colorbar(image, ax=axes, fraction=0.025, pad=0.015)
    bar.set_label("AUROC")
    bar.outline.set_linewidth(0.6)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"probe_transfer.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
