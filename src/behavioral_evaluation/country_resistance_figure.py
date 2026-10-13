"""Figure: resistance to a contradictory norm against country-only alignment, per country.

One panel per model and one point per country. Both axes are the percentage of
unconstrained answers judged aligned, averaged over the two judges: with the
country alone, and with the country and a contradictory norm. The five
Anglophone countries are marked.

PYTHONPATH=src python -m behavioral_evaluation.country_resistance_figure \
  --judgments-root output/behavioral_evaluation/unconstrained/judgments \
  --output paper/figures/behavioral_evaluation
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MODELS = {"llama-3.1-8b": "Llama-3.1-8B", "gemma-3-12b": "Gemma-3-12B", "qwen-3.5-27b": "Qwen-3.5-27B"}
JUDGES = ("gemma-4-31b", "qwen-3.6-27b")
ANGLOPHONE = ("USA", "CAN", "GBR", "AUS", "NZL")
CONDITIONS = ("country_only", "country_opposite_norm")


def rates(judgments: Path, slug: str) -> dict[str, dict[str, float]]:
    """Per condition and country, the percentage aligned, mean of the judges."""

    per_judge = []
    for judge in JUDGES:
        seen: dict[tuple[str, str], list[float]] = defaultdict(list)
        with (judgments / slug / f"{judge}.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    if row["condition"] in CONDITIONS:
                        seen[(row["condition"], str(row["norm_id"]).split("__")[0])].append(
                            float(row["alignment"] == "aligned"))
        per_judge.append({key: 100 * float(np.mean(values)) for key, values in seen.items()})
    countries = sorted({code for _, code in per_judge[0]})
    return {name: {code: float(np.mean([each[(name, code)] for each in per_judge])) for code in countries}
            for name in CONDITIONS}


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    rank = lambda values: np.argsort(np.argsort(values)).astype(float)  # noqa: E731
    return float(np.corrcoef(rank(a), rank(b))[0, 1])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from mechanistic_analysis import figure_style as style

    style.apply(plt)
    fig, axes = plt.subplots(1, len(MODELS), figsize=(7.8, 2.4), sharex=True, sharey=True)
    for axis, (slug, name) in zip(axes, MODELS.items()):
        found = rates(args.judgments_root, slug)
        countries = list(found["country_only"])
        known = np.array([found["country_only"][code] for code in countries])
        resist = np.array([found["country_opposite_norm"][code] for code in countries])
        marked = np.array([code in ANGLOPHONE for code in countries])
        slope, intercept = np.polyfit(known, resist, 1)
        ends = np.array([known.min(), known.max()])
        axis.plot(ends, slope * ends + intercept, color=style.BLACK, lw=1, zorder=1)
        axis.scatter(known[~marked], resist[~marked], s=14, facecolor=style.BLUE, alpha=0.7, linewidths=0,
                     zorder=2, label="Other countries")
        axis.scatter(known[marked], resist[marked], s=30, marker="^", facecolor=style.VERMILLION,
                     edgecolor=style.BLACK, linewidths=0.4, zorder=3, label="Anglophone countries")
        style.grid(axis)
        axis.set_title(name)
        axis.set_xlabel("Country-only alignment (%)")
        print(slug, "rho", round(spearman(known, resist), 3), "range", round(resist.min(), 1), round(resist.max(), 1),
              flush=True)
    axes[0].set_ylabel("Alignment under a\ncontradictory norm (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03),
               handletextpad=0.3, columnspacing=1.6)
    fig.tight_layout(rect=(0, 0, 1, 0.9), w_pad=0.5, pad=0.3)
    args.output.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(args.output / f"country_resistance.{suffix}", dpi=300, bbox_inches="tight")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
