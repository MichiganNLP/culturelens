"""Shared look of the paper's figures: serif type, boxed axes, a colour-blind-safe palette."""

from __future__ import annotations

from typing import Any

# Okabe-Ito palette: distinguishable under the common forms of colour blindness.
BLUE, VERMILLION, GREEN, PURPLE, YELLOW, SKY, BLACK, GREY = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000", "#8c8c8c")
DARK_PURPLE, MAGENTA = "#3B1F6B", "#E7298A"
GRID = "#d9d9d9"


def apply(plt: Any, size: float = 11) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "font.size": size, "axes.titlesize": size + 1, "axes.labelsize": size,
        "xtick.labelsize": size - 1, "ytick.labelsize": size - 1, "legend.fontsize": size - 0.5,
        "axes.edgecolor": BLACK, "axes.labelcolor": BLACK, "xtick.color": BLACK, "ytick.color": BLACK,
        "text.color": BLACK, "axes.linewidth": 0.7,
        "axes.spines.top": True, "axes.spines.right": True,
        "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 3, "ytick.major.size": 3,
        "figure.facecolor": "white", "axes.facecolor": "white", "pdf.fonttype": 42, "ps.fonttype": 42})


def grid(axis: Any) -> None:
    axis.grid(True, color=GRID, lw=0.5, ls=(0, (2, 2)))
    axis.set_axisbelow(True)
