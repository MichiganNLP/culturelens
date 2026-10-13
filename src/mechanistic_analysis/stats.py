"""Small statistics used across the study. NumPy only."""

from __future__ import annotations

import numpy as np


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(1, len(values) + 1)
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.bincount(inverse, weights=ranks) / counts)[inverse]


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """P(a positive outscores a negative), ties counted a half."""

    positive = np.asarray(labels).astype(bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = average_ranks(np.asarray(scores, dtype=np.float64))
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def within_ranks(values: np.ndarray, strata: np.ndarray) -> np.ndarray:
    """Each value's rank inside its stratum, scaled to (0, 1). Ranking within
    country removes any difference between countries from a comparison."""

    out = np.empty(len(values))
    for stratum in np.unique(strata):
        rows = np.flatnonzero(strata == stratum)
        out[rows] = (average_ranks(np.asarray(values, dtype=np.float64)[rows]) - 0.5) / len(rows)
    return out


def bootstrap(statistic, n: int, *, draws: int = 2000, seed: int = 0,
              groups: np.ndarray | None = None) -> tuple[float, float]:
    """95% percentile interval of ``statistic(indices)``.

    Resamples records, or whole ``groups`` (e.g. records sharing a claim) when
    given, so dependent records are drawn together.
    """

    generator = np.random.default_rng(seed)
    values = []
    if groups is None:
        for _ in range(draws):
            values.append(statistic(generator.integers(0, n, n)))
    else:
        members: dict[int, list[int]] = {}
        for index, group in enumerate(np.asarray(groups).tolist()):
            members.setdefault(group, []).append(index)
        keys = list(members)
        for _ in range(draws):
            drawn = generator.integers(0, len(keys), len(keys))
            values.append(statistic(np.concatenate([members[keys[k]] for k in drawn])))
    values = np.asarray(values, dtype=np.float64)
    values = values[~np.isnan(values)]
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def auroc_with_interval(scores: np.ndarray, labels: np.ndarray, strata: np.ndarray | None = None,
                        *, groups: np.ndarray | None = None, draws: int = 1000,
                        seed: int = 0) -> dict[str, float]:
    """AUROC (within strata when given) and its bootstrap interval."""

    scores, labels = np.asarray(scores, dtype=np.float64), np.asarray(labels).astype(bool)
    ranked = within_ranks(scores, strata) if strata is not None else scores
    low, high = bootstrap(lambda rows: auroc(ranked[rows], labels[rows]), len(labels),
                          draws=draws, seed=seed, groups=groups)
    return {"auroc": auroc(ranked, labels), "low": low, "high": high, "n": int(len(labels)),
            "positives": int(labels.sum())}


def rate_with_interval(values: np.ndarray, *, draws: int = 2000, seed: int = 0) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    low, high = bootstrap(lambda rows: values[rows].mean(), len(values), draws=draws, seed=seed)
    return {"rate": float(values.mean()), "low": low, "high": high, "n": int(len(values))}


def paired_difference(after: np.ndarray, before: np.ndarray, *, draws: int = 2000,
                      seed: int = 0) -> dict[str, float]:
    """Mean of ``after - before`` on the same records, with a bootstrap interval
    and the two-sided sign-flip p-value for the mean being zero."""

    delta = np.asarray(after, dtype=np.float64) - np.asarray(before, dtype=np.float64)
    low, high = bootstrap(lambda rows: delta[rows].mean(), len(delta), draws=draws, seed=seed)
    generator = np.random.default_rng(seed + 1)
    observed = abs(delta.mean())
    flips = generator.choice([-1.0, 1.0], size=(draws, len(delta)))
    null = np.abs((flips * delta).mean(axis=1))
    return {"difference": float(delta.mean()), "low": low, "high": high,
            "p": float((1 + (null >= observed - 1e-12).sum()) / (1 + draws)), "n": int(len(delta))}
