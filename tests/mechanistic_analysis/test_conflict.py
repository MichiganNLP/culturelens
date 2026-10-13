"""The label-free conflict test: interactions and their held-out separability."""

from __future__ import annotations

import numpy as np

from mechanistic_analysis.conflict import interactions, separability, summarize


def states(signal: float, records: int = 200, layers: int = 2, width: int = 16, seed: int = 0):
    rng = np.random.default_rng(seed)
    shape = (records, layers, width)
    bare, country = rng.normal(size=shape), rng.normal(size=shape)
    claim_true, claim_false = rng.normal(size=shape), rng.normal(size=shape)
    direction = np.zeros(width); direction[0] = 1.0
    noise = lambda: 0.5 * rng.normal(size=shape)  # noqa: E731
    return {"base": bare, "country_only": bare + country,
            "norm_only": bare + claim_true, "opposite_norm_only": bare + claim_false,
            "country_norm": bare + country + claim_true + noise(),
            "country_opposite_norm": bare + country + claim_false + noise() + signal * direction}


def test_interaction_removes_what_each_piece_of_evidence_adds_alone():
    true, false, country = interactions(states(0.0))
    assert true.shape == false.shape == country.shape == (200, 2, 16)
    # Additive parts cancel exactly: what is left is the noise, far smaller than the country effect.
    assert np.abs(true).mean() < 0.6 * np.abs(country).mean()


def test_separability_is_chance_without_a_conflict_signal_and_high_with_one():
    true, false, _ = interactions(states(0.0))
    area, paired = separability(true[:, 0], false[:, 0], folds=5, seed=1)
    assert 0.4 < area < 0.6 and 0.4 < paired < 0.6
    true, false, _ = interactions(states(3.0))
    area, paired = separability(true[:, 0], false[:, 0], folds=5, seed=1)
    assert area > 0.95 and paired > 0.95
    rows = summarize(states(3.0), folds=5, seed=1)
    assert [row["layer"] for row in rows] == [0, 1] and all(row["auroc"] > 0.95 for row in rows)
