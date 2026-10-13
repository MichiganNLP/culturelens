"""Country geometry and the follow-the-stated-norm probe, on synthetic states."""

from __future__ import annotations

import numpy as np

from mechanistic_analysis.country_space import (
    NORM_PROMPTS, bag_of_words, cosine_rdm, country_vectors, distinctness, follow_labels,
    shift_size, transfer,
)


def test_geometry_measures():
    rng = np.random.default_rng(0)
    country = np.repeat(np.array(["AAA", "BBB", "CCC", "DDD"]), 50)
    centre = {"AAA": [4, 0, 0], "BBB": [0, 4, 0], "CCC": [0, 3.5, 1], "DDD": [-4, -4, 0]}
    states = np.array([centre[c] for c in country], dtype=float) + 0.1 * rng.normal(size=(200, 3))
    vectors = country_vectors(states, country)
    assert vectors.shape == (4, 3) and np.allclose(vectors.mean(axis=0), 0)
    rdm = cosine_rdm(vectors)
    assert np.allclose(np.diag(rdm), 0, atol=1e-9) and np.allclose(rdm, rdm.T)
    assert rdm[1, 2] < rdm[0, 1] and rdm[1, 2] < rdm[0, 3]           # BBB and CCC are the close pair
    assert distinctness(rdm).shape == (4,) and distinctness(rdm)[0] > distinctness(rdm)[1]
    # Naming AAA changes nothing; naming DDD moves the state a long way.
    base = rng.normal(size=(200, 3)) + 5
    move = np.array([{"AAA": 0.0, "BBB": 0.5, "CCC": 0.5, "DDD": 3.0}[c] for c in country])[:, None]
    sizes = shift_size(base, base + move, country)
    assert sizes[0] < 1e-9 and sizes[3] > sizes[1] > 0


def synthetic(shared: bool, records: int = 600, width: int = 24, seed: int = 1):
    """States whose first coordinate says whether the answer goes with the stated norm
    (``shared``), or whether the answer is the documented one (not ``shared``)."""

    rng = np.random.default_rng(seed)
    country = rng.choice(np.array(["AAA", "BBB", "CCC"]), size=records)
    aligned, states = {}, {}
    for name in NORM_PROMPTS:
        signal = rng.normal(size=records)
        outcome = (signal + 0.3 * rng.normal(size=records) > 0).astype(float)
        states[name] = rng.normal(size=(records, width))
        states[name][:, 0] = 3 * signal
        contradictory = "opposite" in name
        # shared: the signal is "follows"; otherwise the signal is "aligned".
        aligned[name] = (1 - outcome) if (shared and contradictory) else outcome
    return states, aligned, country, rng.permutation(records) % 5


def test_transfer_tells_following_from_giving_the_documented_answer():
    states, aligned, country, fold = synthetic(shared=True)
    follow = follow_labels(aligned)
    assert np.array_equal(follow["norm_only"], aligned["norm_only"])
    assert np.array_equal(follow["country_opposite_norm"], 1 - aligned["country_opposite_norm"])
    out = transfer(lambda name: states[name], follow, country, fold)
    assert out["norm_only"]["norm_only"] > 0.9 and out["norm_only"]["country_opposite_norm"] > 0.9
    # When the state tracks the documented answer instead, the transfer is below chance.
    states, aligned, country, fold = synthetic(shared=False)
    out = transfer(lambda name: states[name], follow_labels(aligned), country, fold)
    assert out["norm_only"]["norm_only"] > 0.9 and out["norm_only"]["opposite_norm_only"] < 0.1


def test_bag_of_words_shares_one_vocabulary():
    texts = {"norm_only": ["Guests remove shoes"] * 3 + ["Guests bow"] * 3,
             "country_norm": ["In Japan guests remove shoes"] * 6}
    out = bag_of_words(texts, minimum=3)
    assert out["norm_only"].shape[1] == out["country_norm"].shape[1] and out["norm_only"].shape[0] == 6
    assert out["country_norm"].sum() > out["norm_only"].sum()
