"""Outcomes, statistics, the decision readout and the country x claim interaction."""

from __future__ import annotations

import json

import numpy as np
import pytest

from mechanistic_analysis import outcomes, stats
from mechanistic_analysis.common import assign_folds, claim_groups

torch = pytest.importorskip("torch")

from mechanistic_analysis import interaction, readout  # noqa: E402
from mechanistic_analysis.activations import SCHEMA_VERSION  # noqa: E402
from mechanistic_analysis.activations.cache import sha256_file  # noqa: E402


def records(n):
    return [{"norm_id": f"{'JPN' if i % 2 else 'BRA'}__a__{i:03d}__001", "norm": f"n{i}",
             "values": [], "options": {"A": "a", "B": "b"},
             "scenario": {"country_context": "This interaction takes place in Japan.",
                          "situation": f"s{i}", "question": "q", "norm_evidence": f"norm {i}",
                          "opposite_norm_evidence": f"opposite {i}", "value_evidence": None}}
            for i in range(n)]


def write(root, condition, tensor, norm_ids, digest):
    path = root / "unconstrained" / "toy" / f"{condition}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"schema_version": SCHEMA_VERSION, "condition": condition,
                "variant": "unconstrained", "model_slug": "toy", "norm_ids": norm_ids,
                "layers": [0, 1], "hidden_size": tensor.shape[2], "dataset_sha256": digest,
                "assistant_start": tensor}, path)


def test_statistics():
    assert stats.auroc(np.array([0.1, 0.2, 0.8, 0.9]), np.array([0, 0, 1, 1])) == 1.0
    assert stats.auroc(np.array([1.0, 1.0]), np.array([0, 1])) == 0.5
    values, country = np.array([1.0, 2, 3, 101, 102, 103]), np.array([0, 0, 0, 1, 1, 1])
    ranks = stats.within_ranks(values, country)
    assert np.allclose(ranks[:3], ranks[3:])
    rng = np.random.default_rng(0)
    y = rng.random(400) < 0.4
    score = y + rng.standard_normal(400)
    out = stats.auroc_with_interval(score, y, draws=200)
    assert out["low"] < out["auroc"] < out["high"] and out["positives"] == int(y.sum())
    rate = stats.rate_with_interval(y, draws=200)
    assert rate["low"] < rate["rate"] < rate["high"]
    paired = stats.paired_difference(np.r_[np.ones(60), np.zeros(40)], np.zeros(100), draws=500)
    assert paired["difference"] == pytest.approx(0.6) and paired["p"] < 0.01
    assert stats.paired_difference(rng.random(200), rng.random(200), draws=500)["p"] > 0.01
    # Resampling whole groups: an interval exists and brackets the estimate.
    groups = np.arange(400) // 4
    grouped = stats.auroc_with_interval(score, y, groups=groups, draws=200)
    assert grouped["low"] < grouped["auroc"] < grouped["high"]


def test_folds_keep_shared_claims_together():
    rows = records(40)
    rows[3]["scenario"]["norm_evidence"] = rows[7]["scenario"]["opposite_norm_evidence"]
    groups = claim_groups(rows)
    assert groups[3] == groups[7]
    folds = assign_folds(groups, 5, 1)
    assert folds[3] == folds[7] and set(folds) == set(range(5))


def test_outcome_table_keeps_judges_apart_and_marks_missing(tmp_path):
    ids = ["JPN__a__001__001", "JPN__a__002__001", "JPN__a__003__001"]

    def judge(name, labels):
        path = tmp_path / f"{name}.jsonl"
        path.write_text("\n".join(json.dumps(
            {"norm_id": i, "condition": "country_opposite_norm", "alignment": a})
            for i, a in zip(ids, labels) if a))
        return path

    names, table = outcomes.build(ids, {
        "one": judge("one", ["aligned", "aligned", "misaligned"]),
        "two": judge("two", ["aligned", "partially_aligned", None])})
    column = dict(zip(names, table.T))
    assert column["one__country_opposite_norm"].tolist() == [1, 1, 0]
    assert column["two__country_opposite_norm"].tolist() == [1, 0, -1]
    assert column["both__country_opposite_norm"].tolist() == [1, 0, -1]
    assert column["one__base"].tolist() == [-1, -1, -1]
    np.savez(tmp_path / "t.npz", norm_ids=np.asarray(ids), names=np.asarray(names), outcomes=table)
    subset = outcomes.load(tmp_path / "t.npz", [ids[2], ids[0]])
    assert subset["one__country_opposite_norm"].tolist() == [0, 1]
    with pytest.raises(ValueError):
        outcomes.load(tmp_path / "t.npz", ["absent"])


def test_readout_finds_a_planted_outcome_only_where_it_is(tmp_path):
    generator = torch.Generator().manual_seed(2)
    n, d = 300, 12
    rows = records(n)
    path = tmp_path / "records.json"
    path.write_text(json.dumps(rows))
    digest, ids = sha256_file(path), [row["norm_id"] for row in rows]
    resists = torch.rand(n, generator=generator) > 0.5
    for condition, informative in (("country_only", False), ("country_opposite_norm", True)):
        tensor = torch.randn(n, 2, d, generator=generator)
        if informative:
            tensor[:, 1, 2] += torch.where(resists, 2.0, -2.0)
        write(tmp_path / "activations", condition, tensor, ids, digest)
    table = tmp_path / "outcomes.npz"
    missing = resists.numpy().astype(np.int8).copy()
    missing[:3] = -1
    np.savez(table, norm_ids=np.asarray(ids),
             names=np.asarray(["j__country_opposite_norm", "j__with_gaps", "j__never"]),
             outcomes=np.stack([resists.numpy().astype(np.int8), missing, np.zeros(n, np.int8)], axis=1))
    assert readout.main([
        "--activations-root", str(tmp_path / "activations"), "--model-slug", "toy",
        "--records", str(path), "--outcomes", str(table), "--output-root", str(tmp_path / "out"),
        "--states", "country_only", "country_opposite_norm", "--device", "cpu", "--text",
        "--min-count", "1"]) == 0
    saved = np.load(tmp_path / "out" / "toy" / "scores.npz")
    assert saved["outcomes"].tolist() == ["j__country_opposite_norm", "j__with_gaps"]   # constant one dropped
    assert saved["score__country_opposite_norm"].shape == (2, 2, n)
    y = resists.numpy()
    score = lambda state, layer: saved[f"score__{state}"].astype(float)[layer, 0]  # noqa: E731
    assert stats.auroc(score("country_opposite_norm", 1), y) > 0.9
    assert 0.35 < stats.auroc(score("country_opposite_norm", 0), y) < 0.65
    assert 0.35 < stats.auroc(score("country_only", 1), y) < 0.65
    assert saved["score__text:country_only"].shape == (1, 2, n)


def test_interaction_is_zero_for_additive_states_and_found_when_planted(tmp_path):
    generator = torch.Generator().manual_seed(1)
    n, d = 200, 16
    rows = records(n)
    path = tmp_path / "records.json"
    path.write_text(json.dumps(rows))
    digest, ids = sha256_file(path), [row["norm_id"] for row in rows]

    def draw():
        return torch.randn(n, 2, d, generator=generator)

    base, country, true, false = draw(), draw(), draw(), draw()
    planted = torch.zeros(n, 2, d)
    planted[:, 1, 5] = 4.0       # layer 1: a false claim under its country adds one direction
    for name, tensor in {
        "base": base, "country_only": base + country, "norm_only": base + true,
        "opposite_norm_only": base + false, "country_norm": base + country + true,
        "country_opposite_norm": base + country + false + planted,
    }.items():
        write(tmp_path / "activations", name, tensor, ids, digest)
    assert interaction.main([
        "--activations-root", str(tmp_path / "activations"), "--model-slug", "toy",
        "--records", str(path), "--output-root", str(tmp_path / "out"), "--device", "cpu",
        "--features-step", "1"]) == 0
    layers = json.loads((tmp_path / "out" / "toy" / "metrics.json").read_text())["layers"]
    assert layers[0]["additive_share_false"] == pytest.approx(1.0, abs=1e-5)
    assert layers[1]["interaction_false"] == pytest.approx(4.0, abs=1e-3)
    assert layers[1]["additive_share_true"] == pytest.approx(1.0, abs=1e-5)
    assert layers[1]["additive_share_false"] < 0.9
    assert layers[0]["country_given_false_cosine"] == pytest.approx(1.0, abs=1e-4)
    features = np.load(tmp_path / "out" / "toy" / "features.npz")
    assert features["conflict"].shape == (2, n)
    assert np.allclose(features["conflict"][1], 4.0, atol=1e-3)
