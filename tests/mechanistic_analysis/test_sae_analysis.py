"""SAE analyses: planted structure is recovered, and label-shuffled data are not mistaken for it."""

from __future__ import annotations

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from mechanistic_analysis import JUDGES, RESIST  # noqa: E402
from mechanistic_analysis.sae import analyze, features  # noqa: E402

from conftest import toy_records  # noqa: E402


def store(matrix: torch.Tensor, condition: str, norm_ids: list[str], d_sae: int) -> features.SparseFeatures:
    sparse = matrix.to_sparse_csr()
    return features.SparseFeatures("qwen-3.5-27b", condition, 0, (len(matrix), d_sae),
                                   sparse.crow_indices(), sparse.col_indices().to(torch.int32),
                                   sparse.values(), tuple(norm_ids), {}, {}, {"mean_l0": 1.0})


def test_dense_columns_keeps_only_features_that_are_used():
    matrix = torch.zeros(200, 50)
    matrix[:, 3] = 1.0                       # always active
    matrix[:2, 7] = 2.0                      # 1% of records
    matrix[0, 9] = 5.0                       # 0.5% of records
    ids, dense = analyze.dense_columns({"a": store(matrix, "a", [f"r{i}" for i in range(200)], 50)}, 0.01)
    assert ids.tolist() == [3, 7] and torch.equal(dense["a"], matrix[:, [3, 7]])


def test_sparse_probe_finds_a_planted_feature_and_scores_only_held_out_records():
    generator = torch.Generator().manual_seed(0)
    n = 600
    y = (torch.rand(n, generator=generator) < 0.4).to(torch.int8)
    x = torch.rand(n, 30, generator=generator)
    x[:, 12] = y.float() * 2 + 0.3 * torch.randn(n, generator=generator)
    folds = torch.arange(n) % 5
    scores, chosen = analyze.sparse_probe(x, y, folds, (1, 5), (1e-2, 1.0), 5)
    assert all(columns == [12] for columns in chosen[1])
    assert analyze._auroc(scores[1], y) > 0.95 and analyze._auroc(scores[5], y) > 0.95
    noise, _ = analyze.sparse_probe(torch.rand(n, 30, generator=generator), y, folds, (5,), (1.0,), 5)
    assert 0.4 < analyze._auroc(noise[5], y) < 0.6                     # selection does not leak


def test_conflict_features_are_found_replicated_and_tied_to_resisting():
    generator = torch.Generator().manual_seed(1)
    n, width = 800, 40
    resist = (torch.rand(n, generator=generator) < 0.4).numpy().astype(np.int8)
    base = lambda: torch.rand(n, width, generator=generator) * 0.1  # noqa: E731
    dense = {name: base() for name in (analyze.TRUE, analyze.FALSE, analyze.TRUE_ALONE, analyze.FALSE_ALONE)}
    # Feature 5 responds to a contradictory norm only when the country is named, and more when resisting.
    dense[analyze.FALSE][:, 5] += 1.0 + torch.tensor(resist).float()
    # Feature 9 responds to a contradictory norm with or without the country: truth, not conflict.
    dense[analyze.FALSE][:, 9] += 1.0
    dense[analyze.FALSE_ALONE][:, 9] += 1.0
    folds = torch.arange(n) % 5
    out = analyze.conflict_features(dense, resist, np.ones(n, bool), folds, 5, np.zeros(n, int),
                                    np.arange(n), draws=50, size=1)
    assert out["truth"]["discoveries_at_fdr_0.01"] == 2 and out["conflict"]["discoveries_at_fdr_0.01"] == 1
    assert out["conflict"]["held_out_sign_agreement"] == 1.0
    assert out["conflict"]["predicts_resisting"]["auroc"] > 0.9
    assert 0.35 < out["conflict"]["random_features_predict_resisting"] < 0.65
    assert "right_country" not in out                                   # those conditions were not given


def test_exclusive_features_and_what_shuffled_labels_produce():
    generator = torch.Generator().manual_seed(2)
    n, n_country, n_value = 1200, 6, 3
    country = torch.arange(n) % n_country
    one_hot = torch.nn.functional.one_hot(country).bool()
    values = torch.zeros(n, n_value, dtype=torch.bool)
    tagged = torch.arange(0, n, 3)                                      # a third of the records have a value
    values[tagged, torch.randint(0, n_value, (len(tagged),), generator=generator)] = True
    active = (torch.rand(n, 60, generator=generator) < 0.03).float()    # background noise
    active[:, 0] = (country == 2).float()                               # exclusive to country 2
    active[:, 1] = values[:, 1].float()                                 # exclusive to value 1
    found = analyze.exclusive(active, one_hot)
    assert found[2, 0] and found[:, 0].sum() == 1
    assert analyze.exclusive(active, values)[1, 1]
    audit = analyze.profile_audit(active, one_hot, values, draws=30, halves=4)
    assert audit["country_exclusive_features"] >= 1 and audit["value_exclusive_features"] >= 1
    # Sparse noise alone already yields "exclusive" features, which is what the audit is for.
    assert audit["country_exclusive_features_when_labels_shuffled"]["mean"] > 0


def test_analysis_end_to_end(tmp_path, capsys):
    slug, d_sae = "qwen-3.5-27b", 64
    records = toy_records(per_country=40)
    for index, record in enumerate(records):
        record["values"] = ["face"] if index % 3 == 0 else (["tradition"] if index % 3 == 1 else [])
    norm_ids = [record["norm_id"] for record in records]
    n = len(records)
    generator = torch.Generator().manual_seed(3)
    resist = (torch.rand(n, generator=generator) < 0.4)
    (tmp_path / "records.json").write_text(json.dumps(records))
    names = [f"{judge}__{RESIST}" for judge in (*JUDGES, "both")]
    np.savez(tmp_path / "outcomes.npz", norm_ids=np.asarray(norm_ids), names=np.asarray(names),
             outcomes=np.stack([resist.numpy().astype(np.int8)] * 3, axis=1))
    folder = tmp_path / "features" / "unconstrained" / slug / "layer_0"
    folder.mkdir(parents=True)
    for condition in (analyze.TRUE, analyze.FALSE, analyze.TRUE_ALONE, analyze.FALSE_ALONE, "country_only"):
        matrix = (torch.rand(n, d_sae, generator=generator) < 0.1).float() * torch.rand(n, d_sae, generator=generator)
        if condition == analyze.FALSE:
            matrix[:, 4] = resist.float() + 0.1                          # the decision, in one feature
        features.save(store(matrix, condition, norm_ids, d_sae), folder / f"{condition}.pt")
    arguments = ["--model", slug, "--features-root", str(tmp_path / "features"), "--records",
                 str(tmp_path / "records.json"), "--outcomes", str(tmp_path / "outcomes.npz"),
                 "--output-root", str(tmp_path / "out"), "--layers", "0", "--draws", "30"]
    assert analyze.main(arguments) == 0
    report = json.loads((tmp_path / "out" / "unconstrained" / slug / "analysis.json").read_text())
    layer = report["layers"]["0"]
    assert layer["probes"]["top_k"]["1"]["auroc"] > 0.95
    assert layer["probes"]["chosen_features"]["1"]["in_every_fold"] == [4]
    assert "dense_state" not in layer["probes"]                          # no activation cache was given
    assert set(layer["conflict"]) == {"truth", "conflict"} and "earlier_profile" in layer
    assert analyze.main(arguments) == 0 and "[layer 0]" not in capsys.readouterr().out.split("saved")[-2][-40:]
