"""SAE features: encoding rules, sparse storage, and one end-to-end extraction."""

from __future__ import annotations

import json

import pytest

torch = pytest.importorskip("torch")

from mechanistic_analysis.activations import SCHEMA_VERSION  # noqa: E402
from mechanistic_analysis.sae import extract, features  # noqa: E402
from mechanistic_analysis.sae.backend import MatrixSAE, SAEMetadata  # noqa: E402
from mechanistic_analysis.sae.registry import get_spec, resolve_layers  # noqa: E402

D_MODEL, D_SAE = 8, 32


def make_sae(activation: str, *, k: int | None = None, threshold: float | None = None) -> MatrixSAE:
    generator = torch.Generator().manual_seed(0)
    w = torch.randn(D_SAE, D_MODEL, generator=generator)
    metadata = SAEMetadata("test", "local", "x", "model", 0, D_MODEL, D_SAE, activation, "float32", k=k)
    return MatrixSAE(metadata, w, torch.zeros(D_SAE), w.T.contiguous() / D_SAE, torch.zeros(D_MODEL),
                     threshold=threshold, k=k)


def test_topk_keeps_exactly_k_features_and_jumprelu_applies_its_threshold():
    states = torch.randn(20, D_MODEL, generator=torch.Generator().manual_seed(1))
    top = make_sae("topk", k=4).encode(states)
    assert (top.ne(0).sum(dim=1) == 4).all()
    pre = states @ make_sae("topk", k=4).w_enc.T
    assert torch.equal(top.max(dim=1).values, pre.max(dim=1).values)      # the largest survive
    jump = make_sae("jumprelu", threshold=1.5).encode(states)
    assert torch.equal(jump.ne(0), pre > 1.5) and torch.equal(jump[jump.ne(0)], pre[pre > 1.5])


def test_csr_storage_reproduces_the_dense_features(tmp_path):
    sae = make_sae("topk", k=4)
    states = torch.randn(50, D_MODEL, generator=torch.Generator().manual_seed(2)).to(torch.bfloat16)
    row_ptr, ids, values, quality = features.encode_states(states, sae, batch_size=16)
    stored = features.SparseFeatures("m", "base", 0, (50, D_SAE), row_ptr, ids, values,
                                     tuple(f"r{i}" for i in range(50)), sae.metadata.to_dict(), {}, quality)
    features.save(stored, tmp_path / "f.pt")
    loaded = features.load(tmp_path / "f.pt")
    dense = loaded.to_dense()
    assert torch.allclose(dense, sae.encode(states).float(), atol=1e-5)
    assert quality["mean_l0"] == 4.0 and 0 < quality["features_ever_active"] <= D_SAE
    # Batching does not change the result.
    again = features.encode_states(states, sae, batch_size=50)
    assert torch.equal(again[0], row_ptr) and torch.equal(again[1], ids)
    with pytest.raises(ValueError, match="row_ptr"):
        features.validate(features.SparseFeatures("m", "base", 0, (49, D_SAE), row_ptr, ids, values,
                                                  tuple(f"r{i}" for i in range(49)), {}, {}, {}))


def test_a_perfect_autoencoder_leaves_no_variance_unexplained():
    eye = torch.eye(D_MODEL)
    metadata = SAEMetadata("test", "local", "x", "model", 0, D_MODEL, D_MODEL, "jumprelu", "float32")
    sae = MatrixSAE(metadata, eye, torch.zeros(D_MODEL), eye, torch.zeros(D_MODEL), threshold=-1e9)
    states = torch.randn(30, D_MODEL, generator=torch.Generator().manual_seed(3))
    quality = features.encode_states(states, sae, batch_size=8)[3]
    assert quality["fraction_of_variance_unexplained"] < 1e-10 and quality["cosine"] > 0.999999


def test_registry_layers():
    assert resolve_layers(get_spec("llama-3.1-8b"), "all") == (3, 7, 11, 15, 19, 23, 27)
    assert resolve_layers(get_spec("qwen-3.5-27b"), "5,3") == (3, 5)
    with pytest.raises(ValueError, match="no checkpoint"):
        resolve_layers(get_spec("llama-3.1-8b"), "4")


def test_extraction_end_to_end_and_resumable(tmp_path, capsys):
    slug, layers, records = "qwen-3.5-27b", [0, 1, 2], 12
    generator = torch.Generator().manual_seed(4)
    sae_root = tmp_path / "sae"
    sae_root.mkdir()
    (sae_root / "config.json").write_text(json.dumps(
        {"hook_point": "resid_post", "layers": list(range(64)), "d_model": D_MODEL, "d_sae": D_SAE, "k": 3}))
    for layer in (0, 2):
        torch.save({"W_enc": torch.randn(D_SAE, D_MODEL, generator=generator), "b_enc": torch.zeros(D_SAE),
                    "W_dec": torch.randn(D_MODEL, D_SAE, generator=generator), "b_dec": torch.zeros(D_MODEL)},
                   sae_root / f"layer{layer}.sae.pt")
    cache_root = tmp_path / "activations"
    (cache_root / "unconstrained" / slug).mkdir(parents=True)
    for condition in ("base", "country_only"):
        torch.save({"schema_version": SCHEMA_VERSION, "condition": condition, "variant": "unconstrained",
                    "model_slug": slug, "norm_ids": [f"JPN__a__{i:03d}__001" for i in range(records)],
                    "layers": layers, "hidden_size": D_MODEL, "dataset_sha256": "abc",
                    "assistant_start": torch.randn(records, len(layers), D_MODEL,
                                                   generator=generator).to(torch.bfloat16)},
                   cache_root / "unconstrained" / slug / f"{condition}.pt")
    arguments = ["--model", slug, "--activation-root", str(cache_root), "--output-root",
                 str(tmp_path / "out"), "--sae-root", str(sae_root), "--layers", "0,2",
                 "--device", "cpu", "--sae-dtype", "float32", "--batch-size", "5"]
    assert extract.main(arguments) == 0
    folder = tmp_path / "out" / "unconstrained" / slug
    stored = features.load(folder / "layer_2" / "country_only.pt")
    assert stored.shape == (records, D_SAE) and stored.nnz == records * 3
    assert stored.layer == 2 and stored.norm_ids[0] == "JPN__a__000__001"
    assert stored.cache["dataset_sha256"] == "abc" and stored.sae_metadata["k"] == 3
    # The features are those of the cached state at that layer.
    cache = torch.load(cache_root / "unconstrained" / slug / "country_only.pt", weights_only=False)
    weights = torch.load(sae_root / "layer2.sae.pt", weights_only=False)
    pre = cache["assistant_start"][:, 2, :].float() @ weights["W_enc"].T
    dense = stored.to_dense()
    assert torch.allclose(dense.max(dim=1).values, pre.max(dim=1).values, atol=1e-4)
    summary = json.loads((folder / "metrics.json").read_text())
    assert set(summary["quality"]) == {"0", "2"} and summary["quality"]["0"]["base"]["mean_l0"] == 3.0
    assert "wrote 4, kept 0" in capsys.readouterr().out
    assert extract.main(arguments) == 0                                   # nothing is redone
    assert "wrote 0, kept 4" in capsys.readouterr().out
    with pytest.raises(ValueError, match="no layers"):
        extract.main(arguments[:-8] + ["--layers", "5", "--device", "cpu", "--sae-dtype", "float32",
                                       "--batch-size", "5"])
