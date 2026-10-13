"""Steering one SAE feature: the edit does what it says, where it says, during generation too."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from mechanistic_analysis.interventions.features import steered  # noqa: E402
from mechanistic_analysis.sae.backend import MatrixSAE, SAEMetadata  # noqa: E402


def rotation_sae(width: int) -> MatrixSAE:
    """An SAE that is an exact change of basis, so a feature's value can be checked after an edit."""

    q, _ = torch.linalg.qr(torch.randn(width, width, generator=torch.Generator().manual_seed(0)))
    metadata = SAEMetadata("test", "local", "x", "model", 1, width, width, "jumprelu", "float32")
    return MatrixSAE(metadata, q.contiguous(), torch.zeros(width), q.T.contiguous(), torch.zeros(width),
                     threshold=-1e9)


def states(tiny, ids, layer):
    seen = {}

    def record(_module, _inputs, output) -> None:          # must return None: a value would replace the output
        seen["x"] = (output[0] if isinstance(output, tuple) else output).detach().clone()

    handle = tiny.decoder_layers[layer].register_forward_hook(record)
    try:
        with torch.no_grad():
            tiny.model(input_ids=ids)
    finally:
        handle.remove()
    return seen["x"]


def test_ablate_and_clamp_set_the_feature_from_the_start_position_only(tiny):
    sae, layer, feature = rotation_sae(32), 1, 7
    ids = torch.randint(5, 40, (2, 9), generator=torch.Generator().manual_seed(1))
    plain = states(tiny, ids, layer)
    start = torch.tensor([4, 6])
    for mode, value, wanted in (("ablate", 0.0, 0.0), ("clamp", 3.5, 3.5)):
        with steered(tiny.decoder_layers, layer, sae, feature, mode, value, start, ids.shape[1]):
            edited = states(tiny, ids, layer)        # registered after, so it sees the edited output
        activation = sae.encode(edited)[..., feature]
        for row in range(2):
            assert torch.allclose(activation[row, start[row]:], torch.full((9 - int(start[row]),), wanted),
                                  atol=1e-4)
            assert torch.equal(edited[row, :start[row]], plain[row, :start[row]])   # earlier positions untouched
        # Every other feature keeps its value at the edited positions.
        others = torch.ones(32, dtype=torch.bool)
        others[feature] = False
        assert torch.allclose(sae.encode(edited)[1, 6:, others], sae.encode(plain)[1, 6:, others], atol=1e-4)


def test_steering_changes_what_is_generated_and_leaves_no_hook_behind(tiny):
    sae, layer = rotation_sae(32), 1
    ids = torch.randint(5, 40, (2, 8), generator=torch.Generator().manual_seed(2))
    call = dict(input_ids=ids, max_new_tokens=6, do_sample=False, pad_token_id=0)
    with torch.no_grad():
        plain = tiny.model.generate(**call)
        with steered(tiny.decoder_layers, layer, sae, 3, "clamp", 40.0, torch.tensor([5, 5]), ids.shape[1]):
            pushed = tiny.model.generate(**call)
        again = tiny.model.generate(**call)
    assert torch.equal(plain, again)                  # the hook is gone
    assert not torch.equal(plain, pushed)             # new tokens were steered as well
    with pytest.raises(ValueError, match="Unknown mode"):
        with steered(tiny.decoder_layers, layer, sae, 3, "boost", 1.0, torch.tensor([0, 0]), 8):
            pass


def test_reading_and_steering_end_to_end(tmp_path, tiny, monkeypatch):
    import json

    from conftest import toy_records
    from mechanistic_analysis import ERASE_LAYERS
    from mechanistic_analysis.interventions import generate
    from mechanistic_analysis.sae import read

    slug, width, d_sae = "qwen-3.5-27b", 32, 64
    generator = torch.Generator().manual_seed(5)
    sae_root = tmp_path / "sae"
    sae_root.mkdir()
    (sae_root / "config.json").write_text(json.dumps(
        {"hook_point": "resid_post", "layers": list(range(64)), "d_model": width, "d_sae": d_sae, "k": 4}))
    for layer in (1, 2):
        torch.save({"W_enc": torch.randn(d_sae, width, generator=generator), "b_enc": torch.zeros(d_sae),
                    "W_dec": torch.randn(width, d_sae, generator=generator), "b_dec": torch.zeros(width)},
                   sae_root / f"layer{layer}.sae.pt")
    records = tmp_path / "records.json"
    records.write_text(json.dumps(toy_records()))
    for module in (generate, read):
        monkeypatch.setattr(module, "load", lambda args: tiny)
    monkeypatch.setitem(ERASE_LAYERS, slug, (0, 1, 3))
    common = ["--model", "toy", "--model-slug", slug, "--records", str(records), "--sample", "6",
              "--batch-size", "4", "--sae-root", str(sae_root)]

    assert read.main(common + ["--output-root", str(tmp_path / "readings"), "--features", "1:3", "1:9",
                               "2:5", "--erase-layers", "0"]) == 0
    readings = json.loads((tmp_path / "readings" / slug / "readings.json").read_text())
    assert set(readings["variants"]) == {*read.BENCHMARK, *read.FRAMINGS, "erase_after_0"}
    values = readings["variants"]["country_opposite_norm_guide"]
    assert set(values) == {"1", "2"} and set(values["1"]) == {"3", "9"} and len(values["2"]["5"]) == 6
    # Erasing the norm from the first block changes what is read downstream.
    plain, erased = readings["variants"]["country_opposite_norm"], readings["variants"]["erase_after_0"]
    assert plain != erased

    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"arms": {
        "false-x-off": {"condition": "country_opposite_norm", "layer": 1, "feature": 3, "mode": "ablate"},
        "false-x-on": {"condition": "country_opposite_norm", "layer": 2, "feature": 5, "mode": "clamp",
                       "value": 50.0}}}))
    out = tmp_path / "answers"
    assert generate.main(common + ["--output-root", str(out), "--max-new-tokens", "5", "--arms", "false",
                                   "false-x-off", "false-x-on", "--feature-plan", str(plan)]) == 0
    rows = {arm: [json.loads(line) for line in (out / slug / f"{arm}.jsonl").read_text().splitlines()]
            for arm in ("false", "false-x-off", "false-x-on")}
    assert all(len(found) == 6 for found in rows.values())
    assert [r["response"] for r in rows["false-x-on"]] != [r["response"] for r in rows["false"]]
    manifest = json.loads((out / slug / "metrics.json").read_text())
    assert manifest["feature_arms"]["false-x-on"]["value"] == 50.0
    generate.ARMS.pop("false-x-off"), generate.ARMS.pop("false-x-on")
