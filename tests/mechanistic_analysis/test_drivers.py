"""Behavioural drivers: planted cue effects, predictors and stance tables are recovered."""

from __future__ import annotations

import json

import numpy as np

from mechanistic_analysis import JUDGES, RESIST, drivers

CONDITIONS = ("base", "country_only", "norm_only", "country_norm", RESIST)


def build_tree(tmp_path, n=600):
    rng = np.random.default_rng(0)
    records = [{"norm_id": f"{('JPN', 'BRA', 'KEN')[i % 3]}__{('etiquette', 'family')[i % 2]}__{i:03d}__001",
                "values": ["face"] if i % 4 == 0 else []} for i in range(n)]
    (tmp_path / "records.json").write_text(json.dumps(records))
    knows = rng.random(n) < 0.6
    for shift, slug in enumerate(("llama-3.1-8b", "gemma-3-12b")):
        truth = {"base": rng.random(n) < 0.4, "country_only": knows,
                 "norm_only": rng.random(n) < 0.8, "country_norm": rng.random(n) < 0.95,
                 # resisting needs the country-only answer to be right; same records in both models
                 RESIST: knows & (np.arange(n) % (2 + shift) == 0)}
        for judge in JUDGES:
            folder = tmp_path / "behavioural" / "judgments" / slug
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{judge}.jsonl").write_text("\n".join(json.dumps(
                {"norm_id": records[i]["norm_id"], "condition": condition,
                 "alignment": "aligned" if truth[condition][i] else "misaligned"})
                for condition in CONDITIONS for i in range(n)))
        folder = tmp_path / "behavioural" / "stance" / slug
        folder.mkdir(parents=True)
        (folder / f"{JUDGES[0]}.jsonl").write_text("\n".join(json.dumps(
            {"norm_id": records[i]["norm_id"], "condition": RESIST,
             "stance": "challenges" if truth[RESIST][i] else "follows",
             "restates_claim": bool(not truth[RESIST][i]), "attributes_to_country": True})
            for i in range(n)))
    return knows


def test_drivers_recover_planted_structure(tmp_path):
    knows = build_tree(tmp_path)
    built = drivers.build(tmp_path / "records.json", tmp_path / "behavioural", tmp_path / "root", draws=100)
    assert list(built["models"]) == ["llama-3.1-8b", "gemma-3-12b"]
    llama = built["models"]["llama-3.1-8b"]
    country = llama["effects"]["Country, nothing else stated"]
    assert abs(country["difference"] - (knows.mean() - country["without"])) < 1e-9
    assert country["low"] < country["difference"] < country["high"]
    assert llama["effects"]["False norm, country stated"]["difference"] < -0.2
    assert "Country, false norm stated" not in llama["effects"]            # no claim-only run
    ratios = llama["resisting_model"]["odds_ratio"]
    assert ratios["Right with the country alone"]["or"] > 20              # resisting requires it
    assert 0.5 < ratios["Right with the true norm alone"]["or"] < 2.0
    assert built["cross_model"]["pairwise_kappa"]["llama-3.1-8b / gemma-3-12b"]["kappa"] > 0.2
    assert abs(sum(built["cross_model"]["if_independent"].values()) - 1) < 1e-9
    stance = llama["stance"]["benchmark"]
    assert stance["aligned_given_stance"] == {"follows": 0.0, "challenges": 1.0}
    assert stance["stance_given_aligned"]["challenges"] == 1.0
    text = drivers.markdown(built)
    for heading in ("## 1. Effect of each cue", "## 3. What predicts", "## 5. Do models", "## 7. How answers"):
        assert heading in text
    assert "## 9. Framing" not in text and "Qwen" not in text


def test_logistic_fit_matches_a_known_odds_ratio():
    rng = np.random.default_rng(1)
    x = (rng.random(20000) < 0.5).astype(float)
    y = (rng.random(20000) < np.where(x == 1, 0.5, 0.2)).astype(float)
    beta = drivers.fit_logistic(np.column_stack([np.ones_like(x), x]), y)
    assert abs(np.exp(beta[1]) - 4.0) < 0.4                                # (0.5/0.5) / (0.2/0.8)
