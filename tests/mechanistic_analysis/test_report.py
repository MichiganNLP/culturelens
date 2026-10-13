"""The report: built from a small synthetic output tree, with planted effects."""

from __future__ import annotations

import json

import numpy as np

from mechanistic_analysis import JUDGES, REPORT_LAYER, RESIST, report
from mechanistic_analysis.outcomes import CONDITIONS

SLUG = "llama-3.1-8b"


def build_tree(root, n=240):
    rng = np.random.default_rng(0)
    ids = [f"{'JPN' if i % 2 else 'BRA'}__a__{i:03d}__001" for i in range(n)]
    resist = rng.random(n) < 0.3
    names, columns = [], []
    for judge in (*JUDGES, "both"):
        for condition in CONDITIONS:
            names.append(f"{judge}__{condition}")
            columns.append(resist.astype(np.int8) if condition == RESIST
                           else (rng.random(n) < 0.6).astype(np.int8))
    (root / "outcomes").mkdir(parents=True)
    np.savez(root / "outcomes" / f"{SLUG}.npz", norm_ids=np.asarray(ids),
             names=np.asarray(names), outcomes=np.stack(columns, axis=1))

    layer = REPORT_LAYER[SLUG]
    layers = [0, layer, 31]
    outcome_names = [f"{judge}__{RESIST}" for judge in (*JUDGES, "both")]
    informative = resist + 0.7 * rng.standard_normal(n)
    noise = lambda: rng.standard_normal(n)  # noqa: E731

    def scores(signal_at_layer):
        return np.stack([np.stack([noise() if i != 1 or not signal_at_layer else informative
                                   for _ in outcome_names]) for i in range(3)]).astype(np.float16)

    (root / "readout" / SLUG).mkdir(parents=True)
    np.savez(root / "readout" / SLUG / "scores.npz", norm_ids=np.asarray(ids),
             outcomes=np.asarray(outcome_names), layers=np.asarray(layers),
             groups=np.arange(n), folds=np.zeros(n, int),
             **{f"score__{RESIST}": scores(True), "score__opposite_norm_only": scores(False),
                "score__country_only": scores(False), "score__base": scores(False),
                f"score__text:{RESIST}": scores(False)[:1]})

    (root / "interaction" / SLUG).mkdir(parents=True)
    np.savez(root / "interaction" / SLUG / "features.npz", norm_ids=np.asarray(ids),
             layers=np.asarray(layers), groups=np.arange(n),
             conflict=np.stack([noise(), informative, noise()]).astype(np.float32))
    with (root / "interaction" / SLUG / "layers.csv").open("w") as handle:
        handle.write("layer,state_norm,country,claim_true,claim_false,interaction_true,interaction_false,"
                     "additive_share_true,additive_share_false,country_given_false_ratio,"
                     "country_given_false_cosine,interaction_cosine\n")
        for index in range(32):
            handle.write(f"{index},10,3,4,4,2,2,0.6,0.5,1.0,0.6,{0.98 if index < 8 else 0.66}\n")

    (root / "localize" / SLUG).mkdir(parents=True)
    plain = np.stack([np.full(n, -4.0), np.full(n, 3.0)], axis=1)       # margin -7
    lifted = np.stack([np.full(n, 0.0), np.full(n, 3.0)], axis=1)       # margin -3
    np.savez(root / "localize" / SLUG / "localize.npz", norm_ids=np.asarray(ids),
             completed=np.ones(n, bool), windows=np.asarray([[0, 16], [16, 32], [0, 32]]),
             **{f"baseline__{RESIST}": plain, "baseline__country_norm": plain[:, ::-1],
                f"knock__{RESIST}__claim": np.stack([lifted, plain, lifted])})

    rates = {"false": 0.2, "false-noclaim": 0.6, "claim-only": 0.3}
    for arm, rate in rates.items():
        (root / "answers" / SLUG).mkdir(parents=True, exist_ok=True)
        (root / "answers" / SLUG / f"{arm}.jsonl").write_text("{}\n")
        labels = rng.random(100) < rate
        for judge in JUDGES:
            folder = root / "judgments" / f"{SLUG}.{arm}"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / f"{judge}.jsonl").write_text("\n".join(json.dumps(
                {"norm_id": ids[i], "alignment": "aligned" if labels[i] else "misaligned"})
                for i in range(100)))
    return resist


def test_report_recovers_planted_results_and_skips_what_is_missing(tmp_path):
    build_tree(tmp_path)
    built = report.build(tmp_path, draws=200)
    assert list(built["models"]) == [SLUG]                      # the other models have no outputs
    model = built["models"][SLUG]
    rows = model["readout"]["by_judge"][JUDGES[0]]
    assert model["readout"]["layer"] == REPORT_LAYER[SLUG]
    assert rows[RESIST]["auroc"] > 0.8 and rows[RESIST]["low"] < rows[RESIST]["auroc"] < rows[RESIST]["high"]
    assert 0.35 < rows["country_only"]["auroc"] < 0.65
    assert rows["country_adds"]["difference"] > 0.2 and rows["country_adds"]["low"] > 0
    assert model["interaction"]["conflict"]["both"]["auroc"] > 0.8
    assert model["interaction"]["interaction_cosine_early"] == 0.98
    assert model["interaction"]["interaction_cosine_at_layer"] == 0.66
    change = model["localisation"]["change"][f"{RESIST}__claim"]
    assert change["0-16"]["difference"] == 4.0 and change["16-32"]["difference"] == 0.0
    answers = model["answers"]["by_judge"][JUDGES[0]]
    assert answers["rate"]["false-noclaim"]["rate"] > answers["rate"]["false"]["rate"] + 0.2
    contrast = answers["contrast"]["false-noclaim - false"]
    assert contrast["difference"] > 0.2 and contrast["low"] > 0 and contrast["p"] < 0.05
    assert "false-nocountry - false" not in answers["contrast"]     # that arm was not run
    text = report.markdown(built)
    for heading in ("1. Behaviour", "2. Readout", "3. Country", "4. Where the claim", "5. Answers"):
        assert heading in text
    assert "Gemma-3-12B" not in text


def test_an_empty_tree_gives_an_empty_report(tmp_path):
    built = report.build(tmp_path, draws=10)
    assert built["models"] == {}
    assert "## 1" not in report.markdown(built)
    assert report.figures(built, tmp_path) == []


def test_margin_validity_pairs_each_margin_with_the_answers_of_the_same_intervention(tmp_path):
    rng = np.random.default_rng(1)
    n = 120
    ids = [f"JPN__a__{i:03d}__001" for i in range(n)]
    aligned = rng.random(n) < 0.4
    tracks = np.stack([aligned * 4.0 - 2.0 + rng.standard_normal(n), np.zeros(n)], axis=1)
    blind = np.stack([rng.standard_normal(n), np.zeros(n)], axis=1)
    cuts = list(report.ERASE_LAYERS[SLUG])
    (tmp_path / "erasure" / SLUG).mkdir(parents=True)
    np.savez(tmp_path / "erasure" / SLUG / "erasure.npz", norm_ids=np.asarray(ids),
             completed=np.ones(n, bool), cuts=np.asarray(cuts),
             **{f"baseline__{RESIST}": tracks, "baseline__country_norm": tracks,
                f"erase__{RESIST}": np.stack([blind] * len(cuts)),
                "erase__country_norm": np.stack([blind] * len(cuts))})
    for arm in ("false", "false-erase-first"):
        folder = tmp_path / "judgments" / f"{SLUG}.{arm}"
        folder.mkdir(parents=True)
        (folder / f"{JUDGES[0]}.jsonl").write_text("\n".join(json.dumps(
            {"norm_id": ids[i], "alignment": "aligned" if aligned[i] else "misaligned"}) for i in range(n)))
    found = report.margin_validity(tmp_path, SLUG, draws=100)
    assert set(found) == {"false", "false-erase-first"}          # the other arms were not judged
    assert found["false"]["auroc"] > 0.9
    assert 0.35 < found["false-erase-first"]["auroc"] < 0.65
