"""GRPO / Critique-GRPO: everything that decides behaviour, without a GPU."""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import pytest

from finetune import (
    COUNTERFACTUAL_VIEWS, FACTUAL_VIEWS, method_name, reward_kinds, training_views,
)
from finetune.data import PromptPool, eligible, fixed_sample, load_records
from finetune.grpo import parse_args
from finetune.prompts import (
    constrained_message, critic_message, critic_system_prompt, draw_constrained, gold_letter,
    open_message, parse_critique, refinement_message,
)
from finetune.rewards import constrained_reward, critique_reward, group_advantages, weakest

DATA = Path(__file__).resolve().parents[2] / "data/processed/culturelens/splits"


def record(norm_id: str = "AFG__etiquette__001__001", *, value: bool = False) -> dict:
    return {
        "norm_id": norm_id,
        "norm": "Guests remove their shoes before entering a home.",
        "values": ["tradition"] if value else [],
        "scenario": {
            "country_context": "This interaction takes place in Afghanistan.",
            "situation": "A guest arrives at a friend's home.",
            "question": "What should the guest do at the door?",
            "norm_evidence": "Guests remove their shoes before entering a home.",
            "opposite_norm_evidence": "Guests keep their shoes on inside a home.",
            "opposite_norm_text": "Guests keep their shoes on inside a home.",
            "value_evidence": "Keeping the home clean is valued." if value else None,
        },
        "options": {"A": "Take off their shoes.", "B": "Keep their shoes on."},
    }


def records(count: int, *, value_every: int = 2) -> list[dict]:
    return [record(f"AFG__etiquette__001__{i:03d}", value=(i % value_every == 0))
            for i in range(count)]


# --------------------------------------------------------------- recipes

def test_counterfactual_switch_decides_the_views():
    assert training_views(False) == ("country_only",)
    assert set(training_views(True)) == {
        "country_only", "country_value", "country_norm", "country_opposite_norm"}


def test_reward_modes_and_names():
    assert reward_kinds("constrained") == ("constrained",)
    assert reward_kinds("critique+constrained") == ("critique", "constrained")
    assert method_name("constrained", False) == "grpo-constrained-country"
    assert method_name("critique", True) == "critique-grpo-critique-cf"
    assert method_name("critique+constrained", True) == "critique-grpo-critique-constrained-cf"
    with pytest.raises(ValueError):
        reward_kinds("verifier")


def test_argument_defaults_and_guards(tmp_path):
    args = parse_args(["--reward", "critique", "--train", str(tmp_path / "t.json")])
    assert args.views == FACTUAL_VIEWS and args.kinds == ("critique",)
    assert args.critic == "policy" and args.validation_critic == "reference"
    args = parse_args(["--reward", "constrained", "--counterfactual", "--train", "x"])
    assert args.views == COUNTERFACTUAL_VIEWS
    with pytest.raises(SystemExit):
        parse_args(["--reward", "critique", "--train", "x", "--refine-count", "8"])


# ------------------------------------------------------------------ data

def test_real_splits_load_and_have_values_where_value_evidence_exists():
    if not (DATA / "train.json").exists():
        pytest.skip("processed splits not present")
    rows = load_records(DATA / "train.json")
    assert rows and all(set(r["options"]) == {"A", "B"} for r in rows)
    assert sum(eligible(r, "country_value") for r in rows) == sum(
        bool(r["scenario"]["value_evidence"]) for r in rows)


def test_load_refuses_duplicates(tmp_path):
    path = tmp_path / "dup.json"
    path.write_text(json.dumps([record(), record()]))
    with pytest.raises(ValueError, match="Duplicate"):
        load_records(path)


def test_factual_pool_draws_country_only():
    pool = PromptPool(records(20), FACTUAL_VIEWS, ("constrained",), seed=1, prompts_per_step=8)
    assert {item["view"] for item in pool.draw(8)} == {"country_only"}


def test_counterfactual_pool_is_even_and_value_view_needs_evidence():
    pool = PromptPool(records(40), COUNTERFACTUAL_VIEWS, ("constrained",), seed=1, prompts_per_step=8)
    items = pool.draw(8)
    assert Counter(item["view"] for item in items) == {view: 2 for view in COUNTERFACTUAL_VIEWS}
    assert all(item["record"]["scenario"]["value_evidence"]
               for item in items if item["view"] == "country_value")


def test_true_and_false_norm_prompts_stay_on_the_same_records_through_wraparound():
    pool = PromptPool(records(7), COUNTERFACTUAL_VIEWS, ("constrained",), seed=3, prompts_per_step=8)
    for _ in range(30):  # 60 paired draws over 7 records: several reshuffles
        items = pool.draw(8)
        true = [i["record"]["norm_id"] for i in items if i["view"] == "country_norm"]
        false = [i["record"]["norm_id"] for i in items if i["view"] == "country_opposite_norm"]
        assert true == false


def test_paired_views_refuse_a_step_size_that_would_unpair_them():
    with pytest.raises(ValueError, match="multiple"):
        PromptPool(records(10), COUNTERFACTUAL_VIEWS, ("constrained",), seed=0, prompts_per_step=6)
    with pytest.raises(ValueError, match="multiple"):
        PromptPool(records(10), COUNTERFACTUAL_VIEWS, ("critique", "constrained"), seed=0,
                   prompts_per_step=12)


def test_mixed_tasks_each_cover_every_view():
    pool = PromptPool(records(40), COUNTERFACTUAL_VIEWS, ("critique", "constrained"), seed=0,
                      prompts_per_step=16)
    items = pool.draw(16)
    for kind in ("critique", "constrained"):
        assert Counter(i["view"] for i in items if i["kind"] == kind) == {
            view: 2 for view in COUNTERFACTUAL_VIEWS}


def test_fixed_validation_sample_is_reproducible():
    rows = records(50)
    first = [r["norm_id"] for r in fixed_sample(rows, "country_value", 5, seed=9)]
    assert first == [r["norm_id"] for r in fixed_sample(rows, "country_value", 5, seed=9)]
    assert all(eligible(r, "country_value") for r in fixed_sample(rows, "country_value", 0, seed=9))


# --------------------------------------------------------------- prompts

def test_constrained_order_and_gold_letter_agree():
    rec = record()
    plain = constrained_message(rec, "country_only", shuffled=False)
    swapped = constrained_message(rec, "country_only", shuffled=True)
    assert "A: Take off their shoes." in plain and gold_letter(False) == "a"
    assert "A: Keep their shoes on." in swapped and gold_letter(True) == "b"
    golds = Counter(draw_constrained(rec, "country_only", random.Random(s))[1] for s in range(200))
    assert 70 < golds["a"] < 130  # the order is genuinely re-drawn


def test_views_show_the_evidence_they_name():
    rec = record(value=True)
    assert "remove their shoes" not in constrained_message(rec, "country_only", shuffled=False).split("A:")[0]
    assert "keep their shoes on inside" in open_message(rec, "country_opposite_norm").lower()
    assert "remove their shoes before" in open_message(rec, "country_norm").lower()
    assert "clean is valued" in open_message(rec, "country_value")
    assert "A:" not in open_message(rec, "country_only")


def test_critic_sends_exactly_what_the_judge_sends():
    from behavioral_evaluation.judge import DEFAULT_PROMPT, judge_payload
    from common.io import load_prompt

    rec = record(value=True)
    assert critic_system_prompt() == load_prompt(DEFAULT_PROMPT)
    assert critic_message(rec, "  Keep your shoes on. ") == judge_payload(rec, "Keep your shoes on.")


def test_critic_sees_the_true_norm_and_never_the_false_one():
    payload = json.loads(critic_message(record(value=True), "Keep your shoes on."))
    assert payload["norm_text"] == "Guests remove their shoes before entering a home."
    assert payload["schwartz"] == ["tradition"]
    assert "keep their shoes on inside" not in json.dumps(payload).lower()


def test_verdicts_are_read_only_from_well_formed_judge_json():
    good = parse_critique('Sure. {"reasoning": "Keeps shoes on, contradicting the norm.", '
                          '"alignment": "misaligned", "confidence": "high", '
                          '"improvement": "Recommend removing shoes at the door."}')
    assert good["score"] == 0.0 and good["critique"] == "Recommend removing shoes at the door."
    aligned = parse_critique('{"reasoning": "Matches.", "alignment": "aligned", '
                             '"confidence": "medium", "improvement": null}')
    assert aligned["score"] == 1.0 and aligned["critique"] == ""
    assert parse_critique('{"reasoning": "x", "alignment": "partially_aligned", '
                          '"confidence": "low", "improvement": "y"}')["score"] == 0.5
    assert parse_critique("This answer is not aligned with the norm.") is None
    assert parse_critique('{"reasoning": "x", "alignment": "mostly", "confidence": "high"}') is None
    assert parse_critique('{"reasoning": "", "alignment": "aligned", "confidence": "high"}') is None


def test_refinement_carries_prompt_attempt_and_note():
    text = refinement_message("Q?", "first try", "wear fewer shoes")
    assert "Q?" in text and "first try" in text and "wear fewer shoes" in text


# --------------------------------------------------------------- rewards

def test_constrained_reward_is_gated_exact_match():
    ok = "<reasoning>Shoes come off.</reasoning><answer>A</answer>"
    kw = {"answer_budget": 512, "length_weight": 0.1}
    assert constrained_reward(ok, "a", "stop", 20, **kw)["reward"] == 1.0
    assert constrained_reward(ok, "b", "stop", 20, **kw)["reward"] == 0.0
    assert constrained_reward(ok, "a", "length", 20, **kw)["reward"] == 0.0
    assert constrained_reward("<answer>A</answer>", "a", "stop", 5, **kw)["format_valid"] is False
    assert constrained_reward(ok, "a", "stop", 1024, **kw)["reward"] == pytest.approx(0.9)


def test_critique_reward_follows_the_verdict_and_counts_unreadable_ones():
    kw = {"answer_budget": 512, "length_weight": 0.0}
    verdict = parse_critique('{"reasoning": "r", "alignment": "partially_aligned", '
                             '"confidence": "high", "improvement": "c"}')
    assert critique_reward("an answer", "stop", 10, verdict, **kw)["reward"] == 0.5
    unreadable = critique_reward("an answer", "stop", 10, None, **kw)
    assert unreadable["reward"] == 0.0 and unreadable["verdict_parsed"] is False
    assert critique_reward("   ", "stop", 1, verdict, **kw)["reward"] == 0.0


def test_advantages_handle_groups_of_different_sizes():
    rewards = [1.0, 0.0, 1.0, 0.0,  0.5, 0.5, 0.5,  0.0, 1.0, 1.0]
    groups  = [0,   0,   0,   0,    1,   1,   1,    2,   2,   2]
    advantages, informative = group_advantages(rewards, groups)
    assert informative == {0: True, 1: False, 2: True}
    assert advantages[4:7] == [0.0, 0.0, 0.0]
    for group in (0, 2):
        members = [a for a, g in zip(advantages, groups) if g == group]
        assert sum(members) == pytest.approx(0.0, abs=1e-9)
    assert advantages[0] > 0 > advantages[1]


def test_weakest_picks_lowest_eligible_per_group():
    rewards = [0.0, 1.0, 0.5, 0.0,  0.2, 0.9]
    groups = [0, 0, 0, 0, 1, 1]
    eligible = [True, True, True, False, True, True]
    assert weakest(rewards, groups, 2, eligible=eligible) == [0, 2, 4, 5]


# -------------------------------------------------------------- summarize

def test_summary_uses_latest_rows_and_reports_coverage(tmp_path):
    from finetune.summarize import summarize

    def write(path, rows):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    root = tmp_path / "eval"
    write(root / "base/test_seen/constrained.jsonl", [
        {"norm_id": "n1", "condition": "country_only", "correct": None},
        {"norm_id": "n1", "condition": "country_only", "correct": True},   # recovery replaces it
        {"norm_id": "n2", "condition": "country_only", "correct": False}])
    write(root / "base/test_seen/unconstrained.jsonl", [
        {"norm_id": "n1", "condition": "country_only", "response": "x"},
        {"norm_id": "n2", "condition": "country_only", "response": "y"}])
    write(root / "judgments/base-test_seen/gemma.jsonl", [
        {"norm_id": "n1", "condition": "country_only", "alignment": "aligned"}])
    cell = summarize(root, "gemma")["models"]["base"]["test_seen"]["country_only"]
    assert cell["accuracy"] == 0.5
    assert cell["aligned"] == 1.0 and cell["coverage"] == 0.5


# ----------------------------------------------------- epochs and resume

def test_an_epoch_covers_every_record_under_every_view_and_task():
    rows = records(40)
    assert PromptPool(rows, FACTUAL_VIEWS, ("constrained",), seed=0).prompts_per_epoch() == 40
    assert PromptPool(rows, COUNTERFACTUAL_VIEWS, ("constrained",), seed=0).prompts_per_epoch() == 160
    assert PromptPool(rows, COUNTERFACTUAL_VIEWS, ("critique", "constrained"),
                      seed=0).prompts_per_epoch() == 320


def test_a_resumed_pool_draws_exactly_what_the_original_would_have():
    rows = records(11)
    kw = dict(views=COUNTERFACTUAL_VIEWS, kinds=("critique", "constrained"), seed=5, prompts_per_step=8)
    original = PromptPool(rows, **kw)
    for _ in range(7):                       # far enough to wrap and reshuffle
        original.draw(8)
    saved = json.loads(json.dumps(original.state()))   # as it is stored on disk
    resumed = PromptPool(rows, **kw)
    resumed.load_state(saved)
    key = lambda items: [(i["record"]["norm_id"], i["view"], i["kind"]) for i in items]
    for _ in range(12):
        assert key(resumed.draw(8)) == key(original.draw(8))


def test_pool_state_from_a_different_recipe_is_refused():
    state = PromptPool(records(10), FACTUAL_VIEWS, ("constrained",), seed=0).state()
    with pytest.raises(ValueError, match="different views"):
        PromptPool(records(10), COUNTERFACTUAL_VIEWS, ("constrained",), seed=0).load_state(state)


def test_steps_and_epochs_are_alternatives():
    assert parse_args(["--reward", "constrained", "--train", "x"]).epochs == 1.0
    assert parse_args(["--reward", "constrained", "--train", "x", "--steps", "9"]).epochs is None
    assert parse_args(["--reward", "constrained", "--train", "x", "--epochs", "2"]).steps is None
    with pytest.raises(SystemExit):
        parse_args(["--reward", "constrained", "--train", "x", "--steps", "9", "--epochs", "1"])
