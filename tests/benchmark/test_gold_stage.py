"""Gold-response stages of the DeepSeek generator: what the teacher sees and what is kept."""

from __future__ import annotations

import json

import pytest

from benchmark import generate

RECORD = {
    "norm_id": "AFG__business_culture__001__003",
    "norm": "It is expected to wait to be seated at a meeting.",
    "values": ["conformity_rules"],
    "scenario": {
        "country_context": "This interaction takes place in Afghanistan.",
        "situation": "A visitor enters a meeting room with chairs around a table.",
        "question": "What should the visitor do?",
        "norm_evidence": "It is expected to wait to be seated at a meeting.",
        "value_evidence": "Following formal expectations is important.",
        "opposite_norm_text": "x",
        "opposite_norm_evidence": "It is acceptable to choose a seat immediately.",
    },
    "options": {"A": "The visitor should wait to be seated.",
                "B": "The visitor should choose a seat immediately."},
}
ANSWER = ("In Afghanistan, the visitor should wait until the host shows them where to sit. "
          "Seating at a meeting is usually arranged by the host, so standing until invited "
          "respects the formal expectations of the occasion.")


def test_teacher_sees_the_country_only_prompt_and_the_private_record():
    payload = json.loads(generate.user_payload("gold", RECORD))
    prompt = payload["student_prompt"]
    assert prompt.endswith("This interaction takes place in Afghanistan.")
    assert "wait to be seated" not in prompt and "important" not in prompt   # no norm, no value
    assert payload["private_source"] == {
        "norm": RECORD["norm"],
        "values": ["conformity_rules"],
        "value_evidence": "Following formal expectations is important.",
        "reference_answer": "The visitor should wait to be seated.",
        "contrasting_answer": "The visitor should choose a seat immediately.",
    }


def test_counterfactual_teacher_sees_the_false_claim_in_the_prompt():
    payload = json.loads(generate.user_payload("gold_cf", RECORD))
    assert payload["student_prompt"].endswith("It is acceptable to choose a seat immediately.")
    assert payload["private_source"]["false_claim"] == "It is acceptable to choose a seat immediately."
    assert payload["private_source"]["reference_answer"] == "The visitor should wait to be seated."


def test_kept_row_names_the_views_that_share_the_answer():
    [row] = generate.validate_gold("gold", {"response": ANSWER}, RECORD)
    assert row["condition"] == "country_only"
    assert row["target_for"] == ["country_only", "country_value", "country_norm", "country_value_norm"]
    assert row["response"] == ANSWER and row["norm_id"] == RECORD["norm_id"]
    [row] = generate.validate_gold("gold_cf", {"response": ANSWER}, RECORD)
    assert row["condition"] == "country_opposite_norm" and row["target_for"] == ["country_opposite_norm"]


@pytest.mark.parametrize("reply", [
    {"response": "Wait."},                                             # no reason given
    {"response": ANSWER + " This matches the reference answer."},      # talks about its inputs
    {"response": ANSWER + " As the prompt says, option A is best."},
    {"response": "<think>plan</think> " + ANSWER},
    {"analysis": "x", "response": ANSWER},                             # extra field
    ["not", "an", "object"],
])
def test_unusable_replies_are_rejected(reply):
    with pytest.raises(generate.BadOutput):
        generate.validate_gold("gold", reply, RECORD)


def test_ordinary_words_are_not_mistaken_for_leaks():
    text = ANSWER + " Taking the option at hand without asking can seem presumptuous."
    assert generate.validate_gold("gold", {"response": text}, RECORD)[0]["response"] == text


@pytest.mark.parametrize("extra", [
    " The documented norm is to wait.", " This reflects interpersonal conformity.",
    " The recommendation follows from the value placed on order.",
])
def test_meta_language_and_label_names_are_rejected(extra):
    with pytest.raises(generate.BadOutput):
        generate.validate_gold("gold", {"response": ANSWER + extra}, RECORD)


def test_counterfactual_answers_may_not_open_with_the_stock_sentence():
    with pytest.raises(generate.BadOutput, match="stock sentence"):
        generate.validate_gold("gold_cf", {"response": "That is not the usual expectation in Afghanistan. " + ANSWER}, RECORD)
    assert generate.validate_gold("gold_cf", {"response": ANSWER}, RECORD)
