"""Framings of the false norm: same claim, same country line, only the wording around it."""

from __future__ import annotations

from behavioral_evaluation.prompts import CONDITION_SEGMENTS, DIAGNOSTIC_SEGMENTS, build_user_message

RECORD = {
    "norm_id": "JPN__etiquette__001__001",
    "scenario": {
        "country_context": "This interaction takes place in Japan.",
        "opposite_norm_evidence": "It is acceptable to start eating at once.",
        "norm_evidence": "Guests wait.", "value_evidence": None,
        "situation": "A guest's plate arrives first.", "question": "What should the guest do?",
    },
    "options": {"A": "Wait.", "B": "Start."},
}
HEAD = "Situation: A guest's plate arrives first.\n\nQuestion: What should the guest do?"
COUNTRY, CLAIM = "This interaction takes place in Japan.", "It is acceptable to start eating at once."


def text(condition: str) -> str:
    return build_user_message(RECORD, condition, False)[0]


def test_framings_change_only_the_wording_around_the_claim():
    assert text("country_opposite_norm") == f"{HEAD}\n\n{COUNTRY}\n\n{CLAIM}"
    assert text("country_opposite_norm_hearsay") == f"{HEAD}\n\n{COUNTRY}\n\nSomeone told me this: {CLAIM}"
    assert text("country_opposite_norm_guide") == (
        f"{HEAD}\n\n{COUNTRY}\n\nAccording to a guide to local customs: {CLAIM}")
    assert text("country_opposite_norm_doubt") == f"{HEAD}\n\n{COUNTRY}\n\n{CLAIM} I am not sure this is right."
    assert text("opposite_norm_country") == f"{HEAD}\n\n{CLAIM}\n\n{COUNTRY}"


def test_framings_are_diagnostic_and_leave_the_benchmark_conditions_alone():
    assert not set(DIAGNOSTIC_SEGMENTS) & set(CONDITION_SEGMENTS)
    assert len(CONDITION_SEGMENTS) == 8
    assert text("country_norm") == f"{HEAD}\n\n{COUNTRY}\n\nGuests wait."
