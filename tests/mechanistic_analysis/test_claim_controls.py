"""Claim controls: the claim's wording stays fixed while the country is removed or replaced."""

from __future__ import annotations

import pytest

from mechanistic_analysis.activations.conditions import (
    ACTIVATION_CONDITIONS,
    CLAIM_CONTROL_CONDITIONS,
    build_user_message,
    country_derangement,
    eligible,
)


def record(iso3: str, country: str) -> dict:
    return {
        "norm_id": f"{iso3}__etiquette__001__001", "norm": "n", "values": [],
        "options": {"A": "a", "B": "b"},
        "scenario": {
            "country_context": f"This interaction takes place in {country}.",
            "situation": "A guest arrives.", "question": "What should the guest do?",
            "norm_evidence": f"True norm of {country}.",
            "opposite_norm_evidence": f"Opposite norm of {country}.",
            "value_evidence": None,
        },
    }


RECORDS = [record("JPN", "Japan"), record("BRA", "Brazil"), record("KEN", "Kenya")]
BASE = "Situation: A guest arrives.\n\nQuestion: What should the guest do?"


def test_claim_controls_are_registered_and_always_eligible():
    assert set(CLAIM_CONTROL_CONDITIONS) <= set(ACTIVATION_CONDITIONS)
    assert all(eligible(RECORDS[0], condition) for condition in CLAIM_CONTROL_CONDITIONS)


def test_no_country_twin_matches_norm_only_except_for_the_claim():
    true_claim, _ = build_user_message(RECORDS[0], "norm_only", False)
    opposite, _ = build_user_message(RECORDS[0], "opposite_norm_only", False)
    assert true_claim == f"{BASE}\n\nTrue norm of Japan."
    assert opposite == f"{BASE}\n\nOpposite norm of Japan."
    assert "takes place in" not in opposite


def test_shuffled_country_keeps_the_claim_and_swaps_only_the_country_line():
    contexts, donors = country_derangement(RECORDS, seed=42)
    for row in RECORDS:
        iso3 = row["norm_id"][:3]
        own = row["scenario"]["country_context"]
        assert donors[iso3] != iso3 and contexts[iso3] != own
        for condition, original in (("shuffled_country_norm", "country_norm"),
                                    ("shuffled_country_opposite_norm", "country_opposite_norm")):
            swapped, _ = build_user_message(
                row, condition, False, shuffled_context_by_iso3=contexts)
            factual, _ = build_user_message(row, original, False)
            assert swapped == factual.replace(own, contexts[iso3])
        # The same donor as the claim-free country_shuffled control.
        plain, _ = build_user_message(
            row, "country_shuffled", False, shuffled_context_by_iso3=contexts)
        assert plain == f"{BASE}\n\n{contexts[iso3]}"


def test_claim_controls_refuse_options_and_a_missing_mapping():
    with pytest.raises(ValueError, match="unconstrained"):
        build_user_message(RECORDS[0], "opposite_norm_only", True)
    with pytest.raises(ValueError, match="shuffled context mapping"):
        build_user_message(RECORDS[0], "shuffled_country_norm", False)
