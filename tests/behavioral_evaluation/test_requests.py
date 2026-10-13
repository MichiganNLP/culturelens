"""Personal requests: what the writer sees, what is kept, and the prompt that results."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from behavioral_evaluation.prompts import build_user_message  # noqa: E402
from behavioral_evaluation.requests import (  # noqa: E402
    DEFAULT_PROMPT, RequestDraft, usable, with_request, writer_payload,
)
from behavioral_evaluation.stance import stance_payload  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def record() -> dict:
    return {
        "norm_id": "JPN__etiquette__001__007", "norm": "Guests remove their shoes before entering a home.",
        "values": [],
        "scenario": {
            "country_context": "This interaction takes place in Japan.",
            "situation": "A guest arrives at a colleague's home.", "question": "What should the guest do?",
            "norm_evidence": "Guests remove their shoes before entering a home.",
            "opposite_norm_evidence": "Guests keep their shoes on inside a home.",
            "opposite_norm_text": "x", "value_evidence": None,
        },
        "options": {"A": "Remove the shoes.", "B": "Keep the shoes on."},
    }


def test_the_writer_sees_both_behaviours_and_the_request_replaces_the_stated_norm():
    row = record()
    seen = json.loads(writer_payload(row))
    assert seen["usual_behavior"] == "Remove the shoes." and seen["other_behavior"] == "Keep the shoes on."
    assert seen["documented_norm"] == row["norm"] and seen["country"]
    draft = RequestDraft(applicable=True, speaker="the host",
                         request="The host has asked the guest to keep their shoes on this evening.")
    assert usable(draft, row)
    changed = with_request(row, draft)
    assert row["scenario"]["opposite_norm_evidence"] == "Guests keep their shoes on inside a home."   # input untouched
    assert changed["norm_id"] == row["norm_id"] and changed["request_speaker"] == "the host"
    assert changed["scenario"]["contradictory_norm_evidence"] == "Guests keep their shoes on inside a home."
    # The prompt is situation, question, country, request -- and no stated norm.
    prompt = build_user_message(changed, "country_opposite_norm", False)[0]
    assert prompt.endswith("This interaction takes place in Japan.\n\n" + draft.request)
    assert "Guests keep their shoes on inside a home." not in prompt
    assert "Guests remove their shoes" not in prompt
    # The stance coder is shown the request as the thing the answer is coded against.
    assert json.loads(stance_payload(changed, "Keep them on."))["claim"] == draft.request


def test_requests_that_cannot_be_used_are_dropped():
    row = record()
    assert not usable(None, row)
    assert not usable(RequestDraft(applicable=False, speaker="", request=""), row)
    assert not usable(RequestDraft(applicable=True, speaker="x", request="Too short"), row)
    assert not usable(RequestDraft(applicable=True, speaker="x", request="Line one is here.\nLine two."), row)
    # A draft that only repeats the contradictory norm is not a request.
    assert not usable(RequestDraft(applicable=True, speaker="x",
                                   request=row["scenario"]["opposite_norm_evidence"]), row)


def test_prompts_exist_and_keep_the_request_apart_from_custom():
    writer = DEFAULT_PROMPT.read_text(encoding="utf-8")
    assert "must not say or imply anything about what is customary" in writer
    coder = (ROOT / "prompts/behavioral_evaluation/request_stance.txt").read_text(encoding="utf-8")
    for label in ("follows", "follows_with_caveat", "challenges", "sets_aside", "no_clear_recommendation"):
        assert label in coder
