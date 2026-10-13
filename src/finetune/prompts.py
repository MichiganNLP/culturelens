"""Every prompt the loop sends: policy (constrained and open), critic, refinement.

Policy prompts are built with the evaluation's own pieces, so the text trained
on is the text tested on.

**The A/B order is re-drawn every time a prompt is sampled.** The evaluation
fixes it by norm id, which is right for a reproducible benchmark. In training it
would make the correct letter a fixed function of the prompt, so a policy that
saw an item twice could learn the letter instead of the culture.

**The critic is the policy's own model, applying the judge's rubric.** It sees
exactly what the evaluation judge sees -- country, documented norm, value
labels, situation, question, answer -- and never the false norm a
counterfactual prompt asserted, nor which condition the policy saw.
"""

from __future__ import annotations

import json
import random
import re
from typing import Any

from behavioral_evaluation.prompts import (
    _SEGMENT_FRAME,
    CONDITION_SEGMENTS,
    CONSTRAINED_SYSTEM_PROMPT,
    build_user_message,
)

# How a stated norm can be presented. "plain" is the benchmark's wording; the
# others are the diagnostic framings of behavioral_evaluation.prompts: attributed
# to hearsay, attributed to a guide, followed by the user's doubt, and placed
# before the country line. The same wording is applied to a true norm, so a
# framing never tells the model whether the norm is the documented one.
FRAMINGS: tuple[str, ...] = ("plain", "hearsay", "guide", "doubt", "first")
_NORM_FIELD = {"country_norm": "norm_evidence", "country_opposite_norm": "opposite_norm_evidence"}

_SEGMENT_FIELD = {
    "country": "country_context",
    "value": "value_evidence",
    "norm": "norm_evidence",
    "opposite_norm": "opposite_norm_evidence",
}


# ----------------------------------------------------------------- policy

def _evidence_parts(record: dict[str, Any], view: str, framing: str = "plain") -> list[str]:
    scenario = record["scenario"]
    parts = [f"Situation: {scenario['situation']}\n\nQuestion: {scenario['question']}"]
    if framing != "plain" and view in _NORM_FIELD:
        if framing not in FRAMINGS:
            raise ValueError(f"Unknown framing {framing!r}")
        norm = scenario[_NORM_FIELD[view]]
        template = _SEGMENT_FRAME.get(f"opposite_norm_{framing}")
        norm = template.format(text=norm) if template else norm
        country = scenario["country_context"]
        return parts + ([norm, country] if framing == "first" else [country, norm])
    for segment in CONDITION_SEGMENTS[view]:
        value = scenario[_SEGMENT_FIELD[segment]]
        if not value:
            raise ValueError(f"{record['norm_id']} has no {_SEGMENT_FIELD[segment]} for {view}")
        parts.append(value)
    return parts


def constrained_message(record: dict[str, Any], view: str, *, shuffled: bool,
                        framing: str = "plain") -> str:
    """The A/B user message with the option order chosen by the caller."""

    options = record["options"]
    first, second = (options["B"], options["A"]) if shuffled else (options["A"], options["B"])
    parts = _evidence_parts(record, view, framing)
    parts.append(f"A: {first}\nB: {second}\n\nChoose A or B.")
    return "\n\n".join(parts)


def gold_letter(shuffled: bool) -> str:
    """Which letter carries option A, the norm-aligned answer."""

    return "b" if shuffled else "a"


def draw_constrained(record: dict[str, Any], view: str, rng: random.Random) -> tuple[str, str]:
    shuffled = rng.random() < 0.5
    return constrained_message(record, view, shuffled=shuffled), gold_letter(shuffled)


def open_message(record: dict[str, Any], view: str, framing: str = "plain") -> str:
    """The open-ended user message: exactly what unconstrained evaluation sends.
    With a ``framing``, the stated norm of a norm view is presented that way."""

    if framing != "plain" and view in _NORM_FIELD:
        return "\n\n".join(_evidence_parts(record, view, framing))
    message, _ = build_user_message(record, view, include_options=False)
    return message


# ----------------------------------------------------------------- critic
#
# The critic applies the evaluation judge's own rubric, sent exactly as the
# judge sends it: judge.txt as the system prompt, judge_payload() as the user
# message. Training therefore optimises the same definition of "aligned" the
# benchmark measures; the only difference is who applies it -- the policy
# itself during training, an independent model at evaluation. The payload
# carries the documented norm and its value labels and nothing else: never the
# false norm a counterfactual prompt asserted, never which view the policy saw.

VERDICT_SCORES = {"aligned": 1.0, "partially_aligned": 0.5, "misaligned": 0.0}
_CONFIDENCE = frozenset({"high", "medium", "low"})


def critic_system_prompt() -> str:
    from behavioral_evaluation.judge import DEFAULT_PROMPT
    from common.io import load_prompt

    return load_prompt(DEFAULT_PROMPT)


def critic_message(record: dict[str, Any], answer: str) -> str:
    from behavioral_evaluation.judge import judge_payload

    return judge_payload(record, answer.strip())


def parse_critique(text: str) -> dict[str, Any] | None:
    """``{critique, verdict, score, reasoning}`` from the judge's JSON, or None.

    The judge is decoded under a JSON schema; the in-loop critic is sampled with
    plain generation, so its object is located and validated here instead. Only
    a well-formed verdict counts -- reading "aligned" out of free text would let
    "not aligned" score as aligned. ``critique`` is the judge's ``improvement``
    field, the one concrete change it asks for, which is what a rewrite needs.
    """

    for match in re.finditer(r"\{.*?\}", text, re.DOTALL):
        try:
            payload = json.loads(match.group(0))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue
        verdict = payload.get("alignment")
        if verdict not in VERDICT_SCORES or payload.get("confidence") not in _CONFIDENCE:
            continue
        reasoning = payload.get("reasoning")
        if not isinstance(reasoning, str) or not reasoning.strip():
            continue
        improvement = payload.get("improvement")
        return {"verdict": verdict, "score": VERDICT_SCORES[verdict],
                "reasoning": reasoning.strip(),
                "critique": improvement.strip() if isinstance(improvement, str) else ""}
    return None


# ------------------------------------------------------------- refinement

REFINEMENT_TEMPLATE = (
    "{prompt}\n\n"
    "A previous answer was:\n{attempt}\n\n"
    "A reviewer's note on that answer:\n{critique}\n\n"
    "Write a better answer to the original question. Answer it directly, as if "
    "for the first time, without mentioning the previous answer or the note."
)


def refinement_message(prompt: str, attempt: str, critique: str) -> str:
    return REFINEMENT_TEMPLATE.format(
        prompt=prompt.strip(), attempt=attempt.strip(), critique=critique.strip())


__all__ = [
    "CONSTRAINED_SYSTEM_PROMPT", "VERDICT_SCORES", "critic_system_prompt",
    "constrained_message", "critic_message", "draw_constrained", "gold_letter",
    "open_message", "parse_critique", "refinement_message",
]
