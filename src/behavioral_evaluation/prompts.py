"""Dependency-light, canonical CultureLens behavioral prompt construction."""

from __future__ import annotations

import hashlib


CONSTRAINED_SYSTEM_PROMPT = (
    "You will be given a multiple-choice cultural scenario with options A and B.\n\n"
    "Think about the cultural norms and values relevant to the situation, then select "
    "the best option. Keep your reasoning concise and culturally grounded. If cultural "
    "context is provided, use it. Do not introduce unsupported cultural claims.\n\n"
    "Your response must begin immediately with <reasoning> — do not write anything before it.\n\n"
    "<reasoning>your reasoning here</reasoning>\n"
    "<answer>A or B</answer>"
)


CONDITION_SEGMENTS: dict[str, list[str]] = {
    "base": [],
    "country_only": ["country"],
    "value_only": ["value"],
    "norm_only": ["norm"],
    "country_value": ["country", "value"],
    "country_norm": ["country", "norm"],
    "country_value_norm": ["country", "value", "norm"],
    "country_opposite_norm": ["country", "opposite_norm"],
}
# Diagnostic prompts outside the benchmark's eight conditions. They are never
# run by default; name them with --conditions.
#   opposite_norm_only  the false norm with no country line: against
#                       country_opposite_norm, it shows what naming the
#                       country does to a false claim.
DIAGNOSTIC_SEGMENTS: dict[str, list[str]] = {
    "opposite_norm_only": ["opposite_norm"],
    # Framing of the false norm, country stated (compare with country_opposite_norm):
    # the same claim attributed to hearsay, attributed to a guide, followed by the
    # user's own doubt, and placed before the country line instead of after it.
    "country_opposite_norm_hearsay": ["country", "opposite_norm_hearsay"],
    "country_opposite_norm_guide": ["country", "opposite_norm_guide"],
    "country_opposite_norm_doubt": ["country", "opposite_norm_doubt"],
    "opposite_norm_country": ["opposite_norm", "country"],
}
ALL_SEGMENTS: dict[str, list[str]] = {**CONDITION_SEGMENTS, **DIAGNOSTIC_SEGMENTS}

_SEGMENT_FIELD: dict[str, str] = {
    "country": "country_context",
    "value": "value_evidence",
    "norm": "norm_evidence",
    "opposite_norm": "opposite_norm_evidence",
    "opposite_norm_hearsay": "opposite_norm_evidence",
    "opposite_norm_guide": "opposite_norm_evidence",
    "opposite_norm_doubt": "opposite_norm_evidence",
}
# How a framed segment words its field; segments not listed state it as is.
_SEGMENT_FRAME: dict[str, str] = {
    "opposite_norm_hearsay": "Someone told me this: {text}",
    "opposite_norm_guide": "According to a guide to local customs: {text}",
    "opposite_norm_doubt": "{text} I am not sure this is right.",
}


def is_shuffled(norm_id: str) -> bool:
    """Deterministic A/B swap based on norm_id, consistent across models."""

    return int(hashlib.md5(norm_id.encode()).hexdigest(), 16) % 2 == 1


def build_user_message(
    record: dict, condition: str, include_options: bool
) -> tuple[str, bool]:
    """Return the canonical Phase 2 user text and deterministic shuffle flag."""

    scenario = record["scenario"]
    parts: list[str] = [
        f"Situation: {scenario['situation']}\n\nQuestion: {scenario['question']}",
    ]
    for segment in ALL_SEGMENTS[condition]:
        text = scenario[_SEGMENT_FIELD[segment]]
        parts.append(_SEGMENT_FRAME[segment].format(text=text) if segment in _SEGMENT_FRAME else text)

    if include_options:
        options = record["options"]
        shuffled = is_shuffled(record["norm_id"])
        option_a = options["B"] if shuffled else options["A"]
        option_b = options["A"] if shuffled else options["B"]
        parts.append(f"A: {option_a}\nB: {option_b}\n\nChoose A or B.")
    else:
        shuffled = False
    return "\n\n".join(parts), shuffled


__all__ = [
    "ALL_SEGMENTS",
    "CONDITION_SEGMENTS",
    "CONSTRAINED_SYSTEM_PROMPT",
    "DIAGNOSTIC_SEGMENTS",
    "build_user_message",
    "is_shuffled",
]
