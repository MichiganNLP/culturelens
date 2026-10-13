"""Prompt conditions used by behavioral evaluation and activation extraction."""

from __future__ import annotations

import random
from typing import Any

from behavioral_evaluation.prompts import (
    CONDITION_SEGMENTS,
    build_user_message as build_behavioral_user_message,
)


CORE_CONDITIONS = tuple(CONDITION_SEGMENTS)
CONTROL_CONDITIONS = ("country_neutral", "country_shuffled")
# Controls for the compatibility probe. They hold the claim's wording fixed and
# change only whether, and which, country accompanies it: no country at all
# (``norm_only`` is the documented-norm twin of ``opposite_norm_only``), or the
# country ``country_shuffled`` assigns in place of the record's own.
CLAIM_CONTROL_CONDITIONS = (
    "opposite_norm_only",
    "shuffled_country_norm",
    "shuffled_country_opposite_norm",
)
# The country line with no situation or question: the baseline for how much of
# a country's representation comes from its name alone.
GEOMETRY_CONTROL_CONDITIONS = ("country_line_only",)
ACTIVATION_CONDITIONS = (
    *CORE_CONDITIONS,
    *CONTROL_CONDITIONS,
    *CLAIM_CONTROL_CONDITIONS,
    *GEOMETRY_CONTROL_CONDITIONS,
)
# Conditions whose country line belongs to a donor country, not the record's.
SHUFFLED_COUNTRY_CONDITIONS = frozenset(
    {"country_shuffled", "shuffled_country_norm", "shuffled_country_opposite_norm"}
)
_CLAIM_FIELD = {
    "opposite_norm_only": "opposite_norm_evidence",
    "shuffled_country_norm": "norm_evidence",
    "shuffled_country_opposite_norm": "opposite_norm_evidence",
}

VALUE_CONDITIONS = frozenset(
    {"value_only", "country_value", "country_value_norm"}
)
COUNTRY_CONDITIONS = frozenset(
    {
        "country_only",
        "country_value",
        "country_norm",
        "country_value_norm",
        "country_opposite_norm",
        "country_line_only",
    }
)
COUNTRY_NEUTRAL_CONTEXT = "This interaction takes place in this setting."


def record_metadata(record: dict[str, Any]) -> tuple[str, str]:
    """Return ISO3 and source category encoded in a v2 ``norm_id``."""

    norm_id = str(record.get("norm_id", ""))
    parts = norm_id.split("__")
    if len(parts) < 2 or len(parts[0]) != 3:
        raise ValueError(f"Malformed norm_id: {norm_id!r}")
    return parts[0], parts[1]


def eligible(record: dict[str, Any], condition: str) -> bool:
    """Whether a record has every component needed by ``condition``."""

    if condition not in ACTIVATION_CONDITIONS:
        raise ValueError(f"Unknown activation condition: {condition}")
    scenario = record.get("scenario")
    if not isinstance(scenario, dict) or not isinstance(record.get("options"), dict):
        return False
    if condition in VALUE_CONDITIONS:
        return scenario.get("value_evidence") is not None
    return True


def filter_records(
    records: list[dict[str, Any]], condition: str
) -> list[dict[str, Any]]:
    """Filter without changing dataset order."""

    return [record for record in records if eligible(record, condition)]


def country_derangement(
    records: list[dict[str, Any]], seed: int
) -> tuple[dict[str, str], dict[str, str]]:
    """Map each source ISO3 to a different country context and donor ISO3."""

    context_by_iso3: dict[str, str] = {}
    for record in records:
        iso3, _ = record_metadata(record)
        context = str(record["scenario"]["country_context"])
        prior = context_by_iso3.setdefault(iso3, context)
        if prior != context:
            raise ValueError(f"Country {iso3} has inconsistent country_context values")
    countries = sorted(context_by_iso3)
    if len(countries) < 2:
        raise ValueError("country_shuffled requires at least two countries")
    order = list(countries)
    random.Random(seed).shuffle(order)
    donor_by_iso3 = {
        source: order[(index + 1) % len(order)]
        for index, source in enumerate(order)
    }
    if any(source == donor for source, donor in donor_by_iso3.items()):
        raise RuntimeError("Country derangement unexpectedly contains a fixed point")
    contexts = {
        source: context_by_iso3[donor]
        for source, donor in donor_by_iso3.items()
    }
    return contexts, donor_by_iso3


def build_user_message(
    record: dict[str, Any],
    condition: str,
    include_options: bool,
    *,
    shuffled_context_by_iso3: dict[str, str] | None = None,
) -> tuple[str, bool]:
    """Render a behavioral condition or a matched country control."""

    if condition in CONDITION_SEGMENTS:
        return build_behavioral_user_message(record, condition, include_options)
    if condition == "country_line_only":
        if include_options:
            raise ValueError(f"{condition} is registered for unconstrained prompts only")
        return str(record["scenario"]["country_context"]), False

    base, _ = build_behavioral_user_message(record, "base", include_options=False)
    if condition == "country_neutral":
        context = COUNTRY_NEUTRAL_CONTEXT
    elif condition in SHUFFLED_COUNTRY_CONDITIONS:
        if shuffled_context_by_iso3 is None:
            raise ValueError(f"{condition} requires a shuffled context mapping")
        iso3, _ = record_metadata(record)
        context = shuffled_context_by_iso3[iso3]
    elif condition == "opposite_norm_only":
        context = None
    else:
        raise ValueError(f"Unknown activation condition: {condition}")

    # Same order as the behavioral prompts: situation, country line, claim.
    parts = [base] if context is None else [base, context]
    if condition in _CLAIM_FIELD:
        if include_options:
            raise ValueError(f"{condition} is registered for unconstrained prompts only")
        parts.append(str(record["scenario"][_CLAIM_FIELD[condition]]))
        return "\n\n".join(parts), False
    shuffled = False
    if include_options:
        option_message, shuffled = build_behavioral_user_message(
            record, "base", include_options=True
        )
        prefix = f"{base}\n\n"
        if not option_message.startswith(prefix):
            raise RuntimeError("Behavioral option prompt no longer extends the base prompt")
        parts.append(option_message[len(prefix):])
    return "\n\n".join(parts), shuffled


__all__ = [
    "ACTIVATION_CONDITIONS",
    "CLAIM_CONTROL_CONDITIONS",
    "CONTROL_CONDITIONS",
    "CORE_CONDITIONS",
    "COUNTRY_CONDITIONS",
    "GEOMETRY_CONTROL_CONDITIONS",
    "COUNTRY_NEUTRAL_CONTEXT",
    "SHUFFLED_COUNTRY_CONDITIONS",
    "VALUE_CONDITIONS",
    "build_user_message",
    "country_derangement",
    "eligible",
    "filter_records",
    "record_metadata",
]
