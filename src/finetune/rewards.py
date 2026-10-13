"""Rewards and group-relative advantages, in plain Python.

Nothing here needs a GPU, so every decision about what earns reward is testable
without one.

**Format is a gate, not a term.** Added to the score, a format bonus is farmed:
emit tidy tags, answer at chance, still collect. Multiplied in, a malformed
rollout is worth nothing however tidy it looks, and a well-formed one earns only
what its answer earns. A rollout cut off at the token budget fails the gate.

**Reasoning is never scored for content.** Paying for a norm named in the
reasoning would reward reciting training norms. Correct reasoning pays for
itself through the answer it enables.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

from behavioral_evaluation.generate import parse_choice

TRUNCATED = "length"
LETTERS = ("a", "b")


def length_penalty(tokens: int, budget: int, weight: float) -> float:
    """Zero within budget, then linear in the overrun."""

    if weight <= 0.0 or budget <= 0 or tokens <= budget:
        return 0.0
    return weight * (tokens - budget) / budget


def constrained_reward(text: str, gold: str, finish_reason: str, tokens: int, *,
                       answer_budget: int, length_weight: float) -> dict[str, Any]:
    """Exact match on the A/B letter, gated on a reasoned, complete answer."""

    reasoning, letter = parse_choice(text)
    valid = finish_reason != TRUNCATED and letter in LETTERS and bool(reasoning and reasoning.strip())
    correct = valid and letter == gold
    penalty = length_penalty(tokens, answer_budget, length_weight)
    return {
        "reward": max(0.0, float(correct) - penalty) if valid else 0.0,
        "format_valid": valid,
        "correct": correct,
        # Logged so a collapse onto one letter is visible: with the order
        # re-drawn per prompt, accuracy alone would hide it near chance.
        "chose_a": letter == "a",
        "length_penalty": penalty,
    }


def critique_reward(text: str, finish_reason: str, tokens: int, verdict: dict[str, Any] | None, *,
                    answer_budget: int, length_weight: float) -> dict[str, Any]:
    """The critic's verdict score, gated on a complete, non-empty answer.

    An unreadable verdict is worth nothing and counted, so a critic that stops
    producing verdicts shows up in the log rather than as a quiet zero reward.
    """

    valid = finish_reason != TRUNCATED and bool(text.strip())
    parsed = verdict is not None
    score = verdict["score"] if parsed else 0.0
    penalty = length_penalty(tokens, answer_budget, length_weight)
    return {
        "reward": max(0.0, score - penalty) if valid else 0.0,
        "format_valid": valid,
        "verdict_parsed": parsed,
        "verdict": verdict["verdict"] if parsed else None,
        "length_penalty": penalty,
    }


def group_advantages(rewards: Sequence[float], groups: Sequence[int], *,
                     epsilon: float = 1e-4) -> tuple[list[float], dict[int, bool]]:
    """Standardise rewards within each group; groups may differ in size.

    Refinements join their prompt's group, so critique groups are larger than
    constrained ones. Computing the baseline per group, whatever its size, keeps
    that honest without padding a short group with copies of its best rollout --
    which would raise the group mean and shrink every advantage in it.

    A group whose rollouts all score alike carries no preference and gets zero
    advantage; it is reported as uninformative so the loop can drop it.
    """

    members: dict[int, list[int]] = {}
    for index, group in enumerate(groups):
        members.setdefault(group, []).append(index)
    advantages = [0.0] * len(rewards)
    informative: dict[int, bool] = {}
    for group, indices in members.items():
        values = [rewards[i] for i in indices]
        mean = sum(values) / len(values)
        # Sample standard deviation, matching the usual GRPO estimator.
        spread = (math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
                  if len(values) > 1 else 0.0)
        informative[group] = spread > epsilon
        if informative[group]:
            for i in indices:
                advantages[i] = (rewards[i] - mean) / (spread + epsilon)
    return advantages, informative


def weakest(rewards: Sequence[float], groups: Sequence[int], count: int, *,
            eligible: Sequence[bool]) -> list[int]:
    """The ``count`` lowest-scoring eligible rollouts in each group.

    Refining the weakest is where a critique has something to say; a rollout
    that already scored full marks mostly comes back as a paraphrase.
    """

    by_group: dict[int, list[int]] = {}
    for index, group in enumerate(groups):
        if eligible[index]:
            by_group.setdefault(group, []).append(index)
    chosen: list[int] = []
    for indices in by_group.values():
        indices.sort(key=lambda index: rewards[index])
        chosen.extend(indices[:count])
    return sorted(chosen)


__all__ = ["constrained_reward", "critique_reward", "group_advantages",
           "length_penalty", "weakest"]
