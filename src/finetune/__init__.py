"""CultureLens reinforcement fine-tuning: GRPO and Critique-GRPO.

Two methods, three reward modes, and one counterfactual switch.

**Reward modes** (``--reward``):

* ``constrained``            the A/B task; reward is exact match against the
                             record's keyed option. Nothing in the loop was
                             trained on a judge's output.
* ``critique``               open-ended answers, scored by the model being
                             trained acting as its own critic, with the
                             documented norm (and values, where the record has
                             them) as the evidence it judges against. No
                             external judge, so a gain cannot be attributed to
                             fitting another model's taste.
* ``critique+constrained``   both at once: every step draws half its prompts
                             from each task, and each prompt's group is scored
                             by its own task's reward.

GRPO is ``constrained``. Critique-GRPO is either critique mode: the critique
that produces the score is also fed back, and the weakest rollouts are rewritten
with it in context (``--refine-count``), so the critique shapes the data and not
only the number.

**Counterfactual switch** (``--counterfactual``). Off, the policy sees
``country_only`` alone. On, it also sees ``country_opposite_norm`` (the prompt
asserts a norm that is false for the country; the reward is for the country's
true behaviour), ``country_norm`` (the true norm), and ``country_value``.

``country_norm`` is there so the false-norm prompts cannot be solved by the rule
"always answer against the stated norm": with both present, whether a norm is
stated no longer predicts the answer, and only checking it against knowledge of
the country pays. The two are drawn from the same records in lockstep, one to
one. Their groups mostly carry no gradient while the model trusts true norms --
every rollout agrees -- and start to the moment it stops, which is exactly the
failure they guard against. Validation always reports ``country_norm`` so that
moment is visible.

Conditions without the country (``base``, ``value_only``, ``norm_only``) are
never trained on: their reward would not depend on knowledge of any country.
``country_value_norm`` is not trained on either; it adds nothing the pair above
does not already test.
"""

from __future__ import annotations

SCHEMA = "culturelens-finetune-v1"
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

FACTUAL_VIEWS: tuple[str, ...] = ("country_only",)
COUNTERFACTUAL_VIEWS: tuple[str, ...] = (
    "country_only",
    "country_value",
    "country_norm",
    "country_opposite_norm",
)
# Drawn from the same records in lockstep, so each false-norm prompt has its
# true-norm twin in the same step.
PAIRED_VIEWS: tuple[str, ...] = ("country_norm", "country_opposite_norm")
VALUE_VIEWS = frozenset({"country_value", "country_value_norm", "value_only"})

# Reported on validation whatever the recipe. country_norm is the guard against
# the counterfactual arm learning to reject every stated norm.
MONITOR_VIEWS: tuple[str, ...] = COUNTERFACTUAL_VIEWS

CONSTRAINED = "constrained"
CRITIQUE = "critique"
REWARDS: tuple[str, ...] = (CONSTRAINED, CRITIQUE, f"{CRITIQUE}+{CONSTRAINED}")
CRITICS: tuple[str, ...] = ("policy", "reference")


def training_views(counterfactual: bool) -> tuple[str, ...]:
    return COUNTERFACTUAL_VIEWS if counterfactual else FACTUAL_VIEWS


def reward_kinds(reward: str) -> tuple[str, ...]:
    """Which tasks a reward mode draws prompts from."""

    if reward not in REWARDS:
        raise ValueError(f"Unknown reward {reward!r}; expected one of {REWARDS}")
    return tuple(reward.split("+"))


def method_name(reward: str, counterfactual: bool) -> str:
    kinds = reward_kinds(reward)
    method = "critique-grpo" if CRITIQUE in kinds else "grpo"
    task = "-".join(kinds)
    return f"{method}-{task}-{'cf' if counterfactual else 'country'}"


__all__ = [
    "BASE_MODEL", "COUNTERFACTUAL_VIEWS", "CONSTRAINED", "CRITICS", "CRITIQUE",
    "FACTUAL_VIEWS", "MONITOR_VIEWS", "PAIRED_VIEWS", "REWARDS", "SCHEMA",
    "VALUE_VIEWS", "method_name", "reward_kinds", "training_views",
]
