"""The GRPO objective. Torch is imported inside each function so the rest of
the package, and its tests, need no GPU stack.

Ported unchanged from the v1 package, where it trained stably.
"""

from __future__ import annotations


def token_logprobs(logits, input_ids, prompt_lengths, attention_mask):
    """Per-token log-probabilities of the answer, and the mask selecting them.

    Position j predicts token j+1, so the first answer token is predicted at
    ``prompt_length - 1``.
    """

    import torch
    import torch.nn.functional as F

    targets = input_ids[:, 1:]
    positions = torch.arange(targets.shape[1], device=targets.device).unsqueeze(0)
    answer_mask = (positions >= (prompt_lengths.unsqueeze(1) - 1)) & attention_mask[:, 1:].bool()
    values = -F.cross_entropy(logits[:, :-1, :].float().transpose(1, 2), targets, reduction="none")
    return values.masked_fill(~answer_mask, 0.0), answer_mask


def grpo_loss(policy_logprobs, old_logprobs, reference_logprobs, advantages, answer_mask, *,
              clip_epsilon: float, kl_weight: float):
    """Token-level clipped policy loss with a KL penalty to the reference.

    The KL term is the estimator ``exp(d) - d - 1`` for ``d = log pi_ref - log
    pi``: it cannot go negative, so it cannot be reduced by drifting in either
    direction. Each sequence's loss is its token mean, so a long answer does not
    outweigh a short one with the same advantage.
    """

    import torch

    counts = answer_mask.sum(-1).clamp_min(1)
    ratio = (policy_logprobs - old_logprobs).exp()
    weights = advantages.unsqueeze(1)
    unclipped = ratio * weights
    clipped = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon) * weights
    objective = torch.minimum(unclipped, clipped)
    difference = reference_logprobs - policy_logprobs
    kl = difference.exp() - difference - 1.0
    per_token = -(objective - float(kl_weight) * kl)
    per_token = per_token.masked_fill(~answer_mask, 0.0)
    losses = per_token.sum(-1) / counts
    return losses, {
        "kl": (kl.masked_fill(~answer_mask, 0.0).sum(-1) / counts).detach(),
        "clip_fraction": (((unclipped > clipped) & answer_mask).sum(-1) / counts).detach().float(),
    }


__all__ = ["grpo_loss", "token_logprobs"]
