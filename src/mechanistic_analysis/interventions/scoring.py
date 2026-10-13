"""A teacher-forced decision score.

The margin for a prompt is

    M = log P(reference answer | prompt) - log P(contrast answer | prompt)

where each term sums the model's own token log-probabilities along that answer
(option A and option B of the record; A is the reference). It is a fast proxy
used to localise an effect across layers; conclusions about behaviour rest on
generated answers and their judgments (``generate.py``).
"""

from __future__ import annotations

from typing import Any

from mechanistic_analysis.activations.backend import require_torch


class BatchScorer:
    """Scores ``continuations[r][k]`` after ``prompts[r]``; each row is run in full."""

    def __init__(self, model: Any, pad_id: int, prompts: list[list[int]],
                 continuations: list[list[list[int]]]) -> None:
        torch = require_torch()
        self.model = model
        self.device = model.get_input_embeddings().weight.device
        self.records, self.per_record = len(prompts), len(continuations[0])
        rows = [(prompt, answer) for prompt, options in zip(prompts, continuations)
                for answer in options]
        if any(not answer for _, answer in rows) or len(rows) != self.records * self.per_record:
            raise ValueError("Every record needs the same number of non-empty continuations")
        longest = max(len(answer) for _, answer in rows)
        width = max(len(prompt) + len(answer) - 1 for prompt, answer in rows)
        self.inputs = torch.full((len(rows), width), pad_id, dtype=torch.long)
        self.attention = torch.zeros((len(rows), width), dtype=torch.long)
        self.targets = torch.zeros((len(rows), longest), dtype=torch.long)
        self.target_mask = torch.zeros((len(rows), longest))
        for row, (prompt, answer) in enumerate(rows):
            tokens = prompt + answer[:-1]
            self.inputs[row, :len(tokens)] = torch.tensor(tokens)
            self.attention[row, :len(tokens)] = 1
            self.targets[row, :len(answer)] = torch.tensor(answer)
            self.target_mask[row, :len(answer)] = 1
        self.position = torch.tensor([len(prompt) - 1 for prompt, _ in rows])
        self.width = width

    def score(self) -> Any:
        """(records, continuations): summed log-probability of each continuation."""

        torch = require_torch()
        with torch.no_grad():
            logits = self.model(input_ids=self.inputs.to(self.device),
                                attention_mask=self.attention.to(self.device),
                                use_cache=False, return_dict=True).logits
        steps = self.targets.shape[1]
        index = (self.position[:, None] + torch.arange(steps)[None, :]).clamp_max(self.width - 1)
        rows = torch.arange(len(index))[:, None]
        picked = logits[rows.to(logits.device), index.to(logits.device)].float()
        logprob = torch.log_softmax(picked, dim=-1).gather(
            2, self.targets.to(logits.device)[:, :, None])[:, :, 0].cpu()
        return (logprob * self.target_mask).sum(dim=1).reshape(self.records, self.per_record)


def margin(scores: Any) -> Any:
    """Reference minus contrast; ``scores`` is (..., 2)."""

    return scores[..., 0] - scores[..., 1]
