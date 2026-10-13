"""Chat encoding that matches what generation saw, token for token.

Sampling and log-probability encoding both build their turn list here. A system
prompt present in one and absent from the other would make the importance ratio
a comparison of two different prompts. The system prompt is passed per call
because one step can mix the constrained task (which has one) and the open task
(which, like open-ended evaluation, has none).
"""

from __future__ import annotations

from typing import Any


def messages(prompt: str, system: str | None = None,
             response: str | None = None) -> list[dict[str, str]]:
    turns: list[dict[str, str]] = []
    if system:
        turns.append({"role": "system", "content": system})
    turns.append({"role": "user", "content": prompt})
    if response is not None:
        turns.append({"role": "assistant", "content": response})
    return turns


class ChatEncoder:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer
        self._cache: dict[tuple[str | None, str], tuple[str, list[int]]] = {}

    def render_prompt(self, prompt: str, system: str | None = None) -> str:
        return self._prompt(prompt, system)[0]

    def _prompt(self, prompt: str, system: str | None) -> tuple[str, list[int]]:
        key = (system, prompt)
        cached = self._cache.get(key)
        if cached is None:
            rendered = self.tokenizer.apply_chat_template(
                messages(prompt, system), tokenize=False, add_generation_prompt=True)
            ids = list(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])
            cached = (rendered, ids)
            self._cache[key] = cached
        return cached

    def encode(self, prompt: str, response: str, system: str | None = None) -> tuple[list[int], int]:
        """Ids for prompt plus answer, and the index where the answer begins."""

        rendered_prompt, prompt_ids = self._prompt(prompt, system)
        rendered = self.tokenizer.apply_chat_template(
            messages(prompt, system, response), tokenize=False)
        if not rendered.startswith(rendered_prompt):
            raise ValueError("Chat template does not extend the generation prompt")
        ids = list(self.tokenizer(rendered, add_special_tokens=False)["input_ids"])
        if ids[: len(prompt_ids)] != prompt_ids:
            raise ValueError("Prompt tokens change once the answer is appended")
        if len(ids) <= len(prompt_ids):
            raise ValueError("Encoded answer is empty")
        return ids, len(prompt_ids)


__all__ = ["ChatEncoder", "messages"]
