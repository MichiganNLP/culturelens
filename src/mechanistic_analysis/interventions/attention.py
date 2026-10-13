"""Cutting attention from the answer stream to one part of the prompt.

``blocked`` stops every position at or after a row's ``query_start`` from
attending to that row's marked key positions, in the chosen decoder layers,
for as long as the context is open. Positions before ``query_start`` are
untouched, so the prompt is processed normally up to the end of the user's
text; only the chat-template tail and the answer lose sight of the span.

It works for a full pass (queries and keys cover the same positions) and for
cached decoding (a step's queries are the newest positions): the mask a layer
receives is ``(rows, 1, queries, keys)`` and the queries are always the last
``queries`` of the ``keys`` positions.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from mechanistic_analysis.activations.backend import require_torch


def attention_module(layer: Any) -> Any | None:
    """The layer's maskable self-attention, or None when it has none (hybrid
    models mix tokens in some layers with a recurrent block that takes no mask)."""

    for name in ("self_attn", "attention", "attn"):
        module = getattr(layer, name, None)
        if module is not None:
            return module
    return None


def maskable_layers(decoder_layers: Any) -> list[int]:
    return [index for index, layer in enumerate(decoder_layers)
            if attention_module(layer) is not None]


def head_count(module: Any) -> int:
    for owner in (module, getattr(module, "config", None)):
        for name in ("num_heads", "num_attention_heads"):
            value = getattr(owner, name, None)
            if isinstance(value, int):
                return value
    raise RuntimeError("Could not read the number of attention heads of a layer")


@contextlib.contextmanager
def blocked(decoder_layers: Any, layers: list[int], query_start: Any, keys: Any,
            heads: dict[int, list[int]] | None = None) -> Iterator[None]:
    """``query_start`` is (rows,) and ``keys`` is (rows, positions), both in the
    coordinates of the padded input the model receives. With ``heads`` (layer ->
    head indices) only those heads of those layers are cut and ``layers`` is
    ignored."""

    torch = require_torch()

    def hook(module: Any, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
        hidden = kwargs.get("hidden_states", args[0] if args else None)
        rows, queries = hidden.shape[:2]
        mask = kwargs.get("attention_mask")
        if mask is None:
            if kwargs.get("past_key_values") is not None and queries < keys.shape[1]:
                raise RuntimeError(
                    "The layer received no attention mask during cached decoding; give every "
                    "row at least one padding position so the model builds one")
            mask = torch.zeros(rows, 1, queries, queries, dtype=hidden.dtype, device=hidden.device)
            mask.masked_fill_(torch.ones(queries, queries, dtype=torch.bool,
                                         device=hidden.device).triu(1), torch.finfo(hidden.dtype).min)
        else:
            mask = mask.expand(rows, -1, -1, -1).clone()
        n_queries, n_keys = mask.shape[2], mask.shape[3]
        positions = n_keys - n_queries + torch.arange(n_queries, device=mask.device)
        marked = torch.zeros(rows, n_keys, dtype=torch.bool, device=mask.device)
        width = min(n_keys, keys.shape[1])
        marked[:, :width] = keys.to(mask.device)[:, :width]
        cut = ((positions[None, :, None] >= query_start.to(mask.device)[:, None, None])
               & marked[:, None, :])
        off = False if mask.dtype == torch.bool else torch.finfo(mask.dtype).min
        chosen = selected.get(id(module))
        if chosen is None:
            mask[:, 0][cut] = off
        else:
            # One mask per head, so only the chosen heads lose sight of the span.
            mask = mask.expand(-1, head_count(module), -1, -1).clone()
            for head in chosen:
                mask[:, head][cut] = off
        kwargs["attention_mask"] = mask
        return args, kwargs

    if heads is not None:
        layers = sorted(heads)
    modules = {index: attention_module(decoder_layers[index]) for index in layers}
    selected = {id(module): heads[index] for index, module in modules.items()
                if heads is not None and module is not None}
    handles = [module.register_forward_pre_hook(hook, with_kwargs=True)
               for module in modules.values() if module is not None]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def token_spans(tokenizer: Any, prompt: str, user_message: str, pieces: dict[str, str]
                ) -> tuple[list[int], int, dict[str, tuple[int, int]]]:
    """Token ids of ``prompt``, the index of the first token after the user's
    text, and the token range [start, end) of each named piece of that text."""

    encoded = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    ids, offsets = list(encoded["input_ids"]), list(encoded["offset_mapping"])
    begin = prompt.rfind(user_message)
    if begin < 0:
        raise ValueError("The rendered prompt does not contain the user message verbatim")
    end = begin + len(user_message)
    after = next((index for index, (start, _) in enumerate(offsets) if start >= end), len(ids))
    spans = {}
    for name, text in pieces.items():
        at = prompt.find(text, begin)
        if at < 0:
            raise ValueError(f"The prompt does not contain its {name}")
        inside = [index for index, (start, stop) in enumerate(offsets)
                  if start < at + len(text) and stop > at]
        spans[name] = (inside[0], inside[-1] + 1)
    return ids, after, spans


def prompt_pieces(record: dict[str, Any], condition: str) -> dict[str, str]:
    """The parts of a condition's user message that can be blocked.

    ``control`` (as many tokens as the claim, from the start of the situation)
    is added by the caller once token ranges are known.
    """

    scenario = record["scenario"]
    pieces = {"situation": str(scenario["situation"])}
    if condition in ("country_only", "country_norm", "country_opposite_norm"):
        pieces["country"] = str(scenario["country_context"])
    if condition in ("norm_only", "country_norm"):
        pieces["claim"] = str(scenario["norm_evidence"])
    if condition in ("opposite_norm_only", "country_opposite_norm"):
        pieces["claim"] = str(scenario["opposite_norm_evidence"])
    return pieces


def with_control(spans: dict[str, tuple[int, int]]) -> dict[str, tuple[int, int]]:
    if "claim" in spans:
        length = spans["claim"][1] - spans["claim"][0]
        first, last = spans["situation"]
        spans = {**spans, "control": (first, min(first + length, last))}
    return spans
