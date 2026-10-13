"""Set one sparse-autoencoder feature to a chosen value while the model answers.

``steered`` edits the residual stream at one layer for every position from a
row's ``start`` onward: the chat-template tail and each generated token. At
each edited position the feature's activation is read with the SAE's encoder,
and the difference between the wanted value and that activation is added along
the feature's decoder direction. ``mode="ablate"`` wants 0, so the feature's
contribution is removed and nothing else changes; ``mode="clamp"`` wants
``value``. The residual is never replaced by the SAE's reconstruction.
"""

from __future__ import annotations

import contextlib
from typing import Any, Iterator

from mechanistic_analysis.activations.backend import _first_tensor, require_torch


@contextlib.contextmanager
def steered(decoder_layers: Any, layer: int, sae: Any, feature: int, mode: str, value: float,
            start: Any, width: int) -> Iterator[None]:
    """``start`` is (rows,), in the positions of the full prompt of ``width`` tokens.

    The first pass of generation covers the whole prompt and is edited from
    ``start``; later passes carry only new tokens and are edited everywhere.
    """

    if mode not in ("ablate", "clamp"):
        raise ValueError(f"Unknown mode {mode!r}")
    torch = require_torch()
    target = 0.0 if mode == "ablate" else float(value)

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        state = _first_tensor(output)
        if sae.w_enc.device != state.device:
            sae.to(state.device)
        positions = torch.arange(state.shape[1], device=state.device)
        if state.shape[1] >= width:
            chosen = positions[None, :] >= start.to(state.device)[:, None]
        else:
            chosen = torch.ones(state.shape[:2], dtype=torch.bool, device=state.device)
        picked = state[chosen]
        activation = sae.encode(picked)[:, feature]
        direction = sae.w_dec[:, feature]
        edit = (target - activation.float())[:, None] * direction.float()[None, :]
        state[chosen] = picked + edit.to(state.dtype)

    handle = decoder_layers[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


__all__ = ["steered"]
