"""Loader for the three pretrained sparse-autoencoder suites, one per model.

All three consume the raw output of a decoder block (``resid_post``), which is
what the activation cache stores. The loader checks checkpoint metadata and
tensor shapes against the registry; it never runs the language model.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from mechanistic_analysis.sae.registry import ModelSAESpec

DTYPES = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
}


@dataclass(frozen=True, slots=True)
class SAEMetadata:
    provider: str
    repo: str
    source_id: str
    source_model: str
    layer: int
    d_model: int
    d_sae: int
    activation: str
    dtype: str
    k: int | None = None
    threshold: float | None = None
    expected_l0: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MatrixSAE(nn.Module):
    """Canonical matrices: encoder ``(d_sae,d_model)``, decoder ``(d_model,d_sae)``."""

    def __init__(
        self,
        metadata: SAEMetadata,
        w_enc: torch.Tensor,
        b_enc: torch.Tensor,
        w_dec: torch.Tensor,
        b_dec: torch.Tensor,
        *,
        threshold: float | torch.Tensor | None = None,
        k: int | None = None,
        subtract_decoder_bias: bool = False,
    ) -> None:
        super().__init__()
        expected = {
            "w_enc": (metadata.d_sae, metadata.d_model),
            "b_enc": (metadata.d_sae,),
            "w_dec": (metadata.d_model, metadata.d_sae),
            "b_dec": (metadata.d_model,),
        }
        actual = {
            "w_enc": tuple(w_enc.shape),
            "b_enc": tuple(b_enc.shape),
            "w_dec": tuple(w_dec.shape),
            "b_dec": tuple(b_dec.shape),
        }
        if actual != expected:
            raise ValueError(f"Invalid SAE matrix shapes: expected={expected}, actual={actual}")
        self.metadata = metadata
        self.k = k
        self.subtract_decoder_bias = subtract_decoder_bias
        self.register_buffer("w_enc", w_enc.contiguous())
        self.register_buffer("b_enc", b_enc.contiguous())
        self.register_buffer("w_dec", w_dec.contiguous())
        self.register_buffer("b_dec", b_dec.contiguous())
        value = (
            None
            if threshold is None
            else torch.as_tensor(threshold, device=w_enc.device, dtype=w_enc.dtype)
        )
        self.register_buffer("threshold", value)

    @property
    def d_model(self) -> int:
        return self.metadata.d_model

    @property
    def d_sae(self) -> int:
        return self.metadata.d_sae

    def encode(self, activations: torch.Tensor) -> torch.Tensor:
        if activations.shape[-1] != self.d_model:
            raise ValueError(f"SAE expects width {self.d_model}, got {activations.shape[-1]}")
        x = activations.to(self.w_enc)
        if self.subtract_decoder_bias:
            x = x - self.b_dec
        pre = F.linear(x, self.w_enc, self.b_enc)
        if self.metadata.activation == "topk":
            if self.k is None:
                raise RuntimeError("TopK SAE has no k")
            values, indices = pre.topk(self.k, dim=-1)
            return torch.zeros_like(pre).scatter(-1, indices, values)
        if self.threshold is None:
            raise RuntimeError("JumpReLU SAE has no threshold")
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))

    def decode(self, features: torch.Tensor) -> torch.Tensor:
        if features.shape[-1] != self.d_sae:
            raise ValueError(f"SAE expects {self.d_sae} features")
        return F.linear(features.to(self.w_dec), self.w_dec, self.b_dec)

    def decoder_directions(self, feature_ids: torch.Tensor) -> torch.Tensor:
        """Return unit-free residual directions as ``(features,d_model)``."""

        return self.w_dec[:, feature_ids.to(self.w_dec.device)].T


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    if not isinstance(result, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return result


def _torch_state(path: Path) -> Mapping[str, torch.Tensor]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        state = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(state, Mapping):
        raise TypeError(f"Expected tensor dictionary: {path}")
    return state


def _require(state: Mapping[str, Any], keys: set[str], label: str) -> None:
    missing = sorted(keys - set(state))
    if missing:
        raise ValueError(f"{label} checkpoint missing {missing}")
    bad = sorted(key for key in keys if not torch.is_tensor(state[key]))
    if bad:
        raise ValueError(f"{label} non-tensor entries: {bad}")


def _resolve(root: Path | None, spec: ModelSAESpec, filename: str) -> Path:
    if root is not None:
        path = root.expanduser() / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing SAE checkpoint file: {path}")
        return path
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as error:
        raise RuntimeError("huggingface_hub or safetensors is not installed") from error
    return Path(hf_hub_download(repo_id=spec.default_repo, filename=filename))


def _load_llama(
    spec: ModelSAESpec, layer: int, root: Path | None, device: str, dtype: str
) -> MatrixSAE:
    source = f"resid_post_layer_{layer}/trainer_0"
    config = _read_json(_resolve(root, spec, f"{source}/config.json"))
    trainer = config.get("trainer")
    if not isinstance(trainer, Mapping) or trainer.get("trainer_class") != "BatchTopKTrainer":
        raise ValueError("Unexpected Llama Instruct SAE configuration")
    if int(trainer.get("layer", -1)) != layer or trainer.get("lm_name") != spec.source_model:
        raise ValueError("Llama SAE metadata disagrees with the locked pairing")
    state = _torch_state(_resolve(root, spec, f"{source}/ae.pt"))
    _require(
        state,
        {"encoder.weight", "encoder.bias", "decoder.weight", "b_dec", "k", "threshold"},
        "Llama",
    )
    target = DTYPES[dtype]
    dev = torch.device(device)
    d_model = int(trainer["activation_dim"])
    d_sae = int(trainer["dict_size"])
    threshold = float(state["threshold"].item())
    metadata = SAEMetadata(
        spec.provider,
        str(root or spec.default_repo),
        source,
        spec.source_model,
        layer,
        d_model,
        d_sae,
        "jumprelu",
        dtype,
        k=int(state["k"].item()),
        threshold=threshold,
    )
    return MatrixSAE(
        metadata,
        state["encoder.weight"].to(device=dev, dtype=target),
        state["encoder.bias"].to(device=dev, dtype=target),
        state["decoder.weight"].to(device=dev, dtype=target),
        state["b_dec"].to(device=dev, dtype=target),
        threshold=threshold,
        subtract_decoder_bias=True,
    ).eval()


def _load_gemma(
    spec: ModelSAESpec, layer: int, root: Path | None, device: str, dtype: str
) -> MatrixSAE:
    if dtype != "float32":
        raise ValueError("Gemma Scope JumpReLU inference is locked to float32")
    source = f"resid_post_all/layer_{layer}_width_16k_l0_big"
    config = _read_json(_resolve(root, spec, f"{source}/config.json"))
    if config.get("architecture") != "jump_relu":
        raise ValueError("Unexpected Gemma Scope architecture")
    try:
        from safetensors.torch import load_file
    except ImportError as error:
        raise RuntimeError("huggingface_hub or safetensors is not installed") from error
    state = load_file(str(_resolve(root, spec, f"{source}/params.safetensors")), device="cpu")
    _require(state, {"w_enc", "b_enc", "w_dec", "b_dec", "threshold"}, "Gemma")
    dev, target = torch.device(device), DTYPES[dtype]
    d_model, d_sae = int(state["b_dec"].numel()), int(state["b_enc"].numel())
    metadata = SAEMetadata(
        spec.provider,
        str(root or spec.default_repo),
        source,
        spec.source_model,
        layer,
        d_model,
        d_sae,
        "jumprelu",
        dtype,
        threshold=float(state["threshold"].float().mean()),
        expected_l0=float(config["l0"]),
    )
    return MatrixSAE(
        metadata,
        state["w_enc"].T.to(device=dev, dtype=target),
        state["b_enc"].to(device=dev, dtype=target),
        state["w_dec"].T.to(device=dev, dtype=target),
        state["b_dec"].to(device=dev, dtype=target),
        threshold=state["threshold"].to(device=dev, dtype=target),
    ).eval()


def _load_qwen(
    spec: ModelSAESpec, layer: int, root: Path | None, device: str, dtype: str
) -> MatrixSAE:
    config = _read_json(_resolve(root, spec, "config.json"))
    if config.get("hook_point") != "resid_post" or layer not in set(map(int, config["layers"])):
        raise ValueError("Qwen SAE metadata disagrees with the locked pairing")
    state = _torch_state(_resolve(root, spec, f"layer{layer}.sae.pt"))
    _require(state, {"W_enc", "b_enc", "W_dec", "b_dec"}, "Qwen")
    dev, target = torch.device(device), DTYPES[dtype]
    metadata = SAEMetadata(
        spec.provider,
        str(root or spec.default_repo),
        f"layer{layer}.sae.pt",
        spec.source_model,
        layer,
        int(config["d_model"]),
        int(config["d_sae"]),
        "topk",
        dtype,
        k=int(config["k"]),
    )
    return MatrixSAE(
        metadata,
        state["W_enc"].to(device=dev, dtype=target),
        state["b_enc"].to(device=dev, dtype=target),
        state["W_dec"].to(device=dev, dtype=target),
        state["b_dec"].to(device=dev, dtype=target),
        k=metadata.k,
    ).eval()


def load_sae(
    spec: ModelSAESpec,
    layer: int,
    *,
    root: Path | None,
    device: str,
    dtype: str | None = None,
) -> MatrixSAE:
    if layer not in spec.layers:
        raise ValueError(f"No locked {spec.provider} checkpoint at layer {layer}")
    resolved_dtype = dtype or spec.inference_dtype
    if resolved_dtype not in DTYPES:
        raise ValueError(f"Unsupported SAE dtype {resolved_dtype!r}")
    if spec.provider == "llama_instruct_batchtopk":
        return _load_llama(spec, layer, root, device, resolved_dtype)
    if spec.provider == "gemma_scope":
        return _load_gemma(spec, layer, root, device, resolved_dtype)
    if spec.provider == "qwen_scope":
        return _load_qwen(spec, layer, root, device, resolved_dtype)
    raise ValueError(f"Unsupported provider {spec.provider!r}")


__all__ = ["MatrixSAE", "SAEMetadata", "load_sae"]
