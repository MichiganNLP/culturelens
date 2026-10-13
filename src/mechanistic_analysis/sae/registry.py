"""Which pretrained sparse autoencoder goes with which model, and at which layers.

The pairings are the ones validated in the first CultureLens study: the Llama
Instruct BatchTopK suite (seven layers), Gemma Scope 2 at width 16k (every
layer) and the Qwen residual suite at width 80k (every layer). Feature ids
mean something only inside one checkpoint and layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ModelSAESpec:
    slug: str
    model_id: str
    provider: str
    layers: tuple[int, ...]
    n_model_layers: int
    source_model: str
    default_repo: str
    inference_dtype: str

    def normalized_depth(self, layer: int) -> float:
        return (layer + 1) / self.n_model_layers


MODEL_SAE_REGISTRY: dict[str, ModelSAESpec] = {
    "llama-3.1-8b": ModelSAESpec(
        slug="llama-3.1-8b",
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        provider="llama_instruct_batchtopk",
        layers=(3, 7, 11, 15, 19, 23, 27),
        n_model_layers=32,
        source_model="meta-llama/Llama-3.1-8B-Instruct",
        default_repo="andyrdt/saes-llama-3.1-8b-instruct",
        inference_dtype="float32",
    ),
    "gemma-3-12b": ModelSAESpec(
        slug="gemma-3-12b",
        model_id="google/gemma-3-12b-it",
        provider="gemma_scope",
        layers=tuple(range(48)),
        n_model_layers=48,
        source_model="google/gemma-3-12b-it",
        default_repo="google/gemma-scope-2-12b-it",
        inference_dtype="float32",
    ),
    "qwen-3.5-27b": ModelSAESpec(
        slug="qwen-3.5-27b",
        model_id="Qwen/Qwen3.5-27B",
        provider="qwen_scope",
        layers=tuple(range(64)),
        n_model_layers=64,
        source_model="Qwen/Qwen3.5-27B",
        default_repo="Qwen/SAE-Res-Qwen3.5-27B-W80K-L0_50",
        inference_dtype="bfloat16",
    ),
}


def get_spec(model: str) -> ModelSAESpec:
    try:
        return MODEL_SAE_REGISTRY[model]
    except KeyError as error:
        raise ValueError(
            f"Unknown model {model!r}; choose from {sorted(MODEL_SAE_REGISTRY)}"
        ) from error


def resolve_layers(spec: ModelSAESpec, value: str) -> tuple[int, ...]:
    if value.strip().lower() in {"all", "default"}:
        return spec.layers
    layers = tuple(sorted({int(part) for part in value.split(",") if part.strip()}))
    if not layers:
        raise ValueError("No SAE layers selected")
    unavailable = sorted(set(layers) - set(spec.layers))
    if unavailable:
        raise ValueError(
            f"{spec.provider} has no checkpoint for layers {unavailable}; "
            f"available={list(spec.layers)}"
        )
    return layers


__all__ = ["MODEL_SAE_REGISTRY", "ModelSAESpec", "get_spec", "resolve_layers"]
