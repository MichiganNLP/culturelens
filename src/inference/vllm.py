"""Generic, in-process, batched inference through vLLM's Python API.

This module is intentionally free of project domain knowledge: no cultural
norms, countries, WVS/Schwartz/IWM labels, prompts, or data files. It provides:

1. loading one local/Hugging Face model;
2. batched chat generation;
3. batched JSON-schema-constrained generation;
4. Pydantic validation of structured responses.

vLLM manages the tokenizer and chat template; this wrapper never applies chat
templates manually and never mutates the caller's conversations.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, TypedDict, TypeVar

from pydantic import BaseModel, ValidationError


def _vllm_types() -> tuple[type[Any], type[Any], type[Any]]:
    """Import vLLM only when inference is actually requested.

    Dataset inspection, schema validation, and CLI ``--help`` should work on a
    laptop without the GPU-only vLLM dependency installed.
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:  # pragma: no cover - depends on runtime image
        raise RuntimeError(
            "vLLM is required for model inference. Install CultureLens with "
            "the 'generation' extra in a CUDA environment."
        ) from error
    return LLM, SamplingParams, StructuredOutputsParams


# --------------------------------------------------------------------------- #
# Typing (static clarity only; no runtime message schema)
# --------------------------------------------------------------------------- #
class ChatMessage(TypedDict):
    role: Literal["system", "user", "assistant"]
    content: str


Conversation = list[ChatMessage]
T = TypeVar("T", bound=BaseModel)

_VALID_ROLES = frozenset({"system", "user", "assistant"})
_MAX_SNIPPET = 300

#: chat_template_kwargs that disable a model's built-in reasoning/"thinking"
#: pass. This is the offline-``LLM.chat`` equivalent of the server's
#: ``--chat-template-kwargs '{"enable_thinking": false}'`` and is honoured by
#: Qwen3-style chat templates. Pass it (merged with any other template kwargs)
#: through the ``chat_template_kwargs`` argument of :func:`generate_batch` /
#: :func:`generate_json_batch`; the backend itself stays model-agnostic.
NO_THINKING_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class VLLMBackendError(RuntimeError):
    """Base error raised by the local vLLM backend."""


class VLLMGenerationError(VLLMBackendError):
    """vLLM returned no usable generation for a request."""


class VLLMValidationError(VLLMBackendError):
    """A structured generation failed Pydantic validation."""


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class VLLMConfig:
    model: str
    dtype: str = "auto"
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: int | None = None
    max_num_batched_tokens: int | None = None
    gpu_memory_utilization: float | None = None
    kv_cache_memory_bytes: int | None = None
    disable_custom_all_reduce: bool = False
    disable_custom_ops: bool = False
    disable_norm_quant_fusion: bool = False
    trust_remote_code: bool = False
    enforce_eager: bool = False
    language_model_only: bool = False
    seed: int = 42
    enable_lora: bool = False
    max_lora_rank: int | None = None

    def __post_init__(self) -> None:
        if not self.model or not self.model.strip():
            raise ValueError("VLLMConfig.model must be a non-empty model identifier.")
        if self.tensor_parallel_size < 1:
            raise ValueError("VLLMConfig.tensor_parallel_size must be >= 1.")
        if self.pipeline_parallel_size < 1:
            raise ValueError("VLLMConfig.pipeline_parallel_size must be >= 1.")
        if self.max_model_len is not None and self.max_model_len < 1:
            raise ValueError("VLLMConfig.max_model_len must be >= 1 when provided.")
        if self.max_num_batched_tokens is not None and self.max_num_batched_tokens < 1:
            raise ValueError(
                "VLLMConfig.max_num_batched_tokens must be >= 1 when provided."
            )
        if self.gpu_memory_utilization is not None and not (
            0.0 < self.gpu_memory_utilization <= 1.0
        ):
            raise ValueError(
                "VLLMConfig.gpu_memory_utilization must be in (0, 1] when provided."
            )
        if self.kv_cache_memory_bytes is not None and self.kv_cache_memory_bytes < 1:
            raise ValueError(
                "VLLMConfig.kv_cache_memory_bytes must be >= 1 when provided."
            )
        if self.max_lora_rank is not None and self.max_lora_rank < 1:
            raise ValueError("VLLMConfig.max_lora_rank must be >= 1 when provided.")


# --------------------------------------------------------------------------- #
# Model loading
# --------------------------------------------------------------------------- #
def load_model(config: VLLMConfig) -> Any:
    """Construct and return a vLLM ``LLM`` engine from ``config`` (no tokenizer)."""
    kwargs: dict[str, Any] = {
        "model": config.model,
        "dtype": config.dtype,
        "tensor_parallel_size": config.tensor_parallel_size,
        "disable_custom_all_reduce": config.disable_custom_all_reduce,
        "trust_remote_code": config.trust_remote_code,
        "enforce_eager": config.enforce_eager,
        "seed": config.seed,
    }
    # These options were added after the vLLM release used by the project's
    # validated container image.  Only pass them when the caller actually
    # requests a non-default value, keeping the backend compatible with both
    # the locked image and newer local installations.
    if config.pipeline_parallel_size != 1:
        kwargs["pipeline_parallel_size"] = config.pipeline_parallel_size
    if config.language_model_only:
        kwargs["language_model_only"] = True
    if config.max_model_len is not None:
        kwargs["max_model_len"] = config.max_model_len
    if config.max_num_batched_tokens is not None:
        kwargs["max_num_batched_tokens"] = config.max_num_batched_tokens
    if config.gpu_memory_utilization is not None:
        kwargs["gpu_memory_utilization"] = config.gpu_memory_utilization
    if config.kv_cache_memory_bytes is not None:
        kwargs["kv_cache_memory_bytes"] = config.kv_cache_memory_bytes
    if config.enable_lora:
        kwargs["enable_lora"] = True
    if config.max_lora_rank is not None:
        kwargs["max_lora_rank"] = config.max_lora_rank
    if config.disable_custom_ops or config.disable_norm_quant_fusion:
        # Use native PyTorch implementations instead of vLLM CUDA custom ops.
        # This preserves vLLM scheduling/KV caching/structured decoding while
        # avoiding architecture-specific kernels that may not support Ampere.
        compilation_config: dict[str, Any] = {}
        if config.disable_custom_ops:
            compilation_config["custom_ops"] = ["none"]
        if config.disable_norm_quant_fusion:
            # This fusion remains enabled by default even with custom_ops=none.
            # Gemma-4 has heterogeneous attention head dimensions, and its
            # vLLM profiling pass has segfaulted in this fusion on Ampere GPUs.
            compilation_config["pass_config"] = {"fuse_norm_quant": False}
        kwargs["compilation_config"] = compilation_config
    LLM, _, _ = _vllm_types()
    return LLM(**kwargs)


# --------------------------------------------------------------------------- #
# Input validation (private helpers; never mutate caller inputs)
# --------------------------------------------------------------------------- #
def _validate_messages_batch(
    messages_batch: Sequence[Sequence[ChatMessage]],
) -> None:
    if len(messages_batch) == 0:
        raise ValueError("messages_batch must contain at least one conversation.")
    for conversation_index, conversation in enumerate(messages_batch):
        if len(conversation) == 0:
            raise ValueError(f"Conversation at index {conversation_index} is empty.")
        for message_index, message in enumerate(conversation):
            if not isinstance(message, dict):
                raise TypeError(
                    f"Message at conversation {conversation_index}, position "
                    f"{message_index} must be a mapping with 'role' and 'content'."
                )
            role = message.get("role")
            content = message.get("content")
            if role not in _VALID_ROLES:
                raise ValueError(
                    f"Unsupported role {role!r} at conversation {conversation_index}, "
                    f"message {message_index}; expected one of {sorted(_VALID_ROLES)}."
                )
            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Message content at conversation {conversation_index}, message "
                    f"{message_index} must be a non-blank string."
                )


def _validate_sampling(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    if max_tokens < 1:
        raise ValueError("max_tokens must be >= 1.")
    if temperature < 0:
        raise ValueError("temperature must be >= 0.")
    if not (0 < top_p <= 1):
        raise ValueError("top_p must be in (0, 1].")


def _build_sampling_params(
    *,
    max_tokens: int,
    temperature: float,
    top_p: float,
    seed: int | None,
    stop: str | list[str] | None,
    structured_outputs: Any | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "n": 1,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "stop": stop,
    }
    # Omit seed when None rather than relying on a particular vLLM version
    # accepting seed=None.
    if seed is not None:
        kwargs["seed"] = seed
    if structured_outputs is not None:
        kwargs["structured_outputs"] = structured_outputs
    _, SamplingParams, _ = _vllm_types()
    return SamplingParams(**kwargs)


def _extract_text(output: Any, batch_index: int) -> str:
    candidates = getattr(output, "outputs", None)
    if not candidates:
        raise VLLMGenerationError(
            f"vLLM returned no candidate output for batch index {batch_index}."
        )
    text = candidates[0].text
    if not isinstance(text, str) or not text.strip():
        raise VLLMGenerationError(
            f"vLLM returned an empty generation for batch index {batch_index}."
        )
    return text


# --------------------------------------------------------------------------- #
# Plain batched chat generation
# --------------------------------------------------------------------------- #
def generate_batch(
    llm: Any,
    messages_batch: Sequence[Sequence[ChatMessage]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = 42,
    stop: str | list[str] | None = None,
    use_tqdm: bool = False,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    lora_request: Any | None = None,
) -> list[str]:
    """Generate one chat completion per conversation, in input order.

    ``temperature == 0.0`` is deterministic/greedy; ``> 0.0`` samples. The
    caller's conversations are never mutated and assistant turns are never
    appended.
    """
    _validate_messages_batch(messages_batch)
    _validate_sampling(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    sampling_params = _build_sampling_params(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop=stop,
    )
    chat_kwargs: dict[str, Any] = {
        "messages": [list(conversation) for conversation in messages_batch],
        "sampling_params": sampling_params,
        "use_tqdm": use_tqdm,
        "chat_template": chat_template,
        "chat_template_kwargs": chat_template_kwargs,
    }
    if lora_request is not None:
        chat_kwargs["lora_request"] = lora_request
    outputs = llm.chat(**chat_kwargs)
    return [_extract_text(output, index) for index, output in enumerate(outputs)]


def generate_choice_batch(
    llm: Any,
    messages_batch: Sequence[Sequence[ChatMessage]],
    *,
    choices: Sequence[str],
    max_tokens: int = 8,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = 42,
    use_tqdm: bool = False,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
    lora_request: Any | None = None,
) -> list[str]:
    """Generate exactly one member of ``choices`` for each conversation.

    This is intended for classification-style recovery where free generation
    would introduce an avoidable formatting failure.  vLLM enforces the
    finite choice set during decoding; the returned strings are additionally
    checked so a backend/version mismatch cannot silently enter an audit.
    """

    _validate_messages_batch(messages_batch)
    _validate_sampling(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    normalized = tuple(str(choice).strip() for choice in choices)
    if not normalized or any(not choice for choice in normalized):
        raise ValueError("choices must contain at least one non-blank string")
    if len(set(normalized)) != len(normalized):
        raise ValueError("choices must be unique")

    _, _, StructuredOutputsParams = _vllm_types()
    structured_outputs = StructuredOutputsParams(choice=list(normalized))
    sampling_params = _build_sampling_params(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop=None,
        structured_outputs=structured_outputs,
    )
    chat_kwargs: dict[str, Any] = {
        "messages": [list(conversation) for conversation in messages_batch],
        "sampling_params": sampling_params,
        "use_tqdm": use_tqdm,
        "chat_template": chat_template,
        "chat_template_kwargs": chat_template_kwargs,
    }
    if lora_request is not None:
        chat_kwargs["lora_request"] = lora_request
    outputs = llm.chat(**chat_kwargs)
    results = [_extract_text(output, index).strip() for index, output in enumerate(outputs)]
    invalid = [value for value in results if value not in normalized]
    if invalid:
        raise VLLMGenerationError(
            f"Choice-constrained generation returned invalid values: {invalid[:3]!r}"
        )
    return results


# --------------------------------------------------------------------------- #
# Structured (JSON-schema-constrained) batched generation
# --------------------------------------------------------------------------- #
def generate_json_batch(
    llm: Any,
    messages_batch: Sequence[Sequence[ChatMessage]],
    *,
    output_schema: type[T],
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = 42,
    stop: str | list[str] | None = None,
    use_tqdm: bool = False,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[T]:
    """Generate and Pydantic-validate one ``output_schema`` object per conversation.

    Returns the parsed objects in input order. A response that fails validation
    raises :class:`VLLMValidationError` carrying the batch index, schema name, a
    truncated snippet, and the original Pydantic error as its cause.
    """
    _validate_messages_batch(messages_batch)
    _validate_sampling(max_tokens=max_tokens, temperature=temperature, top_p=top_p)

    _, _, StructuredOutputsParams = _vllm_types()
    structured_outputs = StructuredOutputsParams(json=output_schema.model_json_schema())
    sampling_params = _build_sampling_params(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop=stop,
        structured_outputs=structured_outputs,
    )
    outputs = llm.chat(
        messages=[list(conversation) for conversation in messages_batch],
        sampling_params=sampling_params,
        use_tqdm=use_tqdm,
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
    )

    results: list[T] = []
    for index, output in enumerate(outputs):
        content = _extract_text(output, index)
        try:
            results.append(output_schema.model_validate_json(content))
        except ValidationError as error:
            snippet = content[:_MAX_SNIPPET]
            raise VLLMValidationError(
                f"Structured generation at batch index {index} failed validation "
                f"for schema {output_schema.__name__}. Snippet: {snippet!r}"
            ) from error
    return results


def generate_json_batch_partial(
    llm: Any,
    messages_batch: Sequence[Sequence[ChatMessage]],
    *,
    output_schema: type[T],
    max_tokens: int = 512,
    temperature: float = 0.0,
    top_p: float = 1.0,
    seed: int | None = 42,
    stop: str | list[str] | None = None,
    use_tqdm: bool = False,
    chat_template: str | None = None,
    chat_template_kwargs: dict[str, Any] | None = None,
) -> list[T | None]:
    """Generate a structured batch while preserving individually valid outputs.

    Unlike :func:`generate_json_batch`, malformed or truncated responses become
    ``None`` without discarding valid peers from the same expensive model call.
    This is intended for resumable offline drafting, where failed rows are
    explicitly retained for retry or human completion.
    """
    _validate_messages_batch(messages_batch)
    _validate_sampling(max_tokens=max_tokens, temperature=temperature, top_p=top_p)
    _, _, StructuredOutputsParams = _vllm_types()
    structured_outputs = StructuredOutputsParams(json=output_schema.model_json_schema())
    sampling_params = _build_sampling_params(
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        stop=stop,
        structured_outputs=structured_outputs,
    )
    outputs = llm.chat(
        messages=[list(conversation) for conversation in messages_batch],
        sampling_params=sampling_params,
        use_tqdm=use_tqdm,
        chat_template=chat_template,
        chat_template_kwargs=chat_template_kwargs,
    )
    results: list[T | None] = []
    for index, output in enumerate(outputs):
        try:
            content = _extract_text(output, index)
            results.append(output_schema.model_validate_json(content))
        except (VLLMBackendError, ValidationError):
            results.append(None)
    return results


__all__ = [
    "NO_THINKING_CHAT_TEMPLATE_KWARGS",
    "ChatMessage",
    "Conversation",
    "VLLMBackendError",
    "VLLMConfig",
    "VLLMGenerationError",
    "VLLMValidationError",
    "generate_batch",
    "generate_choice_batch",
    "generate_json_batch",
    "generate_json_batch_partial",
    "load_model",
]
