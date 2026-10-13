"""Raw Hugging Face hooks for decoder-block residual extraction.

The recorded state is the output of each decoder block, before the model's
final normalization, at the final non-padding token of the rendered prompt.
That token is the last token in the assistant-role prefix produced by
``add_generation_prompt=True`` and is therefore the state used to predict the
first generated token.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class LoadedModel:
    model: Any
    tokenizer: Any
    decoder_layers: Any
    layers: list[int]
    hidden_size: int


def require_torch():
    try:
        import torch
    except ImportError as error:  # pragma: no cover - GPU environment dependency
        raise RuntimeError(
            "Activation extraction requires PyTorch. Install the mechanistic extra."
        ) from error
    return torch


def resolve_layers(n_layers: int, requested: list[int] | None) -> list[int]:
    """Validate zero-indexed decoder-block indices."""

    if requested is None:
        return list(range(n_layers))
    layers = sorted(set(requested))
    invalid = [layer for layer in layers if not 0 <= layer < n_layers]
    if invalid:
        raise ValueError(
            f"Layer indices {invalid} are invalid for a {n_layers}-layer model"
        )
    return layers


def resolve_decoder_layers(model: Any) -> Any:
    """Locate the decoder stack in Llama, Gemma, and Qwen model wrappers."""

    torch = require_torch()
    candidates = (
        "model.layers",
        "model.language_model.layers",
        "language_model.layers",
        "transformer.h",
        "gpt_neox.layers",
    )
    for path in candidates:
        current = model
        for component in path.split("."):
            current = getattr(current, component, None)
            if current is None:
                break
        if isinstance(current, torch.nn.ModuleList):
            return current

    expected = None
    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
        getattr(getattr(model, "config", None), "language_config", None),
    ):
        if config is not None and hasattr(config, "num_hidden_layers"):
            expected = int(config.num_hidden_layers)
            break
    matches = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.ModuleList)
        and (expected is None or len(module) == expected)
    ]
    if len(matches) == 1:
        return matches[0][1]
    raise RuntimeError(
        "Could not uniquely locate decoder blocks; "
        f"expected {expected!r} layers and found {[name for name, _ in matches]}"
    )


def require_adapter_weights(model: Any, adapter: str) -> int:
    """How many adapter matrices of ``adapter`` were loaded into ``model``;
    raises when any is missing.

    PEFT matches saved weights to modules by name and skips, without failing,
    those it cannot place, which happens when a model is loaded under another
    class than it was trained with. A LoRA ``B`` matrix starts at zero, so one
    that is still zero after loading was not filled from the file.
    """

    from pathlib import Path

    loaded = sum(1 for name, value in model.named_parameters()
                 if "lora_B" in name and float(value.detach().abs().sum()) > 0)
    expected = None
    weights = Path(adapter) / "adapter_model.safetensors"
    if weights.is_file():
        from safetensors import safe_open

        with safe_open(str(weights), framework="pt") as handle:
            expected = sum(1 for key in handle.keys() if "lora_B" in key)
    if loaded == 0 or (expected is not None and loaded < expected):
        raise RuntimeError(f"Adapter {adapter} was not fully loaded: {loaded} of "
                           f"{expected if expected is not None else 'its'} matrices are in the model")
    return loaded


def attach_adapter(model: Any, adapter: str) -> Any:
    """``model`` with the LoRA ``adapter`` on it (not merged), every matrix loaded.

    An adapter trained on a multimodal class names its weights
    ``...language_model.layers...``; the same model loaded as a text-only class
    has ``...layers...``, and the other way round. When the plain load places
    nothing, the saved names are mapped to the loaded class and loaded again.
    """

    from pathlib import Path

    from peft import PeftModel

    wrapped = PeftModel.from_pretrained(model, adapter)
    try:
        require_adapter_weights(wrapped, adapter)
        return wrapped
    except RuntimeError:
        weights = Path(adapter) / "adapter_model.safetensors"
        if not weights.is_file():
            raise
    from peft.utils import set_peft_model_state_dict
    from safetensors.torch import load_file

    saved = load_file(str(weights))
    for old, new in ((".language_model.", "."), (".model.layers.", ".model.language_model.layers.")):
        set_peft_model_state_dict(wrapped, {key.replace(old, new): value for key, value in saved.items()})
        try:
            require_adapter_weights(wrapped, adapter)
            return wrapped
        except RuntimeError:
            continue
    raise RuntimeError(f"Adapter {adapter} does not fit this model under either naming of its layers")


def load_model(
    model_id: str,
    *,
    dtype: str,
    device: str,
    device_map: str,
    requested_layers: list[int] | None,
    trust_remote_code: bool,
    local_files_only: bool,
    attn_implementation: str | None,
    lora: str | None = None,
) -> LoadedModel:
    """Load a model and tokenizer for native-coordinate hooks. With ``lora``,
    that adapter is merged into the weights first, so every hook and mask sees
    the fine-tuned model as a plain one."""

    torch = require_torch()
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        try:
            from transformers import AutoModelForImageTextToText
        except ImportError:  # pragma: no cover - older Transformers
            AutoModelForImageTextToText = None
    except ImportError as error:  # pragma: no cover - GPU environment dependency
        raise RuntimeError(
            "Activation extraction requires Transformers and Accelerate. "
            "Install the mechanistic extra."
        ) from error

    dtype_map = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code,
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    kwargs: dict[str, Any] = {
        "torch_dtype": dtype_map[dtype],
        "low_cpu_mem_usage": True,
        "trust_remote_code": trust_remote_code,
        "local_files_only": local_files_only,
    }
    if device_map != "none":
        kwargs["device_map"] = device_map
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation

    loaders = [AutoModelForCausalLM]
    if AutoModelForImageTextToText is not None:
        loaders.append(AutoModelForImageTextToText)
    errors: list[str] = []
    model = None
    for loader in loaders:
        try:
            model = loader.from_pretrained(model_id, **kwargs)
            break
        except (TypeError, ValueError) as error:
            errors.append(f"{loader.__name__}: {error}")
    if model is None:
        raise RuntimeError("Unable to load model:\n" + "\n".join(errors))
    if lora:
        model = attach_adapter(model, lora).merge_and_unload()
    if device_map == "none":
        model = model.to(device)
    model.eval()

    decoder_layers = resolve_decoder_layers(model)
    layers = resolve_layers(len(decoder_layers), requested_layers)
    embedding = model.get_input_embeddings()
    hidden_size = int(embedding.weight.shape[1])
    return LoadedModel(model, tokenizer, decoder_layers, layers, hidden_size)


def _first_tensor(value: Any) -> Any | None:
    torch = require_torch()
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = _first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def extract_batch(
    loaded: LoadedModel,
    full_prompts: list[str],
    *,
    storage_dtype: str,
    max_length: int,
) -> tuple[Any, list[int], list[int]]:
    """Return states, unpadded token counts, and selected token IDs.

    States have shape ``(batch, requested_layers, hidden_size)``. Right padding
    and the attention mask are used so the selected position is always the
    final real prompt token rather than a padding position.
    """

    torch = require_torch()
    encoded = loaded.tokenizer(
        full_prompts,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    token_counts = encoded["attention_mask"].sum(dim=1)
    if int(token_counts.min()) < 1:
        raise RuntimeError("A rendered prompt tokenized to an empty sequence")
    if int(token_counts.max()) > max_length:
        raise RuntimeError(
            f"Prompt length {int(token_counts.max())} exceeds --max-length {max_length}; "
            "refusing to truncate the assistant-start position"
        )

    input_device = loaded.model.get_input_embeddings().weight.device
    input_ids = encoded["input_ids"].to(input_device)
    attention_mask = encoded["attention_mask"].to(input_device)
    batch_size = len(full_prompts)
    selected_ids = input_ids[
        torch.arange(batch_size, device=input_device),
        token_counts.to(input_device) - 1,
    ].detach().cpu().tolist()
    output_dtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[storage_dtype]

    captured: dict[int, Any] = {}
    handles = []

    def capture(layer_index: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            state = _first_tensor(output)
            if state is None or state.ndim != 3:
                raise RuntimeError(
                    f"Decoder layer {layer_index} did not return a (batch, tokens, hidden) tensor"
                )
            device = state.device
            rows = torch.arange(batch_size, device=device)
            positions = token_counts.to(device) - 1
            captured[layer_index] = (
                state[rows, positions, :]
                .detach()
                .to(device="cpu", dtype=output_dtype)
                .contiguous()
            )

        return hook

    try:
        for layer in loaded.layers:
            handles.append(
                loaded.decoder_layers[layer].register_forward_hook(capture(layer))
            )
        with torch.inference_mode():
            loaded.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=False,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [layer for layer in loaded.layers if layer not in captured]
    if missing:
        raise RuntimeError(f"Decoder hooks did not fire for layers {missing}")
    states = torch.stack([captured[layer] for layer in loaded.layers], dim=1)
    if tuple(states.shape) != (batch_size, len(loaded.layers), loaded.hidden_size):
        raise RuntimeError(f"Unexpected activation shape: {tuple(states.shape)}")
    if not bool(torch.isfinite(states.float()).all()):
        raise RuntimeError("Extracted activations contain NaN or infinity")
    return states, [int(value) for value in token_counts.tolist()], selected_ids


__all__ = [
    "LoadedModel",
    "extract_batch",
    "load_model",
    "require_torch",
    "resolve_decoder_layers",
    "resolve_layers",
]
