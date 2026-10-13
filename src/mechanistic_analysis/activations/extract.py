"""Extract CultureLens assistant-start residual-stream activations.

For every requested decoder block, this command records the block output at
the final real token of the fully rendered prompt. The prompt is rendered with
``add_generation_prompt=True``; consequently this position is the end of the
assistant-role prefix and its residual state is the one used to predict the
first response token. It is not the first generated answer token.

The main analysis uses unconstrained prompts. A constrained variant is
available as a separate diagnostic. Each condition is written to a distinct,
self-describing ``.pt`` file with shape ``(records, layers, hidden_width)``.
Batch checkpoints make interrupted extraction resumable.

Example
-------

PYTHONPATH=src python -m mechanistic_analysis.activations.extract \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --model-slug llama-3.1-8b \
  --input data/processed/culturelens/records.json \
  --output-root output/mechanistic_analysis/activations
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from behavioral_evaluation.prompts import CONSTRAINED_SYSTEM_PROMPT
from mechanistic_analysis.activations import ACTIVATION_SITE, SCHEMA_VERSION
from mechanistic_analysis.activations.backend import (
    LoadedModel,
    extract_batch,
    load_model,
    require_torch,
)
from mechanistic_analysis.activations.cache import (
    atomic_torch_save,
    canonical_sha256,
    load_torch,
    sha256_file,
    sha256_strings,
    sha256_text,
    write_manifest,
)
from mechanistic_analysis.activations.conditions import (
    ACTIVATION_CONDITIONS,
    CLAIM_CONTROL_CONDITIONS,
    CONTROL_CONDITIONS,
    CORE_CONDITIONS,
    COUNTRY_CONDITIONS,
    GEOMETRY_CONTROL_CONDITIONS,
    SHUFFLED_COUNTRY_CONDITIONS,
    build_user_message,
    country_derangement,
    filter_records,
    record_metadata,
)


NO_THINKING_CHAT_TEMPLATE_KWARGS: dict[str, Any] = {"enable_thinking": False}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", required=True, help="Hugging Face model ID or path")
    parser.add_argument("--model-slug", required=True, help="Stable short model name")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/processed/culturelens/records.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/mechanistic_analysis/activations"),
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=ACTIVATION_CONDITIONS,
        default=None,
        help="Default: all eight evidence conditions plus both controls",
    )
    parser.add_argument(
        "--variant",
        choices=("unconstrained", "constrained"),
        default="unconstrained",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Zero-indexed decoder blocks; default extracts every block",
    )
    parser.add_argument(
        "--dtype",
        choices=("float32", "bfloat16", "float16"),
        default="bfloat16",
        help="Model and cache dtype",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--device-map",
        choices=("balanced", "auto", "none"),
        default="balanced",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--checkpoint-size",
        type=int,
        default=128,
        help="Number of records per resumable checkpoint shard",
    )
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--placebo-seed", type=int, default=42)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--attn-implementation",
        choices=("eager", "sdpa", "flash_attention_2"),
        default=None,
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Chat-template thinking mode; defaults off for Qwen and unchanged otherwise",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-checkpoints", action="store_true")
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.checkpoint_size < args.batch_size:
        parser.error("--checkpoint-size must be at least --batch-size")
    if args.max_length < 1:
        parser.error("--max-length must be positive")
    if args.variant == "constrained" and args.conditions:
        invalid = sorted(
            set(args.conditions)
            & {*CONTROL_CONDITIONS, *CLAIM_CONTROL_CONDITIONS, *GEOMETRY_CONTROL_CONDITIONS}
        )
        if invalid:
            parser.error(
                "Controls are registered for unconstrained extraction only: "
                + ", ".join(invalid)
            )
    return args


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.casefold() == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain a non-empty list of records")
    required = {"norm_id", "norm", "values", "scenario", "options"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Record {index} is missing {sorted(missing)}")
        record_metadata(row)
    norm_ids = [str(row["norm_id"]) for row in rows]
    if len(norm_ids) != len(set(norm_ids)):
        raise ValueError("Input contains duplicate norm_id values")
    return rows


def resolved_conditions(args: argparse.Namespace) -> tuple[str, ...]:
    if args.conditions:
        return tuple(args.conditions)
    if args.variant == "unconstrained":
        return (*CORE_CONDITIONS, *CONTROL_CONDITIONS)
    return CORE_CONDITIONS


def thinking_enabled(args: argparse.Namespace) -> bool | None:
    if args.thinking is not None:
        return bool(args.thinking)
    if "qwen" in args.model_slug.casefold():
        return False
    return None


def render_prompt(
    loaded: LoadedModel,
    user_message: str,
    *,
    constrained: bool,
    thinking: bool | None,
) -> str:
    messages: list[dict[str, str]] = []
    if constrained:
        messages.append({"role": "system", "content": CONSTRAINED_SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_message})
    kwargs: dict[str, Any] = {}
    if thinking is False:
        kwargs.update(NO_THINKING_CHAT_TEMPLATE_KWARGS)
    rendered = loaded.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        **kwargs,
    )
    if not isinstance(rendered, str) or not rendered:
        raise RuntimeError("Tokenizer returned an empty or non-text chat template")
    return rendered


def model_commit(model: Any) -> str | None:
    for config in (
        getattr(model, "config", None),
        getattr(getattr(model, "config", None), "text_config", None),
    ):
        value = getattr(config, "_commit_hash", None)
        if value:
            return str(value)
    return None


def prompt_iso3s(
    records: list[dict[str, Any]],
    condition: str,
    donor_by_iso3: dict[str, str],
) -> list[str | None]:
    output: list[str | None] = []
    for record in records:
        iso3, _ = record_metadata(record)
        if condition in SHUFFLED_COUNTRY_CONDITIONS:
            output.append(donor_by_iso3[iso3])
        elif condition in COUNTRY_CONDITIONS:
            output.append(iso3)
        else:
            output.append(None)
    return output


def base_metadata(
    *,
    args: argparse.Namespace,
    loaded: LoadedModel,
    input_path: Path,
    input_sha256: str,
    condition: str,
    records: list[dict[str, Any]],
    full_prompts: list[str],
    options_shuffled: list[bool],
    donor_by_iso3: dict[str, str],
    thinking: bool | None,
) -> dict[str, Any]:
    norm_ids = [str(record["norm_id"]) for record in records]
    metadata = [record_metadata(record) for record in records]
    chat_template = str(getattr(loaded.tokenizer, "chat_template", "") or "")
    return {
        "schema_version": SCHEMA_VERSION,
        "activation_site": ACTIVATION_SITE,
        "activation_definition": (
            "Output of each decoder block before final model normalization, selected "
            "at the final non-padding token of a chat template rendered with "
            "add_generation_prompt=True."
        ),
        "position_semantics": (
            "Last token of the assistant-role prefix; this state predicts the first "
            "generated response token and is not itself a generated answer token."
        ),
        "layer_indexing": "zero-indexed decoder blocks",
        "dataset_path": str(input_path),
        "dataset_sha256": input_sha256,
        "record_order": "input_file_order_after_condition_eligibility_filter",
        "record_order_sha256": sha256_strings(norm_ids),
        "model": args.model,
        "model_slug": args.model_slug,
        "model_commit": model_commit(loaded.model),
        "model_type": str(getattr(loaded.model.config, "model_type", "unknown")),
        "tokenizer": str(getattr(loaded.tokenizer, "name_or_path", args.model)),
        "tokenizer_commit": getattr(loaded.tokenizer, "_commit_hash", None),
        "chat_template_sha256": sha256_text(chat_template),
        "condition": condition,
        "variant": args.variant,
        "thinking": thinking,
        "options": args.variant == "constrained",
        "option_shuffle": "md5(norm_id) parity" if args.variant == "constrained" else None,
        "requested_dtype": args.dtype,
        "device_map": args.device_map,
        "placebo_seed": (
            args.placebo_seed if condition in SHUFFLED_COUNTRY_CONDITIONS else None
        ),
        "layers": loaded.layers,
        "hidden_size": loaded.hidden_size,
        "norm_ids": norm_ids,
        "iso3s": [iso3 for iso3, _ in metadata],
        "categories": [category for _, category in metadata],
        "prompt_iso3s": prompt_iso3s(records, condition, donor_by_iso3),
        "options_shuffled": options_shuffled,
        "full_prompt_sha256s": [sha256_text(prompt) for prompt in full_prompts],
        "software_versions": {
            "python": platform.python_version(),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "accelerate": package_version("accelerate"),
        },
    }


def signature_fields(metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        key: metadata[key]
        for key in (
            "schema_version",
            "activation_site",
            "dataset_sha256",
            "record_order_sha256",
            "model",
            "model_slug",
            "model_commit",
            "tokenizer",
            "tokenizer_commit",
            "chat_template_sha256",
            "condition",
            "variant",
            "thinking",
            "requested_dtype",
            "layers",
            "hidden_size",
            "norm_ids",
            "prompt_iso3s",
            "full_prompt_sha256s",
        )
    }


def checkpoint_path(parts_dir: Path, start: int, end: int) -> Path:
    return parts_dir / f"{start:08d}-{end:08d}.pt"


def validate_checkpoint(
    payload: dict[str, Any],
    *,
    signature: str,
    start: int,
    end: int,
    norm_ids: list[str],
    n_layers: int,
    hidden_size: int,
) -> None:
    torch = require_torch()
    if payload.get("run_signature") != signature:
        raise ValueError("Checkpoint belongs to a different extraction run")
    if payload.get("start") != start or payload.get("end") != end:
        raise ValueError("Checkpoint range does not match its filename")
    if payload.get("norm_ids") != norm_ids[start:end]:
        raise ValueError("Checkpoint norm_ids do not match the current dataset")
    tensor = payload.get("assistant_start")
    expected = (end - start, n_layers, hidden_size)
    if not torch.is_tensor(tensor) or tuple(tensor.shape) != expected:
        raise ValueError(
            f"Checkpoint tensor shape is {getattr(tensor, 'shape', None)}, expected {expected}"
        )
    if len(payload.get("token_counts", [])) != end - start:
        raise ValueError("Checkpoint token_counts length mismatch")
    if len(payload.get("selected_token_ids", [])) != end - start:
        raise ValueError("Checkpoint selected_token_ids length mismatch")


def existing_output_matches(path: Path, signature: str) -> bool:
    try:
        payload = load_torch(path, mmap=True)
    except Exception as error:
        raise ValueError(f"Cannot validate existing cache {path}: {error}") from error
    return payload.get("run_signature") == signature


def extract_condition(
    *,
    args: argparse.Namespace,
    loaded: LoadedModel,
    all_records: list[dict[str, Any]],
    input_path: Path,
    input_sha256: str,
    condition: str,
    thinking: bool | None,
) -> Path:
    torch = require_torch()
    records = filter_records(all_records, condition)
    if not records:
        raise ValueError(f"No eligible records for condition {condition}")

    shuffled_contexts: dict[str, str] | None = None
    donor_by_iso3: dict[str, str] = {}
    if condition in SHUFFLED_COUNTRY_CONDITIONS:
        shuffled_contexts, donor_by_iso3 = country_derangement(
            records, args.placebo_seed
        )

    built = [
        build_user_message(
            record,
            condition,
            args.variant == "constrained",
            shuffled_context_by_iso3=shuffled_contexts,
        )
        for record in records
    ]
    full_prompts = [
        render_prompt(
            loaded,
            user_message,
            constrained=args.variant == "constrained",
            thinking=thinking,
        )
        for user_message, _ in built
    ]
    metadata = base_metadata(
        args=args,
        loaded=loaded,
        input_path=input_path,
        input_sha256=input_sha256,
        condition=condition,
        records=records,
        full_prompts=full_prompts,
        options_shuffled=[shuffled for _, shuffled in built],
        donor_by_iso3=donor_by_iso3,
        thinking=thinking,
    )
    signature = canonical_sha256(signature_fields(metadata))
    metadata["run_signature"] = signature

    output = (
        args.output_root.expanduser().resolve()
        / args.variant
        / args.model_slug
        / f"{condition}.pt"
    )
    parts_dir = output.with_suffix(output.suffix + ".parts")
    manifest_path = parts_dir / "manifest.json"

    if args.overwrite:
        output.unlink(missing_ok=True)
        shutil.rmtree(parts_dir, ignore_errors=True)
    elif output.exists():
        if existing_output_matches(output, signature):
            print(f"[{condition}] valid cache already exists: {output}", flush=True)
            return output
        raise ValueError(
            f"Existing cache does not match this run: {output}. "
            "Use --overwrite or a different output root."
        )

    parts_dir.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("run_signature") != signature:
            raise ValueError(
                f"Checkpoint manifest does not match this run: {manifest_path}. "
                "Use --overwrite or a different output root."
            )
    else:
        write_manifest(
            {
                "schema_version": SCHEMA_VERSION,
                "run_signature": signature,
                "condition": condition,
                "records": len(records),
                "layers": loaded.layers,
                "hidden_size": loaded.hidden_size,
                "checkpoint_size": args.checkpoint_size,
            },
            manifest_path,
        )

    part_paths: list[Path] = []
    for start in range(0, len(records), args.checkpoint_size):
        end = min(start + args.checkpoint_size, len(records))
        part = checkpoint_path(parts_dir, start, end)
        part_paths.append(part)
        if part.exists():
            payload = load_torch(part)
            validate_checkpoint(
                payload,
                signature=signature,
                start=start,
                end=end,
                norm_ids=metadata["norm_ids"],
                n_layers=len(loaded.layers),
                hidden_size=loaded.hidden_size,
            )
            print(f"[{condition}] resumed {end}/{len(records)}", flush=True)
            continue

        tensors = []
        token_counts: list[int] = []
        selected_token_ids: list[int] = []
        for batch_start in range(start, end, args.batch_size):
            batch_end = min(batch_start + args.batch_size, end)
            states, counts, selected = extract_batch(
                loaded,
                full_prompts[batch_start:batch_end],
                storage_dtype=args.dtype,
                max_length=args.max_length,
            )
            tensors.append(states)
            token_counts.extend(counts)
            selected_token_ids.extend(selected)
        tensor = torch.cat(tensors, dim=0)
        part_payload = {
            "run_signature": signature,
            "start": start,
            "end": end,
            "norm_ids": metadata["norm_ids"][start:end],
            "token_counts": token_counts,
            "selected_token_ids": selected_token_ids,
            "assistant_start": tensor,
        }
        validate_checkpoint(
            part_payload,
            signature=signature,
            start=start,
            end=end,
            norm_ids=metadata["norm_ids"],
            n_layers=len(loaded.layers),
            hidden_size=loaded.hidden_size,
        )
        atomic_torch_save(part_payload, part)
        print(f"[{condition}] extracted {end}/{len(records)}", flush=True)

    tensors = []
    token_counts = []
    selected_token_ids = []
    for part in part_paths:
        payload = load_torch(part)
        tensors.append(payload["assistant_start"])
        token_counts.extend(payload["token_counts"])
        selected_token_ids.extend(payload["selected_token_ids"])
    assistant_start = torch.cat(tensors, dim=0)
    expected_shape = (len(records), len(loaded.layers), loaded.hidden_size)
    if tuple(assistant_start.shape) != expected_shape:
        raise RuntimeError(
            f"Final activation shape is {tuple(assistant_start.shape)}, expected {expected_shape}"
        )
    if len(token_counts) != len(records) or len(selected_token_ids) != len(records):
        raise RuntimeError("Final token metadata does not match the record count")

    selected_tokens = loaded.tokenizer.convert_ids_to_tokens(selected_token_ids)
    final_payload = {
        **metadata,
        "tensor_shape": list(expected_shape),
        "prompt_token_counts": token_counts,
        "selected_token_indices": [count - 1 for count in token_counts],
        "selected_token_ids": selected_token_ids,
        "selected_tokens": [str(token) for token in selected_tokens],
        "assistant_start": assistant_start,
    }
    atomic_torch_save(final_payload, output)
    if not args.keep_checkpoints:
        shutil.rmtree(parts_dir)
    print(
        f"[{condition}] saved {output} shape={expected_shape} dtype={assistant_start.dtype}",
        flush=True,
    )
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    input_path = args.input.expanduser().resolve()
    all_records = read_records(input_path)
    input_sha256 = sha256_file(input_path)
    conditions = resolved_conditions(args)
    thinking = thinking_enabled(args)

    print(
        f"model={args.model} slug={args.model_slug} variant={args.variant} "
        f"conditions={','.join(conditions)} records={len(all_records)} "
        f"thinking={thinking}",
        flush=True,
    )
    loaded = load_model(
        args.model,
        dtype=args.dtype,
        device=args.device,
        device_map=args.device_map,
        requested_layers=args.layers,
        trust_remote_code=args.trust_remote_code,
        local_files_only=args.local_files_only,
        attn_implementation=args.attn_implementation,
    )
    print(
        f"loaded decoder_layers={len(loaded.decoder_layers)} "
        f"selected_layers={loaded.layers} hidden_size={loaded.hidden_size}",
        flush=True,
    )
    for condition in conditions:
        extract_condition(
            args=args,
            loaded=loaded,
            all_records=all_records,
            input_path=input_path,
            input_sha256=input_sha256,
            condition=condition,
            thinking=thinking,
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
