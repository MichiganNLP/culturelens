"""Shared set-up for the attention experiments: the sample, the model, the prompts."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np

from mechanistic_analysis.activations.backend import load_model
from mechanistic_analysis.activations.conditions import build_user_message
from mechanistic_analysis.activations.extract import read_records, render_prompt
from mechanistic_analysis.interventions.attention import prompt_pieces, token_spans, with_control


def add_model_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=400)
    parser.add_argument("--seed", type=int, default=2026,
                        help="Sample seed. 2026 is the confirmation sample; the exploratory "
                             "run that fixed the layer windows used 42.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--device-map", default="balanced")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--lora", default=None,
                        help="A LoRA adapter to merge into the model before the experiment.")


def sample_records(path: Path, size: int, seed: int) -> list[dict[str, Any]]:
    """A uniform sample of the bank, the same for every model and experiment."""

    records = read_records(path)
    chosen = np.sort(np.random.default_rng(seed).choice(
        len(records), size=min(size, len(records)), replace=False))
    return [records[index] for index in chosen]


def load(args: argparse.Namespace) -> Any:
    return load_model(args.model, dtype=args.dtype, device=args.device,
                      device_map=args.device_map, requested_layers=None,
                      trust_remote_code=args.trust_remote_code,
                      local_files_only=args.local_files_only, attn_implementation=None,
                      lora=getattr(args, "lora", None))


def thinking_flag(args: argparse.Namespace) -> bool | None:
    if args.thinking is None and "qwen" in args.model_slug.casefold():
        return False
    return args.thinking


def encode_prompt(loaded: Any, record: dict[str, Any], condition: str, thinking: bool | None
                  ) -> tuple[list[int], int, dict[str, tuple[int, int]]]:
    """Token ids of the rendered prompt, where the user's text ends, and the
    token range of each blockable part (claim, country, control)."""

    message = build_user_message(record, condition, include_options=False)[0]
    rendered = render_prompt(loaded, message, constrained=False, thinking=thinking)
    ids, after, spans = token_spans(loaded.tokenizer, rendered, message,
                                    prompt_pieces(record, condition))
    return ids, after, with_control(spans)
