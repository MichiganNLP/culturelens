"""Generate constrained and unconstrained CultureLens responses with vLLM."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from behavioral_evaluation.prompts import (
    ALL_SEGMENTS,
    CONDITION_SEGMENTS,
    CONSTRAINED_SYSTEM_PROMPT,
    build_user_message,
)
from common.io import read_json, read_jsonl, write_jsonl_atomic
from inference.vllm import (
    NO_THINKING_CHAT_TEMPLATE_KWARGS,
    VLLMConfig,
    generate_batch,
    generate_choice_batch,
    load_model,
)


_REASONING_RE = re.compile(r"<reasoning>(.*?)</reasoning>", re.IGNORECASE | re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>\s*([AB])\s*</answer>", re.IGNORECASE)
_TERMINAL_ANSWER_RES = (
    re.compile(r"(?:^|\n)\s*(?:final\s+)?answer\s*[:=-]\s*([AB])[.)]?\s*$", re.I),
    re.compile(r"\\boxed\{\s*([AB])\s*\}\s*$", re.I),
    re.compile(r"(?:^|\n)\s*([AB])[.)]?\s*$", re.I),
    re.compile(r"(?:^|\n)\s*(?:option|choice)\s*[:=-]?\s*([AB])[.)]?\s*$", re.I),
)

_ANSWER_ONLY_SYSTEM_PROMPT = (
    "Select the better option for the cultural scenario. Return exactly one "
    "character, A or B, with no reasoning or additional text."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", choices=list(ALL_SEGMENTS),
                        default=list(CONDITION_SEGMENTS),
                        help="Default: the benchmark's eight conditions. Diagnostic "
                             "conditions (the false norm alone or reframed) run only when named.")
    parser.add_argument("--options", action="store_true",
                        help="Run constrained A/B evaluation instead of open generation.")
    parser.add_argument("--answer-only-retry", action="store_true",
                        help="Recover unparseable constrained outputs with finite-choice decoding.")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--lora", type=Path, default=None,
                        help="LoRA adapter directory to generate with, e.g. a fine-tuned run's best/.")
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_records(path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(path) if path.suffix.casefold() == ".jsonl" else read_json(path)
    if not isinstance(rows, list) or not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain a non-empty list of records")
    required = {"norm_id", "norm", "values", "scenario", "options"}
    for index, row in enumerate(rows):
        missing = required - set(row)
        if missing:
            raise ValueError(f"Record {index} is missing {sorted(missing)}")
    identifiers = [str(row["norm_id"]) for row in rows]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Input contains duplicate norm_id values")
    return rows


def metadata(norm_id: str) -> tuple[str, str]:
    parts = norm_id.split("__")
    if len(parts) < 2:
        raise ValueError(f"Malformed norm_id: {norm_id}")
    return parts[0], parts[1]


def parse_choice(text: str) -> tuple[str | None, str | None]:
    reasoning_match = _REASONING_RE.search(text)
    answer_match = _ANSWER_RE.search(text)
    reasoning = reasoning_match.group(1).strip() if reasoning_match else None
    if answer_match:
        return reasoning, answer_match.group(1).casefold()
    stripped = text.strip()
    for pattern in _TERMINAL_ANSWER_RES:
        match = pattern.search(stripped)
        if match:
            return reasoning, match.group(1).casefold()
    return reasoning, None


def eligible(record: dict[str, Any], condition: str) -> bool:
    if condition not in {"value_only", "country_value", "country_value_norm"}:
        return True
    return record["scenario"].get("value_evidence") is not None


def successful(row: dict[str, Any], constrained: bool) -> bool:
    if constrained:
        return row.get("pred") in {"a", "b"}
    return isinstance(row.get("response"), str) and bool(row["response"].strip())


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), max(1, size))]


def make_row(
    record: dict[str, Any],
    condition: str,
    user_message: str,
    response: str,
    shuffled: bool,
    constrained: bool,
) -> dict[str, Any]:
    iso3, category = metadata(str(record["norm_id"]))
    row: dict[str, Any] = {
        "norm_id": record["norm_id"],
        "iso3": iso3,
        "category": category,
        "condition": condition,
        "user_message": user_message,
        "response": response,
    }
    if constrained:
        reasoning, prediction = parse_choice(response)
        gold = "b" if shuffled else "a"
        row.update(
            reasoning=reasoning,
            pred=prediction,
            gold=gold,
            correct=(prediction == gold) if prediction else None,
            shuffled=shuffled,
        )
    return row


def generate_plain(
    llm: Any,
    messages: list[list[dict[str, str]]],
    *,
    max_tokens: int,
    seed: int,
    chat_template_kwargs: dict[str, Any] | None,
    lora_request: Any | None = None,
) -> list[str]:
    try:
        return generate_batch(
            llm,
            messages,
            max_tokens=max_tokens,
            temperature=0.0,
            top_p=1.0,
            seed=seed,
            chat_template_kwargs=chat_template_kwargs,
            lora_request=lora_request,
        )
    except Exception as batch_error:
        print(f"Batch generation failed; retrying individually: {batch_error}", file=sys.stderr)
        outputs: list[str] = []
        for message in messages:
            try:
                outputs.extend(
                    generate_batch(
                        llm,
                        [message],
                        max_tokens=max_tokens,
                        temperature=0.0,
                        top_p=1.0,
                        seed=seed,
                        chat_template_kwargs=chat_template_kwargs,
                        lora_request=lora_request,
                    )
                )
            except Exception as error:
                print(f"Individual generation failed: {error}", file=sys.stderr)
                outputs.append("")
        return outputs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.answer_only_retry and not args.options:
        raise ValueError("--answer-only-retry requires --options")
    records = read_records(args.input.expanduser().resolve())
    output = args.output.expanduser().resolve()

    existing: list[dict[str, Any]] = []
    if output.exists() and not args.overwrite:
        existing = read_jsonl(output)
        existing = [row for row in existing if successful(row, args.options)]
    if args.overwrite:
        existing = []
    write_jsonl_atomic(existing, output)
    done = {(str(row["norm_id"]), str(row["condition"])) for row in existing}

    expected = {
        (str(record["norm_id"]), condition)
        for record in records
        for condition in args.conditions
        if eligible(record, condition)
    }
    if done - expected:
        raise ValueError("Existing output contains rows outside the requested evaluation")
    pending_total = len(expected - done)
    print(
        f"model={args.model} mode={'constrained' if args.options else 'unconstrained'} "
        f"records={len(records)} expected={len(expected)} completed={len(done)} "
        f"pending={pending_total}",
        flush=True,
    )
    if not pending_total:
        return 0

    chat_kwargs = dict(NO_THINKING_CHAT_TEMPLATE_KWARGS) if args.no_think else None
    llm = load_model(
        VLLMConfig(
            model=args.model,
            dtype=args.dtype,
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            disable_custom_all_reduce=args.disable_custom_all_reduce,
            trust_remote_code=args.trust_remote_code,
            enforce_eager=args.enforce_eager,
            language_model_only=args.language_model_only,
            seed=args.seed,
            enable_lora=args.lora is not None,
            max_lora_rank=args.max_lora_rank if args.lora is not None else None,
        )
    )
    lora_request = None
    if args.lora is not None:
        adapter = args.lora.expanduser().resolve()
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"{adapter} is not a LoRA adapter directory")
        from vllm.lora.request import LoRARequest

        lora_request = LoRARequest("finetuned", 1, str(adapter))

    all_rows = list(existing)
    for condition in args.conditions:
        pending = [
            record
            for record in records
            if eligible(record, condition)
            and (str(record["norm_id"]), condition) not in done
        ]
        print(f"[{condition}] pending={len(pending)}", flush=True)
        for batch_number, batch in enumerate(chunks(pending, args.batch_size), 1):
            built = [build_user_message(record, condition, args.options) for record in batch]
            conversations = []
            for user_message, _ in built:
                conversation: list[dict[str, str]] = []
                if args.options:
                    conversation.append({"role": "system", "content": CONSTRAINED_SYSTEM_PROMPT})
                conversation.append({"role": "user", "content": user_message})
                conversations.append(conversation)
            responses = generate_plain(
                llm,
                conversations,
                max_tokens=args.max_tokens,
                seed=args.seed,
                chat_template_kwargs=chat_kwargs,
                lora_request=lora_request,
            )
            new_rows = [
                make_row(record, condition, built[index][0], responses[index], built[index][1], args.options)
                for index, record in enumerate(batch)
                if responses[index].strip()
            ]
            append_rows(output, new_rows)
            all_rows.extend(new_rows)
            done.update((str(row["norm_id"]), condition) for row in new_rows if successful(row, args.options))
            if batch_number % 10 == 0 or batch_number * args.batch_size >= len(pending):
                print(
                    f"[{condition}] processed={min(batch_number * args.batch_size, len(pending))}/"
                    f"{len(pending)} parseable={sum(successful(row, args.options) for row in new_rows)}",
                    flush=True,
                )

    if args.options and args.answer_only_retry:
        latest = {(str(row["norm_id"]), str(row["condition"])): row for row in all_rows}
        unresolved = expected - {
            key for key, row in latest.items() if successful(row, constrained=True)
        }
        print(f"answer-only recovery rows={len(unresolved)}", flush=True)
        recovered: dict[tuple[str, str], dict[str, Any]] = {}
        records_by_id = {str(record["norm_id"]): record for record in records}
        for condition in args.conditions:
            keys = sorted(key for key in unresolved if key[1] == condition)
            for batch in chunks(keys, args.batch_size):
                source = [records_by_id[norm_id] for norm_id, _ in batch]
                built = [build_user_message(record, condition, True) for record in source]
                messages = [
                    [
                        {"role": "system", "content": _ANSWER_ONLY_SYSTEM_PROMPT},
                        {"role": "user", "content": user_message},
                    ]
                    for user_message, _ in built
                ]
                answers = generate_choice_batch(
                    llm,
                    messages,
                    choices=("A", "B"),
                    max_tokens=8,
                    temperature=0.0,
                    top_p=1.0,
                    seed=args.seed,
                    chat_template_kwargs=chat_kwargs,
                    lora_request=lora_request,
                )
                for index, record in enumerate(source):
                    row = make_row(
                        record,
                        condition,
                        built[index][0],
                        answers[index],
                        built[index][1],
                        True,
                    )
                    row["reasoning"] = None
                    row["pred"] = answers[index].casefold()
                    row["correct"] = row["pred"] == row["gold"]
                    recovered[(str(record["norm_id"]), condition)] = row
        latest.update(recovered)
        all_rows = [latest[key] for key in sorted(expected, key=lambda item: (args.conditions.index(item[1]), item[0]))]
        write_jsonl_atomic(all_rows, output)

    final_rows = read_jsonl(output)
    final = {(str(row["norm_id"]), str(row["condition"])): row for row in final_rows}
    missing = expected - {key for key, row in final.items() if successful(row, args.options)}
    if missing:
        raise RuntimeError(f"Evaluation is incomplete; {len(missing)} rows remain unresolved")
    print(f"complete rows={len(expected)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
