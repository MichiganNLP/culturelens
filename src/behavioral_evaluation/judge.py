"""Judge unconstrained CultureLens responses with one grounded LLM judge."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from common.io import load_prompt, read_json, read_jsonl, write_jsonl_atomic
from inference.vllm import (
    NO_THINKING_CHAT_TEMPLATE_KWARGS,
    VLLMConfig,
    generate_json_batch_partial,
    load_model,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROOT / "prompts/behavioral_evaluation/judge.txt"


class JudgeVerdict(BaseModel):
    reasoning: str
    alignment: Literal["aligned", "partially_aligned", "misaligned"]
    confidence: Literal["high", "medium", "low"]
    improvement: str | None = None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--judge-slug", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--response",
        action="append",
        required=True,
        metavar="SLUG=PATH",
        help="Response-model slug and unconstrained JSONL. Repeat for multiple models.",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=10,
                        help="Print progress every this many batches.")
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def read_dataset(path: Path) -> dict[str, dict[str, Any]]:
    rows = read_jsonl(path) if path.suffix.casefold() == ".jsonl" else read_json(path)
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path} must contain a list of records")
    mapped = {str(row["norm_id"]): row for row in rows}
    if len(mapped) != len(rows):
        raise ValueError("Dataset contains duplicate norm_id values")
    return mapped


def response_specs(values: list[str]) -> list[tuple[str, Path]]:
    specs: list[tuple[str, Path]] = []
    slugs: set[str] = set()
    for value in values:
        slug, separator, path = value.partition("=")
        if not separator or not slug.strip() or not path.strip():
            raise ValueError(f"Invalid --response {value!r}; expected SLUG=PATH")
        slug = slug.strip()
        if slug in slugs:
            raise ValueError(f"Duplicate response slug: {slug}")
        slugs.add(slug)
        specs.append((slug, Path(path).expanduser().resolve()))
    return specs


def country(record: dict[str, Any]) -> str:
    context = str(record["scenario"]["country_context"]).strip()
    match = re.fullmatch(r"This interaction takes place in (.+)\.", context)
    return match.group(1) if match else context


def judge_payload(record: dict[str, Any], response: str) -> str:
    return json.dumps(
        {
            "country": country(record),
            "norm_text": record["norm"],
            "schwartz": record.get("values") or [],
            "situation": record["scenario"]["situation"],
            "question": record["scenario"]["question"],
            "response": response,
        },
        ensure_ascii=False,
    )


def chunks(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index:index + size] for index in range(0, len(items), max(1, size))]


def successful(row: dict[str, Any]) -> bool:
    return (
        row.get("alignment") in {"aligned", "partially_aligned", "misaligned"}
        and row.get("confidence") in {"high", "medium", "low"}
        and isinstance(row.get("judge_reasoning"), str)
        and bool(row["judge_reasoning"].strip())
    )


def append_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def validate_responses(
    path: Path, dataset: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = read_jsonl(path, require_nonempty=True)
    seen: set[tuple[str, str]] = set()
    for row in rows:
        norm_id = str(row.get("norm_id", ""))
        condition = str(row.get("condition", ""))
        key = (norm_id, condition)
        if norm_id not in dataset:
            raise ValueError(f"Response references unknown norm_id: {norm_id}")
        if key in seen:
            raise ValueError(f"Duplicate response key in {path}: {key}")
        if not isinstance(row.get("response"), str) or not row["response"].strip():
            raise ValueError(f"Response is empty for {key}")
        seen.add(key)
    return rows


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.retries < 0:
        raise ValueError("--retries cannot be negative")
    dataset = read_dataset(args.dataset.expanduser().resolve())
    prompt = load_prompt(args.prompt.expanduser().resolve())
    specs = response_specs(args.response)
    output_root = args.output_root.expanduser().resolve()
    chat_kwargs = dict(NO_THINKING_CHAT_TEMPLATE_KWARGS) if args.no_think else None

    work: list[tuple[str, Path, Path, list[dict[str, Any]], list[dict[str, Any]]]] = []
    total_pending = 0
    for slug, response_path in specs:
        rows = validate_responses(response_path, dataset)
        output = output_root / slug / f"{args.judge_slug}.jsonl"
        existing: list[dict[str, Any]] = []
        if output.exists() and not args.overwrite:
            existing = [row for row in read_jsonl(output) if successful(row)]
        if args.overwrite:
            existing = []
        write_jsonl_atomic(existing, output)
        done = {(str(row["norm_id"]), str(row["condition"])) for row in existing}
        expected = {(str(row["norm_id"]), str(row["condition"])) for row in rows}
        if done - expected:
            raise ValueError(f"Existing judgments contain unexpected rows for {slug}")
        pending = [
            row for row in rows
            if (str(row["norm_id"]), str(row["condition"])) not in done
        ]
        total_pending += len(pending)
        work.append((slug, response_path, output, rows, pending))
        print(
            f"response_model={slug} expected={len(rows)} completed={len(done)} "
            f"pending={len(pending)}",
            flush=True,
        )
    if not total_pending:
        print("All judgments already complete", flush=True)
        return 0

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
        )
    )

    failures = 0
    for slug, _, output, rows, pending in work:
        completed_now = 0
        unresolved: list[dict[str, Any]] = []
        for batch_number, batch in enumerate(chunks(pending, args.batch_size), 1):
            remaining = list(batch)
            verdict_by_key: dict[tuple[str, str], JudgeVerdict] = {}
            for attempt in range(args.retries + 1):
                if not remaining:
                    break
                messages = [
                    [
                        {"role": "system", "content": prompt},
                        {
                            "role": "user",
                            "content": judge_payload(
                                dataset[str(row["norm_id"])], str(row["response"])
                            ),
                        },
                    ]
                    for row in remaining
                ]
                verdicts = generate_json_batch_partial(
                    llm,
                    messages,
                    output_schema=JudgeVerdict,
                    max_tokens=args.max_tokens,
                    temperature=0.0,
                    top_p=1.0,
                    seed=args.seed,
                    chat_template_kwargs=chat_kwargs,
                )
                retry: list[dict[str, Any]] = []
                for row, verdict in zip(remaining, verdicts, strict=True):
                    if verdict is None:
                        retry.append(row)
                    else:
                        verdict_by_key[(str(row["norm_id"]), str(row["condition"]))] = verdict
                remaining = retry
                if remaining:
                    print(
                        f"[{slug}] batch={batch_number} attempt={attempt + 1} "
                        f"retry={len(remaining)}",
                        file=sys.stderr,
                        flush=True,
                    )
            new_rows: list[dict[str, Any]] = []
            for row in batch:
                key = (str(row["norm_id"]), str(row["condition"]))
                verdict = verdict_by_key.get(key)
                if verdict is None:
                    unresolved.append(row)
                    continue
                new_rows.append(
                    {
                        "norm_id": row["norm_id"],
                        "iso3": row.get("iso3"),
                        "category": row.get("category"),
                        "condition": row["condition"],
                        "alignment": verdict.alignment,
                        "confidence": verdict.confidence,
                        "judge_reasoning": verdict.reasoning,
                        "improvement": verdict.improvement,
                    }
                )
            append_rows(output, new_rows)
            completed_now += len(new_rows)
            if (batch_number % max(1, args.progress_every) == 0
                    or batch_number * args.batch_size >= len(pending)):
                print(
                    f"[{slug}] processed={min(batch_number * args.batch_size, len(pending))}/"
                    f"{len(pending)} judged={completed_now} "
                    f"unresolved={len(unresolved)}",
                    flush=True,
                )
        failures += len(unresolved)
        if unresolved:
            print(f"[{slug}] unresolved={len(unresolved)}", file=sys.stderr, flush=True)
        else:
            print(f"[{slug}] complete output={output}", flush=True)

    if failures:
        raise RuntimeError(f"Judging incomplete; {failures} rows remain unresolved")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
