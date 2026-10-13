"""Code how each open-ended answer treats a stated (false) norm.

A second pass over answers already written, separate from the alignment judge:
the coder sees the claim that was (or would have been) stated in the prompt and
the answer, and says whether the answer follows the claim, follows it with
doubt, challenges it, or sets it aside. It never sees the documented norm and
does not grade correctness, so the coding is independent of the alignment
label it is later crossed with.

PYTHONPATH=src python -m behavioral_evaluation.stance \
  --model <judge model> --coder-slug gemma-4-31b \
  --dataset data/processed/culturelens/records.json \
  --response llama-3.1-8b=output/behavioral_evaluation/unconstrained/responses/llama-3.1-8b.jsonl \
  --conditions country_opposite_norm \
  --output-root output/behavioral_evaluation/unconstrained/stance
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

from behavioral_evaluation.judge import (
    append_rows,
    chunks,
    country,
    read_dataset,
    response_specs,
    validate_responses,
)
from common.io import load_prompt, read_jsonl, write_jsonl_atomic

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROOT / "prompts/behavioral_evaluation/stance.txt"
STANCES = ("follows", "follows_with_caveat", "challenges", "sets_aside", "no_clear_recommendation")


class StanceCoding(BaseModel):
    evidence: str
    stance: Literal["follows", "follows_with_caveat", "challenges", "sets_aside",
                    "no_clear_recommendation"]
    restates_claim: bool
    attributes_to_country: bool


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--coder-slug", required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--response", action="append", required=True, metavar="SLUG=PATH",
                        help="Name for this set of answers and its JSONL. Repeat for several.")
    parser.add_argument("--conditions", nargs="+", default=["country_opposite_norm"],
                        help="Only answers to these prompt conditions are coded.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--progress-every", type=int, default=10)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args(argv)


def stance_payload(record: dict[str, Any], response: str) -> str:
    """What the coder sees: the claim and the answer, never the documented norm."""

    return json.dumps({
        "country": country(record),
        "claim": record["scenario"]["opposite_norm_evidence"],
        "situation": record["scenario"]["situation"],
        "question": record["scenario"]["question"],
        "response": response,
    }, ensure_ascii=False)


def successful(row: dict[str, Any]) -> bool:
    return (row.get("stance") in STANCES and isinstance(row.get("restates_claim"), bool)
            and isinstance(row.get("attributes_to_country"), bool))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    dataset = read_dataset(args.dataset.expanduser().resolve())
    prompt = load_prompt(args.prompt.expanduser().resolve())
    output_root = args.output_root.expanduser().resolve()
    wanted = set(args.conditions)

    work: list[tuple[str, Path, list[dict[str, Any]]]] = []
    for slug, path in response_specs(args.response):
        rows = [row for row in validate_responses(path, dataset) if str(row["condition"]) in wanted]
        output = output_root / slug / f"{args.coder_slug}.jsonl"
        existing = [row for row in read_jsonl(output) if successful(row)] if output.exists() else []
        write_jsonl_atomic(existing, output)
        done = {(str(row["norm_id"]), str(row["condition"])) for row in existing}
        pending = [row for row in rows if (str(row["norm_id"]), str(row["condition"])) not in done]
        work.append((slug, output, pending))
        print(f"answers={slug} expected={len(rows)} completed={len(done)} pending={len(pending)}",
              flush=True)
    if not any(pending for _, _, pending in work):
        print("All stance codings already complete", flush=True)
        return 0

    from inference.vllm import (NO_THINKING_CHAT_TEMPLATE_KWARGS, VLLMConfig,
                                generate_json_batch_partial, load_model)
    chat_kwargs = dict(NO_THINKING_CHAT_TEMPLATE_KWARGS) if args.no_think else None
    llm = load_model(VLLMConfig(
        model=args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        trust_remote_code=args.trust_remote_code, enforce_eager=args.enforce_eager,
        language_model_only=args.language_model_only, seed=args.seed))

    failures = 0
    for slug, output, pending in work:
        coded = unresolved = 0
        for number, batch in enumerate(chunks(pending, args.batch_size), 1):
            messages = [[{"role": "system", "content": prompt},
                         {"role": "user", "content": stance_payload(
                             dataset[str(row["norm_id"])], str(row["response"]))}] for row in batch]
            codings = generate_json_batch_partial(
                llm, messages, output_schema=StanceCoding, max_tokens=args.max_tokens,
                temperature=0.0, top_p=1.0, seed=args.seed, chat_template_kwargs=chat_kwargs)
            rows = [{"norm_id": row["norm_id"], "condition": row["condition"],
                     "stance": coding.stance, "restates_claim": coding.restates_claim,
                     "attributes_to_country": coding.attributes_to_country,
                     "evidence": coding.evidence}
                    for row, coding in zip(batch, codings, strict=True) if coding is not None]
            append_rows(output, rows)
            coded += len(rows)
            unresolved += len(batch) - len(rows)
            if number % max(1, args.progress_every) == 0 or number * args.batch_size >= len(pending):
                print(f"[{slug}] processed={min(number * args.batch_size, len(pending))}/{len(pending)} "
                      f"coded={coded} unresolved={unresolved}", flush=True)
        failures += unresolved
        print(f"[{slug}] {'complete' if not unresolved else f'unresolved={unresolved}'} output={output}",
              file=sys.stderr if unresolved else sys.stdout, flush=True)
    if failures:
        raise RuntimeError(f"Stance coding incomplete; {failures} answers remain; rerun to retry them")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
