"""Write, for each record, a personal request for the behaviour the custom does not prescribe.

A contradictory norm is a false statement about a country, and the right answer
keeps to the documented custom. A *request* is different: a person present in
the situation (the host, the elder, the organiser) asks, for this occasion, for
the other behaviour. Honouring it is their call, so the right answer does what
was asked. A model trained to hold to the documented custom against a
contradictory norm should still do that; one that holds to the custom here too
has learned to reject whatever departs from it.

A writer model drafts one sentence per record and says whether such a request
is plausible at all (``prompts/behavioral_evaluation/request_writer.txt``).
The output is a records file with only the plausible ones, in which the request
takes the place of ``opposite_norm_evidence`` (the original is kept as
``contradictory_norm_evidence``). The ``country_opposite_norm`` prompt of that
file is therefore situation, question, country, request, and every tool that
reads a records file works on it unchanged; ``behavioral_evaluation.stance``
with ``request_stance.txt`` then codes whether an answer does what was asked.

PYTHONPATH=src python -m behavioral_evaluation.requests --model <writer model> \
  --records output/finetune/mechanism/test_all.json \
  --output output/finetune/mechanism/test_requests.json
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from common.io import load_prompt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT = ROOT / "prompts/behavioral_evaluation/request_writer.txt"


class RequestDraft(BaseModel):
    applicable: bool
    speaker: str
    request: str


def writer_payload(record: dict[str, Any]) -> str:
    from behavioral_evaluation.judge import country

    scenario = record["scenario"]
    return json.dumps({
        "country": country(record), "situation": scenario["situation"], "question": scenario["question"],
        "documented_norm": record["norm"], "usual_behavior": record["options"]["A"],
        "other_behavior": record["options"]["B"]}, ensure_ascii=False)


def usable(draft: RequestDraft | None, record: dict[str, Any]) -> bool:
    """A plausible request that is one sentence of its own, not the stated norms."""

    if draft is None or not draft.applicable:
        return False
    text = draft.request.strip()
    scenario = record["scenario"]
    return (10 <= len(text) <= 400 and "\n" not in text
            and text not in (scenario["norm_evidence"], scenario["opposite_norm_evidence"]))


def with_request(record: dict[str, Any], draft: RequestDraft) -> dict[str, Any]:
    changed = copy.deepcopy(record)
    scenario = changed["scenario"]
    scenario["contradictory_norm_evidence"] = scenario["opposite_norm_evidence"]
    scenario["opposite_norm_evidence"] = draft.request.strip()
    changed["request_speaker"] = draft.speaker.strip()
    return changed


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--disable-custom-all-reduce", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from inference.vllm import (NO_THINKING_CHAT_TEMPLATE_KWARGS, VLLMConfig,
                                generate_json_batch_partial, load_model)

    records = json.loads(args.records.read_text(encoding="utf-8"))
    prompt = load_prompt(args.prompt.expanduser().resolve())
    llm = load_model(VLLMConfig(
        model=args.model, dtype=args.dtype, tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization,
        disable_custom_all_reduce=args.disable_custom_all_reduce,
        trust_remote_code=args.trust_remote_code, enforce_eager=args.enforce_eager,
        language_model_only=args.language_model_only, seed=args.seed))
    chat_kwargs = dict(NO_THINKING_CHAT_TEMPLATE_KWARGS) if args.no_think else None
    kept: list[dict[str, Any]] = []
    counts = {"records": len(records), "unparsed": 0, "not_applicable": 0, "rejected": 0}
    for start in range(0, len(records), args.batch_size):
        batch = records[start:start + args.batch_size]
        drafts = generate_json_batch_partial(
            llm, [[{"role": "system", "content": prompt},
                   {"role": "user", "content": writer_payload(record)}] for record in batch],
            output_schema=RequestDraft, max_tokens=args.max_tokens, temperature=0.0, top_p=1.0,
            seed=args.seed, chat_template_kwargs=chat_kwargs)
        for record, draft in zip(batch, drafts, strict=True):
            if draft is None:
                counts["unparsed"] += 1
            elif not draft.applicable:
                counts["not_applicable"] += 1
            elif not usable(draft, record):
                counts["rejected"] += 1
            else:
                kept.append(with_request(record, draft))
        print(f"written {min(start + args.batch_size, len(records))}/{len(records)} kept={len(kept)}", flush=True)
    counts["kept"] = len(kept)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(kept, ensure_ascii=False), encoding="utf-8")
    args.output.with_suffix(".counts.json").write_text(json.dumps(counts, indent=2), encoding="utf-8")
    print(json.dumps(counts), flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
