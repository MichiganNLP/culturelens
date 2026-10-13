"""Value-first answering: name the value at stake, then answer through it.

With the country alone, a tuned model often gives advice that fits the culture
but never addresses the behaviour at issue; stating the value the norm serves
removes most of those answers. Here the model supplies that value itself. One
adapter is trained on the usual views (``finetune.sft --counterfactual
--value-first``) and on one more task:

* **value**: given the situation, the question and the country, write the
  value at stake, in the form of the dataset's value evidence ("... is
  important."). Only records that have value evidence are trained on.

At evaluation (this module, with ``--predict``) a ``country_only`` prompt is
answered in two steps: the model writes the value, and then answers the
``country_value`` prompt built with that sentence. This is done for every
record, with or without annotated value evidence. Every other condition, and
``country_only`` without ``--predict``, is answered directly.

    python -m finetune.value_first --model <model> --lora runs/sft-cf-value/best --predict \
        --input .../splits/test_seen.json --output eval/sft-cf-value/test_seen/unconstrained.jsonl

The output rows are the ones ``behavioral_evaluation.generate`` writes, with
the predicted value beside the answer, so the judge reads them unchanged.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .mirror import _same, clean_mirror

VALUE_SYSTEM = (
    "You will be given a situation, a question, and the country it takes place in. "
    "Write one sentence naming what is important to most people there in that situation, "
    "ending with \"is important.\" Return only the sentence."
)
VALUE_VIEW = "country_only"


def value_message(record: dict[str, Any]) -> str:
    """The evaluation's own country-only prompt, then the request."""

    from behavioral_evaluation.prompts import build_user_message

    return build_user_message(record, VALUE_VIEW, False)[0] + "\n\nWhat is important to most people there?"


def answer_message(record: dict[str, Any], value: str) -> str:
    """The open-ended prompt of ``country_value`` with ``value`` as the value evidence."""

    scenario = record["scenario"]
    return "\n\n".join([f"Situation: {scenario['situation']}\n\nQuestion: {scenario['question']}",
                        scenario["country_context"], value.strip()])


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    found = [row for row in rows if "predicted_value" in row]
    annotated = [row for row in found if row["dataset_value"]]
    return {"prompts": len(found),
            "without_a_prediction": sum(not row["predicted_value"] for row in found),
            "with_annotated_value": len(annotated),
            # Word-for-word matches only, so this is a lower bound.
            "prediction_is_annotated_value": (round(sum(row["value_matches_dataset"] for row in annotated)
                                                    / len(annotated), 4) if annotated else None)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=[
        "country_only", "country_value", "country_norm", "country_opposite_norm"])
    parser.add_argument("--predict", action="store_true",
                        help="Answer country_only prompts through a value the model writes first.")
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--value-tokens", type=int, default=64)
    parser.add_argument("--max-lora-rank", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from behavioral_evaluation.generate import (
        append_rows, chunks, eligible, generate_plain, make_row, read_records, successful,
    )
    from behavioral_evaluation.prompts import build_user_message
    from common.io import read_jsonl, write_jsonl_atomic
    from inference.vllm import VLLMConfig, load_model

    records = read_records(args.input.expanduser().resolve())
    output = args.output.expanduser().resolve()
    existing = [row for row in read_jsonl(output) if successful(row, False)] if output.exists() else []
    write_jsonl_atomic(existing, output)
    done = {(str(row["norm_id"]), str(row["condition"])) for row in existing}
    expected = {(str(record["norm_id"]), condition) for record in records
                for condition in args.conditions if eligible(record, condition)}
    print(f"model={args.model} mode=value-first predict={args.predict} records={len(records)} "
          f"expected={len(expected)} completed={len(done & expected)} pending={len(expected - done)}",
          flush=True)

    if expected - done:
        adapter = args.lora.expanduser().resolve()
        if not (adapter / "adapter_config.json").is_file():
            raise FileNotFoundError(f"{adapter} is not a LoRA adapter directory")
        llm = load_model(VLLMConfig(
            model=args.model, dtype=args.dtype, max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization, seed=args.seed,
            enable_lora=True, max_lora_rank=args.max_lora_rank))
        from vllm.lora.request import LoRARequest

        lora = LoRARequest("finetuned", 1, str(adapter))

        def write(conversations: list[list[dict[str, str]]], max_tokens: int) -> list[str]:
            return generate_plain(llm, conversations, max_tokens=max_tokens, seed=args.seed,
                                  chat_template_kwargs=None, lora_request=lora)

        for condition in args.conditions:
            pending = [record for record in records if eligible(record, condition)
                       and (str(record["norm_id"]), condition) not in done]
            print(f"[{condition}] pending={len(pending)}", flush=True)
            for number, batch in enumerate(chunks(pending, args.batch_size), 1):
                prompts = [build_user_message(record, condition, False)[0] for record in batch]
                extras: list[dict[str, Any]] = [{} for _ in batch]
                if args.predict and condition == VALUE_VIEW:
                    written = write([[{"role": "system", "content": VALUE_SYSTEM},
                                      {"role": "user", "content": value_message(record)}]
                                     for record in batch], args.value_tokens)
                    for i, record in enumerate(batch):
                        value = clean_mirror(written[i])
                        gold = record["scenario"].get("value_evidence") or ""
                        extras[i] = {"original_user_message": prompts[i], "predicted_value": value,
                                     "dataset_value": gold,
                                     "value_matches_dataset": bool(gold) and _same(value, gold)}
                        if value:                       # with nothing written, the prompt stands
                            prompts[i] = answer_message(record, value)
                responses = write([[{"role": "user", "content": prompt}] for prompt in prompts],
                                  args.max_tokens)
                rows = [{**make_row(record, condition, prompts[i], responses[i], False, False), **extras[i]}
                        for i, record in enumerate(batch) if responses[i].strip()]
                append_rows(output, rows)
                done.update((str(row["norm_id"]), condition) for row in rows)
                if number % 10 == 0 or number * args.batch_size >= len(pending):
                    print(f"[{condition}] processed={min(number * args.batch_size, len(pending))}/"
                          f"{len(pending)}", flush=True)

    final = {(str(row["norm_id"]), str(row["condition"])): row for row in read_jsonl(output)}
    missing = expected - set(final)
    if missing:
        raise RuntimeError(f"Evaluation is incomplete; {len(missing)} rows remain unresolved")
    if args.predict:
        steps = summarize(list(final.values()))
        output.with_suffix(".value.json").write_text(json.dumps(steps, indent=2))
        print(json.dumps(steps, indent=2), flush=True)
    print(f"complete rows={len(expected)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
