"""Mirror verification: compare a stated norm with its opposite before answering.

A model that judges a stated norm on its own can accept a norm and its
opposite, or reject both. Here the verdict comes from a comparison instead.
One adapter is trained on three tasks (``finetune.sft --mirror``):

* **mirror**: given a statement, write its opposite. No country and no
  situation is shown, so this step is about wording only;
* **compare**: given the situation, the country and the two statements in an
  order drawn at random, with nothing marking which one the user stated, name
  the one that is accurate for the country;
* **answer**: the open-ended answer, for the country alone, the country with a
  value, and the country with the true norm.

At evaluation (this module) a prompt that states a norm is answered in three
steps: the model writes the opposite of the stated norm, compares the two in
both orders, and answers the original question with the norm it chose in place
of the stated one. When the two orders disagree the stated norm is kept: with
no evidence either way, the user's statement stands. Prompts that state no norm
are answered directly.

**Proposing a norm, with ``--propose`` (mirror verification v2).** A prompt
that states no norm is answered the same way: the model first writes the norm
it thinks applies (a fourth task, trained on the documented norms with
``finetune.sft --mirror --propose``), and that proposal is compared with its
opposite like a stated norm. The proposal only has to be about the right
behaviour: if its direction is wrong, the comparison can still pick the
opposite. Without ``--propose`` such prompts are answered directly.

``control`` repeats the comparison with the country line removed. If it picks
the documented norm as often as the real comparison does, the comparison is
reading the wording of the statements and not the country.

    python -m finetune.mirror --model <model> --lora runs/sft-mirror/best \
        --input .../splits/test_seen.json --output eval/sft-mirror/test_seen/unconstrained.jsonl

The output rows are the ones ``behavioral_evaluation.generate`` writes, with
the steps recorded beside the answer, so the judge reads them unchanged.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

MIRROR_SYSTEM = (
    "You will be given a statement about what most people do or expect. Write the "
    "opposite statement: the same topic and the same form, with the behaviour or "
    "expectation reversed. Return only the statement."
)
COMPARE_SYSTEM = (
    "You will be given a situation, a question, and two statements about what most "
    "people do or expect. Exactly one of the statements is accurate for the setting "
    "described. Answer with the number of the accurate statement, 1 or 2, and nothing else."
)
PROPOSE_SYSTEM = (
    "You will be given a situation, a question, and the country it takes place in. "
    "Write one sentence stating what most people there do or expect in that situation. "
    "Return only the sentence."
)
PROPOSE_VIEWS = ("country_only", "country_value")
STATED_FIELD = {"country_norm": "norm_evidence", "country_opposite_norm": "opposite_norm_evidence"}
OTHER_FIELD = {"country_norm": "opposite_norm_evidence", "country_opposite_norm": "norm_evidence"}


def mirror_message(statement: str) -> str:
    return f"Statement: {statement.strip()}"


def propose_message(record: dict[str, Any], view: str = "country_only") -> str:
    """The evaluation's own prompt for a view that states no norm, then the request."""

    from behavioral_evaluation.prompts import build_user_message

    if view not in PROPOSE_VIEWS:
        raise ValueError(f"A norm is proposed for {PROPOSE_VIEWS}, not {view!r}")
    return build_user_message(record, view, False)[0] + "\n\nWhat do most people there do or expect?"


def compare_message(record: dict[str, Any], first: str, second: str, *, country: bool = True) -> str:
    scenario = record["scenario"]
    parts = [f"Situation: {scenario['situation']}\n\nQuestion: {scenario['question']}"]
    if country:
        parts.append(scenario["country_context"])
    parts.append(f"Statement 1: {first.strip()}\nStatement 2: {second.strip()}")
    parts.append("Which statement is accurate? Answer 1 or 2.")
    return "\n\n".join(parts)


def answer_message(record: dict[str, Any], norm: str) -> str:
    """The open-ended prompt of ``country_norm`` with ``norm`` as the stated norm."""

    scenario = record["scenario"]
    return "\n\n".join([f"Situation: {scenario['situation']}\n\nQuestion: {scenario['question']}",
                        scenario["country_context"], norm.strip()])


def _same(left: str, right: str) -> bool:
    squeeze = lambda text: re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()  # noqa: E731
    return squeeze(left) == squeeze(right)


def clean_mirror(text: str) -> str:
    """The first line of what the model wrote, without a ``Statement:`` label."""

    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    return re.sub(r"^(opposite\s+)?statement\s*:\s*", "", lines[0], flags=re.I).strip() if lines else ""


def decide(stated_first: str, mirror_first: str) -> tuple[str, bool]:
    """``("stated" | "mirror", orders agree)`` from the two comparisons.

    ``stated_first`` is the choice when the stated norm was statement 1 and
    ``mirror_first`` the choice when it was statement 2.
    """

    picks_mirror = (stated_first == "2", mirror_first == "1")
    if all(picks_mirror):
        return "mirror", True
    return "stated", not any(picks_mirror)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """How the steps behaved, per norm condition. ``chose_documented`` is the
    share of prompts answered with the documented norm's side of the pair."""

    out: dict[str, Any] = {}
    for condition in PROPOSE_VIEWS:
        found = [row for row in rows if row["condition"] == condition and "proposed_norm" in row]
        if found:
            share = lambda flags: round(sum(flags) / len(found), 4)  # noqa: E731
            out[condition] = {
                "prompts": len(found),
                "orders_agree": share([row["orders_agree"] for row in found]),
                "replaced_by_opposite": share([row["chosen"] == "mirror" for row in found]),
                # Word-for-word matches only, so these are lower bounds.
                "proposal_is_documented_norm": share([row["proposal_is_documented"] for row in found]),
                "proposal_is_contradictory_norm": share([row["proposal_is_contradictory"] for row in found]),
                "control_replaced_by_opposite": share([row["control_chosen"] == "mirror" for row in found]),
            }
    for condition in STATED_FIELD:
        found = [row for row in rows if row["condition"] == condition and "chosen" in row]
        if not found:
            continue
        keep_is_right = condition == "country_norm"
        share = lambda flags: round(sum(flags) / len(found), 4)  # noqa: E731
        out[condition] = {
            "prompts": len(found),
            "mirror_matches_dataset": share([row["mirror_matches_dataset"] for row in found]),
            "orders_agree": share([row["orders_agree"] for row in found]),
            "chose_documented": share([(row["chosen"] == "stated") == keep_is_right for row in found]),
            "control_orders_agree": share([row["control_orders_agree"] for row in found]),
            "control_chose_documented": share([(row["control_chosen"] == "stated") == keep_is_right
                                              for row in found]),
        }
    return out


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=[
        "country_only", "country_value", "country_norm", "country_opposite_norm"])
    parser.add_argument("--dtype", default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--propose", action="store_true",
                        help="Also verify prompts that state no norm, on a norm the model proposes.")
    parser.add_argument("--mirror-tokens", type=int, default=96)
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
    from inference.vllm import VLLMConfig, generate_choice_batch, load_model

    unknown = [name for name in args.conditions
               if name not in STATED_FIELD and name not in ("country_only", "country_value")]
    if unknown:
        raise ValueError(f"Mirror verification is defined for the four evaluated conditions, not {unknown}")
    records = read_records(args.input.expanduser().resolve())
    output = args.output.expanduser().resolve()
    existing = [row for row in read_jsonl(output) if successful(row, False)] if output.exists() else []
    write_jsonl_atomic(existing, output)
    done = {(str(row["norm_id"]), str(row["condition"])) for row in existing}
    expected = {(str(record["norm_id"]), condition) for record in records
                for condition in args.conditions if eligible(record, condition)}
    print(f"model={args.model} mode=mirror records={len(records)} expected={len(expected)} "
          f"completed={len(done & expected)} pending={len(expected - done)}", flush=True)

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

        def choose(prompts: list[str]) -> list[str]:
            return generate_choice_batch(
                llm, [[{"role": "system", "content": COMPARE_SYSTEM},
                       {"role": "user", "content": prompt}] for prompt in prompts],
                choices=("1", "2"), max_tokens=4, temperature=0.0, top_p=1.0, seed=args.seed,
                chat_template_kwargs=None, lora_request=lora)

        for condition in args.conditions:
            pending = [record for record in records if eligible(record, condition)
                       and (str(record["norm_id"]), condition) not in done]
            print(f"[{condition}] pending={len(pending)}", flush=True)
            for number, batch in enumerate(chunks(pending, args.batch_size), 1):
                extras: list[dict[str, Any]] = [{} for _ in batch]
                proposed = args.propose and condition in PROPOSE_VIEWS
                if proposed or condition in STATED_FIELD:
                    if proposed:
                        written = write([[{"role": "system", "content": PROPOSE_SYSTEM},
                                          {"role": "user", "content": propose_message(record, condition)}]
                                         for record in batch], args.mirror_tokens)
                        stated = [clean_mirror(text) for text in written]
                    else:
                        stated = [record["scenario"][STATED_FIELD[condition]] for record in batch]
                    written = write([[{"role": "system", "content": MIRROR_SYSTEM},
                                      {"role": "user", "content": mirror_message(text)}]
                                     for text in stated], args.mirror_tokens)
                    mirrors = [clean_mirror(text) or stated[index] for index, text in enumerate(written)]
                    direct = [build_user_message(record, condition, False)[0] for record in batch]
                    picks = {}
                    for name, country in (("compare", True), ("control", False)):
                        answers = choose(
                            [compare_message(record, stated[i], mirrors[i], country=country)
                             for i, record in enumerate(batch)]
                            + [compare_message(record, mirrors[i], stated[i], country=country)
                               for i, record in enumerate(batch)])
                        picks[name] = [decide(answers[i], answers[len(batch) + i]) for i in range(len(batch))]
                    prompts = []
                    for i, record in enumerate(batch):
                        chosen, agree = picks["compare"][i]
                        if not stated[i] or _same(mirrors[i], stated[i]):   # nothing to compare
                            chosen = "stated"
                        norm = mirrors[i] if chosen == "mirror" else stated[i]
                        # With nothing proposed, the prompt is answered as it stands.
                        prompts.append(answer_message(record, norm) if norm else direct[i])
                        extras[i] = {
                            "original_user_message": direct[i], "mirror_norm": mirrors[i],
                            "chosen": chosen, "orders_agree": agree, "answered_with": norm,
                            "control_chosen": picks["control"][i][0],
                            "control_orders_agree": picks["control"][i][1]}
                        scenario = record["scenario"]
                        if proposed:
                            extras[i].update(
                                proposed_norm=stated[i],
                                proposal_is_documented=_same(stated[i], scenario["norm_evidence"]),
                                proposal_is_contradictory=_same(stated[i], scenario["opposite_norm_evidence"]))
                        else:
                            extras[i].update(
                                stated_norm=stated[i],
                                mirror_matches_dataset=_same(mirrors[i], scenario[OTHER_FIELD[condition]]))
                else:
                    prompts = [build_user_message(record, condition, False)[0] for record in batch]
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
    steps = summarize(list(final.values()))
    output.with_suffix(".mirror.json").write_text(json.dumps(steps, indent=2))
    print(json.dumps(steps, indent=2), flush=True)
    print(f"complete rows={len(expected)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
