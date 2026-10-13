"""GRPO and Critique-GRPO for CultureLens.

    python -m finetune.grpo --reward constrained --train .../train.json \
        --validation .../validation.json [--counterfactual]

    python -m finetune.grpo --reward critique [--critic policy|reference] ...
    python -m finetune.grpo --reward critique+constrained ...

One step:

1. draw prompts, evenly across the recipe's views (and across the two tasks
   under ``critique+constrained``);
2. sample ``--group-size`` rollouts per prompt from the policy;
3. score them: exact match on the A/B letter for constrained prompts; for open
   prompts, the critic -- this same model, given the documented norm -- writes a
   critique and a verdict;
4. Critique-GRPO only: the weakest open rollouts are rewritten with their
   critique in context, scored the same way, and join their prompt's group;
5. standardise rewards within each group, drop groups where every rollout
   scored alike, and take a clipped policy-gradient step with a KL penalty to
   the initialisation.

**Refinements are credited to the plain prompt.** At test time there is no
critique in context, so a rewrite is scored and trained as an answer to the
original question. That makes it off-policy -- it was sampled from the prompt
plus critique -- and ``--updates-per-step`` above 1 is what lets the clipped
ratio bound how far one such sample can move the policy.

**Reference is the initialisation.** The KL anchor, and the critic under
``--critic reference``, is a frozen second adapter loaded from ``--init`` (or a
fresh zero-initialised adapter, which is exactly the base model).

**Every visible GPU samples.** Sampling and critiquing are most of a step, so
one copy of the model sits on each GPU and requests are split across them; the
policy adapter is copied to the others before each pass. The update itself
runs on the first GPU.

**A killed job resumes.** With ``--resume`` the run restores its adapter,
optimiser, random streams and position in the data from the last saved state,
so resubmitting the same command continues the same run rather than a new one.

**Validation is judge-free first.** Every evaluation reports constrained A/B
accuracy on each monitor view with the benchmark's own fixed option order, so
whatever the reward, the run has a number no model graded. Under a critique
reward it also reports the critic's score on open answers, judged by the frozen
reference by default so the yardstick does not move with the policy.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from . import (
    BASE_MODEL, CONSTRAINED, CRITICS, CRITIQUE, MONITOR_VIEWS, REWARDS, SCHEMA,
    method_name, reward_kinds, training_views,
)
from .data import PromptPool, fixed_sample, load_records
from .prompts import (
    CONSTRAINED_SYSTEM_PROMPT, constrained_message, critic_message, critic_system_prompt,
    draw_constrained, gold_letter, open_message, parse_critique, refinement_message,
)
from .rewards import constrained_reward, critique_reward, group_advantages, weakest

POLICY = "default"
REFERENCE = "reference"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--reward", choices=REWARDS, required=True)
    parser.add_argument("--counterfactual", action="store_true",
                        help="Add country_opposite_norm, country_norm and country_value.")
    parser.add_argument("--critic", choices=CRITICS, default="policy",
                        help="Which weights critique during training: the live policy, "
                             "or the frozen initialisation.")
    parser.add_argument("--validation-critic", choices=CRITICS, default="reference",
                        help="Which weights score open answers on validation.")
    parser.add_argument("--refine-count", type=int, default=2,
                        help="Weakest open rollouts per group rewritten with their critique. "
                             "0 turns Critique-GRPO into GRPO with a self-critic reward.")
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", default=None, help="Run name; derived from the flags if omitted.")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--init", type=Path, default=None,
                        help="Adapter to start from, also the KL reference. Omit for a fresh adapter.")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=None,
                        help="Update steps. Give this or --epochs; --epochs 1 if neither.")
    parser.add_argument("--epochs", type=float, default=None,
                        help="Passes over the data: every record drawn once under every "
                             "trained view, by every task.")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from the state saved in --output-dir, if there is one.")
    parser.add_argument("--state-every", type=int, default=20,
                        help="Steps between resumable state saves.")
    parser.add_argument("--sampling-devices", type=int, default=0,
                        help="GPUs used for sampling; 0 means every visible GPU.")
    parser.add_argument("--prompts-per-step", type=int, default=8)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--updates-per-step", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=4)
    parser.add_argument("--generation-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--clip-epsilon", type=float, default=0.2)
    parser.add_argument("--kl-weight", type=float, default=0.04)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--critic-tokens", type=int, default=512)
    parser.add_argument("--answer-budget", type=int, default=512,
                        help="Tokens allowed before the length penalty engages.")
    parser.add_argument("--length-weight", type=float, default=0.1)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    parser.add_argument("--eval-every", type=int, default=25)
    parser.add_argument("--eval-per-view", type=int, default=120)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    args.kinds = reward_kinds(args.reward)
    args.views = training_views(args.counterfactual)
    if args.method is None:
        args.method = method_name(args.reward, args.counterfactual)
    if args.output_dir is None:
        model_tag = args.model.rstrip("/").split("/")[-1].lower()
        args.output_dir = Path("output/finetune/runs") / model_tag / args.method
    if args.refine_count < 0 or args.refine_count >= args.group_size:
        parser.error("--refine-count must be in [0, --group-size)")
    if args.steps is not None and args.epochs is not None:
        parser.error("give --steps or --epochs, not both")
    if args.steps is None and args.epochs is None:
        args.epochs = 1.0
    return args


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from behavioral_evaluation.prompts import is_shuffled

    from .losses import grpo_loss, token_logprobs
    from .tokenization import ChatEncoder, messages

    output_dir: Path = args.output_dir
    state_dir = output_dir / "state"
    resuming = args.resume and (state_dir / "state.json").is_file()
    if output_dir.exists() and any(output_dir.iterdir()) and not resuming:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; pass --resume to continue it")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    set_seed(args.seed)
    rng = random.Random(args.seed)

    def log(payload: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        print(json.dumps(payload), flush=True)

    # ------------------------------------------------------------ model
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"  # generation needs every prompt flush right
    encoder = ChatEncoder(tokenizer)

    count = torch.cuda.device_count()
    if count < 1:
        raise RuntimeError("GRPO needs at least one GPU")
    devices = [f"cuda:{i}" for i in range(min(count, args.sampling_devices or count))]

    def build(device: str):
        base = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=torch.bfloat16, device_map={"": device},
            local_files_only=args.local_files_only)
        base.config.use_cache = True
        if args.init is None:
            # A fresh adapter's B matrices are zero, so policy and reference
            # both start as the base model exactly: the result is
            # attributable to RL alone.
            config = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                                lora_dropout=args.lora_dropout, target_modules=list(LORA_TARGETS),
                                bias="none", task_type="CAUSAL_LM")
            built = get_peft_model(base, config, adapter_name=POLICY)
            built.add_adapter(REFERENCE, config)
        else:
            built = PeftModel.from_pretrained(base, str(args.init), adapter_name=POLICY,
                                              is_trainable=True)
            built.load_adapter(str(args.init), adapter_name=REFERENCE)
        # Both adapters in fp32: were only the policy upcast, policy and
        # reference would differ numerically at step 0 and the KL would start
        # above zero with identical weights.
        for name, parameter in built.named_parameters():
            if "lora_" in name and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        # Dropout off for every pass: under LoRA dropout the old and new
        # log-probabilities would disagree on identical weights and the ratio
        # would be noise. eval() does not stop gradients.
        built.eval()
        built.set_adapter(POLICY)
        return built

    model = build(devices[0])
    for name, parameter in model.named_parameters():
        if REFERENCE in name:
            parameter.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.0)

    # Sampling copies, one per extra GPU. Only the policy adapter ever changes,
    # so that is all that is copied across; the reference is identical on
    # every copy by construction.
    replicas = [build(device) for device in devices[1:]]
    for replica in replicas:
        replica.requires_grad_(False)

    def policy_parameters(module) -> dict[str, Any]:
        return {name: parameter for name, parameter in module.named_parameters()
                if "lora_" in name and f".{POLICY}." in name}

    source = policy_parameters(model)
    targets = [policy_parameters(replica) for replica in replicas]
    for target in targets:
        if target.keys() != source.keys():
            raise RuntimeError("A sampling copy has a different adapter layout from the policy")

    def sync() -> None:
        with torch.no_grad():
            for target in targets:
                for name, parameter in target.items():
                    parameter.copy_(source[name])

    # ------------------------------------------------------------- data
    train_records = load_records(args.train)
    validation_records = load_records(args.validation) if args.validation else []
    pool = PromptPool(train_records, args.views, args.kinds, args.seed,
                      prompts_per_step=args.prompts_per_step)
    total_steps = args.steps if args.steps is not None else math.ceil(
        args.epochs * pool.prompts_per_epoch() / args.prompts_per_step)
    log({"event": "resume" if resuming else "plan", "schema": SCHEMA, "method": args.method,
         "steps": total_steps, "epochs": total_steps * args.prompts_per_step / pool.prompts_per_epoch(),
         "devices": devices, "reward": args.reward,
         "counterfactual": args.counterfactual, "views": list(args.views),
         "kinds": list(args.kinds), "pool": pool.sizes(),
         "critic": args.critic if CRITIQUE in args.kinds else None,
         "refine_count": args.refine_count if CRITIQUE in args.kinds else 0,
         "rollouts_per_step": args.prompts_per_step * args.group_size,
         "init": str(args.init) if args.init else "fresh"})

    # ------------------------------------------------------- generation
    models = [model, *replicas]

    def generate(requests: list[tuple[str, str | None]], *, adapter: str, sample: bool,
                 budget: int) -> list[dict[str, Any]]:
        """One completion per (prompt, system) request, split across the GPUs."""

        if not requests:
            return []
        if adapter == POLICY and replicas:
            sync()
        rendered = [encoder.render_prompt(prompt, system) for prompt, system in requests]
        share = math.ceil(len(rendered) / len(models))
        # Tokenised here, on the calling thread: a fast tokenizer is not safe
        # to share between threads, so workers only ever see tensors.
        work = []
        for index, worker in enumerate(models):
            indices = list(range(index * share, min((index + 1) * share, len(rendered))))
            batches = []
            for start in range(0, len(indices), args.generation_batch_size):
                chunk = indices[start : start + args.generation_batch_size]
                encoded = tokenizer([rendered[i] for i in chunk], add_special_tokens=False,
                                    return_tensors="pt", padding=True)
                if encoded["input_ids"].shape[1] > args.max_prompt_tokens:
                    raise ValueError(f"Prompt of {encoded['input_ids'].shape[1]} tokens exceeds "
                                     f"--max-prompt-tokens {args.max_prompt_tokens}")
                batches.append((chunk, encoded))
            if batches:
                work.append((worker, batches))
        options = ({"do_sample": True, "temperature": args.temperature, "top_p": args.top_p}
                   if sample else {"do_sample": False})

        def run(item):
            worker, batches = item
            worker.set_adapter(adapter)
            produced = []
            with torch.no_grad():
                for chunk, encoded in batches:
                    encoded = encoded.to(worker.device)
                    generated = worker.generate(**encoded, max_new_tokens=budget,
                                                pad_token_id=tokenizer.pad_token_id, **options)
                    produced.append((chunk, generated[:, encoded["input_ids"].shape[1]:].cpu()))
            return produced

        if len(work) == 1:
            results = [run(work[0])]
        else:
            with ThreadPoolExecutor(max_workers=len(work)) as executor:
                results = list(executor.map(run, work))
        out: list[dict[str, Any] | None] = [None] * len(rendered)
        for produced in results:
            for chunk, completions in produced:
                for index, ids in zip(chunk, completions):
                    kept = [int(v) for v in ids if int(v) != tokenizer.pad_token_id]
                    out[index] = {"text": tokenizer.decode(kept, skip_special_tokens=True).strip(),
                                  "tokens": len(kept),
                                  "finish": "length" if len(kept) >= budget else "stop"}
        model.set_adapter(POLICY)
        return out  # type: ignore[return-value]

    def critique(rows: list[dict[str, Any]], adapter: str) -> list[dict[str, Any] | None]:
        """The critic's verdict on each open rollout. Greedy: a reward should
        not change because the critic rolled differently."""

        requests = [(critic_message(r["record"], r["text"]), critic_rubric) for r in rows]
        judged = generate(requests, adapter=adapter, sample=False, budget=args.critic_tokens)
        return [parse_critique(j["text"]) if j["finish"] != "length" else None for j in judged]

    critic_adapter = REFERENCE if args.critic == "reference" else POLICY
    critic_rubric = critic_system_prompt()

    def prompt_for(item: dict[str, Any]) -> dict[str, Any]:
        """A policy prompt. Constrained order is drawn here, once per prompt,
        so every rollout in a group answers the same question."""

        if item["kind"] == CONSTRAINED:
            prompt, gold = draw_constrained(item["record"], item["view"], rng)
            return {**item, "prompt": prompt, "system": CONSTRAINED_SYSTEM_PROMPT, "gold": gold}
        return {**item, "prompt": open_message(item["record"], item["view"]), "system": None}

    def score(rollouts: list[dict[str, Any]], adapter: str) -> None:
        """Fill ``reward`` and diagnostics in place."""

        open_rows = [r for r in rollouts if r["kind"] == CRITIQUE]
        verdicts = critique(open_rows, adapter) if open_rows else []
        for row, verdict in zip(open_rows, verdicts):
            row["verdict_raw"] = verdict
            row.update(critique_reward(row["text"], row["finish"], row["tokens"], verdict,
                                       answer_budget=args.answer_budget,
                                       length_weight=args.length_weight))
        for row in rollouts:
            if row["kind"] == CONSTRAINED:
                row.update(constrained_reward(row["text"], row["gold"], row["finish"], row["tokens"],
                                              answer_budget=args.answer_budget,
                                              length_weight=args.length_weight))

    def rollout(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        prompted = [prompt_for(item) for item in items]
        requests = [(p["prompt"], p["system"]) for p in prompted for _ in range(args.group_size)]
        results = generate(requests, adapter=POLICY, sample=True, budget=args.max_new_tokens)
        rows = []
        for index, result in enumerate(results):
            group = index // args.group_size
            rows.append({**prompted[group], **result, "group": group, "refined": False})
        return rows

    def refine(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Critique-GRPO: rewrite the weakest open rollouts with their critique."""

        if args.refine_count == 0:
            return []
        usable = [r["kind"] == CRITIQUE and r.get("verdict_raw") is not None
                  and r["verdict_raw"]["score"] < 1.0 and bool(r["verdict_raw"]["critique"])
                  for r in rows]
        targets = weakest([r["reward"] for r in rows], [r["group"] for r in rows],
                          args.refine_count, eligible=usable)
        if not targets:
            return []
        requests = [(refinement_message(rows[i]["prompt"], rows[i]["text"],
                                        rows[i]["verdict_raw"]["critique"]), None)
                    for i in targets]
        results = generate(requests, adapter=POLICY, sample=True, budget=args.max_new_tokens)
        # Credited to the plain prompt: that is the distribution it is tested under.
        refined = [{**rows[i], **result, "refined": True} for i, result in zip(targets, results)]
        score(refined, critic_adapter)
        return refined

    # ---------------------------------------------------- log-probabilities
    def batches(rows: list[dict[str, Any]]):
        for start in range(0, len(rows), args.micro_batch_size):
            yield rows[start : start + args.micro_batch_size]

    def encode(chunk: list[dict[str, Any]]):
        encoded = [encoder.encode(r["prompt"], r["text"], r["system"]) for r in chunk]
        width = max(len(ids) for ids, _ in encoded)
        input_ids = torch.full((len(chunk), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(chunk), width), dtype=torch.long)
        for row, (ids, _) in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[row, : len(ids)] = 1
        lengths = torch.tensor([length for _, length in encoded], dtype=torch.long)
        return input_ids.to(model.device), mask.to(model.device), lengths.to(model.device)

    def frozen(rows: list[dict[str, Any]], adapter: str) -> list[Any]:
        model.set_adapter(adapter)
        values = []
        with torch.no_grad():
            for chunk in batches(rows):
                ids, mask, lengths = encode(chunk)
                logits = model(input_ids=ids, attention_mask=mask).logits
                chunk_values, _ = token_logprobs(logits, ids, lengths, mask)
                values.append(chunk_values.detach())
                del logits
        model.set_adapter(POLICY)
        return values

    # ------------------------------------------------------------ validation
    fixed = {view: fixed_sample(validation_records, view, args.eval_per_view, args.seed)
             for view in MONITOR_VIEWS} if validation_records else {}
    baseline: dict[str, float] = {}

    def evaluate(step: int) -> float | None:
        """Constrained accuracy per monitor view; returns the selection score."""

        if not fixed:
            return None
        report: dict[str, Any] = {"event": "validation", "step": step, "constrained": {}}
        for view, records in fixed.items():
            if not records:
                continue
            # The benchmark's own fixed order, so these numbers compare with eval.
            shuffles = [is_shuffled(r["norm_id"]) for r in records]
            requests = [(constrained_message(r, view, shuffled=s), CONSTRAINED_SYSTEM_PROMPT)
                        for r, s in zip(records, shuffles)]
            results = generate(requests, adapter=POLICY, sample=False, budget=args.max_new_tokens)
            scored = [constrained_reward(res["text"], gold_letter(s), res["finish"], res["tokens"],
                                         answer_budget=args.answer_budget, length_weight=0.0)
                      for res, s in zip(results, shuffles)]
            n = len(scored)
            report["constrained"][view] = {
                "accuracy": sum(s["correct"] for s in scored) / n,
                "format_valid": sum(s["format_valid"] for s in scored) / n,
                "chose_a": sum(s["chose_a"] for s in scored) / n, "n": n}
        accuracy = {v: m["accuracy"] for v, m in report["constrained"].items()}
        if step == 0:
            baseline.update(accuracy)
        # The guard: a counterfactual arm that learns to reject every stated
        # norm shows up first as country_norm falling below where it started.
        if "country_norm" in baseline:
            report["country_norm_change"] = accuracy["country_norm"] - baseline["country_norm"]
        if CRITIQUE in args.kinds:
            report["open"] = {}
            adapter = REFERENCE if args.validation_critic == "reference" else POLICY
            for view in args.views:
                records = fixed.get(view) or []
                if not records:
                    continue
                items = [{"record": r, "view": view, "kind": CRITIQUE, "country": _country(r)}
                         for r in records]
                answers = generate([(open_message(r, view), None) for r in records],
                                   adapter=POLICY, sample=False, budget=args.max_new_tokens)
                rows = [{**item, **answer} for item, answer in zip(items, answers)]
                verdicts = critique(rows, adapter)
                parsed = [v for v in verdicts if v is not None]
                report["open"][view] = {
                    "critic_score": (sum(v["score"] for v in parsed) / len(parsed)) if parsed else None,
                    "verdict_parse_rate": len(parsed) / len(verdicts), "critic": args.validation_critic}
        selection = sum(accuracy[v] for v in args.views if v in accuracy) / max(
            1, sum(1 for v in args.views if v in accuracy))
        report["selection_score"] = selection
        log(report)
        return selection

    def save(name: str, step: int, extra: dict[str, Any]) -> None:
        target = output_dir / name
        model.set_adapter(POLICY)
        model.save_pretrained(str(target), selected_adapters=[POLICY])
        tokenizer.save_pretrained(str(target))
        (target / "training_config.json").write_text(json.dumps(
            {"schema": SCHEMA, "step": step, "method": args.method,
             "arguments": {k: _plain(v) for k, v in vars(args).items()}, **extra}, indent=2))

    # --------------------------------------------------------------- state
    def save_state(step: int, hours: float) -> None:
        """Everything a resubmitted job needs to continue this exact run.

        Written beside the live state and swapped in, so a job killed while
        saving leaves the previous state intact.
        """

        fresh = output_dir / "state.new"
        shutil.rmtree(fresh, ignore_errors=True)
        fresh.mkdir(parents=True)
        model.set_adapter(POLICY)
        model.save_pretrained(str(fresh / "adapter"), selected_adapters=[POLICY])
        torch.save({"optimizer": optimizer.state_dict(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state_all()}, fresh / "trainer.pt")
        version, internal, gauss = rng.getstate()
        (fresh / "state.json").write_text(json.dumps({
            "schema": SCHEMA, "step": step, "hours": hours, "best": best, "baseline": baseline,
            "rng": [version, list(internal), gauss], "pool": pool.state(),
            "devices": len(devices), "method": args.method, "total_steps": total_steps}))
        old = output_dir / "state.old"
        shutil.rmtree(old, ignore_errors=True)
        if state_dir.exists():
            state_dir.rename(old)
        fresh.rename(state_dir)
        shutil.rmtree(old, ignore_errors=True)

    # ---------------------------------------------------------------- loop
    best = {"score": -math.inf, "step": 0}
    first_step, elapsed = 1, 0.0
    if resuming:
        from peft.utils import load_peft_weights, set_peft_model_state_dict

        saved = json.loads((state_dir / "state.json").read_text())
        if saved["method"] != args.method:
            raise ValueError(f"{state_dir} belongs to {saved['method']}, not {args.method}")
        set_peft_model_state_dict(
            model, load_peft_weights(str(state_dir / "adapter"), device="cpu"), adapter_name=POLICY)
        for parameter in source.values():
            if parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        trainer = torch.load(state_dir / "trainer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(trainer["optimizer"])
        torch.set_rng_state(trainer["torch"])
        if len(trainer["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(trainer["cuda"])
        version, internal, gauss = saved["rng"]
        rng.setstate((version, tuple(internal), gauss))
        pool.load_state(saved["pool"])
        best, elapsed = saved["best"], saved["hours"]
        baseline.update(saved["baseline"])
        first_step = saved["step"] + 1
        log({"event": "resumed", "from_step": saved["step"], "of": total_steps, "best": best})
    else:
        initial = evaluate(0)
        if initial is not None:
            # The initialisation is a candidate too: if nothing beats it, best/ is it.
            best = {"score": initial, "step": 0}
            save("best", 0, {"selection_score": initial})
    started = time.time() - elapsed * 3600
    for step in range(first_step, total_steps + 1):
        rows = rollout(pool.draw(args.prompts_per_step))
        score(rows, critic_adapter)
        refined = refine(rows) if CRITIQUE in args.kinds else []
        rows = rows + refined

        rewards = [r["reward"] for r in rows]
        advantages, informative = group_advantages(rewards, [r["group"] for r in rows])
        keep = [i for i, r in enumerate(rows) if informative[r["group"]]]
        by_kind: dict[str, list[float]] = defaultdict(list)
        for r in rows:
            by_kind[r["kind"]].append(r["reward"])
        stats: dict[str, Any] = {
            "event": "train", "step": step, "of": total_steps,
            "reward": sum(rewards) / len(rewards),
            "reward_by_kind": {k: sum(v) / len(v) for k, v in by_kind.items()},
            "reward_by_view": _mean_by(rows, "view"),
            "format_valid": sum(bool(r.get("format_valid")) for r in rows) / len(rows),
            "answer_tokens": sum(r["tokens"] for r in rows) / len(rows),
            "informative_groups": sum(informative.values()), "of_groups": len(informative),
            "refined": len(refined),
            "hours": round((time.time() - started) / 3600, 3)}
        constrained_rows = [r for r in rows if r["kind"] == CONSTRAINED]
        if constrained_rows:
            stats["accuracy"] = sum(r["correct"] for r in constrained_rows) / len(constrained_rows)
            stats["chose_a"] = sum(r["chose_a"] for r in constrained_rows) / len(constrained_rows)
        open_rows = [r for r in rows if r["kind"] == CRITIQUE]
        if open_rows:
            stats["verdict_parse_rate"] = sum(r["verdict_parsed"] for r in open_rows) / len(open_rows)
            if refined:
                gained = [r["reward"] for r in refined]
                stats["refined_reward"] = sum(gained) / len(gained)
        if not keep:
            # Still a step: validation, checkpoints and saved state below must
            # not be skipped because this one carried no gradient.
            log({**stats, "skipped": "no_informative_groups"})
        else:
            kept = [rows[i] for i in keep]
            kept_advantages = torch.tensor([advantages[i] for i in keep], dtype=torch.float32,
                                           device=model.device)
            old = frozen(kept, POLICY)
            reference = frozen(kept, REFERENCE)
            collected: dict[str, list[float]] = defaultdict(list)
            total = 0.0
            for _ in range(args.updates_per_step):
                optimizer.zero_grad(set_to_none=True)
                for index, chunk in enumerate(batches(kept)):
                    ids, mask, lengths = encode(chunk)
                    logits = model(input_ids=ids, attention_mask=mask).logits
                    policy, answer_mask = token_logprobs(logits, ids, lengths, mask)
                    start = index * args.micro_batch_size
                    losses, diagnostics = grpo_loss(
                        policy, old[index], reference[index],
                        kept_advantages[start : start + len(chunk)], answer_mask,
                        clip_epsilon=args.clip_epsilon, kl_weight=args.kl_weight)
                    loss = losses.sum() / len(kept)
                    if not torch.isfinite(loss):
                        raise FloatingPointError(f"Non-finite loss at step {step}")
                    loss.backward()
                    total += float(loss.detach())
                    for key, value in diagnostics.items():
                        collected[key].extend(value.tolist())
                    del logits, policy
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
            log({**stats, "loss": total / args.updates_per_step, "rollouts_used": len(kept),
                 "kl": _mean(collected["kl"]), "clip_fraction": _mean(collected["clip_fraction"])})

        if args.eval_every and step % args.eval_every == 0:
            score_now = evaluate(step)
            if score_now is not None and score_now > best["score"]:
                best = {"score": score_now, "step": step}
                save("best", step, {"selection_score": score_now})
        if args.save_every and step % args.save_every == 0:
            save(f"step-{step}", step, {})
        if args.state_every and step % args.state_every == 0:
            save_state(step, (time.time() - started) / 3600)

    save("final", total_steps, {"best": best})
    save_state(total_steps, (time.time() - started) / 3600)
    (output_dir / "manifest.json").write_text(json.dumps(
        {"schema": SCHEMA, "method": args.method, "steps": total_steps, "best": best,
         "hours": round((time.time() - started) / 3600, 3),
         "arguments": {k: _plain(v) for k, v in vars(args).items()}}, indent=2))
    log({"event": "done", "best": best})
    return 0


def _country(record: dict[str, Any]) -> str:
    from .data import country_name

    return country_name(record)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _mean_by(rows: list[dict[str, Any]], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row[key]].append(row["reward"])
    return {k: sum(v) / len(v) for k, v in sorted(grouped.items())}


if __name__ == "__main__":
    raise SystemExit(main())
