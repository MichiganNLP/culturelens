"""Supervised fine-tuning on the teacher's gold answers.

    python -m finetune.sft --train .../splits/train.json --gold-dir .../gold \
        --validation .../splits/validation.json [--counterfactual]

The policy is trained to reproduce a gold answer for the open-ended prompt,
with the loss on the answer tokens only. The views are the GRPO recipes':

* without ``--counterfactual``: ``country_only``;
* with it: ``country_only``, ``country_value``, ``country_norm`` and
  ``country_opposite_norm``.

**Two gold answers per record, by design.** The teacher wrote one answer for
the country-only prompt, and it is the target for every truthful view: stating
the norm or the value in the prompt does not change what the right answer is.
The false-norm view has its own answer, which corrects the claim before giving
the documented one. Gold files are the ``benchmark.generate gold`` and
``gold_cf`` outputs, named ``<split>.country_only.jsonl`` and
``<split>.country_opposite_norm.jsonl`` under ``--gold-dir``.

**An epoch is every example once.** The counterfactual recipe therefore has
about 3.4 times as many updates per epoch as the country recipe (three views
for every record, plus the value view where the record has a value). The plan
line in the log gives the exact counts.

**Selection is held-out loss.** The best adapter is the one with the lowest
answer loss on validation gold, averaged over the trained views. When there is
no validation gold, ``--dev-fraction`` of the training records is held out
instead (never trained on) and said so in the log.

**Both tasks, with ``--constrained``.** Every record and view then also gives
an A/B example: the evaluation's constrained prompt (its system prompt, the two
options) with the target ``<reasoning>gold answer</reasoning>`` followed by
``<answer>`` and the letter of the documented option. The reasoning is the same
gold answer the open-ended example is trained on, so the two tasks teach one
answer in two formats. The A/B order is re-drawn every epoch, so the letter is
never a fixed function of the prompt; development examples use the
evaluation's own fixed order. An epoch is then twice as long.

**Where the adapter goes.** By default every decoder block is adapted.
``--lora-layers`` restricts the adapter to the listed blocks, and
``--lora-heads`` (``layer.head`` pairs) to single attention heads: only the
query rows and the output-projection columns that belong to those heads are
trained (keys and values are shared between heads in grouped-query attention,
so they are left alone). These test whether training only where the stated
norm is read (``mechanistic_analysis.CLAIM_WINDOW``) is enough.

**Conflict loss, with ``--conflict-weight``.** An auxiliary loss on the state
at the last prompt token after block ``--conflict-layer``. For a record, the
part of that state that neither the country nor the stated norm explains alone
is ``interaction = state(country + norm) - state(norm) - state(country) +
state(bare)``. A linear probe, trained with the adapter, has to tell the
interaction under the contradictory norm from the one under the true norm, so
the model is pushed to register at that depth whether a stated norm fits the
country. Prompts only; no answer is involved. Validation reports the probe's
accuracy on held-out records.

**Framings, with ``--framings``.** Each time a norm view (``country_norm`` or
``country_opposite_norm``) is trained on, the stated norm is presented in one
of five ways drawn at random: as is, attributed to hearsay, attributed to a
guide, followed by the user's doubt, or placed before the country line. True
and contradictory norms get the same framings, so a framing never reveals
which one it is. Development examples keep the plain wording.

**Knockout-teacher distillation, with ``--distill-weight``.** The teacher is
the untuned model itself. On a contradictory-norm prompt the teacher's answer
stream cannot attend to the stated norm in any layer, which the mechanistic
analysis showed restores the answer the model gives from the country alone; on
a true-norm prompt the teacher is left untouched. The student (the model with
its adapter, attention intact) is trained to match the teacher's next-token
distribution along the truthful gold answer, and with
``--distill-activation-weight`` also its state at the last prompt token after
block ``--distill-layer``. The targets are the model's own, so nothing in this
loss asks it to change how it writes.

**Attention gating, with ``--gate-weight``.** A loss on the heads named by
``--gate-heads``: the share of attention the answer stream gives the stated
norm should go to zero when the norm is contradictory and stay where the
untuned model has it when the norm is true. It needs attention weights, so the
model is loaded with eager attention.

**Mirror verification, with ``--mirror``.** The adapter learns three tasks
(``finetune.mirror``): writing the opposite of a statement, naming which of a
norm and its opposite is accurate for the country, and the open-ended answer
for the truthful views. The contradictory-norm view is not trained on: at
evaluation a stated norm is compared with its opposite first, and the answer is
written with the one chosen. The two short tasks have targets of a few tokens,
so each of their examples is weighted to count as ``--task-tokens`` answer
tokens in the batch's token mean. ``--propose`` adds a fourth task for
``finetune.mirror --propose``: writing the documented norm for a prompt that
states none (the country alone, and the country with a value).

**Value first, with ``--value-first``.** Beside the usual views, the adapter
learns to write the value at stake for a country-only prompt, with the
dataset's value evidence as the target (records that have it). At evaluation
``finetune.value_first --predict`` answers a country-only prompt through the
value the model writes. These examples are weighted by ``--task-tokens`` too.

**Country memory, with ``--country-memory K``.** ``K`` trainable vectors per
training country are read at the position a stated norm occupies
(``finetune.country_memory``): every open-ended prompt that names the country
gets ``K`` placeholder tokens after the country line, and the embedding layer
swaps them for that country's vectors. They are trained with the adapter and
saved beside it as ``country_memory.pt``.

The adapter, output layout (``best/``, ``final/``, ``manifest.json``,
``metrics.jsonl``, ``state/``) and ``--resume`` behave as in ``finetune.grpo``,
so the same evaluation commands work on either.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import shutil
import time
from pathlib import Path
from typing import Any

from . import BASE_MODEL, SCHEMA, training_views
from behavioral_evaluation.prompts import is_shuffled

from .data import eligible, load_records
from .mirror import (
    COMPARE_SYSTEM, MIRROR_SYSTEM, PROPOSE_SYSTEM, PROPOSE_VIEWS, compare_message, mirror_message,
    propose_message,
)
from .country_memory import MEMORY_FILE, MEMORY_TOKEN, with_memory
from .value_first import VALUE_SYSTEM, value_message
from .prompts import (
    CONSTRAINED_SYSTEM_PROMPT, FRAMINGS, constrained_message, gold_letter, open_message,
)

POLICY = "default"
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
HEAD_TARGETS = ("q_proj", "o_proj")
FALSE_NORM = "country_opposite_norm"
# The six prompts whose states give the two interactions of the conflict loss.
CONFLICT_VIEWS = ("base", "country_only", "norm_only", "opposite_norm_only", "country_norm", FALSE_NORM)
GOLD_FILE = {False: "country_only", True: FALSE_NORM}  # keyed by "is the false-norm view"
# Mirror verification: the answer is trained on the truthful views only, and
# the two short tasks are scored on development like views of their own.
MIRROR_VIEWS = ("country_only", "country_value", "country_norm")
MIRROR_TASKS = ("mirror", "compare")
TASK_KINDS = (*MIRROR_TASKS, "propose", "value")   # examples with short targets, weighted by --task-tokens


def method_name(counterfactual: bool, constrained: bool = False) -> str:
    if constrained:
        return "sft-all-cf" if counterfactual else "sft-all"
    return f"sft-{'cf' if counterfactual else 'country'}"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--validation", type=Path, default=None)
    parser.add_argument("--gold-dir", type=Path, required=True,
                        help="Folder holding <split>.country_only.jsonl and "
                             "<split>.country_opposite_norm.jsonl.")
    parser.add_argument("--counterfactual", action="store_true",
                        help="Also train on country_value, country_norm and country_opposite_norm.")
    parser.add_argument("--constrained", action="store_true",
                        help="Also train on the A/B task for the same records and views.")
    parser.add_argument("--requests", type=Path, default=None,
                        help="Training records in which a person in the situation asks for the "
                             "behaviour the custom does not prescribe (behavioral_evaluation.requests). "
                             "Each one with an answer in <gold-dir>/<split>.request.jsonl is trained on "
                             "as a further view, so the adapter also learns to honour a request.")
    parser.add_argument("--validation-requests", type=Path, default=None,
                        help="The same for the validation records, scored as a view of its own.")
    parser.add_argument("--dev-fraction", type=float, default=0.03,
                        help="Share of training records held out for selection when there is "
                             "no validation gold.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--method", default=None, help="Run name; derived from the flags if omitted.")
    parser.add_argument("--model", default=BASE_MODEL)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-layers", type=int, nargs="+", default=None,
                        help="Decoder blocks to adapt (zero-indexed); all when omitted.")
    parser.add_argument("--lora-heads", nargs="+", default=None, metavar="LAYER.HEAD",
                        help="Adapt only these attention heads (their query rows and "
                             "output-projection columns).")
    parser.add_argument("--conflict-weight", type=float, default=0.0,
                        help="Weight of the auxiliary conflict loss; 0 turns it off.")
    parser.add_argument("--conflict-layer", type=int, default=16,
                        help="Decoder block whose output the conflict loss reads.")
    parser.add_argument("--conflict-records", type=int, default=4,
                        help="Records per step for the conflict loss (six prompts each).")
    parser.add_argument("--conflict-learning-rate", type=float, default=1e-3,
                        help="Learning rate of the conflict probe.")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--batch-size", type=int, default=16, help="Examples per optimiser step.")
    parser.add_argument("--micro-batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--warmup-fraction", type=float, default=0.05)
    parser.add_argument("--max-tokens", type=int, default=2048,
                        help="Longest prompt plus answer accepted, in tokens.")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-per-view", type=int, default=200)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--state-every", type=int, default=50)
    parser.add_argument("--framings", action="store_true",
                        help="Draw a presentation of the stated norm each time a norm view is trained on.")
    parser.add_argument("--distill-weight", type=float, default=0.0,
                        help="Weight of the knockout-teacher distillation loss; 0 turns it off.")
    parser.add_argument("--distill-activation-weight", type=float, default=0.0,
                        help="Weight of matching the teacher's state at the last prompt token.")
    parser.add_argument("--distill-layer", type=int, default=16)
    parser.add_argument("--gate-weight", type=float, default=0.0,
                        help="Weight of the attention-gating loss; 0 turns it off.")
    parser.add_argument("--gate-heads", nargs="+", default=None, metavar="LAYER.HEAD")
    parser.add_argument("--mirror", action="store_true",
                        help="Train the three tasks of mirror verification (finetune.mirror).")
    parser.add_argument("--country-memory", type=int, default=0, metavar="K",
                        help="Trainable vectors per country, read after the country line; 0 is off.")
    parser.add_argument("--memory-token", default=MEMORY_TOKEN,
                        help="The spare token that marks a memory position.")
    parser.add_argument("--memory-learning-rate", type=float, default=1e-3)
    parser.add_argument("--value-first", action="store_true",
                        help="Also train the model to write the value at stake (finetune.value_first).")
    parser.add_argument("--propose", action="store_true",
                        help="With --mirror: also train the model to write the norm that applies.")
    parser.add_argument("--task-tokens", type=float, default=300.0,
                        help="With --mirror: how many answer tokens one mirror or compare "
                             "example counts as in the batch's token mean.")
    parser.add_argument("--aux-records", type=int, default=4,
                        help="Records per step for the distillation and gating losses "
                             "(one true-norm and one contradictory-norm prompt each).")
    parser.add_argument("--resume", action="store_true",
                        help="Continue the run in --output-dir from its last saved state.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args(argv)
    args.views = MIRROR_VIEWS if args.mirror else training_views(args.counterfactual)
    if args.method is None:
        args.method = (("sft-mirror-v2" if args.propose else "sft-mirror") if args.mirror
                       else method_name(args.counterfactual, args.constrained))
    if args.output_dir is None:
        model_tag = args.model.rstrip("/").split("/")[-1].lower()
        args.output_dir = Path("output/finetune/runs") / model_tag / args.method
    if args.batch_size < 1 or args.micro_batch_size < 1 or args.epochs <= 0:
        parser.error("--batch-size, --micro-batch-size and --epochs must be positive")
    if not 0 < args.dev_fraction < 0.5:
        parser.error("--dev-fraction must be between 0 and 0.5")
    if args.lora_layers and args.lora_heads:
        parser.error("--lora-layers and --lora-heads are alternatives")
    if args.mirror and (args.constrained or args.framings or args.conflict_weight or args.gate_weight
                        or args.distill_weight or args.distill_activation_weight):
        parser.error("--mirror is a recipe of its own; it does not combine with the other arms")
    if args.country_memory < 0:
        parser.error("--country-memory cannot be negative")
    if args.country_memory and (args.mirror or args.value_first or args.constrained or args.framings
                                or args.conflict_weight or args.gate_weight or args.distill_weight
                                or args.distill_activation_weight):
        parser.error("--country-memory is a recipe of its own; it does not combine with the other arms")
    if args.value_first and (args.mirror or args.constrained):
        parser.error("--value-first does not combine with --mirror or --constrained")
    if args.propose and not args.mirror:
        parser.error("--propose needs --mirror")
    if args.task_tokens <= 0:
        parser.error("--task-tokens must be positive")
    if args.lora_heads:
        parse_heads(args.lora_heads)
    if args.gate_weight:
        if not args.gate_heads:
            parser.error("--gate-weight needs --gate-heads")
        parse_heads(args.gate_heads)
    if min(args.distill_weight, args.distill_activation_weight, args.gate_weight) < 0 or args.aux_records < 1:
        parser.error("loss weights cannot be negative and --aux-records must be positive")
    if args.conflict_weight < 0 or args.conflict_records < 1:
        parser.error("--conflict-weight cannot be negative and --conflict-records must be positive")
    return args


def parse_heads(items: list[str]) -> dict[int, list[int]]:
    """``["13.4", "13.22", "14.5"] -> {13: [4, 22], 14: [5]}``."""

    heads: dict[int, list[int]] = {}
    for item in items:
        if not re.fullmatch(r"\d+\.\d+", item):
            raise ValueError(f"A head is written LAYER.HEAD, not {item!r}")
        layer, head = (int(part) for part in item.split("."))
        if head in heads.setdefault(layer, []):
            raise ValueError(f"Head {item} is listed twice")
        heads[layer].append(head)
    return {layer: sorted(found) for layer, found in sorted(heads.items())}


def head_mask(name: str, shape: tuple[int, ...], heads: dict[int, list[int]], head_count: int
              ) -> list[bool] | None:
    """For one adapter matrix, which rows (the query projection's ``lora_B``)
    or columns (the output projection's ``lora_A``) belong to the selected
    heads; ``None`` when the matrix is not sliced by head.

    The query projection writes one block of rows per head, and the output
    projection reads one block of columns per head, so those are the two
    places a low-rank update can be confined to a head.
    """

    found = re.search(r"layers\.(\d+)\.self_attn\.(q_proj|o_proj)\.(lora_[AB])\.", name)
    if not found:
        return None
    layer, module, part = int(found.group(1)), found.group(2), found.group(3)
    if (module, part) not in {("q_proj", "lora_B"), ("o_proj", "lora_A")}:
        return None
    size = shape[0] if part == "lora_B" else shape[1]
    width, rest = divmod(size, head_count)
    if rest:
        raise ValueError(f"{name} has {size} units, not a multiple of {head_count} heads")
    chosen = set(heads.get(layer, ()))
    return [index // width in chosen for index in range(size)]


# -------------------------------------------------------------------- data


def load_gold(path: Path) -> dict[str, str]:
    """``norm_id -> answer`` from one gold file; empty when the file is absent."""

    if not path.is_file():
        return {}
    gold: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            answer = str(row.get("response") or "").strip()
            if answer:
                gold[str(row["norm_id"])] = answer
    return gold


def constrained_target(answer: str, shuffled: bool) -> str:
    """The A/B answer in the format the constrained system prompt asks for."""

    return f"<reasoning>{answer.strip()}</reasoning>\n<answer>{gold_letter(shuffled).upper()}</answer>"


def resolve(example: dict[str, Any], seed: int, epoch: int | None,
            framings: bool = False) -> dict[str, Any]:
    """The prompt, target and system prompt an example is trained or scored on.

    Open-ended examples are fixed unless ``framings`` is on. An A/B example
    draws its option order from the seed, the epoch and the item, so it changes
    between epochs and is the same when a run is resumed; ``epoch=None`` gives
    the evaluation's fixed order. With ``framings``, the presentation of the
    stated norm is drawn the same way; ``epoch=None`` keeps the plain wording.
    """

    if example["kind"] in ("mirror", "propose", "value"):
        return example
    record, view = example["record"], example["view"]
    if example["kind"] == "compare":
        # The two copies of a record always show the two orders, and which copy
        # shows which is drawn per epoch (fixed for development).
        flipped = (is_shuffled(record["norm_id"]) if epoch is None else
                   random.Random(f"{seed}:compare:{epoch}:{record['norm_id']}").random() < 0.5)
        true_first = flipped == bool(example["copy"])
        true, false = record["scenario"]["norm_evidence"], record["scenario"]["opposite_norm_evidence"]
        first, second = (true, false) if true_first else (false, true)
        return {**example, "prompt": compare_message(record, first, second),
                "response": "1" if true_first else "2", "system": COMPARE_SYSTEM}
    framing = "plain"
    if framings and epoch is not None:
        framing = random.Random(
            f"{seed}:framing:{epoch}:{record['norm_id']}:{view}:{example['kind']}").choice(FRAMINGS)
    if example["kind"] == "open":
        return (example if framing == "plain" else
                {**example, "prompt": open_message(record, view, framing)})
    shuffled = (is_shuffled(record["norm_id"]) if epoch is None else
                random.Random(f"{seed}:order:{epoch}:{record['norm_id']}:{view}").random() < 0.5)
    return {**example, "prompt": constrained_message(record, view, shuffled=shuffled, framing=framing),
            "response": constrained_target(example["answer"], shuffled),
            "system": CONSTRAINED_SYSTEM_PROMPT}


REQUEST_VIEW = "request"


def request_examples(path: Path | None, gold_path: Path, keep: set[str]) -> list[dict[str, Any]]:
    """One open-ended example per request record that is in ``keep`` and has a
    gold answer. The request stands where a contradictory norm would, so the
    prompt is that view's; the target does what was asked."""

    if path is None:
        return []
    gold = load_gold(gold_path)
    if not gold:
        raise FileNotFoundError(f"No request answers at {gold_path}")
    return [{"norm_id": record["norm_id"], "view": REQUEST_VIEW, "kind": "open", "record": record,
             "prompt": open_message(record, FALSE_NORM), "response": gold[str(record["norm_id"])]}
            for record in load_records(path)
            if str(record["norm_id"]) in keep and str(record["norm_id"]) in gold]


def build_examples(records: list[dict[str, Any]], views: tuple[str, ...],
                   gold: dict[str, str], gold_cf: dict[str, str], constrained: bool = False,
                   mirror: bool = False, propose: bool = False, value_first: bool = False
                   ) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """One example per record and view that has a prompt and a gold answer, and
    with ``constrained`` an A/B example beside each (see ``resolve``).

    With ``mirror``, each record also gives two mirror examples (the true norm
    to the contradictory one and back) and two compare examples (one per order).
    With ``propose``, each record also gives the documented norm as the target
    of a prompt that states none, once per such view the record has.
    With ``value_first``, each record that has value evidence also gives it as
    the target of its country-only prompt.

    Returns the examples and, per view, how many records had none.
    """

    examples: list[dict[str, Any]] = []
    missing = {view: 0 for view in views}
    for record in records:
        for view in views:
            if not eligible(record, view):
                continue
            answer = (gold_cf if view == FALSE_NORM else gold).get(record["norm_id"])
            if answer is None:
                missing[view] += 1
                continue
            examples.append({"norm_id": record["norm_id"], "view": view, "kind": "open",
                             "record": record, "prompt": open_message(record, view), "response": answer})
            if constrained:
                examples.append({"norm_id": record["norm_id"], "view": view, "kind": "constrained",
                                 "record": record, "answer": answer})
        if mirror:
            true, false = record["scenario"]["norm_evidence"], record["scenario"]["opposite_norm_evidence"]
            for source, target in ((true, false), (false, true)):
                examples.append({"norm_id": record["norm_id"], "view": "mirror", "kind": "mirror",
                                 "record": record, "prompt": mirror_message(source),
                                 "response": target, "system": MIRROR_SYSTEM})
            for copy in (0, 1):
                examples.append({"norm_id": record["norm_id"], "view": "compare", "kind": "compare",
                                 "record": record, "copy": copy})
        if value_first and record["scenario"].get("value_evidence"):
            examples.append({"norm_id": record["norm_id"], "view": "value", "kind": "value",
                             "record": record, "prompt": value_message(record),
                             "response": record["scenario"]["value_evidence"], "system": VALUE_SYSTEM})
        if propose:
            for view in PROPOSE_VIEWS:
                if eligible(record, view):
                    examples.append({"norm_id": record["norm_id"], "view": "propose", "kind": "propose",
                                     "record": record, "prompt": propose_message(record, view),
                                     "response": record["scenario"]["norm_evidence"],
                                     "system": PROPOSE_SYSTEM})
    return examples, missing


def hold_out(records: list[dict[str, Any]], fraction: float, seed: int
             ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Training and development records, split by record so no item is in both."""

    order = list(records)
    random.Random(f"{seed}:dev").shuffle(order)
    count = max(1, round(len(order) * fraction))
    held = {row["norm_id"] for row in order[:count]}
    return ([row for row in records if row["norm_id"] not in held],
            [row for row in records if row["norm_id"] in held])


def epoch_order(count: int, epoch: int, seed: int) -> list[int]:
    """The example order of one epoch: a function of the seed and epoch only,
    so a resumed run walks the same sequence."""

    order = list(range(count))
    random.Random(f"{seed}:epoch:{epoch}").shuffle(order)
    return order


def learning_rate_scale(step: int, total: int, warmup: int) -> float:
    """Linear warm-up, then cosine decay to zero; ``step`` counts from 1."""

    if step <= warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def _plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


# -------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

    from .losses import token_logprobs
    from .tokenization import ChatEncoder

    output_dir: Path = args.output_dir
    state_dir = output_dir / "state"
    resuming = args.resume and (state_dir / "state.json").is_file()
    if output_dir.exists() and any(output_dir.iterdir()) and not resuming:
        raise FileExistsError(f"Refusing to overwrite {output_dir}; pass --resume to continue it")
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.jsonl"
    set_seed(args.seed)

    def log(payload: dict[str, Any]) -> None:
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
        print(json.dumps(payload), flush=True)

    # ------------------------------------------------------------- data
    def gold_for(split: Path) -> tuple[dict[str, str], dict[str, str]]:
        stem = split.stem
        return (load_gold(args.gold_dir / f"{stem}.{GOLD_FILE[False]}.jsonl"),
                load_gold(args.gold_dir / f"{stem}.{GOLD_FILE[True]}.jsonl"))

    train_records = load_records(args.train)
    gold, gold_cf = gold_for(args.train)
    if not gold:
        raise FileNotFoundError(f"No gold answers for {args.train.stem} under {args.gold_dir}")
    if args.counterfactual and not gold_cf:
        raise FileNotFoundError(f"No false-norm gold answers for {args.train.stem} under {args.gold_dir}")

    dev_source = "validation gold"
    dev_examples: list[dict[str, Any]] = []
    dev_records: list[dict[str, Any]] = []
    if args.validation:
        validation_gold, validation_gold_cf = gold_for(args.validation)
        dev_records = load_records(args.validation)
        dev_examples, _ = build_examples(dev_records, args.views, validation_gold,
                                         validation_gold_cf, args.constrained, args.mirror, args.propose,
                                         args.value_first)
    if not dev_examples:
        train_records, dev_records = hold_out(train_records, args.dev_fraction, args.seed)
        dev_examples, _ = build_examples(dev_records, args.views, gold, gold_cf, args.constrained,
                                         args.mirror, args.propose, args.value_first)
        dev_source = f"{len(dev_records)} training records held out (no validation gold found)"
    examples, missing = build_examples(train_records, args.views, gold, gold_cf, args.constrained,
                                       args.mirror, args.propose, args.value_first)
    if not examples:
        raise ValueError("No training example has both a prompt and a gold answer")
    # Requests: only for records that are trained on (never the held-out ones).
    examples += request_examples(args.requests, args.gold_dir / f"{args.train.stem}.{REQUEST_VIEW}.jsonl",
                                 {str(record["norm_id"]) for record in train_records})
    if args.validation and args.validation_requests:
        dev_examples += request_examples(
            args.validation_requests, args.gold_dir / f"{args.validation.stem}.{REQUEST_VIEW}.jsonl",
            {str(record["norm_id"]) for record in dev_records})
    # The same development items at every evaluation, capped per view.
    # With both tasks, each view is scored once per task ("country_only" and
    # "country_only:constrained"), A/B examples in the evaluation's fixed order.
    slot = lambda example: (f"{example['view']}:constrained"  # noqa: E731
                            if example["kind"] == "constrained" else example["view"])
    dev_by_view: dict[str, list[dict[str, Any]]] = {
        name: [] for view in args.views
        for name in ((view, f"{view}:constrained") if args.constrained else (view,))}
    if args.mirror:
        dev_by_view.update({task: [] for task in ((*MIRROR_TASKS, "propose") if args.propose
                                                  else MIRROR_TASKS)})
    if args.value_first:
        dev_by_view["value"] = []
    if args.requests:
        dev_by_view[REQUEST_VIEW] = []
    random.Random(f"{args.seed}:dev-order").shuffle(dev_examples)
    def placed(example: dict[str, Any]) -> dict[str, Any]:
        """The example with memory positions marked in its prompt, when the memory is on."""

        if not args.country_memory:
            return example
        return {**example, "prompt": with_memory(example["prompt"], example["record"],
                                                 args.country_memory, args.memory_token)}

    for example in dev_examples:
        if len(dev_by_view[slot(example)]) < args.eval_per_view or args.eval_per_view <= 0:
            dev_by_view[slot(example)].append(placed(resolve(example, args.seed, None)))

    steps_per_epoch = math.ceil(len(examples) / args.batch_size)
    total_steps = math.ceil(args.epochs * len(examples) / args.batch_size)
    warmup = round(args.warmup_fraction * total_steps)
    per_view = {name: sum(slot(example) == name for example in examples) for name in dev_by_view}
    log({"event": "resume" if resuming else "plan", "schema": SCHEMA, "method": args.method,
         "counterfactual": args.counterfactual, "constrained": args.constrained,
         "views": list(args.views),
         "examples": len(examples), "examples_by_view": per_view, "without_gold": missing,
         "epochs": args.epochs, "steps": total_steps, "steps_per_epoch": steps_per_epoch,
         "batch_size": args.batch_size, "learning_rate": args.learning_rate, "warmup_steps": warmup,
         "lora_layers": args.lora_layers, "lora_heads": args.lora_heads,
         "country_memory": args.country_memory,
         "mirror": args.mirror, "propose": args.propose, "value_first": args.value_first,
         "task_tokens": args.task_tokens if (args.mirror or args.value_first) else None,
         "requests": str(args.requests) if args.requests else None,
         "framings": args.framings, "distill_weight": args.distill_weight,
         "distill_activation_weight": args.distill_activation_weight,
         "gate_weight": args.gate_weight, "gate_heads": args.gate_heads,
         "conflict_weight": args.conflict_weight,
         "conflict_layer": args.conflict_layer if args.conflict_weight else None,
         "selection": "lowest held-out answer loss, mean over views",
         "development": dev_source,
         "development_by_view": {view: len(rows) for view, rows in dev_by_view.items()}})

    # ------------------------------------------------------------ model
    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("No GPU is visible; pass --device cpu only for a smoke test")
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    encoder = ChatEncoder(tokenizer)
    # Attention weights are only returned by the eager implementation.
    attention = {"attn_implementation": "eager"} if args.gate_weight else {}
    base = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=getattr(torch, args.dtype), device_map={"": device},
        local_files_only=args.local_files_only, **attention)
    base.config.use_cache = False
    heads = parse_heads(args.lora_heads) if args.lora_heads else {}
    placement: dict[str, Any] = {}
    if heads or args.lora_layers:
        placement = {"layers_to_transform": sorted(heads) if heads else sorted(set(args.lora_layers)),
                     "layers_pattern": "layers"}
    model = get_peft_model(base, LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=list(HEAD_TARGETS if heads else LORA_TARGETS), bias="none",
        task_type="CAUSAL_LM", **placement), adapter_name=POLICY)
    for name, parameter in model.named_parameters():
        if "lora_" in name and parameter.dtype != torch.float32:
            parameter.data = parameter.data.float()

    def confine_to_heads() -> int:
        """Zero what lies outside the selected heads and keep it at zero: the
        gradient there is masked, and the optimiser has no weight decay."""

        confined = 0
        for name, parameter in model.named_parameters():
            if not parameter.requires_grad:
                continue
            keep = head_mask(name, tuple(parameter.shape), heads, base.config.num_attention_heads)
            if keep is None:
                continue
            mask = torch.tensor(keep, dtype=parameter.dtype, device=parameter.device)
            mask = mask[:, None] if ".lora_B." in name else mask[None, :]
            parameter.data.mul_(mask)
            parameter.register_hook(lambda gradient, mask=mask: gradient * mask)
            confined += 1
        return confined

    if heads and confine_to_heads() != 2 * len(heads):
        raise RuntimeError("Did not find one query and one output projection per selected layer")
    trainable = [p for p in model.parameters() if p.requires_grad]
    log({"event": "adapter", "trainable_parameters": sum(p.numel() for p in trainable),
         "adapted_modules": sum(".lora_A." in name for name, _ in model.named_parameters())})

    # The conflict probe reads the interaction between country and stated norm.
    probe = None
    groups = [{"params": trainable, "lr": args.learning_rate, "base_lr": args.learning_rate}]
    if args.conflict_weight:
        probe = torch.nn.Linear(base.config.hidden_size, 1).to(device=device, dtype=torch.float32)
        torch.nn.init.zeros_(probe.weight)
        torch.nn.init.zeros_(probe.bias)
        groups.append({"params": list(probe.parameters()), "lr": args.conflict_learning_rate,
                       "base_lr": args.conflict_learning_rate})
    # Country memory: vectors per training country, swapped in by the embedding layer.
    memory = None
    if args.country_memory:
        from .country_memory import initial_vectors, make_memory, placeholder_id
        from .data import iso3

        memory = make_memory(torch, sorted({iso3(record) for record in train_records}),
                             initial_vectors(torch, base, tokenizer, args.country_memory),
                             placeholder_id(tokenizer, args.memory_token)).to(device)
        base.get_input_embeddings().register_forward_hook(memory.hook)
        groups.append({"params": [memory.table], "lr": args.memory_learning_rate,
                       "base_lr": args.memory_learning_rate})
        log({"event": "memory", "countries": len(memory.countries), "slots": args.country_memory,
             "parameters": memory.table.numel()})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)

    def encode(batch: list[dict[str, Any]]):
        """Right-padded ids, mask and prompt lengths for a batch of examples."""

        rows = [encoder.encode(example["prompt"], example["response"], example.get("system"))
                for example in batch]
        width = max(len(ids) for ids, _ in rows)
        if width > args.max_tokens:
            raise ValueError(f"An example of {width} tokens exceeds --max-tokens {args.max_tokens}")
        ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for row, (tokens, _) in enumerate(rows):
            ids[row, :len(tokens)] = torch.tensor(tokens)
            mask[row, :len(tokens)] = 1
        lengths = torch.tensor([start for _, start in rows])
        return ids.to(device), mask.to(device), lengths.to(device)

    def answer_loss(batch: list[dict[str, Any]], weights: list[float] | None = None):
        """Summed negative log-likelihood of the answer tokens, and their count.
        With ``weights``, each example's sum is multiplied by its weight."""

        ids, mask, lengths = encode(batch)
        if memory is not None:
            memory.rows = [example["norm_id"].split("__")[0] for example in batch]
        logits = model(input_ids=ids, attention_mask=mask).logits
        if memory is not None:
            memory.rows = None
        logprobs, answer_mask = token_logprobs(logits, ids, lengths, mask)
        if weights is None:
            return -logprobs.sum(), int(answer_mask.sum())
        scale = torch.tensor(weights, device=logprobs.device, dtype=logprobs.dtype)
        return -(logprobs.sum(dim=1) * scale).sum(), int(answer_mask.sum())

    class _Stop(Exception):
        pass

    def states_at(prompts: list[str], layer: int):
        """The state at each prompt's last token after decoder block ``layer``.
        The forward pass stops there: later blocks are not needed."""

        rows = [encoder._prompt(prompt, None)[1] for prompt in prompts]
        width = max(len(tokens) for tokens in rows)
        ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        for row, tokens in enumerate(rows):
            ids[row, :len(tokens)] = torch.tensor(tokens)
            mask[row, :len(tokens)] = 1
        last = torch.tensor([len(tokens) - 1 for tokens in rows], device=device)
        caught: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            caught["state"] = output[0] if isinstance(output, tuple) else output
            raise _Stop

        handle = base.model.layers[layer].register_forward_hook(hook)
        try:
            model(input_ids=ids.to(device), attention_mask=mask.to(device))
        except _Stop:
            pass
        finally:
            handle.remove()
        return caught["state"][torch.arange(len(rows), device=device), last].float()

    def conflict_loss(records: list[dict[str, Any]]):
        """Probe loss and accuracy on the two interactions of each record:
        label 1 for the contradictory norm, 0 for the true one."""

        prompts = [open_message(record, view) for record in records for view in CONFLICT_VIEWS]
        bare, country, norm, opposite, country_norm, country_opposite = states_at(
            prompts, args.conflict_layer).view(len(records), len(CONFLICT_VIEWS), -1).unbind(dim=1)
        interactions = torch.cat([country_norm - norm - country + bare,
                                  country_opposite - opposite - country + bare])
        # Unit length times sqrt(width): the probe reads direction, so the loss
        # cannot be lowered by inflating the interaction.
        interactions = torch.nn.functional.normalize(interactions, dim=-1) * interactions.shape[-1] ** 0.5
        labels = torch.cat([torch.zeros(len(records)), torch.ones(len(records))]).to(device)
        logits = probe(interactions).squeeze(-1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return loss, float(((logits > 0).float() == labels).float().mean())

    # ---- losses that need to know where the stated norm sits in the prompt
    from mechanistic_analysis.interventions.attention import blocked, token_spans

    NORM_TEXT = {"country_norm": "norm_evidence", FALSE_NORM: "opposite_norm_evidence"}
    decoder_layers = base.model.layers
    gate_heads = parse_heads(args.gate_heads) if args.gate_weight else {}

    def encode_marked(items: list[tuple[dict[str, Any], str, str]]):
        """Padded ids and mask for (record, view, answer) items, with each row's
        prompt length, the index of the first token after the user's text, and
        a mask over the tokens of the stated norm."""

        rows = []
        for record, view, answer in items:
            message = open_message(record, view)
            prompt_ids, after, spans = token_spans(
                tokenizer, encoder.render_prompt(message), message,
                {"norm": str(record["scenario"][NORM_TEXT[view]])})
            ids, start = encoder.encode(message, answer)
            if ids[:start] != prompt_ids:
                raise ValueError("The marked prompt does not tokenize like the training prompt")
            rows.append((ids, start, after, spans["norm"]))
        width = max(len(ids) for ids, *_ in rows)
        ids = torch.full((len(rows), width), tokenizer.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(rows), width), dtype=torch.long)
        norm = torch.zeros((len(rows), width), dtype=torch.bool)
        for row, (tokens, _, _, (first, last)) in enumerate(rows):
            ids[row, :len(tokens)] = torch.tensor(tokens)
            mask[row, :len(tokens)] = 1
            norm[row, first:last] = True
        return (ids.to(device), mask.to(device), torch.tensor([r[1] for r in rows], device=device),
                torch.tensor([r[2] for r in rows], device=device), norm.to(device))

    def state_hook(layer: int, store: dict[str, Any]):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            store["state"] = output[0] if isinstance(output, tuple) else output
        return decoder_layers[layer].register_forward_hook(hook)

    def distill_loss(records: list[dict[str, Any]]):
        """KL from the knockout teacher to the student over the answer tokens,
        and one minus the cosine between their states at the last prompt token."""

        items = [(record, view, gold[record["norm_id"]]) for record in records
                 for view in ("country_norm", FALSE_NORM)]
        false = torch.tensor([view == FALSE_NORM for _, view, _ in items], device=device)
        ids, mask, lengths, after, norm = encode_marked(items)
        rows = torch.arange(len(items), device=device)
        positions = torch.arange(ids.shape[1], device=device)[None, :]
        # Position j predicts token j + 1: the answer's tokens are predicted from
        # the last prompt token up to the one before the end.
        answer = (positions >= (lengths[:, None] - 1)) & (positions < mask.sum(1)[:, None] - 1)
        seen: dict[str, Any] = {}
        handle = state_hook(args.distill_layer, seen)
        try:
            with torch.no_grad(), model.disable_adapter(), blocked(
                    decoder_layers, list(range(len(decoder_layers))), after,
                    norm & false[:, None]):
                teacher = model(input_ids=ids, attention_mask=mask).logits[answer].float().log_softmax(-1)
            teacher_state = seen["state"][rows, lengths - 1].float()
            student = model(input_ids=ids, attention_mask=mask).logits[answer].float().log_softmax(-1)
            student_state = seen["state"][rows, lengths - 1].float()
        finally:
            handle.remove()
        kl = (teacher.exp() * (teacher - student)).sum(-1).mean()
        apart = 1.0 - torch.nn.functional.cosine_similarity(student_state, teacher_state, dim=-1).mean()
        return kl, apart

    def norm_attention(ids, mask, after, norm):
        """Per row: the share of attention the selected heads give the stated
        norm, averaged over the positions after the user's text and the heads."""

        caught: dict[int, Any] = {}
        handles = []
        for layer in gate_heads:
            def hook(_module: Any, _inputs: Any, output: Any, layer: int = layer) -> None:
                weights = [value for value in output if torch.is_tensor(value) and value.ndim == 4]
                if not weights:
                    raise RuntimeError("The attention module returned no weights; eager attention is needed")
                caught[layer] = weights[0]
            handles.append(decoder_layers[layer].self_attn.register_forward_hook(hook))
        try:
            model(input_ids=ids, attention_mask=mask)
        finally:
            for handle in handles:
                handle.remove()
        positions = torch.arange(ids.shape[1], device=device)[None, :]
        queries = ((positions >= after[:, None]) & mask.bool()).float()
        shares = []
        for layer, heads in gate_heads.items():
            on_norm = (caught[layer][:, heads].float() * norm[:, None, None, :]).sum(-1)   # rows, heads, queries
            shares.append((on_norm * queries[:, None, :]).sum(-1) / queries.sum(-1, keepdim=True).clamp_min(1))
        return torch.cat(shares, dim=1).mean(dim=1)

    def gate_loss(records: list[dict[str, Any]]):
        """Attention to a contradictory norm should vanish; attention to a true
        norm should stay where the untuned model has it."""

        items = [(record, view, (gold_cf if view == FALSE_NORM else gold)[record["norm_id"]])
                 for record in records for view in ("country_norm", FALSE_NORM)]
        false = torch.tensor([view == FALSE_NORM for _, view, _ in items], device=device)
        ids, mask, _, after, norm = encode_marked(items)
        with torch.no_grad(), model.disable_adapter():
            untuned = norm_attention(ids, mask, after, norm)
        share = norm_attention(ids, mask, after, norm)
        loss = share[false].mean() + (share[~false] - untuned[~false]).abs().mean()
        return loss, float(share[false].mean().detach()), float(share[~false].mean().detach())

    aux_pool = [record for record in train_records
                if record["norm_id"] in gold and record["norm_id"] in gold_cf]
    if (args.distill_weight or args.distill_activation_weight or args.gate_weight) and not aux_pool:
        raise ValueError("No training record has both gold answers the auxiliary losses need")

    conflict_pool = [record for record in train_records
                     if all(eligible(record, view) for view in CONFLICT_VIEWS)]
    conflict_dev = [record for record in dev_records
                    if all(eligible(record, view) for view in CONFLICT_VIEWS)]
    random.Random(f"{args.seed}:conflict-dev").shuffle(conflict_dev)
    conflict_dev = conflict_dev[:96]
    if probe is not None and not conflict_pool:
        raise ValueError("No training record has every prompt the conflict loss needs")

    def evaluate(step: int) -> float:
        model.eval()
        report: dict[str, Any] = {"event": "validation", "step": step, "loss": {}}
        with torch.no_grad():
            for view, rows in dev_by_view.items():
                total, tokens = 0.0, 0
                for start in range(0, len(rows), args.micro_batch_size):
                    loss, count = answer_loss(rows[start:start + args.micro_batch_size])
                    total += float(loss)
                    tokens += count
                if tokens:
                    report["loss"][view] = total / tokens
            if probe is not None and conflict_dev:
                losses, hits = [], []
                for start in range(0, len(conflict_dev), 8):
                    loss, accuracy = conflict_loss(conflict_dev[start:start + 8])
                    losses.append(float(loss))
                    hits.append(accuracy)
                report["conflict"] = {"loss": sum(losses) / len(losses),
                                      "accuracy": sum(hits) / len(hits), "records": len(conflict_dev)}
        model.train()
        report["selection_score"] = sum(report["loss"].values()) / max(1, len(report["loss"]))
        log(report)
        return report["selection_score"]

    def save(name: str, step: int, extra: dict[str, Any]) -> None:
        target = output_dir / name
        model.save_pretrained(str(target), selected_adapters=[POLICY])
        tokenizer.save_pretrained(str(target))
        if probe is not None:
            torch.save(probe.state_dict(), target / "conflict_probe.pt")
        if memory is not None:
            torch.save({"countries": memory.countries, "token": args.memory_token,
                        "table": memory.table.detach().float().cpu()}, target / MEMORY_FILE)
        (target / "training_config.json").write_text(json.dumps(
            {"schema": SCHEMA, "step": step, "method": args.method,
             "arguments": {k: _plain(v) for k, v in vars(args).items()}, **extra}, indent=2))

    def save_state(step: int, hours: float) -> None:
        """Written beside the live state and swapped in, so a job killed while
        saving leaves the previous state intact."""

        fresh = output_dir / "state.new"
        shutil.rmtree(fresh, ignore_errors=True)
        fresh.mkdir(parents=True)
        model.save_pretrained(str(fresh / "adapter"), selected_adapters=[POLICY])
        torch.save({"optimizer": optimizer.state_dict(), "torch": torch.get_rng_state(),
                    "probe": probe.state_dict() if probe is not None else None,
                    "memory": memory.table.detach().float().cpu() if memory is not None else None,
                    "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []},
                   fresh / "trainer.pt")
        (fresh / "state.json").write_text(json.dumps({
            "schema": SCHEMA, "step": step, "hours": hours, "best": best,
            "method": args.method, "total_steps": total_steps, "examples": len(examples)}))
        old = output_dir / "state.old"
        shutil.rmtree(old, ignore_errors=True)
        if state_dir.exists():
            state_dir.rename(old)
        fresh.rename(state_dir)
        shutil.rmtree(old, ignore_errors=True)

    # ------------------------------------------------------------- loop
    best = {"score": math.inf, "step": 0}
    first_step, elapsed = 1, 0.0
    if resuming:
        from peft.utils import load_peft_weights, set_peft_model_state_dict

        saved = json.loads((state_dir / "state.json").read_text())
        if (saved["method"], saved["total_steps"], saved["examples"]) != (
                args.method, total_steps, len(examples)):
            raise ValueError(f"{state_dir} was saved by a different run "
                             f"({saved['method']}, {saved['total_steps']} steps, "
                             f"{saved['examples']} examples)")
        set_peft_model_state_dict(
            model, load_peft_weights(str(state_dir / "adapter"), device="cpu"), adapter_name=POLICY)
        for name, parameter in model.named_parameters():
            if "lora_" in name and parameter.dtype != torch.float32:
                parameter.data = parameter.data.float()
        trainer = torch.load(state_dir / "trainer.pt", map_location="cpu", weights_only=False)
        optimizer.load_state_dict(trainer["optimizer"])
        if probe is not None:
            probe.load_state_dict(trainer["probe"])
        if memory is not None:
            memory.table.data.copy_(trainer["memory"].to(memory.table.device))
        torch.set_rng_state(trainer["torch"])
        if trainer["cuda"] and len(trainer["cuda"]) == torch.cuda.device_count():
            torch.cuda.set_rng_state_all(trainer["cuda"])
        best, elapsed, first_step = saved["best"], saved["hours"], saved["step"] + 1
        log({"event": "resumed", "from_step": saved["step"], "of": total_steps, "best": best})
    else:
        best = {"score": evaluate(0), "step": 0}

    model.train()
    started = time.time() - elapsed * 3600
    orders: dict[int, list[int]] = {}
    for step in range(first_step, total_steps + 1):
        # Positions are counted over the whole run, so a step's batch does not
        # depend on where the run was resumed.
        batch = []
        for position in range((step - 1) * args.batch_size, step * args.batch_size):
            epoch, offset = divmod(position, len(examples))
            if epoch not in orders:
                orders = {epoch: epoch_order(len(examples), epoch, args.seed)}
            batch.append(placed(resolve(examples[orders[epoch][offset]], args.seed, epoch, args.framings)))
        for group in optimizer.param_groups:
            group["lr"] = group.get("base_lr", args.learning_rate) * learning_rate_scale(step, total_steps, warmup)
        optimizer.zero_grad(set_to_none=True)
        chunks = [batch[start:start + args.micro_batch_size]
                  for start in range(0, len(batch), args.micro_batch_size)]
        # Answer tokens in the whole batch, counted first so each chunk's
        # gradient is already its share of the batch's token mean and no graph
        # is kept between chunks.
        # A mirror or compare example has a target of a few tokens; it is weighted
        # to count as --task-tokens of them, in the sum and in the denominator.
        tokens, weights = 0.0, []
        for example in batch:
            ids, start = encoder.encode(example["prompt"], example["response"], example.get("system"))
            count = len(ids) - start
            weights.append(args.task_tokens / count if example["kind"] in TASK_KINDS else 1.0)
            tokens += weights[-1] * count
        sums = []
        for index, chunk in enumerate(chunks):
            first = index * args.micro_batch_size
            loss, _ = answer_loss(chunk, weights[first:first + len(chunk)]
                                  if (args.mirror or args.value_first) else None)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite loss at step {step}")
            (loss / tokens).backward()
            sums.append(float(loss.detach()))
        conflict: dict[str, float] = {}
        if probe is not None:
            drawn = random.Random(f"{args.seed}:conflict:{step}").sample(
                conflict_pool, min(args.conflict_records, len(conflict_pool)))
            loss, accuracy = conflict_loss(drawn)
            (args.conflict_weight * loss).backward()
            conflict = {"conflict_loss": float(loss.detach()), "conflict_accuracy": accuracy}
        if args.distill_weight or args.distill_activation_weight or args.gate_weight:
            drawn = random.Random(f"{args.seed}:aux:{step}").sample(
                aux_pool, min(args.aux_records, len(aux_pool)))
            if args.distill_weight or args.distill_activation_weight:
                kl, apart = distill_loss(drawn)
                (args.distill_weight * kl + args.distill_activation_weight * apart).backward()
                conflict.update(distill_kl=float(kl.detach()), distill_state=float(apart.detach()))
            if args.gate_weight:
                loss, on_false, on_true = gate_loss(drawn)
                (args.gate_weight * loss).backward()
                conflict.update(gate_loss=float(loss.detach()), attention_false=on_false,
                                attention_true=on_true)
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        if memory is not None:
            torch.nn.utils.clip_grad_norm_([memory.table], 1.0)
        optimizer.step()
        if step % 10 == 0 or step == total_steps:
            log({"event": "train", "step": step, "of": total_steps, "loss": sum(sums) / tokens,
                 **conflict,
                 "epoch": round(step * args.batch_size / len(examples), 3),
                 "learning_rate": optimizer.param_groups[0]["lr"],
                 "hours": round((time.time() - started) / 3600, 3)})

        if (args.eval_every and step % args.eval_every == 0) or step == total_steps:
            score = evaluate(step)
            if score < best["score"]:
                best = {"score": score, "step": step}
                save("best", step, {"selection_score": score})
        if args.save_every and step % args.save_every == 0:
            save(f"step-{step}", step, {})
        if args.state_every and step % args.state_every == 0:
            save_state(step, (time.time() - started) / 3600)

    if not (output_dir / "best" / "adapter_config.json").is_file():
        # Nothing beat the untuned model on held-out loss; keep the last adapter
        # so evaluation still has one, and say so.
        save("best", total_steps, {"selection_score": None, "note": "no step improved on step 0"})
    save("final", total_steps, {"best": best})
    save_state(total_steps, (time.time() - started) / 3600)
    (output_dir / "manifest.json").write_text(json.dumps(
        {"schema": SCHEMA, "method": args.method, "steps": total_steps, "best": best,
         "hours": round((time.time() - started) / 3600, 3),
         "arguments": {k: _plain(v) for k, v in vars(args).items()}}, indent=2))
    log({"event": "done", "best": best})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
