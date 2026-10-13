"""What does the model write when it cannot read the claim? (answers under knockout)

Generates the open-ended answer for each sampled record under a set of *arms*.
An arm is a prompt condition plus, optionally, a part of the prompt the answer
is stopped from attending to and the layers in which it is stopped. The answers
are written in the behavioural evaluation's format, one file per arm, so the
same judges score them and the outcome is the paper's own endpoint.

Arms (the layer windows are fixed in ``mechanistic_analysis/__init__.py``):

    false               country + false norm, untouched
    false-noclaim       ... the claim blocked in every layer
    false-noclaim-win   ... the claim blocked in the claim window only
    false-noclaim-late  ... the claim blocked in a later window of the same width
    false-nocountry     ... the country line blocked in every layer
    false-nocontrol     ... a same-length piece of the situation blocked in every layer
    true                country + true norm, untouched
    true-noclaim        ... the claim blocked in every layer
    country             country only, untouched
    claim-only          the false norm with no country line, untouched
    false-erase-first   country + false norm, the claim's positions erased after
                        the first block (``erasure.py``)
    false-erase-before  ... erased after a block before the band where the claim is read
    false-erase-after   ... erased after a block past that band
    false-noclaim-heads the claim blocked for the selected heads only (``heads.py``)
    false-noclaim-randheads  ... for as many randomly chosen heads of the window

``claim-only`` against ``false`` is a behavioural test that needs no
intervention: does naming the country help the model resist the false norm?

Decoding is greedy with the behavioural evaluation's token limit. Models whose
layers cannot all be masked run the untouched and the erasure arms; the head
arms need ``--heads`` (the selection ``heads.py`` wrote).

PYTHONPATH=src python -m mechanistic_analysis.interventions.generate \
  --model /path/to/model --model-slug llama-3.1-8b \
  --records data/processed/culturelens/records.json \
  --output-root output/mechanistic_analysis/answers
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from pathlib import Path

from mechanistic_analysis import CLAIM_WINDOW, CONTROL_WINDOW, ERASE_LAYERS
from mechanistic_analysis.activations.backend import require_torch
from mechanistic_analysis.activations.cache import sha256_file, write_manifest
from mechanistic_analysis.interventions.attention import blocked, maskable_layers
from mechanistic_analysis.interventions.erasure import erased
from mechanistic_analysis.interventions.features import steered
from mechanistic_analysis.interventions.common import (
    add_model_arguments, encode_prompt, load, sample_records, thinking_flag,
)

FALSE, TRUE = "country_opposite_norm", "country_norm"
# name: (prompt condition, part acted on or None, how). ``how`` is None (untouched),
# "all" | "claim" | "control" (attention blocked in those layers), "heads" |
# "random_heads" (blocked for those heads only), or "erase:<i>" (the part's
# positions erased after ERASE_LAYERS[model][i]).
ARMS: dict[str, tuple[str, str | None, str | None]] = {
    "false": (FALSE, None, None),
    "false-noclaim": (FALSE, "claim", "all"),
    "false-noclaim-win": (FALSE, "claim", "claim"),
    "false-noclaim-late": (FALSE, "claim", "control"),
    "false-nocountry": (FALSE, "country", "all"),
    "false-nocontrol": (FALSE, "control", "all"),
    "true": (TRUE, None, None),
    "true-noclaim": (TRUE, "claim", "all"),
    "country": ("country_only", None, None),
    "claim-only": ("opposite_norm_only", None, None),
    "false-erase-first": (FALSE, "claim", "erase:0"),
    "false-erase-before": (FALSE, "claim", "erase:1"),
    "false-erase-after": (FALSE, "claim", "erase:2"),
    "false-noclaim-heads": (FALSE, "claim", "heads"),
    "false-noclaim-randheads": (FALSE, "claim", "random_heads"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--arms", nargs="+", default=None,
                        help="Default: every arm the model supports.")
    parser.add_argument("--feature-plan", type=Path,
                        help="JSON with an 'arms' map: name -> {condition, layer, feature, mode "
                             "(ablate|clamp), value}. Each becomes an arm that sets that SAE "
                             "feature while the model answers.")
    parser.add_argument("--sae-root", type=Path, help="Local SAE checkpoints, for --feature-plan.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--heads", type=Path, help="selection.json written by heads.py.")
    return parser.parse_args(argv)


def arm_layers(arm: str, slug: str, n_layers: int) -> list[int] | None:
    """Layers an arm blocks attention in; None for arms that do not block by layer."""

    which = ARMS[arm][2]
    if which not in ("all", "claim", "control"):
        return None
    if which == "all":
        return list(range(n_layers))
    low, high = (CLAIM_WINDOW if which == "claim" else CONTROL_WINDOW)[slug]
    return list(range(low, high))


def supported_arms(slug: str, fully_maskable: bool, heads: bool = False) -> list[str]:
    def runs(which: str | None) -> bool:
        if which is None:
            return True
        if which.startswith("erase"):
            return slug in ERASE_LAYERS
        if which.startswith("feature"):
            return True
        if which in ("heads", "random_heads"):
            return fully_maskable and heads
        return fully_maskable and (which == "all" or slug in CLAIM_WINDOW)

    return [arm for arm, (_, _, which) in ARMS.items() if runs(which)]


def left_padded(prompts: list[list[int]], pad_id: int) -> tuple[Any, Any, list[int]]:
    """Ids and mask with every row flush right, and each row's padding count.

    One extra padding position is added to every row, so the batch always
    contains padding and the model builds an explicit attention mask, which the
    knockout edits.
    """

    torch = require_torch()
    width = max(len(prompt) for prompt in prompts) + 1
    ids = torch.full((len(prompts), width), pad_id, dtype=torch.long)
    mask = torch.zeros((len(prompts), width), dtype=torch.long)
    pads = []
    for row, prompt in enumerate(prompts):
        pads.append(width - len(prompt))
        ids[row, pads[-1]:] = torch.tensor(prompt)
        mask[row, pads[-1]:] = 1
    return ids, mask, pads


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    started = time.time()
    records = sample_records(args.records, args.sample, args.seed)
    loaded = load(args)
    thinking = thinking_flag(args)
    tokenizer = loaded.tokenizer
    n_layers = len(loaded.decoder_layers)
    fully = len(maskable_layers(loaded.decoder_layers)) == n_layers
    selection = json.loads(args.heads.read_text(encoding="utf-8")) if args.heads else None
    plan = json.loads(args.feature_plan.read_text(encoding="utf-8"))["arms"] if args.feature_plan else {}
    for name, spec in plan.items():
        ARMS[name] = (spec["condition"], None, f"feature:{name}")
    saes: dict[int, Any] = {}
    possible = supported_arms(args.model_slug, fully, selection is not None)
    arms = args.arms or possible
    unsupported = sorted(set(arms) - set(possible))
    if unsupported:
        raise ValueError(f"{args.model_slug} cannot run {unsupported}")
    device = loaded.model.get_input_embeddings().weight.device
    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    print(f"model={args.model_slug} sample={len(records)} arms={arms} "
          f"fully_maskable={fully}", flush=True)

    counts: dict[str, int] = {}
    for arm in arms:
        condition, part, how = ARMS[arm]
        layers = arm_layers(arm, args.model_slug, n_layers)
        path = output / f"{arm}.jsonl"
        done = set()
        if path.exists():
            done = {json.loads(line)["norm_id"] for line in path.read_text(encoding="utf-8").splitlines()
                    if line.strip()}
        pending = [record for record in records if str(record["norm_id"]) not in done]
        encoded = [encode_prompt(loaded, record, condition, thinking) for record in pending]
        order = np.argsort([len(item[0]) for item in encoded], kind="stable")
        for start in range(0, len(order), args.batch_size):
            if args.max_seconds and time.time() - started > args.max_seconds:
                print(f"time limit reached in arm {arm}", flush=True)
                break
            rows = order[start:start + args.batch_size]
            ids, mask, pads = left_padded([encoded[row][0] for row in rows], tokenizer.pad_token_id)
            call = dict(input_ids=ids.to(device), attention_mask=mask.to(device),
                        max_new_tokens=args.max_new_tokens, do_sample=False,
                        pad_token_id=tokenizer.pad_token_id)
            def marked(name: str) -> Any:
                keys = torch.zeros(ids.shape, dtype=torch.bool)
                for position, (pad, row) in enumerate(zip(pads, rows)):
                    first, last = encoded[row][2][name]
                    keys[position, pad + first:pad + last] = True
                return keys

            with torch.no_grad():
                if how is None:
                    generated = loaded.model.generate(**call)
                elif how.startswith("feature"):
                    spec = plan[arm]
                    if spec["layer"] not in saes:
                        from mechanistic_analysis.sae.backend import load_sae
                        from mechanistic_analysis.sae.registry import get_spec

                        saes[spec["layer"]] = load_sae(get_spec(args.model_slug), spec["layer"],
                                                       root=args.sae_root, device="cpu", dtype="float32")
                    start = torch.tensor([pad + encoded[row][1] for pad, row in zip(pads, rows)])
                    with steered(loaded.decoder_layers, spec["layer"], saes[spec["layer"]],
                                 spec["feature"], spec["mode"], spec.get("value", 0.0), start,
                                 ids.shape[1]):
                        generated = loaded.model.generate(**call)
                elif how.startswith("erase"):
                    layer = ERASE_LAYERS[args.model_slug][int(how.split(":")[1])]
                    with erased(loaded.decoder_layers, layer, marked(part), marked("situation")):
                        generated = loaded.model.generate(**call)
                else:
                    query = torch.tensor([pad + encoded[row][1] for pad, row in zip(pads, rows)])
                    chosen = None
                    if how in ("heads", "random_heads"):
                        chosen = {int(layer): heads for layer, heads in
                                  selection[how].items()}
                    with blocked(loaded.decoder_layers, layers or [], query, marked(part),
                                 heads=chosen):
                        generated = loaded.model.generate(**call)
            with path.open("a", encoding="utf-8") as handle:
                for position, row in enumerate(rows):
                    record = pending[row]
                    text = tokenizer.decode(generated[position, ids.shape[1]:],
                                            skip_special_tokens=True).strip()
                    iso3, category = str(record["norm_id"]).split("__")[:2]
                    handle.write(json.dumps({
                        "norm_id": record["norm_id"], "iso3": iso3, "category": category,
                        "condition": condition, "arm": arm, "response": text,
                    }, ensure_ascii=False) + "\n")
        counts[arm] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) \
            if path.exists() else 0
        print(f"arm {arm}: {counts[arm]}/{len(records)} answers  {time.time() - started:.0f}s",
              flush=True)

    # Arms written by earlier runs stay in the manifest.
    earlier = {}
    if (output / "metrics.json").is_file():
        earlier = json.loads((output / "metrics.json").read_text(encoding="utf-8")).get("arms", {})
    write_manifest({
        "analysis": "answers_under_attention_knockout",
        "model_slug": args.model_slug,
        "dataset_sha256": sha256_file(args.records),
        "sample": len(records),
        "sample_rule": f"uniform without replacement over all records, seed {args.seed}",
        "arms": {**earlier, **{arm: {"condition": ARMS[arm][0], "part": ARMS[arm][1], "how": ARMS[arm][2],
                       "layers": arm_layers(arm, args.model_slug, n_layers),
                       "answers": counts[arm]} for arm in arms}},
        "erase_layers": ERASE_LAYERS.get(args.model_slug),
        "heads": selection,
        "feature_arms": plan or None,
        "queries": "every position after the user's text: chat-template tail and answer tokens",
        "decoding": {"sampling": False, "max_new_tokens": args.max_new_tokens},
        "fully_maskable": fully,
        "seconds": round(time.time() - started, 1),
    }, output / "metrics.json")
    complete = all(counts[arm] == len(records) for arm in arms)
    print(f"saved {output}  complete={complete}  {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
