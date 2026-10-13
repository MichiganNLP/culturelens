"""Country memory: learned vectors per country, placed where a stated norm goes.

Supervised fine-tuning learns one shift for every country: the tuned model
answers almost as well with no country named. Here the model is given a place
to keep what is specific to a country. ``--country-memory K``
(``finetune.sft``) adds ``K`` trainable vectors for every training country.
A prompt that names a country gets ``K`` placeholder tokens directly after the
country line, the position a stated norm occupies, and the embedding layer
swaps those tokens for the country's vectors. Everything after the embedding
layer is the ordinary model with its adapter, so the vectors are read the way
a stated norm is read. They are trained with the adapter, on the same answers.

**Countries that were not trained on.** Their vectors are a weighted mean of
those of the nearest training countries. Distances are the cosine distances
between country vectors of the *untuned* model (``mechanistic_analysis.
country_space``, ``rdm_shift`` at one layer), which track survey-measured
cultural distance. They are computed from prompts alone, with no answer and no
label, but over all records, the evaluated ones included.

This module is the evaluation. The embedding swap needs the model itself, so
answers are generated with transformers, not vLLM:

    python -m finetune.country_memory --model <model> --lora runs/sft-cf-memory/best \
        --geometry output/mechanistic_analysis/country_space/llama-3.1-8b/geometry.npz \
        --input .../splits/test_unseen.json --output eval/sft-cf-memory/test_unseen/unconstrained.jsonl

``--no-memory`` sends the same prompts without the placeholder tokens, which
shows what the memory adds. Output rows are those of
``behavioral_evaluation.generate``, so the judge reads them unchanged.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

MEMORY_TOKEN = "<|reserved_special_token_100|>"   # a spare token of Llama-3.1; never in real text
MEMORY_FILE = "country_memory.pt"
INIT_TEXT = "The customs and norms that most people in this country follow"


def with_memory(prompt: str, record: dict[str, Any], slots: int, token: str = MEMORY_TOKEN) -> str:
    """``prompt`` with ``slots`` placeholder tokens after the country line.
    A prompt that does not name the country is returned unchanged."""

    line = record["scenario"]["country_context"]
    if slots <= 0 or line not in prompt:
        return prompt
    return prompt.replace(line, line + token * slots, 1)


def neighbour_weights(countries: list[str], distance: Any, known: list[str], target: str,
                      nearest: int = 5, temperature: float = 0.1) -> dict[str, float]:
    """Weights over the ``nearest`` known countries closest to ``target``:
    a softmax of minus distance over ``temperature``."""

    position = {code: index for index, code in enumerate(countries)}
    if target not in position:
        return {}
    ranked = sorted((float(distance[position[target]][position[code]]), code)
                    for code in known if code in position and code != target)[:nearest]
    if not ranked:
        return {}
    scores = [math.exp(-(value - ranked[0][0]) / temperature) for value, _ in ranked]
    return {code: score / sum(scores) for (_, code), score in zip(ranked, scores)}


def make_memory(torch: Any, countries: list[str], init: Any, token_id: int) -> Any:
    """The memory module. ``init`` is ``(slots, width)``: every country starts from it."""

    class CountryMemory(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.countries = list(countries)
            self.index = {code: row for row, code in enumerate(self.countries)}
            self.table = torch.nn.Parameter(
                init.detach().float().unsqueeze(0).repeat(len(self.countries), 1, 1))
            self.token_id = token_id
            self.rows: list[str] | None = None      # the country of each row of the next batch
            self.blends: dict[str, dict[str, float]] = {}

        def vectors(self, code: str) -> Any:
            if code in self.index:
                return self.table[self.index[code]]
            blend = self.blends.get(code)
            if blend:
                return sum(weight * self.table[self.index[other]] for other, weight in blend.items())
            return self.table.mean(dim=0)

        def hook(self, _module: Any, inputs: Any, output: Any) -> Any:
            """Forward hook of the embedding layer: placeholder positions take
            the vectors of the row's country."""

            ids = inputs[0]
            marked = ids == self.token_id
            if self.rows is None or not bool(marked.any()):
                return None
            if len(self.rows) != ids.shape[0]:
                raise RuntimeError("The memory was given countries for a different batch")
            swapped = output.clone()
            for row in marked.any(dim=1).nonzero().flatten().tolist():
                positions = marked[row].nonzero().flatten()
                vectors = self.vectors(self.rows[row])
                if len(positions) != vectors.shape[0]:
                    raise RuntimeError(f"A prompt has {len(positions)} memory tokens, not {vectors.shape[0]}")
                swapped[row, positions] = vectors.to(swapped.dtype)
            return swapped

    return CountryMemory()


def initial_vectors(torch: Any, model: Any, tokenizer: Any, slots: int) -> Any:
    """``slots`` word embeddings of a neutral phrase, cycled if it is shorter."""

    ids = tokenizer(INIT_TEXT, add_special_tokens=False)["input_ids"]
    chosen = torch.tensor([ids[index % len(ids)] for index in range(slots)],
                          device=model.get_input_embeddings().weight.device)
    return model.get_input_embeddings().weight[chosen].detach().float().cpu()


def placeholder_id(tokenizer: Any, token: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(token.strip())
    if token_id is None or token_id == tokenizer.unk_token_id:
        raise ValueError(f"The tokenizer has no token {token.strip()!r} to mark memory positions")
    if tokenizer(token.strip(), add_special_tokens=False)["input_ids"] != [token_id]:
        raise ValueError(f"{token.strip()!r} is not encoded as a single token")
    return int(token_id)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    parser.add_argument("--lora", type=Path, required=True,
                        help=f"Adapter directory; it must also hold {MEMORY_FILE}.")
    parser.add_argument("--geometry", type=Path, default=None,
                        help="geometry.npz of the untuned model, for countries not trained on.")
    parser.add_argument("--geometry-layer", type=int, default=16)
    parser.add_argument("--nearest", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--no-memory", action="store_true",
                        help="Send the prompts without memory tokens.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+", default=[
        "country_only", "country_value", "country_norm", "country_opposite_norm"])
    parser.add_argument("--dtype", choices=("bfloat16", "float32", "auto"), default="bfloat16")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    from behavioral_evaluation.generate import append_rows, eligible, make_row, read_records, successful
    from behavioral_evaluation.prompts import build_user_message
    from common.io import read_jsonl, write_jsonl_atomic

    from .data import iso3

    records = read_records(args.input.expanduser().resolve())
    output = args.output.expanduser().resolve()
    existing = [row for row in read_jsonl(output) if successful(row, False)] if output.exists() else []
    write_jsonl_atomic(existing, output)
    done = {(str(row["norm_id"]), str(row["condition"])) for row in existing}
    expected = {(str(record["norm_id"]), condition) for record in records
                for condition in args.conditions if eligible(record, condition)}
    print(f"model={args.model} mode=country-memory memory={not args.no_memory} records={len(records)} "
          f"expected={len(expected)} completed={len(done & expected)} pending={len(expected - done)}",
          flush=True)

    if expected - done:
        import numpy as np
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        adapter = args.lora.expanduser().resolve()
        saved = torch.load(adapter / MEMORY_FILE, map_location="cpu", weights_only=False)
        tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=args.local_files_only)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, device_map={"": args.device}, local_files_only=args.local_files_only)
        model = PeftModel.from_pretrained(model, str(adapter)).merge_and_unload()
        model.eval()
        token, slots = saved["token"], int(saved["table"].shape[1])
        memory = make_memory(torch, saved["countries"], saved["table"][0], placeholder_id(tokenizer, token))
        memory.table.data.copy_(saved["table"])
        memory.to(args.device)
        model.get_input_embeddings().register_forward_hook(memory.hook)

        unseen = sorted({iso3(record) for record in records} - set(memory.countries))
        if unseen and not args.no_memory:
            if args.geometry is None:
                raise ValueError(f"--geometry is needed for countries not trained on: {unseen}")
            geometry = np.load(args.geometry)
            layer = [int(value) for value in geometry["layers"]].index(args.geometry_layer)
            countries = [str(code) for code in geometry["countries"]]
            for code in unseen:
                memory.blends[code] = neighbour_weights(
                    countries, geometry["rdm_shift"][layer], memory.countries, code,
                    args.nearest, args.temperature)
            output.with_suffix(".neighbours.json").write_text(json.dumps(memory.blends, indent=2))
            print(f"neighbours for {len(unseen)} countries not trained on", flush=True)

        stops = [value for value in {tokenizer.eos_token_id,
                                     tokenizer.convert_tokens_to_ids("<|eot_id|>")}
                 if isinstance(value, int) and value >= 0 and value != tokenizer.unk_token_id]
        for condition in args.conditions:
            pending = [record for record in records if eligible(record, condition)
                       and (str(record["norm_id"]), condition) not in done]
            print(f"[{condition}] pending={len(pending)}", flush=True)
            plain = {str(record["norm_id"]): build_user_message(record, condition, False)[0]
                     for record in pending}
            sent = {key: (plain[key] if args.no_memory else
                          with_memory(plain[key], record, slots, token))
                    for key, record in ((str(record["norm_id"]), record) for record in pending)}
            rendered = {key: tokenizer.apply_chat_template(
                [{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True)
                for key, text in sent.items()}
            pending.sort(key=lambda record: len(rendered[str(record["norm_id"])]))
            for start in range(0, len(pending), args.batch_size):
                batch = pending[start:start + args.batch_size]
                keys = [str(record["norm_id"]) for record in batch]
                encoded = tokenizer([rendered[key] for key in keys], return_tensors="pt", padding=True,
                                    add_special_tokens=False).to(args.device)
                memory.rows = [iso3(record) for record in batch]
                with torch.no_grad():
                    generated = model.generate(**encoded, max_new_tokens=args.max_tokens, do_sample=False,
                                               eos_token_id=stops, pad_token_id=tokenizer.pad_token_id)
                memory.rows = None
                texts = tokenizer.batch_decode(generated[:, encoded["input_ids"].shape[1]:],
                                               skip_special_tokens=True)
                rows = [{**make_row(record, condition, plain[keys[i]], texts[i].strip(), False, False),
                         "memory": "none" if args.no_memory else
                         ("trained" if iso3(record) in memory.index else "neighbours")}
                        for i, record in enumerate(batch) if texts[i].strip()]
                append_rows(output, rows)
                done.update((str(row["norm_id"]), condition) for row in rows)
                if (start // args.batch_size) % 10 == 9 or start + args.batch_size >= len(pending):
                    print(f"[{condition}] processed={min(start + args.batch_size, len(pending))}/"
                          f"{len(pending)}", flush=True)

    final = {(str(row["norm_id"]), str(row["condition"])) for row in read_jsonl(output)}
    if expected - final:
        raise RuntimeError(f"Evaluation is incomplete; {len(expected - final)} rows remain unresolved")
    print(f"complete rows={len(expected)} output={output}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
