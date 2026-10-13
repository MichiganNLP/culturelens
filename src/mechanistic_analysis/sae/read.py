"""Read chosen SAE features at the last prompt token under prompts that are not cached.

For the intervention sample and a list of features (layer, id), run the model
once per prompt variant, with no generation, and record each feature's
activation:

* the benchmark prompts with a contradictory norm, a true norm, and the country
  alone (a check against the cached features);
* the contradictory norm under each framing (guide, hearsay, doubt, norm first);
* the contradictory norm with its positions erased after a given layer.

    python -m mechanistic_analysis.sae.read --model <path> --model-slug qwen-3.5-27b \
        --records data/processed/culturelens/records.json --sample 200 \
        --sae-root <checkpoints> --features 34:57357 36:17338 --erase-layers 24 36 \
        --output-root output/mechanistic_analysis/sae/readings
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

import numpy as np

from behavioral_evaluation.prompts import build_user_message as behavioral_message
from mechanistic_analysis import RESIST
from mechanistic_analysis.activations.backend import _first_tensor, require_torch
from mechanistic_analysis.activations.extract import render_prompt
from mechanistic_analysis.interventions.common import (
    add_model_arguments, encode_prompt, load, sample_records, thinking_flag,
)
from mechanistic_analysis.interventions.erasure import erased
from mechanistic_analysis.interventions.generate import left_padded
from mechanistic_analysis.sae.backend import load_sae
from mechanistic_analysis.sae.registry import get_spec

BENCHMARK = (RESIST, "country_norm", "country_only", "opposite_norm_only")
FRAMINGS = ("country_opposite_norm_guide", "country_opposite_norm_hearsay",
            "country_opposite_norm_doubt", "opposite_norm_country")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    add_model_arguments(parser)
    parser.add_argument("--sae-root", type=str, required=True)
    parser.add_argument("--features", nargs="+", required=True, metavar="LAYER:FEATURE")
    parser.add_argument("--erase-layers", type=int, nargs="*", default=[])
    args = parser.parse_args(argv)
    torch = require_torch()
    from pathlib import Path

    wanted: dict[int, list[int]] = {}
    for item in args.features:
        layer, feature = item.split(":")
        wanted.setdefault(int(layer), []).append(int(feature))
    records = sample_records(args.records, args.sample, args.seed)
    loaded = load(args)
    thinking = thinking_flag(args)
    tokenizer = loaded.tokenizer
    device = loaded.model.get_input_embeddings().weight.device
    spec = get_spec(args.model_slug)
    saes = {layer: load_sae(spec, layer, root=Path(args.sae_root), device="cpu", dtype="float32")
            for layer in wanted}
    started = time.time()

    variants: list[tuple[str, str, int | None]] = [(name, name, None) for name in BENCHMARK]
    variants += [(name, name, None) for name in FRAMINGS]
    variants += [(f"erase_after_{layer}", RESIST, layer) for layer in args.erase_layers]

    out: dict[str, Any] = {"model": args.model_slug, "sample": len(records),
                           "norm_ids": [str(r["norm_id"]) for r in records],
                           "features": {str(k): v for k, v in wanted.items()}, "variants": {}}
    for name, condition, erase_layer in variants:
        if condition in FRAMINGS:
            prompts = [(tokenizer(render_prompt(loaded, behavioral_message(r, condition, False)[0],
                                                constrained=False, thinking=thinking),
                                  add_special_tokens=False)["input_ids"], None, None) for r in records]
        else:
            prompts = [encode_prompt(loaded, r, condition, thinking) for r in records]
        values = {layer: np.zeros((len(records), len(ids))) for layer, ids in wanted.items()}
        order = np.argsort([len(p[0]) for p in prompts], kind="stable")
        for begin in range(0, len(order), args.batch_size):
            rows = order[begin:begin + args.batch_size]
            ids, mask, pads = left_padded([prompts[row][0] for row in rows], tokenizer.pad_token_id)
            seen: dict[int, Any] = {}

            def capture(layer: int):
                def hook(_module: Any, _inputs: Any, output: Any) -> None:
                    seen[layer] = _first_tensor(output)[:, -1, :].detach().float().cpu()
                return hook

            handles = [loaded.decoder_layers[layer].register_forward_hook(capture(layer))
                       for layer in wanted]

            def marked(part: str) -> Any:
                keys = torch.zeros(ids.shape, dtype=torch.bool)
                for position, (pad, row) in enumerate(zip(pads, rows)):
                    first, last = prompts[row][2][part]
                    keys[position, pad + first:pad + last] = True
                return keys

            try:
                with torch.no_grad():
                    call = dict(input_ids=ids.to(device), attention_mask=mask.to(device))
                    if erase_layer is None:
                        loaded.model(**call)
                    else:
                        with erased(loaded.decoder_layers, erase_layer, marked("claim"),
                                    marked("situation")):
                            loaded.model(**call)
            finally:
                for handle in handles:
                    handle.remove()
            with torch.inference_mode():
                for layer, feature_ids in wanted.items():
                    encoded = saes[layer].encode(seen[layer])
                    values[layer][rows] = encoded[:, feature_ids].numpy()
        out["variants"][name] = {str(layer): {str(f): values[layer][:, i].round(4).tolist()
                                              for i, f in enumerate(wanted[layer])} for layer in wanted}
        summary = "  ".join(f"L{layer}.{f} active {100 * (values[layer][:, i] > 0).mean():.0f}%"
                            for layer in wanted for i, f in enumerate(wanted[layer]))
        print(f"[{name}] {summary}  {time.time() - started:.0f}s", flush=True)

    folder = args.output_root / args.model_slug
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "readings.json").write_text(json.dumps(out), encoding="utf-8")
    print(f"saved {folder / 'readings.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
