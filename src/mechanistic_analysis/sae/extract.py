"""Sparse-autoencoder features for the cached assistant-start states.

For one model, every layer that has a pretrained SAE and every cached prompt
condition: encode the state at the last prompt token and write the features in
CSR form, with how well the SAE reconstructs those states. No language model is
run; the input is the activation cache.

    python -m mechanistic_analysis.sae.extract --model llama-3.1-8b \
        --activation-root <caches> --sae-root <checkpoint folder> \
        --output-root output/mechanistic_analysis/sae/features

Files land at ``<output-root>/<variant>/<model>/layer_<L>/<condition>.pt`` and a
summary at ``<output-root>/<variant>/<model>/metrics.json``. Existing files are
kept unless ``--overwrite`` is given, so an interrupted run can be repeated.
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import torch

from mechanistic_analysis.common import load_condition
from mechanistic_analysis.sae.backend import DTYPES, load_sae
from mechanistic_analysis.sae.features import SparseFeatures, encode_states, load, save
from mechanistic_analysis.sae.registry import MODEL_SAE_REGISTRY, get_spec, resolve_layers


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODEL_SAE_REGISTRY), required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", default="unconstrained")
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Prompt conditions to encode. Default: every cached condition.")
    parser.add_argument("--layers", default="all",
                        help="Comma-separated layers, or 'all' for every layer with an SAE.")
    parser.add_argument("--sae-root", type=Path, default=None,
                        help="Local checkpoint folder. Default: download from the Hugging Face Hub.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--sae-dtype", choices=tuple(DTYPES), default=None)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def cached_conditions(root: Path, variant: str, model: str) -> list[str]:
    return sorted(path.stem for path in (root / variant / model).glob("*.pt"))


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = get_spec(args.model)
    layers = resolve_layers(spec, args.layers)
    conditions = args.conditions or cached_conditions(args.activation_root, args.variant, args.model)
    if not conditions:
        raise FileNotFoundError(
            f"No activation caches under {args.activation_root / args.variant / args.model}")
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    folder = args.output_root / args.variant / args.model
    print(f"SAE features: {args.model}  layers={list(layers)}  conditions={conditions}  "
          f"device={device}  sae={args.sae_root or spec.default_repo}  output={folder}", flush=True)

    caches = {name: load_condition(args.activation_root, args.variant, args.model, name)
              for name in conditions}
    missing = sorted(set(layers) - set(next(iter(caches.values()))["layers"]))
    if missing:
        raise ValueError(f"The activation cache has no layers {missing}")

    written = skipped = 0
    started = time.time()
    for layer in layers:
        pending = [name for name in conditions
                   if args.overwrite or not (folder / f"layer_{layer}" / f"{name}.pt").exists()]
        skipped += len(conditions) - len(pending)
        if not pending:
            continue
        sae = load_sae(spec, layer, root=args.sae_root, device=device, dtype=args.sae_dtype)
        for name in pending:
            cache = caches[name]
            # Copied out of the memory-mapped cache once; the layer's rows are not
            # adjacent on disk, so reading them batch by batch would repeat the seeks.
            states = cache["assistant_start"][:, cache["layers"].index(layer), :].clone()
            row_ptr, ids, values, quality = encode_states(states, sae, batch_size=args.batch_size)
            save(SparseFeatures(
                model=args.model, condition=name, layer=layer,
                shape=(len(cache["norm_ids"]), sae.d_sae), row_ptr=row_ptr, feature_ids=ids,
                feature_values=values, norm_ids=tuple(map(str, cache["norm_ids"])),
                sae_metadata=sae.metadata.to_dict(),
                cache={"dataset_sha256": cache.get("dataset_sha256"),
                       "schema_version": cache.get("schema_version"), "variant": args.variant},
                quality=quality), folder / f"layer_{layer}" / f"{name}.pt")
            written += 1
            print(f"[layer {layer} / {name}] L0 {quality['mean_l0']:.1f}  "
                  f"relative error {quality['relative_error']:.4f}  "
                  f"FVU {quality['fraction_of_variance_unexplained']:.3f}  "
                  f"cosine {quality['cosine']:.3f}  {time.time() - started:.0f}s", flush=True)
        del sae
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # The summary is rebuilt from the files, so it is complete after a resumed run.
    summary: dict[str, dict[str, dict[str, float]]] = {}
    for layer in layers:
        for name in conditions:
            path = folder / f"layer_{layer}" / f"{name}.pt"
            if path.exists():
                features = load(path)
                summary.setdefault(str(layer), {})[name] = {
                    **features.quality, "records": features.n_rows, "d_sae": features.d_sae}
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "metrics.json").write_text(json.dumps(
        {"model": args.model, "sae": spec.default_repo, "provider": spec.provider,
         "layers": list(layers), "conditions": conditions, "quality": summary}, indent=1),
        encoding="utf-8")
    print(f"done: wrote {written}, kept {skipped}; summary in {folder / 'metrics.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
