"""What the single most predictive SAE feature at each layer is.

For the feature that every fold of ``analyze.py`` chose as its best single
predictor of resisting, at each analysed layer:

* how often it is active under each cached prompt condition, and by outcome;
* the records on which it is most active;
* whether it is one direction carried through the layers or a different one at
  each: the cosine between its decoder direction and the next layer's chosen
  feature, and with the nearest feature of the next layer;
* how much of its predictive value is in the direction itself: the AUROC of the
  dense state projected on the decoder direction, with no SAE nonlinearity;
* with ``--model-dir``, the output tokens the direction promotes and suppresses
  when written straight to the unembedding (through the final norm's gain, and
  ignoring every later layer).

CPU only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mechanistic_analysis import JUDGES, RESIST
from mechanistic_analysis import outcomes as outcome_table
from mechanistic_analysis.common import country_index, load_condition
from mechanistic_analysis.sae.backend import load_sae
from mechanistic_analysis.sae.features import SparseFeatures, load
from mechanistic_analysis.sae.registry import MODEL_SAE_REGISTRY, get_spec
from mechanistic_analysis.stats import auroc, within_ranks


def column(store: SparseFeatures, feature: int) -> np.ndarray:
    rows = torch.repeat_interleave(torch.arange(store.n_rows), store.row_ptr.diff())
    chosen = store.feature_ids.long() == feature
    values = torch.zeros(store.n_rows)
    values[rows[chosen]] = store.feature_values.float()[chosen]
    return values.numpy()


def unembedding(model_dir: Path) -> tuple[torch.Tensor, torch.Tensor, Any] | None:
    """(lm_head weight, final norm gain, tokenizer) read straight from the checkpoint."""

    index = model_dir / "model.safetensors.index.json"
    if not index.is_file():
        return None
    from safetensors import safe_open
    from transformers import AutoTokenizer

    weights = json.loads(index.read_text())["weight_map"]
    head = next((k for k in weights if k.endswith("lm_head.weight")), None) or next(
        k for k in weights if k.endswith("embed_tokens.weight"))
    norm = next(k for k in weights if k.endswith("norm.weight") and "layers" not in k
                and "visual" not in k and "mtp" not in k and "vision" not in k)

    def read(name: str) -> torch.Tensor:
        with safe_open(str(model_dir / weights[name]), framework="pt") as handle:
            return handle.get_tensor(name).float()

    return read(head), read(norm), AutoTokenizer.from_pretrained(str(model_dir))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODEL_SAE_REGISTRY), required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--analysis-root", type=Path, required=True)
    parser.add_argument("--sae-root", type=Path, required=True)
    parser.add_argument("--activation-root", type=Path, required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--variant", default="unconstrained")
    parser.add_argument("--top-records", type=int, default=12)
    args = parser.parse_args(argv)

    spec = get_spec(args.model)
    folder = args.analysis_root / args.variant / args.model
    analysis = json.loads((folder / "analysis.json").read_text())
    chosen = {int(layer): entry["probes"]["chosen_features"]["1"]["in_every_fold"][0]
              for layer, entry in analysis["layers"].items()
              if entry["probes"]["chosen_features"].get("1", {}).get("in_every_fold")}
    layers = sorted(chosen)
    records = {str(r["norm_id"]): r for r in json.loads(args.records.read_text(encoding="utf-8"))}
    readout = None
    if args.model_dir:
        try:
            readout = unembedding(args.model_dir)
        except (StopIteration, OSError, KeyError, ImportError) as error:
            print(f"no unembedding read from {args.model_dir}: {error!r}", flush=True)
    cache = load_condition(args.activation_root, args.variant, args.model, RESIST)

    unit = lambda v: v / v.norm().clamp_min(1e-12)  # noqa: E731
    previous: tuple[int, torch.Tensor] | None = None
    out: dict[str, Any] = {"model": args.model, "layers": {}}
    for layer in layers:
        feature = chosen[layer]
        sae = load_sae(spec, layer, root=args.sae_root, device="cpu", dtype="float32")
        decoder = sae.w_dec                                   # (d_model, d_sae)
        direction = decoder[:, feature].clone()
        if previous is not None:
            # Is the earlier layer's feature the same direction as this layer's?
            earlier, earlier_direction = previous
            similarity = (decoder / decoder.norm(dim=0, keepdim=True).clamp_min(1e-12)).T @ unit(
                earlier_direction)
            nearest = int(similarity.argmax())
            out["layers"][str(earlier)]["next_layer"] = {
                "layer": layer, "cosine_with_its_chosen_feature": float(similarity[feature]),
                "nearest_feature": nearest, "nearest_cosine": float(similarity[nearest]),
                "nearest_is_its_chosen_feature": nearest == feature}
        previous = (layer, direction)
        stores = {p.stem: load(p) for p in sorted(
            (args.features_root / args.variant / args.model / f"layer_{layer}").glob("*.pt"))}
        norm_ids = list(stores[RESIST].norm_ids)
        table = outcome_table.load(args.outcomes, norm_ids)
        country = np.asarray(country_index(norm_ids))
        resist = table[f"{JUDGES[0]}__{RESIST}"]
        valid, label = resist >= 0, resist == 1
        entry: dict[str, Any] = {"feature": feature, "decoder_norm": float(direction.norm()),
                                 "active_by_condition": {}, "mean_by_condition": {}}
        for name, store in stores.items():
            values = column(store, feature)
            entry["active_by_condition"][name] = float((values > 0).mean())
            entry["mean_by_condition"][name] = float(values.mean())
        values = column(stores[RESIST], feature)
        entry["active_when_resisting"] = float((values[valid & label] > 0).mean())
        entry["active_when_not_resisting"] = float((values[valid & ~label] > 0).mean())
        entry["feature_auroc"] = auroc(within_ranks(values[valid], country[valid]), label[valid])
        positive = values[values > 0]
        entry["active_percentiles"] = {str(q): float(np.percentile(positive, q)) for q in (50, 90)} \
            if len(positive) else {}
        for condition, key in (("country_norm", "true_norm"), ("country_only", "country_only")):
            if condition in stores and f"{JUDGES[0]}__{condition}" in table:
                aligned = table[f"{JUDGES[0]}__{condition}"]
                other = column(stores[condition], feature)
                entry[f"active_{key}_aligned"] = float((other[aligned == 1] > 0).mean())
                entry[f"active_{key}_not_aligned"] = float((other[aligned == 0] > 0).mean())
        # The direction alone, without the SAE's threshold or top-k.
        state = cache["assistant_start"][:, cache["layers"].index(layer), :].float()
        projection = (state @ direction).numpy()
        signed = projection if entry["active_when_resisting"] >= entry["active_when_not_resisting"] \
            else -projection
        entry["direction_auroc"] = auroc(within_ranks(signed[valid], country[valid]), label[valid])
        order = np.argsort(-values)[:args.top_records]
        entry["top_records"] = [{
            "norm_id": norm_ids[i], "activation": float(values[i]), "resists": bool(label[i]),
            "norm": records[norm_ids[i]]["norm"],
            "contradictory_norm": records[norm_ids[i]]["scenario"]["opposite_norm_evidence"]}
            for i in order if values[i] > 0]
        if readout is not None:
            head, gain, tokenizer = readout
            logits = head @ (direction * gain)
            entry["promotes"] = [tokenizer.decode([int(i)]) for i in torch.topk(logits, 20).indices]
            entry["suppresses"] = [tokenizer.decode([int(i)]) for i in torch.topk(-logits, 20).indices]
        out["layers"][str(layer)] = entry
        print(f"[layer {layer}] feature {feature}  feature AUROC {entry['feature_auroc']:.3f}  "
              f"direction AUROC {entry['direction_auroc']:.3f}", flush=True)
        del sae

    (folder / "features.json").write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"saved {folder / 'features.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
