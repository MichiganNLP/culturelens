"""What the sparse-autoencoder features say about resisting a contradictory norm.

Three analyses on the extracted features (``extract.py``), per model and layer.
All selection is done on training records and all scores are on held-out ones.

1. **Is the decision carried by a few features?** Predict resisting from the
   SAE features of the country + contradictory-norm state: all active features,
   and the top ``k`` chosen on training records for a range of ``k``. The same
   readout on the dense state the SAE was given is the reference. If a handful
   of features match the dense state, the decision is localised in the SAE
   basis; if even all of them fall short, the SAE loses the signal.

2. **Do features register the conflict between country and norm?** For each
   feature and record, how much more it responds to the contradictory norm
   than to the true one, with the country named, minus the same difference
   without the country (``conflict``) or with a wrong country (``right
   country``). Counted against a false-discovery threshold, checked for
   replication on held-out records, and scored, without any behavioural label,
   for how well the selected features predict resisting.

3. **Audit of the earlier country-by-value profile.** The first CultureLens
   draft called a feature exclusive to a country (or a Schwartz value) when it
   fires on at least 1% of that group's records and at least twice as often as
   elsewhere, and profiled each country by the values whose exclusive features
   it shares. This reruns that definition and adds what it lacked: how many
   exclusive features label-shuffled data produce, and whether a country's
   profile is more like itself on a second half of the records than like other
   countries' profiles.

No language model is run. Dense states come from the activation cache.

    python -m mechanistic_analysis.sae.analyze --model gemma-3-12b \
        --features-root output/mechanistic_analysis/sae/features \
        --activation-root <caches> --records data/processed/culturelens/records.json \
        --outcomes output/mechanistic_analysis/outcomes/gemma-3-12b.npz \
        --output-root output/mechanistic_analysis/sae/analysis
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from mechanistic_analysis import JUDGES, RESIST
from mechanistic_analysis import outcomes as outcome_table
from mechanistic_analysis.common import assign_folds, claim_groups, country_index, load_condition
from mechanistic_analysis.readout import DEFAULT_PENALTIES, _auroc, ridge_readout
from mechanistic_analysis.sae.features import SparseFeatures, load
from mechanistic_analysis.sae.registry import MODEL_SAE_REGISTRY, get_spec
from mechanistic_analysis.stats import auroc, auroc_with_interval, within_ranks

# Layers analysed by default: every SAE layer for Llama, a grid for the others.
DEFAULT_LAYERS = {
    "llama-3.1-8b": (3, 7, 11, 15, 19, 23, 27),
    "gemma-3-12b": (0, 4, 8, 12, 14, 16, 18, 20, 22, 24, 26, 28, 32, 36, 40, 44, 47),
    "qwen-3.5-27b": (0, 8, 16, 20, 24, 28, 30, 32, 34, 36, 40, 44, 48, 52, 56, 60, 63),
}
SPARSE_SIZES = (1, 2, 5, 10, 20, 50, 100)
TRUE, FALSE = "country_norm", RESIST
TRUE_ALONE, FALSE_ALONE = "norm_only", "opposite_norm_only"
TRUE_WRONG, FALSE_WRONG = "shuffled_country_norm", "shuffled_country_opposite_norm"


# ------------------------------------------------------------------ inputs --


def dense_columns(stores: dict[str, SparseFeatures], min_rate: float
                  ) -> tuple[np.ndarray, dict[str, torch.Tensor]]:
    """The features active on at least ``min_rate`` of records in some condition,
    and each condition's activations on those features as a dense matrix."""

    first = next(iter(stores.values()))
    keep = torch.zeros(first.d_sae, dtype=torch.bool)
    for store in stores.values():
        rate = torch.bincount(store.feature_ids.long(), minlength=store.d_sae).float() / store.n_rows
        keep |= rate >= min_rate
    ids = keep.nonzero().flatten()
    position = torch.full((first.d_sae,), -1, dtype=torch.long)
    position[ids] = torch.arange(len(ids))
    dense = {}
    for name, store in stores.items():
        rows = torch.repeat_interleave(torch.arange(store.n_rows), store.row_ptr.diff())
        columns = position[store.feature_ids.long()]
        chosen = columns >= 0
        matrix = torch.zeros(store.n_rows, len(ids), dtype=torch.float32)
        matrix[rows[chosen], columns[chosen]] = store.feature_values.float()[chosen]
        dense[name] = matrix
    return ids.numpy(), dense


def scored(scores: np.ndarray, labels: np.ndarray, country: np.ndarray, groups: np.ndarray,
           draws: int) -> dict[str, float]:
    return auroc_with_interval(scores, labels, country, groups=groups, draws=draws)


# ------------------------------------------------------- 1. feature probes --


def sparse_probe(x: torch.Tensor, y: torch.Tensor, folds: torch.Tensor, sizes: tuple[int, ...],
                 penalties: tuple[float, ...], n_folds: int
                 ) -> tuple[dict[int, torch.Tensor], dict[int, list[list[int]]]]:
    """Held-out scores from a ridge readout on the ``k`` features that best separate
    the outcome on training records, for each ``k``, and the features each fold chose.

    The ranking uses training records only (test fold ``f``, development fold
    ``f + 1``): the absolute difference in a feature's mean between the two
    outcomes, in pooled standard deviations.
    """

    scores = {k: torch.zeros(len(y), dtype=torch.float64) for k in sizes}
    chosen: dict[int, list[list[int]]] = {k: [] for k in sizes}
    label = y.bool()
    for fold in range(n_folds):
        test, dev = folds == fold, folds == (fold + 1) % n_folds
        train = ~(test | dev)
        mean, scale = x[train].mean(dim=0), x[train].std(dim=0)
        usable = scale > 1e-8
        z = ((x - mean) / torch.where(usable, scale, torch.ones_like(scale))).double()
        gap = (z[train & label].mean(dim=0) - z[train & ~label].mean(dim=0)).abs()
        order = torch.argsort(torch.where(usable, gap, torch.zeros_like(gap)), descending=True)
        target = y[train].double()
        offset = target.mean()
        for k in sizes:
            columns = order[:min(k, int(usable.sum()))]
            chosen[k].append(columns.tolist())
            part = z[:, columns]
            gram, moment = part[train].T @ part[train], part[train].T @ (target - offset)
            eye = torch.eye(len(columns), dtype=gram.dtype)
            best = -1.0
            for penalty in penalties:
                weights = torch.linalg.solve(gram + penalty * eye, moment)
                value = _auroc(part[dev] @ weights, y[dev])
                if value == value and value > best:
                    best = value
                    scores[k][test] = part[test] @ weights + offset
    return scores, chosen


def feature_probes(x: torch.Tensor, state: torch.Tensor | None, resist: np.ndarray, valid: np.ndarray,
                   folds: torch.Tensor, n_folds: int, country: np.ndarray, groups: np.ndarray,
                   ids: np.ndarray, draws: int) -> dict[str, Any]:
    y = torch.tensor(resist.astype(np.int8))
    penalties = tuple(sorted(DEFAULT_PENALTIES))
    evaluate = lambda s: scored(np.asarray(s)[valid], resist[valid].astype(bool),  # noqa: E731
                                country[valid], groups[valid], draws)
    out: dict[str, Any] = {"features_used": int(x.shape[1])}
    held, _ = ridge_readout(x, y[:, None], folds, list(penalties), n_folds)
    out["all_features"] = evaluate(held[:, 0].numpy())
    if state is not None:
        held, _ = ridge_readout(state, y[:, None], folds, list(penalties), n_folds)
        out["dense_state"] = evaluate(held[:, 0].numpy())
    sizes = tuple(k for k in SPARSE_SIZES if k <= x.shape[1])
    scores, chosen = sparse_probe(x, y, folds, sizes, (1e-2, 1e0, 1e1, 1e2, 1e3), n_folds)
    out["top_k"] = {str(k): evaluate(scores[k].numpy()) for k in sizes}
    # How consistently the folds pick the same features, and which ones.
    out["chosen_features"] = {}
    for k in sizes:
        sets = [set(columns) for columns in chosen[k]]
        shared = set.intersection(*sets)
        out["chosen_features"][str(k)] = {
            "in_every_fold": sorted(int(ids[c]) for c in shared),
            "share_in_every_fold": len(shared) / max(1, len(sets[0]))}
    return out


# ------------------------------------------------------ 2. conflict features --


def paired_t(delta: torch.Tensor) -> torch.Tensor:
    """One-sample t statistic per column; 0 where a column does not vary."""

    scale = delta.std(dim=0)
    return torch.where(scale > 1e-8, delta.mean(dim=0) / (scale / len(delta) ** 0.5),
                       torch.zeros_like(scale))


def discoveries(t: torch.Tensor, rate: float = 0.01) -> int:
    """Number of columns passing Benjamini-Hochberg at ``rate`` (two-sided normal p)."""

    p = torch.erfc(t.abs().double() / 2 ** 0.5)
    ordered = torch.sort(p).values
    passed = ordered <= rate * torch.arange(1, len(p) + 1, dtype=torch.float64) / len(p)
    return int(passed.nonzero().max() + 1) if passed.any() else 0


def conflict_features(dense: dict[str, torch.Tensor], resist: np.ndarray, valid: np.ndarray,
                      folds: torch.Tensor, n_folds: int, country: np.ndarray, groups: np.ndarray,
                      draws: int, size: int = 20, seed: int = 0) -> dict[str, Any]:
    truth = dense[FALSE] - dense[TRUE]
    contrasts = {"truth": truth}
    if TRUE_ALONE in dense and FALSE_ALONE in dense:
        contrasts["conflict"] = truth - (dense[FALSE_ALONE] - dense[TRUE_ALONE])
    if TRUE_WRONG in dense and FALSE_WRONG in dense:
        contrasts["right_country"] = truth - (dense[FALSE_WRONG] - dense[TRUE_WRONG])
    generator = torch.Generator().manual_seed(seed)
    labels = resist[valid].astype(bool)
    out: dict[str, Any] = {}
    for name, delta in contrasts.items():
        varying = int((delta.std(dim=0) > 1e-8).sum())
        entry: dict[str, Any] = {"features_tested": varying,
                                 "discoveries_at_fdr_0.01": discoveries(paired_t(delta))}
        score = torch.zeros(len(delta), dtype=torch.float64)
        random_score = torch.zeros(len(delta), dtype=torch.float64)
        agreement, replication = [], []
        for fold in range(n_folds):
            test = folds == fold
            t = paired_t(delta[~test])
            top = torch.argsort(t.abs(), descending=True)[:min(size, varying)]
            scale = delta[~test][:, top].std(dim=0).clamp_min(1e-8)
            score[test] = ((delta[test][:, top] / scale) * torch.sign(t[top])).sum(dim=1).double()
            held = paired_t(delta[test])
            agreement.append(float((torch.sign(held[top]) == torch.sign(t[top])).float().mean()))
            wide = torch.argsort(t.abs(), descending=True)[:min(200, varying)]
            replication.append(float(np.corrcoef(delta[~test][:, wide].mean(dim=0).numpy(),
                                                 delta[test][:, wide].mean(dim=0).numpy())[0, 1]))
            pick = torch.randperm(delta.shape[1], generator=generator)[:len(top)]
            scale = delta[~test][:, pick].std(dim=0).clamp_min(1e-8)
            random_score[test] = ((delta[test][:, pick] / scale)
                                  * torch.sign(paired_t(delta[~test])[pick])).sum(dim=1).double()
        entry["held_out_sign_agreement"] = float(np.mean(agreement))
        entry["held_out_effect_correlation"] = float(np.mean(replication))
        entry["predicts_resisting"] = scored(score.numpy()[valid], labels, country[valid],
                                             groups[valid], draws)
        entry["random_features_predict_resisting"] = auroc(
            within_ranks(random_score.numpy()[valid], country[valid]), labels)
        out[name] = entry
    return out


# ------------------------------------------- 3. audit of the earlier profile --


def exclusive(active: torch.Tensor, membership: torch.Tensor, floor: float = 0.01,
              ratio: float = 2.0) -> torch.Tensor:
    """(groups, features) boolean: the feature is enriched for that group and no other.

    Enriched means it fires on at least ``floor`` of the group's records and at
    least ``ratio`` times as often as on the records outside the group.
    """

    members = membership.float()
    size = members.sum(dim=0)
    inside = members.T @ active / size[:, None].clamp_min(1)
    total = active.sum(dim=0)
    outside = (total[None, :] - members.T @ active) / (len(active) - size)[:, None].clamp_min(1)
    enriched = (inside >= floor) & (inside >= ratio * outside) & (size[:, None] > 0)
    return enriched & (enriched.sum(dim=0, keepdim=True) == 1)


def value_profile(active: torch.Tensor, countries: torch.Tensor, values: torch.Tensor
                  ) -> tuple[torch.Tensor, int, int]:
    """The earlier draft's ``w`` (countries, values), and how many features are
    exclusive to a country and to a value."""

    by_value = exclusive(active, values)
    # As in the draft: a country's group leaves out records tagged with a value
    # that has exclusive features of its own.
    tagged = (values & by_value.any(dim=1)[None, :]).any(dim=1)
    by_country = exclusive(active[~tagged], countries[~tagged])
    shared = by_country.float() @ by_value.float().T
    share = shared / by_value.sum(dim=1).clamp_min(1)[None, :]
    weight = share / share.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return weight, int(by_country.any(dim=0).sum()), int(by_value.any(dim=0).sum())


def top_sets(weight: torch.Tensor, size: int = 5) -> list[frozenset[int]]:
    sets = []
    for row in weight:
        order = torch.argsort(row, descending=True)[:size]
        sets.append(frozenset(int(i) for i in order if row[i] > 0))
    return sets


def jaccard(a: frozenset[int], b: frozenset[int]) -> float:
    return len(a & b) / len(a | b) if a | b else float("nan")


def profile_audit(active: torch.Tensor, countries: torch.Tensor, values: torch.Tensor,
                  draws: int = 200, halves: int = 20, seed: int = 0) -> dict[str, Any]:
    generator = torch.Generator().manual_seed(seed)
    weight, n_country, n_value = value_profile(active, countries, values)
    sets = top_sets(weight)
    shuffled = []
    for _ in range(draws):
        order = torch.randperm(len(active), generator=generator)
        shuffled.append(value_profile(active, countries[order], values)[1])
    same, other = [], []
    for _ in range(halves):
        order = torch.randperm(len(active), generator=generator)
        first, second = order[:len(order) // 2], order[len(order) // 2:]
        a = top_sets(value_profile(active[first], countries[first], values[first])[0])
        b = top_sets(value_profile(active[second], countries[second], values[second])[0])
        pairs = [(i, j) for i in range(len(a)) for j in range(len(b)) if a[i] and b[j]]
        same.extend(jaccard(a[i], b[j]) for i, j in pairs if i == j)
        other.extend(jaccard(a[i], b[j]) for i, j in pairs if i != j)
    return {
        "features_active": int(active.any(dim=0).sum()),
        "country_exclusive_features": n_country,
        "country_exclusive_features_when_labels_shuffled": {
            "mean": float(np.mean(shuffled)), "95th_percentile": float(np.percentile(shuffled, 95))},
        "value_exclusive_features": n_value,
        "countries_with_a_profile": int(sum(bool(s) for s in sets)),
        "distinct_profiles": len({s for s in sets if s}),
        "split_half_overlap_same_country": float(np.nanmean(same)) if same else float("nan"),
        "split_half_overlap_other_country": float(np.nanmean(other)) if other else float("nan"),
    }


# ------------------------------------------------------------------- driver --


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODEL_SAE_REGISTRY), required=True)
    parser.add_argument("--features-root", type=Path, required=True)
    parser.add_argument("--activation-root", type=Path, default=None,
                        help="Activation caches, for the dense-state reference. Optional.")
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--variant", default="unconstrained")
    parser.add_argument("--layers", type=int, nargs="+", default=None)
    parser.add_argument("--min-rate", type=float, default=0.005,
                        help="A feature is kept if it is active on this share of records in some condition.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--draws", type=int, default=500)
    parser.add_argument("--skip-profile", action="store_true")
    parser.add_argument("--judge", choices=JUDGES, default=JUDGES[0],
                        help="Whose labels define resisting. The second judge's results are "
                             "written to analysis.<judge>.json.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    spec = get_spec(args.model)
    layers = tuple(args.layers or DEFAULT_LAYERS[args.model])
    folder = args.features_root / args.variant / args.model
    output = args.output_root / args.variant / args.model
    output.mkdir(parents=True, exist_ok=True)
    path = output / ("analysis.json" if args.judge == JUDGES[0] else f"analysis.{args.judge}.json")
    report: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {
        "model": args.model, "sae": spec.default_repo, "judge": args.judge, "layers": {}}

    records = json.loads(args.records.read_text(encoding="utf-8"))
    by_id = {str(record["norm_id"]): record for record in records}
    started = time.time()
    for layer in layers:
        if str(layer) in report["layers"]:
            continue
        stores = {p.stem: load(p) for p in sorted((folder / f"layer_{layer}").glob("*.pt"))}
        if FALSE not in stores or TRUE not in stores:
            print(f"[layer {layer}] no features; skipped", flush=True)
            continue
        norm_ids = list(stores[FALSE].norm_ids)
        ordered = [by_id[norm_id] for norm_id in norm_ids]
        table = outcome_table.load(args.outcomes, norm_ids)
        resist = table[f"{args.judge}__{RESIST}"]
        valid = resist >= 0
        resist = (resist == 1).astype(np.int8)
        country = np.asarray(country_index(norm_ids))
        groups = np.asarray(claim_groups(ordered))
        folds = torch.tensor(assign_folds(groups.tolist(), args.folds, args.seed))
        ids, dense = dense_columns(stores, args.min_rate)

        state = None
        if args.activation_root is not None:
            cache = load_condition(args.activation_root, args.variant, args.model, FALSE)
            if [str(v) for v in cache["norm_ids"]] != norm_ids:
                raise ValueError("The activation cache and the features list different records")
            state = cache["assistant_start"][:, cache["layers"].index(layer), :].float().clone()

        entry: dict[str, Any] = {
            "depth": spec.normalized_depth(layer),
            "quality": stores[FALSE].quality,
            "probes": feature_probes(dense[FALSE], state, resist, valid, folds, args.folds, country,
                                     groups, ids, args.draws),
            "conflict": conflict_features(dense, resist, valid, folds, args.folds, country, groups,
                                          args.draws),
        }
        if not args.skip_profile and "country_only" in stores:
            names = sorted({value for record in ordered for value in record.get("values") or []})
            values = torch.tensor([[name in (record.get("values") or []) for name in names]
                                   for record in ordered])
            one_hot = torch.nn.functional.one_hot(torch.tensor(country)).bool()
            store = stores["country_only"]
            used = torch.unique(store.feature_ids.long())
            position = torch.full((store.d_sae,), -1, dtype=torch.long)
            position[used] = torch.arange(len(used))
            rows = torch.repeat_interleave(torch.arange(store.n_rows), store.row_ptr.diff())
            active = torch.zeros(store.n_rows, len(used))
            active[rows, position[store.feature_ids.long()]] = 1.0
            entry["earlier_profile"] = profile_audit(active, one_hot, values)
        report["layers"][str(layer)] = entry
        path.write_text(json.dumps(report, indent=1), encoding="utf-8")
        probes = entry["probes"]
        print(f"[layer {layer}] features {probes['features_used']}  all {probes['all_features']['auroc']:.3f}"
              + (f"  dense {probes['dense_state']['auroc']:.3f}" if "dense_state" in probes else "")
              + "  top-k " + " ".join(f"{k}:{v['auroc']:.3f}" for k, v in probes["top_k"].items())
              + f"  {time.time() - started:.0f}s", flush=True)
    print(f"saved {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
