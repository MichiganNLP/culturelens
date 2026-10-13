"""How the country is represented, and whether following a stated norm is one variable.

Two analyses on the cached last-prompt-token states, with no intervention and
(for the first) no behavioural label.

**Country geometry.** A country's vector at a layer is the mean state of its
records, centred across countries. Three states are used: the ``country_only``
prompt, the ``base`` prompt (no country), and their per-record difference, the
*shift* that naming the country causes. Per layer this module saves

* ``rdm_<state>``: cosine distance between every pair of country vectors, also
  under ``country_norm`` and ``country_opposite_norm`` so the geometry with a
  norm stated can be compared with the geometry without one;
* ``shift_size``: per country, the mean over its records of the length of the
  shift relative to the length of the base state. A small value means naming
  the country changes little, i.e. the model was already close to it;
* ``distinctness`` is derived from an RDM as a country's mean distance to the
  others (``distinctness``).

**Following a stated norm.** ``follow`` is 1 when the judged answer goes with
the norm the prompt states: aligned for a true norm, not aligned for a
contradictory one. A ridge probe is fitted on the state of one prompt
(``norm_only``, ``country_norm``, ``opposite_norm_only`` or
``country_opposite_norm``) and scored, on held-out records, on all four. If a
probe fitted on true norms predicts following contradictory ones, the state
carries "will go along with what is stated" and not "will give the documented
answer": a probe for the latter would transfer with AUROC below 0.5.
``text_transfer`` is the same test on a bag of words of the prompt, the
control for wording. AUROCs rank scores within country.

PYTHONPATH=src python -m mechanistic_analysis.country_space \
  --activations-root <caches> --model-slug llama-3.1-8b \
  --records data/processed/culturelens/records.json \
  --judgments-root output/behavioral_evaluation/unconstrained/judgments \
  --output-root output/mechanistic_analysis/country_space
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from mechanistic_analysis import JUDGES, REPORT_LAYER
from mechanistic_analysis.stats import auroc, within_ranks

GEOMETRY_STATES = ("base", "country_only", "shift", "country_norm", "country_opposite_norm")
NORM_PROMPTS = ("norm_only", "country_norm", "opposite_norm_only", "country_opposite_norm")
CONTRADICTORY = frozenset({"opposite_norm_only", "country_opposite_norm"})
CONDITIONS = ("base", "country_only", *NORM_PROMPTS)


# ----------------------------------------------------------------- geometry --


def country_vectors(states: np.ndarray, country: np.ndarray) -> np.ndarray:
    """Mean state per country (rows in sorted country order), centred across countries."""

    vectors = np.stack([states[country == code].mean(axis=0) for code in np.unique(country)])
    return vectors - vectors.mean(axis=0)


def cosine_rdm(vectors: np.ndarray) -> np.ndarray:
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    return 1.0 - unit @ unit.T


def distinctness(rdm: np.ndarray) -> np.ndarray:
    """Each country's mean distance to every other country."""

    return rdm.sum(axis=1) / (len(rdm) - 1)


def shift_size(base: np.ndarray, with_country: np.ndarray, country: np.ndarray) -> np.ndarray:
    """Per country, the mean relative length of the change that naming it causes."""

    relative = np.linalg.norm(with_country - base, axis=1) / np.maximum(np.linalg.norm(base, axis=1), 1e-12)
    return np.array([relative[country == code].mean() for code in np.unique(country)])


# -------------------------------------------------------------------- probe --


def follow_labels(aligned: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """1 when the answer goes with the stated norm; NaN where there is no judged answer."""

    return {name: (1.0 - aligned[name]) if name in CONTRADICTORY else aligned[name] for name in NORM_PROMPTS}


def ridge_scores(train: np.ndarray, target: np.ndarray, tests: list[np.ndarray], penalty: float) -> list[np.ndarray]:
    """Ridge regression in its dual form (records are fewer than a cached file is wide
    for the text baseline, and the solve is the same size either way)."""

    mean, scale = train.mean(axis=0), train.std(axis=0) + 1e-6
    design = (train - mean) / scale
    weights = design.T @ np.linalg.solve(design @ design.T + penalty * np.eye(len(design)),
                                         target - target.mean())
    return [((test - mean) / scale) @ weights for test in tests]


def transfer(features: Callable[[str], np.ndarray], follow: dict[str, np.ndarray], country: np.ndarray,
             fold: np.ndarray, penalty_per_record: float = 10.0) -> dict[str, dict[str, float]]:
    """``trained on -> scored on -> within-country AUROC`` for following the stated norm."""

    matrices = {name: features(name) for name in NORM_PROMPTS}
    out: dict[str, dict[str, float]] = {}
    for source in NORM_PROMPTS:
        scores = {name: np.full(len(country), np.nan) for name in NORM_PROMPTS}
        for held in np.unique(fold):
            fit = (fold != held) & ~np.isnan(follow[source])
            test = fold == held
            parts = ridge_scores(matrices[source][fit], follow[source][fit],
                                 [matrices[name][test] for name in NORM_PROMPTS],
                                 penalty_per_record * float(fit.sum()))
            for name, part in zip(NORM_PROMPTS, parts):
                scores[name][test] = part
        out[source] = {}
        for name in NORM_PROMPTS:
            keep = ~np.isnan(follow[name])
            out[source][name] = float(auroc(within_ranks(scores[name][keep], country[keep]),
                                            follow[name][keep] == 1))
    return out


def bag_of_words(texts: dict[str, list[str]], minimum: int = 20) -> dict[str, np.ndarray]:
    """Binary word indicators over one vocabulary shared by every prompt."""

    split = {name: [set(re.findall(r"[a-z']+", text.lower())) for text in rows] for name, rows in texts.items()}
    counts: dict[str, int] = {}
    for rows in split.values():
        for words in rows:
            for word in words:
                counts[word] = counts.get(word, 0) + 1
    index = {word: column for column, word in enumerate(sorted(w for w, n in counts.items() if n >= minimum))}
    out = {}
    for name, rows in split.items():
        matrix = np.zeros((len(rows), len(index)), dtype=np.float32)
        for row, words in enumerate(rows):
            for word in words:
                if word in index:
                    matrix[row, index[word]] = 1.0
        out[name] = matrix
    return out


# --------------------------------------------------------------------- main --


def read_aligned(root: Path, slug: str, judge: str, norm_ids: list[str]) -> dict[str, np.ndarray]:
    found: dict[tuple[str, str], float] = {}
    for folder in (slug, f"{slug}.opposite_norm_only"):
        path = root / folder / f"{judge}.jsonl"
        if path.is_file():
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        row = json.loads(line)
                        found[(str(row["norm_id"]), str(row["condition"]))] = float(row["alignment"] == "aligned")
    return {name: np.array([found.get((key, name), np.nan) for key in norm_ids]) for name in NORM_PROMPTS}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations-root", type=Path, required=True)
    parser.add_argument("--variant", default="unconstrained")
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--judgments-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--layer-step", type=int, default=2, help="Geometry is saved at every n-th layer.")
    parser.add_argument("--probe-layers", type=int, nargs="+",
                        help="Layers for the probe; the model's reporting layer when omitted.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    import torch

    from behavioral_evaluation.prompts import build_user_message
    from mechanistic_analysis.activations.cache import sha256_file
    from mechanistic_analysis.activations.extract import read_records
    from mechanistic_analysis.common import load_conditions

    started = time.time()
    caches = load_conditions(args.activations_root, args.variant, args.model_slug, CONDITIONS)
    norm_ids = [str(key) for key in caches["base"]["norm_ids"]]
    layers = [int(layer) for layer in caches["base"]["layers"]]
    country = np.array([key.split("__")[0] for key in norm_ids])

    def state(name: str, layer: int) -> np.ndarray:
        return caches[name]["assistant_start"][:, layers.index(layer)].to(torch.float32).numpy().astype(np.float64)

    chosen = sorted(set(layers[::args.layer_step]) | {layers[-1]})
    rdms: dict[str, list[np.ndarray]] = {name: [] for name in GEOMETRY_STATES}
    sizes = []
    for layer in chosen:
        base, named = state("base", layer), state("country_only", layer)
        for name, values in (("base", base), ("country_only", named), ("shift", named - base),
                             ("country_norm", state("country_norm", layer)),
                             ("country_opposite_norm", state("country_opposite_norm", layer))):
            rdms[name].append(cosine_rdm(country_vectors(values, country)))
        sizes.append(shift_size(base, named, country))
        print(f"geometry layer {layer}  {time.time() - started:.0f}s", flush=True)
    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "geometry.npz", countries=np.unique(country), layers=np.asarray(chosen),
             shift_size=np.asarray(sizes), **{f"rdm_{name}": np.asarray(values) for name, values in rdms.items()})

    fold = np.random.default_rng(args.seed).permutation(len(norm_ids)) % args.folds
    probe_layers = args.probe_layers or [REPORT_LAYER[args.model_slug]]
    by_id = {str(record["norm_id"]): record for record in read_records(args.records)}
    words = bag_of_words({name: [build_user_message(by_id[key], name, False)[0] for key in norm_ids]
                          for name in NORM_PROMPTS})
    probe: dict[str, Any] = {}
    for judge in JUDGES:
        aligned = read_aligned(args.judgments_root, args.model_slug, judge, norm_ids)
        if all(np.isnan(values).all() for values in aligned.values()):
            continue
        follow = follow_labels(aligned)
        probe[judge] = {
            "follow_rate": {name: float(np.nanmean(values)) for name, values in follow.items()},
            "state_transfer": {str(layer): transfer(lambda name, layer=layer: state(name, layer),
                                                    follow, country, fold) for layer in probe_layers},
            "text_transfer": transfer(lambda name: words[name].astype(np.float64), follow, country, fold)}
        print(f"probe {judge}  {time.time() - started:.0f}s", flush=True)
    (output / "metrics.json").write_text(json.dumps({
        "analysis": "country_space", "model_slug": args.model_slug,
        "dataset_sha256": sha256_file(args.records), "records": len(norm_ids),
        "countries": len(np.unique(country)), "geometry_layers": chosen, "probe_layers": probe_layers,
        "definitions": {
            "shift_size": "mean over a country's records of |country_only - base| / |base|",
            "rdm": "cosine distance between country mean states, centred across countries",
            "follow": "aligned for a true norm, not aligned for a contradictory norm",
            "transfer": "ridge probe fitted on one prompt's state, scored on every prompt's; "
                        "5 folds over records; AUROC ranked within country"},
        "follow_probe": probe, "seconds": round(time.time() - started, 1)}, indent=2))
    print(f"saved {output}  {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
