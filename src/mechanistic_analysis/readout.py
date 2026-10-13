"""Is the model's answer readable from its state before it starts answering?

For each cached prompt condition (the *state*) and each judged outcome
(``outcomes.py``), a ridge readout is fitted on the assistant-start state and
scored on held-out records, at every requested layer. Pairing states with
outcomes answers separate questions about resisting a false norm:

* ``country_opposite_norm`` state: is the decision present before generation?
* ``opposite_norm_only`` state: is it a property of the claim alone?
* ``country_only`` or ``base`` state: was it settled before the claim was shown?

``--text`` adds the same readouts from a bag of words of the rendered prompt
(layer -1): what the wording alone predicts.

Protocol: five folds over claim groups (test fold k, development fold k+1,
training on the rest), features standardised on training rows, ridge penalty
chosen per outcome by development AUROC. All outcomes for one state and layer
share one factorisation. A record whose judgment is missing (-1) is fitted as
0 and must be left out when the scores are evaluated; ``report.py`` does so.

PYTHONPATH=src python -m mechanistic_analysis.readout \
  --activations-root <caches> --model-slug llama-3.1-8b \
  --records data/processed/culturelens/records.json \
  --outcomes output/mechanistic_analysis/outcomes/llama-3.1-8b.npz \
  --output-root output/mechanistic_analysis/readout
"""

from __future__ import annotations

import argparse
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from mechanistic_analysis.activations.backend import require_torch
from mechanistic_analysis.activations.cache import sha256_file, write_manifest
from mechanistic_analysis.activations.conditions import build_user_message
from mechanistic_analysis.activations.extract import read_records
from mechanistic_analysis import outcomes as outcome_table
from mechanistic_analysis.common import (
    assign_folds, claim_groups, load_conditions, resolve_device,
)

DEFAULT_STATES = ("base", "country_only", "opposite_norm_only", "country_opposite_norm",
                  "norm_only", "country_norm")
DEFAULT_PENALTIES = (1e1, 1e2, 1e3, 1e4, 1e5)
_TOKEN = re.compile(r"[a-z']+")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--activations-root", type=Path, required=True)
    parser.add_argument("--variant", default="unconstrained")
    parser.add_argument("--model-slug", required=True)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--outcomes", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--states", nargs="+", default=list(DEFAULT_STATES))
    parser.add_argument("--layers", type=int, nargs="+")
    parser.add_argument("--layer-step", type=int, default=1,
                        help="Use every n-th cached layer (and always the last).")
    parser.add_argument("--text", action="store_true")
    parser.add_argument("--min-count", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--penalties", type=float, nargs="+", default=list(DEFAULT_PENALTIES))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    return parser.parse_args(argv)


def _auroc(scores: Any, labels: Any) -> float:
    torch = require_torch()
    mask = labels.bool()
    positive, negative = scores[mask].double(), scores[~mask].double()
    if positive.numel() == 0 or negative.numel() == 0:
        return float("nan")
    ordered = torch.sort(negative).values
    wins = (torch.searchsorted(ordered, positive, right=False)
            + torch.searchsorted(ordered, positive, right=True)).double().sum() / 2
    return float(wins / (positive.numel() * negative.numel()))


def ridge_readout(x: Any, y: Any, folds: Any, penalties: list[float], n_folds: int
                  ) -> tuple[Any, list[list[float]]]:
    """Held-out scores, shape (records, outcomes), and the penalty each fold chose.

    ``x`` is (records, features); ``y`` is (records, outcomes) in {0, 1}.
    """

    torch = require_torch()
    scores = torch.zeros(y.shape, dtype=torch.float64, device=x.device)
    chosen: list[list[float]] = []
    for test_fold in range(n_folds):
        test, dev = folds == test_fold, folds == (test_fold + 1) % n_folds
        train = ~(test | dev)
        mean = x[train].mean(dim=0)
        scale = x[train].std(dim=0)
        scale = torch.where(scale > 1e-8, scale, torch.ones_like(scale))
        z = ((x - mean) / scale).double()
        target = y[train].double()
        offset = target.mean(dim=0)
        gram = z[train].T @ z[train]
        moment = z[train].T @ (target - offset)
        eye = torch.eye(gram.shape[0], dtype=gram.dtype, device=gram.device)
        best_auroc = [-1.0] * y.shape[1]
        best_penalty = [penalties[0]] * y.shape[1]
        for penalty in penalties:
            weights = torch.cholesky_solve(moment, torch.linalg.cholesky(gram + penalty * eye))
            dev_scores = z[dev] @ weights
            held = z[test] @ weights + offset
            for column in range(y.shape[1]):
                value = _auroc(dev_scores[:, column], y[dev][:, column])
                if value == value and value > best_auroc[column]:
                    best_auroc[column], best_penalty[column] = value, penalty
                    scores[test, column] = held[:, column]
        chosen.append(best_penalty)
    return scores, chosen


def text_features(prompts: list[str], min_count: int) -> np.ndarray:
    """Binary bag of words and word pairs. The vocabulary uses every prompt but
    no label, so it is not fold-specific."""

    documents = []
    for text in prompts:
        words = _TOKEN.findall(text.casefold())
        documents.append(set(words) | {f"{a} {b}" for a, b in zip(words, words[1:])})
    counts = Counter(word for words in documents for word in words)
    vocabulary = {word: column for column, word in
                  enumerate(sorted(w for w, c in counts.items() if c >= min_count))}
    matrix = np.zeros((len(prompts), len(vocabulary)), dtype=np.float32)
    for row, words in enumerate(documents):
        for word in words:
            column = vocabulary.get(word)
            if column is not None:
                matrix[row, column] = 1.0
    return matrix


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    torch = require_torch()
    device = resolve_device(args.device)
    started = time.time()
    caches = load_conditions(args.activations_root, args.variant, args.model_slug,
                             tuple(args.states))
    first = caches[args.states[0]]
    if sha256_file(args.records) != first["dataset_sha256"]:
        raise ValueError(f"{args.records} is not the dataset these activations were extracted from")
    norm_ids = [str(value) for value in first["norm_ids"]]
    by_id = {str(record["norm_id"]): record for record in read_records(args.records)}
    records = [by_id[norm_id] for norm_id in norm_ids]
    table = outcome_table.load(args.outcomes, norm_ids)
    names = [name for name, values in table.items() if 0 < (values == 1).mean() < 1]
    y = torch.tensor(np.stack([(table[name] == 1) for name in names], axis=1).astype(np.int8),
                     device=device)
    cached_layers = list(first["layers"])
    layers = args.layers or sorted(set(cached_layers[::args.layer_step]) | {cached_layers[-1]})
    groups = claim_groups(records)
    folds = torch.tensor(assign_folds(groups, args.folds, args.seed), device=device)
    print(f"model={args.model_slug} records={len(norm_ids)} states={args.states} "
          f"outcomes={len(names)} layers={len(layers)} device={device}", flush=True)

    scores: dict[str, list[np.ndarray]] = {}

    def run(state: str, x: Any) -> None:
        held, _ = ridge_readout(x, y, folds, sorted(args.penalties), args.folds)
        scores.setdefault(state, []).append(held.T.to(torch.float32).cpu().numpy())

    if args.text:
        for state in args.states:
            prompts = [build_user_message(record, state, include_options=False)[0]
                       for record in records]
            run(f"text:{state}", torch.tensor(text_features(prompts, args.min_count), device=device))
            print(f"text:{state} done", flush=True)
    for state in args.states:
        tick = time.time()
        for layer in layers:
            run(state, caches[state]["assistant_start"][:, cached_layers.index(layer)].to(
                device, torch.float32))
        print(f"{state:24s} {len(layers)} layers  {time.time() - tick:.0f}s", flush=True)

    output = args.output_root / args.model_slug
    output.mkdir(parents=True, exist_ok=True)
    np.savez(output / "scores.npz", norm_ids=np.asarray(norm_ids), outcomes=np.asarray(names),
             layers=np.asarray(layers), folds=folds.cpu().numpy(), groups=np.asarray(groups),
             **{f"score__{state}": np.stack(values).astype(np.float16)
                for state, values in scores.items()})
    write_manifest({
        "analysis": "decision_readout",
        "model_slug": args.model_slug,
        "states": list(args.states),
        "outcomes": names,
        "outcomes_sha256": sha256_file(args.outcomes),
        "dataset_sha256": first["dataset_sha256"],
        "layers": layers,
        "protocol": {
            "readout": "ridge regression on the 0/1 outcome, features standardised on training rows",
            "folds": args.folds,
            "split": "test fold k, development fold k+1, training on the rest",
            "grouping": "records sharing any claim text share a fold",
            "penalties": sorted(args.penalties),
            "penalty_selection": "development AUROC, per outcome",
            "missing_judgments": "fitted as 0; excluded when scores are evaluated",
            "text": "binary unigrams and bigrams of the rendered prompt, layer -1" if args.text else None,
            "scores": "score__<state>: (layers, outcomes, records), held out",
            "seed": args.seed,
        },
        "seconds": round(time.time() - started, 1),
    }, output / "metrics.json")
    print(f"saved {output}  {time.time() - started:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        raise SystemExit(f"ERROR: {error}")
