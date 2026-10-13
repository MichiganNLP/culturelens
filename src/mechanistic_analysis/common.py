"""Record grouping, folds and cache loading shared by the cached-state analyses."""

from __future__ import annotations

import random
import re
from pathlib import Path
from typing import Any

from mechanistic_analysis.activations import SCHEMA_VERSION
from mechanistic_analysis.activations.backend import require_torch
from mechanistic_analysis.activations.cache import load_torch

CLAIM_FIELDS = ("norm_evidence", "opposite_norm_evidence")
_SPACE = re.compile(r"\s+")


def claim_groups(records: list[dict[str, Any]]) -> list[int]:
    """A group index per record; records sharing any claim text share a group,
    whichever role (documented or opposite) the text plays in each."""

    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    owner: dict[str, int] = {}
    for index, record in enumerate(records):
        for field in CLAIM_FIELDS:
            text = _SPACE.sub(" ", str(record["scenario"][field])).strip().casefold()
            if text in owner:
                parent[find(index)] = find(owner[text])
            else:
                owner[text] = index
    labels: dict[int, int] = {}
    return [labels.setdefault(find(index), len(labels)) for index in range(len(records))]


def assign_folds(groups: list[int], n_folds: int, seed: int) -> list[int]:
    """A fold per record: groups are shuffled and dealt round-robin, so no claim
    is in both the fitting and the held-out records."""

    order = sorted(set(groups))
    random.Random(seed).shuffle(order)
    fold_of = {group: position % n_folds for position, group in enumerate(order)}
    return [fold_of[group] for group in groups]


def load_condition(root: Path, variant: str, slug: str, condition: str) -> dict[str, Any]:
    """One cached condition, memory-mapped and checked against its own metadata."""

    path = root / variant / slug / f"{condition}.pt"
    if not path.exists():
        raise FileNotFoundError(f"No activation cache at {path}")
    payload = load_torch(path, mmap=True)
    expected = {"schema_version": SCHEMA_VERSION, "condition": condition, "variant": variant,
                "model_slug": slug}
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"{path}: {key} is {payload.get(key)!r}, expected {value!r}")
    shape = (len(payload["norm_ids"]), len(payload["layers"]), payload["hidden_size"])
    if tuple(payload["assistant_start"].shape) != shape:
        raise ValueError(f"{path}: tensor shape is not {shape}")
    return payload


def load_conditions(root: Path, variant: str, slug: str, conditions: tuple[str, ...]
                    ) -> dict[str, dict[str, Any]]:
    """Several conditions, checked to hold the same records, layers and dataset."""

    caches = {name: load_condition(root, variant, slug, name) for name in conditions}
    first = caches[conditions[0]]
    for name, payload in caches.items():
        for key in ("norm_ids", "layers", "hidden_size", "dataset_sha256"):
            if payload.get(key) != first.get(key):
                raise ValueError(f"{name} disagrees with {conditions[0]} on {key}")
    return caches


def resolve_device(name: str) -> str:
    torch = require_torch()
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def country_index(norm_ids: list[str]) -> list[int]:
    codes = sorted({norm_id.split("__")[0] for norm_id in norm_ids})
    return [codes.index(norm_id.split("__")[0]) for norm_id in norm_ids]
