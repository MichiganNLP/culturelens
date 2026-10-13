"""Encode cached states with a sparse autoencoder and store the result sparsely.

One file per model, layer and prompt condition. Features are kept in CSR form
(``row_ptr``, ``feature_ids``, ``feature_values``), rows in the cache's record
order, so a file is a few megabytes where the dense matrix would be gigabytes.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from mechanistic_analysis.sae.backend import MatrixSAE

FORMAT_VERSION = "culturelens-v2-sae-features-v1"


@dataclass(frozen=True, slots=True)
class SparseFeatures:
    model: str
    condition: str
    layer: int
    shape: tuple[int, int]
    row_ptr: torch.Tensor
    feature_ids: torch.Tensor
    feature_values: torch.Tensor
    norm_ids: tuple[str, ...]
    sae_metadata: dict[str, Any]
    cache: dict[str, Any]
    quality: dict[str, float]

    @property
    def n_rows(self) -> int:
        return self.shape[0]

    @property
    def d_sae(self) -> int:
        return self.shape[1]

    @property
    def nnz(self) -> int:
        return int(self.feature_ids.numel())

    def to_dense(self) -> torch.Tensor:
        """The full (records, d_sae) matrix; for small inputs and checks only."""

        return torch.sparse_csr_tensor(self.row_ptr.long(), self.feature_ids.long(),
                                       self.feature_values.float(), size=self.shape).to_dense()

    def to_scipy(self):
        from scipy import sparse

        return sparse.csr_matrix(
            (self.feature_values.float().numpy(), self.feature_ids.long().numpy(),
             self.row_ptr.long().numpy()), shape=self.shape)

    def payload(self) -> dict[str, Any]:
        return {"format_version": FORMAT_VERSION, "model": self.model, "condition": self.condition,
                "layer": self.layer, "shape": self.shape, "row_ptr": self.row_ptr,
                "feature_ids": self.feature_ids, "feature_values": self.feature_values,
                "norm_ids": list(self.norm_ids), "sae_metadata": self.sae_metadata,
                "cache": self.cache, "quality": self.quality}


def encode_states(states: torch.Tensor, sae: MatrixSAE, *, batch_size: int,
                  value_dtype: torch.dtype = torch.float32
                  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """CSR features for ``states`` (records, d_model), and how well the SAE fits them.

    ``relative_error`` is the squared reconstruction error over the squared norm
    of the states, and ``cosine`` the mean cosine between a state and its
    reconstruction. ``fraction_of_variance_unexplained`` divides the same error
    by the squared distance of the states from their mean instead. The states
    here all come from one position (the last prompt token), so they vary far
    less than activations in general and this last number is a much harder
    test than the figure SAE papers report over all tokens: above 1 means the
    SAE's error is larger than the differences between records. All are
    computed in float32 on the states as cached.
    """

    if states.ndim != 2 or states.shape[1] != sae.d_model:
        raise ValueError(f"Expected (records, {sae.d_model}) states, got {tuple(states.shape)}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = sae.w_enc.device
    mean = states.float().mean(dim=0).to(device)
    row_ptr = [torch.zeros(1, dtype=torch.int64)]
    ids: list[torch.Tensor] = []
    values: list[torch.Tensor] = []
    cumulative, error, total, cosine, norm = 0, 0.0, 0.0, 0.0, 0.0
    with torch.inference_mode():
        for start in range(0, len(states), batch_size):
            x = states[start:start + batch_size].to(device)
            features = sae.encode(x)
            rebuilt = sae.decode(features).float()
            x = x.float()
            error += float((x - rebuilt).pow(2).sum())
            total += float((x - mean).pow(2).sum())
            norm += float(x.pow(2).sum())
            cosine += float(torch.nn.functional.cosine_similarity(x, rebuilt, dim=-1).sum())
            active = features.ne(0)
            counts = active.sum(dim=1, dtype=torch.int64).cpu()
            row_ptr.append(counts.cumsum(0) + cumulative)
            cumulative += int(counts.sum())
            ids.append(active.nonzero(as_tuple=False)[:, 1].to(device="cpu", dtype=torch.int32))
            values.append(features[active].to(device="cpu", dtype=value_dtype))
    feature_ids = torch.cat(ids) if ids else torch.empty(0, dtype=torch.int32)
    quality = {"mean_l0": cumulative / max(1, len(states)),
               "relative_error": error / norm if norm > 0 else float("nan"),
               "fraction_of_variance_unexplained": error / total if total > 0 else float("nan"),
               "cosine": cosine / max(1, len(states)),
               "features_ever_active": int(feature_ids.unique().numel())}
    return (torch.cat(row_ptr), feature_ids,
            torch.cat(values) if values else torch.empty(0, dtype=value_dtype), quality)


def validate(features: SparseFeatures) -> None:
    if tuple(features.row_ptr.shape) != (features.n_rows + 1,):
        raise ValueError("row_ptr has the wrong length")
    if int(features.row_ptr[0]) != 0 or int(features.row_ptr[-1]) != features.nnz:
        raise ValueError("row_ptr endpoints disagree with the number of stored values")
    if torch.any(features.row_ptr[1:] < features.row_ptr[:-1]):
        raise ValueError("row_ptr is not monotonic")
    if features.feature_values.numel() != features.nnz:
        raise ValueError("feature ids and values differ in length")
    if features.nnz and not 0 <= int(features.feature_ids.min()) <= int(
            features.feature_ids.max()) < features.d_sae:
        raise ValueError("a feature id is outside the SAE's width")
    if len(features.norm_ids) != features.n_rows or len(set(features.norm_ids)) != features.n_rows:
        raise ValueError("record ids are misaligned or duplicated")


def save(features: SparseFeatures, path: Path) -> None:
    """Written beside the target and renamed, so a killed job leaves no partial file."""

    validate(features)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".tmp")
    os.close(descriptor)
    temporary = Path(name)
    try:
        torch.save(features.payload(), temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load(path: Path) -> SparseFeatures:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path} is not a {FORMAT_VERSION} file")
    features = SparseFeatures(
        model=str(payload["model"]), condition=str(payload["condition"]), layer=int(payload["layer"]),
        shape=tuple(map(int, payload["shape"])), row_ptr=payload["row_ptr"],
        feature_ids=payload["feature_ids"], feature_values=payload["feature_values"],
        norm_ids=tuple(map(str, payload["norm_ids"])), sae_metadata=dict(payload["sae_metadata"]),
        cache=dict(payload["cache"]), quality=dict(payload["quality"]))
    validate(features)
    return features


__all__ = ["FORMAT_VERSION", "SparseFeatures", "encode_states", "load", "save", "validate"]
