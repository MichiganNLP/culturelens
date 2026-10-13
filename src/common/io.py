"""Shared, dependency-light helpers used across CultureLens.

This top-level module is the home for genuinely cross-package utilities: file
I/O (JSONL / JSON / text / YAML), prompt loading, and atomic writes.
Domain-specific or heavier-dependency helpers belong in
the per-package ``utils.py`` modules (e.g. ``benchmark/common.py``) or
dedicated modules (e.g. ``inference/vllm.py``).

The project uses a standard ``src`` package layout. Install it with
``python -m pip install -e .`` and import these helpers from
``common.io``.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import yaml


# --------------------------------------------------------------------------- #
# JSONL I/O
# --------------------------------------------------------------------------- #
def read_jsonl(
    path: str | Path,
    *,
    required_fields: Iterable[str] | None = None,
    require_nonempty: bool = False,
) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dict records.

    Each non-blank line must be a JSON object. ``required_fields`` (if given) are
    checked on every record; ``require_nonempty`` raises when no records are read.
    """
    jsonl_path = Path(path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

    required = set(required_fields) if required_fields else None
    records: list[dict[str, Any]] = []
    with jsonl_path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid JSON at {jsonl_path}:{line_number}: {error}"
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at {jsonl_path}:{line_number}."
                )
            if required:
                missing = required - set(record)
                if missing:
                    raise ValueError(
                        f"{jsonl_path}:{line_number} is missing {sorted(missing)}"
                    )
            records.append(record)

    if require_nonempty and not records:
        raise ValueError(f"No records found in {jsonl_path}.")
    return records


def write_jsonl(
    records: Iterable[dict[str, Any]] | Any,
    path: str | Path,
    *,
    sort_key: str | Callable[[dict[str, Any]], Any] | None = None,
) -> None:
    """Write dict records (or a pandas DataFrame) to a JSONL file.

    ``sort_key`` may be a field name or a callable; for a DataFrame it must be a
    column name. The parent directory is created if needed.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if hasattr(records, "to_dict"):  # pandas DataFrame
        frame = records
        if sort_key is not None:
            if not isinstance(sort_key, str):
                raise TypeError("DataFrame sort_key must be a column name.")
            frame = frame.sort_values(sort_key)
        rows = frame.to_dict(orient="records")
    else:
        rows = list(records)
        if sort_key is not None:
            key = sort_key if callable(sort_key) else (lambda item: item[sort_key])
            rows = sorted(rows, key=key)

    with out_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


# --------------------------------------------------------------------------- #
# JSON I/O
# --------------------------------------------------------------------------- #
def read_json(path: str | Path) -> Any:
    """Read and parse a JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(data: Any, path: str | Path, *, indent: int = 2) -> None:
    """Write ``data`` to a JSON file (parent dirs created, UTF-8, trailing newline)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=indent) + "\n", encoding="utf-8"
    )


def write_json_atomic(data: Any, path: str | Path, *, indent: int = 2) -> None:
    """Atomically write JSON so interrupted runs cannot leave a partial audit."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=f"{out_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_json(data, temporary, indent=indent)
        os.replace(temporary, out_path)
    finally:
        if temporary.exists():
            temporary.unlink()


# --------------------------------------------------------------------------- #
# Text / YAML / prompt loading
# --------------------------------------------------------------------------- #
def read_text(path: str | Path) -> str:
    """Read a UTF-8 text file."""
    text_path = Path(path)
    if not text_path.exists():
        raise FileNotFoundError(f"Text file not found: {text_path}")
    return text_path.read_text(encoding="utf-8")


def load_yaml(path: str | Path) -> Any:
    """Parse a YAML file."""
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")
    return yaml.safe_load(yaml_path.read_text(encoding="utf-8"))


def load_prompt(path: str | Path) -> str:
    """Load a stripped, non-empty prompt text file."""
    prompt = read_text(path).strip()
    if not prompt:
        raise ValueError(f"Prompt file is empty: {path}")
    return prompt


# --------------------------------------------------------------------------- #
# Atomic writes
# --------------------------------------------------------------------------- #
def write_jsonl_atomic(
    records: Iterable[dict[str, Any]] | Any,
    path: str | Path,
    *,
    sort_key: str | Callable[[dict[str, Any]], Any] | None = None,
) -> None:
    """Write JSONL via a temp file + ``os.replace`` so the destination is never
    left partially written if the process is interrupted mid-write.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=out_path.parent, prefix=f"{out_path.name}.", suffix=".tmp"
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        write_jsonl(records, temporary, sort_key=sort_key)
        os.replace(temporary, out_path)
    finally:
        if temporary.exists():
            temporary.unlink()
