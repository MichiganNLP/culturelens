"""Records, and the pool that turns them into training prompts.

The split files are JSON arrays of records::

    {"norm_id": "AFG__communication__001__013", "norm": "...", "values": [...],
     "scenario": {"country_context", "situation", "question", "norm_evidence",
                  "opposite_norm_evidence", "opposite_norm_text", "value_evidence"},
     "options": {"A": "...", "B": "..."}}

Option ``A`` is the norm-aligned answer; the evaluation shuffles by norm id, and
training re-draws the order every time a prompt is sampled (see
:mod:`finetune.prompts`).
"""

from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from . import PAIRED_VIEWS, VALUE_VIEWS

REQUIRED = ("norm_id", "norm", "scenario", "options")
SCENARIO_FIELDS = (
    "country_context", "situation", "question", "norm_evidence",
    "opposite_norm_evidence", "value_evidence",
)
_COUNTRY = re.compile(r"takes place in (?P<name>[^.]+)\.")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """A split file: a JSON array, or JSON Lines for older builds."""

    path = Path(path)
    text = path.read_text(encoding="utf-8")
    stripped = text.lstrip()
    if stripped.startswith("["):
        rows = json.loads(text)
    else:
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
    if not rows:
        raise ValueError(f"{path} holds no records")
    seen: set[str] = set()
    for row in rows:
        missing = [key for key in REQUIRED if key not in row]
        missing += [f"scenario.{key}" for key in SCENARIO_FIELDS if key not in row.get("scenario", {})]
        if missing:
            raise ValueError(f"{row.get('norm_id')!r} in {path} lacks {missing}")
        if set(row["options"]) != {"A", "B"}:
            raise ValueError(f"{row['norm_id']!r} in {path} does not have exactly options A and B")
        if row["norm_id"] in seen:
            raise ValueError(f"Duplicate norm_id {row['norm_id']!r} in {path}")
        seen.add(row["norm_id"])
    return rows


def iso3(record: dict[str, Any]) -> str:
    return record["norm_id"].split("__")[0]


def category(record: dict[str, Any]) -> str:
    return record["norm_id"].split("__")[1]


def country_name(record: dict[str, Any]) -> str:
    match = _COUNTRY.search(str(record["scenario"].get("country_context") or ""))
    if match is None:
        raise ValueError(f"Cannot read a country name for {record['norm_id']}")
    return match.group("name").strip()


def eligible(record: dict[str, Any], view: str) -> bool:
    """Whether a record can be shown under a view: value views need evidence."""

    return view not in VALUE_VIEWS or bool(record["scenario"].get("value_evidence"))


class PromptPool:
    """Records drawn evenly across views, and across tasks when there are two.

    **Even across views.** Sampling records as they come would spend most
    rollouts on whatever the policy already handles, where every rollout in a
    group agrees and the group-relative advantage is zero.

    **Paired views move in lockstep.** ``country_norm`` and
    ``country_opposite_norm`` read one shared ordering of records with cursors
    that advance together, so the k-th false-norm prompt and the k-th true-norm
    prompt are the same record. That requires both to be drawn equally often per
    step, which :meth:`draw` guarantees when the per-task count divides evenly
    across views; the constructor refuses a count that would break it.

    **Even across tasks.** With two tasks, each draws half the step's prompts and
    runs its own full rotation over the views, so no view is tied to one task.
    """

    def __init__(self, records: list[dict[str, Any]], views: tuple[str, ...],
                 kinds: tuple[str, ...], seed: int, *, prompts_per_step: int | None = None) -> None:
        self.views = tuple(views)
        self.kinds = tuple(kinds)
        self.random = random.Random(seed)
        paired = [view for view in self.views if view in PAIRED_VIEWS]
        if paired and len(paired) != len(PAIRED_VIEWS):
            raise ValueError(f"{PAIRED_VIEWS} must be trained together, got {paired}")
        if paired and prompts_per_step is not None:
            per_kind, remainder = divmod(prompts_per_step, len(self.kinds))
            if remainder or per_kind % len(self.views):
                raise ValueError(
                    f"prompts_per_step={prompts_per_step} must be a multiple of "
                    f"{len(self.kinds) * len(self.views)} (tasks x views) so the paired "
                    "views are drawn equally often and stay on the same records")

        # One ordering per group; paired views share theirs.
        self.group_of = {view: ("paired" if view in PAIRED_VIEWS else view) for view in self.views}
        self.orders: dict[str, list[dict[str, Any]]] = {}
        self.epoch: dict[str, int] = {}
        for view in self.views:
            group = self.group_of[view]
            if group in self.orders:
                continue
            rows = [record for record in records if eligible(record, view)]
            if not rows:
                raise ValueError(f"No records are eligible for {view}")
            self.random.shuffle(rows)
            self.orders[group] = rows
            self.epoch[group] = 0
        # Cursor and epoch per (task, view): each task runs its own rotation.
        self.cursor = {(kind, view): 0 for kind in self.kinds for view in self.views}
        self.seen_epoch = {(kind, view): 0 for kind in self.kinds for view in self.views}
        # Each task gets its own shuffle, so the two tasks do not walk the same
        # records in the same order, and its own epochs, so one task wrapping
        # does not reshuffle an order the other is half way through. Within a
        # task the paired views still share one order.
        self.task_orders: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for kind in self.kinds:
            self.task_orders[kind] = {}
            for group, rows in self.orders.items():
                copy = list(rows)
                self.random.shuffle(copy)
                self.task_orders[kind][group] = copy
        self.task_epoch = {kind: dict(self.epoch) for kind in self.kinds}

    def sizes(self) -> dict[str, int]:
        return {view: len(self.orders[self.group_of[view]]) for view in self.views}

    def prompts_per_epoch(self) -> int:
        """Prompts in one epoch: every record drawn once under every trained
        view, by every task. The largest view sets the length; a smaller one
        (the value view, which needs value evidence) cycles within it."""

        return max(self.sizes().values()) * len(self.views) * len(self.kinds)

    # A resumed run must draw exactly the prompts the interrupted one would
    # have: the same orders, the same positions, the same random stream.
    def state(self) -> dict[str, Any]:
        return {
            "views": list(self.views), "kinds": list(self.kinds),
            "orders": {kind: {group: [row["norm_id"] for row in rows]
                              for group, rows in groups.items()}
                       for kind, groups in self.task_orders.items()},
            "cursor": [[kind, view, value] for (kind, view), value in self.cursor.items()],
            "seen_epoch": [[kind, view, value] for (kind, view), value in self.seen_epoch.items()],
            "task_epoch": self.task_epoch,
            "random": _encode_random(self.random.getstate()),
        }

    def load_state(self, state: dict[str, Any]) -> None:
        if state["views"] != list(self.views) or state["kinds"] != list(self.kinds):
            raise ValueError("Saved prompt pool was built for different views or tasks")
        by_id = {row["norm_id"]: row for rows in self.orders.values() for row in rows}
        for kind, groups in state["orders"].items():
            for group, ids in groups.items():
                if sorted(ids) != sorted(row["norm_id"] for row in self.task_orders[kind][group]):
                    raise ValueError("Saved prompt pool covers different records")
                self.task_orders[kind][group] = [by_id[norm_id] for norm_id in ids]
        self.cursor = {(kind, view): value for kind, view, value in state["cursor"]}
        self.seen_epoch = {(kind, view): value for kind, view, value in state["seen_epoch"]}
        self.task_epoch = {kind: dict(groups) for kind, groups in state["task_epoch"].items()}
        self.random.setstate(_decode_random(state["random"]))

    def _next(self, kind: str, view: str) -> dict[str, Any]:
        group = self.group_of[view]
        rows = self.task_orders[kind][group]
        key = (kind, view)
        if self.cursor[key] >= len(rows):
            # The first paired view to run out reshuffles the shared order; its
            # twin finds the epoch already advanced and just restarts on it.
            if self.seen_epoch[key] == self.task_epoch[kind][group]:
                self.random.shuffle(rows)
                self.task_epoch[kind][group] += 1
            self.seen_epoch[key] = self.task_epoch[kind][group]
            self.cursor[key] = 0
        record = rows[self.cursor[key]]
        self.cursor[key] += 1
        return record

    def draw(self, count: int) -> list[dict[str, Any]]:
        """``count`` prompt items, each ``{record, view, kind, country}``."""

        items: list[dict[str, Any]] = []
        per_kind = [count // len(self.kinds)] * len(self.kinds)
        for index in range(count % len(self.kinds)):
            per_kind[index] += 1
        for kind, share in zip(self.kinds, per_kind):
            for index in range(share):
                view = self.views[index % len(self.views)]
                record = self._next(kind, view)
                items.append({"record": record, "view": view, "kind": kind,
                              "country": country_name(record)})
        return items


def _encode_random(state: tuple) -> list:
    version, internal, gauss = state
    return [version, list(internal), gauss]


def _decode_random(state: list) -> tuple:
    version, internal, gauss = state
    return (version, tuple(internal), gauss)


def fixed_sample(records: list[dict[str, Any]], view: str, count: int, seed: int) -> list[dict[str, Any]]:
    """A reproducible validation subset for one view: the same items every evaluation."""

    rows = [record for record in records if eligible(record, view)]
    random.Random(f"{seed}:{view}").shuffle(rows)
    return rows[:count] if count > 0 else rows


__all__ = ["PromptPool", "category", "country_name", "eligible", "fixed_sample",
           "iso3", "load_records"]
