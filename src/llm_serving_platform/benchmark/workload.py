"""Deterministic JSONL workload loader and selection helpers for Week 7."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class WorkloadError(ValueError):
    """Raised when a workload JSONL file is malformed or contains insufficient valid entries."""


@dataclass(frozen=True)
class WorkloadEntry:
    """One validated chat-completion prompt entry from a JSONL workload."""

    index: int
    messages: tuple[dict[str, str], ...]
    entry_id: str | None = None

    def as_messages_list(self) -> list[dict[str, str]]:
        return [dict(m) for m in self.messages]


@dataclass(frozen=True)
class ScheduledWorkloadEntry:
    sequence: int
    scheduled_offset_ms: float
    entry: WorkloadEntry


def _validate_entry(raw: Any, *, line_no: int, path: Path) -> WorkloadEntry:
    if not isinstance(raw, dict):
        raise WorkloadError(f"{path}: line {line_no}: entry must be a JSON object")

    allowed_keys = {"messages", "id"}
    extra_keys = set(raw.keys()) - allowed_keys
    if extra_keys:
        raise WorkloadError(
            f"{path}: line {line_no}: unexpected workload keys {sorted(extra_keys)}"
        )

    messages = raw.get("messages")
    if not isinstance(messages, list) or not messages:
        raise WorkloadError(f"{path}: line {line_no}: 'messages' must be a non-empty list")

    validated_messages: list[dict[str, str]] = []
    for msg_idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            raise WorkloadError(f"{path}: line {line_no}: messages[{msg_idx}] must be an object")
        role = msg.get("role")
        content = msg.get("content")
        if not isinstance(role, str) or not role.strip():
            raise WorkloadError(
                f"{path}: line {line_no}: messages[{msg_idx}].role must be a non-empty string"
            )
        if not isinstance(content, str) or not content.strip():
            raise WorkloadError(
                f"{path}: line {line_no}: messages[{msg_idx}].content must be a non-empty string"
            )
        validated_messages.append({"role": role.strip(), "content": content})

    entry_id = raw.get("id")
    if entry_id is not None and (not isinstance(entry_id, str) or not entry_id.strip()):
        raise WorkloadError(f"{path}: line {line_no}: optional 'id' must be a non-empty string")

    return WorkloadEntry(
        index=line_no - 1,
        messages=tuple(validated_messages),
        entry_id=entry_id.strip() if isinstance(entry_id, str) else None,
    )


def load_workload(
    path: Path | str,
    *,
    min_entries: int = 1,
    schema_version: int = 1,
    expected_count: int | None = None,
    measurement_window_ms: float | None = None,
) -> list[WorkloadEntry] | list[ScheduledWorkloadEntry]:
    """Load and validate a deterministic JSONL workload file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise WorkloadError(f"Workload file not found: {file_path}")

    try:
        raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise WorkloadError(f"Unable to read workload file {file_path}: {exc}") from exc

    if type(schema_version) is not int or schema_version not in {1, 2}:
        raise WorkloadError("Unsupported workload schema_version")
    entries: list[Any] = []
    seen_ids: set[str] = set()
    previous_offset = -1.0
    for line_no, raw_line in enumerate(raw_lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise WorkloadError(f"{file_path}: line {line_no}: malformed JSON ({exc.msg})") from exc
        if schema_version == 1:
            entries.append(_validate_entry(parsed, line_no=line_no, path=file_path))
            continue
        required = {"id", "sequence", "scheduled_offset_ms", "messages"}
        if not isinstance(parsed, dict) or set(parsed) != required:
            raise WorkloadError(f"Line {line_no}: expected scheduled workload fields")
        entry = _validate_entry(
            {"id": parsed["id"], "messages": parsed["messages"]},
            line_no=line_no,
            path=file_path,
        )
        if entry.entry_id is None or entry.entry_id in seen_ids:
            raise WorkloadError(f"Line {line_no}: scheduled IDs must be nonempty and unique")
        seen_ids.add(entry.entry_id)
        seq, offset = parsed["sequence"], parsed["scheduled_offset_ms"]
        if isinstance(seq, bool) or not isinstance(seq, int) or seq != len(entries):
            raise WorkloadError(f"Line {line_no}: sequence must be contiguous from zero")
        if (
            isinstance(offset, bool)
            or not isinstance(offset, (float, int))
            or not math.isfinite(offset)
            or offset < 0
            or offset < previous_offset
        ):
            raise WorkloadError(f"Line {line_no}: invalid or decreasing scheduled offset")
        if measurement_window_ms is not None and offset >= measurement_window_ms:
            raise WorkloadError(f"Line {line_no}: offset lies outside measurement window")
        previous_offset = float(offset)
        entries.append(ScheduledWorkloadEntry(seq, float(offset), entry))

    if len(entries) < min_entries:
        raise WorkloadError(
            f"Workload {file_path} contains {len(entries)} valid entries; "
            f"expected at least {min_entries}"
        )
    if expected_count is not None and len(entries) != expected_count:
        raise WorkloadError(f"Expected exactly {expected_count} entries, found {len(entries)}")
    return entries


def select_scheduled_entries(
    entries: list[ScheduledWorkloadEntry],
    request_count: int,
) -> list[ScheduledWorkloadEntry]:
    """Preserve the materialized schedule, with no sampling, shuffle or silent truncation."""
    if len(entries) != request_count or any(
        not isinstance(item, ScheduledWorkloadEntry) or item.sequence != i
        for i, item in enumerate(entries)
    ):
        raise WorkloadError("Scheduled workload count/order mismatch")
    return list(entries)


def select_warmup_entries(entries: list[WorkloadEntry], count: int) -> list[WorkloadEntry]:
    """Deterministically select warm-up entries (cycling if count > len(entries))."""
    if count <= 0:
        return []
    if not entries:
        raise WorkloadError("Cannot select warm-up entries from an empty workload")
    return [entries[i % len(entries)] for i in range(count)]


def select_measured_entries(
    entries: list[WorkloadEntry],
    request_count: int,
    *,
    seed: int,
) -> list[WorkloadEntry]:
    """Deterministically select and order measured workload entries using the manifest seed."""
    if len(entries) < request_count:
        raise WorkloadError(
            f"Workload contains {len(entries)} entries; need at least {request_count}"
        )
    indices = list(range(len(entries)))
    rng = random.Random(seed)
    rng.shuffle(indices)
    return [entries[idx] for idx in indices[:request_count]]
