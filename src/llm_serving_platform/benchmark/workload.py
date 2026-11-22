"""Deterministic JSONL workload loader and selection helpers for Week 7."""

from __future__ import annotations

import json
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
            raise WorkloadError(
                f"{path}: line {line_no}: messages[{msg_idx}] must be an object"
            )
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


def load_workload(path: Path | str, *, min_entries: int = 1) -> list[WorkloadEntry]:
    """Load and validate a deterministic JSONL workload file."""
    file_path = Path(path)
    if not file_path.is_file():
        raise WorkloadError(f"Workload file not found: {file_path}")

    try:
        raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise WorkloadError(f"Unable to read workload file {file_path}: {exc}") from exc

    entries: list[WorkloadEntry] = []
    for line_no, raw_line in enumerate(raw_lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise WorkloadError(
                f"{file_path}: line {line_no}: malformed JSON ({exc.msg})"
            ) from exc
        entries.append(_validate_entry(parsed, line_no=line_no, path=file_path))

    if len(entries) < min_entries:
        raise WorkloadError(
            f"Workload {file_path} contains {len(entries)} valid entries; "
            f"expected at least {min_entries}"
        )
    return entries


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
