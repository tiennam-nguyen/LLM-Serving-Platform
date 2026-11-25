"""Pilot-only Week 8 v2 manifest and ordered workload adapter.

IMPORTANT: This module is intentionally scoped to infra/kaggle/week8_open_loop_pilot/
for Member A's dual-T4 pilot execution. Member B owns the final production
llm_serving_platform.benchmark.config and workload v2 implementations.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from llm_serving_platform.benchmark.artifacts import write_text_atomic
from llm_serving_platform.benchmark.workload import WorkloadEntry

SUPPORTED_V2_POLICIES = frozenset({"round_robin", "least_loaded"})
SUPPORTED_V2_PROFILES = frozenset({"steady", "burst", "heterogeneous"})
SUPPORTED_LOAD_LEVELS = frozenset({"pilot", "low", "medium", "near_saturation"})


class PilotManifestError(ValueError):
    """Raised when a pilot v2 manifest or workload violates the Week 8 contract."""


def _require_non_bool_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer (not bool)")
    if minimum is not None and value < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    return value


def _require_finite_number(
    value: Any,
    field_name: str,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    strictly_positive: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a finite number")
    num = float(value)
    if not math.isfinite(num):
        raise ValueError(f"{field_name} must be finite")
    if strictly_positive and num <= 0.0:
        raise ValueError(f"{field_name} must be > 0")
    if minimum is not None and num < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and num > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return num


class PilotModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str

    @field_validator("id", mode="before")
    @classmethod
    def _val_id(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("model.id must be a non-empty string")
        return v.strip()


class PilotTopologySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    worker_count: int

    @field_validator("worker_count", mode="before")
    @classmethod
    def _val_wc(cls, v: Any) -> int:
        return _require_non_bool_int(v, "topology.worker_count", minimum=1)


class PilotPolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str

    @field_validator("name", mode="before")
    @classmethod
    def _val_name(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("policy.name must be a non-empty string")
        norm = v.strip().lower()
        if norm not in SUPPORTED_V2_POLICIES:
            raise ValueError(f"Unsupported policy {v!r}; expected {sorted(SUPPORTED_V2_POLICIES)}")
        return norm


class PilotWorkloadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    path: str
    profile: str
    request_count: int

    @field_validator("path", mode="before")
    @classmethod
    def _val_path(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip() or "\x00" in v:
            raise ValueError("workload.path must be a non-empty valid string")
        return v.strip()

    @field_validator("profile", mode="before")
    @classmethod
    def _val_profile(cls, v: Any) -> str:
        if not isinstance(v, str) or v.strip() not in SUPPORTED_V2_PROFILES:
            raise ValueError(f"workload.profile must be one of {sorted(SUPPORTED_V2_PROFILES)}")
        return v.strip()

    @field_validator("request_count", mode="before")
    @classmethod
    def _val_count(cls, v: Any) -> int:
        return _require_non_bool_int(v, "workload.request_count", minimum=1)


class PilotWarmupSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    request_count: int

    @field_validator("request_count", mode="before")
    @classmethod
    def _val_count(cls, v: Any) -> int:
        return _require_non_bool_int(v, "warmup.request_count", minimum=0)


class PilotGenerationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    temperature: float
    max_tokens: int
    top_p: float | None = None

    @field_validator("temperature", mode="before")
    @classmethod
    def _val_temp(cls, v: Any) -> float:
        return _require_finite_number(v, "generation.temperature", minimum=0.0, maximum=2.0)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _val_max_tokens(cls, v: Any) -> int:
        return _require_non_bool_int(v, "generation.max_tokens", minimum=1)

    @field_validator("top_p", mode="before")
    @classmethod
    def _val_top_p(cls, v: Any) -> float | None:
        if v is None:
            return None
        return _require_finite_number(v, "generation.top_p", strictly_positive=True, maximum=1.0)


class PilotRunnerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    gateway_url: str
    request_timeout_s: float
    max_in_flight: int
    max_dispatch_lag_ms: float

    @field_validator("gateway_url", mode="before")
    @classmethod
    def _val_url(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("runner.gateway_url must be a non-empty string")
        cleaned = v.strip().rstrip("/")
        parsed = urlsplit(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(f"Invalid runner.gateway_url {v!r}")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("runner.gateway_url must not contain credentials")
        return cleaned

    @field_validator("request_timeout_s", mode="before")
    @classmethod
    def _val_timeout(cls, v: Any) -> float:
        return _require_finite_number(v, "runner.request_timeout_s", strictly_positive=True)

    @field_validator("max_in_flight", mode="before")
    @classmethod
    def _val_mif(cls, v: Any) -> int:
        return _require_non_bool_int(v, "runner.max_in_flight", minimum=1)

    @field_validator("max_dispatch_lag_ms", mode="before")
    @classmethod
    def _val_lag(cls, v: Any) -> float:
        return _require_finite_number(v, "runner.max_dispatch_lag_ms", strictly_positive=True)


class PilotLoadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: str
    measurement_window_ms: float

    @field_validator("mode", mode="before")
    @classmethod
    def _val_mode(cls, v: Any) -> str:
        if v != "open_loop":
            raise ValueError(f"load.mode must be 'open_loop', got {v!r}")
        return v

    @field_validator("measurement_window_ms", mode="before")
    @classmethod
    def _val_win(cls, v: Any) -> float:
        return _require_finite_number(v, "load.measurement_window_ms", strictly_positive=True)


class PilotComparisonSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    load_level: str
    repetition: int

    @field_validator("id", mode="before")
    @classmethod
    def _val_id(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip():
            raise ValueError("comparison.id must be a non-empty string")
        return v.strip()

    @field_validator("load_level", mode="before")
    @classmethod
    def _val_level(cls, v: Any) -> str:
        if not isinstance(v, str) or v.strip() not in SUPPORTED_LOAD_LEVELS:
            raise ValueError(
                f"comparison.load_level must be one of {sorted(SUPPORTED_LOAD_LEVELS)}"
            )
        return v.strip()

    @field_validator("repetition", mode="before")
    @classmethod
    def _val_rep(cls, v: Any) -> int:
        return _require_non_bool_int(v, "comparison.repetition", minimum=1)


class PilotManifestV2(BaseModel):
    """Pilot-only schema_version=2 manifest conforming to contract_w8.md."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    name: str
    model: PilotModelSpec
    topology: PilotTopologySpec
    policy: PilotPolicySpec
    workload: PilotWorkloadSpec
    seed: int
    warmup: PilotWarmupSpec
    generation: PilotGenerationSpec
    runner: PilotRunnerSpec
    load: PilotLoadSpec
    comparison: PilotComparisonSpec

    @field_validator("schema_version", mode="before")
    @classmethod
    def _val_ver(cls, v: Any) -> int:
        ver = _require_non_bool_int(v, "schema_version")
        if ver != 2:
            raise ValueError(f"PilotManifestV2 requires schema_version=2, got {ver!r}")
        return ver

    @field_validator("name", mode="before")
    @classmethod
    def _val_name(cls, v: Any) -> str:
        if not isinstance(v, str) or not v.strip() or "/" in v or "\\" in v or ".." in v:
            raise ValueError(f"Invalid manifest name {v!r}")
        return v.strip()

    @field_validator("seed", mode="before")
    @classmethod
    def _val_seed(cls, v: Any) -> int:
        return _require_non_bool_int(v, "seed")

    def to_normalized_dict(self) -> dict[str, Any]:
        gen: dict[str, Any] = {
            "temperature": self.generation.temperature,
            "max_tokens": self.generation.max_tokens,
        }
        if self.generation.top_p is not None:
            gen["top_p"] = self.generation.top_p
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "pilot_only": True,
            "model": {"id": self.model.id},
            "topology": {"worker_count": self.topology.worker_count},
            "policy": {"name": self.policy.name},
            "workload": {
                "path": self.workload.path,
                "profile": self.workload.profile,
                "request_count": self.workload.request_count,
            },
            "seed": self.seed,
            "warmup": {"request_count": self.warmup.request_count},
            "generation": gen,
            "runner": {
                "gateway_url": self.runner.gateway_url,
                "request_timeout_s": self.runner.request_timeout_s,
                "max_in_flight": self.runner.max_in_flight,
                "max_dispatch_lag_ms": self.runner.max_dispatch_lag_ms,
            },
            "load": {
                "mode": self.load.mode,
                "measurement_window_ms": self.load.measurement_window_ms,
            },
            "comparison": {
                "id": self.comparison.id,
                "load_level": self.comparison.load_level,
                "repetition": self.comparison.repetition,
            },
        }


@dataclass(frozen=True)
class PilotScheduledEntry:
    """Pilot-only scheduled entry implementing ScheduledEntryProtocol."""

    sequence: int
    scheduled_offset_ms: float
    entry: WorkloadEntry


def load_pilot_workload_v2(
    path: Path | str,
    *,
    expected_count: int,
    measurement_window_ms: float,
) -> list[PilotScheduledEntry]:
    """Load and validate a Week 8 v2 ordered JSONL workload strictly per contract."""
    file_path = Path(path)
    if not file_path.is_file():
        raise PilotManifestError(f"Workload file not found: {file_path}")

    raw_lines = file_path.read_text(encoding="utf-8").splitlines()
    entries: list[PilotScheduledEntry] = []
    seen_ids: set[str] = set()
    prev_offset = -1.0
    required_keys = {"id", "sequence", "scheduled_offset_ms", "messages"}

    for line_no, raw_line in enumerate(raw_lines, start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            row = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise PilotManifestError(f"{file_path}:{line_no}: malformed JSON ({exc.msg})") from exc

        if not isinstance(row, dict) or set(row.keys()) != required_keys:
            raise PilotManifestError(
                f"{file_path}:{line_no}: expected exact keys {sorted(required_keys)}"
            )

        entry_id = row["id"]
        if not isinstance(entry_id, str) or not entry_id.strip():
            raise PilotManifestError(f"{file_path}:{line_no}: invalid id")
        entry_id = entry_id.strip()
        if entry_id in seen_ids:
            raise PilotManifestError(f"{file_path}:{line_no}: duplicate id {entry_id!r}")
        seen_ids.add(entry_id)

        seq = row["sequence"]
        expected_seq = len(entries)
        if isinstance(seq, bool) or not isinstance(seq, int) or seq != expected_seq:
            raise PilotManifestError(
                f"{file_path}:{line_no}: expected contiguous sequence {expected_seq}, got {seq!r}"
            )

        offset_raw = row["scheduled_offset_ms"]
        if isinstance(offset_raw, bool) or not isinstance(offset_raw, (int, float)):
            raise PilotManifestError(f"{file_path}:{line_no}: invalid scheduled_offset_ms")
        offset = float(offset_raw)
        if not math.isfinite(offset) or offset < 0.0 or offset >= measurement_window_ms:
            raise PilotManifestError(
                f"{file_path}:{line_no}: scheduled_offset_ms={offset} outside "
                f"[0, {measurement_window_ms})"
            )
        if offset < prev_offset:
            raise PilotManifestError(
                f"{file_path}:{line_no}: decreasing scheduled_offset_ms={offset} < {prev_offset}"
            )
        prev_offset = offset

        msgs = row["messages"]
        if not isinstance(msgs, list) or not msgs:
            raise PilotManifestError(f"{file_path}:{line_no}: messages must be a non-empty list")
        validated_msgs: list[dict[str, str]] = []
        for msg in msgs:
            if not isinstance(msg, dict):
                raise PilotManifestError(f"{file_path}:{line_no}: message item must be an object")
            role = msg.get("role")
            content = msg.get("content")
            if not isinstance(role, str) or not role.strip():
                raise PilotManifestError(f"{file_path}:{line_no}: empty message role")
            if not isinstance(content, str) or not content.strip():
                raise PilotManifestError(f"{file_path}:{line_no}: empty message content")
            validated_msgs.append({"role": role.strip(), "content": content})

        entries.append(
            PilotScheduledEntry(
                sequence=seq,
                scheduled_offset_ms=offset,
                entry=WorkloadEntry(
                    index=seq,
                    entry_id=entry_id,
                    messages=tuple(validated_msgs),
                ),
            )
        )

    if len(entries) != expected_count:
        raise PilotManifestError(
            f"{file_path}: expected exactly {expected_count} entries, got {len(entries)}"
        )
    return entries


_OPENERS = (
    "Briefly summarize how",
    "In two concise sentences, explain why",
    "State one practical engineering reason why",
    "Describe the operational trade-off when",
    "Give a clear technical overview of how",
    "Outline two key factors determining how",
    "Contrast the runtime behavior when",
    "Explain in plain systems terminology why",
    "Identify the primary bottleneck when",
    "Summarize the tail-latency impact when",
)

_SUBJECTS = (
    "continuous batching schedulers",
    "KV-cache block allocators",
    "chunked prefill pipelines",
    "tensor-parallel attention heads",
    "round-robin HTTP proxies",
    "least-loaded active-request trackers",
    "server-sent event token flushers",
    "CUDA stream synchronization barriers",
    "paged attention memory pools",
    "open-loop Poisson load generators",
    "prefill vs decode compute phases",
    "head-of-line HTTP connection pools",
    "asynchronous health check probers",
    "inflight reservation counters",
    "vLLM worker GPU isolation",
    "tokenization worker threads",
    "backpressure admission controllers",
    "FIFO trace store ring buffers",
    "Prometheus histogram buckets",
    "request cancellation disconnect handlers",
)

_CONTEXTS = (
    "under steady concurrency on a 16 GB T4 GPU.",
    "when decode sequences reach sixty-four tokens.",
    "while multiple streaming clients read SSE chunks.",
    "across two independent single-GPU inference replicas.",
    "during sudden transitions from idle to moderate batch size.",
    "when prompts arrive at fixed monotonic deadlines.",
    "under memory bandwidth pressure during autoregressive decoding.",
    "while background metrics scrapes run concurrently.",
    "when one worker stalls on a slightly longer prompt.",
    "during clean shutdown and active-request draining.",
)


def generate_steady_pilot_workload(
    output_path: Path | str,
    *,
    request_count: int,
    measurement_window_ms: float,
    seed: int = 42,
) -> tuple[Path, list[PilotScheduledEntry], str]:
    """Generate a deterministic, unique-prompt steady open-loop workload for the pilot."""
    if request_count < 1:
        raise PilotManifestError(f"request_count must be >= 1, got {request_count}")
    if measurement_window_ms <= 0.0:
        raise PilotManifestError(
            f"measurement_window_ms must be > 0, got {measurement_window_ms}"
        )

    combos = [
        f"{opener} {subject} behave {ctx}"
        for opener in _OPENERS
        for subject in _SUBJECTS
        for ctx in _CONTEXTS
    ]
    rng = random.Random(seed)
    rng.shuffle(combos)
    if request_count > len(combos):
        raise PilotManifestError(
            f"Pilot generator supports up to {len(combos)} unique prompts, got {request_count}"
        )

    step_ms = float(measurement_window_ms) / float(request_count)
    rows: list[dict[str, Any]] = []
    for idx in range(request_count):
        offset_ms = round(idx * step_ms, 3)
        if offset_ms >= measurement_window_ms:
            offset_ms = round(measurement_window_ms - 0.001, 3)
        # Put unique case identifier inside distinct content so token prefixes do not repeat.
        prompt = f"[Case {idx + 1:04d}/{seed}] {combos[idx]}"
        rows.append(
            {
                "id": f"pilot-{idx:04d}",
                "sequence": idx,
                "scheduled_offset_ms": offset_ms,
                "messages": [{"role": "user", "content": prompt}],
            }
        )

    path = Path(output_path)
    text = "".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows)
    write_text_atomic(path, text)
    workload_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    scheduled = load_pilot_workload_v2(
        path,
        expected_count=request_count,
        measurement_window_ms=measurement_window_ms,
    )
    return path, scheduled, workload_sha256


def validate_pilot_manifest(raw: dict[str, Any]) -> PilotManifestV2:
    """Validate a raw manifest dict against the pilot-only v2 schema."""
    try:
        return PilotManifestV2.model_validate(raw)
    except ValidationError as exc:
        raise PilotManifestError(f"Invalid pilot manifest: {exc}") from exc
