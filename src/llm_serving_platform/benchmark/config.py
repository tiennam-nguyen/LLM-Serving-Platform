"""Versioned experiment manifest schema and strict validation for Week 7."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from llm_serving_platform.common.config import REPO_ROOT

SUPPORTED_SCHEMA_VERSIONS = frozenset({1})
SUPPORTED_WEEK7_POLICIES = frozenset({"single", "round_robin", "least_loaded"})
_SAFE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class ExperimentConfigError(ValueError):
    """Raised when an experiment manifest is missing, unreadable, or fails validation."""


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
    if strictly_positive and num <= 0:
        raise ValueError(f"{field_name} must be > 0")
    if minimum is not None and num < minimum:
        raise ValueError(f"{field_name} must be >= {minimum}")
    if maximum is not None and num > maximum:
        raise ValueError(f"{field_name} must be <= {maximum}")
    return num


class ModelSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str

    @field_validator("id", mode="before")
    @classmethod
    def _validate_id(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("model.id must be a non-empty string")
        return value.strip()


class TopologySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    worker_count: int

    @field_validator("worker_count", mode="before")
    @classmethod
    def _validate_worker_count(cls, value: Any) -> int:
        return _require_non_bool_int(value, "topology.worker_count", minimum=1)


class PolicySpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str

    @field_validator("name", mode="before")
    @classmethod
    def _validate_policy_name(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("policy.name must be a non-empty string")
        normalized = value.strip().lower()
        if normalized not in SUPPORTED_WEEK7_POLICIES:
            raise ValueError(
                f"Unsupported policy.name {value!r}. "
                f"Supported Week 7 policies: {sorted(SUPPORTED_WEEK7_POLICIES)}"
            )
        return normalized


class WorkloadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    request_count: int

    @field_validator("path", mode="before")
    @classmethod
    def _validate_path(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("workload.path must be a non-empty string")
        cleaned = value.strip()
        if "\x00" in cleaned:
            raise ValueError("workload.path must not contain null bytes")
        return cleaned

    @field_validator("request_count", mode="before")
    @classmethod
    def _validate_request_count(cls, value: Any) -> int:
        return _require_non_bool_int(value, "workload.request_count", minimum=1)


class WarmupSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_count: int

    @field_validator("request_count", mode="before")
    @classmethod
    def _validate_request_count(cls, value: Any) -> int:
        return _require_non_bool_int(value, "warmup.request_count", minimum=0)


class GenerationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    temperature: float
    max_tokens: int
    top_p: float | None = None

    @field_validator("temperature", mode="before")
    @classmethod
    def _validate_temperature(cls, value: Any) -> float:
        return _require_finite_number(value, "generation.temperature", minimum=0.0, maximum=2.0)

    @field_validator("max_tokens", mode="before")
    @classmethod
    def _validate_max_tokens(cls, value: Any) -> int:
        return _require_non_bool_int(value, "generation.max_tokens", minimum=1)

    @field_validator("top_p", mode="before")
    @classmethod
    def _validate_top_p(cls, value: Any) -> float | None:
        if value is None:
            return None
        return _require_finite_number(
            value, "generation.top_p", strictly_positive=True, maximum=1.0
        )


class RunnerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    concurrency: int
    gateway_url: str
    request_timeout_s: float

    @field_validator("concurrency", mode="before")
    @classmethod
    def _validate_concurrency(cls, value: Any) -> int:
        return _require_non_bool_int(value, "runner.concurrency", minimum=1)

    @field_validator("gateway_url", mode="before")
    @classmethod
    def _validate_gateway_url(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("runner.gateway_url must be a non-empty URL string")
        cleaned = value.strip().rstrip("/")
        parsed = urlsplit(cleaned)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError(
                f"runner.gateway_url must be a valid http(s) URL with hostname, got {value!r}"
            )
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("runner.gateway_url must not contain embedded credentials")
        if parsed.query or parsed.fragment:
            raise ValueError("runner.gateway_url must not contain query parameters or fragments")
        return cleaned

    @field_validator("request_timeout_s", mode="before")
    @classmethod
    def _validate_timeout(cls, value: Any) -> float:
        return _require_finite_number(value, "runner.request_timeout_s", strictly_positive=True)


class ExperimentConfig(BaseModel):
    """Canonical versioned manifest contract (schema_version=1)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int
    name: str
    model: ModelSpec
    topology: TopologySpec
    policy: PolicySpec
    workload: WorkloadSpec
    seed: int
    warmup: WarmupSpec
    generation: GenerationSpec
    runner: RunnerSpec

    @field_validator("schema_version", mode="before")
    @classmethod
    def _validate_schema_version(cls, value: Any) -> int:
        version = _require_non_bool_int(value, "schema_version")
        if version not in SUPPORTED_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version {version!r}; "
                f"supported versions: {sorted(SUPPORTED_SCHEMA_VERSIONS)}"
            )
        return version

    @field_validator("name", mode="before")
    @classmethod
    def _validate_name(cls, value: Any) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("name must be a non-empty string")
        cleaned = value.strip()
        if ".." in cleaned or "/" in cleaned or "\\" in cleaned or not _SAFE_NAME_RE.match(cleaned):
            raise ValueError(
                f"name {value!r} must match {_SAFE_NAME_RE.pattern} and contain no path separators"
            )
        return cleaned

    @field_validator("seed", mode="before")
    @classmethod
    def _validate_seed(cls, value: Any) -> int:
        return _require_non_bool_int(value, "seed")

    def to_normalized_dict(self) -> dict[str, Any]:
        """Return a canonical, deterministic dictionary representation of the validated config."""
        generation: dict[str, Any] = {
            "temperature": self.generation.temperature,
            "max_tokens": self.generation.max_tokens,
        }
        if self.generation.top_p is not None:
            generation["top_p"] = self.generation.top_p
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "model": {"id": self.model.id},
            "topology": {"worker_count": self.topology.worker_count},
            "policy": {"name": self.policy.name},
            "workload": {
                "path": self.workload.path,
                "request_count": self.workload.request_count,
            },
            "seed": self.seed,
            "warmup": {"request_count": self.warmup.request_count},
            "generation": generation,
            "runner": {
                "concurrency": self.runner.concurrency,
                "gateway_url": self.runner.gateway_url,
                "request_timeout_s": self.runner.request_timeout_s,
            },
        }


class ScheduledWorkloadSpec(WorkloadSpec):
    profile: Literal[
        "steady",
        "burst",
        "heterogeneous",
        "steady_mixed",
        "irregular_mixed",
        "burst_mixed",
        "ramp_mixed",
        "microburst_mixed",
    ]


class OpenLoopRunnerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    gateway_url: str
    request_timeout_s: float
    max_in_flight: int
    # Observation replay warns; the historical pilot retains its strict limit semantics.
    max_dispatch_lag_ms: float

    @field_validator("gateway_url", mode="before")
    @classmethod
    def _url(cls, value: Any) -> str:
        return RunnerSpec._validate_gateway_url(value)

    @field_validator("request_timeout_s", "max_dispatch_lag_ms", mode="before")
    @classmethod
    def _positive_number(cls, value: Any) -> float:
        return _require_finite_number(value, "runner timing", strictly_positive=True)

    @field_validator("max_in_flight", mode="before")
    @classmethod
    def _capacity(cls, value: Any) -> int:
        return _require_non_bool_int(value, "runner.max_in_flight", minimum=1)


class OpenLoopLoadSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["open_loop"]
    measurement_window_ms: float

    @field_validator("measurement_window_ms", mode="before")
    @classmethod
    def _window(cls, value: Any) -> float:
        return _require_finite_number(value, "load.measurement_window_ms", strictly_positive=True)


class ComparisonSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    load_level: Literal["pilot", "low", "medium", "near_saturation"]
    repetition: int

    @field_validator("id", mode="before")
    @classmethod
    def _id(cls, value: Any) -> str:
        return ExperimentConfig._validate_name(value)

    @field_validator("repetition", mode="before")
    @classmethod
    def _repetition(cls, value: Any) -> int:
        return _require_non_bool_int(value, "comparison.repetition", minimum=1)


class OpenLoopPolicySpec(PolicySpec):
    @field_validator("name", mode="before")
    @classmethod
    def _rr_or_ll(cls, value: Any) -> str:
        policy = PolicySpec._validate_policy_name(value)
        if policy not in {"round_robin", "least_loaded"}:
            raise ValueError("Schema v2 supports round_robin and least_loaded only")
        return policy


class ExperimentConfigV2(BaseModel):
    """Production W8 contract. V1 remains an independent, backward-compatible schema."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[2]
    name: str
    model: ModelSpec
    topology: TopologySpec
    policy: OpenLoopPolicySpec
    workload: ScheduledWorkloadSpec
    seed: int
    warmup: WarmupSpec
    generation: GenerationSpec
    runner: OpenLoopRunnerSpec
    load: OpenLoopLoadSpec
    comparison: ComparisonSpec

    @field_validator("schema_version", mode="before")
    @classmethod
    def _version(cls, value: Any) -> int:
        version = _require_non_bool_int(value, "schema_version")
        if version != 2:
            raise ValueError("ExperimentConfigV2 requires schema_version=2")
        return version

    @field_validator("name", mode="before")
    @classmethod
    def _name(cls, value: Any) -> str:
        return ExperimentConfig._validate_name(value)

    @field_validator("seed", mode="before")
    @classmethod
    def _seed(cls, value: Any) -> int:
        return _require_non_bool_int(value, "seed")

    def to_normalized_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def validate_manifest_data(raw: Any) -> ExperimentConfig | ExperimentConfigV2:
    """Choose a schema explicitly; never coerce a v1 manifest to open-loop execution."""
    if not isinstance(raw, dict):
        raise ExperimentConfigError("Manifest must be a JSON object")
    version = raw.get("schema_version")
    if isinstance(version, bool) or not isinstance(version, int) or version not in {1, 2}:
        raise ExperimentConfigError("schema_version must be integer 1 or 2")
    try:
        schema = ExperimentConfig if version == 1 else ExperimentConfigV2
        return schema.model_validate(raw)
    except ValidationError as exc:
        # ValidationError can include user inputs: persist/report only field locations.
        locations = [".".join(map(str, error["loc"])) for error in exc.errors()]
        raise ExperimentConfigError(f"Invalid manifest fields: {', '.join(locations)}") from exc


def resolve_workload_path(
    workload_path: str,
    *,
    base_dir: Path | None = None,
    manifest_dir: Path | None = None,
) -> Path:
    """Safely resolve a manifest workload path, rejecting relative path-traversal escapes."""
    raw = Path(workload_path)
    if raw.is_absolute():
        resolved = raw.resolve()
        if not resolved.is_file():
            raise ExperimentConfigError(f"Workload file does not exist: {workload_path!r}")
        return resolved

    roots: list[Path] = []
    if base_dir is not None:
        roots.append(base_dir.resolve())
    else:
        roots.append(REPO_ROOT.resolve())
        if manifest_dir is not None and manifest_dir.resolve() not in roots:
            roots.append(manifest_dir.resolve())

    escaped = False
    for root in roots:
        candidate = (root / raw).resolve()
        if not candidate.is_relative_to(root):
            escaped = True
            continue
        if candidate.is_file():
            return candidate

    if escaped:
        raise ExperimentConfigError(
            f"Workload path {workload_path!r} escapes allowed root directory "
            "(path traversal rejected)"
        )
    raise ExperimentConfigError(f"Workload file not found: {workload_path!r}")


def load_manifest(
    manifest_path: Path | str,
    *,
    base_dir: Path | None = None,
    gateway_url_override: str | None = None,
) -> tuple[ExperimentConfig | ExperimentConfigV2, Path]:
    """Load, validate, and resolve a JSON experiment manifest file."""
    path = Path(manifest_path)
    if not path.is_file():
        raise ExperimentConfigError(f"Manifest file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ExperimentConfigError(f"Unable to read manifest {path}: {exc}") from exc

    try:
        raw_data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ExperimentConfigError(
            f"Malformed JSON in manifest {path} at line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc

    if not isinstance(raw_data, dict):
        raise ExperimentConfigError(f"Manifest {path} must be a JSON object")

    if gateway_url_override is not None:
        runner_section = raw_data.get("runner")
        if isinstance(runner_section, dict):
            raw_data = {
                **raw_data,
                "runner": {**runner_section, "gateway_url": gateway_url_override},
            }

    config = validate_manifest_data(raw_data)

    resolved_workload = resolve_workload_path(
        config.workload.path,
        base_dir=base_dir,
        manifest_dir=path.resolve().parent,
    )
    return config, resolved_workload
