"""Canonical run directory management, atomic artifact writes, and safe environment metadata."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from llm_serving_platform.benchmark.config import ExperimentConfig
from llm_serving_platform.common.config import REPO_ROOT

DEFAULT_RUNS_ROOT = REPO_ROOT / "experiments" / "runs"

_RELEVANT_PACKAGES = (
    "llm-serving-platform",
    "fastapi",
    "httpx",
    "pydantic",
    "prometheus-client",
    "uvicorn",
    "vllm",
    "torch",
    "transformers",
)

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9-]+")


def utc_now_iso() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""
    return datetime.now(timezone.utc).isoformat()


def _slugify(name: str, max_len: int = 32) -> str:
    slug = _SLUG_CLEAN_RE.sub("-", name.strip().lower()).strip("-")
    slug = re.sub(r"-{2,}", "-", slug)
    return (slug or "run")[:max_len].rstrip("-") or "run"


def generate_run_id(
    name: str,
    git_revision: str | None = None,
    *,
    started_at: datetime | None = None,
) -> str:
    """Create a collision-resistant, human-readable run identifier."""
    ts = (started_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
    stamp = ts.strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(name)
    short_sha = (git_revision or "nogit")[:7].lower()
    suffix = uuid.uuid4().hex[:6]
    return f"{stamp}-{slug}-{short_sha}-{suffix}"


def _safe_git(*args: str, cwd: Path = REPO_ROOT) -> str | None:
    try:
        proc = subprocess.run(
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout.strip()


def capture_git_info(cwd: Path = REPO_ROOT) -> dict[str, Any]:
    """Capture Git revision, branch, and dirty flag safely without failing outside Git repos."""
    revision = _safe_git("rev-parse", "HEAD", cwd=cwd)
    branch = _safe_git("branch", "--show-current", cwd=cwd)
    status = _safe_git("status", "--porcelain", cwd=cwd)
    return {
        "git_revision": revision,
        "branch": branch if branch else None,
        "git_dirty": bool(status) if status is not None else None,
    }


def capture_installed_packages() -> dict[str, str]:
    """Return versions only for installed packages from the explicit allowlist."""
    versions: dict[str, str] = {}
    for pkg in _RELEVANT_PACKAGES:
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def capture_gpu_inventory() -> list[dict[str, str]] | None:
    """Safely query local NVIDIA GPUs if nvidia-smi is present; returns None otherwise."""
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    gpus: list[dict[str, str]] = []
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            gpus.append(
                {
                    "index": parts[0],
                    "name": parts[1],
                    "memory_total_mib": parts[2],
                    "driver_version": parts[3],
                }
            )
    return gpus


def _sanitize_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    authority = f"{host}:{parsed.port}" if parsed.port else host
    return f"{parsed.scheme}://{authority}"


def build_environment_metadata(
    *,
    run_id: str,
    config: ExperimentConfig,
    started_at: str,
    ended_at: str | None = None,
    observed_workers: list[dict[str, Any]] | None = None,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Capture reproducibility metadata without ever reading or dumping environment secrets."""
    git_info = capture_git_info(cwd=repo_root)
    safe_workers = [
        {
            "worker_id": row.get("worker_id"),
            "endpoint": _sanitize_url(row["endpoint"])
            if isinstance(row.get("endpoint"), str) and "://" in row["endpoint"]
            else row.get("endpoint"),
            "model": row.get("model"),
            "status": row.get("status"),
            "active_requests": row.get("active_requests"),
        }
        for row in (observed_workers or [])
        if isinstance(row, dict)
    ]
    return {
        "schema_version": 1,
        "run_id": run_id,
        "started_at": started_at,
        "ended_at": ended_at,
        "git_revision": git_info["git_revision"],
        "git_dirty": git_info["git_dirty"],
        "branch": git_info["branch"],
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "platform": platform.platform(),
        },
        "packages": capture_installed_packages(),
        "gpu": capture_gpu_inventory(),
        "model_id": config.model.id,
        "routing_policy": config.policy.name,
        "expected_worker_count": config.topology.worker_count,
        "gateway_url": _sanitize_url(config.runner.gateway_url),
        "observed_workers": safe_workers,
    }


def write_text_atomic(path: Path, text: str) -> None:
    """Atomically write UTF-8 text with portable LF newlines."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.parent / f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
    try:
        with tmp_path.open("w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    """Atomically write formatted JSON with trailing newline."""
    write_text_atomic(path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


class ArtifactWriter:
    """Manages one canonical Week 7 experiment run directory."""

    def __init__(self, run_dir: Path, run_id: str) -> None:
        self.run_dir = run_dir
        self.run_id = run_id
        self.manifest_path = run_dir / "manifest.json"
        self.environment_path = run_dir / "environment.json"
        self.traces_path = run_dir / "traces.jsonl"
        self.metrics_path = run_dir / "metrics.prom"
        self.summary_path = run_dir / "summary.json"

    @classmethod
    def create(
        cls,
        *,
        runs_root: Path | None = None,
        config: ExperimentConfig,
        run_id: str | None = None,
        repo_root: Path = REPO_ROOT,
    ) -> ArtifactWriter:
        root = (runs_root or DEFAULT_RUNS_ROOT).resolve()
        root.mkdir(parents=True, exist_ok=True)
        resolved_run_id = run_id
        if resolved_run_id is None:
            git_rev = capture_git_info(cwd=repo_root).get("git_revision")
            resolved_run_id = generate_run_id(config.name, git_rev)
        run_dir = root / resolved_run_id
        run_dir.mkdir(parents=True, exist_ok=False)
        writer = cls(run_dir=run_dir, run_id=resolved_run_id)
        # Initialize empty traces.jsonl so readers always see a valid JSONL file even on early fail.
        writer.traces_path.write_text("", encoding="utf-8", newline="\n")
        return writer

    def write_manifest(self, config: ExperimentConfig) -> None:
        write_json_atomic(self.manifest_path, config.to_normalized_dict())

    def write_environment(self, metadata: dict[str, Any]) -> None:
        write_json_atomic(self.environment_path, metadata)

    def append_trace(self, trace: dict[str, Any]) -> None:
        line = json.dumps(trace, ensure_ascii=False, separators=(",", ":")) + "\n"
        with self.traces_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(line)
            handle.flush()

    def write_metrics(self, metrics_text: str) -> None:
        write_text_atomic(self.metrics_path, metrics_text)

    def write_summary(self, summary: dict[str, Any]) -> None:
        write_json_atomic(self.summary_path, summary)
