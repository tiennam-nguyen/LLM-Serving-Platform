"""Unit tests for Week 7 experiment artifact writer and environment capture."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from llm_serving_platform.benchmark.artifacts import (
    ArtifactWriter,
    build_environment_metadata,
    generate_run_id,
)
from llm_serving_platform.benchmark.config import ExperimentConfig


def _sample_config() -> ExperimentConfig:
    return ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "name": "w7-smoke-least-loaded",
            "model": {"id": "Qwen/Qwen2.5-3B-Instruct"},
            "topology": {"worker_count": 2},
            "policy": {"name": "least_loaded"},
            "workload": {"path": "benchmarks/workloads/w7_smoke.jsonl", "request_count": 5},
            "seed": 42,
            "warmup": {"request_count": 1},
            "generation": {"temperature": 0, "max_tokens": 32},
            "runner": {
                "concurrency": 2,
                "gateway_url": "http://127.0.0.1:8080",
                "request_timeout_s": 60,
            },
        }
    )


def test_generate_run_id_is_human_readable_and_collision_resistant():
    ts = datetime(2026, 11, 22, 4, 15, 12, tzinfo=timezone.utc)
    id_a = generate_run_id("w7-smoke-ll", "abcdef1234567890", started_at=ts)
    id_b = generate_run_id("w7-smoke-ll", "abcdef1234567890", started_at=ts)
    assert id_a.startswith("20261122T041512Z-w7-smoke-ll-abcdef1")
    assert id_a != id_b


def test_artifact_writer_creates_canonical_files_atomically(tmp_path: Path):
    cfg = _sample_config()
    writer = ArtifactWriter.create(runs_root=tmp_path, config=cfg, run_id="20261122T000000Z-test")
    writer.write_manifest(cfg)
    writer.write_environment({"run_id": writer.run_id, "git_revision": "abc1234"})
    writer.append_trace({"request_id": "req-1", "status": "completed"})
    writer.append_trace({"request_id": "req-2", "status": "error"})
    writer.write_metrics("llm_gateway_requests_total{status=\"completed\"} 1.0\n")
    writer.write_summary({"run_id": writer.run_id, "run_status": "completed"})

    expected_files = {
        "manifest.json",
        "environment.json",
        "traces.jsonl",
        "metrics.prom",
        "summary.json",
    }
    actual_files = {p.name for p in writer.run_dir.iterdir()}
    assert actual_files == expected_files  # No leftover .tmp files

    assert json.loads(writer.manifest_path.read_text(encoding="utf-8"))["name"] == cfg.name
    lines = writer.traces_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["request_id"] == "req-1"
    assert json.loads(lines[1])["request_id"] == "req-2"
    assert "llm_gateway_requests_total" in writer.metrics_path.read_text(encoding="utf-8")


def test_environment_metadata_never_dumps_secrets(monkeypatch):
    monkeypatch.setenv("KAGGLE_KEY", "SUPER_SECRET_KAGGLE_TOKEN_999")
    monkeypatch.setenv("HF_TOKEN", "hf_SUPER_SECRET_TOKEN_888")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_SUPER_SECRET_GITHUB_777")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-SUPER_SECRET_API_666")

    cfg = _sample_config()
    env_meta = build_environment_metadata(
        run_id="run-safe-1",
        config=cfg,
        started_at="2026-11-22T04:15:12+00:00",
        ended_at="2026-11-22T04:15:30+00:00",
        observed_workers=[
            {
                "worker_id": "worker-mock-0",
                "endpoint": "mock://worker-mock-0",
                "model": cfg.model.id,
                "status": "HEALTHY",
                "active_requests": 0,
            }
        ],
    )
    serialized = json.dumps(env_meta)
    for secret in (
        "SUPER_SECRET_KAGGLE_TOKEN_999",
        "hf_SUPER_SECRET_TOKEN_888",
        "ghp_SUPER_SECRET_GITHUB_777",
        "sk-SUPER_SECRET_API_666",
    ):
        assert secret not in serialized
    assert env_meta["run_id"] == "run-safe-1"
    assert env_meta["python_version"]
    assert env_meta["model_id"] == cfg.model.id
    assert env_meta["routing_policy"] == "least_loaded"
    assert env_meta["expected_worker_count"] == 2
    assert "fastapi" in env_meta["packages"]
    assert "httpx" in env_meta["packages"]
    assert "pydantic" in env_meta["packages"]
    assert "prometheus-client" in env_meta["packages"]
