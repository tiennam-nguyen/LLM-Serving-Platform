"""Unit tests for deterministic Week 7 summary computation from raw traces.jsonl."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_serving_platform.benchmark.summary import (
    compute_percentile,
    compute_summary_from_traces_file,
)


def _trace_record(
    request_id: str,
    *,
    status: str = "completed",
    worker_id: str = "worker-mock-0",
    policy: str | None = "least_loaded",
    routing_ms: float | None = 0.5,
    ttft_ms: float | None = 100.0,
    e2e_ms: float | None = 500.0,
) -> dict:
    return {
        "request_id": request_id,
        "model": "test-model",
        "worker_id": worker_id,
        "status": status,
        "token_chunks": 4 if status == "completed" else 0,
        "completion_tokens": None,
        "error": None if status == "completed" else "worker_unavailable",
        "routing": {
            "policy": policy,
            "candidate_loads": {},
            "selected_worker_id": worker_id if worker_id != "none" else None,
            "selected_load_before": 0,
            "selected_load_after": 1,
            "reason": policy,
        },
        "milestones": {
            "arrival": "2026-11-22T00:00:00+00:00",
            "routing_start": "2026-11-22T00:00:00+00:00",
            "routing_end": "2026-11-22T00:00:00+00:00",
            "worker_start": "2026-11-22T00:00:00+00:00",
            "first_token": "2026-11-22T00:00:00+00:00" if ttft_ms is not None else None,
            "last_token": "2026-11-22T00:00:01+00:00",
        },
        "metrics_ms": {
            "routing_ms": routing_ms,
            "ttft_ms": ttft_ms,
            "tpot_ms": None,
            "inter_content_chunk_ms": None,
            "e2e_ms": e2e_ms,
        },
    }


def _write_traces(path: Path, traces: list[dict]) -> Path:
    content = "".join(json.dumps(t) + "\n" for t in traces)
    path.write_text(content, encoding="utf-8")
    return path


def test_percentile_zero_one_and_many_samples():
    assert compute_percentile([], 50) is None
    assert compute_percentile([], 95) is None
    assert compute_percentile([42.5], 50) == 42.5
    assert compute_percentile([42.5], 95) == 42.5

    # Linear interpolation (R-7): for [10, 20, 30, 40, 50]
    # p50 -> index 0.5 * 4 = 2.0 -> 30.0
    # p95 -> index 0.95 * 4 = 3.8 -> 40 + 0.8 * (50 - 40) = 48.0
    values = [50.0, 10.0, 40.0, 20.0, 30.0]
    assert compute_percentile(values, 50) == pytest.approx(30.0)
    assert compute_percentile(values, 95) == pytest.approx(48.0)


def test_summary_handles_empty_traces_without_error(tmp_path: Path):
    traces_file = _write_traces(tmp_path / "traces.jsonl", [])
    summary = compute_summary_from_traces_file(
        traces_file,
        run_id="run-empty",
        run_status="failed",
        planned_requests=10,
        expected_policy="least_loaded",
        failure={"type": "gateway_unreachable"},
    )
    assert summary["run_id"] == "run-empty"
    assert summary["run_status"] == "failed"
    assert summary["failure"] == {"type": "gateway_unreachable"}
    assert summary["requests"] == {
        "planned": 10,
        "observed": 0,
        "completed": 0,
        "error": 0,
        "cancelled": 0,
    }
    assert all(v is None for v in summary["latency_ms"].values())
    assert summary["worker_distribution"] == {}


def test_summary_from_raw_jsonl_handles_missing_ttft_mixed_statuses_and_workers(tmp_path: Path):
    traces = [
        _trace_record(
            "r1",
            status="completed",
            worker_id="worker-vllm-0",
            routing_ms=0.2,
            ttft_ms=100.0,
            e2e_ms=1000.0,
        ),
        _trace_record(
            "r2",
            status="completed",
            worker_id="worker-vllm-1",
            routing_ms=0.4,
            ttft_ms=200.0,
            e2e_ms=1200.0,
        ),
        _trace_record(
            "r3",
            status="completed",
            worker_id="worker-vllm-1",
            routing_ms=0.6,
            ttft_ms=300.0,
            e2e_ms=1400.0,
        ),
        _trace_record(
            "r4",
            status="error",
            worker_id="none",
            routing_ms=0.8,
            ttft_ms=None,
            e2e_ms=50.0,
        ),
        _trace_record(
            "r5",
            status="cancelled",
            worker_id="worker-vllm-0",
            routing_ms=1.0,
            ttft_ms=None,
            e2e_ms=80.0,
        ),
    ]
    traces_file = _write_traces(tmp_path / "traces.jsonl", traces)
    summary = compute_summary_from_traces_file(
        traces_file,
        run_id="run-mixed",
        run_status="completed",
        planned_requests=5,
        expected_policy="least_loaded",
    )

    assert summary["policy"] == "least_loaded"
    assert summary["observed_policies"] == ["least_loaded"]
    assert summary["requests"] == {
        "planned": 5,
        "observed": 5,
        "completed": 3,
        "error": 1,
        "cancelled": 1,
    }
    # Missing TTFT values must NOT be treated as 0: only [100, 200, 300] count!
    assert summary["latency_ms"]["ttft_p50"] == pytest.approx(200.0)
    assert summary["latency_ms"]["ttft_p95"] == pytest.approx(290.0)
    assert summary["latency_sample_counts"]["ttft"] == {"present": 3, "missing": 2}
    assert summary["latency_sample_counts"]["routing"] == {"present": 5, "missing": 0}
    assert summary["latency_sample_counts"]["e2e"] == {"present": 5, "missing": 0}
    assert summary["worker_distribution"] == {
        "worker-vllm-0": 2,
        "worker-vllm-1": 2,
    }
    # Re-reading raw traces.jsonl is deterministic
    summary_again = compute_summary_from_traces_file(
        traces_file,
        run_id="run-mixed",
        run_status="completed",
        planned_requests=5,
        expected_policy="least_loaded",
    )
    assert summary == summary_again
