"""Unit tests for Week 8 pilot-only v2 schema, metrics recomputation, and load selection."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from infra.kaggle.week8_open_loop_pilot.pilot_metrics import (  # noqa: E402
    compute_pilot_metrics_from_dir,
    select_common_load_points,
    validate_pilot_run_dir,
)
from infra.kaggle.week8_open_loop_pilot.pilot_schema import (  # noqa: E402
    PilotManifestError,
    PilotManifestV2,
    generate_steady_pilot_workload,
    load_pilot_workload_v2,
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    content = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    path.write_text(content, encoding="utf-8", newline="\n")


def _sample_manifest_dict(workload_path: str, policy: str = "round_robin") -> dict[str, Any]:
    return {
        "schema_version": 2,
        "name": f"w8-pilot-{policy}",
        "model": {"id": "Qwen/Qwen2.5-3B-Instruct"},
        "topology": {"worker_count": 2},
        "policy": {"name": policy},
        "workload": {
            "path": workload_path,
            "profile": "steady",
            "request_count": 4,
        },
        "seed": 42,
        "warmup": {"request_count": 2},
        "generation": {"temperature": 0, "max_tokens": 64, "top_p": 1},
        "runner": {
            "gateway_url": "http://127.0.0.1:8080",
            "request_timeout_s": 120,
            "max_in_flight": 64,
            "max_dispatch_lag_ms": 250,
        },
        "load": {"mode": "open_loop", "measurement_window_ms": 2000},
        "comparison": {"id": "pilot-2rps", "load_level": "pilot", "repetition": 1},
    }


def test_pilot_manifest_validation_rejects_concurrency_unknowns_and_bad_policy(tmp_path: Path):
    base = _sample_manifest_dict("workload.jsonl")
    cfg = PilotManifestV2.model_validate(base)
    assert cfg.schema_version == 2
    assert cfg.policy.name == "round_robin"

    bad_conc = json.loads(json.dumps(base))
    bad_conc["runner"]["concurrency"] = 2
    with pytest.raises(Exception):
        PilotManifestV2.model_validate(bad_conc)

    bad_bool = json.loads(json.dumps(base))
    bad_bool["runner"]["max_in_flight"] = True
    with pytest.raises(Exception):
        PilotManifestV2.model_validate(bad_bool)

    bad_pol = json.loads(json.dumps(base))
    bad_pol["policy"]["name"] = "single"
    with pytest.raises(Exception):
        PilotManifestV2.model_validate(bad_pol)


def test_generate_and_load_steady_pilot_workload_are_deterministic_and_unique(tmp_path: Path):
    path_a, sched_a, sha_a = generate_steady_pilot_workload(
        tmp_path / "w_a.jsonl",
        request_count=10,
        measurement_window_ms=5000.0,
        seed=42,
    )
    path_b, sched_b, sha_b = generate_steady_pilot_workload(
        tmp_path / "w_b.jsonl",
        request_count=10,
        measurement_window_ms=5000.0,
        seed=42,
    )
    assert sha_a == sha_b
    assert len(sched_a) == 10
    assert [s.sequence for s in sched_a] == list(range(10))
    assert [s.scheduled_offset_ms for s in sched_a] == pytest.approx(
        [i * 500.0 for i in range(10)]
    )
    prompts = [s.entry.messages[0]["content"] for s in sched_a]
    assert len(set(prompts)) == 10  # Avoid identical prefix/prompt cache distortion

    reloaded = load_pilot_workload_v2(
        path_a, expected_count=10, measurement_window_ms=5000.0
    )
    assert [r.entry.entry_id for r in reloaded] == [s.entry.entry_id for s in sched_b]

    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(
        json.dumps(
            {
                "id": "q0",
                "sequence": 0,
                "scheduled_offset_ms": 2000.0,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(PilotManifestError):
        load_pilot_workload_v2(bad_file, expected_count=1, measurement_window_ms=2000.0)


def test_compute_pilot_metrics_exact_denominators_r7_and_ooo_join(tmp_path: Path):
    run_dir = tmp_path / "run-1"
    run_dir.mkdir()
    w_path, _, w_sha = generate_steady_pilot_workload(
        tmp_path / "w.jsonl", request_count=4, measurement_window_ms=2000.0, seed=42
    )
    manifest = _sample_manifest_dict(str(w_path), policy="round_robin")
    _write_json(run_dir / "manifest.json", manifest)

    schedule_rows = [
        {
            "sequence": i,
            "entry_id": f"pilot-{i:04d}",
            "workload_index": i,
            "scheduled_offset_ms": i * 500.0,
        }
        for i in range(4)
    ]
    _write_jsonl(run_dir / "schedule.jsonl", schedule_rows)
    sched_sha = hashlib.sha256((run_dir / "schedule.jsonl").read_bytes()).hexdigest()

    _write_json(
        run_dir / "environment.json",
        {
            "schema_version": 2,
            "run_id": "run-1",
            "started_at": "2026-11-25T00:00:00+00:00",
            "ended_at": "2026-11-25T00:00:03+00:00",
            "measured_started_at": "2026-11-25T00:00:01+00:00",
            "measurement_window_ms": 2000.0,
            "drain_end_offset_ms": 2500.0,
            "git_revision": "21e5de5d3403642328b647aa8dfda2cc58d7753d",
            "git_dirty": False,
            "workload_sha256": w_sha,
            "schedule_sha256": sched_sha,
            "final_workers": [
                {"worker_id": "worker-vllm-0", "active_requests": 0},
                {"worker_id": "worker-vllm-1", "active_requests": 0},
            ],
        },
    )
    # Out-of-order requests.jsonl and traces.jsonl:
    # seq 0: completed within window (client_end=800ms)
    # seq 1: completed AFTER window (client_end=2500ms -> counts in drain_rps, NOT window_rps)
    # seq 2: client completed (900ms), but gateway trace says "error" -> NOT successful throughput!
    # seq 3: client error (http 503), gateway trace "error"
    req_rows = [
        {
            "sequence": 2,
            "request_id": "run-1-req-00002",
            "dispatch_status": "sent",
            "not_dispatched_reason": None,
            "dispatch_offset_ms": 1002.0,
            "dispatch_lag_ms": 2.0,
            "client_end_offset_ms": 1900.0,
            "http_status": 200,
            "client_status": "completed",
            "client_error_type": None,
            "trace_observed": True,
        },
        {
            "sequence": 0,
            "request_id": "run-1-req-00000",
            "dispatch_status": "sent",
            "not_dispatched_reason": None,
            "dispatch_offset_ms": 0.5,
            "dispatch_lag_ms": 0.5,
            "client_end_offset_ms": 800.0,
            "http_status": 200,
            "client_status": "completed",
            "client_error_type": None,
            "trace_observed": True,
        },
        {
            "sequence": 3,
            "request_id": "run-1-req-00003",
            "dispatch_status": "sent",
            "not_dispatched_reason": None,
            "dispatch_offset_ms": 1504.0,
            "dispatch_lag_ms": 4.0,
            "client_end_offset_ms": 1520.0,
            "http_status": 503,
            "client_status": "error",
            "client_error_type": "http_error",
            "trace_observed": True,
        },
        {
            "sequence": 1,
            "request_id": "run-1-req-00001",
            "dispatch_status": "sent",
            "not_dispatched_reason": None,
            "dispatch_offset_ms": 501.0,
            "dispatch_lag_ms": 1.0,
            "client_end_offset_ms": 2500.0,
            "http_status": 200,
            "client_status": "completed",
            "client_error_type": None,
            "trace_observed": True,
        },
    ]
    _write_jsonl(run_dir / "requests.jsonl", req_rows)

    traces = [
        {
            "request_id": "run-1-req-00001",
            "worker_id": "worker-vllm-1",
            "status": "completed",
            "routing": {"policy": "round_robin"},
            "metrics_ms": {"routing_ms": 0.2, "ttft_ms": 80.0, "e2e_ms": 2000.0},
        },
        {
            "request_id": "run-1-req-00000",
            "worker_id": "worker-vllm-0",
            "status": "completed",
            "routing": {"policy": "round_robin"},
            "metrics_ms": {"routing_ms": 0.1, "ttft_ms": 40.0, "e2e_ms": 800.0},
        },
        {
            "request_id": "run-1-req-00003",
            "worker_id": "none",
            "status": "error",
            "routing": {"policy": "round_robin"},
            "metrics_ms": {"routing_ms": 0.4, "ttft_ms": None, "e2e_ms": 20.0},
        },
        {
            "request_id": "run-1-req-00002",
            "worker_id": "worker-vllm-0",
            "status": "error",
            "routing": {"policy": "round_robin"},
            "metrics_ms": {"routing_ms": 0.3, "ttft_ms": 50.0, "e2e_ms": 900.0},
        },
    ]
    _write_jsonl(run_dir / "traces.jsonl", traces)
    (run_dir / "metrics.before.prom").write_text("", encoding="utf-8")
    (run_dir / "metrics.prom").write_text(
        'llm_worker_active_requests{worker_id="worker-vllm-0"} 0.0\n'
        'llm_worker_active_requests{worker_id="worker-vllm-1"} 0.0\n',
        encoding="utf-8",
    )

    metrics = compute_pilot_metrics_from_dir(run_dir)
    assert metrics["artifact_role"] == "pilot_selection_evidence_not_member_b_summary"
    assert metrics["requests"] == {
        "planned": 4,
        "sent": 4,
        "not_dispatched": 0,
        "observed": 4,
        "completed": 2,
        "error": 2,
        "cancelled": 0,
        "missing_trace": 0,
        "both_completed": 2,
    }
    # T = 2.0s; drain = max(2.0, 2.5) = 2.5s
    assert metrics["rates"]["offered_rps"] == pytest.approx(2.0)
    assert metrics["rates"]["dispatched_rps"] == pytest.approx(2.0)
    assert metrics["rates"]["completed_window_rps"] == pytest.approx(1 / 2.0)
    assert metrics["rates"]["completed_drain_rps"] == pytest.approx(2 / 2.5)
    assert metrics["rates"]["noncompletion_rate"] == pytest.approx(1.0 - 2 / 4)
    assert metrics["rates"]["gateway_error_rate"] == pytest.approx(2 / 4)
    assert metrics["rates"]["gateway_cancelled_rate"] == pytest.approx(0.0)
    # Completed traces ONLY for TTFT/E2E: [40, 80] and [800, 2000]
    assert metrics["latency_ms"]["ttft_p50"] == pytest.approx(60.0)
    assert metrics["latency_ms"]["e2e_p50"] == pytest.approx(1400.0)
    assert metrics["latency_sample_counts"]["e2e"] == {"present": 2, "missing": 0}
    # Routing includes all 4 observed traces [0.1, 0.2, 0.3, 0.4]
    assert metrics["latency_sample_counts"]["routing"] == {"present": 4, "missing": 0}
    assert metrics["dispatch_lag_ms"]["max"] == pytest.approx(4.0)
    assert metrics["worker_distribution"] == {"worker-vllm-0": 2, "worker-vllm-1": 1}
    assert metrics["no_worker_count"] == 1
    assert metrics["backlog_at_window_end"] == 1

    # Write pilot_metrics.json and validate run directory passes
    _write_json(run_dir / "pilot_metrics.json", metrics)
    val = validate_pilot_run_dir(
        run_dir,
        expected_policy="round_robin",
        expected_planned=4,
        max_dispatch_lag_ms=250.0,
    )
    assert val["valid"] is True

    # Tampering pilot_metrics.json must fail validation
    bad_metrics = dict(metrics)
    bad_metrics["requests"] = {**metrics["requests"], "sent": 99}
    _write_json(run_dir / "pilot_metrics.json", bad_metrics)
    with pytest.raises(AssertionError):
        validate_pilot_run_dir(run_dir, expected_policy="round_robin", expected_planned=4)


def test_select_common_load_points_picks_knee_and_flags_no_knee():
    def _pair(
        rate: float,
        drain_rr: float,
        drain_ll: float,
        p95_rr: float,
        p95_ll: float,
        bl: int,
        *,
        ttft_p95: float = 120.0,
        window_ms: float = 20000.0,
        noncomp: float = 0.0,
    ):
        planned = int(rate * (window_ms / 1000.0))
        return {
            "offered_rps": rate,
            "measurement_window_ms": window_ms,
            "pair_valid": True,
            "schedule_sha256": f"sha-{rate}",
            "round_robin": {
                "run_id": f"rr-{rate}",
                "valid": True,
                "planned": planned,
                "completed_drain_rps": drain_rr,
                "noncompletion_rate": noncomp,
                "e2e_p95": p95_rr,
                "ttft_p95": ttft_p95,
                "backlog_at_window_end": bl,
                "dispatch_lag_max_ms": 5.0,
            },
            "least_loaded": {
                "run_id": f"ll-{rate}",
                "valid": True,
                "planned": planned,
                "completed_drain_rps": drain_ll,
                "noncompletion_rate": noncomp,
                "e2e_p95": p95_ll,
                "ttft_p95": ttft_p95,
                "backlog_at_window_end": bl,
                "dispatch_lag_max_ms": 5.0,
            },
        }

    # Finite-window drain tail (~2.35s on 20s window => 0.895*offered) WITHOUT tail/TTFT spike
    # must NOT falsely trigger saturation (mirrors Pilot Run #1 1..20 RPS behavior).
    finite_window_no_sat = [
        _pair(1.0, 0.96, 0.96, 1750.0, 2000.0, 2, ttft_p95=48.0),
        _pair(4.0, 3.69, 3.66, 2005.0, 2011.0, 8, ttft_p95=104.0),
        _pair(16.0, 14.43, 14.42, 2899.0, 2853.0, 44, ttft_p95=140.0),
        _pair(20.0, 17.89, 17.91, 3122.0, 3092.0, 60, ttft_p95=148.0),
    ]
    res_no_sat = select_common_load_points(finite_window_no_sat)
    assert res_no_sat["knee_reached"] is False
    assert res_no_sat["selected"] is None

    # Clear true server knee between 24.0 (stable) and 36.0 (saturated latency/TTFT/throughput)
    pairs = [
        _pair(1.0, 0.96, 0.96, 1750.0, 2000.0, 2, ttft_p95=48.0),
        _pair(4.0, 3.69, 3.66, 2005.0, 2011.0, 8, ttft_p95=104.0),
        _pair(12.0, 10.92, 10.92, 2374.0, 2365.0, 26, ttft_p95=116.0),
        _pair(24.0, 21.10, 21.20, 3300.0, 3250.0, 75, ttft_p95=165.0),
        _pair(36.0, 21.50, 22.00, 6500.0, 6200.0, 280, ttft_p95=850.0),
    ]
    sel = select_common_load_points(pairs)
    assert sel["knee_reached"] is True
    assert sel["selected"]["low"]["offered_rps"] == 1.0
    assert sel["selected"]["near_saturation"]["offered_rps"] == 24.0
    # Geometric midpoint between 1.0 and 24.0 is sqrt(24) ~= 4.90 -> closest tested is 4.0
    assert sel["selected"]["medium"]["offered_rps"] == 4.0

    # All stable -> knee_reached must be False (never fabricate load levels)
    no_knee = select_common_load_points(pairs[:3])
    assert no_knee["knee_reached"] is False
    assert no_knee["selected"] is None
