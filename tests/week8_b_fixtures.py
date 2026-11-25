"""Small, explicitly synthetic W8 fixtures; these are not experimental evidence."""

import hashlib
import json
from pathlib import Path

from llm_serving_platform.benchmark.campaign import generate_workload


def manifest_v2(
    workload_path: Path, *, policy="round_robin", profile="steady", count=4, window=200
):
    return {
        "schema_version": 2,
        "name": "w8-test",
        "model": {"id": "test-model"},
        "topology": {"worker_count": 2},
        "policy": {"name": policy},
        "workload": {"path": str(workload_path), "profile": profile, "request_count": count},
        "seed": 42,
        "warmup": {"request_count": 0},
        "generation": {"temperature": 0, "max_tokens": 8, "top_p": 1},
        "runner": {
            "gateway_url": "http://mock-gateway",
            "request_timeout_s": 5,
            "max_in_flight": 64,
            "max_dispatch_lag_ms": 1000,
        },
        "load": {"mode": "open_loop", "measurement_window_ms": window},
        "comparison": {"id": "synthetic-pair", "load_level": "pilot", "repetition": 1},
    }


def write_manifest(tmp_path: Path, *, policy="round_robin", profile="steady", count=4, window=200):
    workload = tmp_path / f"{profile}.jsonl"
    rows = generate_workload(profile, request_count=count, measurement_window_ms=window, seed=42)
    workload.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    raw = manifest_v2(workload, policy=policy, profile=profile, count=count, window=window)
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return path, raw, rows


def raw_run(root: Path, *, policy="round_robin", repetition=1) -> Path:
    """Independent two-request raw fixture with hand-checkable expected statistics."""
    run_id = f"synthetic-{policy}-r{repetition}"
    path = root / run_id
    path.mkdir(parents=True)
    manifest = manifest_v2(root / "input.jsonl", policy=policy, count=2, window=1000)
    manifest["comparison"]["repetition"] = repetition
    manifest["name"] = run_id
    schedule = [
        {
            "sequence": i,
            "entry_id": f"q-{i}",
            "workload_index": i,
            "scheduled_offset_ms": float(i * 500),
        }
        for i in range(2)
    ]
    schedule_text = "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in schedule)
    (path / "schedule.jsonl").write_text(schedule_text, encoding="utf-8", newline="\n")
    requests, traces = [], []
    for i, end in enumerate((200.0, 1500.0)):
        rid = f"{run_id}-req-{i:05d}"
        requests.append(
            {
                "sequence": i,
                "request_id": rid,
                "dispatch_status": "sent",
                "not_dispatched_reason": None,
                "dispatch_offset_ms": float(i * 500),
                "dispatch_lag_ms": 0.0,
                "client_end_offset_ms": end,
                "http_status": 200,
                "client_status": "completed",
                "client_error_type": None,
                "trace_observed": True,
            }
        )
        traces.append(
            {
                "request_id": rid,
                "model": "test-model",
                "status": "completed",
                "worker_id": f"worker-mock-{i}",
                "routing": {"policy": policy},
                "metrics_ms": {
                    "routing_ms": float(i + 1),
                    "ttft_ms": float(10 + i * 20),
                    "e2e_ms": float(100 + i * 900),
                },
            }
        )
    for name, rows in (("requests.jsonl", requests), ("traces.jsonl", traces)):
        (path / name).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    env = {
        "schema_version": 2,
        "run_id": run_id,
        "run_status": "completed",
        "git_revision": "a" * 40,
        "git_dirty": False,
        "packages": {"test-fixture": "1"},
        "gpu": None,
        "started_at": "2000-01-01T00:00:00Z",
        "ended_at": "2000-01-01T00:00:02Z",
        "measured_started_at": "2000-01-01T00:00:00Z",
        "measurement_window_ms": 1000,
        "workload_sha256": "b" * 64,
        "schedule_sha256": hashlib.sha256(schedule_text.encode()).hexdigest(),
        "final_workers": [
            {"worker_id": f"worker-mock-{i}", "active_requests": 0} for i in range(2)
        ],
    }
    for name, value in (("manifest.json", manifest), ("environment.json", env)):
        (path / name).write_text(json.dumps(value), encoding="utf-8")
    for name in ("metrics.before.prom", "metrics.prom"):
        (path / name).write_text(
            'llm_worker_active_requests{worker_id="worker-mock-0"} 0\n', encoding="utf-8"
        )
    return path
