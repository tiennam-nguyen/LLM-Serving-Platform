"""Integration tests for the Week 7 HTTP-only experiment runner against real ASGI Gateway."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from llm_serving_platform.benchmark.config import load_manifest
from llm_serving_platform.benchmark.runner import ExperimentRunner
from llm_serving_platform.benchmark.summary import compute_summary_from_traces_file
from tests.week6_workers import mock_gateway

SECRET_PROMPT_SENTINEL = "UNIQUE_SECRET_PROMPT_DO_NOT_LEAK_4412"


def _create_manifest_and_workload(
    tmp_path: Path,
    *,
    policy: str = "least_loaded",
    model: str = "test-model",
    worker_count: int = 2,
    request_count: int = 6,
    warmup_count: int = 2,
    concurrency: int = 2,
    workload_entries: int = 8,
) -> Path:
    workload_path = tmp_path / "workload.jsonl"
    lines = [
        json.dumps(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": f"{SECRET_PROMPT_SENTINEL} question {idx}",
                    }
                ]
            }
        )
        for idx in range(workload_entries)
    ]
    workload_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "schema_version": 1,
        "name": f"w7-test-{policy}",
        "model": {"id": model},
        "topology": {"worker_count": worker_count},
        "policy": {"name": policy},
        "workload": {"path": str(workload_path), "request_count": request_count},
        "seed": 42,
        "warmup": {"request_count": warmup_count},
        "generation": {"temperature": 0, "max_tokens": 16},
        "runner": {
            "concurrency": concurrency,
            "gateway_url": "http://mock-gateway",
            "request_timeout_s": 30,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest_path


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["least_loaded", "round_robin"])
async def test_runner_e2e_warmup_excluded_artifacts_complete_and_no_prompt_leak(
    tmp_path: Path, policy: str
):
    manifest_path = _create_manifest_and_workload(
        tmp_path,
        policy=policy,
        request_count=6,
        warmup_count=2,
        concurrency=2,
    )
    runs_root = tmp_path / "runs"

    async with mock_gateway(policy, "test-model") as (_, client, controls):
        runner = ExperimentRunner(runs_root=runs_root, client=client)
        result = await runner.run_from_manifest(manifest_path, base_dir=tmp_path)

        assert result.succeeded is True
        assert result.run_dir.is_dir()

        # Verify exact canonical artifact layout
        expected_files = {
            "manifest.json",
            "environment.json",
            "traces.jsonl",
            "metrics.prom",
            "summary.json",
        }
        assert {p.name for p in result.run_dir.iterdir()} == expected_files

        # Verify warm-up ran (2 warmup + 6 measured = 8 total worker calls)
        total_calls = sum(controls.calls.values())
        assert total_calls == 8

        # Verify traces.jsonl has ONLY the 6 measured traces, all unique IDs, all terminal
        trace_lines = (result.run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(trace_lines) == 6
        traces = [json.loads(line) for line in trace_lines]
        request_ids = [t["request_id"] for t in traces]
        assert len(set(request_ids)) == 6
        assert all("-warmup-" not in rid for rid in request_ids)
        assert all(t["status"] == "completed" for t in traces)
        assert all(t["routing"]["policy"] == policy for t in traces)

        # Verify summary.json strictly matches recomputing from traces.jsonl
        summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
        recomputed = compute_summary_from_traces_file(
            result.run_dir / "traces.jsonl",
            run_id=result.run_id,
            run_status="completed",
            planned_requests=6,
            expected_policy=policy,
        )
        assert summary == recomputed
        assert summary["run_status"] == "completed"
        assert summary["policy"] == policy
        assert summary["requests"] == {
            "planned": 6,
            "observed": 6,
            "completed": 6,
            "error": 0,
            "cancelled": 0,
        }
        assert sum(summary["worker_distribution"].values()) == 6

        # Verify metrics.prom is valid Prometheus exposition format
        metrics_text = (result.run_dir / "metrics.prom").read_text(encoding="utf-8")
        families = {f.name: f for f in text_string_to_metric_families(metrics_text)}
        assert "llm_gateway_requests" in families
        assert "llm_worker_active_requests" in families

        # Verify active load returned to 0
        workers = (await client.get("/control/v1/workers")).json()
        assert all(w["active_requests"] == 0 for w in workers)

        # Verify ZERO prompt leakage across ALL generated observability artifacts
        for filename in expected_files:
            content = (result.run_dir / filename).read_text(encoding="utf-8")
            assert SECRET_PROMPT_SENTINEL not in content


@pytest.mark.asyncio
async def test_runner_records_sse_error_under_http_200_as_experiment_data(tmp_path: Path):
    manifest_path = _create_manifest_and_workload(
        tmp_path,
        policy="least_loaded",
        request_count=4,
        warmup_count=0,
        concurrency=1,
    )
    runs_root = tmp_path / "runs"
    run_id = "20261122T000000Z-sse-error-test"

    async with mock_gateway("least_loaded", "test-model") as (_, client, controls):
        # Fail measured request index 1 before first token -> HTTP 200 with SSE error chunk
        controls.failures[f"{run_id}-req-00001"] = "pre"

        runner = ExperimentRunner(runs_root=runs_root, client=client)
        result = await runner.run_from_manifest(manifest_path, base_dir=tmp_path, run_id=run_id)

        assert result.succeeded is True
        summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["run_status"] == "completed"
        assert summary["requests"] == {
            "planned": 4,
            "observed": 4,
            "completed": 3,
            "error": 1,
            "cancelled": 0,
        }
        # Missing TTFT on the failed request must be counted explicitly
        assert summary["latency_sample_counts"]["ttft"] == {"present": 3, "missing": 1}


@pytest.mark.asyncio
async def test_runner_records_gateway_503_mid_run_without_crashing(tmp_path: Path):
    manifest_path = _create_manifest_and_workload(
        tmp_path,
        policy="least_loaded",
        request_count=3,
        warmup_count=0,
        concurrency=1,
    )
    runs_root = tmp_path / "runs"

    async with mock_gateway("least_loaded", "test-model") as (app, client, controls):
        # Make all workers unhealthy AFTER preflight by wrapping transport
        original_send = client.send
        chat_posts = 0

        async def drop_workers_on_second_post(request: httpx.Request, **kwargs):
            nonlocal chat_posts
            if request.url.path == "/v1/chat/completions":
                chat_posts += 1
                if chat_posts == 2:
                    for w in controls.workers:
                        w.healthy = False
                    await app.state.health_monitor.check_once()
            return await original_send(request, **kwargs)

        client.send = drop_workers_on_second_post  # type: ignore[method-assign]
        runner = ExperimentRunner(runs_root=runs_root, client=client)
        result = await runner.run_from_manifest(manifest_path, base_dir=tmp_path)

        assert result.succeeded is True
        summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["requests"]["planned"] == 3
        assert summary["requests"]["observed"] == 3
        assert summary["requests"]["completed"] == 1
        assert summary["requests"]["error"] == 2


@pytest.mark.asyncio
async def test_runner_polls_until_trace_becomes_terminal(tmp_path: Path):
    manifest_path = _create_manifest_and_workload(
        tmp_path,
        policy="least_loaded",
        request_count=1,
        warmup_count=0,
        concurrency=1,
    )
    runs_root = tmp_path / "runs"

    async with mock_gateway("least_loaded", "test-model") as (_, client, _):
        original_get = client.get
        trace_polls = 0

        async def delayed_terminal_get(url, *args, **kwargs):
            nonlocal trace_polls
            resp = await original_get(url, *args, **kwargs)
            if isinstance(url, str) and url.startswith("/v1/traces/"):
                trace_polls += 1
                if trace_polls < 3:
                    data = resp.json()
                    data["status"] = "in_flight"
                    return httpx.Response(200, json=data, request=resp.request)
            return resp

        client.get = delayed_terminal_get  # type: ignore[method-assign]
        runner = ExperimentRunner(
            runs_root=runs_root,
            client=client,
            trace_poll_interval_s=0.005,
            trace_poll_timeout_s=2.0,
        )
        result = await runner.run_from_manifest(manifest_path, base_dir=tmp_path)

        assert result.succeeded is True
        assert trace_polls >= 3
        traces = [
            json.loads(line)
            for line in (result.run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(traces) == 1
        assert traces[0]["status"] == "completed"


@pytest.mark.asyncio
async def test_runner_fatal_network_error_preserves_partial_evidence(tmp_path: Path):
    manifest_path = _create_manifest_and_workload(
        tmp_path,
        policy="least_loaded",
        request_count=5,
        warmup_count=0,
        concurrency=1,
    )
    runs_root = tmp_path / "runs"

    async with mock_gateway("least_loaded", "test-model") as (_, client, _):
        original_send = client.send
        chat_posts = 0

        async def fail_network_after_two(request: httpx.Request, **kwargs):
            nonlocal chat_posts
            if request.url.path == "/v1/chat/completions":
                chat_posts += 1
                if chat_posts >= 3:
                    raise httpx.ConnectError("Gateway connection dropped", request=request)
            return await original_send(request, **kwargs)

        client.send = fail_network_after_two  # type: ignore[method-assign]
        runner = ExperimentRunner(runs_root=runs_root, client=client)
        result = await runner.run_from_manifest(manifest_path, base_dir=tmp_path)

        assert result.succeeded is False
        assert result.run_dir.is_dir()
        assert (result.run_dir / "manifest.json").is_file()
        assert (result.run_dir / "environment.json").is_file()
        assert (result.run_dir / "traces.jsonl").is_file()
        assert (result.run_dir / "summary.json").is_file()

        lines = (result.run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2

        summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["run_status"] == "failed"
        assert summary["requests"]["planned"] == 5
        assert summary["requests"]["observed"] == 2
        assert summary["failure"]["type"] == "gateway_unreachable"
        # Raw exception text must not leak into failure metadata
        assert "Gateway connection dropped" not in json.dumps(summary)


@pytest.mark.asyncio
async def test_runner_detects_policy_mismatch_from_traces(tmp_path: Path):
    # Manifest declares least_loaded, but Gateway actually runs round_robin
    manifest_path = _create_manifest_and_workload(
        tmp_path,
        policy="least_loaded",
        request_count=2,
        warmup_count=0,
        concurrency=1,
    )
    runs_root = tmp_path / "runs"

    async with mock_gateway("round_robin", "test-model") as (_, client, _):
        runner = ExperimentRunner(runs_root=runs_root, client=client)
        result = await runner.run_from_manifest(manifest_path, base_dir=tmp_path)

        assert result.succeeded is False
        summary = json.loads((result.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["run_status"] == "failed"
        assert summary["failure"]["type"] == "policy_mismatch"
        assert summary["observed_policies"] == ["round_robin"]


@pytest.mark.asyncio
async def test_runner_preflight_rejects_topology_and_model_mismatch(tmp_path: Path):
    # Topology mismatch: manifest expects 3 workers, Gateway has 2
    manifest_topo = _create_manifest_and_workload(tmp_path, worker_count=3, request_count=2)
    async with mock_gateway("least_loaded", "test-model") as (_, client, _):
        runner = ExperimentRunner(runs_root=tmp_path / "runs-topo", client=client)
        res_topo = await runner.run_from_manifest(manifest_topo, base_dir=tmp_path)
        assert res_topo.succeeded is False
        summary = json.loads((res_topo.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["failure"]["type"] == "topology_mismatch"

    # Model mismatch: manifest expects other-model, Gateway serves test-model
    manifest_model = _create_manifest_and_workload(tmp_path, model="other-model", request_count=2)
    async with mock_gateway("least_loaded", "test-model") as (_, client, _):
        runner = ExperimentRunner(runs_root=tmp_path / "runs-model", client=client)
        res_model = await runner.run_from_manifest(manifest_model, base_dir=tmp_path)
        assert res_model.succeeded is False
        summary = json.loads((res_model.run_dir / "summary.json").read_text(encoding="utf-8"))
        assert summary["failure"]["type"] == "model_mismatch"


def test_benchmark_package_does_not_import_gateway_internals():
    forbidden = {
        "RoundRobinRouter",
        "LeastLoadedRouter",
        "WorkerRegistry",
        "InMemoryTraceStore",
    }
    benchmark_dir = Path("src/llm_serving_platform/benchmark")
    for py_file in benchmark_dir.glob("*.py"):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    assert alias.name not in forbidden, f"{py_file} imports {alias.name}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name not in forbidden, f"{py_file} imports {alias.name}"


def test_benchmark_cli_returns_nonzero_for_invalid_manifest(tmp_path: Path):
    bad_manifest = tmp_path / "bad.json"
    bad_manifest.write_text('{"schema_version": 99}', encoding="utf-8")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_serving_platform.benchmark",
            "run",
            "--manifest",
            str(bad_manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0


@pytest.mark.parametrize(
    "manifest_rel",
    [
        "experiments/manifests/w7_smoke_least_loaded.json",
        "experiments/manifests/w7_t4_round_robin.json",
        "experiments/manifests/w7_t4_least_loaded.json",
    ],
)
def test_repo_manifests_and_workloads_are_valid(manifest_rel: str):
    from llm_serving_platform.benchmark.workload import load_workload

    cfg, workload_path = load_manifest(manifest_rel)
    assert cfg.schema_version == 1
    assert workload_path.is_file()
    entries = load_workload(workload_path, min_entries=cfg.workload.request_count)
    assert len(entries) >= cfg.workload.request_count
