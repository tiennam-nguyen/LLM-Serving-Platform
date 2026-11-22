"""Full Week 7 CPU/mock end-to-end experiment pipeline verification gate & artifact validator."""

from __future__ import annotations

import asyncio
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx
from prometheus_client.parser import text_string_to_metric_families

REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (REPO_ROOT, REPO_ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from llm_serving_platform.benchmark.config import (  # noqa: E402
    ExperimentConfigError,
    load_manifest,
)
from llm_serving_platform.benchmark.runner import ExperimentRunner  # noqa: E402
from llm_serving_platform.benchmark.summary import (  # noqa: E402
    compute_summary_from_traces_file,
    read_traces_jsonl,
)

REQUIRED_ARTIFACT_FILES = (
    "manifest.json",
    "environment.json",
    "traces.jsonl",
    "metrics.prom",
    "summary.json",
)
TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def validate_run_artifacts(
    run_dir: Path | str,
    *,
    expected_policy: str | None = None,
    expected_planned: int | None = None,
    expected_status: str = "completed",
    expected_revision: str | None = None,
    forbidden_substrings: list[str] | None = None,
    require_clean_git: bool = False,
) -> dict[str, Any]:
    """Independently validate a Week 7 experiment run directory against the canonical contract."""
    path = Path(run_dir)
    if not path.is_dir():
        raise AssertionError(f"Run directory does not exist: {path}")

    for filename in REQUIRED_ARTIFACT_FILES:
        file_path = path / filename
        if not file_path.is_file():
            raise AssertionError(f"Missing required artifact file: {file_path}")

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    summary = json.loads((path / "summary.json").read_text(encoding="utf-8"))
    metrics_text = (path / "metrics.prom").read_text(encoding="utf-8")
    traces = read_traces_jsonl(path / "traces.jsonl")

    if manifest.get("schema_version") != 1:
        raise AssertionError(
            f"Unexpected manifest schema_version: {manifest.get('schema_version')}"
        )

    if expected_revision is not None and environment.get("git_revision") != expected_revision:
        raise AssertionError(
            f"environment.json git_revision {environment.get('git_revision')!r} "
            f"!= expected {expected_revision!r}"
        )
    if require_clean_git and environment.get("git_dirty") is not False:
        raise AssertionError(
            f"Expected clean git checkout, got git_dirty={environment.get('git_dirty')!r}"
        )

    if not all(environment.get(k) for k in ("run_id", "started_at", "ended_at")):
        raise AssertionError("environment.json missing run_id, started_at, or ended_at")

    # Verify unique terminal request IDs and no warm-up traces in traces.jsonl
    request_ids = [t.get("request_id") for t in traces]
    if len(request_ids) != len(set(request_ids)):
        raise AssertionError("Duplicate request_id found in traces.jsonl")
    for trace in traces:
        if "-warmup-" in str(trace.get("request_id", "")):
            raise AssertionError(f"Warm-up trace leaked into traces.jsonl: {trace['request_id']}")
        if trace.get("status") not in TERMINAL_STATUSES:
            raise AssertionError(f"Non-terminal trace in traces.jsonl: {trace.get('status')!r}")

    # Independently recompute summary strictly from traces.jsonl and require exact match
    recomputed = compute_summary_from_traces_file(
        path / "traces.jsonl",
        run_id=summary["run_id"],
        run_status=summary["run_status"],
        planned_requests=manifest["workload"]["request_count"],
        expected_policy=manifest["policy"]["name"],
        failure=summary.get("failure"),
    )
    if summary != recomputed:
        raise AssertionError(
            f"summary.json does not match recomputed traces.jsonl:\n{summary}\n!=\n{recomputed}"
        )

    if summary["run_status"] != expected_status:
        raise AssertionError(
            f"Expected run_status={expected_status!r}, got {summary['run_status']!r}"
        )

    if expected_policy is not None:
        if summary["policy"] != expected_policy:
            raise AssertionError(f"Expected policy={expected_policy!r}, got {summary['policy']!r}")
        if traces and summary["observed_policies"] != [expected_policy]:
            raise AssertionError(
                f"Expected observed_policies={[expected_policy]!r}, "
                f"got {summary['observed_policies']!r}"
            )

    if expected_planned is not None:
        if summary["requests"]["planned"] != expected_planned:
            raise AssertionError(
                f"Expected planned={expected_planned}, got {summary['requests']['planned']}"
            )
        if expected_status == "completed" and summary["requests"]["observed"] != expected_planned:
            raise AssertionError(
                f"Expected observed={expected_planned}, got {summary['requests']['observed']}"
            )

    if expected_status == "completed" and traces:
        if not summary["worker_distribution"]:
            raise AssertionError("Expected non-empty worker_distribution for completed run")

    # Validate metrics.prom parses cleanly as Prometheus exposition format
    families = {f.name: f for f in text_string_to_metric_families(metrics_text)}
    required_families = {"llm_gateway_requests", "llm_worker_active_requests", "llm_worker_health"}
    missing_families = required_families - set(families)
    if missing_families:
        raise AssertionError(
            f"metrics.prom missing expected metric families: {sorted(missing_families)}"
        )

    active_samples = [
        s.value
        for s in families["llm_worker_active_requests"].samples
        if s.name == "llm_worker_active_requests"
    ]
    if expected_status == "completed" and any(v != 0 for v in active_samples):
        raise AssertionError(f"Non-zero final llm_worker_active_requests: {active_samples}")

    # Check forbidden substrings (prompt text, secrets) across all generated artifacts
    if forbidden_substrings:
        for filename in REQUIRED_ARTIFACT_FILES:
            content = (path / filename).read_text(encoding="utf-8")
            for forbidden in forbidden_substrings:
                if forbidden and forbidden in content:
                    raise AssertionError(f"Forbidden text {forbidden!r} leaked into {filename}")

    return {
        "valid": True,
        "run_id": summary["run_id"],
        "run_status": summary["run_status"],
        "policy": summary["policy"],
        "requests": summary["requests"],
        "worker_distribution": summary["worker_distribution"],
        "metrics_families": sorted(families.keys()),
    }


def _wait_gateway_ready(
    gateway_url: str, proc: subprocess.Popen[str], timeout_s: float = 20.0
) -> None:
    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=2.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                raise RuntimeError(f"Mock Gateway exited early with code {proc.returncode}")
            try:
                resp = client.get(f"{gateway_url}/health")
                if resp.status_code == 200 and resp.json().get("status") == "healthy":
                    return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.15)
    raise TimeoutError(f"Mock Gateway at {gateway_url} did not become healthy within {timeout_s}s")


async def _verify_partial_failure_preservation(temp_dir: Path, workload_path: Path) -> None:
    from tests.week6_workers import mock_gateway

    manifest_path = temp_dir / "partial_manifest.json"
    manifest_data = {
        "schema_version": 1,
        "name": "w7-gate-partial-failure",
        "model": {"id": "test-model"},
        "topology": {"worker_count": 2},
        "policy": {"name": "least_loaded"},
        "workload": {"path": str(workload_path), "request_count": 4},
        "seed": 42,
        "warmup": {"request_count": 0},
        "generation": {"temperature": 0, "max_tokens": 16},
        "runner": {
            "concurrency": 1,
            "gateway_url": "http://mock-gateway",
            "request_timeout_s": 30,
        },
    }
    manifest_path.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")

    async with mock_gateway("least_loaded", "test-model") as (_, client, _):
        original_send = client.send
        chat_calls = 0

        async def drop_on_third_request(request: httpx.Request, **kwargs):
            nonlocal chat_calls
            if request.url.path == "/v1/chat/completions":
                chat_calls += 1
                if chat_calls == 3:
                    raise httpx.ConnectError("Simulated gateway drop", request=request)
            return await original_send(request, **kwargs)

        client.send = drop_on_third_request  # type: ignore[method-assign]
        runner = ExperimentRunner(runs_root=temp_dir / "partial-runs", client=client)
        result = await runner.run_from_manifest(manifest_path, base_dir=temp_dir)
        if result.succeeded:
            raise AssertionError("Expected simulated network drop to fail the run")

        validate_run_artifacts(
            result.run_dir,
            expected_policy="least_loaded",
            expected_planned=4,
            expected_status="failed",
        )
        if result.summary["requests"]["observed"] != 2:
            raise AssertionError(
                f"Expected 2 preserved traces on partial failure, got {result.summary['requests']}"
            )
        if result.summary.get("failure", {}).get("type") != "gateway_unreachable":
            raise AssertionError(f"Unexpected failure metadata: {result.summary.get('failure')}")


def run_week7_e2e_gate() -> dict[str, Any]:
    """Execute the complete deterministic CPU/mock Week 7 E2E experiment gate."""
    with tempfile.TemporaryDirectory(prefix="w7-gate-") as tmp:
        temp_dir = Path(tmp)
        prompt_sentinel = "W7_GATE_SECRET_PROMPT_SENTINEL_8841"
        workload_path = temp_dir / "gate_workload.jsonl"
        workload_lines = [
            json.dumps(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": f"{prompt_sentinel} item {idx}",
                        }
                    ]
                }
            )
            for idx in range(10)
        ]
        workload_path.write_text("\n".join(workload_lines) + "\n", encoding="utf-8")

        # 1. Verify invalid manifest is rejected
        bad_manifest = temp_dir / "invalid_manifest.json"
        bad_manifest.write_text(json.dumps({"schema_version": 99}), encoding="utf-8")
        try:
            load_manifest(bad_manifest, base_dir=temp_dir)
        except ExperimentConfigError:
            pass
        else:
            raise AssertionError("Expected invalid manifest to be rejected")

        # 2. Start a real local mock multi-worker Gateway subprocess
        port = _find_free_port()
        gateway_url = f"http://127.0.0.1:{port}"
        model_id = "Qwen/Qwen2.5-0.5B-Instruct"

        valid_manifest = temp_dir / "valid_manifest.json"
        manifest_data = {
            "schema_version": 1,
            "name": "w7-gate-least-loaded",
            "model": {"id": model_id},
            "topology": {"worker_count": 2},
            "policy": {"name": "least_loaded"},
            "workload": {"path": str(workload_path), "request_count": 8},
            "seed": 42,
            "warmup": {"request_count": 2},
            "generation": {"temperature": 0, "max_tokens": 16},
            "runner": {
                "concurrency": 2,
                "gateway_url": gateway_url,
                "request_timeout_s": 30,
            },
        }
        valid_manifest.write_text(json.dumps(manifest_data, indent=2), encoding="utf-8")
        # Confirm valid manifest loads cleanly
        load_manifest(valid_manifest, base_dir=temp_dir)

        env = os.environ.copy()
        py_path = f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'src'}"
        env["PYTHONPATH"] = f"{py_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
        env.update(
            {
                "NO_GPU": "1",
                "WORKER_MODE": "mock",
                "MOCK_WORKER_COUNT": "2",
                "ROUTING_POLICY": "least_loaded",
                "MODEL_NAME": model_id,
                "HEALTH_CHECK_INTERVAL_S": "0",
            }
        )

        gw_proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "llm_serving_platform.gateway.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--log-level",
                "warning",
            ],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            _wait_gateway_ready(gateway_url, gw_proc)

            # 3. Execute the real CLI runner against the live mock Gateway
            runs_root = temp_dir / "runs"
            cli_proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "llm_serving_platform.benchmark",
                    "run",
                    "--manifest",
                    str(valid_manifest),
                    "--output-dir",
                    str(runs_root),
                ],
                cwd=str(REPO_ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
            if cli_proc.returncode != 0:
                raise RuntimeError(
                    f"Benchmark CLI failed (exit {cli_proc.returncode}):\n"
                    f"STDOUT:\n{cli_proc.stdout}\nSTDERR:\n{cli_proc.stderr}"
                )
            cli_output = json.loads(cli_proc.stdout)
            run_dir = Path(cli_output["run_dir"])

            # 4. Validate all generated artifacts
            validation = validate_run_artifacts(
                run_dir,
                expected_policy="least_loaded",
                expected_planned=8,
                expected_status="completed",
                forbidden_substrings=[prompt_sentinel],
            )

            # Confirm warm-up ran (2 warmup + 8 measured = 10 total completed gateway requests)
            metrics_text = (run_dir / "metrics.prom").read_text(encoding="utf-8")
            families = {f.name: f for f in text_string_to_metric_families(metrics_text)}
            completed_total = sum(
                s.value
                for s in families["llm_gateway_requests"].samples
                if s.name == "llm_gateway_requests_total" and s.labels.get("status") == "completed"
            )
            if int(completed_total) != 10:
                raise AssertionError(
                    f"Expected 10 total gateway requests (2 warmup + 8 measured), "
                    f"got {completed_total}"
                )
        finally:
            gw_proc.terminate()
            try:
                gw_proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                gw_proc.kill()
                gw_proc.wait(timeout=5)

        # 5. Verify partial/failure path preserves evidence
        asyncio.run(_verify_partial_failure_preservation(temp_dir, workload_path))

        print(
            f"  [w7] Run ID: {validation['run_id']} | "
            f"Requests: {validation['requests']} | "
            f"Workers: {validation['worker_distribution']}"
        )
        return validation


if __name__ == "__main__":
    run_week7_e2e_gate()
