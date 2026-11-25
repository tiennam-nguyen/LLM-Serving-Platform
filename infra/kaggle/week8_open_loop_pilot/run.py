"""Week 8 open-loop dual-T4 pilot launcher (Member A scope, pilot-only).

Reuses the verified Week 6/7 dual-T4 GPU harness (infra/kaggle/dual_t4_validation/run.py)
to sweep open-loop steady offered rates across separate RR and LL runs with explicit
worker restarts (clean cache state), alternating policy order, and identical paired
ordered workloads/schedules.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
for _p in (ROOT, ROOT / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from infra.kaggle.dual_t4_validation import run as dual_t4  # noqa: E402
from infra.kaggle.week6_routing_evidence.run import (  # noqa: E402
    checked_inventory,
    require_unused_port,
)
from infra.kaggle.week8_open_loop_pilot.pilot_metrics import (  # noqa: E402
    DEFAULT_SELECTION_CRITERIA,
    compute_pilot_metrics_from_dir,
    select_common_load_points,
    validate_pilot_run_dir,
    write_pilot_reports,
)
from infra.kaggle.week8_open_loop_pilot.pilot_schema import (  # noqa: E402
    PilotManifestV2,
    generate_steady_pilot_workload,
    validate_pilot_manifest,
)
from llm_serving_platform.benchmark.runner import ExperimentRunner  # noqa: E402
from llm_serving_platform.benchmark.summary import (  # noqa: E402
    compute_percentile,
    read_traces_jsonl,
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _installed_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for pkg in (
        "llm-serving-platform",
        "vllm",
        "torch",
        "transformers",
        "fastapi",
        "httpx",
        "pydantic",
        "prometheus-client",
        "uvicorn",
    ):
        try:
            versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            continue
    return versions


def load_w7_historical_anchor(repo_root: Path) -> dict[str, Any] | None:
    """Recompute W7 historical closed-loop anchor purely as starting-rate context."""
    live_root = repo_root / "experiments" / "raw" / "week7" / "live"
    if not live_root.is_dir():
        return None
    policies: dict[str, Any] = {}
    for policy in ("round_robin", "least_loaded"):
        p_dir = live_root / policy
        traces_path = p_dir / "traces.jsonl"
        manifest_path = p_dir / "manifest.json"
        if not traces_path.is_file() or not manifest_path.is_file():
            continue
        traces = read_traces_jsonl(traces_path)
        e2e = [
            float(t["metrics_ms"]["e2e_ms"])
            for t in traces
            if t.get("status") == "completed" and t.get("metrics_ms", {}).get("e2e_ms") is not None
        ]
        arrivals = [
            datetime.fromisoformat(t["milestones"]["arrival"]).timestamp() for t in traces
        ]
        last_toks = [
            datetime.fromisoformat(t["milestones"]["last_token"]).timestamp() for t in traces
        ]
        span_s = max(last_toks) - min(arrivals) if arrivals and last_toks else 0.0
        policies[policy] = {
            "completed": len(e2e),
            "achieved_rps": round(len(e2e) / span_s, 3) if span_s > 0 else None,
            "e2e_p50_ms": compute_percentile(e2e, 50),
            "e2e_p95_ms": compute_percentile(e2e, 95),
        }
    return {
        "label": "historical W7 closed/concurrency-limited context — not W8 controlled evidence",
        "policies": policies,
    }


def _supports_no_prefix_caching_flag() -> bool:
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        return "--no-enable-prefix-caching" in (proc.stdout + proc.stderr)
    except (OSError, subprocess.SubprocessError):
        return False


def _start_vllm_replica(
    index: int,
    env: dict[str, str],
    model: str,
    *,
    disable_prefix_caching: bool,
) -> tuple[Any, dict[str, Any]]:
    """Start one TP=1 float16 vLLM worker strictly pinned to physical GPU `index`."""
    require_unused_port(8000 + index)
    worker_env = {**env, "CUDA_VISIBLE_DEVICES": str(index), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
    visible = dual_t4.torch_info(worker_env)
    dual_t4.require(
        len(visible["devices"]) == 1 and "T4" in visible["devices"][0],
        f"Expected one visible T4 for worker {index}, got {visible['devices']}",
    )
    args = [
        sys.executable,
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model,
        "--host",
        "127.0.0.1",
        "--port",
        str(8000 + index),
        "--tensor-parallel-size",
        "1",
        "--dtype",
        "float16",
        "--gpu-memory-utilization",
        dual_t4.MEMORY,
        "--max-model-len",
        dual_t4.CONTEXT,
    ]
    if disable_prefix_caching:
        args.append("--no-enable-prefix-caching")
    launch = {
        "CUDA_VISIBLE_DEVICES": str(index),
        "args": args[1:],
        "visible_devices": visible["devices"],
    }
    return dual_t4.Service(args, f"worker{index}.log", worker_env), launch


def _wait_both_replicas_ready(
    services: dict[int, Any], model: str, timeout_s: float = 480.0
) -> None:
    import httpx

    deadline = time.monotonic() + timeout_s
    with httpx.Client(timeout=4.0, trust_env=False) as client:
        while time.monotonic() < deadline:
            for svc in services.values():
                svc.check_running()
            ready0 = dual_t4.replica_ready(client, 0, model) is not None
            ready1 = dual_t4.replica_ready(client, 1, model) is not None
            if ready0 and ready1:
                print("READY worker-vllm-0 and worker-vllm-1", flush=True)
                return
            time.sleep(2)
    raise TimeoutError("Timed out waiting for both vLLM replicas to become ready")


def _wait_port_free(port: int, timeout_s: float = 15.0) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            probe.settimeout(0.5)
            if probe.connect_ex(("127.0.0.1", port)) != 0:
                return
        time.sleep(0.25)
    raise RuntimeError(f"Port {port} remained in use after {timeout_s}s")


def _wait_gateway_ready(
    gateway: Any,
    gateway_url: str,
    expected_workers: dict[str, tuple[str, str, str]],
    services: tuple[Any, ...],
) -> None:
    import httpx

    with httpx.Client(timeout=4.0, trust_env=False) as client:
        deadline = time.monotonic() + 60.0
        while time.monotonic() < deadline:
            gateway.check_running()
            for svc in services:
                svc.check_running()
            try:
                health = client.get(f"{gateway_url}/health")
                if health.status_code == 200:
                    resp = client.get(f"{gateway_url}/control/v1/workers")
                    if resp.status_code == 200:
                        workers = resp.json()
                        actual = {
                            row["worker_id"]: (row["endpoint"], row["model"], row["status"])
                            for row in workers
                        }
                        if actual == expected_workers:
                            return
            except (httpx.HTTPError, ValueError):
                pass
            time.sleep(0.3)
    raise TimeoutError(f"Timed out waiting for Gateway at {gateway_url}")


def _build_pilot_manifest(
    *,
    name: str,
    model: str,
    policy: str,
    workload_path: Path,
    request_count: int,
    warmup_count: int,
    gateway_url: str,
    request_timeout_s: float,
    max_in_flight: int,
    max_dispatch_lag_ms: float,
    window_ms: float,
    comparison_id: str,
    seed: int = 42,
) -> PilotManifestV2:
    raw = {
        "schema_version": 2,
        "name": name,
        "model": {"id": model},
        "topology": {"worker_count": 2},
        "policy": {"name": policy},
        "workload": {
            "path": str(workload_path),
            "profile": "steady",
            "request_count": request_count,
        },
        "seed": seed,
        "warmup": {"request_count": warmup_count},
        "generation": {"temperature": 0, "max_tokens": 64, "top_p": 1},
        "runner": {
            "gateway_url": gateway_url,
            "request_timeout_s": request_timeout_s,
            "max_in_flight": max_in_flight,
            "max_dispatch_lag_ms": max_dispatch_lag_ms,
        },
        "load": {"mode": "open_loop", "measurement_window_ms": window_ms},
        "comparison": {
            "id": comparison_id,
            "load_level": "pilot",
            "repetition": 1,
        },
    }
    return validate_pilot_manifest(raw)


def _rate_tag(rate_rps: float) -> str:
    return f"{rate_rps:.2f}".rstrip("0").rstrip(".").replace(".", "p")


def _summarize_run_row(
    *,
    metrics: dict[str, Any],
    policy_order: str,
    valid: bool,
    invalid_reason: str | None,
) -> dict[str, Any]:
    lat = metrics["latency_ms"]
    lag = metrics["dispatch_lag_ms"]
    rates = metrics["rates"]
    reqs = metrics["requests"]
    return {
        "policy": metrics["policy"],
        "policy_order": policy_order,
        "offered_rps": rates["offered_rps"],
        "sent": reqs["sent"],
        "planned": reqs["planned"],
        "completed": reqs["completed"],
        "both_completed": reqs["both_completed"],
        "completed_window_rps": rates["completed_window_rps"],
        "completed_drain_rps": rates["completed_drain_rps"],
        "ttft_p50_ms": lat["ttft_p50"],
        "ttft_p95_ms": lat["ttft_p95"],
        "ttft_p99_ms": lat["ttft_p99"],
        "e2e_p50_ms": lat["e2e_p50"],
        "e2e_p95_ms": lat["e2e_p95"],
        "e2e_p99_ms": lat["e2e_p99"],
        "routing_p50_ms": lat["routing_p50"],
        "routing_p95_ms": lat["routing_p95"],
        "routing_p99_ms": lat["routing_p99"],
        "dispatch_lag_p95_ms": lag["p95"],
        "dispatch_lag_p99_ms": lag["p99"],
        "dispatch_lag_max_ms": lag["max"],
        "noncompletion_rate": rates["noncompletion_rate"],
        "gateway_error_rate": rates["gateway_error_rate"],
        "gateway_cancelled_rate": rates["gateway_cancelled_rate"],
        "backlog_at_window_end": metrics["backlog_at_window_end"],
        "worker_distribution": json.dumps(metrics["worker_distribution"], sort_keys=True),
        "run_id": metrics["run_id"],
        "schedule_sha256": metrics["schedule_sha256"],
        "valid": valid,
        "invalid_reason": invalid_reason or "",
    }


def _build_pair_entry(
    *,
    offered_rps: float,
    policy_order: str,
    rr_metrics: dict[str, Any],
    ll_metrics: dict[str, Any],
    rr_valid: bool,
    ll_valid: bool,
    rr_reason: str | None,
    ll_reason: str | None,
) -> dict[str, Any]:
    same_sched = rr_metrics["schedule_sha256"] == ll_metrics["schedule_sha256"]
    same_workload = rr_metrics["workload_sha256"] == ll_metrics["workload_sha256"]
    pair_valid = rr_valid and ll_valid and same_sched and same_workload

    def _compact(m: dict[str, Any], ok: bool, reason: str | None) -> dict[str, Any]:
        return {
            "run_id": m["run_id"],
            "valid": ok,
            "invalid_reason": reason,
            "planned": m["requests"]["planned"],
            "sent": m["requests"]["sent"],
            "both_completed": m["requests"]["both_completed"],
            "missing_trace": m["requests"]["missing_trace"],
            "completed_drain_rps": m["rates"]["completed_drain_rps"],
            "completed_window_rps": m["rates"]["completed_window_rps"],
            "noncompletion_rate": m["rates"]["noncompletion_rate"],
            "e2e_p50": m["latency_ms"]["e2e_p50"],
            "e2e_p95": m["latency_ms"]["e2e_p95"],
            "e2e_p99": m["latency_ms"]["e2e_p99"],
            "ttft_p50": m["latency_ms"]["ttft_p50"],
            "ttft_p95": m["latency_ms"]["ttft_p95"],
            "ttft_p99": m["latency_ms"]["ttft_p99"],
            "routing_p50": m["latency_ms"]["routing_p50"],
            "routing_p95": m["latency_ms"]["routing_p95"],
            "routing_p99": m["latency_ms"]["routing_p99"],
            "dispatch_lag_max_ms": m["dispatch_lag_ms"]["max"],
            "backlog_at_window_end": m["backlog_at_window_end"],
            "worker_distribution": m["worker_distribution"],
        }

    return {
        "offered_rps": offered_rps,
        "policy_order": policy_order,
        "pair_valid": pair_valid,
        "workload_sha256": rr_metrics["workload_sha256"],
        "schedule_sha256": rr_metrics["schedule_sha256"],
        "same_hashes": same_sched and same_workload,
        "round_robin": _compact(rr_metrics, rr_valid, rr_reason),
        "least_loaded": _compact(ll_metrics, ll_valid, ll_reason),
    }


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class _LocalMockGatewayProc:
    """Cross-platform subprocess wrapper used only in --dry-run-mock CPU smoke checks."""

    def __init__(self, args: list[str], log_path: Path, env: dict[str, str]) -> None:
        self._log_file = log_path.open("ab")
        self.process = subprocess.Popen(
            args,
            cwd=str(ROOT),
            env=env,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
        )

    def check_running(self) -> None:
        dual_t4.require(
            self.process.poll() is None,
            f"Mock gateway exited early with code {self.process.returncode}",
        )

    def stop(self) -> None:
        try:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            self._log_file.close()


def run_pilot_campaign(args: argparse.Namespace, inventory: list[str] | None) -> dict[str, Any]:
    """Execute adaptive steady open-loop sweep on live dual-T4s or CPU mock mode."""
    start_mono = time.monotonic()
    deadline_mono = start_mono + float(args.time_budget_s)
    dry_run = bool(args.dry_run_mock)

    output_dir: Path = args.output_dir.resolve()
    runs_dir = output_dir / "runs"
    workloads_dir = output_dir / "workloads"
    runs_dir.mkdir(parents=True, exist_ok=True)
    workloads_dir.mkdir(parents=True, exist_ok=True)

    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    branch = dual_t4.command("git", "branch", "--show-current", cwd=ROOT, capture=True)

    if dry_run:
        torch_meta = {"torch": "mock", "cuda": "cpu", "devices": ["MockGPU-0", "MockGPU-1"]}
        disable_prefix_caching = False
        cache_control = {
            "mode": "dry_run_mock_fresh_gateway_per_run",
            "workers_restarted_between_runs": False,
        }
        gw_port = _find_free_port()
        gateway_url = f"http://127.0.0.1:{gw_port}"
        expected_workers = {
            "worker-mock-0": ("mock://worker-mock-0", args.model, "HEALTHY"),
            "worker-mock-1": ("mock://worker-mock-1", args.model, "HEALTHY"),
        }
    else:
        torch_meta = dual_t4.torch_info()
        dual_t4.require(
            len(torch_meta["devices"]) == 2
            and all("T4" in name for name in torch_meta["devices"]),
            f"Expected exactly two CUDA-visible T4 GPUs, got {torch_meta['devices']}",
        )
        disable_prefix_caching = _supports_no_prefix_caching_flag()
        cache_control = {
            "mode": "full_vllm_worker_restart_between_every_policy_run",
            "workers_restarted_between_runs": True,
            "vllm_no_enable_prefix_caching_flag_applied": disable_prefix_caching,
        }
        gw_port = 8080
        gateway_url = dual_t4.GATEWAY
        expected_workers = {
            worker_id: (endpoint, args.model, "HEALTHY")
            for worker_id, endpoint in zip(dual_t4.IDS, dual_t4.ENDPOINTS)
        }

    environment_info: dict[str, Any] = {
        "schema_version": 2,
        "artifact_role": "pilot_selection_evidence_not_member_b_summary",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": args.revision,
        "git_dirty": False if not dry_run else None,
        "branch": branch,
        "python": platform.python_version(),
        "pytorch": torch_meta["torch"],
        "cuda": torch_meta["cuda"],
        "packages": _installed_versions(),
        "gpu_inventory": inventory,
        "model": args.model,
        "tensor_parallel_size": 1,
        "vllm_settings": {
            "dtype": "float16",
            "gpu_memory_utilization": dual_t4.MEMORY,
            "max_model_len": dual_t4.CONTEXT,
            "disable_prefix_caching_flag": disable_prefix_caching,
        },
        "generation": {"temperature": 0, "max_tokens": 64, "top_p": 1},
        "warmup_requests": args.warmup_count,
        "measurement_window_ms": args.window_ms,
        "max_in_flight": args.max_in_flight,
        "max_dispatch_lag_ms": args.max_dispatch_lag_ms,
        "request_timeout_s": args.request_timeout_s,
        "cache_control": cache_control,
        "test_mode": "dry_run_mock" if dry_run else "live_dual_t4",
        "dual_t4_topology_verified": not dry_run,
        "worker_gpu_bindings": {},
        "historical_w7_anchor": load_w7_historical_anchor(ROOT),
    }

    services: dict[int, Any] = {}
    cold_boot_done = False

    def _stop_workers() -> None:
        for idx in list(services.keys()):
            svc = services.pop(idx)
            svc.stop()
            dual_t4.require(
                not dual_t4.group_pids(svc.process.pid),
                f"Worker {idx} process group did not terminate cleanly",
            )
        if not dry_run:
            _wait_port_free(8000)
            _wait_port_free(8001)

    def _start_fresh_workers(phase_label: str) -> None:
        nonlocal cold_boot_done
        if dry_run:
            return
        _stop_workers()
        if not cold_boot_done:
            # First cold start runs sequentially to avoid concurrent HF weight download races.
            svc0, _ = _start_vllm_replica(
                0, env, args.model, disable_prefix_caching=disable_prefix_caching
            )
            services[0] = svc0
            import httpx

            with httpx.Client(timeout=4.0, trust_env=False) as c:
                dual_t4.wait_for(
                    dual_t4.IDS[0],
                    lambda: dual_t4.replica_ready(c, 0, args.model),
                    480,
                    (svc0,),
                )
            svc1, _ = _start_vllm_replica(
                1, env, args.model, disable_prefix_caching=disable_prefix_caching
            )
            services[1] = svc1
            with httpx.Client(timeout=4.0, trust_env=False) as c:
                dual_t4.wait_for(
                    dual_t4.IDS[1],
                    lambda: dual_t4.replica_ready(c, 1, args.model),
                    480,
                    (svc0, svc1),
                )
            cold_boot_done = True
        else:
            # Subsequent restarts load weights from local HF cache in parallel.
            svc0, _ = _start_vllm_replica(
                0, env, args.model, disable_prefix_caching=disable_prefix_caching
            )
            services[0] = svc0
            svc1, _ = _start_vllm_replica(
                1, env, args.model, disable_prefix_caching=disable_prefix_caching
            )
            services[1] = svc1
            _wait_both_replicas_ready(services, args.model, timeout_s=480.0)

        binding = dual_t4.capture_processes(phase_label, services)
        if phase_label in {"initial", "final"} or len(environment_info["worker_gpu_bindings"]) < 4:
            environment_info["worker_gpu_bindings"][phase_label] = {
                wid: {
                    "gpu_index": rec["gpu_index"],
                    "gpu_uuid": rec["gpu_uuid"],
                    "owned_compute_processes": len(rec["compute_processes"]),
                }
                for wid, rec in binding.items()
            }

    run_counter = 0
    all_run_rows: list[dict[str, Any]] = []
    pair_summaries: list[dict[str, Any]] = []
    tested_rates: set[float] = set()

    def _run_one_pair(rate_rps: float, pair_idx: int) -> dict[str, Any]:
        nonlocal run_counter
        req_count = max(1, round(rate_rps * (args.window_ms / 1000.0)))
        actual_offered_rps = round(req_count / (args.window_ms / 1000.0), 4)
        tag = _rate_tag(actual_offered_rps)

        workload_path = workloads_dir / f"steady_{tag}rps.jsonl"
        _, scheduled_entries, workload_sha = generate_steady_pilot_workload(
            workload_path,
            request_count=req_count,
            measurement_window_ms=float(args.window_ms),
            seed=args.seed,
        )
        prompts = [s.entry.messages[0]["content"] for s in scheduled_entries]

        if pair_idx % 2 == 0:
            policy_pair = ("round_robin", "least_loaded")
            policy_order = "RR->LL"
        else:
            policy_pair = ("least_loaded", "round_robin")
            policy_order = "LL->RR"

        pair_results: dict[str, tuple[dict[str, Any], bool, str | None]] = {}
        comparison_id = f"pilot-{tag}rps"

        for policy in policy_pair:
            run_counter += 1
            bind_label = "initial" if run_counter == 1 else f"run_{run_counter}_{policy}"
            _start_fresh_workers(bind_label)

            _wait_port_free(gw_port)
            gateway_env = {
                **env,
                "MODEL_NAME": args.model,
                "ROUTING_POLICY": policy,
                "HEALTH_CHECK_INTERVAL_S": "2" if not dry_run else "0",
                "HEALTH_CHECK_TIMEOUT_S": "3",
                "NO_GPU": "1" if dry_run else "0",
                "WORKER_MODE": "mock" if dry_run else "live",
                "MOCK_WORKER_COUNT": "2",
            }
            if not dry_run:
                gateway_env["WORKER_ENDPOINTS"] = ",".join(
                    f"{k}={u}" for k, u in zip(dual_t4.IDS, dual_t4.ENDPOINTS)
                )
            gw_cmd = [
                sys.executable,
                "-m",
                "uvicorn",
                "llm_serving_platform.gateway.main:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(gw_port),
                "--log-level",
                "warning",
            ]
            gw_log_name = f"gateway-p{pair_idx + 1}-{policy}-{tag}.log"
            gateway: Any = (
                _LocalMockGatewayProc(gw_cmd, dual_t4.OUTPUT / gw_log_name, gateway_env)
                if dry_run
                else dual_t4.Service(gw_cmd, gw_log_name, gateway_env)
            )
            try:
                _wait_gateway_ready(
                    gateway,
                    gateway_url,
                    expected_workers,
                    tuple(services.values()),
                )
                cfg = _build_pilot_manifest(
                    name=f"w8-pilot-{tag}rps-{policy}",
                    model=args.model,
                    policy=policy,
                    workload_path=workload_path,
                    request_count=req_count,
                    warmup_count=args.warmup_count,
                    gateway_url=gateway_url,
                    request_timeout_s=args.request_timeout_s,
                    max_in_flight=args.max_in_flight,
                    max_dispatch_lag_ms=args.max_dispatch_lag_ms,
                    window_ms=args.window_ms,
                    comparison_id=comparison_id,
                    seed=args.seed,
                )
                runner = ExperimentRunner(runs_root=runs_dir, repo_root=ROOT)
                res = asyncio.run(
                    runner.run_open_loop(
                        cfg,
                        scheduled_entries,
                        workload_sha256=workload_sha,
                    )
                )
                metrics = compute_pilot_metrics_from_dir(res.run_dir)
                _write_json(res.run_dir / "pilot_metrics.json", metrics)

                valid = False
                invalid_reason: str | None = None
                try:
                    validate_pilot_run_dir(
                        res.run_dir,
                        expected_policy=policy,
                        expected_planned=req_count,
                        expected_revision=args.revision if not dry_run else None,
                        require_clean_git=not dry_run,
                        forbidden_substrings=prompts,
                        max_dispatch_lag_ms=args.max_dispatch_lag_ms,
                        require_completed=(res.run_status == "completed"),
                    )
                    if res.run_status != "completed":
                        invalid_reason = f"run_status={res.run_status}:{res.failure_type}"
                    else:
                        valid = True
                except AssertionError as exc:
                    invalid_reason = f"validation_failed:{exc}"

                pair_results[policy] = (metrics, valid, invalid_reason)
                row = _summarize_run_row(
                    metrics=metrics,
                    policy_order=policy_order,
                    valid=valid,
                    invalid_reason=invalid_reason,
                )
                all_run_rows.append(row)
                print(
                    f"[PILOT {tag} RPS | {policy_order}] {policy}: valid={valid} "
                    f"sent={metrics['requests']['sent']}/{req_count} "
                    f"drain_rps={metrics['rates']['completed_drain_rps']:.3f} "
                    f"e2e_p50={metrics['latency_ms']['e2e_p50']} "
                    f"e2e_p95={metrics['latency_ms']['e2e_p95']} "
                    f"backlog={metrics['backlog_at_window_end']} "
                    f"lag_max={metrics['dispatch_lag_ms']['max']}",
                    flush=True,
                )
            finally:
                gateway.stop()

        rr_m, rr_ok, rr_err = pair_results["round_robin"]
        ll_m, ll_ok, ll_err = pair_results["least_loaded"]
        pair_entry = _build_pair_entry(
            offered_rps=actual_offered_rps,
            policy_order=policy_order,
            rr_metrics=rr_m,
            ll_metrics=ll_m,
            rr_valid=rr_ok,
            ll_valid=ll_ok,
            rr_reason=rr_err,
            ll_reason=ll_err,
        )
        pair_summaries.append(pair_entry)
        tested_rates.add(actual_offered_rps)

        # Save incremental report after every pair so partial progress is never lost.
        selection_now = select_common_load_points(
            pair_summaries, criteria=DEFAULT_SELECTION_CRITERIA
        )
        write_pilot_reports(
            output_dir,
            run_rows=all_run_rows,
            pair_summaries=pair_summaries,
            criteria=DEFAULT_SELECTION_CRITERIA,
            selection=selection_now,
            environment=environment_info,
        )
        return pair_entry

    try:
        coarse_rates = [float(x.strip()) for x in args.coarse_rates.split(",") if x.strip()]
        consecutive_unstable = 0
        saw_stable = False

        for rate in coarse_rates:
            if time.monotonic() + 240.0 > deadline_mono and pair_summaries:
                print("Time budget guard reached during coarse sweep; finalizing.", flush=True)
                break
            pair = _run_one_pair(rate, len(pair_summaries))
            sel_mid = select_common_load_points(pair_summaries)
            eval_map = {e["offered_rps"]: e for e in sel_mid["evaluations"]}
            ev = eval_map[pair["offered_rps"]]
            if ev["both_stable"]:
                saw_stable = True
                consecutive_unstable = 0
            else:
                consecutive_unstable += 1
                rr_drain = float(pair["round_robin"]["completed_drain_rps"] or 0.0)
                ll_drain = float(pair["least_loaded"]["completed_drain_rps"] or 0.0)
                rr_nc = float(pair["round_robin"]["noncompletion_rate"] or 0.0)
                ll_nc = float(pair["least_loaded"]["noncompletion_rate"] or 0.0)
                clear_saturation = (
                    not pair["pair_valid"]
                    or rr_drain < 0.85 * pair["offered_rps"]
                    or ll_drain < 0.85 * pair["offered_rps"]
                    or rr_nc > 0.05
                    or ll_nc > 0.05
                )
                if saw_stable and (consecutive_unstable >= 2 or clear_saturation):
                    print(
                        f"Coarse sweep reached saturation at {pair['offered_rps']} RPS; "
                        "moving to knee refinement.",
                        flush=True,
                    )
                    break

        # If all coarse rates remained stable and time budget allows, check higher extension rates.
        if (
            not args.no_refine
            and saw_stable
            and consecutive_unstable == 0
            and not dry_run
            and time.monotonic() + 360.0 < deadline_mono
        ):
            for extra_rate in (20.0, 24.0):
                if extra_rate <= max(tested_rates) or time.monotonic() + 240.0 > deadline_mono:
                    continue
                pair = _run_one_pair(extra_rate, len(pair_summaries))
                sel_mid = select_common_load_points(pair_summaries)
                ev = {e["offered_rps"]: e for e in sel_mid["evaluations"]}[pair["offered_rps"]]
                if not ev["both_stable"]:
                    break

        # Stage 2: Knee refinement around transition boundary.
        if not args.no_refine and args.max_refine_pairs > 0:
            for _ in range(args.max_refine_pairs):
                if time.monotonic() + 240.0 > deadline_mono:
                    break
                sel_now = select_common_load_points(pair_summaries)
                evals = sorted(sel_now["evaluations"], key=lambda e: float(e["offered_rps"]))
                stable_evals = [e for e in evals if e["both_stable"]]
                unstable_evals = [e for e in evals if not e["both_stable"]]

                next_probe: float | None = None
                if not stable_evals and unstable_evals:
                    lowest = float(unstable_evals[0]["offered_rps"])
                    cand = round(lowest * 0.5, 2)
                    if cand >= 0.25 and cand not in tested_rates:
                        next_probe = cand
                elif stable_evals and unstable_evals:
                    hi_stable = float(stable_evals[-1]["offered_rps"])
                    first_unstable = next(
                        (
                            float(e["offered_rps"])
                            for e in unstable_evals
                            if float(e["offered_rps"]) > hi_stable
                        ),
                        None,
                    )
                    if first_unstable is not None:
                        mid = round(0.5 * (hi_stable + first_unstable), 2)
                        if mid not in tested_rates and abs(first_unstable - hi_stable) >= 0.5:
                            next_probe = mid
                    if next_probe is None and not sel_now["knee_reached"]:
                        # Check missing geometric midpoint between low and near_saturation.
                        low_r = float(stable_evals[0]["offered_rps"])
                        if low_r < hi_stable:
                            geo = round(math.sqrt(low_r * hi_stable), 2)
                            if geo not in tested_rates:
                                next_probe = geo

                if next_probe is None:
                    break
                _run_one_pair(next_probe, len(pair_summaries))

        final_selection = select_common_load_points(
            pair_summaries, criteria=DEFAULT_SELECTION_CRITERIA
        )
        write_pilot_reports(
            output_dir,
            run_rows=all_run_rows,
            pair_summaries=pair_summaries,
            criteria=DEFAULT_SELECTION_CRITERIA,
            selection=final_selection,
            environment=environment_info,
        )
        return {
            "passed": True,
            "selection": final_selection,
            "pairs_tested": len(pair_summaries),
            "output_dir": str(output_dir),
        }
    finally:
        _stop_workers()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=False, default="HEAD")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--vllm-version", default="0.33.0")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("/kaggle/working/week8/pilot")
    )
    parser.add_argument("--window-ms", type=float, default=20000.0)
    parser.add_argument("--warmup-count", type=int, default=2)
    parser.add_argument("--request-timeout-s", type=float, default=120.0)
    parser.add_argument("--max-in-flight", type=int, default=180)
    parser.add_argument("--max-dispatch-lag-ms", type=float, default=250.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coarse-rates", default="1.0,2.0,4.0,8.0,12.0,16.0")
    parser.add_argument("--max-refine-pairs", type=int, default=2)
    parser.add_argument("--no-refine", action="store_true")
    parser.add_argument("--time-budget-s", type=float, default=2900.0)
    parser.add_argument("--dry-run-mock", action="store_true")
    args = parser.parse_args()

    dual_t4.REPO = ROOT
    if not args.dry_run_mock:
        installed_rev = dual_t4.command("git", "rev-parse", "HEAD", cwd=ROOT, capture=True)
        if args.revision != "HEAD":
            dual_t4.require(installed_rev == args.revision, "Checkout revision mismatch")
        else:
            args.revision = installed_rev
        dual_t4.require(
            not dual_t4.command("git", "status", "--porcelain", cwd=ROOT, capture=True),
            "Expected a clean source checkout",
        )
        inventory = checked_inventory()
    else:
        args.revision = dual_t4.command("git", "rev-parse", "HEAD", cwd=ROOT, capture=True)
        inventory = None

    if args.bootstrap and not args.dry_run_mock:
        python = dual_t4.serving_python()
        requirement = f"vllm=={args.vllm_version}" if args.vllm_version else "vllm"
        dual_t4.install(python, vllm_requirement=requirement)
        cmd = [
            python,
            str(Path(__file__).resolve()),
            "--revision",
            args.revision,
            "--model",
            args.model,
            "--output-dir",
            str(args.output_dir),
            "--window-ms",
            str(args.window_ms),
            "--warmup-count",
            str(args.warmup_count),
            "--request-timeout-s",
            str(args.request_timeout_s),
            "--max-in-flight",
            str(args.max_in_flight),
            "--max-dispatch-lag-ms",
            str(args.max_dispatch_lag_ms),
            "--seed",
            str(args.seed),
            "--coarse-rates",
            args.coarse_rates,
            "--max-refine-pairs",
            str(args.max_refine_pairs),
            "--time-budget-s",
            str(args.time_budget_s),
        ]
        if args.no_refine:
            cmd.append("--no-refine")
        dual_t4.command(*cmd, timeout=3500, cwd=ROOT)
        return

    dual_t4.OUTPUT = args.output_dir.resolve().parent / "service-logs"
    dual_t4.OUTPUT.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = run_pilot_campaign(args, inventory)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
