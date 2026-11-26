"""Execute a materialized W8 campaign with fresh services per run, CPU/mock or dual-T4."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm_serving_platform.benchmark.artifacts import (  # noqa: E402
    capture_git_info,
    write_json_atomic,
)
from llm_serving_platform.benchmark.config import ExperimentConfigV2, load_manifest  # noqa: E402
from llm_serving_platform.benchmark.runner import ExperimentRunner  # noqa: E402
from llm_serving_platform.benchmark.workload import load_workload  # noqa: E402


class GpuTelemetry:
    """Owned, read-only 1 Hz NVML/SMI sampler; constructed only on explicit live runs."""

    fields = (
        "timestamp,index,uuid,name,temperature.gpu,utilization.gpu,"
        "clocks.current.sm,pstate,power.draw,memory.used"
    )

    def __init__(self, path: Path):
        self.output = path.open("wb")
        self.output.write((self.fields + "\n").encode())
        self.output.flush()
        try:
            self.process = subprocess.Popen(
                [
                    "nvidia-smi",
                    "--query-gpu=" + self.fields,
                    "--format=csv,noheader,nounits",
                    "--loop=1",
                ],
                stdout=self.output,
                stderr=subprocess.STDOUT,
            )
        except BaseException:
            self.output.close()
            raise

    def stop(self):
        try:
            if self.process.poll() is not None:
                raise RuntimeError("GPU telemetry sampler terminated unexpectedly")
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        finally:
            self.output.close()


def execute_campaign(
    protocol_path: Path,
    *,
    output_dir: Path,
    no_gpu: bool,
    revision: str | None = None,
    limit: int | None = None,
    calibration: bool = False,
    prefetched_model: bool = False,
    model_revision: str | None = None,
    gpu_telemetry: bool = False,
) -> dict:
    """Never install an engine or download weights in mock mode. Preserve failed runs."""
    from infra.kaggle.week8_open_loop_pilot import run as pilot

    protocol_bytes = protocol_path.read_bytes().replace(b"\r\n", b"\n")
    protocol = json.loads(protocol_bytes)
    records = protocol.get("runs")
    if protocol.get("schema_version") != 1 or not isinstance(records, list) or not records:
        raise ValueError("Invalid campaign protocol")
    if limit is not None and (not no_gpu or type(limit) is not int or limit < 1):
        raise ValueError("--limit is for explicit CPU smoke checks only")
    phase = protocol.get("experiment_phase", "main")
    if phase not in {"main", "calibration", "conditioning"}:
        raise ValueError("Unknown study phase")
    if calibration != (phase in {"calibration", "conditioning"}):
        raise ValueError("Calibration requires both the explicit flag and calibration protocol")
    if no_gpu and (prefetched_model or gpu_telemetry or model_revision is not None):
        raise ValueError("GPU preparation/telemetry options require explicit live mode")
    if prefetched_model and not model_revision:
        raise ValueError("Prefetched model requires its recorded snapshot revision")
    selected = records[:limit] if limit is not None else records
    manifests = []
    for row in selected:
        manifest = (protocol_path.parent / row["manifest"]).resolve()
        if not manifest.is_relative_to(protocol_path.parent.resolve()):
            raise ValueError("Campaign manifest escapes protocol directory")
        cfg, workload = load_manifest(manifest)
        if not isinstance(cfg, ExperimentConfigV2):
            raise ValueError("Campaign manifests must use schema v2")
        load_workload(
            workload,
            schema_version=2,
            expected_count=cfg.workload.request_count,
            measurement_window_ms=cfg.load.measurement_window_ms,
        )
        digest = hashlib.sha256(workload.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if (
            row["workload_sha256"] != digest
            or row["policy"] != cfg.policy.name
            or row["comparison_id"] != cfg.comparison.id
            or row["repetition"] != cfg.comparison.repetition
        ):
            raise ValueError("Campaign manifest/workload differs from frozen protocol")
        manifests.append((manifest, cfg))
    if output_dir.exists():
        raise FileExistsError("Use a new campaign output directory")
    if not no_gpu:
        source = capture_git_info(ROOT)
        if (
            sys.platform == "win32"
            or not revision
            or source["git_revision"] != revision
            or source["git_dirty"] is not False
        ):
            raise ValueError("Live dual-T4 campaign requires a clean Linux checkout at --revision")
        if not calibration and protocol.get("repeatability", {}).get("status") != "predeclared":
            raise ValueError("Freeze repeated-pilot tolerances before the live campaign")
        if calibration and protocol.get("repeatability", {}).get("status") != "calibration_only":
            raise ValueError("Calibration protocol must remain separate from main acceptance")
        pilot.checked_inventory()
        torch = pilot.dual_t4.torch_info()
        if len(torch["devices"]) != 2 or not all("T4" in name for name in torch["devices"]):
            raise ValueError("Live campaign requires exactly two CUDA-visible T4s")
    output_dir.mkdir(parents=True)
    protocol_hash = hashlib.sha256(protocol_bytes).hexdigest()
    report = {
        "schema_version": 1,
        "test_mode": "cpu_mock" if no_gpu else "live_dual_t4",
        "experiment_phase": phase,
        "protocol_sha256": protocol_hash,
        "source_git": capture_git_info(ROOT),
        "full_campaign": len(selected) == len(records),
        "planned_runs": len(records),
        "selected_runs": len(selected),
        "run_status": "running",
        "runs": [],
    }
    report_path = output_dir / "campaign_report.json"
    write_json_atomic(report_path, report)
    old_repo, old_output = pilot.dual_t4.REPO, pilot.dual_t4.OUTPUT
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
    try:
        prefix_flag = False if no_gpu else pilot._supports_no_prefix_caching_flag()
        for index, (manifest, cfg) in enumerate(manifests):
            print(
                json.dumps(
                    {
                        "event": "run_started",
                        "phase": phase,
                        "index": index + 1,
                        "total": len(manifests),
                        "manifest": manifest.name,
                    }
                ),
                flush=True,
            )
            services, gateway, launches, bindings = {}, None, {}, {}
            sampler = None
            logs = output_dir / "service-logs" / f"run-{index:03d}"
            logs.mkdir(parents=True)
            pilot.dual_t4.REPO, pilot.dual_t4.OUTPUT = ROOT, logs
            port = pilot._find_free_port() if no_gpu else 8080
            url = f"http://127.0.0.1:{port}"
            try:
                if not no_gpu:
                    if gpu_telemetry:
                        sampler = GpuTelemetry(logs / "gpu_telemetry.csv")
                    if prefetched_model:
                        env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
                    # Restart every replica before each run; use first-start serialization
                    # to avoid concurrent model-download races on an empty HF cache.
                    for gpu in (0, 1):
                        services[gpu], launches[gpu] = pilot._start_vllm_replica(
                            gpu,
                            env,
                            cfg.model.id,
                            disable_prefix_caching=prefix_flag,
                            model_revision=model_revision,
                        )
                        if index == 0 and gpu == 0 and not prefetched_model:
                            import httpx

                            with httpx.Client(timeout=4, trust_env=False) as client:
                                pilot.dual_t4.wait_for(
                                    "worker-vllm-0",
                                    lambda: pilot.dual_t4.replica_ready(client, 0, cfg.model.id),
                                    480,
                                    (services[0],),
                                )
                    pilot._wait_both_replicas_ready(services, cfg.model.id)
                    if sampler is not None and sampler.process.poll() is not None:
                        raise RuntimeError("GPU telemetry failed; see gpu_telemetry.csv")
                    bindings = pilot.dual_t4.capture_processes("initial", services)
                pilot.require_unused_port(port)
                gateway_env = {
                    **env,
                    "MODEL_NAME": cfg.model.id,
                    "ROUTING_POLICY": cfg.policy.name,
                    "NO_GPU": "1" if no_gpu else "0",
                    "WORKER_MODE": "mock" if no_gpu else "live",
                    "MOCK_WORKER_COUNT": "2",
                    "HEALTH_CHECK_INTERVAL_S": "0",
                }
                if no_gpu:
                    expected = {
                        f"worker-mock-{i}": (f"mock://worker-mock-{i}", cfg.model.id, "HEALTHY")
                        for i in range(2)
                    }
                else:
                    gateway_env["WORKER_ENDPOINTS"] = ",".join(
                        f"{wid}={endpoint}"
                        for wid, endpoint in zip(pilot.dual_t4.IDS, pilot.dual_t4.ENDPOINTS)
                    )
                    expected = {
                        wid: (endpoint, cfg.model.id, "HEALTHY")
                        for wid, endpoint in zip(pilot.dual_t4.IDS, pilot.dual_t4.ENDPOINTS)
                    }
                command = [
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
                ]
                gateway = (
                    pilot._LocalMockGatewayProc(command, logs / "gateway.log", gateway_env)
                    if no_gpu
                    else pilot.dual_t4.Service(command, "gateway.log", gateway_env)
                )
                pilot._wait_gateway_ready(gateway, url, expected, tuple(services.values()))
                runner = ExperimentRunner(
                    runs_root=output_dir / "runs",
                    run_context={
                        "protocol_sha256": protocol_hash,
                        "test_mode": report["test_mode"],
                        "experiment_phase": phase,
                        "campaign_run_index": index,
                        "model_revision": model_revision,
                        "gpu_telemetry_interval_s": 1 if gpu_telemetry else None,
                        "cache_control": {
                            "workers_restarted_between_runs": not no_gpu,
                            "fresh_mock_gateway": no_gpu,
                            "prefix_cache_disabled": prefix_flag,
                            "worker_startup": "fresh_mock_gateway"
                            if no_gpu
                            else "parallel_prefetched"
                            if prefetched_model
                            else "legacy_first_start_serialization",
                            "model_cache_offline": prefetched_model,
                        },
                        "worker_runtime_args": {str(i): v["args"] for i, v in launches.items()},
                        "worker_gpu_bindings": bindings,
                    },
                )
                result = asyncio.run(runner.run_from_manifest(manifest, gateway_url_override=url))
                report["runs"].append(
                    {
                        "manifest": str(manifest),
                        "run_id": result.run_id,
                        "run_dir": str(result.run_dir),
                        "run_status": result.run_status,
                        "failure_type": result.failure_type,
                    }
                )
                write_json_atomic(report_path, report)
                print(
                    json.dumps(
                        {
                            "event": "run_finished",
                            "phase": phase,
                            "index": index + 1,
                            "total": len(manifests),
                            "run_id": result.run_id,
                            "status": result.run_status,
                            "failure_type": result.failure_type,
                        }
                    ),
                    flush=True,
                )
                if not result.succeeded:
                    report["run_status"] = "failed"
                    break
            finally:
                try:
                    if gateway is not None:
                        gateway.stop()
                    for service in services.values():
                        service.stop()
                    if not no_gpu:
                        for service in services.values():
                            if pilot.dual_t4.group_pids(service.process.pid):
                                raise RuntimeError("Owned worker processes did not terminate")
                        for worker_port in (8000, 8001):
                            pilot._wait_port_free(worker_port)
                finally:
                    if sampler is not None:
                        sampler.stop()
        if report["run_status"] == "running":
            report["run_status"] = "completed"
    except BaseException as exc:
        report["run_status"] = "failed"
        report["failure_type"] = type(exc).__name__
        raise
    finally:
        pilot.dual_t4.REPO, pilot.dual_t4.OUTPUT = old_repo, old_output
        write_json_atomic(report_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--no-gpu", action="store_true")
    mode.add_argument("--dual-t4", action="store_true")
    parser.add_argument("--revision")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--prefetched-model-revision")
    parser.add_argument("--gpu-telemetry", action="store_true")
    args = parser.parse_args(argv)
    report = execute_campaign(
        args.campaign,
        output_dir=args.output_dir,
        no_gpu=args.no_gpu,
        revision=args.revision,
        limit=args.limit,
        calibration=args.calibration,
        prefetched_model=args.prefetched_model_revision is not None,
        model_revision=args.prefetched_model_revision,
        gpu_telemetry=args.gpu_telemetry,
    )
    print(json.dumps(report, indent=2))
    return 0 if report["run_status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
