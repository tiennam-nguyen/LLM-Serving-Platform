"""Run Week 7 reproducible experiment pipeline on two harness-owned T4 replicas."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
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
from infra.scripts.verify_week7_e2e import (  # noqa: E402
    REQUIRED_ARTIFACT_FILES,
    validate_run_artifacts,
)
from llm_serving_platform.benchmark.workload import load_workload  # noqa: E402

LIVE_PHASES = (
    ("round_robin", "experiments/manifests/w7_t4_round_robin.json"),
    ("least_loaded", "experiments/manifests/w7_t4_least_loaded.json"),
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _wait_gateway_ready(gateway: Any, services: dict[int, Any], model: str, policy: str) -> None:
    import httpx

    with httpx.Client(timeout=4.0, trust_env=False) as client:

        def workers_ready():
            try:
                health = client.get(dual_t4.GATEWAY + "/health")
                if health.status_code != 200:
                    return None
                resp = client.get(dual_t4.GATEWAY + "/control/v1/workers")
                resp.raise_for_status()
                workers = resp.json()
            except (httpx.HTTPError, ValueError):
                return None
            expected = {
                worker_id: (endpoint, model, "HEALTHY")
                for worker_id, endpoint in zip(dual_t4.IDS, dual_t4.ENDPOINTS)
            }
            actual = {
                row["worker_id"]: (row["endpoint"], row["model"], row["status"])
                for row in workers
            }
            return workers if actual == expected else None

        dual_t4.wait_for(
            f"Gateway {policy} with two HEALTHY replicas",
            workers_ready,
            60,
            (gateway, *services.values()),
        )


def _verify_zero_active_workers() -> list[dict[str, Any]]:
    import httpx

    with httpx.Client(timeout=4.0, trust_env=False) as client:
        resp = client.get(dual_t4.GATEWAY + "/control/v1/workers")
        resp.raise_for_status()
        workers = resp.json()
    dual_t4.require(
        all(row.get("active_requests") == 0 for row in workers),
        f"Expected zero active requests after run, got {workers}",
    )
    return workers


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


def _collect_prompt_strings(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    workload_path = ROOT / manifest["workload"]["path"]
    entries = load_workload(workload_path, min_entries=1)
    prompts: list[str] = []
    for entry in entries:
        for msg in entry.messages:
            if msg.get("content"):
                prompts.append(msg["content"])
    return prompts


def run_live_experiments(args: argparse.Namespace, inventory: list[str]) -> None:
    import httpx

    torch = dual_t4.torch_info()
    dual_t4.require(
        len(torch["devices"]) == 2 and all("T4" in name for name in torch["devices"]),
        f"Expected exactly two CUDA-visible T4 GPUs, got {torch['devices']}",
    )

    branch = dual_t4.command("git", "branch", "--show-current", cwd=ROOT, capture=True)
    environment: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": args.revision,
        "git_dirty": False,
        "branch": branch,
        "python": platform.python_version(),
        "pytorch": torch["torch"],
        "cuda": torch["cuda"],
        "packages": _installed_versions(),
        "gpu_inventory": inventory,
        "model": args.model,
        "worker_endpoints": dict(zip(dual_t4.IDS, dual_t4.ENDPOINTS)),
        "gateway_endpoint": dual_t4.GATEWAY,
        "policies": [policy for policy, _ in LIVE_PHASES],
        "test_mode": "live_gpu",
        "dual_t4_topology_verified": False,
        "tensor_parallel_size": 1,
        "worker_gpu_bindings": {},
        "scope": (
            "Week 7 live dual-T4 observability and experiment artifact pipeline verification; "
            "not a policy performance comparison."
        ),
    }

    services: dict[int, Any] = {}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}

    def capture_bindings(label: str) -> None:
        binding = dual_t4.capture_processes(label, services)
        environment["worker_gpu_bindings"][label] = {
            worker_id: {
                "gpu_index": record["gpu_index"],
                "gpu_uuid": record["gpu_uuid"],
                "owned_compute_processes": len(record["compute_processes"]),
            }
            for worker_id, record in binding.items()
        }

    def start_worker(index: int) -> None:
        require_unused_port(8000 + index)
        service, _ = dual_t4.start_replica(index, env, args.model)
        services[index] = service
        with httpx.Client(timeout=4.0, trust_env=False) as client:
            dual_t4.wait_for(
                dual_t4.IDS[index],
                lambda: dual_t4.replica_ready(client, index, args.model),
                480,
                tuple(services.values()),
            )

    validation: dict[str, Any] = {
        "passed": False,
        "git_revision": args.revision,
        "model": args.model,
        "dual_t4_topology_verified": False,
        "phases": {},
    }

    try:
        for index in range(2):
            start_worker(index)
        capture_bindings("initial")
        environment["dual_t4_topology_verified"] = True
        validation["dual_t4_topology_verified"] = True
        _write_json(args.output_dir / "environment.json", environment)

        with tempfile.TemporaryDirectory(prefix="w7-live-runs-") as runs_tmp:
            runs_root = Path(runs_tmp)
            for phase_num, (policy, rel_manifest) in enumerate(LIVE_PHASES, start=1):
                manifest_path = ROOT / rel_manifest
                manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
                dual_t4.require(
                    manifest_data["model"]["id"] == args.model,
                    f"Manifest {rel_manifest} model {manifest_data['model']['id']!r} "
                    f"!= launcher model {args.model!r}",
                )
                require_unused_port(8080)
                gateway_env = {
                    **env,
                    "MODEL_NAME": args.model,
                    "WORKER_ENDPOINTS": ",".join(
                        f"{key}={url}" for key, url in zip(dual_t4.IDS, dual_t4.ENDPOINTS)
                    ),
                    "ROUTING_POLICY": policy,
                    "HEALTH_CHECK_INTERVAL_S": "2",
                    "HEALTH_CHECK_TIMEOUT_S": "3",
                    "NO_GPU": "0",
                    "WORKER_MODE": "live",
                }
                gateway = dual_t4.Service(
                    [
                        sys.executable,
                        "-m",
                        "uvicorn",
                        "llm_serving_platform.gateway.main:app",
                        "--host",
                        "127.0.0.1",
                        "--port",
                        "8080",
                    ],
                    f"gateway-{phase_num}-{policy}.log",
                    gateway_env,
                )
                try:
                    _wait_gateway_ready(gateway, services, args.model, policy)
                    proc = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "llm_serving_platform.benchmark",
                            "run",
                            "--manifest",
                            str(manifest_path),
                            "--output-dir",
                            str(runs_root),
                            "--gateway-url",
                            dual_t4.GATEWAY,
                        ],
                        cwd=str(ROOT),
                        env=env,
                        capture_output=True,
                        text=True,
                        timeout=1200,
                        check=False,
                    )
                    if proc.returncode != 0:
                        raise RuntimeError(
                            f"Week 7 runner failed for {policy} (exit {proc.returncode}):\n"
                            f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
                        )
                    cli_out = json.loads(proc.stdout)
                    src_run_dir = Path(cli_out["run_dir"])
                    dst_phase_dir = args.output_dir / policy
                    dst_phase_dir.mkdir(parents=True, exist_ok=True)
                    for filename in REQUIRED_ARTIFACT_FILES:
                        shutil.copy2(src_run_dir / filename, dst_phase_dir / filename)

                    final_workers = _verify_zero_active_workers()
                    prompts = _collect_prompt_strings(manifest_path)
                    phase_validation = validate_run_artifacts(
                        dst_phase_dir,
                        expected_policy=policy,
                        expected_planned=manifest_data["workload"]["request_count"],
                        expected_status="completed",
                        expected_revision=args.revision,
                        forbidden_substrings=prompts,
                        require_clean_git=True,
                    )
                    phase_validation["final_active_requests"] = {
                        w["worker_id"]: w["active_requests"] for w in final_workers
                    }
                    validation["phases"][policy] = phase_validation
                    print(
                        f"WEEK7 PHASE {policy} PASSED: run_id={phase_validation['run_id']} "
                        f"requests={phase_validation['requests']} "
                        f"workers={phase_validation['worker_distribution']}",
                        flush=True,
                    )
                finally:
                    gateway.stop()

        capture_bindings("final")
        validation["passed"] = True
        _write_json(args.output_dir / "environment.json", environment)
        _write_json(args.output_dir / "validation.json", validation)
        print("WEEK 7 DUAL-T4 EVIDENCE PASSED", flush=True)
    except Exception as exc:
        validation["failure"] = f"{type(exc).__name__}: {exc}"
        _write_json(args.output_dir / "environment.json", environment)
        _write_json(args.output_dir / "validation.json", validation)
        raise
    finally:
        errors: list[BaseException] = []
        for service in list(services.values()):
            try:
                service.stop()
            except BaseException as exc:
                errors.append(exc)
        if errors:
            raise errors[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", required=True, help="Exact tested commit SHA of this checkout"
    )
    parser.add_argument("--model", required=True, help="Model ID served by both replicas")
    parser.add_argument("--bootstrap", action="store_true", help="Install the serving environment")
    parser.add_argument("--vllm-version", help="Pin the vLLM wheel when bootstrapping")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/week7/live"))
    args = parser.parse_args()

    dual_t4.REPO = ROOT
    installed_revision = dual_t4.command("git", "rev-parse", "HEAD", cwd=ROOT, capture=True)
    dual_t4.require(installed_revision == args.revision, "Checkout revision does not match")
    dual_t4.require(
        not dual_t4.command("git", "status", "--porcelain", cwd=ROOT, capture=True),
        "Expected a clean source checkout",
    )

    inventory = checked_inventory()

    if args.bootstrap:
        python = dual_t4.serving_python()
        requirement = f"vllm=={args.vllm_version}" if args.vllm_version else "vllm"
        dual_t4.install(python, vllm_requirement=requirement)
        subprocess_args = [
            python,
            str(Path(__file__).resolve()),
            "--revision",
            args.revision,
            "--model",
            args.model,
            "--output-dir",
            str(args.output_dir),
        ]
        dual_t4.command(*subprocess_args, timeout=3600, cwd=ROOT)
        return

    dual_t4.OUTPUT = args.output_dir.resolve().parent / "service-logs"
    dual_t4.OUTPUT.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_live_experiments(args, inventory)


if __name__ == "__main__":
    main()
