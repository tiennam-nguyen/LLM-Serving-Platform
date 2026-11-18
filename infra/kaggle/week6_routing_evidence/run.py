"""Run Week 6 with two harness-owned T4 replicas and fresh Gateway policy phases."""

import argparse
import asyncio
import os
import socket
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from infra.kaggle.dual_t4_validation import run as dual_t4  # noqa: E402


def checked_inventory():
    dual_t4.command("nvidia-smi")
    rows = dual_t4.gpu_query("name,memory.total,driver_version").splitlines()
    dual_t4.require(
        len(rows) == 2 and all("T4" in row.split(",")[0] for row in rows),
        "Expected exactly two physical T4 GPUs",
    )
    return rows


def require_unused_port(port):
    with socket.socket() as probe:
        probe.settimeout(1)
        dual_t4.require(probe.connect_ex(("127.0.0.1", port)) != 0, f"Port {port} is in use")


async def experiment(args, inventory):
    import httpx

    from infra.scripts.verify_week6_e2e import (
        DEFAULT_LEVELS,
        live_phase,
        run_week6_e2e,
    )

    torch = dual_t4.torch_info()
    dual_t4.require(
        len(torch["devices"]) == 2 and all("T4" in name for name in torch["devices"]),
        "Expected exactly two CUDA-visible T4 GPUs",
    )
    hardware = {
        "dual_t4_topology_verified": False,
        "gpu_inventory": inventory,
        "pytorch": torch["torch"],
        "cuda": torch["cuda"],
        "tensor_parallel_size": 1,
        "worker_gpu_bindings": {},
    }
    services = {}
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}

    def capture_bindings(label):
        binding = dual_t4.capture_processes(label, services)
        hardware["worker_gpu_bindings"][label] = {
            worker_id: {
                "gpu_index": record["gpu_index"],
                "owned_compute_processes": len(record["compute_processes"]),
            }
            for worker_id, record in binding.items()
        }

    def start_worker(index):
        require_unused_port(8000 + index)
        service, _ = dual_t4.start_replica(index, env, args.model)
        services[index] = service
        with httpx.Client(timeout=4, trust_env=False) as client:
            dual_t4.wait_for(
                dual_t4.IDS[index],
                lambda: dual_t4.replica_ready(client, index, args.model),
                480,
                tuple(services.values()),
            )

    async def set_worker_health(worker_id, healthy):
        index = dual_t4.IDS.index(worker_id)
        if healthy:
            await asyncio.to_thread(start_worker, index)
            await asyncio.to_thread(capture_bindings, "recovered")
            return

        service = services[index]
        await asyncio.to_thread(service.stop)
        dual_t4.require(not dual_t4.group_pids(service.process.pid), "Worker group still running")
        del services[index]
        await asyncio.to_thread(capture_bindings, "worker0_down")

    @asynccontextmanager
    async def fresh_phase(policy):
        nonlocal phase_number
        require_unused_port(8080)
        phase_number += 1
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
            f"gateway-{phase_number}-{policy}.log",
            gateway_env,
        )
        try:

            def wait_for_gateway():
                with httpx.Client(timeout=4, trust_env=False) as client:

                    def workers_ready():
                        try:
                            response = client.get(dual_t4.GATEWAY + "/control/v1/workers")
                            response.raise_for_status()
                            workers = response.json()
                        except (httpx.HTTPError, ValueError):
                            return None
                        expected = {
                            worker_id: (endpoint, args.model, "HEALTHY")
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

            await asyncio.to_thread(wait_for_gateway)
            async with live_phase(
                policy,
                args.model,
                dual_t4.GATEWAY,
                set_worker_health,
            ) as phase:
                yield phase
        finally:
            await asyncio.to_thread(gateway.stop)

    phase_number = 0
    try:
        for index in range(2):
            await asyncio.to_thread(start_worker, index)
        await asyncio.to_thread(capture_bindings, "initial")
        hardware["dual_t4_topology_verified"] = True
        levels = (*DEFAULT_LEVELS, 16) if args.extended else DEFAULT_LEVELS
        validation, _ = await run_week6_e2e(
            no_gpu=False,
            model=args.model,
            gateway_url=dual_t4.GATEWAY,
            levels=levels,
            output_dir=args.output_dir,
            phase_factory=fresh_phase,
            hardware=hardware,
        )
        dual_t4.require(validation["passed"], "Week 6 live gate did not pass. See validation.json.")
    finally:
        stopped = await asyncio.gather(
            *(asyncio.to_thread(service.stop) for service in services.values()),
            return_exceptions=True,
        )
        for result in stopped:
            if isinstance(result, BaseException):
                raise result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--revision", required=True, help="Exact tested commit SHA of this checkout"
    )
    parser.add_argument("--model", required=True, help="Model ID served by both replicas")
    parser.add_argument("--bootstrap", action="store_true", help="Install the serving environment")
    parser.add_argument("--extended", action="store_true", help="Add a concurrency-16 burst")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/week6/live"))
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
        dual_t4.install(python)
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
        if args.extended:
            subprocess_args.append("--extended")
        dual_t4.command(*subprocess_args, timeout=3600, cwd=ROOT)
        return

    # Service stdout and raw process diagnostics remain separate from request evidence.
    dual_t4.OUTPUT = args.output_dir.resolve().parent / "service-logs"
    dual_t4.OUTPUT.mkdir(parents=True, exist_ok=True)
    asyncio.run(experiment(args, inventory))


if __name__ == "__main__":
    main()
