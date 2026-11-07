"""Validate two independent replicas on one Kaggle node, never a LAN gate."""

import asyncio
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

REPOSITORY = "https://github.com/tiennam-nguyen/LLM-Serving-Platform.git"
REVISION = "db60212fadeaf71f8c0092a65dffd6c5629fa1d0"
MODEL = "Qwen/Qwen2.5-3B-Instruct"
VLLM_REQUIREMENT = "vllm"
MEMORY = "0.85"
CONTEXT = "4096"
ACCELERATOR = "NvidiaTeslaT4"
OUTPUT = Path("/kaggle/working")
REPO = Path("/tmp/llm-serving-dual-t4")
GATEWAY = "http://127.0.0.1:8080"
IDS = ("worker-vllm-0", "worker-vllm-1")
ENDPOINTS = ("http://127.0.0.1:8000", "http://127.0.0.1:8001")
MILESTONES = (
    "arrival",
    "routing_start",
    "routing_end",
    "worker_start",
    "first_token",
    "last_token",
)
TORCH_PROBE = """
import json, torch
print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
    'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}))
"""


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def save(name, value):
    (OUTPUT / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def command(*args, timeout=120, cwd=None, env=None, capture=False):
    result = subprocess.run(
        args,
        check=True,
        timeout=timeout,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE if capture else None,
    )
    return result.stdout.strip() if capture else None


def gpu_query(fields, kind="gpu"):
    return command(
        "nvidia-smi", f"--query-{kind}={fields}", "--format=csv,noheader,nounits", capture=True
    )


def torch_info(env=None):
    return json.loads(
        command(sys.executable, "-c", TORCH_PROBE, env=env, capture=True).splitlines()[-1]
    )


def serving_python():
    if (3, 11) <= sys.version_info[:2] < (3, 13):
        return sys.executable
    # Kaggle v1 ran 3.13; honor the project's interpreter bound in an isolated venv.
    command(sys.executable, "-m", "pip", "--isolated", "install", "--only-binary=:all:", "uv")
    path = "/tmp/llm-serving-python312"
    command("uv", "venv", "--python", "3.12", "--managed-python", "--seed", path, timeout=300)
    return f"{path}/bin/python"


def install(python):
    pip = (python, "-m", "pip", "--isolated")
    command(*pip, "install", "-e", ".[dev]", cwd=REPO, timeout=300)
    command(*pip, "index", "versions", "vllm", "--only-binary=:all:")
    command(*pip, "install", "--only-binary=:all:", VLLM_REQUIREMENT, timeout=1200)


class Service:
    def __init__(self, args, log_name, env):
        self.log = OUTPUT / log_name
        with self.log.open("ab") as output:
            self.process = subprocess.Popen(
                args,
                cwd=REPO,
                env=env,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

    def check_running(self):
        require(
            self.process.poll() is None, f"{self.log.name} exited with {self.process.returncode}"
        )

    def stop(self):
        # Descendants can still own the GPU after the HTTP parent exits.
        for sig, grace in ((signal.SIGTERM, 20), (signal.SIGKILL, 5)):
            try:
                os.killpg(self.process.pid, sig)
            except ProcessLookupError:
                break
            try:
                self.process.wait(timeout=grace)
                if not group_pids(self.process.pid):
                    break
            except subprocess.TimeoutExpired:
                continue
        self.process.wait(timeout=5)


def group_pids(group):
    pids = set()
    for path in Path("/proc").glob("[0-9]*/stat"):
        try:
            fields = path.read_text().rsplit(")", 1)[1].split()
            if int(fields[2]) == group and fields[0] != "Z":
                pids.add(int(path.parent.name))
        except (OSError, ValueError, IndexError):
            continue
    return pids


def wait_for(label, probe, seconds, services=()):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        for service in services:
            service.check_running()
        value = probe()
        if value is not None:
            print(f"READY {label}", flush=True)
            return value
        time.sleep(2)
    raise TimeoutError(f"Timed out waiting for {label}")


def capture_processes(phase, workers):
    devices = gpu_query("index,uuid,name,memory.total,memory.used")
    processes = gpu_query("gpu_uuid,pid,process_name,used_gpu_memory", "compute-apps")
    (OUTPUT / f"gpu_processes_{phase}.txt").write_text(
        "index,uuid,name,memory.total,memory.used (MiB)\n"
        + devices
        + "\n\ngpu_uuid,pid,process_name,used_gpu_memory (MiB)\n"
        + processes
        + "\n",
        encoding="utf-8",
    )
    uuids = {int(row[0]): row[1] for row in csv.reader(devices.splitlines(), skipinitialspace=True)}
    rows = list(csv.reader(processes.splitlines(), skipinitialspace=True))
    mapping = {}
    for index, service in workers.items():
        pids = group_pids(service.process.pid)
        owned = [row for row in rows if len(row) == 4 and row[1].isdigit() and int(row[1]) in pids]
        require(
            owned and {row[0] for row in owned} == {uuids[index]},
            f"Unproven GPU binding for {IDS[index]}",
        )
        mapping[IDS[index]] = {
            "gpu_index": index,
            "gpu_uuid": uuids[index],
            "server_pid": service.process.pid,
            "compute_processes": owned,
        }
    return mapping


def experiment(result, services):
    import httpx

    sys.path.insert(0, str(REPO / "src"))
    sys.path.insert(0, str(REPO))
    from infra.scripts.verify_week4_e2e import _matches_round_robin_cycle, _send_streaming_chat

    env = {**os.environ, "PYTHONPATH": str(REPO / "src"), "PYTHONUNBUFFERED": "1"}

    def start_worker(index):
        worker_env = {**env, "CUDA_VISIBLE_DEVICES": str(index), "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
        visible = torch_info(worker_env)
        require(
            len(visible["devices"]) == 1 and "T4" in visible["devices"][0],
            "Expected one visible T4 per replica",
        )
        args = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            MODEL,
            "--host",
            "127.0.0.1",
            "--port",
            str(8000 + index),
            "--tensor-parallel-size",
            "1",
            "--dtype",
            "float16",
            "--gpu-memory-utilization",
            MEMORY,
            "--max-model-len",
            CONTEXT,
        ]
        result["worker_launch"][IDS[index]] = {
            "CUDA_VISIBLE_DEVICES": str(index),
            "args": args[1:],
            "visible_devices": visible["devices"],
        }
        service = Service(args, f"worker{index}.log", worker_env)
        services.append(service)
        return service

    with httpx.Client(timeout=4, trust_env=False) as client:

        def json_ok(url):
            try:
                response = client.get(url)
                return response.json() if response.status_code == 200 else None
            except httpx.RequestError:
                return None

        def worker_ready(index):
            try:
                if client.get(ENDPOINTS[index] + "/health").status_code != 200:
                    return None
                models = json_ok(ENDPOINTS[index] + "/v1/models")
                return (
                    models if models and any(row["id"] == MODEL for row in models["data"]) else None
                )
            except httpx.RequestError:
                return None

        worker0 = start_worker(0)
        wait_for(IDS[0], lambda: worker_ready(0), 480, (worker0,))
        worker1 = start_worker(1)
        wait_for(IDS[1], lambda: worker_ready(1), 480, (worker0, worker1))
        result["worker_health"] = dict.fromkeys(IDS, 200)
        result["gpu_mapping_before"] = capture_processes("before", {0: worker0, 1: worker1})
        gateway_env = {
            **env,
            "MODEL_NAME": MODEL,
            "WORKER_ENDPOINTS": ",".join(f"{key}={url}" for key, url in zip(IDS, ENDPOINTS)),
            "ROUTING_POLICY": "round_robin",
            "HEALTH_CHECK_INTERVAL_S": "2",
            "HEALTH_CHECK_TIMEOUT_S": "3",
            "NO_GPU": "0",
            "WORKER_MODE": "live",
        }
        gateway = Service(
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
            "gateway.log",
            gateway_env,
        )
        services.append(gateway)
        wait_for("gateway health", lambda: json_ok(GATEWAY + "/health"), 60, (gateway,))

        def registry(expected):
            records = json_ok(GATEWAY + "/control/v1/workers")
            if not records or {row["worker_id"]: row["status"] for row in records} != expected:
                return None
            require(len(records) == 2, "Registry must contain exactly two workers")
            for row in records:
                require(
                    row["model"] == MODEL
                    and row["endpoint"] == ENDPOINTS[IDS.index(row["worker_id"])],
                    "Registry metadata mismatch",
                )
            return records

        healthy = dict.fromkeys(IDS, "HEALTHY")
        result["initial_workers"] = wait_for(
            "both workers HEALTHY", lambda: registry(healthy), 60, (gateway, worker0, worker1)
        )
        save("kaggle_dual_t4_validation.json", result)
        command(
            sys.executable,
            "infra/scripts/verify_week4_e2e.py",
            "--gateway-url",
            GATEWAY,
            "--model",
            MODEL,
            "--require-multi-worker",
            "--save-evidence",
            cwd=REPO,
            env=env,
            timeout=360,
        )
        baseline = json.loads(
            (REPO / "experiments/raw/week4_e2e_gpu_evidence.json").read_text(encoding="utf-8")
        )
        require(
            baseline["passed"] is True and baseline["mode"] == "live-gpu",
            "Live verifier did not pass",
        )
        save("week4_e2e_gpu_evidence.json", baseline)

        async def probe_phase(label, expected):
            requests = []
            async with httpx.AsyncClient(timeout=60, trust_env=False) as stream_client:
                for _ in range(4):
                    request_id = f"{label}-{uuid.uuid4().hex}"
                    response = await _send_streaming_chat(
                        stream_client, GATEWAY, MODEL, "Explain batching briefly.", request_id
                    )
                    response.pop("preview", None)
                    requests.append(response)
                    require(
                        response["status_code"] == 200
                        and response["chunks"] > 0
                        and response["request_id"] == request_id
                        and response["worker_id"] in expected,
                        f"Invalid stream: {request_id}",
                    )
                    trace = await stream_client.get(f"{GATEWAY}/v1/traces/{request_id}")
                    trace.raise_for_status()
                    row = trace.json()
                    response["trace_correlated"] = (
                        row["request_id"] == request_id
                        and row["worker_id"] == response["worker_id"]
                        and row["status"] == "completed"
                        and all(row["milestones"].get(key) for key in MILESTONES)
                    )
                    require(response["trace_correlated"], f"Trace mismatch: {request_id}")
            sequence = [row["worker_id"] for row in requests]
            require(
                _matches_round_robin_cycle(sequence, list(expected)),
                f"Unexpected {label} sequence: {sequence}",
            )
            result[label] = {"sequence": sequence, "requests": requests}
            save("kaggle_dual_t4_validation.json", result)
            print(label, sequence, flush=True)

        asyncio.run(probe_phase("baseline", IDS))
        worker0.stop()
        require(not group_pids(worker0.process.pid), "Worker 0 process group did not stop")
        down = {IDS[0]: "UNHEALTHY", IDS[1]: "HEALTHY"}
        result["worker0_down_registry"] = wait_for(
            "worker 0 excluded", lambda: registry(down), 45, (gateway, worker1)
        )
        result["gpu_mapping_worker0_down"] = capture_processes("worker0_down", {1: worker1})
        asyncio.run(probe_phase("failover", IDS[1:]))
        worker0 = start_worker(0)
        wait_for("restarted worker 0", lambda: worker_ready(0), 480, (gateway, worker0, worker1))
        result["recovered_registry"] = wait_for(
            "both workers recovered", lambda: registry(healthy), 45, (gateway, worker0, worker1)
        )
        result["gpu_mapping_recovered"] = capture_processes("recovered", {0: worker0, 1: worker1})
        asyncio.run(probe_phase("post_recovery", IDS))


def main():
    print("Same-node real-replica failover experiment; NOT two-host LAN validation", flush=True)
    environment = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "accelerator": ACCELERATOR,
        "repository": REPOSITORY,
        "git_revision": REVISION,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    result = {
        "passed": False,
        "environment": environment,
        "model": MODEL,
        "worker_launch": {},
        "scope": "same-node real-replica failover experiment; NOT two-host LAN validation",
    }
    try:
        command(sys.executable, "--version")
        command("nvidia-smi")
        environment["gpu_inventory"] = gpu_query("index,uuid,name,memory.total,driver_version")
        environment["before_install"] = torch_info()
        print(environment["before_install"], flush=True)
        save("kaggle_dual_t4_environment.json", environment)
        devices = environment["before_install"]["devices"]
        require(
            len(devices) == 2 and all("T4" in name for name in devices),
            "Expected exactly two CUDA-visible T4 GPUs",
        )
        command("git", "init", str(REPO))
        command("git", "fetch", "--depth", "1", REPOSITORY, REVISION, cwd=REPO)
        command("git", "checkout", "--detach", "FETCH_HEAD", cwd=REPO)
        require(
            command("git", "rev-parse", "HEAD", cwd=REPO, capture=True) == REVISION,
            "Wrong repository revision",
        )
        environment["git_status"] = command("git", "status", "--porcelain", cwd=REPO, capture=True)
        require(not environment["git_status"], "Expected clean checkout")
        python = serving_python()
        environment["serving_python"] = command(python, "--version", capture=True)
        print(environment["serving_python"], flush=True)
        save("kaggle_dual_t4_environment.json", environment)
        install(python)
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        save("kaggle_dual_t4_environment.json", environment)
        save("kaggle_dual_t4_validation.json", result)
        raise
    # Keep the Kaggle launcher intact; execute all serving/verification code with the venv.
    command(python, str(Path(__file__).resolve()), "--validate-installed", timeout=2200)


def validate_installed():
    environment = json.loads((OUTPUT / "kaggle_dual_t4_environment.json").read_text())
    result = {
        "passed": False,
        "environment": environment,
        "model": MODEL,
        "worker_launch": {},
        "scope": "same-node real-replica failover experiment; NOT two-host LAN validation",
    }
    services = []
    try:
        environment["installed"] = {
            name: importlib.metadata.version(name)
            for name in ("torch", "vllm", "transformers", "fastapi", "httpx")
        }
        environment["after_install"] = torch_info()
        print(environment["installed"], environment["after_install"], flush=True)
        save("kaggle_dual_t4_environment.json", environment)
        require(len(environment["after_install"]["devices"]) == 2, "CUDA lost after installation")
        experiment(result, services)
        result["passed"] = True
        print("DUAL T4 ACCEPTANCE PASSED", flush=True)
    except Exception as error:
        result["failure"] = f"{type(error).__name__}: {error}"
        for service in services:
            with service.log.open("rb") as log:
                log.seek(max(0, service.log.stat().st_size - 6000))
                print(
                    f"--- {service.log.name} tail ---\n{log.read().decode(errors='replace')}",
                    flush=True,
                )
        raise
    finally:
        save("kaggle_dual_t4_validation.json", result)
        for service in reversed(services):
            service.stop()


if __name__ == "__main__":
    if sys.argv[1:] == ["--validate-installed"]:
        validate_installed()
    else:
        main()
