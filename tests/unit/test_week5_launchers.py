"""Check lab launcher arguments and fail-fast behavior without starting GPU processes."""

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def bash_executable():
    bash = (
        "C:/Program Files/Git/bin/bash.exe" if sys.platform == "win32" else shutil.which("bash")
    )
    if not bash or not Path(bash).exists():
        pytest.skip("Bash required for lab launcher contracts")
    return bash


def launch(tmp_path, script, config):
    bash = bash_executable()
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    python = fake_bin / "python"
    python.write_text(
        '#!/usr/bin/env bash\n'
        'if [[ "${1:-}" == -m ]]; then\n'
        '  exec "$REAL_PYTHON" -c \'import json,os,sys; '
        'print(json.dumps({"argv":sys.argv[1:], "env":{k:os.getenv(k) for k in '
        '("WORKER_ENDPOINTS","ROUTING_POLICY","NO_GPU","WORKER_MODE",'
        '"PYTHONPATH","CUDA_VISIBLE_DEVICES")}}))\' "$@"\n'
        'fi\nexec "$REAL_PYTHON" "$@"\n',
        encoding="utf-8", newline="\n",
    )
    python.chmod(0o755)
    env_file = tmp_path / "lab.env"
    env_file.write_text(config, encoding="utf-8", newline="\n")
    env = os.environ.copy()
    for key in ("WORKER_ENDPOINTS", "HOST_A_IP", "HOST_B_IP", "BASH_ENV"):
        env.pop(key, None)
    env["REAL_PYTHON"] = Path(sys.executable).as_posix()
    env["PATH"] = str(fake_bin) + os.pathsep + env["PATH"]
    # Deliberately inherit conflicting mock settings: Gateway must override them.
    env["NO_GPU"], env["WORKER_MODE"] = "1", "mock"
    return subprocess.run(
        [bash, str(ROOT / "infra/scripts" / script), env_file.as_posix()],
        env=env, text=True, capture_output=True, timeout=15, cwd=ROOT,
    )


def test_worker_launcher_arguments(tmp_path):
    result = launch(tmp_path, "start_worker_lab.sh", "MAX_MODEL_LEN=1024\n")
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout.splitlines()[-1])
    args = captured["argv"]
    assert args[:2] == ["-m", "vllm.entrypoints.openai.api_server"]
    for option, value in {
        "--model": "Qwen/Qwen2.5-0.5B-Instruct", "--dtype": "float16",
        "--gpu-memory-utilization": "0.6", "--max-model-len": "1024",
        "--swap-space": "1", "--host": "0.0.0.0", "--port": "8000",
        "--tensor-parallel-size": "1",
    }.items():
        assert args[args.index(option) + 1] == value
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == "0"


def test_gateway_explicit_endpoints_and_live_mode(tmp_path):
    endpoints = "worker-vllm-a=http://a.test:8000,worker-vllm-b=http://b.test:8000"
    result = launch(tmp_path, "start_gateway_lab.sh", f"WORKER_ENDPOINTS={endpoints}\n")
    assert result.returncode == 0, result.stderr
    captured = json.loads(result.stdout.splitlines()[-1])
    assert captured["argv"] == [
        "-m", "uvicorn", "llm_serving_platform.gateway.main:app",
        "--host", "0.0.0.0", "--port", "8080",
    ]
    assert captured["env"]["WORKER_ENDPOINTS"] == endpoints
    assert captured["env"]["NO_GPU"] == "0"
    assert captured["env"]["WORKER_MODE"] == "live"
    assert captured["env"]["ROUTING_POLICY"] == "round_robin"
    assert "/src" in captured["env"]["PYTHONPATH"]


@pytest.mark.parametrize("config", [
    "HOST_A_IP=REPLACE_WITH_HOST_A_LAN_IP\nHOST_B_IP=REPLACE_WITH_HOST_B_LAN_IP\n",
    "WORKER_ENDPOINTS=a=http://a.test:8000,a=http://b.test:8000\n",
    "WORKER_ENDPOINTS=a=http://a.test:8000,b=http://a.test:8000\n",
    "WORKER_ENDPOINTS=a=http://a.test:8000,b=http://b.test:8000\nHEALTH_CHECK_INTERVAL_S=0\n",
    "WORKER_ENDPOINTS=a=http://sensitive:secret@a.test:8000,b=http://b.test:8000\n",
])
def test_gateway_rejects_bad_configuration_without_launch(tmp_path, config):
    result = launch(tmp_path, "start_gateway_lab.sh", config)
    assert result.returncode != 0
    assert '"argv"' not in result.stdout
    assert "sensitive" not in result.stdout + result.stderr
    assert "secret@" not in result.stdout + result.stderr


def test_preflight_missing_tools_reports_gpu_failure():
    env = os.environ.copy()
    env["PATH"] = ""
    env.pop("BASH_ENV", None)
    result = subprocess.run(
        [bash_executable(), str(ROOT / "infra/scripts/week5_lab_preflight.sh")],
        env=env, text=True, capture_output=True, timeout=15,
    )
    assert result.returncode != 0
    assert "WARN: uv unavailable" in result.stdout
    assert "FAIL: nvidia-smi unavailable" in result.stdout
    assert "FAIL GPU/runtime: 1" in result.stdout
