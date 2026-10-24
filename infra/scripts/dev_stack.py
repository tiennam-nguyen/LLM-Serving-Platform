"""Interactive User Stack Launcher (opens Gateway on :8080 and Web UI on :3000 for browser testing).

Usage:
  1. Default Mode (Live GPU - connects Gateway :8080 to live vLLM GPU worker on :8000):
     python infra/scripts/dev_stack.py

  2. No-GPU Mode (--no-gpu - runs real Gateway :8080 with MockInferenceWorker + Web UI :3000):
     python infra/scripts/dev_stack.py --no-gpu

  3. Gateway-Only Mode (if you want to start only the Gateway on :8080):
     python infra/scripts/dev_stack.py --gateway-only [--no-gpu]
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_serving_platform.common.config import load_dotenv_file  # noqa: E402

load_dotenv_file()


def _wait_http_ready(url: str, timeout_s: float = 20.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = httpx.get(url, timeout=1.5)
            if resp.status_code in {200, 503}:
                return True
        except httpx.HTTPError:
            pass
        time.sleep(0.4)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Launch Gateway (:8080) and Next.js Web UI (:3000) for interactive browser testing."
        )
    )
    parser.add_argument(
        "--no-gpu",
        "--mock",
        dest="no_gpu",
        action="store_true",
        help="Run Gateway with in-memory MockInferenceWorker (no GPU or vLLM required).",
    )
    parser.add_argument(
        "--gateway-port",
        type=int,
        default=int(os.getenv("GATEWAY_PORT", "8080")),
        help="Port for the FastAPI Gateway (default: 8080).",
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=3000,
        help="Port for the Next.js Web UI (default: 3000).",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"),
        help="vLLM worker URL when running in default Live GPU mode (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="Default model name configured for Gateway and Web UI.",
    )
    parser.add_argument(
        "--gateway-only",
        action="store_true",
        help="Start only the FastAPI Gateway (without starting Next.js dev server).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help=(
            "Start the requested service(s), verify HTTP readiness + 1 E2E trace, and exit cleanly."
        ),
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["GATEWAY_HOST"] = "0.0.0.0"
    env["GATEWAY_PORT"] = str(args.gateway_port)
    env["VLLM_BASE_URL"] = args.vllm_url
    env["MODEL_NAME"] = args.model
    env["NO_GPU"] = "1" if args.no_gpu else "0"
    env["CORS_ALLOWED_ORIGINS"] = (
        f"http://localhost:{args.web_port},http://127.0.0.1:{args.web_port},"
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001"
    )

    mode_str = (
        "NO-GPU (MockInferenceWorker)" if args.no_gpu else f"LIVE-GPU (vLLM @ {args.vllm_url})"
    )
    print("=" * 84)
    print(f"[*] Starting Interactive User Stack — Mode: {mode_str}")
    print("=" * 84)

    if not args.no_gpu:
        try:
            r = httpx.get(f"{args.vllm_url.rstrip('/')}/health", timeout=2.0)
            if r.status_code == 200:
                print(f"[+] Live vLLM GPU worker is ONLINE at {args.vllm_url}")
            else:
                print(f"[!] Warning: vLLM at {args.vllm_url} returned HTTP {r.status_code}")
        except httpx.HTTPError:
            print(
                f"[!] Warning: Live vLLM worker is not reachable at {args.vllm_url}.\n"
                "    -> Start vLLM first (`bash infra/scripts/start_worker_wsl.sh`), OR\n"
                "    -> Run with `--no-gpu` (`python infra/scripts/dev_stack.py --no-gpu`) "
                "to test on browser without a GPU!"
            )

    processes: list[subprocess.Popen[bytes]] = []
    try:
        gw_cmd = [
            sys.executable,
            "-m",
            "uvicorn",
            "llm_serving_platform.gateway.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            str(args.gateway_port),
        ]
        print(f"[*] Launching FastAPI Gateway on http://127.0.0.1:{args.gateway_port} ...")
        gw_proc = subprocess.Popen(gw_cmd, cwd=str(REPO_ROOT), env=env)
        processes.append(gw_proc)

        gw_health_url = f"http://127.0.0.1:{args.gateway_port}/health"
        if not _wait_http_ready(gw_health_url, timeout_s=15.0):
            print("[-] FastAPI Gateway failed to become ready.", file=sys.stderr)
            return 1
        print(f"[+] FastAPI Gateway ready at http://127.0.0.1:{args.gateway_port}")

        if not args.gateway_only:
            npm_bin = shutil.which("npm") or "npm"
            web_env = env.copy()
            web_env["NEXT_PUBLIC_DEFAULT_MODEL"] = args.model
            web_env["NEXT_PUBLIC_GATEWAY_URL"] = f"http://127.0.0.1:{args.gateway_port}"
            web_env["GATEWAY_URL"] = f"http://127.0.0.1:{args.gateway_port}"
            web_cmd = [npm_bin, "run", "dev", "--", "--port", str(args.web_port)]
            print(f"[*] Launching Next.js Web UI on http://127.0.0.1:{args.web_port} ...")
            web_proc = subprocess.Popen(web_cmd, cwd=str(REPO_ROOT / "apps" / "web"), env=web_env)
            processes.append(web_proc)

        print("\n" + "=" * 84)
        print(">>> STACK IS LIVE FOR INTERACTIVE BROWSER TESTING <<<")
        print(f"  - Mode        : {mode_str}")
        if not args.gateway_only:
            print(f"  - Web UI URL  : http://127.0.0.1:{args.web_port}")
        print(f"  - Gateway URL : http://127.0.0.1:{args.gateway_port}/health")
        print(f"  - Traces URL  : http://127.0.0.1:{args.gateway_port}/v1/traces")
        print("  Press Ctrl+C to stop all services.")
        print("=" * 84 + "\n")

        if args.smoke_test:
            import asyncio

            from infra.scripts.verify_week3_e2e import run_week3_e2e_check

            evidence = asyncio.run(
                run_week3_e2e_check(
                    gateway_url=f"http://127.0.0.1:{args.gateway_port}",
                    model=args.model,
                    prompt="Kiểm tra kết nối trực tiếp tới Gateway port thật.",
                    no_gpu=False,
                )
            )
            worker_hdr = evidence["client_observed"]["worker_id_header"]
            print(
                f"[+] Live HTTP Smoke Test against 127.0.0.1:{args.gateway_port} PASSED! "
                f"(Request ID: {evidence['request_id']}, Worker: {worker_hdr})"
            )
            return 0 if evidence["passed"] else 1

        while True:
            for p in processes:
                ret = p.poll()
                if ret is not None:
                    print(f"[!] Process PID {p.pid} exited with code {ret}.")
                    return ret
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n[*] Shutting down interactive stack...")
        return 0
    finally:
        for p in processes:
            if p.poll() is None:
                p.terminate()
        for p in processes:
            try:
                p.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                p.kill()


if __name__ == "__main__":
    sys.exit(main())
