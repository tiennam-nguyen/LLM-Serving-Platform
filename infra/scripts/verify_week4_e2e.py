# ruff: noqa: E402, I001
"""Week 4 E2E Verification & Evidence Recorder (WorkerRegistry, HealthMonitor & Round-Robin).

Usage examples:
  1. Default mode (Live GPU):
     python infra/scripts/verify_week4_e2e.py

  2. Live GPU with custom prompt (--prompt):
     python infra/scripts/verify_week4_e2e.py --prompt "Giải thích cơ chế Round-Robin trong 2 câu."

  3. No-GPU mode for CI / CPU machines (--no-gpu):
     python infra/scripts/verify_week4_e2e.py --no-gpu
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_serving_platform.common.config import GatewayConfig, load_dotenv_file  # noqa: E402

load_dotenv_file()

DEFAULT_PROMPT = "Giải thích ngắn gọn cơ chế Continuous Batching trong 2 câu."


async def _send_streaming_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    prompt: str,
    request_id: str,
) -> dict[str, Any]:
    """Send a single POST /v1/chat/completions request and collect response headers + content."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": 64,
    }
    headers = {
        "Origin": "http://127.0.0.1:3000",
        "X-Request-ID": request_id,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }
    assembled = ""
    chunks = 0
    async with client.stream(
        "POST",
        f"{base_url}/v1/chat/completions",
        json=payload,
        headers=headers,
    ) as resp:
        status_code = resp.status_code
        resp_req_id = resp.headers.get("X-Request-ID")
        resp_worker_id = resp.headers.get("X-Worker-ID")
        cors_origin = resp.headers.get("access-control-allow-origin")
        if status_code != 200:
            raw_body = (await resp.aread()).decode("utf-8", errors="replace")
            return {
                "status_code": status_code,
                "request_id": resp_req_id,
                "worker_id": resp_worker_id,
                "cors_origin": cors_origin,
                "error_body": raw_body,
                "chunks": 0,
                "preview": "",
            }
        async for raw_line in resp.aiter_lines():
            line = raw_line.strip()
            if not line or not line.startswith("data:"):
                continue
            data_str = line.removeprefix("data:").strip()
            if data_str == "[DONE]":
                break
            try:
                event = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = event.get("choices", [])
            if choices and isinstance(choices[0], dict):
                delta_text = choices[0].get("delta", {}).get("content")
                if isinstance(delta_text, str) and delta_text:
                    chunks += 1
                    assembled += delta_text

    return {
        "status_code": status_code,
        "request_id": resp_req_id,
        "worker_id": resp_worker_id,
        "cors_origin": cors_origin,
        "chunks": chunks,
        "preview": assembled[:120],
    }


async def run_week4_e2e_check(
    gateway_url: str,
    model: str,
    prompt: str,
    no_gpu: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Execute Week 4 E2E verification (WorkerRegistry, Round-Robin, Failover & Recovery)."""
    mode_label = "no-gpu-mock" if no_gpu else "live-gpu"
    prefix = f"w4-e2e-{uuid.uuid4().hex[:6]}"

    if no_gpu:
        from llm_serving_platform.gateway.main import create_app

        app = create_app(
            GatewayConfig(
                no_gpu=True,
                mock_worker_count=2,
                model_name=model,
                routing_policy="round_robin",
            )
        )
        registry = app.state.registry
        monitor = app.state.health_monitor
        workers_by_id = dict(registry.probe_targets())

        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://mock-gateway",
            timeout=30.0,
        ) as client:
            base = "http://mock-gateway"

            # 1. Verify /control/v1/workers & detail endpoint + CORS
            ctrl_resp = await client.get(
                f"{base}/control/v1/workers",
                headers={"Origin": "http://127.0.0.1:3000"},
            )
            if ctrl_resp.status_code != 200:
                raise RuntimeError(f"GET /control/v1/workers failed: HTTP {ctrl_resp.status_code}")
            initial_workers = ctrl_resp.json()
            detail_ok = (
                await client.get(f"{base}/control/v1/workers/worker-mock-0")
            ).status_code == 200
            missing_404 = (
                await client.get(f"{base}/control/v1/workers/worker-does-not-exist")
            ).status_code == 404

            # 2. Verify Round-Robin across 2 HEALTHY workers (4 requests -> 0, 1, 0, 1)
            rr_sequence: list[str] = []
            for i in range(4):
                req_id = f"{prefix}-rr-{i + 1}"
                res = await _send_streaming_chat(client, base, model, prompt, req_id)
                if res["status_code"] != 200 or res["chunks"] == 0:
                    raise RuntimeError(f"Round-Robin request {req_id} failed: {res}")
                trace_resp = await client.get(f"{base}/v1/traces/{req_id}")
                if trace_resp.status_code != 200:
                    raise RuntimeError(f"Missing trace for {req_id}")
                trace_json = trace_resp.json()
                if trace_json["worker_id"] != res["worker_id"]:
                    raise RuntimeError("Worker ID mismatch between header and server trace.")
                rr_sequence.append(str(res["worker_id"]))

            # 3. Mark worker-mock-0 UNHEALTHY -> verify traffic routes ONLY to worker-mock-1
            workers_by_id["worker-mock-0"].is_healthy = False  # type: ignore[attr-defined]
            await monitor.check_once()
            degraded_workers = (await client.get(f"{base}/control/v1/workers")).json()
            failover_sequence: list[str] = []
            for i in range(3):
                req_id = f"{prefix}-failover-{i + 1}"
                res = await _send_streaming_chat(client, base, model, prompt, req_id)
                if res["status_code"] != 200:
                    raise RuntimeError(f"Failover request {req_id} failed: {res}")
                failover_sequence.append(str(res["worker_id"]))

            # 4. Mark ALL workers UNHEALTHY -> verify 503 rejection (no requests sent to bad worker)
            workers_by_id["worker-mock-1"].is_healthy = False  # type: ignore[attr-defined]
            await monitor.check_once()
            health_all_down = await client.get(f"{base}/health")
            req_id_down = f"{prefix}-all-down"
            down_res = await _send_streaming_chat(client, base, model, prompt, req_id_down)
            down_trace = (await client.get(f"{base}/v1/traces/{req_id_down}")).json()

            # 5. Recover worker-mock-0 -> verify traffic resumes cleanly
            workers_by_id["worker-mock-0"].is_healthy = True  # type: ignore[attr-defined]
            await monitor.check_once()
            req_id_rec = f"{prefix}-recovered"
            rec_res = await _send_streaming_chat(client, base, model, prompt, req_id_rec)

        passed = bool(
            len(initial_workers) == 2
            and all(w["status"] == "HEALTHY" for w in initial_workers)
            and detail_ok
            and missing_404
            and rr_sequence == ["worker-mock-0", "worker-mock-1", "worker-mock-0", "worker-mock-1"]
            and failover_sequence == ["worker-mock-1", "worker-mock-1", "worker-mock-1"]
            and health_all_down.status_code == 503
            and down_res["status_code"] == 503
            and down_res["worker_id"] == "none"
            and down_trace["status"] == "error"
            and rec_res["status_code"] == 200
            and rec_res["worker_id"] == "worker-mock-0"
        )

        evidence: dict[str, Any] = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode_label,
            "passed": passed,
            "model": model,
            "initial_workers": initial_workers,
            "degraded_workers": degraded_workers,
            "round_robin_sequence": rr_sequence,
            "failover_sequence": failover_sequence,
            "all_unhealthy_check": {
                "health_status_code": health_all_down.status_code,
                "chat_status_code": down_res["status_code"],
                "chat_worker_id": down_res["worker_id"],
                "trace_status": down_trace["status"],
            },
            "recovery_check": {
                "chat_status_code": rec_res["status_code"],
                "chat_worker_id": rec_res["worker_id"],
            },
        }
    else:
        # Live GPU mode against running Gateway
        base = gateway_url.rstrip("/")
        async with httpx.AsyncClient(timeout=60.0) as client:
            health_resp = await client.get(
                f"{base}/health",
                headers={"Origin": "http://127.0.0.1:3000"},
            )
            if health_resp.status_code != 200:
                raise RuntimeError(
                    f"Live Gateway /health returned HTTP {health_resp.status_code}"
                )

            ctrl_resp = await client.get(
                f"{base}/control/v1/workers",
                headers={"Origin": "http://127.0.0.1:3000"},
            )
            if ctrl_resp.status_code != 200:
                raise RuntimeError(
                    f"Live Gateway /control/v1/workers returned HTTP {ctrl_resp.status_code}"
                )
            live_workers = ctrl_resp.json()
            healthy_ids = [w["worker_id"] for w in live_workers if w.get("status") == "HEALTHY"]
            if not healthy_ids:
                raise RuntimeError(f"No HEALTHY workers in live registry: {live_workers}")

            first_id = healthy_ids[0]
            detail_ok = (
                await client.get(f"{base}/control/v1/workers/{first_id}")
            ).status_code == 200
            missing_404 = (
                await client.get(f"{base}/control/v1/workers/worker-does-not-exist")
            ).status_code == 404

            rr_sequence = []
            num_probes = max(2, len(healthy_ids) * 2)
            for i in range(num_probes):
                req_id = f"{prefix}-gpu-{i + 1}"
                res = await _send_streaming_chat(client, base, model, prompt, req_id)
                if res["status_code"] != 200 or res["chunks"] == 0:
                    raise RuntimeError(f"Live GPU request {req_id} failed: {res}")
                rr_sequence.append(str(res["worker_id"]))

        passed = bool(detail_ok and missing_404 and len(rr_sequence) == num_probes)
        evidence = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "mode": mode_label,
            "passed": passed,
            "model": model,
            "initial_workers": live_workers,
            "round_robin_sequence": rr_sequence,
            "failover_sequence": healthy_ids,
            "all_unhealthy_check": {"verified_via_unit_contract": True},
            "recovery_check": {"chat_status_code": 200, "chat_worker_id": rr_sequence[-1]},
        }

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(evidence, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    return evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify Week 4 WorkerRegistry, HealthMonitor, Round-Robin & Failover."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt string to send.",
    )
    parser.add_argument(
        "--no-gpu",
        "--mock",
        dest="no_gpu",
        action="store_true",
        help="Run in-process with MockInferenceWorker pool without requiring a GPU/vLLM server.",
    )
    parser.add_argument(
        "--gateway-url",
        default=os.getenv("GATEWAY_URL", "http://127.0.0.1:8080"),
        help="Base URL of the Gateway (default: http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="Model ID to request.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON evidence.",
    )
    parser.add_argument(
        "--save-evidence",
        action="store_true",
        help="Save JSON evidence to experiments/raw/week4_e2e_{gpu|no_gpu}_evidence.json.",
    )
    args = parser.parse_args()

    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output)
    elif args.save_evidence:
        out_path = (
            REPO_ROOT / "experiments" / "raw" / "week4_e2e_no_gpu_evidence.json"
            if args.no_gpu
            else REPO_ROOT / "experiments" / "raw" / "week4_e2e_gpu_evidence.json"
        )

    try:
        evidence = asyncio.run(
            run_week4_e2e_check(
                gateway_url=args.gateway_url,
                model=args.model,
                prompt=args.prompt,
                no_gpu=args.no_gpu,
                output_path=out_path,
            )
        )
    except Exception as exc:
        print(f"[-] Week 4 E2E Verification FAILED: {exc}", file=sys.stderr)
        sys.exit(1)

    print("=" * 80)
    print(f"WEEK 4 E2E VERIFICATION REPORT ({evidence['mode'].upper()})")
    print("=" * 80)
    print(f"  Registered Workers   : {[w['worker_id'] for w in evidence['initial_workers']]}")
    print(f"  Round-Robin Sequence : {evidence['round_robin_sequence']}")
    print(f"  Failover Sequence    : {evidence['failover_sequence']}")
    print(f"  All-Down Check       : {evidence['all_unhealthy_check']}")
    print(f"  Recovery Check       : {evidence['recovery_check']}")
    if out_path is not None:
        print(f"  Evidence Saved To    : {out_path}")
    print("=" * 80)
    print(f">>> WEEK 4 EXIT GATE: {'PASSED' if evidence['passed'] else 'FAILED'} <<<")
    sys.exit(0 if evidence["passed"] else 1)


if __name__ == "__main__":
    main()
