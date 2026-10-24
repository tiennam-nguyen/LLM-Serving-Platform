"""Week 3 E2E Verification & Evidence Recorder (Browser/Client -> Gateway -> Worker -> Trace).

Usage examples:
  1. Default mode (Live GPU, 1 default prompt):
     python infra/scripts/verify_week3_e2e.py

  2. Live GPU with custom prompt (--prompt):
     python infra/scripts/verify_week3_e2e.py --prompt "Giải thích KV cache trong 2 câu."

  3. No-GPU mode for CI / CPU machines (--no-gpu):
     python infra/scripts/verify_week3_e2e.py --no-gpu
"""

import argparse
import asyncio
import json
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import httpx

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from llm_serving_platform.common.config import GatewayConfig, load_dotenv_file  # noqa: E402

load_dotenv_file()

DEFAULT_PROMPT = "Giải thích ngắn gọn cơ chế Continuous Batching trong 2 câu."


async def run_week3_e2e_check(
    gateway_url: str,
    model: str,
    prompt: str,
    no_gpu: bool = False,
    request_id: str | None = None,
    output_path: Path | None = None,
) -> dict[str, Any]:
    """Execute a full Week 3 E2E request and correlate client metrics with server /v1/traces."""
    req_id = request_id or f"w3-e2e-{uuid.uuid4().hex[:8]}"
    mode_label = "no-gpu-mock" if no_gpu else "live-gpu"

    if no_gpu:
        from llm_serving_platform.gateway.main import create_app

        app = create_app(GatewayConfig(no_gpu=True))
        client_ctx = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://mock-gateway",
            timeout=30.0,
        )
        base = "http://mock-gateway"
    else:
        client_ctx = httpx.AsyncClient(timeout=60.0)
        base = gateway_url.rstrip("/")

    async with client_ctx as client:
        health_resp = await client.get(
            f"{base}/health",
            headers={"Origin": "http://127.0.0.1:3000"},
        )
        health_ok = health_resp.status_code == 200
        health_worker_id = health_resp.headers.get("X-Worker-ID")

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": True,
            "max_tokens": 128,
        }
        headers = {
            "Origin": "http://127.0.0.1:3000",
            "X-Request-ID": req_id,
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        t_submit = time.perf_counter()
        t_accepted: float | None = None
        t_first_content: float | None = None
        assembled_text = ""
        chunk_count = 0
        resp_request_id: str | None = None
        resp_worker_id: str | None = None
        cors_origin: str | None = None
        stream_error: str | None = None

        async with client.stream(
            "POST",
            f"{base}/v1/chat/completions",
            json=payload,
            headers=headers,
        ) as resp:
            t_accepted = time.perf_counter()
            resp_request_id = resp.headers.get("X-Request-ID")
            resp_worker_id = resp.headers.get("X-Worker-ID")
            cors_origin = resp.headers.get("access-control-allow-origin")

            if resp.status_code != 200:
                raw_err = (await resp.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"Gateway returned HTTP {resp.status_code}: {raw_err}")

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
                if "error" in event:
                    stream_error = event["error"].get("message", "Stream error")
                    break
                choices = event.get("choices", [])
                if choices and isinstance(choices[0], dict):
                    delta_text = choices[0].get("delta", {}).get("content")
                    if isinstance(delta_text, str) and delta_text:
                        if t_first_content is None:
                            t_first_content = time.perf_counter()
                        chunk_count += 1
                        assembled_text += delta_text

        t_completed = time.perf_counter()

        trace_resp = await client.get(
            f"{base}/v1/traces/{req_id}",
            headers={"Origin": "http://127.0.0.1:3000"},
        )
        if trace_resp.status_code != 200:
            raise RuntimeError(
                f"Failed to fetch /v1/traces/{req_id}: HTTP {trace_resp.status_code}"
            )
        server_trace = trace_resp.json()

    client_first_content_ms = (
        round((t_first_content - t_submit) * 1000.0, 2) if t_first_content is not None else None
    )
    client_e2e_ms = round((t_completed - t_submit) * 1000.0, 2)
    client_accepted_ms = (
        round((t_accepted - t_submit) * 1000.0, 2) if t_accepted is not None else None
    )

    milestones = server_trace.get("milestones", {})
    all_milestones_present = all(
        milestones.get(k) is not None
        for k in (
            "arrival",
            "routing_start",
            "routing_end",
            "worker_start",
            "first_token",
            "last_token",
        )
    )

    passed = bool(
        health_ok
        and stream_error is None
        and resp_request_id == req_id
        and bool(resp_worker_id)
        and server_trace.get("request_id") == req_id
        and server_trace.get("worker_id") == resp_worker_id
        and server_trace.get("status") == "completed"
        and all_milestones_present
        and chunk_count > 0
    )

    evidence: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "mode": mode_label,
        "passed": passed,
        "request_id": req_id,
        "model": model,
        "prompt": prompt,
        "health_check": {
            "healthy": health_ok,
            "worker_id_header": health_worker_id,
        },
        "client_observed": {
            "request_id_header": resp_request_id,
            "worker_id_header": resp_worker_id,
            "cors_allow_origin": cors_origin,
            "accepted_ms": client_accepted_ms,
            "first_content_ms": client_first_content_ms,
            "e2e_ms": client_e2e_ms,
            "content_chunks": chunk_count,
            "response_preview": assembled_text[:200],
            "error": stream_error,
        },
        "server_trace": server_trace,
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
        description="Verify Week 3 E2E flow and correlate Request ID with backend trace."
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Prompt string to send (default: 1-sentence Vietnamese prompt).",
    )
    parser.add_argument(
        "--no-gpu",
        "--mock",
        dest="no_gpu",
        action="store_true",
        help="Run in-process with MockInferenceWorker without requiring a GPU/vLLM server.",
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
        "--request-id",
        default=None,
        help="Optional explicit X-Request-ID for correlation.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save JSON evidence (pass --save-evidence for default path).",
    )
    parser.add_argument(
        "--save-evidence",
        action="store_true",
        help="Save JSON evidence to experiments/raw/week3_e2e_{gpu|no_gpu}_evidence.json.",
    )
    args = parser.parse_args()

    out_path: Path | None = None
    if args.output:
        out_path = Path(args.output)
    elif args.save_evidence:
        out_path = (
            REPO_ROOT / "experiments" / "raw" / "week3_e2e_no_gpu_evidence.json"
            if args.no_gpu
            else REPO_ROOT / "experiments" / "raw" / "week3_e2e_gpu_evidence.json"
        )

    try:
        evidence = asyncio.run(
            run_week3_e2e_check(
                gateway_url=args.gateway_url,
                model=args.model,
                prompt=args.prompt,
                no_gpu=args.no_gpu,
                request_id=args.request_id,
                output_path=out_path,
            )
        )
    except Exception as exc:
        print(f"[-] Week 3 E2E Verification FAILED: {exc}", file=sys.stderr)
        if not args.no_gpu:
            print(
                "    Tip: Start vLLM + Gateway first, or pass --no-gpu for mock/CPU verification.",
                file=sys.stderr,
            )
        sys.exit(1)

    print("=" * 80)
    print(f"WEEK 3 E2E VERIFICATION REPORT ({evidence['mode'].upper()})")
    print("=" * 80)
    print(f"  Request ID           : {evidence['request_id']}")
    print(f"  Worker ID            : {evidence['client_observed']['worker_id_header']}")
    print(f"  Model                : {evidence['model']}")
    print(f"  Prompt               : {evidence['prompt']}")
    print(f"  Client First Content : {evidence['client_observed']['first_content_ms']} ms")
    print(f"  Client E2E Latency   : {evidence['client_observed']['e2e_ms']} ms")
    st_metrics = evidence["server_trace"]["metrics_ms"]
    print(f"  Server Routing       : {st_metrics['routing_ms']} ms")
    print(f"  Server TTFT          : {st_metrics['ttft_ms']} ms")
    print(f"  Server Inter-Chunk   : {st_metrics['inter_content_chunk_ms']} ms")
    print(f"  Server TPOT          : {st_metrics['tpot_ms']}")
    print(f"  Server E2E           : {st_metrics['e2e_ms']} ms")
    print(f"  Response Preview     : {evidence['client_observed']['response_preview']}")
    if out_path is not None:
        print(f"  Evidence Saved To    : {out_path}")
    print("=" * 80)
    print(f">>> WEEK 3 EXIT GATE: {'PASSED' if evidence['passed'] else 'FAILED'} <<<")
    sys.exit(0 if evidence["passed"] else 1)


if __name__ == "__main__":
    main()
