"""Exit Gate Verification Script for Week 2 Data Plane Gateway.

Executes 30 streaming chat completion requests through the Gateway,
measures Time-to-First-Token (TTFT) and End-to-End (E2E) latency,
and outputs a structured validation summary table.
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

import httpx

# Ensure project root and src/ are in sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))


class RequestMetric:
    """Stores performance and status metrics for a single inference request."""

    def __init__(self, index: int, request_id: str) -> None:
        self.index = index
        self.request_id = request_id
        self.success = False
        self.ttft_ms: float = 0.0
        self.e2e_ms: float = 0.0
        self.token_count: int = 0
        self.error_message: Optional[str] = None


async def send_streaming_request(
    client: httpx.AsyncClient,
    gateway_url: str,
    index: int,
    model: str,
    timeout_s: float,
) -> RequestMetric:
    """Send a single streaming chat completion request and measure latencies."""
    request_id = f"exit-gate-req-{index:02d}"
    metric = RequestMetric(index, request_id)

    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"Request {index}: Giải thích ngắn gọn cơ chế Continuous Batching.",
            }
        ],
        "stream": True,
        "max_tokens": 50,
    }
    headers = {
        "X-Request-ID": request_id,
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
    }

    start_time = time.perf_counter()
    first_token_time: Optional[float] = None

    try:
        async with client.stream(
            "POST",
            f"{gateway_url}/v1/chat/completions",
            json=payload,
            headers=headers,
            timeout=timeout_s,
        ) as response:
            if response.status_code != 200:
                body = await response.aread()
                err_text = body.decode("utf-8", errors="replace")
                metric.error_message = f"HTTP {response.status_code}: {err_text}"
                metric.e2e_ms = (time.perf_counter() - start_time) * 1000
                return metric

            async for raw_line in response.aiter_lines():
                line = raw_line.strip()
                if not line:
                    continue

                if line.startswith("data:"):
                    data_str = line.removeprefix("data:").strip()
                    if data_str == "[DONE]":
                        break

                    try:
                        data = json.loads(data_str)
                        if "error" in data:
                            metric.error_message = data["error"].get("message", "Unknown error")
                            break

                        if first_token_time is None:
                            first_token_time = time.perf_counter()
                            metric.ttft_ms = (first_token_time - start_time) * 1000

                        metric.token_count += 1
                    except json.JSONDecodeError:
                        continue

        end_time = time.perf_counter()
        metric.e2e_ms = (end_time - start_time) * 1000

        if metric.error_message is None and metric.token_count > 0:
            metric.success = True
        elif metric.error_message is None and metric.token_count == 0:
            metric.error_message = "No tokens received"

    except Exception as exc:
        metric.error_message = str(exc)
        metric.e2e_ms = (time.perf_counter() - start_time) * 1000

    return metric


def calculate_percentile(data: List[float], p: float) -> float:
    """Calculate the p-th percentile from a list of float metrics."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    idx = int(len(sorted_data) * (p / 100.0))
    idx = min(idx, len(sorted_data) - 1)
    return sorted_data[idx]


def print_results_table(metrics: List[RequestMetric]) -> bool:
    """Format and print the benchmark results table and summary statistics."""
    print("\n" + "=" * 90)
    print("          WEEK 2 DATA PLANE GATEWAY - EXIT GATE VERIFICATION REPORT")
    print("=" * 90)
    header = (
        f"{'Req #':<6} | {'Request ID':<20} | {'Status':<8} | "
        f"{'TTFT (ms)':<11} | {'E2E (ms)':<11} | {'Tokens':<6} | {'Notes'}"
    )
    print(header)
    print("-" * 90)

    for m in metrics:
        status_str = "PASS" if m.success else "FAIL"
        ttft_str = f"{m.ttft_ms:.2f}" if m.success else "N/A"
        e2e_str = f"{m.e2e_ms:.2f}"
        note = m.error_message or "OK"
        if len(note) > 16:
            note = note[:13] + "..."
        row = (
            f"{m.index:<6} | {m.request_id:<20} | {status_str:<8} | "
            f"{ttft_str:<11} | {e2e_str:<11} | {m.token_count:<6} | {note}"
        )
        print(row)

    success_metrics = [m for m in metrics if m.success]
    total_reqs = len(metrics)
    success_count = len(success_metrics)
    fail_count = total_reqs - success_count
    pct = (success_count / total_reqs * 100.0) if total_reqs else 0.0

    print("=" * 90)
    print("SUMMARY STATISTICS:")
    print(f"  Total Requests Executed : {total_reqs}")
    print(f"  Successful Requests     : {success_count} / {total_reqs} ({pct:.1f}%)")
    print(f"  Failed Requests         : {fail_count}")

    if success_metrics:
        ttfts = [m.ttft_ms for m in success_metrics]
        e2es = [m.e2e_ms for m in success_metrics]

        print("-" * 90)
        table_hdr = (
            f"{'Metric':<15} | {'Min (ms)':<10} | {'Avg (ms)':<10} | "
            f"{'P95 (ms)':<10} | {'Max (ms)':<10}"
        )
        print(table_hdr)
        print("-" * 90)
        ttft_row = (
            f"{'TTFT':<15} | {min(ttfts):<10.2f} | {sum(ttfts)/len(ttfts):<10.2f} | "
            f"{calculate_percentile(ttfts, 95):<10.2f} | {max(ttfts):<10.2f}"
        )
        print(ttft_row)
        e2e_row = (
            f"{'E2E Latency':<15} | {min(e2es):<10.2f} | {sum(e2es)/len(e2es):<10.2f} | "
            f"{calculate_percentile(e2es, 95):<10.2f} | {max(e2es):<10.2f}"
        )
        print(e2e_row)

    print("=" * 90)
    passed = success_count == total_reqs and total_reqs == 30
    if passed:
        print(">>> EXIT GATE STATUS: PASSED (Target 30/30 successfully met) <<<\n")
    else:
        print(f">>> EXIT GATE STATUS: FAILED ({success_count}/{total_reqs} passed) <<<\n")

    return passed


async def run_verification(
    gateway_url: str,
    num_requests: int = 30,
    model: str = "Qwen/Qwen2.5-0.5B-Instruct",
    concurrency: int = 1,
    timeout_s: float = 60.0,
) -> bool:
    """Execute live requests against the running gateway."""
    print(f"Starting Exit Gate Verification against: {gateway_url}")
    print(f"Target: {num_requests} requests, Model: {model}, Concurrency: {concurrency}")

    limits = httpx.Limits(max_keepalive_connections=50, max_connections=200)
    async with httpx.AsyncClient(limits=limits, timeout=timeout_s) as client:
        try:
            health_resp = await client.get(f"{gateway_url}/health", timeout=5.0)
            print(f"Gateway /health check: {health_resp.status_code} -> {health_resp.text}")
        except Exception as e:
            print(f"[Warning] Gateway /health check failed: {e}")

        sem = asyncio.Semaphore(concurrency)

        async def worker(idx: int) -> RequestMetric:
            async with sem:
                return await send_streaming_request(
                    client=client,
                    gateway_url=gateway_url,
                    index=idx,
                    model=model,
                    timeout_s=timeout_s,
                )

        tasks = [worker(i + 1) for i in range(num_requests)]
        metrics: List[RequestMetric] = []
        for coro in asyncio.as_completed(tasks):
            m = await coro
            metrics.append(m)
            status_tag = "OK" if m.success else f"FAIL ({m.error_message})"
            log_line = (
                f"  [{m.index:02d}/30] ID: {m.request_id} - {status_tag} "
                f"(TTFT: {m.ttft_ms:.1f}ms, E2E: {m.e2e_ms:.1f}ms)"
            )
            print(log_line)

        metrics.sort(key=lambda x: x.index)
        return print_results_table(metrics)


async def run_mock_verification(num_requests: int = 30) -> bool:
    """Run verification in-process using an ASGI TestClient and MockInferenceWorker."""
    print("Running Exit Gate Verification with in-process MockInferenceWorker...")
    from llm_serving_platform.gateway.main import create_app, get_worker
    from tests.mocks import MockInferenceWorker

    mock_worker = MockInferenceWorker(
        canned_chunks=["Batching", " mechanism", " dynamically", " groups", " tokens", "."],
        chunk_delay_s=0.005,
    )
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: mock_worker

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://mock-gateway") as client:
        metrics: List[RequestMetric] = []
        for i in range(1, num_requests + 1):
            m = await send_streaming_request(
                client=client,
                gateway_url="http://mock-gateway",
                index=i,
                model="mock-qwen",
                timeout_s=10.0,
            )
            metrics.append(m)
            status_tag = "OK" if m.success else f"FAIL ({m.error_message})"
            log_line = (
                f"  [{m.index:02d}/30] ID: {m.request_id} - {status_tag} "
                f"(TTFT: {m.ttft_ms:.1f}ms, E2E: {m.e2e_ms:.1f}ms)"
            )
            print(log_line)

        metrics.sort(key=lambda x: x.index)
        return print_results_table(metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Week 2 Gateway Exit Gate Verification")
    parser.add_argument(
        "--gateway-url",
        default=os.getenv("GATEWAY_URL", "http://127.0.0.1:8080"),
        help="Base URL of the Gateway (default: http://127.0.0.1:8080)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="Model ID to request",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=30,
        help="Number of requests to fire (default: 30)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent streams (default: 1)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in-process against MockInferenceWorker without needing live server",
    )
    args = parser.parse_args()

    if args.mock:
        success = asyncio.run(run_mock_verification(num_requests=args.requests))
    else:
        success = asyncio.run(
            run_verification(
                gateway_url=args.gateway_url,
                num_requests=args.requests,
                model=args.model,
                concurrency=args.concurrency,
            )
        )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
