"""Observe operator-controlled baseline, worker-B stop, and recovery; never manage workers."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from infra.scripts.verify_week4_e2e import (  # noqa: E402
    _matches_round_robin_cycle,
    _send_streaming_chat,
)

MILESTONES = (
    "arrival", "routing_start", "routing_end", "worker_start", "first_token", "last_token",
)
PHASES = ("baseline", "worker-b-down", "recovered")


def validate_url(value: str) -> str:
    """Accept an HTTP origin without credentials or placeholders; keep evidence non-secret."""
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"} or not parts.hostname
        or parts.username is not None or parts.password is not None
        or parts.query or parts.fragment or parts.path not in {"", "/"}
        or "REPLACE" in value.upper() or (parts.port is not None and parts.port < 1)
    ):
        raise ValueError("Use http(s)://host[:port], without placeholders or credentials")
    return value.rstrip("/")


def git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
            stderr=subprocess.DEVNULL, timeout=3,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


async def _health(client: httpx.AsyncClient, url: str) -> dict[str, Any]:
    try:
        response = await client.get(f"{url}/health", timeout=3)
        return {"status_code": response.status_code, "reachable": True}
    except httpx.RequestError as exc:
        return {"status_code": None, "reachable": False, "error_type": type(exc).__name__}


async def run_phase(
    *, phase: str, gateway_url: str, worker_a_url: str, worker_b_url: str,
    worker_a_id: str = "worker-vllm-a", worker_b_id: str = "worker-vllm-b",
    model: str = "Qwen/Qwen2.5-0.5B-Instruct", wait_timeout_s: float = 30,
    request_timeout_s: float = 90, output_path: Path | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Return and optionally persist evidence even if a phase fails.

    A supplied client enables controlled CPU fixtures; it does not establish physical LAN evidence.
    Registry polling deliberately precedes Gateway /health, which itself forces a health probe.
    """
    evidence: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": git_revision(), "phase": phase, "gateway_url": gateway_url,
        "workers": {worker_a_id: worker_a_url, worker_b_id: worker_b_url},
        "model": model, "registry_snapshot": None, "requests": [],
        "worker_assignments": [], "passed": False, "failure_details": [],
        "scope": "HTTP observations only; physical host/GPU identity requires operator inventory",
    }
    owned = client is None
    try:
        if phase not in PHASES:
            raise ValueError("Unknown phase")
        for value in (wait_timeout_s, request_timeout_s):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Timeouts must be finite and positive")
        base, a_url, b_url = map(validate_url, (gateway_url, worker_a_url, worker_b_url))
        if not worker_a_id or not worker_b_id or worker_a_id == worker_b_id or a_url == b_url:
            raise ValueError("Two distinct worker IDs and endpoints required")
        if client is None:
            # Direct LAN HTTP: ambient HTTP_PROXY must not proxy private lab traffic.
            client = httpx.AsyncClient(timeout=request_timeout_s, trust_env=False)
        evidence["direct_health"] = {
            worker_a_id: await _health(client, a_url), worker_b_id: await _health(client, b_url),
        }
        direct = evidence["direct_health"]
        if direct[worker_a_id]["status_code"] != 200:
            raise RuntimeError("Worker A direct /health must return 200")
        b_healthy = direct[worker_b_id]["status_code"] == 200
        if b_healthy == (phase == "worker-b-down"):
            raise RuntimeError(
                "Worker B direct /health contradicts requested phase; stop/restart B manually"
            )

        expected = {
            worker_a_id: "HEALTHY",
            worker_b_id: "UNHEALTHY" if phase == "worker-b-down" else "HEALTHY",
        }
        expected_urls = {worker_a_id: a_url, worker_b_id: b_url}
        loop = asyncio.get_running_loop()
        deadline = loop.time() + wait_timeout_s
        evidence["registry_poll_count"] = 0
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise TimeoutError("Registry did not reach expected states within wait timeout")
            async with asyncio.timeout(remaining):
                response = await client.get(f"{base}/control/v1/workers")
            response.raise_for_status()
            records = response.json()
            evidence["registry_snapshot"] = records
            evidence["registry_poll_count"] += 1
            if not isinstance(records, list):
                raise RuntimeError("Registry response must be a list")
            by_id = {w["worker_id"]: w for w in records}
            if len(records) != 2 or set(by_id) != set(expected):
                raise RuntimeError("Registry must contain exactly the configured A/B IDs")
            if any(by_id[key]["endpoint"].rstrip("/") != expected_urls[key] for key in expected):
                raise RuntimeError(
                    "Registry endpoint mismatch; check WORKER_ENDPOINTS and verifier URLs"
                )
            if all(by_id[key]["status"] == status for key, status in expected.items()):
                break
            await asyncio.sleep(min(0.5, max(0, deadline - loop.time())))

        evidence["gateway_health"] = await _health(client, base)
        if evidence["gateway_health"]["status_code"] != 200:
            raise RuntimeError("Gateway /health must remain 200")
        candidates = sorted(key for key, status in expected.items() if status == "HEALTHY")
        prefix = f"w5-{phase}-{uuid.uuid4().hex[:12]}"
        # Two full cycles, with no other client issuing inference during this observation.
        for index in range(4):
            request_id = f"{prefix}-{index + 1}"
            observation: dict[str, Any] = {"request_id": request_id}
            evidence["requests"].append(observation)
            async with asyncio.timeout(request_timeout_s):
                result = await _send_streaming_chat(
                    client, base, model, "Reply with a short greeting.", request_id,
                )
                observation.update({key: result[key] for key in (
                    "status_code", "request_id", "worker_id", "chunks",
                )})
                observation["sent_request_id"] = request_id
                if (result["status_code"] != 200 or result["chunks"] < 1
                        or result["request_id"] != request_id
                        or result["worker_id"] not in candidates):
                    raise RuntimeError(f"Streaming/header contract failed for {request_id}")
                evidence["worker_assignments"].append(result["worker_id"])
                # Completion storage can lag receipt of [DONE] by a scheduler turn.
                trace_deadline = loop.time() + min(5, request_timeout_s)
                while True:
                    trace_response = await client.get(f"{base}/v1/traces/{request_id}")
                    trace_response.raise_for_status()
                    trace = trace_response.json()
                    if trace.get("status") != "in_flight" or loop.time() >= trace_deadline:
                        break
                    await asyncio.sleep(0.05)
                observation["trace"] = {key: trace.get(key) for key in (
                    "request_id", "worker_id", "status", "milestones",
                )}
                if (trace.get("request_id") != request_id
                        or trace.get("worker_id") != result["worker_id"]
                        or trace.get("status") != "completed"
                        or any(trace.get("milestones", {}).get(key) is None for key in MILESTONES)):
                    raise RuntimeError(
                        f"Completed six-milestone trace contract failed for {request_id}"
                    )
        if not _matches_round_robin_cycle(evidence["worker_assignments"], candidates):
            raise RuntimeError(
                "Assignments do not form a strict Round-Robin cycle; stop other inference clients"
            )
        evidence["passed"] = True
    except Exception as exc:
        # Do not retain response bodies, generated content, prompts or proxy credentials.
        evidence["failure_details"].append({"type": type(exc).__name__, "message": str(exc)})
    finally:
        if owned and client is not None:
            await client.aclose()
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8080", type=validate_url)
    parser.add_argument("--worker-a-url", required=True, type=validate_url)
    parser.add_argument("--worker-b-url", required=True, type=validate_url)
    parser.add_argument("--worker-a-id", default="worker-vllm-a")
    parser.add_argument("--worker-b-id", default="worker-vllm-b")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--wait-timeout-s", type=float, default=30)
    parser.add_argument("--request-timeout-s", type=float, default=90)
    parser.add_argument("--output", type=Path, required=True)
    args = vars(parser.parse_args())
    args["output_path"] = args.pop("output")
    evidence = asyncio.run(run_phase(**args))
    print(json.dumps({key: evidence[key] for key in (
        "phase", "passed", "worker_assignments", "failure_details",
    )}, indent=2))
    print(f"Evidence: {args['output_path']}")
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
