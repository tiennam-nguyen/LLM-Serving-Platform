"""Dual-mode E2E verification test (supports both live GPU and --no-gpu mock execution)."""

import os

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from llm_serving_platform.common.config import GatewayConfig
from llm_serving_platform.gateway.main import create_app


@pytest.mark.asyncio
async def test_dual_mode_week3_e2e_trace_and_stream(no_gpu: bool, prompt_text: str):
    """Verify full E2E request -> stream -> X-Worker-ID -> 6-milestone trace correlation."""
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    request_id = "pytest-week3-e2e-01"

    if no_gpu:
        app = create_app(GatewayConfig(no_gpu=True))
    else:
        app = create_app(GatewayConfig(no_gpu=False))
        vllm_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=2.0) as probe:
                resp = await probe.get(f"{vllm_url}/health")
                live_available = resp.status_code == 200
        except httpx.HTTPError:
            live_available = False

        if not live_available:
            if os.getenv("REQUIRE_LIVE_GPU") == "1":
                pytest.fail(
                    f"Live vLLM GPU worker is not reachable at {vllm_url}/health "
                    "(pass --no-gpu for CPU/CI testing)."
                )
            pytest.skip(
                f"Live vLLM worker not running at {vllm_url}; "
                "start vLLM or pass --no-gpu to run with MockInferenceWorker."
            )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Origin": "http://127.0.0.1:3000", "X-Request-ID": request_id},
            json={
                "model": model_name,
                "messages": [{"role": "user", "content": prompt_text}],
                "stream": True,
                "max_tokens": 64,
            },
        )
        assert response.status_code == 200
        assert response.headers.get("X-Request-ID") == request_id
        worker_id = response.headers.get("X-Worker-ID")
        assert worker_id is not None and len(worker_id) > 0

        trace_resp = await client.get(f"/v1/traces/{request_id}")
        assert trace_resp.status_code == 200
        trace = trace_resp.json()
        assert trace["request_id"] == request_id
        assert trace["worker_id"] == worker_id
        assert trace["status"] == "completed"
        assert trace["token_chunks"] >= 1
        for key in (
            "arrival",
            "routing_start",
            "routing_end",
            "worker_start",
            "first_token",
            "last_token",
        ):
            assert trace["milestones"][key] is not None


@pytest.mark.asyncio
async def test_dual_mode_week4_registry_and_round_robin_e2e(no_gpu: bool, prompt_text: str):
    """Verify Week 4 E2E flow: /control/v1/workers + Round-Robin routing across healthy pool."""
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

    if no_gpu:
        app = create_app(GatewayConfig(no_gpu=True, mock_worker_count=2, model_name=model_name))
    else:
        app = create_app(GatewayConfig(no_gpu=False, model_name=model_name))
        vllm_url = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=2.0) as probe:
                resp = await probe.get(f"{vllm_url}/health")
                live_available = resp.status_code == 200
        except httpx.HTTPError:
            live_available = False

        if not live_available:
            if os.getenv("REQUIRE_LIVE_GPU") == "1":
                pytest.fail(
                    f"Live vLLM GPU worker is not reachable at {vllm_url}/health "
                    "(pass --no-gpu for CPU/CI testing)."
                )
            pytest.skip(
                f"Live vLLM worker not running at {vllm_url}; "
                "start vLLM or pass --no-gpu to run with MockInferenceWorker."
            )

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Ensure health check has run and inspect /control/v1/workers
        health_resp = await client.get("/health")
        assert health_resp.status_code == 200

        workers_resp = await client.get("/control/v1/workers")
        assert workers_resp.status_code == 200
        workers = workers_resp.json()
        assert len(workers) >= 1
        healthy_ids = [w["worker_id"] for w in workers if w["status"] == "HEALTHY"]
        assert len(healthy_ids) >= 1

        observed_workers: list[str] = []
        for idx in range(len(healthy_ids) * 2):
            req_id = f"pytest-week4-e2e-{idx + 1}"
            resp = await client.post(
                "/v1/chat/completions",
                headers={"Origin": "http://127.0.0.1:3000", "X-Request-ID": req_id},
                json={
                    "model": model_name,
                    "messages": [{"role": "user", "content": prompt_text}],
                    "stream": True,
                    "max_tokens": 64,
                },
            )
            assert resp.status_code == 200
            w_id = resp.headers.get("X-Worker-ID")
            assert w_id in healthy_ids
            observed_workers.append(str(w_id))

        if no_gpu:
            assert observed_workers == [
                "worker-mock-0",
                "worker-mock-1",
                "worker-mock-0",
                "worker-mock-1",
            ]
