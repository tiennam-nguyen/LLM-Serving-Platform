"""Dual-mode E2E verification test (supports both live GPU and --no-gpu mock execution)."""

import os

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from llm_serving_platform.gateway.main import create_app, get_worker
from tests.mocks import MockInferenceWorker


@pytest.mark.asyncio
async def test_dual_mode_week3_e2e_trace_and_stream(no_gpu: bool, prompt_text: str):
    """Verify full E2E request -> stream -> X-Worker-ID -> 6-milestone trace correlation."""
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    request_id = "pytest-week3-e2e-01"

    app = create_app()
    if no_gpu:
        mock_worker = MockInferenceWorker(
            canned_chunks=["Continuous", " batching", " ghép", " động", " các", " request."],
            worker_id="worker-mock-ci-0",
        )
        app.dependency_overrides[get_worker] = lambda: mock_worker
    else:
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
        assert response.headers.get("X-Worker-ID") is not None

        trace_resp = await client.get(f"/v1/traces/{request_id}")
        assert trace_resp.status_code == 200
        trace = trace_resp.json()
        assert trace["request_id"] == request_id
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
