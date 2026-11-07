"""Dual-mode E2E verification test (supports both live GPU and --no-gpu mock execution)."""

import os

import pytest
from httpx import ASGITransport, AsyncClient

from llm_serving_platform.common.config import GatewayConfig
from llm_serving_platform.gateway.main import create_app, lifespan


def _require_or_skip_live_gateway(status_code: int) -> None:
    if status_code == 200:
        return
    message = (
        "Live Gateway has no healthy vLLM worker. Configure VLLM_BASE_URL/WORKER_ENDPOINTS "
        "or pass --no-gpu for CPU/CI testing."
    )
    if os.getenv("REQUIRE_LIVE_GPU") == "1":
        pytest.fail(message)
    pytest.skip(message)


@pytest.mark.asyncio
async def test_dual_mode_week3_e2e_trace_and_stream(no_gpu: bool, prompt_text: str):
    """Verify request -> stream -> worker ID -> six-milestone trace correlation."""
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    request_id = "pytest-week3-e2e-01"
    app = create_app(GatewayConfig(no_gpu=no_gpu, model_name=model_name))

    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            if not no_gpu:
                _require_or_skip_live_gateway((await client.get("/health")).status_code)

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
    """Verify registry state and strict Round-Robin order on the healthy worker snapshot."""
    model_name = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")
    app = create_app(
        GatewayConfig(
            no_gpu=no_gpu,
            mock_worker_count=2,
            model_name=model_name,
            routing_policy="round_robin",
        )
    )

    async with lifespan(app):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            health_resp = await client.get("/health")
            if not no_gpu:
                _require_or_skip_live_gateway(health_resp.status_code)
            else:
                assert health_resp.status_code == 200

            workers_resp = await client.get("/control/v1/workers")
            assert workers_resp.status_code == 200
            workers = workers_resp.json()
            assert len(workers) >= 1
            healthy_ids = [w["worker_id"] for w in workers if w["status"] == "HEALTHY"]
            assert healthy_ids

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
                assert resp.headers.get("X-Request-ID") == req_id
                worker_id = resp.headers.get("X-Worker-ID")
                assert worker_id in healthy_ids
                observed_workers.append(str(worker_id))

                trace_resp = await client.get(f"/v1/traces/{req_id}")
                assert trace_resp.status_code == 200
                trace = trace_resp.json()
                assert trace["worker_id"] == worker_id
                assert trace["status"] == "completed"

            # Fresh app => cursor starts at zero, and registry ordering is deterministic.
            assert observed_workers == healthy_ids * 2
