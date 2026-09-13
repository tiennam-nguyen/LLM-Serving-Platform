"""Unit tests for payload validation, health checks, and request ID tracking."""

import uuid

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from llm_serving_platform.gateway.main import create_app, get_worker
from llm_serving_platform.workers.vllm_adapter import VLLMWorkerAdapter
from tests.mocks import MockInferenceWorker


@pytest.fixture
def mock_worker() -> MockInferenceWorker:
    return MockInferenceWorker()


@pytest.fixture
def test_app(mock_worker: MockInferenceWorker):
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: mock_worker
    return app


@pytest.mark.asyncio
async def test_validation_missing_messages(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "Qwen/Qwen2.5-0.5B-Instruct", "stream": True},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data


@pytest.mark.asyncio
async def test_validation_missing_model(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"messages": [{"role": "user", "content": "hi"}], "stream": True},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data


@pytest.mark.asyncio
async def test_validation_empty_messages(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [], "stream": True},
        )
        assert response.status_code == 400
        data = response.json()
        assert "error" in data


@pytest.mark.asyncio
async def test_validation_invalid_message_format(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # messages contains non-dict
        response = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": ["just a string"], "stream": True},
        )
        assert response.status_code == 400

        # message missing role/content
        response2 = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"wrong_key": "val"}], "stream": True},
        )
        assert response2.status_code == 400


@pytest.mark.asyncio
async def test_validation_stream_false_rejected(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        response = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": False,
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert "Only streaming mode (stream=True) is currently supported" in data.get("detail", "")


@pytest.mark.asyncio
async def test_request_id_generation_and_propagation(test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Case 1: Custom request ID provided
        custom_id = "test-custom-id-12345"
        resp1 = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
            headers={"X-Request-ID": custom_id},
        )
        assert resp1.headers.get("X-Request-ID") == custom_id

        # Case 2: No request ID provided -> generated UUID4
        resp2 = await client.post(
            "/v1/chat/completions",
            json={"model": "test-model", "messages": [{"role": "user", "content": "hi"}]},
        )
        generated_id = resp2.headers.get("X-Request-ID")
        assert generated_id is not None
        # Verify it's a valid UUID
        parsed_uuid = uuid.UUID(generated_id)
        assert parsed_uuid.version == 4


@pytest.mark.asyncio
async def test_gateway_health_check(mock_worker, test_app):
    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Healthy worker
        mock_worker.is_healthy = True
        resp = await client.get("/health")
        assert resp.status_code == 200
        assert resp.json() == {"status": "healthy", "worker": "up"}

        # Unhealthy worker
        mock_worker.is_healthy = False
        resp_unhealthy = await client.get("/health")
        assert resp_unhealthy.status_code == 503
        assert resp_unhealthy.json() == {"status": "unhealthy", "worker": "down"}


@pytest.mark.asyncio
async def test_vllm_adapter_health_check():
    # Mock httpx responses for VLLMWorkerAdapter.check_health()
    def mock_handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            if request.headers.get("simulate-down"):
                return httpx.Response(503, text="Service Unavailable")
            return httpx.Response(200, text="OK")
        return httpx.Response(404)

    mock_transport = httpx.MockTransport(mock_handler)
    async with httpx.AsyncClient(transport=mock_transport) as client:
        adapter = VLLMWorkerAdapter(base_url="http://mock-vllm:8000", client=client)
        assert await adapter.check_health() is True

    # Test failure case
    def fail_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Internal Server Error")

    mock_fail_transport = httpx.MockTransport(fail_handler)
    async with httpx.AsyncClient(transport=mock_fail_transport) as client:
        adapter_fail = VLLMWorkerAdapter(base_url="http://mock-vllm:8000", client=client)
        assert await adapter_fail.check_health() is False
