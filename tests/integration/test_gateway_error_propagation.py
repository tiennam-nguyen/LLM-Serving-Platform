"""Integration tests for gateway error propagation and streaming resilience."""

import json
from typing import AsyncIterator

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from llm_serving_platform.gateway.main import create_app, get_worker
from llm_serving_platform.workers.base import (
    WorkerTimeoutError,
    WorkerUnavailableError,
)
from llm_serving_platform.workers.vllm_adapter import VLLMWorkerAdapter
from tests.mocks import MockInferenceWorker


@pytest.mark.asyncio
async def test_successful_streaming_chat():
    """Verify normal streaming responses deliver expected tokens and terminate with [DONE]."""
    mock_worker = MockInferenceWorker(canned_chunks=["One", " Two", " Three"])
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: mock_worker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "temperature": 0.7,
            "stream": True,
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]

        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        assert len(lines) >= 4
        assert lines[-1] == "data: [DONE]"

        # Check payload received by worker has exclude_none=True
        assert len(mock_worker.received_requests) == 1
        rec = mock_worker.received_requests[0]
        assert rec["model"] == "test-model"
        assert rec["temperature"] == 0.7
        assert "top_p" not in rec  # None was excluded


@pytest.mark.asyncio
async def test_worker_timeout_error_propagation():
    """Verify WorkerTimeoutError yields a standard SSE error event and terminates with [DONE]."""
    mock_worker = MockInferenceWorker(
        raise_error_before_stream=WorkerTimeoutError("Inference timed out after 30s")
    )
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: mock_worker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        assert len(lines) == 2
        assert lines[0].startswith("data: ")
        err_json = json.loads(lines[0].removeprefix("data: "))
        assert "error" in err_json
        assert err_json["error"]["code"] == 504
        assert "Inference timed out" in err_json["error"]["message"]
        assert lines[1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_worker_unavailable_error_propagation():
    """Verify WorkerUnavailableError yields a 503 error event and terminates with [DONE]."""
    mock_worker = MockInferenceWorker(
        raise_error_before_stream=WorkerUnavailableError("vLLM backend is down", status_code=503)
    )
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: mock_worker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        assert len(lines) == 2
        err_json = json.loads(lines[0].removeprefix("data: "))
        assert err_json["error"]["code"] == 503
        assert "vLLM backend is down" in err_json["error"]["message"]
        assert lines[1] == "data: [DONE]"


@pytest.mark.asyncio
async def test_mid_stream_worker_error_propagation():
    """Verify mid-stream WorkerException yields tokens received so far,
    then error event and [DONE].
    """
    mock_worker = MockInferenceWorker(
        canned_chunks=["Token1", "Token2", "Token3"],
        raise_error_after_tokens=2,
        mid_stream_error=WorkerTimeoutError("GPU pipeline crashed mid-stream"),
    )
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: mock_worker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        payload = {
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "stream": True,
        }
        response = await client.post("/v1/chat/completions", json=payload)
        assert response.status_code == 200

        lines = [line.strip() for line in response.text.split("\n") if line.strip()]
        # Should have 2 token chunks, 1 error chunk, 1 [DONE] chunk
        assert len(lines) == 4

        # Check first token
        chunk0 = json.loads(lines[0].removeprefix("data: "))
        assert chunk0["choices"][0]["delta"]["content"] == "Token1"

        # Check second token
        chunk1 = json.loads(lines[1].removeprefix("data: "))
        assert chunk1["choices"][0]["delta"]["content"] == "Token2"

        # Check error chunk
        err_chunk = json.loads(lines[2].removeprefix("data: "))
        assert err_chunk["error"]["code"] == 504
        assert "GPU pipeline crashed mid-stream" in err_chunk["error"]["message"]

        # Check final termination
        assert lines[3] == "data: [DONE]"


@pytest.mark.asyncio
async def test_vllm_adapter_streaming_and_error_handling():
    """Test VLLMWorkerAdapter mapping httpx timeouts and errors into custom WorkerExceptions."""

    # 1. Normal streaming
    async def sse_response_generator() -> AsyncIterator[bytes]:
        yield b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    def normal_handler(request: httpx.Request) -> httpx.Response:
        assert request.headers.get("X-Request-ID") == "req-test-1"
        return httpx.Response(200, content=sse_response_generator())

    async with httpx.AsyncClient(transport=httpx.MockTransport(normal_handler)) as client:
        adapter = VLLMWorkerAdapter(base_url="http://vllm:8000", client=client)
        chunks = [c async for c in adapter.stream_chat({"model": "test"}, request_id="req-test-1")]
        assert len(chunks) == 2
        assert "Hi" in chunks[0]
        assert chunks[1] == "data: [DONE]\n\n"

    # 2. Timeout error mapping
    def timeout_handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("Read timed out")

    async with httpx.AsyncClient(transport=httpx.MockTransport(timeout_handler)) as client:
        adapter_timeout = VLLMWorkerAdapter(base_url="http://vllm:8000", client=client)
        with pytest.raises(WorkerTimeoutError) as exc_info:
            async for _ in adapter_timeout.stream_chat({}, request_id="req-test-2"):
                pass
        assert exc_info.value.status_code == 504

    # 3. Upstream non-200 error mapping
    def error_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="Engine internal failure")

    async with httpx.AsyncClient(transport=httpx.MockTransport(error_handler)) as client:
        adapter_err = VLLMWorkerAdapter(base_url="http://vllm:8000", client=client)
        with pytest.raises(WorkerUnavailableError) as exc_info:
            async for _ in adapter_err.stream_chat({}, request_id="req-test-3"):
                pass
        assert exc_info.value.status_code == 500
        assert "Engine internal failure" in exc_info.value.message


@pytest.mark.asyncio
async def test_client_cancellation_raises_cancelled_error():
    """Verify that asyncio.CancelledError is re-raised cleanly without yielding error payloads."""
    import asyncio

    from llm_serving_platform.gateway.main import safe_event_generator

    class CancellingWorker(MockInferenceWorker):
        async def stream_chat(self, payload, request_id):
            yield 'data: {"token": "hello"}\n\n'
            raise asyncio.CancelledError()

    worker = CancellingWorker()
    gen = safe_event_generator(worker, {"model": "test"}, "req-cancel", "127.0.0.1")

    # First token yields normally
    first_chunk = await anext(gen)
    assert "hello" in first_chunk

    # Second pull raises CancelledError
    with pytest.raises(asyncio.CancelledError):
        await anext(gen)

