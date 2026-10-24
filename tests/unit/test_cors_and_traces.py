from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from llm_serving_platform.common.config import GatewayConfig, load_dotenv_file
from llm_serving_platform.gateway.main import create_app, get_worker
from llm_serving_platform.telemetry.trace import InMemoryTraceStore, RequestTrace
from llm_serving_platform.workers.base import WorkerUnavailableError
from llm_serving_platform.workers.factory import create_worker
from llm_serving_platform.workers.mock_adapter import MockInferenceWorker
from llm_serving_platform.workers.vllm_adapter import VLLMWorkerAdapter


def test_dotenv_loader_reads_env_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Verify load_dotenv_file handles comments, export prefixes, quotes, and fallback defaults."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# Comment line\n"
        "export GATEWAY_PORT=9090 # inline comment\n"
        'WORKER_ID="worker-test-dotenv"\n'
        "VLLM_TIMEOUT_S=invalid_float\n"
        "NO_GPU=1\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("GATEWAY_PORT", raising=False)
    monkeypatch.delenv("WORKER_ID", raising=False)
    monkeypatch.delenv("VLLM_TIMEOUT_S", raising=False)
    monkeypatch.delenv("NO_GPU", raising=False)

    load_dotenv_file(env_file, override=True)
    cfg = GatewayConfig()
    assert cfg.gateway_port == 9090
    assert cfg.worker_id == "worker-test-dotenv"
    assert cfg.worker_timeout_s == 60.0
    assert cfg.no_gpu is True


def test_worker_factory_no_gpu_and_live_gpu_modes(monkeypatch: pytest.MonkeyPatch):
    """Verify create_worker selects MockInferenceWorker for no_gpu and VLLMWorkerAdapter for GPU."""
    monkeypatch.delenv("WORKER_ID", raising=False)
    no_gpu_worker = create_worker(GatewayConfig(no_gpu=True))
    assert isinstance(no_gpu_worker, MockInferenceWorker)
    assert no_gpu_worker.worker_id == "worker-mock-0"

    gpu_worker = create_worker(GatewayConfig(no_gpu=False, worker_id="worker-gpu-live"))
    assert isinstance(gpu_worker, VLLMWorkerAdapter)
    assert gpu_worker.worker_id == "worker-gpu-live"


@pytest.mark.asyncio
async def test_backend_cors_preflight_and_exposed_headers():
    """Verify FastAPI CORSMiddleware allows browser origins and records 6 milestones."""
    mock_worker = MockInferenceWorker(worker_id="worker-gpu-0")
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: mock_worker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. Preflight OPTIONS request
        preflight = await client.options(
            "/v1/chat/completions",
            headers={
                "Origin": "http://127.0.0.1:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type, X-Request-ID",
            },
        )
        assert preflight.status_code == 200
        assert preflight.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"

        # 2. Streaming POST request exposes X-Request-ID and X-Worker-ID
        resp = await client.post(
            "/v1/chat/completions",
            headers={"Origin": "http://127.0.0.1:3000", "X-Request-ID": "trace-req-01"},
            json={
                "model": "Qwen/Qwen2.5-0.5B-Instruct",
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200
        assert resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"
        exposed = resp.headers.get("access-control-expose-headers", "")
        assert "X-Request-ID" in exposed
        assert "X-Worker-ID" in exposed
        assert resp.headers.get("X-Worker-ID") == "worker-gpu-0"

        # 3. Verify all 6 Week 3 trace milestones in /v1/traces/{request_id}
        trace_resp = await client.get("/v1/traces/trace-req-01")
        assert trace_resp.status_code == 200
        trace_data = trace_resp.json()
        assert trace_data["request_id"] == "trace-req-01"
        assert trace_data["worker_id"] == "worker-gpu-0"
        assert trace_data["status"] == "completed"
        assert trace_data["token_chunks"] == 5
        milestones = trace_data["milestones"]
        for key in (
            "arrival",
            "routing_start",
            "routing_end",
            "worker_start",
            "first_token",
            "last_token",
        ):
            assert milestones[key] is not None, f"Missing milestone: {key}"
        assert trace_data["metrics_ms"]["routing_ms"] is not None
        assert trace_data["metrics_ms"]["ttft_ms"] is not None
        assert trace_data["metrics_ms"]["tpot_ms"] is None
        assert trace_data["metrics_ms"]["inter_content_chunk_ms"] is not None
        assert trace_data["metrics_ms"]["e2e_ms"] is not None

        # 4. Verify /v1/traces list endpoint
        list_resp = await client.get("/v1/traces")
        assert list_resp.status_code == 200
        assert len(list_resp.json()["traces"]) >= 1


@pytest.mark.asyncio
async def test_cors_headers_preserved_on_400_and_503_errors():
    """Verify outermost CORSMiddleware attaches CORS headers on 400 and 503 error responses."""
    mock_worker = MockInferenceWorker(worker_id="worker-gpu-down", is_healthy=False)
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: mock_worker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # 1. 400 Validation Error includes CORS header
        bad_resp = await client.post(
            "/v1/chat/completions",
            headers={"Origin": "http://127.0.0.1:3000"},
            json={"model": "test-model", "messages": []},
        )
        assert bad_resp.status_code == 400
        assert bad_resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"

        # 2. 503 Unhealthy /health response includes CORS header and X-Worker-ID
        health_resp = await client.get(
            "/health",
            headers={"Origin": "http://127.0.0.1:3000"},
        )
        assert health_resp.status_code == 503
        assert health_resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"
        assert health_resp.headers.get("X-Worker-ID") == "worker-gpu-down"


@pytest.mark.asyncio
async def test_trace_records_error_status_and_404_on_unknown_trace():
    """Verify failed worker streams record status='error' in trace store and unknown IDs 404."""
    failing_worker = MockInferenceWorker(
        worker_id="worker-err-0",
        raise_error_before_stream=WorkerUnavailableError("Backend GPU offline", status_code=503),
    )
    app = create_app()
    app.dependency_overrides[get_worker] = lambda: failing_worker

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/v1/chat/completions",
            headers={"X-Request-ID": "trace-err-01"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            },
        )
        assert resp.status_code == 200

        trace_resp = await client.get("/v1/traces/trace-err-01")
        assert trace_resp.status_code == 200
        data = trace_resp.json()
        assert data["status"] == "error"
        assert "Backend GPU offline" in (data["error"] or "")
        assert data["milestones"]["arrival"] is not None
        assert data["milestones"]["last_token"] is not None

        not_found = await client.get("/v1/traces/does-not-exist")
        assert not_found.status_code == 404


def test_in_memory_trace_store_fifo_eviction():
    """Verify InMemoryTraceStore evicts oldest trace when max_size is exceeded."""
    store = InMemoryTraceStore(max_size=3)
    for i in range(1, 5):
        t = RequestTrace(request_id=f"req-{i}", model="m", worker_id="w-0")
        t.mark_arrival()
        store.save(t)

    assert store.get("req-1") is None
    assert store.get("req-2") is not None
    assert store.get("req-4") is not None
    recent_ids = [item["request_id"] for item in store.list_recent(limit=10)]
    assert recent_ids == ["req-4", "req-3", "req-2"]
