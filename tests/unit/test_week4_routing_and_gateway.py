"""Week 4 unit tests: RoundRobinRouter, multi-worker factory, control mounting & failover."""

from collections.abc import AsyncIterator
from datetime import datetime, timezone

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from llm_serving_platform.common.config import GatewayConfig
from llm_serving_platform.control.registry import WorkerRegistry, WorkerStatus
from llm_serving_platform.gateway.main import create_app, lifespan
from llm_serving_platform.routing.router import RoundRobinRouter
from llm_serving_platform.telemetry.trace import RequestTrace
from llm_serving_platform.workers.base import WorkerUnavailableError
from llm_serving_platform.workers.factory import populate_registry
from llm_serving_platform.workers.mock_adapter import MockInferenceWorker
from llm_serving_platform.workers.vllm_adapter import VLLMWorkerAdapter

NOW = datetime(2026, 11, 2, tzinfo=timezone.utc)


def test_round_robin_router_cycles_healthy_workers_and_skips_unhealthy():
    """Verify RoundRobinRouter distributes evenly across HEALTHY workers and excludes UNHEALTHY."""
    registry = WorkerRegistry()
    for wid in ("worker-0", "worker-1", "worker-2"):
        w = MockInferenceWorker(worker_id=wid)
        registry.register(w, f"mock://{wid}", "test-model")
        registry.record_health(wid, True, checked_at=NOW)

    router = RoundRobinRouter(registry)
    assert router.registry is registry

    selected_ids: list[str] = []
    for i in range(6):
        trace = RequestTrace(request_id=f"req-{i}", model="test-model")
        trace.mark_arrival()
        chosen = router.select_worker(trace)
        assert trace.routing_start is not None
        assert trace.routing_end is not None
        assert trace.worker_id == chosen.worker_id
        selected_ids.append(chosen.worker_id)

    assert selected_ids == [
        "worker-0",
        "worker-1",
        "worker-2",
        "worker-0",
        "worker-1",
        "worker-2",
    ]

    # Mark worker-1 UNHEALTHY -> router must only select worker-0 and worker-2
    registry.record_health("worker-1", False, checked_at=NOW)
    failover_ids: list[str] = []
    for i in range(4):
        trace = RequestTrace(request_id=f"failover-{i}", model="test-model")
        trace.mark_arrival()
        chosen = router.select_worker(trace)
        failover_ids.append(chosen.worker_id)

    assert "worker-1" not in failover_ids
    assert set(failover_ids) == {"worker-0", "worker-2"}


def test_round_robin_router_raises_503_when_no_healthy_workers():
    """Verify RoundRobinRouter raises WorkerUnavailableError(503) when 0 healthy workers exist."""
    registry = WorkerRegistry()
    registry.register(MockInferenceWorker(worker_id="worker-0"), "mock://worker-0", "test-model")
    router = RoundRobinRouter(registry)

    trace = RequestTrace(request_id="req-none-1", model="test-model")
    trace.mark_arrival()
    with pytest.raises(WorkerUnavailableError) as exc_info:
        router.select_worker(trace)
    assert exc_info.value.status_code == 503
    assert trace.worker_id == "none"

    registry.record_health("worker-0", False, checked_at=NOW)
    trace2 = RequestTrace(request_id="req-none-2", model="test-model")
    trace2.mark_arrival()
    with pytest.raises(WorkerUnavailableError):
        router.select_worker(trace2)
    assert trace2.worker_id == "none"


def test_populate_registry_both_gpu_and_no_gpu():
    """Verify populate_registry reuses create_worker for both No-GPU and Live GPU configs."""
    cfg_mock = GatewayConfig(no_gpu=True, mock_worker_count=3, model_name="mock-qwen")
    reg_mock = WorkerRegistry()
    workers_mock = populate_registry(reg_mock, cfg_mock)
    assert len(workers_mock) == 3
    assert [w.worker_id for w in reg_mock.healthy_workers()] == [
        "worker-mock-0",
        "worker-mock-1",
        "worker-mock-2",
    ]

    cfg_gpu = GatewayConfig(
        no_gpu=False,
        worker_endpoints=[
            "gpu-node-a=http://10.0.0.1:8000",
            "gpu-node-b=http://10.0.0.2:8000",
        ],
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )
    reg_gpu = WorkerRegistry()
    workers_gpu = populate_registry(reg_gpu, cfg_gpu)
    assert len(workers_gpu) == 2
    assert isinstance(workers_gpu[0], VLLMWorkerAdapter)
    assert workers_gpu[0].worker_id == "gpu-node-a"
    assert reg_gpu.get_record("gpu-node-a").endpoint == "http://10.0.0.1:8000"
    assert workers_gpu[1].worker_id == "gpu-node-b"
    assert reg_gpu.get_record("gpu-node-b").endpoint == "http://10.0.0.2:8000"


@pytest.mark.asyncio
async def test_gateway_mounts_control_router_and_round_robin_failover():
    """Verify Gateway mounts /control/v1/workers with CORS, routes Round-Robin, and handles 503."""
    cfg = GatewayConfig(no_gpu=True, mock_worker_count=2, model_name="test-model")
    app = create_app(cfg)
    registry: WorkerRegistry = app.state.registry
    monitor = app.state.health_monitor
    workers = dict(registry.probe_targets())

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        ctrl_resp = await client.get(
            "/control/v1/workers",
            headers={"Origin": "http://127.0.0.1:3000"},
        )
        assert ctrl_resp.status_code == 200
        assert ctrl_resp.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"
        records = ctrl_resp.json()
        assert [r["worker_id"] for r in records] == ["worker-mock-0", "worker-mock-1"]
        assert all(r["status"] == "HEALTHY" for r in records)

        routed: list[str] = []
        for idx in range(4):
            resp = await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": f"w4-rr-{idx}"},
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            routed.append(resp.headers["X-Worker-ID"])
        assert routed == ["worker-mock-0", "worker-mock-1", "worker-mock-0", "worker-mock-1"]

        workers["worker-mock-0"].is_healthy = False
        await monitor.check_once()
        for idx in range(3):
            resp = await client.post(
                "/v1/chat/completions",
                headers={"X-Request-ID": f"w4-degraded-{idx}"},
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "ping"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert resp.headers["X-Worker-ID"] == "worker-mock-1"

        workers["worker-mock-1"].is_healthy = False
        await monitor.check_once()
        resp_503 = await client.post(
            "/v1/chat/completions",
            headers={"Origin": "http://127.0.0.1:3000", "X-Request-ID": "w4-no-worker"},
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "ping"}],
                "stream": True,
            },
        )
        assert resp_503.status_code == 503
        assert resp_503.headers.get("access-control-allow-origin") == "http://127.0.0.1:3000"
        assert resp_503.headers.get("X-Worker-ID") == "none"
        err_payload = resp_503.json()
        assert err_payload["error"]["code"] == 503

        trace_503 = (await client.get("/v1/traces/w4-no-worker")).json()
        assert trace_503["status"] == "error"
        assert trace_503["worker_id"] == "none"


@pytest.mark.asyncio
async def test_live_gpu_mode_vllm_worker_pool_round_robin_and_failover():
    """Verify no_gpu=False mode with multiple VLLMWorkerAdapters via httpx.MockTransport."""
    healthy_hosts = {"vllm-gpu-0", "vllm-gpu-1"}

    async def sse_stream() -> AsyncIterator[bytes]:
        yield b'data: {"choices": [{"delta": {"content": "GPU token"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    def vllm_wire_handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if request.url.path == "/health":
            return httpx.Response(200 if host in healthy_hosts else 503)
        if request.url.path == "/v1/chat/completions":
            if host not in healthy_hosts:
                return httpx.Response(503, text="GPU Down")
            return httpx.Response(200, content=sse_stream())
        return httpx.Response(404)

    cfg = GatewayConfig(
        no_gpu=False,
        worker_endpoints=[
            "worker-vllm-0=http://vllm-gpu-0:8000",
            "worker-vllm-1=http://vllm-gpu-1:8000",
        ],
        model_name="Qwen/Qwen2.5-0.5B-Instruct",
    )
    app = create_app(cfg)
    mock_vllm_client = httpx.AsyncClient(transport=httpx.MockTransport(vllm_wire_handler))
    app.state.client = mock_vllm_client
    for _, worker in app.state.registry.probe_targets():
        if isinstance(worker, VLLMWorkerAdapter):
            worker.client = mock_vllm_client

    try:
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            routed: list[str] = []
            for idx in range(4):
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Request-ID": f"gpu-rr-{idx}"},
                    json={
                        "model": "Qwen/Qwen2.5-0.5B-Instruct",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                assert resp.status_code == 200
                routed.append(resp.headers["X-Worker-ID"])
            assert routed == [
                "worker-vllm-0",
                "worker-vllm-1",
                "worker-vllm-0",
                "worker-vllm-1",
            ]

            healthy_hosts.remove("vllm-gpu-0")
            await app.state.health_monitor.check_once()
            for idx in range(2):
                resp = await client.post(
                    "/v1/chat/completions",
                    headers={"X-Request-ID": f"gpu-failover-{idx}"},
                    json={
                        "model": "Qwen/Qwen2.5-0.5B-Instruct",
                        "messages": [{"role": "user", "content": "hello"}],
                        "stream": True,
                    },
                )
                assert resp.status_code == 200
                assert resp.headers["X-Worker-ID"] == "worker-vllm-1"
    finally:
        await mock_vllm_client.aclose()


@pytest.mark.asyncio
async def test_gateway_lifespan_runs_initial_probe_and_cancels_background_task():
    """Verify FastAPI lifespan runs initial health check and cleanly cancels periodic task."""
    cfg = GatewayConfig(
        no_gpu=True,
        mock_worker_count=2,
        health_check_interval_s=0.05,
        health_check_timeout_s=1.0,
    )
    app = create_app(cfg)
    async with lifespan(app):
        assert app.state.initial_health_checked is True
        assert app.state.health_task is not None
        assert not app.state.health_task.done()
        workers = app.state.registry.list_workers()
        assert len(workers) == 2
        assert all(w.status == WorkerStatus.HEALTHY for w in workers)
    assert app.state.health_task.cancelled() or app.state.health_task.done()
