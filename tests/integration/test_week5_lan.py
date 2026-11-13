"""CPU fixtures exercise real Gateway/adapters and reject misleading Week 5 evidence."""

import json

import httpx
import pytest

from infra.scripts.verify_week5_lan import run_phase
from llm_serving_platform.common.config import GatewayConfig
from llm_serving_platform.gateway import main as gateway

A, B, GATEWAY = "http://a.test:8000", "http://b.test:8000", "http://gateway.test"
IDS = ["worker-vllm-a", "worker-vllm-b"]


@pytest.mark.asyncio
async def test_operator_phases_with_real_gateway_and_remote_adapters(monkeypatch, tmp_path):
    healthy = {"a.test": True, "b.test": True}

    def worker(request):
        if not healthy[request.url.host]:
            raise httpx.ConnectError("fixture worker stopped", request=request)
        if request.url.path == "/health":
            return httpx.Response(200)
        assert request.url.path == "/v1/chat/completions"
        event = {"choices": [{"delta": {"content": "hello"}}]}
        return httpx.Response(200, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")

    worker_transport = httpx.MockTransport(worker)
    monkeypatch.setattr(gateway, "_build_http_client", lambda _: httpx.AsyncClient(
        transport=worker_transport,
    ))
    app = gateway.create_app(GatewayConfig(
        no_gpu=False, worker_endpoints=[f"{IDS[0]}={A}", f"{IDS[1]}={B}"],
        health_check_interval_s=0.01, health_check_timeout_s=0.1,
    ))
    asgi = httpx.ASGITransport(app=app)

    async def dispatch(request):
        if request.url.host == "gateway.test":
            return await asgi.handle_async_request(request)
        return worker(request)

    async with gateway.lifespan(app), httpx.AsyncClient(
        transport=httpx.MockTransport(dispatch),
    ) as client:
        # Start at the other valid cursor phase: B A B A must pass too.
        app.state.router._cursor = 1
        for phase, b_up in (("baseline", True), ("worker-b-down", False), ("recovered", True)):
            healthy["b.test"] = b_up
            output = tmp_path / f"{phase}.json"
            evidence = await run_phase(
                phase=phase, gateway_url=GATEWAY, worker_a_url=A, worker_b_url=B,
                wait_timeout_s=2, request_timeout_s=2, output_path=output, client=client,
            )
            assert evidence["passed"], evidence["failure_details"]
            assert json.loads(output.read_text()) == evidence
            assert len(evidence["requests"]) == 4
            assert set(evidence["worker_assignments"]) == (set(IDS) if b_up else {IDS[0]})
            assert "preview" not in output.read_text()
            assert "prompt" not in output.read_text()


@pytest.mark.asyncio
@pytest.mark.parametrize("fault", [
    "stale-registry", "wrong-endpoint", "bad-request-id", "bad-trace-worker",
    "missing-milestone", "empty-stream", "not-round-robin", "b-still-up",
])
async def test_bad_observations_fail_and_save_evidence(fault, tmp_path):
    calls = 0
    assignments = {}

    def fixture(request):
        nonlocal calls
        if request.url.path == "/health":
            return httpx.Response(200)
        if request.url.path == "/control/v1/workers":
            return httpx.Response(200, json=[
                {"worker_id": worker, "endpoint": "http://wrong.test" if fault == "wrong-endpoint"
                 else url, "status": "UNHEALTHY" if fault == "stale-registry" else "HEALTHY"}
                for worker, url in zip(IDS, (A, B))
            ])
        if request.url.path == "/v1/chat/completions":
            worker_id = IDS[0] if fault == "not-round-robin" else IDS[calls % 2]
            calls += 1
            request_id = request.headers["X-Request-ID"]
            assignments[request_id] = worker_id
            content = "" if fault == "empty-stream" else "hello"
            event = {"choices": [{"delta": {"content": content}}]}
            return httpx.Response(200, headers={
                "X-Request-ID": "wrong" if fault == "bad-request-id" else request_id,
                "X-Worker-ID": worker_id,
            }, text=f"data: {json.dumps(event)}\n\ndata: [DONE]\n\n")
        request_id = request.url.path.rsplit("/", 1)[1]
        return httpx.Response(200, json={
            "request_id": request_id, "worker_id": "wrong" if fault == "bad-trace-worker"
            else assignments[request_id], "status": "completed",
            "milestones": {key: "timestamp" for key in (
                "arrival", "routing_start", "routing_end", "worker_start", "first_token",
                *(() if fault == "missing-milestone" else ("last_token",)),
            )},
        })

    output = tmp_path / "failed.json"
    async with httpx.AsyncClient(transport=httpx.MockTransport(fixture)) as client:
        evidence = await run_phase(
            phase="worker-b-down" if fault == "b-still-up" else "baseline",
            gateway_url=GATEWAY, worker_a_url=A, worker_b_url=B,
            client=client, wait_timeout_s=0.02, output_path=output,
        )
    assert not evidence["passed"]
    assert evidence["failure_details"]
    assert json.loads(output.read_text())["passed"] is False
    if fault == "stale-registry":
        assert evidence["failure_details"][0]["type"] == "TimeoutError"
