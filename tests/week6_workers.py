"""Controlled workers and ASGI disconnect driver for Week 6 tests and the CPU verifier."""

import asyncio
import json
from collections import Counter
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx

from llm_serving_platform.common.config import GatewayConfig
from llm_serving_platform.control.registry import WorkerRegistry
from llm_serving_platform.gateway.main import create_app
from llm_serving_platform.workers.base import (
    InferenceWorker,
    WorkerTimeoutError,
    WorkerUnavailableError,
)


class ControlledWorker(InferenceWorker):
    def __init__(self, worker_id, controls):
        self.worker_id = worker_id
        self.controls = controls
        self.healthy = True

    async def check_health(self) -> bool:
        return self.healthy

    async def stream_chat(self, payload, request_id) -> AsyncIterator[str]:
        controls = self.controls
        controls.calls[request_id, self.worker_id] += 1
        try:
            if gate := controls.gates.get(request_id):
                await asyncio.wait_for(gate.wait(), timeout=10)
            else:
                await asyncio.sleep(0.30 if payload["max_tokens"] >= 64 else 0.04)
            failure = controls.failures.get(request_id)
            if failure == "pre":
                raise WorkerUnavailableError("Controlled failure before content")
            if failure == "timeout":
                try:
                    await asyncio.wait_for(asyncio.Event().wait(), timeout=0.01)
                except TimeoutError as exc:
                    raise WorkerTimeoutError("Controlled worker deadline") from exc
            if failure == "unexpected":
                raise RuntimeError("Controlled unexpected worker failure")
            yield 'data: {"choices": [{"delta": {"content": "OK"}}]}\n\n'
            if failure == "mid":
                raise WorkerUnavailableError("Controlled failure after content")
            yield "data: [DONE]\n\n"
        finally:
            controls.closed[request_id] += 1


class WorkerControls:
    def __init__(self):
        self.workers = [ControlledWorker(f"worker-mock-{i}", self) for i in range(2)]
        self.gates: dict[str, asyncio.Event] = {}
        self.failures: dict[str, str] = {}
        self.calls: Counter = Counter()
        self.closed: Counter = Counter()

    def hold(self, request_id):
        self.gates[request_id] = asyncio.Event()

    def finish(self, request_id):
        self.gates[request_id].set()

    def calls_for(self, request_id):
        return sum(self.calls[request_id, worker.worker_id] for worker in self.workers)

    def release_all(self):
        for gate in self.gates.values():
            gate.set()


@asynccontextmanager
async def mock_gateway(policy, model):
    controls = WorkerControls()
    registry = WorkerRegistry()
    for worker in controls.workers:
        registry.register(worker, f"mock://{worker.worker_id}", model)
    config = GatewayConfig(no_gpu=True, routing_policy=policy, health_check_interval_s=0)
    app = create_app(config=config, registry=registry)
    async with app.router.lifespan_context(app):
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://mock-gateway"
        ) as client:
            try:
                yield app, client, controls
            finally:
                controls.release_all()


async def disconnect_chat(
    app,
    request_id,
    payload,
    disconnect,
    on_message,
    *,
    spec_version="2.3",
):
    """Send a real ASGI disconnect while the worker is blocked producing its stream."""
    sent_request = False

    async def receive():
        nonlocal sent_request
        if not sent_request:
            sent_request = True
            return {"type": "http.request", "body": json.dumps(payload).encode()}
        await disconnect.wait()
        return {"type": "http.disconnect"}

    await app(
        {
            "type": "http",
            "asgi": {"version": "3.0", "spec_version": spec_version},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/v1/chat/completions",
            "query_string": b"",
            "headers": [
                (b"content-type", b"application/json"),
                (b"x-request-id", request_id.encode()),
            ],
        },
        receive,
        on_message,
    )
