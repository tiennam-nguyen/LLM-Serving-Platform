"""CPU-only contracts for the independent Member B control plane."""

import asyncio
from dataclasses import FrozenInstanceError, asdict
from datetime import datetime, timedelta, timezone

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from llm_serving_platform.control.api import create_control_router
from llm_serving_platform.control.health import HealthMonitor
from llm_serving_platform.control.registry import (
    DuplicateWorkerError,
    UnknownWorkerError,
    WorkerRegistry,
    WorkerStatus,
)
from llm_serving_platform.workers.mock_adapter import MockInferenceWorker

NOW = datetime(2026, 11, 2, tzinfo=timezone.utc)


def test_registry_snapshots_identity_and_errors():
    registry = WorkerRegistry()
    b = MockInferenceWorker(worker_id="b")
    a = MockInferenceWorker(worker_id="a")
    registry.register(b, "mock://b", "model")
    original = registry.register(a, "mock://a", "model")
    assert original.status == WorkerStatus.STARTING
    assert original.last_checked_at is None
    assert registry.healthy_workers() == []
    assert [r.worker_id for r in registry.list_workers()] == ["a", "b"]
    with pytest.raises(DuplicateWorkerError):
        registry.register(MockInferenceWorker(worker_id="a"), "other", "other")
    with pytest.raises(UnknownWorkerError):
        registry.record_health("missing", True)
    with pytest.raises(UnknownWorkerError):
        registry.get_record("missing")
    with pytest.raises(FrozenInstanceError):
        original.model = "changed"
    registry.list_workers().clear()
    registry.record_health("a", True, checked_at=NOW)
    assert registry.healthy_workers()[0] is a
    registry.healthy_workers().clear()
    assert len(registry.healthy_workers()) == 1
    assert original.status == WorkerStatus.STARTING
    assert "worker" not in asdict(registry.get_record("a"))
    registry.record_health("a", False, checked_at=NOW + timedelta(seconds=1))
    assert registry.healthy_workers() == []
    registry.record_health("a", True, checked_at=NOW)
    assert registry.get_record("a").status == WorkerStatus.HEALTHY


def test_health_timestamps_and_validation():
    registry = WorkerRegistry()
    registry.register(MockInferenceWorker(worker_id="a"), "mock://a", "model")
    checked = registry.record_health(
        "a", True, checked_at=NOW.astimezone(timezone(timedelta(hours=7)))
    )
    assert checked.last_checked_at == NOW
    assert checked.last_checked_at.tzinfo == timezone.utc
    with pytest.raises(ValueError):
        registry.record_health("a", True, checked_at=datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        registry.record_health("a", "yes")
    with pytest.raises(ValueError):
        registry.register(MockInferenceWorker(worker_id=""), "mock://a", "model")
    assert registry.record_health("a", True).last_checked_at.tzinfo == timezone.utc


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan")])
def test_invalid_timeout(timeout):
    with pytest.raises(ValueError):
        HealthMonitor(WorkerRegistry(), timeout_s=timeout)


@pytest.mark.asyncio
async def test_cycle_failure_recovery_and_per_worker_updates():
    registry = WorkerRegistry()
    fast = MockInferenceWorker(worker_id="fast")
    failing = MockInferenceWorker(worker_id="failing", is_healthy=False)
    entered, release = asyncio.Event(), asyncio.Event()

    class SlowWorker(MockInferenceWorker):
        async def check_health(self):
            entered.set()
            await release.wait()
            return True

    class RaisingWorker(MockInferenceWorker):
        async def check_health(self):
            raise RuntimeError("probe failed")

    slow = SlowWorker(worker_id="slow")
    raising = RaisingWorker(worker_id="raising")
    for worker in (slow, fast, failing, raising):
        registry.register(worker, f"mock://{worker.worker_id}", "model")
    monitor = HealthMonitor(registry, timeout_s=0.05, clock=lambda: NOW)
    cycle = asyncio.create_task(monitor.check_once())
    await entered.wait()
    # Fast probes complete before the slow probe is released or times out.
    while registry.get_record("fast").status == WorkerStatus.STARTING:
        await asyncio.sleep(0)
    assert not cycle.done()
    assert registry.healthy_workers() == [fast]
    await cycle
    assert all(r.last_checked_at == NOW for r in registry.list_workers())
    assert [r.worker_id for r in registry.list_workers() if r.status == WorkerStatus.UNHEALTHY] == [
        "failing",
        "raising",
        "slow",
    ]
    failing.is_healthy = True
    release.set()
    await monitor.check_once()
    assert registry.healthy_workers() == [failing, fast, slow]


@pytest.mark.asyncio
async def test_cancelled_cycle_does_not_mark_failure():
    entered = asyncio.Event()

    class WaitingWorker(MockInferenceWorker):
        async def check_health(self):
            entered.set()
            await asyncio.Event().wait()

    registry = WorkerRegistry()
    registry.register(WaitingWorker(worker_id="wait"), "mock://wait", "model")
    cycle = asyncio.create_task(HealthMonitor(registry).check_once())
    await entered.wait()
    cycle.cancel()
    with pytest.raises(asyncio.CancelledError):
        await cycle
    assert registry.get_record("wait").status == WorkerStatus.STARTING
    await HealthMonitor(WorkerRegistry()).check_once()


@pytest.mark.asyncio
async def test_read_only_api_contract_without_gateway():
    registry = WorkerRegistry()
    app = FastAPI()
    app.include_router(create_control_router(registry))
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        assert (await client.get("/control/v1/workers")).json() == []
        registry.register(MockInferenceWorker(worker_id="one"), "mock://one", "model")
        registry.record_health("one", True, checked_at=NOW)
        response = await client.get("/control/v1/workers")
        assert response.status_code == 200
        row = response.json()[0]
        assert row == {
            "worker_id": "one",
            "endpoint": "mock://one",
            "model": "model",
            "status": "HEALTHY",
            "last_checked_at": "2026-11-02T00:00:00Z",
        }
        assert (await client.get("/control/v1/workers/one")).json() == row
        assert (await client.get("/control/v1/workers/missing")).status_code == 404
        assert (await client.post("/control/v1/workers", json={})).status_code == 405
