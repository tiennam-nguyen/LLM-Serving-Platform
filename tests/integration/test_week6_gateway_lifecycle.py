"""Exercise reservation ownership through the real Gateway ASGI app."""

import asyncio
from unittest.mock import patch

import pytest
from starlette.requests import ClientDisconnect

from infra.scripts.verify_week6_e2e import (
    CHAT,
    Scenario,
    cancel,
    chat_payload,
    duplicate,
    loads,
    mock_phase,
    worker_failure,
)
from llm_serving_platform.control.registry import DuplicateRequestReservationError
from tests.week6_workers import disconnect_chat


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["pre", "mid", "timeout", "unexpected"])
async def test_reserved_worker_failure_drains_without_reroute(failure):
    code, chunks = {"pre": (503, 0), "mid": (503, 1), "timeout": (504, 0), "unexpected": (500, 0)}[
        failure
    ]
    async with mock_phase("least_loaded", "test-model") as phase:
        result = await Scenario(phase, failure, "Worker failure").run(
            lambda s: worker_failure(s, failure, code, chunks)
        )
        assert result["status"] == "PASS", result


@pytest.mark.asyncio
async def test_http_disconnect_releases_reserved_stream():
    async with mock_phase("least_loaded", "test-model") as phase:
        result = await Scenario(phase, "cancel", "ASGI client disconnect").run(cancel)
        assert result["status"] == "PASS", result


@pytest.mark.asyncio
async def test_duplicate_active_request_preserves_original_trace_and_exact_load():
    async with mock_phase("least_loaded", "test-model") as phase:
        result = await Scenario(phase, "duplicate", "Active duplicate").run(duplicate)
        assert result["status"] == "PASS", result


@pytest.mark.asyncio
async def test_duplicate_preserves_original_even_when_all_workers_become_unhealthy():
    async with mock_phase("least_loaded", "test-model") as phase:
        scenario = Scenario(phase, "duplicate_health", "Active duplicate when all workers are down")
        await phase.workers()
        row, task = scenario.start("original", hold=True, request_id="duplicate-active-1")
        request_id = row["request_id"]
        try:
            before = await scenario.snapshot(
                "active", lambda state: sum(loads(state).values()) == 1
            )
            assert before is not None
            trace = await phase.trace(request_id)
            for worker in phase.controls.workers:
                worker.healthy = False
            await phase.app.state.health_monitor.check_once()
            down = await phase.workers()
            assert all(row["status"] == "UNHEALTHY" for row in down)
            _, rejected = scenario.start("duplicate", request_id=request_id, expected="duplicate")
            await rejected
            assert await phase.trace(request_id) == trace
            assert loads(await phase.workers()) == loads(before)
            phase.controls.finish(request_id)
            await asyncio.wait_for(task, 2)
        finally:
            phase.controls.release_all()
            await asyncio.gather(task, return_exceptions=True)
        assert phase.controls.calls_for(request_id) == 1
        assert not any(loads(await phase.workers()).values())


@pytest.mark.asyncio
async def test_duplicate_during_pre_trace_window_writes_no_trace():
    async with mock_phase("least_loaded", "test-model") as phase:
        # Recreate the reservation state that precedes the original trace save.
        registry = phase.app.state.registry
        worker_id = (await phase.workers())[0]["worker_id"]
        request_id = "duplicate-active-1"
        registry.reserve_request(worker_id, request_id)
        try:
            response = await phase.client.post(
                CHAT,
                headers={"X-Request-ID": request_id},
                json=chat_payload(phase.model, "short", 8),
            )
            assert response.status_code == 409
            assert response.headers["X-Request-ID"] == request_id
            assert response.headers["X-Worker-ID"] == "none"
            assert await phase.trace(request_id) is None
            assert registry.active_request_count(worker_id) == 1
            with pytest.raises(DuplicateRequestReservationError):
                registry.reserve_request(worker_id, request_id)
        finally:
            assert registry.release_request(request_id)
        assert not any(loads(await phase.workers()).values())
        assert phase.controls.calls_for(request_id) == 0


@pytest.mark.asyncio
async def test_response_setup_failure_releases_before_propagating():
    async with mock_phase("least_loaded", "test-model") as phase:
        request_id = "w6-response-setup-failure"
        headers = {"X-Request-ID": request_id}
        payload = chat_payload(phase.model, "short", 8)
        with patch(
            "llm_serving_platform.gateway.main._ReservationStreamResponse",
            side_effect=RuntimeError("controlled response setup failure"),
        ):
            with pytest.raises(RuntimeError):
                await phase.client.post(CHAT, headers=headers, json=payload)

        trace = await phase.trace(request_id)
        assert trace["status"] == "error"
        assert trace["routing"]["selected_load_after"] == 1
        assert not any(loads(await phase.workers()).values())
        assert phase.controls.calls_for(request_id) == 0
        response = await phase.client.post(CHAT, headers=headers, json=payload)
        assert response.status_code == 200
        assert (await phase.trace(request_id))["status"] == "completed"
        assert not any(loads(await phase.workers()).values())


@pytest.mark.asyncio
async def test_lost_asgi_send_closes_worker_and_releases_reservation():
    async with mock_phase("least_loaded", "test-model") as phase:
        request_id = "w6-disconnect-during-body-send"
        disconnect = asyncio.Event()

        async def on_message(message):
            if message["type"] == "http.response.body" and message.get("more_body"):
                raise OSError("client closed the ASGI connection")

        with pytest.raises((ClientDisconnect, OSError)):
            await asyncio.wait_for(
                disconnect_chat(
                    phase.app,
                    request_id,
                    chat_payload(phase.model, "short", 8),
                    disconnect,
                    on_message,
                    spec_version="2.4",
                ),
                timeout=2,
            )
        trace = await phase.trace(request_id)
        assert trace["milestones"]["first_token"] is not None
        assert trace["status"] == "cancelled"
        assert phase.controls.closed[request_id] == 1
        assert not any(loads(await phase.workers()).values())
