"""Member A's public HTTP contracts and lifecycle checks, using CPU-only workers."""

import asyncio
import json
import logging
from unittest.mock import patch

import pytest
from prometheus_client.parser import text_string_to_metric_families
from starlette.requests import ClientDisconnect

from infra.scripts.verify_week6_e2e import chat_payload
from llm_serving_platform.gateway.main import _ReservationStreamResponse
from llm_serving_platform.workers.base import WorkerUnavailableError
from tests.week6_workers import disconnect_chat, mock_gateway

CHAT = "/v1/chat/completions"
POLICIES = ("round_robin", "least_loaded", "single")


async def scrape(client):
    response = await client.get("/metrics")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain; version=")
    samples = [
        sample
        for family in text_string_to_metric_families(response.text)
        for sample in family.samples
    ]
    allowed = {
        "llm_gateway_requests": {"status"},
        "llm_gateway_request_e2e_seconds": set(),
        "llm_gateway_ttft_seconds": set(),
        "llm_router_decision_seconds": {"policy"},
        "llm_router_requests": {"policy", "worker_id"},
        "llm_worker_active_requests": {"worker_id"},
        "llm_worker_health": {"worker_id"},
    }
    for family in text_string_to_metric_families(response.text):
        name = family.name.removesuffix("_created")
        assert name in allowed, family.name
        for sample in family.samples:
            assert set(sample.labels) == allowed[name] | (
                {"le"} if sample.name.endswith("_bucket") else set()
            )
    return samples


def value(samples, name, **labels):
    matches = [s.value for s in samples if s.name == name and s.labels == labels]
    assert len(matches) == 1, (name, labels, samples)
    return matches[0]


def total(samples, name):
    return sum(s.value for s in samples if s.name == name)


async def wait_for_active(client, count):
    async with asyncio.timeout(3):
        while True:
            samples = await scrape(client)
            if total(samples, "llm_worker_active_requests") == count:
                return samples
            await asyncio.sleep(0.01)


async def assert_terminal(client, request_id, status):
    response = await client.get(f"/v1/traces/{request_id}")
    assert response.status_code == 200
    trace = response.json()
    assert trace["request_id"] == request_id
    assert trace["status"] == status
    assert trace["milestones"]["arrival"] is not None
    assert trace["milestones"]["last_token"] is not None
    samples = await scrape(client)
    assert value(samples, "llm_gateway_requests_total", status=status) == 1
    assert total(samples, "llm_gateway_requests_total") == 1
    assert value(samples, "llm_gateway_request_e2e_seconds_count") == 1
    assert value(samples, "llm_gateway_request_e2e_seconds_sum") == pytest.approx(
        trace["metrics_ms"]["e2e_ms"] / 1000
    )
    assert value(samples, "llm_gateway_ttft_seconds_count") == int(
        trace["milestones"]["first_token"] is not None
    )
    assert total(samples, "llm_worker_active_requests") == 0
    records = (await client.get("/control/v1/workers")).json()
    assert all(row["active_requests"] == 0 for row in records)
    # Repeated scrapes must not record observations or change cumulative metrics.
    assert await scrape(client) == samples
    return trace, samples


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", POLICIES)
async def test_completion_trace_metrics_identity_and_prompt_privacy(policy, caplog):
    caplog.set_level(logging.INFO, logger="llm_serving_platform.gateway")
    async with mock_gateway(policy, "test-model") as (_, client, _):
        payload = chat_payload("test-model", "short", 8)
        prompt = "private prompt that must never appear in observability"
        payload["messages"][0]["content"] = prompt
        response = await client.post(CHAT, json=payload)
        assert response.status_code == 200
        request_id = response.headers["X-Request-ID"]
        trace, samples = await assert_terminal(client, request_id, "completed")
        worker_id = response.headers["X-Worker-ID"]
        assert trace["model"] == "test-model"
        assert trace["worker_id"] == worker_id
        routing = trace["routing"]
        assert routing["policy"] == policy
        assert routing["selected_worker_id"] == worker_id
        assert {"candidate_loads", "selected_load_before", "selected_load_after", "reason"} <= (
            routing.keys()
        )
        assert all(trace["milestones"].values())
        assert set(trace["metrics_ms"]) == {
            "routing_ms",
            "ttft_ms",
            "tpot_ms",
            "inter_content_chunk_ms",
            "e2e_ms",
        }
        assert value(samples, "llm_router_requests_total", policy=policy, worker_id=worker_id) == 1
        assert value(samples, "llm_router_decision_seconds_count", policy=policy) == 1
        assert value(samples, "llm_router_decision_seconds_sum", policy=policy) == pytest.approx(
            trace["metrics_ms"]["routing_ms"] / 1000
        )
        assert value(samples, "llm_gateway_ttft_seconds_sum") == pytest.approx(
            trace["metrics_ms"]["ttft_ms"] / 1000
        )
        events = [json.loads(r.message) for r in caplog.records if r.name.endswith(".gateway")]
        terminal = [
            e for e in events if e["request_id"] == request_id and e["status"] == "completed"
        ]
        assert len(terminal) == 1
        assert prompt not in json.dumps(trace)
        assert prompt not in (await client.get("/metrics")).text
        assert prompt not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("failure", ["pre", "mid", "timeout", "unexpected"])
async def test_worker_failures_record_one_outcome_without_rerouting(policy, failure):
    async with mock_gateway(policy, "test-model") as (_, client, controls):
        request_id = "failed-request"
        controls.failures[request_id] = failure
        response = await client.post(
            CHAT, headers={"X-Request-ID": request_id}, json=chat_payload("test-model", "short", 8)
        )
        assert response.status_code == 200
        assert '"error"' in response.text
        trace, samples = await assert_terminal(client, request_id, "error")
        assert trace["error"]
        assert controls.calls_for(request_id) == 1
        assert controls.closed[request_id] == 1
        assert total(samples, "llm_router_requests_total") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["round_robin", "least_loaded"])
async def test_no_healthy_worker_preserves_failure_trace_and_observes_routing(policy):
    async with mock_gateway(policy, "test-model") as (app, client, controls):
        for worker in controls.workers:
            worker.healthy = False
        await app.state.health_monitor.check_once()
        response = await client.post(
            CHAT, headers={"X-Request-ID": "no-worker"}, json=chat_payload("test-model", "short", 8)
        )
        assert response.status_code == 503
        assert response.headers["X-Request-ID"] == "no-worker"
        trace, samples = await assert_terminal(client, "no-worker", "error")
        assert trace["worker_id"] == "none"
        assert trace["routing"]["policy"] == policy
        assert trace["routing"]["selected_worker_id"] is None
        assert total(samples, "llm_router_requests_total") == 0
        assert value(samples, "llm_router_decision_seconds_count", policy=policy) == 1
        assert total(samples, "llm_worker_health") == 0
        controls.workers[0].healthy = True
        # Scraping reads the cache; it never triggers another network health check.
        assert total(await scrape(client), "llm_worker_health") == 0
        await app.state.health_monitor.check_once()
        assert value(await scrape(client), "llm_worker_health", worker_id="worker-mock-0") == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", POLICIES)
async def test_real_client_disconnect_tracks_live_load_and_cleans_up_once(policy):
    async with mock_gateway(policy, "test-model") as (app, client, controls):
        request_id = "cancel-me"
        controls.hold(request_id)
        disconnect = asyncio.Event()

        async def send(_message):
            pass

        task = asyncio.create_task(
            disconnect_chat(
                app, request_id, chat_payload("test-model", "short", 8), disconnect, send
            )
        )
        try:
            samples = await wait_for_active(client, 1)
            assert total(samples, "llm_gateway_requests_total") == 0
            assert total(samples, "llm_router_requests_total") == 1
            trace = (await client.get(f"/v1/traces/{request_id}")).json()
            worker_id = trace["worker_id"]
            for worker in controls.workers:
                if worker.worker_id == worker_id:
                    worker.healthy = False
            await app.state.health_monitor.check_once()
            samples = await scrape(client)
            assert value(samples, "llm_worker_health", worker_id=worker_id) == 0
            assert value(samples, "llm_worker_active_requests", worker_id=worker_id) == 1
            disconnect.set()
            await asyncio.wait_for(task, 3)
        finally:
            disconnect.set()
            controls.release_all()
            await asyncio.gather(task, return_exceptions=True)
        await assert_terminal(client, request_id, "cancelled")
        assert controls.closed[request_id] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("failed_message", ["http.response.start", "http.response.body"])
async def test_lost_asgi_send_finalizes_even_when_generator_never_started(policy, failed_message):
    async with mock_gateway(policy, "test-model") as (app, client, controls):
        request_id = "lost-send"

        async def send(message):
            if message["type"] == failed_message:
                raise OSError("Client closed connection")

        with pytest.raises((ClientDisconnect, OSError)):
            await asyncio.wait_for(
                disconnect_chat(
                    app,
                    request_id,
                    chat_payload("test-model", "short", 8),
                    asyncio.Event(),
                    send,
                    spec_version="2.4",
                ),
                3,
            )
        await assert_terminal(client, request_id, "cancelled")
        assert controls.closed[request_id] <= 1


@pytest.mark.asyncio
async def test_duplicate_active_id_preserves_trace_and_does_not_add_an_inference():
    async with mock_gateway("least_loaded", "test-model") as (_, client, controls):
        request_id = "duplicate"
        controls.hold(request_id)
        arguments = {
            "headers": {"X-Request-ID": request_id},
            "json": chat_payload("test-model", "short", 8),
        }
        task = asyncio.create_task(client.post(CHAT, **arguments))
        try:
            await wait_for_active(client, 1)
            before = (await client.get(f"/v1/traces/{request_id}")).json()
            response = await client.post(CHAT, **arguments)
            assert response.status_code == 409
            assert (await client.get(f"/v1/traces/{request_id}")).json() == before
            samples = await scrape(client)
            assert total(samples, "llm_worker_active_requests") == 1
            assert total(samples, "llm_gateway_requests_total") == 0
            assert total(samples, "llm_router_requests_total") == 1
            assert total(samples, "llm_router_decision_seconds_count") == 1
        finally:
            controls.release_all()
            await asyncio.wait_for(task, 3)
        await assert_terminal(client, request_id, "completed")


@pytest.mark.asyncio
async def test_response_setup_failure_records_error_and_releases_load():
    async with mock_gateway("least_loaded", "test-model") as (_, client, _):
        with patch(
            "llm_serving_platform.gateway.main._ReservationStreamResponse",
            side_effect=RuntimeError("Controlled response construction failure"),
        ):
            with pytest.raises(RuntimeError):
                await client.post(
                    CHAT,
                    headers={"X-Request-ID": "setup-failure"},
                    json=chat_payload("test-model", "short", 8),
                )
        await assert_terminal(client, "setup-failure", "error")


@pytest.mark.asyncio
async def test_multiple_apps_have_isolated_metrics_and_trace_eviction_does_not_lose_counts():
    async with mock_gateway("round_robin", "test-model") as (first_app, first, _):
        async with mock_gateway("least_loaded", "test-model") as (_, second, _):
            first_app.state.trace_store.max_size = 1
            for index in range(4):
                response = await first.post(
                    CHAT,
                    headers={"X-Request-ID": f"req-{index}"},
                    json=chat_payload("test-model", "short", 8),
                )
                assert response.status_code == 200
            assert (await first.get("/v1/traces/req-0")).status_code == 404
            samples = await scrape(first)
            assert value(samples, "llm_gateway_requests_total", status="completed") == 4
            assert total(samples, "llm_router_requests_total") == 4
            assert total(samples, "llm_worker_active_requests") == 0
            assert total(await scrape(second), "llm_gateway_requests_total") == 0
            assert total(await scrape(second), "llm_router_requests_total") == 0


@pytest.mark.asyncio
async def test_repeated_cleanup_does_not_double_count_or_release_another_request():
    async with mock_gateway("least_loaded", "test-model") as (_, client, controls):
        payload = chat_payload("test-model", "short", 8)
        with patch(
            "llm_serving_platform.gateway.main._ReservationStreamResponse",
            wraps=_ReservationStreamResponse,
        ) as response_class:
            await client.post(CHAT, headers={"X-Request-ID": "finished"}, json=payload)
            finalize = response_class.call_args.args[2]
        controls.hold("still-running")
        task = asyncio.create_task(
            client.post(CHAT, headers={"X-Request-ID": "still-running"}, json=payload)
        )
        try:
            await wait_for_active(client, 1)
            finalize()
            finalize()
            samples = await scrape(client)
            assert total(samples, "llm_gateway_requests_total") == 1
            assert total(samples, "llm_worker_active_requests") == 1
        finally:
            controls.release_all()
            await asyncio.wait_for(task, 3)
        samples = await scrape(client)
        assert total(samples, "llm_gateway_requests_total") == 2
        assert total(samples, "llm_worker_active_requests") == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("error_kind", ["exception", "internal_exception", "sse"])
async def test_prompt_echoed_by_worker_error_is_not_persisted(error_kind, caplog):
    caplog.set_level(logging.INFO, logger="llm_serving_platform.gateway")
    async with mock_gateway("single", "test-model") as (_, client, controls):
        prompt = 'Nội dung riêng: "không lưu"\nprivate data'
        payload = chat_payload("test-model", "short", 8)
        payload["messages"][0]["content"] = prompt

        async def stream_chat(_payload, _request_id):
            message = f"Upstream rejected {json.dumps(prompt)}"
            if error_kind == "exception":
                raise WorkerUnavailableError(message)
            if error_kind == "internal_exception":
                raise RuntimeError(message)
            yield f"data: {json.dumps({'error': {'message': message, 'code': 500}})}\n\n"
            yield "data: [DONE]\n\n"

        with patch.object(controls.workers[0], "stream_chat", side_effect=stream_chat):
            response = await client.post(
                CHAT, headers={"X-Request-ID": "private-error"}, json=payload
            )
        assert response.status_code == 200
        assert '"error"' in response.text
        trace, _ = await assert_terminal(client, "private-error", "error")
        for text in (
            prompt,
            json.dumps(prompt)[1:-1],
            json.dumps(prompt, ensure_ascii=False)[1:-1],
        ):
            assert text not in trace["error"]
        assert "private data" not in json.dumps(trace)
        assert "private data" not in caplog.text
