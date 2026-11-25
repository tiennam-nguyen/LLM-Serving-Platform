"""Tests verifying isolated inference and control/trace HTTP connection pools."""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from llm_serving_platform.benchmark.client import GatewayHTTPClient


def test_gateway_http_client_creates_distinct_explicit_pools():
    client_wrapper = GatewayHTTPClient(
        "http://127.0.0.1:8080",
        timeout_s=60.0,
        inference_max_connections=96,
        control_max_connections=16,
        control_timeout_s=8.0,
    )

    async def _check() -> None:
        async with client_wrapper as gw:
            assert gw._inference_client is not gw._control_client
            meta = gw.pool_limits_metadata()
            assert meta["isolated_pools"] is True
            assert meta["inference_max_connections"] == 96
            assert meta["control_max_connections"] == 16
            assert meta["inference_timeout_s"] == 60.0
            assert meta["control_timeout_s"] == 8.0

            inf_pool = gw._inference_client._transport._pool  # type: ignore[attr-defined]
            ctrl_pool = gw._control_client._transport._pool  # type: ignore[attr-defined]
            assert inf_pool._max_connections == 96
            assert ctrl_pool._max_connections == 16
            assert inf_pool is not ctrl_pool

    asyncio.run(_check())


def test_gateway_http_client_preserves_single_injected_client_compat():
    injected = httpx.AsyncClient(base_url="http://mock-gateway")

    async def _check() -> None:
        async with GatewayHTTPClient("http://mock-gateway", client=injected) as gw:
            assert gw._inference_client is injected
            assert gw._control_client is injected
            assert gw.pool_limits_metadata()["isolated_pools"] is False
        await injected.aclose()

    asyncio.run(_check())


@pytest.mark.asyncio
async def test_blocked_trace_polling_cannot_block_scheduled_inference_dispatch():
    """Saturating the trace/control pool must never head-of-line block inference dispatch."""
    trace_entered = 0
    all_control_slots_blocked = asyncio.Event()
    release_traces = asyncio.Event()

    async def handle_conn(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        nonlocal trace_entered
        try:
            raw = await reader.readuntil(b"\r\n\r\n")
            first_line = raw.decode("ascii", errors="replace").splitlines()[0]
            method, path, _ = first_line.split(" ", 2)
            # Drain Content-Length if present.
            content_len = 0
            for line in raw.decode("ascii", errors="replace").splitlines()[1:]:
                if line.lower().startswith("content-length:"):
                    content_len = int(line.split(":", 1)[1].strip())
            if content_len:
                await reader.readexactly(content_len)

            if method == "GET" and path.startswith("/v1/traces/"):
                trace_entered += 1
                if trace_entered >= 2:
                    all_control_slots_blocked.set()
                await release_traces.wait()
                req_id = path.rsplit("/", 1)[-1]
                body = json.dumps(
                    {
                        "request_id": req_id,
                        "status": "completed",
                        "worker_id": "worker-mock-0",
                        "routing": {"policy": "round_robin"},
                        "metrics_ms": {"routing_ms": 0.1, "ttft_ms": 1.0, "e2e_ms": 2.0},
                    }
                ).encode("utf-8")
                resp = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: application/json\r\n"
                    + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
                    + body
                )
                writer.write(resp)
                await writer.drain()
            elif method == "POST" and path == "/v1/chat/completions":
                sse = (
                    b'data: {"choices":[{"delta":{"content":"OK"}}]}\n\n'
                    b"data: [DONE]\n\n"
                )
                resp = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/event-stream\r\n"
                    b"X-Worker-ID: worker-mock-0\r\n"
                    + f"Content-Length: {len(sse)}\r\n\r\n".encode("ascii")
                    + sse
                )
                writer.write(resp)
                await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except OSError:
                pass

    server = await asyncio.start_server(handle_conn, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]

    try:
        async with GatewayHTTPClient(
            f"http://127.0.0.1:{port}",
            timeout_s=5.0,
            inference_max_connections=4,
            control_max_connections=2,
            control_timeout_s=5.0,
        ) as gw:
            # Saturate ALL 2 control connections + 1 queued waiter in the control pool.
            blocked_trace_tasks = [
                asyncio.create_task(gw.fetch_terminal_trace(f"trace-hold-{i}", timeout_s=5.0))
                for i in range(3)
            ]
            await asyncio.wait_for(all_control_slots_blocked.wait(), timeout=2.0)

            # While control pool is saturated, 4 inference calls must finish immediately!
            payload = {
                "model": "test",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            }
            chat_results = await asyncio.wait_for(
                asyncio.gather(
                    *(gw.stream_chat_completion(payload, f"inf-req-{i}") for i in range(4))
                ),
                timeout=1.5,
            )
            assert all(res.status_code == 200 and res.saw_done for res in chat_results)

            # Now unblock trace polling and verify clean completion.
            release_traces.set()
            traces = await asyncio.wait_for(asyncio.gather(*blocked_trace_tasks), timeout=2.0)
            assert len(traces) == 3
            assert all(t["status"] == "completed" for t in traces)
    finally:
        release_traces.set()
        server.close()
        await server.wait_closed()
