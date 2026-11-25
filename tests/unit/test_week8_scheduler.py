"""Deterministic unit tests for the Week 8 open-loop scheduler."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

import pytest

from llm_serving_platform.benchmark.client import ChatCallResult, RunnerFatalError
from llm_serving_platform.benchmark.scheduler import OpenLoopScheduler
from llm_serving_platform.benchmark.workload import WorkloadEntry


@dataclass(frozen=True)
class FakeScheduledEntry:
    sequence: int
    scheduled_offset_ms: float
    entry: WorkloadEntry


class FakeClock:
    """Deterministic virtual clock and sleep_until driver (no real wall-clock sleep races)."""

    def __init__(self, start_s: float = 1000.0) -> None:
        self.now = start_s
        self.lag_schedule_ms: dict[int, float] = {}
        self._sleep_calls = 0

    def clock(self) -> float:
        return self.now

    async def sleep_until(self, deadline: float) -> None:
        # Flush ready tasks so previously spawned dispatches reach their await points.
        for _ in range(5):
            await asyncio.sleep(0)
        lag_s = self.lag_schedule_ms.get(self._sleep_calls, 0.0) / 1000.0
        self._sleep_calls += 1
        if deadline > self.now:
            self.now = deadline + lag_s
        elif lag_s > 0:
            self.now += lag_s
        for _ in range(3):
            await asyncio.sleep(0)


def _entry(idx: int) -> WorkloadEntry:
    return WorkloadEntry(
        index=idx,
        messages=({"role": "user", "content": f"Synthetic prompt {idx}"},),
        entry_id=f"q-{idx:03d}",
    )


def _ok_call(request_id: str, worker_id: str = "worker-mock-0") -> ChatCallResult:
    return ChatCallResult(
        status_code=200,
        request_id=request_id,
        response_request_id=request_id,
        response_worker_id=worker_id,
        content_chunks=2,
        saw_done=True,
        saw_sse_error=False,
        sse_error_code=None,
    )


def _trace(
    request_id: str,
    *,
    status: str = "completed",
    worker_id: str = "worker-mock-0",
    e2e_ms: float = 120.0,
) -> dict[str, Any]:
    return {
        "request_id": request_id,
        "model": "test-model",
        "worker_id": worker_id,
        "status": status,
        "routing": {"policy": "round_robin"},
        "metrics_ms": {"routing_ms": 0.1, "ttft_ms": 20.0, "e2e_ms": e2e_ms},
    }


@pytest.mark.asyncio
async def test_slow_response_does_not_delay_later_dispatches():
    """Requirement 2.1: Req 0 held active past 300ms must not delay dispatches at 100/200/300ms."""
    clock = FakeClock()
    schedule = [
        FakeScheduledEntry(sequence=0, scheduled_offset_ms=0.0, entry=_entry(0)),
        FakeScheduledEntry(sequence=1, scheduled_offset_ms=100.0, entry=_entry(1)),
        FakeScheduledEntry(sequence=2, scheduled_offset_ms=200.0, entry=_entry(2)),
        FakeScheduledEntry(sequence=3, scheduled_offset_ms=300.0, entry=_entry(3)),
    ]
    req0_gate = asyncio.Event()
    dispatched: list[tuple[str, float]] = []
    req0_still_active_when_req3_started = False

    async def send_chat(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        nonlocal req0_still_active_when_req3_started
        dispatched.append((request_id, (clock.now - 1000.0) * 1000.0))
        if request_id.endswith("-00000"):
            await req0_gate.wait()
            clock.now += 0.05  # finishes after 300ms
            return _ok_call(request_id)
        if request_id.endswith("-00003"):
            req0_still_active_when_req3_started = not req0_gate.is_set()
            req0_gate.set()
        return _ok_call(request_id)

    async def fetch_trace(request_id: str) -> dict[str, Any]:
        return _trace(request_id)

    scheduler = OpenLoopScheduler(
        send_chat=send_chat,
        fetch_trace=fetch_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
    )
    result = await scheduler.execute(
        run_id="run-2-1",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=16,
        max_dispatch_lag_ms=5.0,
    )

    assert result.run_status == "completed"
    assert result.failure_type is None
    assert req0_still_active_when_req3_started is True
    assert [off for _, off in dispatched] == pytest.approx([0.0, 100.0, 200.0, 300.0], abs=0.01)
    assert [row["dispatch_offset_ms"] for row in result.ledger] == pytest.approx(
        [0.0, 100.0, 200.0, 300.0], abs=0.01
    )
    assert all(row["dispatch_status"] == "sent" for row in result.ledger)
    assert all(row["trace_observed"] is True for row in result.ledger)
    assert result.ledger[0]["client_end_offset_ms"] >= 300.0


@pytest.mark.asyncio
async def test_simultaneous_burst_offsets_release_together():
    """Requirement 2.2: Entries sharing scheduled_offset_ms release as a concurrent burst."""
    clock = FakeClock()
    schedule = [
        FakeScheduledEntry(sequence=0, scheduled_offset_ms=50.0, entry=_entry(0)),
        FakeScheduledEntry(sequence=1, scheduled_offset_ms=50.0, entry=_entry(1)),
        FakeScheduledEntry(sequence=2, scheduled_offset_ms=50.0, entry=_entry(2)),
        FakeScheduledEntry(sequence=3, scheduled_offset_ms=250.0, entry=_entry(3)),
        FakeScheduledEntry(sequence=4, scheduled_offset_ms=250.0, entry=_entry(4)),
    ]
    burst1_entered = 0
    burst1_barrier = asyncio.Event()

    async def send_chat(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        nonlocal burst1_entered
        if entry.index in {0, 1, 2}:
            burst1_entered += 1
            if burst1_entered == 3:
                burst1_barrier.set()
            # All 3 must have entered concurrently before any of them completes!
            await asyncio.wait_for(burst1_barrier.wait(), timeout=1.0)
        return _ok_call(request_id)

    async def fetch_trace(request_id: str) -> dict[str, Any]:
        return _trace(request_id)

    scheduler = OpenLoopScheduler(
        send_chat=send_chat,
        fetch_trace=fetch_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
    )
    result = await scheduler.execute(
        run_id="run-2-2",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=8,
        max_dispatch_lag_ms=5.0,
    )

    assert result.run_status == "completed"
    assert result.max_observed_in_flight >= 3
    assert [row["dispatch_offset_ms"] for row in result.ledger[:3]] == pytest.approx(
        [50.0, 50.0, 50.0], abs=0.01
    )
    assert [row["dispatch_offset_ms"] for row in result.ledger[3:]] == pytest.approx(
        [250.0, 250.0], abs=0.01
    )


@pytest.mark.asyncio
async def test_max_in_flight_triggers_generator_limited_and_preserves_evidence():
    """Requirement 2.3: Hitting max_in_flight stops future sends without queueing."""
    clock = FakeClock()
    schedule = [
        FakeScheduledEntry(sequence=i, scheduled_offset_ms=float(i * 50), entry=_entry(i))
        for i in range(5)
    ]
    gates: dict[str, asyncio.Event] = {}
    flushed_requests: list[dict[str, Any]] = []
    flushed_traces: list[dict[str, Any]] = []

    async def send_chat(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        gate = asyncio.Event()
        gates[request_id] = gate
        await gate.wait()
        return _ok_call(request_id)

    async def fetch_trace(request_id: str) -> dict[str, Any]:
        return _trace(request_id, status="cancelled")

    scheduler = OpenLoopScheduler(
        send_chat=send_chat,
        fetch_trace=fetch_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
        on_request=flushed_requests.append,
        on_trace=flushed_traces.append,
        generator_limited_drain_timeout_s=0.02,
    )
    result = await scheduler.execute(
        run_id="run-2-3",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=2,
        max_dispatch_lag_ms=10.0,
    )

    assert result.run_status == "failed"
    assert result.failure_type == "generator_limited"
    assert result.sent_count == 2
    assert result.not_dispatched_count == 3
    assert len(flushed_requests) == 5
    for idx in (0, 1):
        assert result.ledger[idx]["dispatch_status"] == "sent"
        assert result.ledger[idx]["not_dispatched_reason"] is None
    for idx in (2, 3, 4):
        assert result.ledger[idx]["dispatch_status"] == "not_dispatched"
        assert result.ledger[idx]["not_dispatched_reason"] == "aborted"
        assert result.ledger[idx]["dispatch_offset_ms"] is None
        assert result.ledger[idx]["client_end_offset_ms"] is None
        assert result.ledger[idx]["http_status"] is None
        assert result.ledger[idx]["client_status"] is None
        assert result.ledger[idx]["trace_observed"] is False
    assert len(flushed_traces) == 2


@pytest.mark.asyncio
async def test_slow_trace_poll_does_not_hold_max_in_flight_slot():
    """Contract 5: Trace polling must not hold an inference max_in_flight slot."""
    clock = FakeClock()
    schedule = [
        FakeScheduledEntry(sequence=i, scheduled_offset_ms=float(i * 50), entry=_entry(i))
        for i in range(4)
    ]
    trace_gate = asyncio.Event()

    async def send_chat(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        return _ok_call(request_id)

    async def fetch_trace(request_id: str) -> dict[str, Any]:
        if request_id.endswith("-00000") or request_id.endswith("-00001"):
            await trace_gate.wait()
        if request_id.endswith("-00003"):
            trace_gate.set()
        return _trace(request_id)

    scheduler = OpenLoopScheduler(
        send_chat=send_chat,
        fetch_trace=fetch_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
    )
    result = await scheduler.execute(
        run_id="run-slot-free",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=1,
        max_dispatch_lag_ms=10.0,
    )

    assert result.run_status == "completed"
    assert result.sent_count == 4


@pytest.mark.asyncio
async def test_dispatch_lag_guard_triggers_generator_limited():
    """Requirement 2.4: Dispatch lag exceeding max_dispatch_lag_ms fails as generator_limited."""
    clock = FakeClock()
    # Sleep index 0 is deadline 0ms; sleep index 1 is deadline 100ms -> inject 45ms lag.
    clock.lag_schedule_ms[1] = 45.0
    schedule = [
        FakeScheduledEntry(sequence=0, scheduled_offset_ms=0.0, entry=_entry(0)),
        FakeScheduledEntry(sequence=1, scheduled_offset_ms=100.0, entry=_entry(1)),
        FakeScheduledEntry(sequence=2, scheduled_offset_ms=200.0, entry=_entry(2)),
    ]

    async def send_chat(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        return _ok_call(request_id)

    async def fetch_trace(request_id: str) -> dict[str, Any]:
        return _trace(request_id)

    scheduler = OpenLoopScheduler(
        send_chat=send_chat,
        fetch_trace=fetch_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
    )
    result = await scheduler.execute(
        run_id="run-2-4",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=16,
        max_dispatch_lag_ms=25.0,
    )

    assert result.run_status == "failed"
    assert result.failure_type == "generator_limited"
    assert result.sent_count == 1
    assert result.not_dispatched_count == 2
    assert result.ledger[0]["dispatch_status"] == "sent"
    assert result.ledger[0]["trace_observed"] is True
    assert result.ledger[1]["dispatch_status"] == "not_dispatched"
    assert result.ledger[1]["not_dispatched_reason"] == "aborted"
    assert result.ledger[2]["dispatch_status"] == "not_dispatched"


@pytest.mark.asyncio
async def test_request_level_inference_errors_do_not_stop_schedule():
    """Requirement 2.5: HTTP 503 and SSE error with terminal trace do not abort remaining sends."""
    clock = FakeClock()
    schedule = [
        FakeScheduledEntry(sequence=0, scheduled_offset_ms=0.0, entry=_entry(0)),
        FakeScheduledEntry(sequence=1, scheduled_offset_ms=50.0, entry=_entry(1)),
        FakeScheduledEntry(sequence=2, scheduled_offset_ms=100.0, entry=_entry(2)),
        FakeScheduledEntry(sequence=3, scheduled_offset_ms=150.0, entry=_entry(3)),
    ]

    async def send_chat(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        if entry.index == 1:
            return ChatCallResult(
                status_code=503,
                request_id=request_id,
                response_request_id=request_id,
                response_worker_id=None,
                content_chunks=0,
                saw_done=False,
                saw_sse_error=False,
                sse_error_code=503,
            )
        if entry.index == 2:
            return ChatCallResult(
                status_code=200,
                request_id=request_id,
                response_request_id=request_id,
                response_worker_id="worker-mock-1",
                content_chunks=0,
                saw_done=False,
                saw_sse_error=True,
                sse_error_code=500,
            )
        return _ok_call(request_id)

    async def fetch_trace(request_id: str) -> dict[str, Any]:
        if request_id.endswith("-00001") or request_id.endswith("-00002"):
            return _trace(request_id, status="error", worker_id="none")
        return _trace(request_id, status="completed")

    scheduler = OpenLoopScheduler(
        send_chat=send_chat,
        fetch_trace=fetch_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
    )
    result = await scheduler.execute(
        run_id="run-2-5",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=8,
        max_dispatch_lag_ms=10.0,
    )

    assert result.run_status == "completed"
    assert result.sent_count == 4
    assert result.ledger[1]["http_status"] == 503
    assert result.ledger[1]["client_status"] == "error"
    assert result.ledger[1]["client_error_type"] == "http_error"
    assert result.ledger[1]["trace_observed"] is True
    assert result.ledger[2]["http_status"] == 200
    assert result.ledger[2]["client_status"] == "error"
    assert result.ledger[2]["client_error_type"] == "sse_error"
    assert result.ledger[2]["trace_observed"] is True
    assert result.ledger[3]["client_status"] == "completed"


@pytest.mark.asyncio
async def test_client_timeout_with_and_without_terminal_trace():
    """Requirement 2.7: Timeout with real trace continues; timeout missing trace fails."""
    clock = FakeClock()
    schedule = [
        FakeScheduledEntry(sequence=0, scheduled_offset_ms=0.0, entry=_entry(0)),
        FakeScheduledEntry(sequence=1, scheduled_offset_ms=50.0, entry=_entry(1)),
    ]

    async def send_chat_timeout(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        if entry.index == 0:
            raise RunnerFatalError("gateway_timeout", "Timed out waiting for stream")
        return _ok_call(request_id)

    async def fetch_trace_found(request_id: str) -> dict[str, Any]:
        status = "cancelled" if request_id.endswith("-00000") else "completed"
        return _trace(request_id, status=status)

    scheduler_ok = OpenLoopScheduler(
        send_chat=send_chat_timeout,
        fetch_trace=fetch_trace_found,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
    )
    res_ok = await scheduler_ok.execute(
        run_id="run-timeout-found",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=8,
        max_dispatch_lag_ms=10.0,
    )
    assert res_ok.run_status == "completed"
    assert res_ok.ledger[0]["client_status"] == "error"
    assert res_ok.ledger[0]["client_error_type"] == "client_timeout"
    assert res_ok.ledger[0]["http_status"] is None
    assert res_ok.ledger[0]["trace_observed"] is True
    assert res_ok.ledger[1]["client_status"] == "completed"

    # Now missing trace on timeout -> run fails as missing_trace, never creates fake trace.
    clock2 = FakeClock()
    traces_written: list[dict[str, Any]] = []

    async def fetch_trace_missing(request_id: str) -> dict[str, Any]:
        if request_id.endswith("-00000"):
            raise RunnerFatalError("trace_timeout", "Missing trace")
        return _trace(request_id)

    scheduler_missing = OpenLoopScheduler(
        send_chat=send_chat_timeout,
        fetch_trace=fetch_trace_missing,
        clock=clock2.clock,
        sleep_until=clock2.sleep_until,
        on_trace=traces_written.append,
    )
    res_fail = await scheduler_missing.execute(
        run_id="run-timeout-missing",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=8,
        max_dispatch_lag_ms=10.0,
    )
    assert res_fail.run_status == "failed"
    assert res_fail.failure_type == "missing_trace"
    assert res_fail.ledger[0]["trace_observed"] is False
    assert all(not t["request_id"].endswith("-00000") for t in traces_written)


@pytest.mark.asyncio
async def test_controlled_cancellation_stops_future_sends_and_cleans_up():
    """Requirement 2.7: Controlled cancel halts sends, closes HTTP, fetches real trace if any."""
    clock = FakeClock()
    schedule = [
        FakeScheduledEntry(sequence=i, scheduled_offset_ms=float(i * 100), entry=_entry(i))
        for i in range(4)
    ]
    cancel_event = asyncio.Event()
    req0_started = asyncio.Event()
    closed_requests: list[str] = []

    async def send_chat(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        if entry.index == 0:
            req0_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                closed_requests.append(request_id)
                raise
        return _ok_call(request_id)

    async def fetch_trace(request_id: str) -> dict[str, Any]:
        return _trace(request_id, status="cancelled")

    async def trigger_cancel() -> None:
        await req0_started.wait()
        cancel_event.set()

    canceller = asyncio.create_task(trigger_cancel())
    scheduler = OpenLoopScheduler(
        send_chat=send_chat,
        fetch_trace=fetch_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
        cancel_cleanup_timeout_s=1.0,
    )
    result = await scheduler.execute(
        run_id="run-cancel",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=8,
        max_dispatch_lag_ms=10.0,
        cancel_event=cancel_event,
    )
    await canceller

    assert result.run_status == "failed"
    assert result.failure_type == "runner_cancelled"
    assert closed_requests == ["run-cancel-req-00000"]
    assert result.ledger[0]["dispatch_status"] == "sent"
    assert result.ledger[0]["client_status"] == "cancelled"
    assert result.ledger[0]["client_error_type"] == "client_cancelled"
    assert result.ledger[0]["trace_observed"] is True
    for idx in (1, 2, 3):
        assert result.ledger[idx]["dispatch_status"] == "not_dispatched"
        assert result.ledger[idx]["not_dispatched_reason"] == "aborted"


@pytest.mark.asyncio
async def test_external_task_cancellation_flushes_evidence_and_closes_http():
    """Cancelling execute() task runs shielded cleanup and flushes ledger without fake traces."""
    clock = FakeClock()
    schedule = [
        FakeScheduledEntry(sequence=i, scheduled_offset_ms=float(i * 100), entry=_entry(i))
        for i in range(3)
    ]
    req0_started = asyncio.Event()
    closed_requests: list[str] = []
    flushed_requests: list[dict[str, Any]] = []
    flushed_traces: list[dict[str, Any]] = []

    async def send_chat(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        if entry.index == 0:
            req0_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                closed_requests.append(request_id)
                raise
        return _ok_call(request_id)

    async def fetch_trace(request_id: str) -> dict[str, Any]:
        raise RunnerFatalError("trace_timeout", "No trace")

    scheduler = OpenLoopScheduler(
        send_chat=send_chat,
        fetch_trace=fetch_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
        on_request=flushed_requests.append,
        on_trace=flushed_traces.append,
        cancel_cleanup_timeout_s=1.0,
    )
    task = asyncio.create_task(
        scheduler.execute(
            run_id="run-ext-cancel",
            scheduled_entries=schedule,
            measurement_window_ms=1000.0,
            max_in_flight=8,
            max_dispatch_lag_ms=10.0,
        )
    )
    await req0_started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert closed_requests == ["run-ext-cancel-req-00000"]
    assert flushed_traces == []
    by_seq = {row["sequence"]: row for row in flushed_requests}
    assert set(by_seq.keys()) == {0, 1, 2}
    assert by_seq[0]["client_status"] == "cancelled"
    assert by_seq[0]["trace_observed"] is False
    assert by_seq[1]["dispatch_status"] == "not_dispatched"
    assert by_seq[2]["dispatch_status"] == "not_dispatched"


@pytest.mark.asyncio
async def test_out_of_order_completion_correlates_by_request_id():
    """Requirement 2.8: Req 3 finishing before Req 1 must still match exact request_id."""
    clock = FakeClock()
    schedule = [
        FakeScheduledEntry(sequence=i, scheduled_offset_ms=float(i * 20), entry=_entry(i))
        for i in range(4)
    ]
    req1_gate = asyncio.Event()

    async def send_chat(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        if entry.index == 1:
            await req1_gate.wait()
            return _ok_call(request_id, worker_id="worker-mock-1")
        if entry.index == 3:
            req1_gate.set()
            return _ok_call(request_id, worker_id="worker-mock-0")
        return _ok_call(request_id, worker_id="worker-mock-0")

    async def fetch_trace(request_id: str) -> dict[str, Any]:
        wid = "worker-mock-1" if request_id.endswith("-00001") else "worker-mock-0"
        return _trace(request_id, worker_id=wid)

    scheduler = OpenLoopScheduler(
        send_chat=send_chat,
        fetch_trace=fetch_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
    )
    result = await scheduler.execute(
        run_id="run-ooo",
        scheduled_entries=schedule,
        measurement_window_ms=1000.0,
        max_in_flight=8,
        max_dispatch_lag_ms=10.0,
    )

    assert result.run_status == "completed"
    by_id = {t["request_id"]: t for t in result.traces}
    assert by_id["run-ooo-req-00001"]["worker_id"] == "worker-mock-1"
    assert by_id["run-ooo-req-00003"]["worker_id"] == "worker-mock-0"
    assert [row["sequence"] for row in result.ledger] == [0, 1, 2, 3]
    assert result.ledger[1]["request_id"] == "run-ooo-req-00001"
    assert result.ledger[3]["request_id"] == "run-ooo-req-00003"


@pytest.mark.asyncio
async def test_rejects_invalid_schedule_order_or_window_overflow():
    """Contract 4: Invalid sequence/offset schedule must fail fast before dispatch."""
    clock = FakeClock()

    async def noop_send(entry: WorkloadEntry, request_id: str) -> ChatCallResult:
        raise AssertionError("Must not dispatch invalid schedule")

    async def noop_trace(request_id: str) -> dict[str, Any]:
        raise AssertionError("Must not fetch trace for invalid schedule")

    scheduler = OpenLoopScheduler(
        send_chat=noop_send,
        fetch_trace=noop_trace,
        clock=clock.clock,
        sleep_until=clock.sleep_until,
    )

    with pytest.raises(ValueError):
        await scheduler.execute(
            run_id="bad-seq",
            scheduled_entries=[
                FakeScheduledEntry(sequence=1, scheduled_offset_ms=0.0, entry=_entry(0))
            ],
            measurement_window_ms=1000.0,
            max_in_flight=4,
            max_dispatch_lag_ms=10.0,
        )

    with pytest.raises(ValueError):
        await scheduler.execute(
            run_id="bad-window",
            scheduled_entries=[
                FakeScheduledEntry(sequence=0, scheduled_offset_ms=1000.0, entry=_entry(0))
            ],
            measurement_window_ms=1000.0,
            max_in_flight=4,
            max_dispatch_lag_ms=10.0,
        )
