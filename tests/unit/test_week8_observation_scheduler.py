"""Observation-mode replay must finish late schedules without weakening resource limits."""

import asyncio

import pytest

from llm_serving_platform.benchmark.client import RunnerFatalError
from llm_serving_platform.benchmark.scheduler import OpenLoopScheduler
from tests.unit.test_week8_scheduler import FakeClock, FakeScheduledEntry, _entry, _ok_call, _trace


@pytest.mark.asyncio
@pytest.mark.parametrize("lag,warning", [(249.999, False), (250.0, False), (250.001, True)])
async def test_lag_boundary_is_warning_not_stop(lag, warning):
    clock = FakeClock()
    clock.lag_schedule_ms[0] = lag
    sent = []

    async def send(entry, rid):
        sent.append((entry.index, clock.clock()))
        return _ok_call(rid)

    async def trace(rid):
        return _trace(rid)

    schedule = [FakeScheduledEntry(i, float(i * 100), _entry(i)) for i in range(3)]
    result = await OpenLoopScheduler(
        send_chat=send, fetch_trace=trace, clock=clock.clock, sleep_until=clock.sleep_until
    ).execute(
        run_id="boundary",
        scheduled_entries=schedule,
        measurement_window_ms=1000,
        max_in_flight=16,
        max_dispatch_lag_ms=250,
        observe_limits=True,
    )
    assert result.run_status == "completed"
    assert [i for i, _ in sent] == [0, 1, 2]
    assert any(w["type"] == "dispatch_late" for w in result.warnings) is warning
    first = min(result.ledger, key=lambda row: row["sequence"])
    assert first["dispatch_offset_ms"] == pytest.approx((sent[0][1] - result.t0_mono) * 1000)


@pytest.mark.asyncio
async def test_late_dispatch_beyond_window_still_sends_every_entry():
    clock = FakeClock()
    clock.lag_schedule_ms[0] = 1200

    async def send(_entry, rid):
        return _ok_call(rid)

    async def trace(rid):
        return _trace(rid)

    result = await OpenLoopScheduler(
        send_chat=send, fetch_trace=trace, clock=clock.clock, sleep_until=clock.sleep_until
    ).execute(
        run_id="late-window",
        scheduled_entries=[FakeScheduledEntry(i, i * 10, _entry(i)) for i in range(4)],
        measurement_window_ms=100,
        max_in_flight=16,
        max_dispatch_lag_ms=250,
        observe_limits=True,
    )
    assert result.run_status == "completed"
    assert result.sent_count == 4
    assert result.not_dispatched_count == 0
    assert sum(w["type"] == "dispatch_after_window" for w in result.warnings) == 4


@pytest.mark.asyncio
async def test_capacity_skip_does_not_abort_future_scheduled_entries():
    clock = FakeClock()
    release = asyncio.Event()
    sent = []

    async def send(entry, rid):
        sent.append(entry.index)
        if entry.index == 0:
            await release.wait()
        return _ok_call(rid)

    async def trace(rid):
        return _trace(rid)

    async def advance(deadline):
        if deadline >= 1000.1:
            release.set()
        await clock.sleep_until(deadline)

    result = await OpenLoopScheduler(
        send_chat=send, fetch_trace=trace, clock=clock.clock, sleep_until=advance
    ).execute(
        run_id="capacity",
        scheduled_entries=[
            FakeScheduledEntry(i, offset, _entry(i)) for i, offset in enumerate((0, 0, 100))
        ],
        measurement_window_ms=500,
        max_in_flight=1,
        max_dispatch_lag_ms=250,
        observe_limits=True,
    )
    assert result.run_status == "completed"
    assert sent == [0, 2]
    assert result.max_observed_in_flight == 1
    assert result.ledger[1]["not_dispatched_reason"] == "in_flight_limit"
    assert result.ledger[1]["dispatch_offset_ms"] is None
    assert any(w["type"] == "in_flight_limit" and w["sequence"] == 1 for w in result.warnings)


@pytest.mark.asyncio
async def test_missing_trace_warns_without_stopping_later_dispatches():
    clock = FakeClock()

    async def send(_entry, rid):
        return _ok_call(rid)

    async def trace(_rid):
        raise RunnerFatalError("trace_timeout", "synthetic")

    result = await OpenLoopScheduler(
        send_chat=send, fetch_trace=trace, clock=clock.clock, sleep_until=clock.sleep_until
    ).execute(
        run_id="trace-gap",
        scheduled_entries=[FakeScheduledEntry(i, i * 100, _entry(i)) for i in range(3)],
        measurement_window_ms=500,
        max_in_flight=16,
        max_dispatch_lag_ms=250,
        observe_limits=True,
    )
    assert result.run_status == "completed"
    assert result.sent_count == 3
    assert result.traces == []
    assert all(not row["trace_observed"] for row in result.ledger)
    assert sum(w["type"] == "missing_trace" for w in result.warnings) == 3
