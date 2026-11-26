"""Deterministic open-loop benchmark scheduler for Week 8."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

from llm_serving_platform.benchmark.client import ChatCallResult, RunnerFatalError
from llm_serving_platform.benchmark.workload import WorkloadEntry

VALID_CLIENT_STATUSES = frozenset({"completed", "error", "cancelled"})
VALID_CLIENT_ERROR_TYPES = frozenset(
    {
        "client_timeout",
        "client_cancelled",
        "transport_error",
        "sse_error",
        "http_error",
        "incomplete_sse",
    }
)


@runtime_checkable
class ScheduledEntryProtocol(Protocol):
    """Structural interface for one pre-ordered open-loop scheduled workload entry."""

    @property
    def sequence(self) -> int: ...

    @property
    def scheduled_offset_ms(self) -> float: ...

    @property
    def entry(self) -> WorkloadEntry: ...


@dataclass(frozen=True)
class SchedulerResult:
    """Outcome of executing one open-loop measured schedule."""

    run_status: str
    failure_type: str | None
    t0_mono: float
    measured_started_at: str
    drain_end_offset_ms: float
    planned_count: int
    sent_count: int
    not_dispatched_count: int
    ledger: list[dict[str, Any]]
    traces: list[dict[str, Any]]
    max_observed_in_flight: int
    warnings: list[dict[str, Any]] = field(default_factory=list)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _is_non_bool_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def validate_open_loop_schedule(
    scheduled_entries: Sequence[ScheduledEntryProtocol],
    *,
    measurement_window_ms: float,
    max_in_flight: int,
    max_dispatch_lag_ms: float,
) -> None:
    """Validate open-loop schedule invariants before starting measured traffic."""
    if not scheduled_entries:
        raise ValueError("scheduled_entries must be non-empty")
    if not _is_finite_number(measurement_window_ms) or float(measurement_window_ms) <= 0.0:
        raise ValueError(f"measurement_window_ms must be finite > 0, got {measurement_window_ms!r}")
    if not _is_non_bool_int(max_in_flight) or max_in_flight < 1:
        raise ValueError(f"max_in_flight must be an integer >= 1, got {max_in_flight!r}")
    if not _is_finite_number(max_dispatch_lag_ms) or float(max_dispatch_lag_ms) <= 0.0:
        raise ValueError(f"max_dispatch_lag_ms must be finite > 0, got {max_dispatch_lag_ms!r}")

    window_ms = float(measurement_window_ms)
    prev_offset = -1.0
    for expected_seq, item in enumerate(scheduled_entries):
        if not _is_non_bool_int(item.sequence) or item.sequence != expected_seq:
            raise ValueError(
                f"Schedule sequence must be contiguous from 0: expected {expected_seq}, "
                f"got {item.sequence!r}"
            )
        if not _is_finite_number(item.scheduled_offset_ms):
            raise ValueError(
                f"scheduled_offset_ms at sequence {expected_seq} must be finite, "
                f"got {item.scheduled_offset_ms!r}"
            )
        offset = float(item.scheduled_offset_ms)
        if offset < 0.0:
            raise ValueError(f"scheduled_offset_ms at sequence {expected_seq} must be >= 0")
        if offset < prev_offset:
            raise ValueError(
                f"scheduled_offset_ms must be non-decreasing: {offset} < {prev_offset} "
                f"at sequence {expected_seq}"
            )
        if offset >= window_ms:
            raise ValueError(
                f"scheduled_offset_ms ({offset}) at sequence {expected_seq} must be < "
                f"measurement_window_ms ({window_ms})"
            )
        prev_offset = offset


def classify_chat_outcome(call: ChatCallResult) -> tuple[str, str | None]:
    """Classify an HTTP/SSE response strictly per the Week 8 contract."""
    if call.status_code != 200:
        return "error", "http_error"
    if call.saw_sse_error:
        return "error", "sse_error"
    if call.content_chunks <= 0 or not call.saw_done:
        return "error", "incomplete_sse"
    return "completed", None


def make_not_dispatched_record(
    sequence: int, request_id: str, *, reason: str = "aborted"
) -> dict[str, Any]:
    """Build a canonical requests.jsonl ledger entry for an aborted/unscheduled turn."""
    return {
        "sequence": sequence,
        "request_id": request_id,
        "dispatch_status": "not_dispatched",
        "not_dispatched_reason": reason,
        "dispatch_offset_ms": None,
        "dispatch_lag_ms": None,
        "client_end_offset_ms": None,
        "http_status": None,
        "client_status": None,
        "client_error_type": None,
        "trace_observed": False,
    }


def make_sent_record(
    *,
    sequence: int,
    request_id: str,
    dispatch_offset_ms: float,
    dispatch_lag_ms: float,
    client_end_offset_ms: float,
    http_status: int | None,
    client_status: str,
    client_error_type: str | None,
    trace_observed: bool,
) -> dict[str, Any]:
    """Build a canonical requests.jsonl ledger entry for a dispatched request."""
    if client_status not in VALID_CLIENT_STATUSES:
        raise ValueError(f"Unsupported client_status {client_status!r}")
    if client_error_type is not None and client_error_type not in VALID_CLIENT_ERROR_TYPES:
        raise ValueError(f"Unsupported client_error_type {client_error_type!r}")
    return {
        "sequence": sequence,
        "request_id": request_id,
        "dispatch_status": "sent",
        "not_dispatched_reason": None,
        "dispatch_offset_ms": round(dispatch_offset_ms, 3),
        "dispatch_lag_ms": round(dispatch_lag_ms, 3),
        "client_end_offset_ms": round(client_end_offset_ms, 3),
        "http_status": http_status,
        "client_status": client_status,
        "client_error_type": client_error_type,
        "trace_observed": trace_observed,
    }


async def _maybe_await(result: Awaitable[None] | None) -> None:
    if result is not None:
        await result


class OpenLoopScheduler:
    """Dispatches requests on an absolute monotonic schedule independent of worker load."""

    def __init__(
        self,
        *,
        send_chat: Callable[[WorkloadEntry, str], Awaitable[ChatCallResult]],
        fetch_trace: Callable[[str], Awaitable[dict[str, Any]]],
        validate_trace: Callable[[ChatCallResult, dict[str, Any]], None] | None = None,
        on_trace: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        on_request: Callable[[dict[str, Any]], Awaitable[None] | None] | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep_until: Callable[[float], Awaitable[None]] | None = None,
        request_timeout_s: float | None = None,
        generator_limited_drain_timeout_s: float = 30.0,
        cancel_cleanup_timeout_s: float = 10.0,
    ) -> None:
        self._send_chat = send_chat
        self._fetch_trace = fetch_trace
        self._validate_trace = validate_trace
        self._on_trace = on_trace
        self._on_request = on_request
        self._clock = clock
        self._custom_sleep_until = sleep_until
        self._request_timeout_s = request_timeout_s
        self._generator_limited_drain_timeout_s = generator_limited_drain_timeout_s
        self._cancel_cleanup_timeout_s = cancel_cleanup_timeout_s

    async def _default_sleep_until(self, deadline: float) -> None:
        # Event-loop timers may wake within their clock resolution before a deadline
        # (notably on Windows). Recheck the same clock instead of dispatching early.
        while (remaining := deadline - self._clock()) > 0.0:
            await asyncio.sleep(remaining)

    async def _wait_until_deadline_or_stop(
        self,
        deadline: float,
        stop_event: asyncio.Event,
    ) -> bool:
        """Wait for deadline unless stop_event fires first. Returns True if stopped."""
        if stop_event.is_set():
            return True

        # An already-due replay entry needs no timer tasks. The injected clock driver
        # retains its explicit yield points for deterministic scheduler tests.
        if self._custom_sleep_until is None and deadline <= self._clock():
            return False

        sleep_fn = self._custom_sleep_until or self._default_sleep_until
        sleep_task = asyncio.create_task(sleep_fn(deadline))
        stop_task = asyncio.create_task(stop_event.wait())
        try:
            done, _ = await asyncio.wait(
                (sleep_task, stop_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if stop_task in done or stop_event.is_set():
                if not sleep_task.done():
                    sleep_task.cancel()
                    await asyncio.gather(sleep_task, return_exceptions=True)
                return True
            # Propagate unexpected sleep errors if any; cancel stop_task cleanly.
            await sleep_task
            return False
        finally:
            for task in (sleep_task, stop_task):
                if not task.done():
                    task.cancel()
            await asyncio.gather(sleep_task, stop_task, return_exceptions=True)

    async def execute(
        self,
        *,
        run_id: str,
        scheduled_entries: Sequence[ScheduledEntryProtocol],
        measurement_window_ms: float,
        max_in_flight: int,
        max_dispatch_lag_ms: float,
        cancel_event: asyncio.Event | None = None,
        observe_limits: bool = False,
    ) -> SchedulerResult:
        """Execute an open-loop schedule and return full request/trace evidence."""
        validate_open_loop_schedule(
            scheduled_entries,
            measurement_window_ms=measurement_window_ms,
            max_in_flight=max_in_flight,
            max_dispatch_lag_ms=max_dispatch_lag_ms,
        )

        effective_cancel = cancel_event or asyncio.Event()
        stop_dispatch = asyncio.Event()
        write_lock = asyncio.Lock()

        t0 = self._clock()
        measured_started_at = _utc_now_iso()
        window_end_mono = t0 + float(measurement_window_ms) / 1000.0

        active_http_in_flight = 0
        max_observed_in_flight = 0
        first_fatal_type: str | None = None

        http_tasks: dict[int, asyncio.Task[ChatCallResult]] = {}
        request_tasks: dict[int, asyncio.Task[None]] = {}
        ledger_by_seq: dict[int, dict[str, Any]] = {}
        traces_by_seq: dict[int, dict[str, Any]] = {}
        sent_sequences: set[int] = set()
        warnings: list[dict[str, Any]] = []
        actual_dispatch: dict[int, tuple[float, float]] = {}

        def _warn(kind: str, **details: Any) -> None:
            warnings.append({"type": kind, **details})

        def _set_fatal(failure_type: str) -> None:
            nonlocal first_fatal_type
            if first_fatal_type is None:
                first_fatal_type = failure_type
            stop_dispatch.set()

        async def _emit_trace(sequence: int, trace: dict[str, Any]) -> None:
            async with write_lock:
                traces_by_seq[sequence] = trace
                if self._on_trace is not None:
                    await _maybe_await(self._on_trace(trace))

        async def _emit_request(record: dict[str, Any]) -> None:
            seq = int(record["sequence"])
            async with write_lock:
                if seq in ledger_by_seq:
                    return
                ledger_by_seq[seq] = record
                if self._on_request is not None:
                    await _maybe_await(self._on_request(record))

        async def _run_http_call(item: ScheduledEntryProtocol, request_id: str) -> ChatCallResult:
            if observe_limits:
                offset = (self._clock() - t0) * 1000.0
                lag = offset - float(item.scheduled_offset_ms)
                actual_dispatch[item.sequence] = (offset, lag)
                sent_sequences.add(item.sequence)
                if round(lag, 3) > max_dispatch_lag_ms:
                    _warn(
                        "dispatch_late",
                        sequence=item.sequence,
                        scheduled_offset_ms=float(item.scheduled_offset_ms),
                        actual_dispatch_offset_ms=offset,
                        dispatch_lag_ms=lag,
                        limit_ms=max_dispatch_lag_ms,
                    )
                if round(offset, 3) >= measurement_window_ms:
                    _warn(
                        "dispatch_after_window",
                        sequence=item.sequence,
                        actual_dispatch_offset_ms=offset,
                    )
            coro = self._send_chat(item.entry, request_id)
            if self._request_timeout_s is not None:
                return await asyncio.wait_for(coro, timeout=self._request_timeout_s)
            return await coro

        async def _handle_dispatched(
            item: ScheduledEntryProtocol,
            request_id: str,
            dispatch_offset_ms: float,
            dispatch_lag_ms: float,
            http_task: asyncio.Task[ChatCallResult],
        ) -> None:
            nonlocal active_http_in_flight
            http_status: int | None = None
            client_status = "error"
            client_error_type: str | None = None
            call: ChatCallResult | None = None
            skip_trace_poll = False

            try:
                call = await http_task
                client_end_mono = self._clock()
                http_status = call.status_code
                client_status, client_error_type = classify_chat_outcome(call)
            except asyncio.CancelledError:
                client_end_mono = self._clock()
                client_status = "cancelled"
                client_error_type = "client_cancelled"
            except TimeoutError:
                client_end_mono = self._clock()
                client_status = "error"
                client_error_type = "client_timeout"
            except RunnerFatalError as exc:
                client_end_mono = self._clock()
                if exc.failure_type == "gateway_timeout":
                    client_status = "error"
                    client_error_type = "client_timeout"
                elif exc.failure_type == "gateway_transport_error":
                    client_status = "error"
                    client_error_type = "transport_error"
                elif exc.failure_type in {"gateway_unreachable", "contract_error"}:
                    client_status = "error"
                    client_error_type = "transport_error"
                    skip_trace_poll = True
                    _set_fatal(exc.failure_type)
                else:
                    client_status = "error"
                    client_error_type = "transport_error"
                    skip_trace_poll = True
                    _set_fatal(exc.failure_type)
            except Exception:
                client_end_mono = self._clock()
                client_status = "error"
                client_error_type = "transport_error"
                skip_trace_poll = True
                _set_fatal("runner_internal_error")
            finally:
                active_http_in_flight -= 1
                http_tasks.pop(item.sequence, None)

            client_end_offset_ms = (client_end_mono - t0) * 1000.0
            trace_observed = False

            if observe_limits:
                if item.sequence not in actual_dispatch:
                    await _emit_request(
                        make_not_dispatched_record(
                            item.sequence, request_id, reason="cancelled_before_dispatch"
                        )
                    )
                    return
                dispatch_offset_ms, dispatch_lag_ms = actual_dispatch[item.sequence]

            if not skip_trace_poll:
                try:
                    trace = await self._fetch_trace(request_id)
                    if call is not None and self._validate_trace is not None:
                        self._validate_trace(call, trace)
                    trace_observed = True
                    await _emit_trace(item.sequence, trace)
                except RunnerFatalError as exc:
                    if exc.failure_type == "trace_timeout":
                        if client_status != "cancelled":
                            if observe_limits:
                                _warn("missing_trace", sequence=item.sequence)
                            else:
                                _set_fatal("missing_trace")
                    elif exc.failure_type in {"gateway_unreachable", "contract_error"}:
                        _set_fatal(exc.failure_type)
                    else:
                        _set_fatal(exc.failure_type)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    _set_fatal("runner_internal_error")

            record = make_sent_record(
                sequence=item.sequence,
                request_id=request_id,
                dispatch_offset_ms=dispatch_offset_ms,
                dispatch_lag_ms=dispatch_lag_ms,
                client_end_offset_ms=client_end_offset_ms,
                http_status=http_status,
                client_status=client_status,
                client_error_type=client_error_type,
                trace_observed=trace_observed,
            )
            await _emit_request(record)

        async def _watch_cancel_event() -> None:
            await effective_cancel.wait()
            _set_fatal("runner_cancelled")

        cancel_watcher = asyncio.create_task(_watch_cancel_event())
        externally_cancelled = False

        try:
            for item in scheduled_entries:
                if effective_cancel.is_set() or stop_dispatch.is_set():
                    break

                deadline = t0 + float(item.scheduled_offset_ms) / 1000.0
                stopped = await self._wait_until_deadline_or_stop(deadline, stop_dispatch)
                if stopped or effective_cancel.is_set() or stop_dispatch.is_set():
                    break

                dispatch_mono = self._clock()
                dispatch_offset_ms = (dispatch_mono - t0) * 1000.0
                dispatch_lag_ms = dispatch_offset_ms - float(item.scheduled_offset_ms)

                if dispatch_lag_ms > float(max_dispatch_lag_ms):
                    if not observe_limits:
                        _set_fatal("generator_limited")
                        break

                if dispatch_mono >= window_end_mono:
                    if not observe_limits:
                        _set_fatal("generator_limited")
                        break

                if active_http_in_flight >= max_in_flight:
                    if not observe_limits:
                        _set_fatal("generator_limited")
                        break
                    request_id = f"{run_id}-req-{item.sequence:05d}"
                    _warn(
                        "in_flight_limit",
                        sequence=item.sequence,
                        scheduled_offset_ms=float(item.scheduled_offset_ms),
                        attempt_offset_ms=dispatch_offset_ms,
                        active_http_in_flight=active_http_in_flight,
                        limit=max_in_flight,
                    )
                    await _emit_request(
                        make_not_dispatched_record(
                            item.sequence, request_id, reason="in_flight_limit"
                        )
                    )
                    continue

                request_id = f"{run_id}-req-{item.sequence:05d}"
                if not observe_limits:
                    sent_sequences.add(item.sequence)
                active_http_in_flight += 1
                if active_http_in_flight > max_observed_in_flight:
                    max_observed_in_flight = active_http_in_flight

                http_task = asyncio.create_task(_run_http_call(item, request_id))
                http_tasks[item.sequence] = http_task
                request_tasks[item.sequence] = asyncio.create_task(
                    _handle_dispatched(
                        item,
                        request_id,
                        dispatch_offset_ms,
                        dispatch_lag_ms,
                        http_task,
                    )
                )
        except asyncio.CancelledError:
            externally_cancelled = True
            _set_fatal("runner_cancelled")
        finally:
            if not cancel_watcher.done():
                cancel_watcher.cancel()
            await asyncio.gather(cancel_watcher, return_exceptions=True)

            await asyncio.shield(
                self._drain_and_cleanup(
                    run_id=run_id,
                    scheduled_entries=scheduled_entries,
                    sent_sequences=sent_sequences,
                    http_tasks=http_tasks,
                    request_tasks=request_tasks,
                    ledger_by_seq=ledger_by_seq,
                    emit_request=_emit_request,
                    first_fatal_type=first_fatal_type,
                    actual_dispatch=actual_dispatch if observe_limits else None,
                    t0_mono=t0,
                )
            )

        if externally_cancelled:
            raise asyncio.CancelledError

        drain_end_offset_ms = round((self._clock() - t0) * 1000.0, 3)
        ledger = [ledger_by_seq[item.sequence] for item in scheduled_entries]
        traces = [traces_by_seq[seq] for seq in sorted(traces_by_seq)]
        sent_count = len(sent_sequences)
        not_dispatched_count = len(scheduled_entries) - sent_count

        return SchedulerResult(
            run_status=(
                "completed"
                if first_fatal_type is None
                else "failed"
                if not observe_limits
                else "cancelled"
                if first_fatal_type == "runner_cancelled"
                else "blocked"
            ),
            failure_type=first_fatal_type,
            t0_mono=t0,
            measured_started_at=measured_started_at,
            drain_end_offset_ms=drain_end_offset_ms,
            planned_count=len(scheduled_entries),
            sent_count=sent_count,
            not_dispatched_count=not_dispatched_count,
            ledger=ledger,
            traces=traces,
            max_observed_in_flight=max_observed_in_flight,
            warnings=warnings,
        )

    async def _drain_and_cleanup(
        self,
        *,
        run_id: str,
        scheduled_entries: Sequence[ScheduledEntryProtocol],
        sent_sequences: set[int],
        http_tasks: dict[int, asyncio.Task[ChatCallResult]],
        request_tasks: dict[int, asyncio.Task[None]],
        ledger_by_seq: dict[int, dict[str, Any]],
        emit_request: Callable[[dict[str, Any]], Awaitable[None]],
        first_fatal_type: str | None,
        actual_dispatch: dict[int, tuple[float, float]] | None = None,
        t0_mono: float = 0.0,
    ) -> None:
        pending_requests = [task for task in request_tasks.values() if not task.done()]
        if pending_requests:
            if first_fatal_type is None:
                await asyncio.gather(*pending_requests, return_exceptions=True)
            elif first_fatal_type == "generator_limited":
                # Allow already-dispatched calls to drain up to bounded deadline, then cancel.
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending_requests, return_exceptions=True),
                        timeout=self._generator_limited_drain_timeout_s,
                    )
                except TimeoutError:
                    for h_task in list(http_tasks.values()):
                        if not h_task.done():
                            h_task.cancel()
                    still_pending = [t for t in pending_requests if not t.done()]
                    if still_pending:
                        try:
                            await asyncio.wait_for(
                                asyncio.gather(*still_pending, return_exceptions=True),
                                timeout=self._cancel_cleanup_timeout_s,
                            )
                        except TimeoutError:
                            for r_task in still_pending:
                                if not r_task.done():
                                    r_task.cancel()
                            await asyncio.gather(*still_pending, return_exceptions=True)
            else:
                # Controlled cancellation or fatal infra/contract failure: close open HTTP calls.
                for h_task in list(http_tasks.values()):
                    if not h_task.done():
                        h_task.cancel()
                still_pending = [t for t in pending_requests if not t.done()]
                if still_pending:
                    try:
                        await asyncio.wait_for(
                            asyncio.gather(*still_pending, return_exceptions=True),
                            timeout=self._cancel_cleanup_timeout_s,
                        )
                    except TimeoutError:
                        for r_task in still_pending:
                            if not r_task.done():
                                r_task.cancel()
                        await asyncio.gather(*still_pending, return_exceptions=True)

        # Guarantee every planned sequence has a deterministic ledger entry on controlled shutdown.
        for item in scheduled_entries:
            if item.sequence not in ledger_by_seq:
                request_id = f"{run_id}-req-{item.sequence:05d}"
                if item.sequence not in sent_sequences:
                    await emit_request(make_not_dispatched_record(item.sequence, request_id))
                else:
                    dispatch, lag = (
                        actual_dispatch[item.sequence]
                        if actual_dispatch is not None
                        else (float(item.scheduled_offset_ms), 0.0)
                    )
                    end = (
                        max(dispatch, (self._clock() - t0_mono) * 1000)
                        if actual_dispatch is not None
                        else float(item.scheduled_offset_ms)
                    )
                    await emit_request(
                        make_sent_record(
                            sequence=item.sequence,
                            request_id=request_id,
                            dispatch_offset_ms=dispatch,
                            dispatch_lag_ms=lag,
                            client_end_offset_ms=end,
                            http_status=None,
                            client_status="cancelled",
                            client_error_type="client_cancelled",
                            trace_observed=False,
                        )
                    )
