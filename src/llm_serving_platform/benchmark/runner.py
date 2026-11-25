"""Reproducible Week 7 experiment runner over the Gateway HTTP boundary."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import urlsplit

import httpx

from llm_serving_platform.benchmark.artifacts import (
    DEFAULT_RUNS_ROOT,
    ArtifactWriter,
    build_environment_metadata,
    utc_now_iso,
    write_text_atomic,
)
from llm_serving_platform.benchmark.client import (
    ChatCallResult,
    GatewayHTTPClient,
    RunnerFatalError,
)
from llm_serving_platform.benchmark.config import (
    ExperimentConfig,
    ExperimentConfigV2,
    load_manifest,
)
from llm_serving_platform.benchmark.scheduler import (
    OpenLoopScheduler,
    ScheduledEntryProtocol,
    SchedulerResult,
    validate_open_loop_schedule,
)
from llm_serving_platform.benchmark.summary import (
    compute_summary_from_traces_file,
    compute_summary_v2,
    read_traces_jsonl,
)
from llm_serving_platform.benchmark.workload import (
    WorkloadEntry,
    load_workload,
    select_measured_entries,
    select_scheduled_entries,
    select_warmup_entries,
)
from llm_serving_platform.common.config import REPO_ROOT


@runtime_checkable
class OpenLoopConfigProtocol(Protocol):
    """Structural contract for Week 8 v2 open-loop experiment configs."""

    @property
    def schema_version(self) -> int: ...

    @property
    def name(self) -> str: ...

    @property
    def model(self) -> Any: ...

    @property
    def topology(self) -> Any: ...

    @property
    def policy(self) -> Any: ...

    @property
    def workload(self) -> Any: ...

    @property
    def seed(self) -> int: ...

    @property
    def warmup(self) -> Any: ...

    @property
    def generation(self) -> Any: ...

    @property
    def runner(self) -> Any: ...

    @property
    def load(self) -> Any: ...

    def to_normalized_dict(self) -> dict[str, Any]: ...


@runtime_checkable
class OpenLoopArtifactWriterProtocol(Protocol):
    """Structural writer seam for Week 8 open-loop artifacts."""

    run_dir: Path
    run_id: str
    traces_path: Path

    def write_manifest(self, config: Any) -> None: ...

    def write_environment(self, metadata: dict[str, Any]) -> None: ...

    def write_schedule(self, records: Sequence[dict[str, Any]]) -> None: ...

    def append_request(self, record: dict[str, Any]) -> None: ...

    def append_trace(self, trace: dict[str, Any]) -> None: ...

    def write_metrics(self, metrics_text: str) -> None: ...

    def write_summary(self, summary: dict[str, Any]) -> None: ...


@dataclass(frozen=True)
class RunResult:
    """Outcome of a single ExperimentRunner execution."""

    run_id: str
    run_dir: Path
    succeeded: bool
    run_status: str
    failure_type: str | None
    summary: dict[str, Any]


def _build_chat_payload(config: Any, entry: WorkloadEntry) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": config.model.id,
        "messages": entry.as_messages_list(),
        "stream": True,
        "temperature": config.generation.temperature,
        "max_tokens": config.generation.max_tokens,
    }
    if config.generation.top_p is not None:
        payload["top_p"] = config.generation.top_p
    return payload


def _sanitize_endpoint_url(url: str) -> str:
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    authority = f"{host}:{parsed.port}" if parsed.port else host
    return f"{parsed.scheme}://{authority}"


def _sanitize_workers(workers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "worker_id": row.get("worker_id"),
            "endpoint": _sanitize_endpoint_url(row["endpoint"])
            if isinstance(row.get("endpoint"), str) and "://" in row["endpoint"]
            else row.get("endpoint"),
            "model": row.get("model"),
            "status": row.get("status"),
            "active_requests": row.get("active_requests"),
        }
        for row in workers
        if isinstance(row, dict)
    ]


def _resolve_workload_sha256(
    config: OpenLoopConfigProtocol,
    *,
    repo_root: Path,
    explicit_sha256: str | None,
) -> str | None:
    if explicit_sha256 is not None:
        return explicit_sha256
    raw_path = Path(config.workload.path)
    candidates = [raw_path] if raw_path.is_absolute() else [repo_root / raw_path, raw_path]
    for candidate in candidates:
        if candidate.is_file():
            return hashlib.sha256(candidate.read_bytes()).hexdigest()
    return None


def _build_schedule_records(
    scheduled_entries: Sequence[ScheduledEntryProtocol],
) -> tuple[list[dict[str, Any]], str]:
    records = [
        {
            "sequence": item.sequence,
            "entry_id": item.entry.entry_id,
            "workload_index": item.entry.index,
            "scheduled_offset_ms": float(item.scheduled_offset_ms),
        }
        for item in scheduled_entries
    ]
    raw_bytes = "".join(
        json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n" for rec in records
    ).encode("utf-8")
    return records, hashlib.sha256(raw_bytes).hexdigest()


def _verify_preflight_workers(
    config: Any,
    workers: list[dict[str, Any]],
) -> None:
    if len(workers) != config.topology.worker_count:
        raise RunnerFatalError(
            "topology_mismatch",
            f"Expected {config.topology.worker_count} workers from manifest, "
            f"observed {len(workers)}",
        )
    unhealthy = [w.get("worker_id") for w in workers if w.get("status") != "HEALTHY"]
    if unhealthy:
        raise RunnerFatalError(
            "gateway_unhealthy",
            f"All workers must be HEALTHY for preflight; non-healthy: {unhealthy}",
        )
    mismatched_models = [
        (w.get("worker_id"), w.get("model"))
        for w in workers
        if w.get("model") is not None and w.get("model") != config.model.id
    ]
    if mismatched_models:
        raise RunnerFatalError(
            "model_mismatch",
            f"Worker model mismatch: expected {config.model.id!r}, got {mismatched_models}",
        )


def _validate_trace_correlation(call: ChatCallResult, trace: dict[str, Any]) -> None:
    if trace.get("request_id") != call.request_id:
        raise RunnerFatalError("contract_error", "Trace request_id mismatch")
    if call.response_worker_id and trace.get("worker_id"):
        if trace["worker_id"] != call.response_worker_id:
            raise RunnerFatalError(
                "contract_error",
                f"X-Worker-ID header {call.response_worker_id!r} != "
                f"trace worker_id {trace['worker_id']!r}",
            )
    if call.saw_sse_error and trace.get("status") == "completed":
        raise RunnerFatalError(
            "contract_error",
            "Gateway reported completed trace despite SSE error payload",
        )


class ExperimentRunner:
    """Orchestrates preflight, warm-up, measured traffic, trace/metric capture, and summaries."""

    def __init__(
        self,
        *,
        runs_root: Path | None = None,
        client: httpx.AsyncClient | None = None,
        control_client: httpx.AsyncClient | None = None,
        repo_root: Path = REPO_ROOT,
        trace_poll_timeout_s: float = 5.0,
        trace_poll_interval_s: float = 0.02,
        clock: Callable[[], float] = time.monotonic,
        sleep_until: Callable[[float], Awaitable[None]] | None = None,
        generator_limited_drain_timeout_s: float = 30.0,
        cancel_cleanup_timeout_s: float = 10.0,
        run_context: dict[str, Any] | None = None,
    ) -> None:
        self._runs_root = runs_root or DEFAULT_RUNS_ROOT
        self._client = client
        self._control_client = control_client
        self._repo_root = repo_root
        self._trace_poll_timeout_s = trace_poll_timeout_s
        self._trace_poll_interval_s = trace_poll_interval_s
        self._clock = clock
        self._sleep_until = sleep_until
        self._generator_limited_drain_timeout_s = generator_limited_drain_timeout_s
        self._cancel_cleanup_timeout_s = cancel_cleanup_timeout_s
        allowed_context = {
            "protocol_sha256",
            "test_mode",
            "cache_control",
            "worker_runtime_args",
            "worker_gpu_bindings",
            "campaign_run_index",
        }
        self._run_context = {k: v for k, v in (run_context or {}).items() if k in allowed_context}

    async def run_from_manifest(
        self,
        manifest_path: Path | str,
        *,
        base_dir: Path | None = None,
        gateway_url_override: str | None = None,
        run_id: str | None = None,
    ) -> RunResult:
        """Validate manifest and workload, then execute the full reproducible experiment."""
        config, workload_path = load_manifest(
            manifest_path,
            base_dir=base_dir,
            gateway_url_override=gateway_url_override,
        )
        if isinstance(config, ExperimentConfigV2):
            entries = load_workload(
                workload_path,
                schema_version=2,
                expected_count=config.workload.request_count,
                measurement_window_ms=config.load.measurement_window_ms,
            )
            schedule = select_scheduled_entries(entries, config.workload.request_count)
            return await self.run_open_loop(
                config,
                schedule,
                run_id=run_id,
                workload_sha256=hashlib.sha256(
                    workload_path.read_bytes().replace(b"\r\n", b"\n")
                ).hexdigest(),
            )
        entries = load_workload(workload_path, min_entries=config.workload.request_count)
        return await self.run(config, entries, run_id=run_id)

    async def run(
        self,
        config: ExperimentConfig,
        entries: list[WorkloadEntry],
        *,
        run_id: str | None = None,
    ) -> RunResult:
        """Execute one experiment run and persist canonical artifacts (including on failure)."""
        started_at = utc_now_iso()
        writer = ArtifactWriter.create(
            runs_root=self._runs_root,
            config=config,
            run_id=run_id,
            repo_root=self._repo_root,
        )
        writer.write_manifest(config)

        observed_workers: list[dict[str, Any]] = []
        env_meta = build_environment_metadata(
            run_id=writer.run_id,
            config=config,
            started_at=started_at,
            ended_at=None,
            observed_workers=observed_workers,
            repo_root=self._repo_root,
        )
        writer.write_environment(env_meta)

        failure_type: str | None = None
        run_status = "completed"

        async with GatewayHTTPClient(
            config.runner.gateway_url,
            timeout_s=config.runner.request_timeout_s,
            client=self._client,
        ) as gw:
            try:
                # 1. Preflight
                await gw.check_health()
                observed_workers = await gw.list_workers()
                _verify_preflight_workers(config, observed_workers)
                env_meta = build_environment_metadata(
                    run_id=writer.run_id,
                    config=config,
                    started_at=started_at,
                    ended_at=None,
                    observed_workers=observed_workers,
                    repo_root=self._repo_root,
                )
                writer.write_environment(env_meta)

                # 2. Warm-up (excluded from measured traces.jsonl and summary.json)
                warmup_entries = select_warmup_entries(entries, config.warmup.request_count)
                for idx, entry in enumerate(warmup_entries):
                    warmup_id = f"{writer.run_id}-warmup-{idx:04d}"
                    call = await gw.stream_chat_completion(
                        _build_chat_payload(config, entry),
                        warmup_id,
                    )
                    if call.status_code != 200 or call.saw_sse_error:
                        raise RunnerFatalError(
                            "warmup_failed",
                            f"Warm-up request {warmup_id} failed "
                            f"(http={call.status_code}, sse_error={call.saw_sse_error})",
                        )
                    warmup_trace = await gw.fetch_terminal_trace(
                        warmup_id,
                        timeout_s=self._trace_poll_timeout_s,
                        poll_interval_s=self._trace_poll_interval_s,
                    )
                    if warmup_trace.get("status") != "completed":
                        raise RunnerFatalError(
                            "warmup_failed",
                            f"Warm-up request {warmup_id} ended with "
                            f"status={warmup_trace.get('status')!r}",
                        )

                # 3. Measured workload (bounded concurrency, deterministic ordering)
                measured_entries = select_measured_entries(
                    entries,
                    config.workload.request_count,
                    seed=config.seed,
                )
                await self._execute_measured(gw, config, writer, measured_entries)

                # 4. Verify observed routing policy from persisted traces
                persisted = read_traces_jsonl(writer.traces_path)
                observed_policies = {
                    t["routing"]["policy"]
                    for t in persisted
                    if isinstance(t.get("routing"), dict) and t["routing"].get("policy")
                }
                if observed_policies and observed_policies != {config.policy.name}:
                    raise RunnerFatalError(
                        "policy_mismatch",
                        f"Expected policy {config.policy.name!r}, "
                        f"observed {sorted(observed_policies)}",
                    )

            except RunnerFatalError as exc:
                run_status = "failed"
                failure_type = exc.failure_type
            except Exception:
                run_status = "failed"
                failure_type = "runner_internal_error"
            finally:
                # 5. Scrape final /metrics snapshot (best-effort on failure)
                try:
                    metrics_text = await gw.scrape_metrics()
                    writer.write_metrics(metrics_text)
                except RunnerFatalError:
                    pass

                # 6. Always regenerate summary.json strictly by re-reading traces.jsonl
                failure_obj = {"type": failure_type} if failure_type is not None else None
                summary = compute_summary_from_traces_file(
                    writer.traces_path,
                    run_id=writer.run_id,
                    run_status=run_status,
                    planned_requests=config.workload.request_count,
                    expected_policy=config.policy.name,
                    failure=failure_obj,
                )
                writer.write_summary(summary)

                # 7. Finalize environment.json with ended_at
                ended_at = utc_now_iso()
                env_meta = build_environment_metadata(
                    run_id=writer.run_id,
                    config=config,
                    started_at=started_at,
                    ended_at=ended_at,
                    observed_workers=observed_workers,
                    repo_root=self._repo_root,
                )
                writer.write_environment(env_meta)

        return RunResult(
            run_id=writer.run_id,
            run_dir=writer.run_dir,
            succeeded=(run_status == "completed"),
            run_status=run_status,
            failure_type=failure_type,
            summary=summary,
        )

    async def _run_warmup(
        self,
        gw: GatewayHTTPClient,
        config: Any,
        run_id: str,
        warmup_entries: Sequence[WorkloadEntry],
    ) -> None:
        for idx, entry in enumerate(warmup_entries):
            warmup_id = f"{run_id}-warmup-{idx:04d}"
            call = await gw.stream_chat_completion(
                _build_chat_payload(config, entry),
                warmup_id,
            )
            if call.status_code != 200 or call.saw_sse_error:
                raise RunnerFatalError(
                    "warmup_failed",
                    f"Warm-up request {warmup_id} failed "
                    f"(http={call.status_code}, sse_error={call.saw_sse_error})",
                )
            warmup_trace = await gw.fetch_terminal_trace(
                warmup_id,
                timeout_s=self._trace_poll_timeout_s,
                poll_interval_s=self._trace_poll_interval_s,
            )
            if warmup_trace.get("status") != "completed":
                raise RunnerFatalError(
                    "warmup_failed",
                    f"Warm-up request {warmup_id} ended with status={warmup_trace.get('status')!r}",
                )

    async def _wait_for_workers_idle(
        self,
        gw: GatewayHTTPClient,
        *,
        timeout_s: float = 5.0,
        poll_interval_s: float = 0.02,
    ) -> list[dict[str, Any]]:
        deadline = time.monotonic() + timeout_s
        workers = await gw.list_workers()
        while any(row.get("active_requests", 0) != 0 for row in workers):
            if time.monotonic() >= deadline:
                return workers
            await asyncio.sleep(poll_interval_s)
            workers = await gw.list_workers()
        return workers

    async def run_open_loop(
        self,
        config: OpenLoopConfigProtocol,
        scheduled_entries: Sequence[ScheduledEntryProtocol],
        *,
        run_id: str | None = None,
        warmup_entries: Sequence[WorkloadEntry] | None = None,
        workload_sha256: str | None = None,
        writer: OpenLoopArtifactWriterProtocol | None = None,
        cancel_event: asyncio.Event | None = None,
        summary_builder: Callable[..., dict[str, Any]] | None = None,
    ) -> RunResult:
        """Execute a Week 8 open-loop controlled benchmark run over the Gateway HTTP API."""
        if config.schema_version != 2:
            raise ValueError(
                f"Open-loop execution requires schema_version=2, got {config.schema_version!r}"
            )
        if config.load.mode != "open_loop":
            raise ValueError(f"Expected load.mode='open_loop', got {config.load.mode!r}")
        if len(scheduled_entries) != config.workload.request_count:
            raise ValueError(
                f"Scheduled entry count ({len(scheduled_entries)}) != "
                f"workload.request_count ({config.workload.request_count})"
            )
        validate_open_loop_schedule(
            scheduled_entries,
            measurement_window_ms=float(config.load.measurement_window_ms),
            max_in_flight=int(config.runner.max_in_flight),
            max_dispatch_lag_ms=float(config.runner.max_dispatch_lag_ms),
        )

        started_at = utc_now_iso()
        active_writer: OpenLoopArtifactWriterProtocol = (
            writer
            if writer is not None
            else ArtifactWriter.create(
                runs_root=self._runs_root,
                config=config,
                run_id=run_id,
                repo_root=self._repo_root,
            )
        )
        active_writer.write_manifest(config)

        schedule_records, schedule_sha256 = _build_schedule_records(scheduled_entries)
        active_writer.write_schedule(schedule_records)
        resolved_workload_sha = _resolve_workload_sha256(
            config,
            repo_root=self._repo_root,
            explicit_sha256=workload_sha256,
        )

        observed_workers: list[dict[str, Any]] = []
        final_workers: list[dict[str, Any]] = []
        measured_started_at: str | None = None
        drain_end_offset_ms: float | None = None
        pool_limits: dict[str, Any] = {}
        sched_result: SchedulerResult | None = None
        measurement_started = False
        failure_type: str | None = None
        run_status = "completed"
        if summary_builder is None and isinstance(config, ExperimentConfigV2):
            summary_builder = compute_summary_v2

        def _build_v2_env(ended_at: str | None) -> dict[str, Any]:
            env_meta = build_environment_metadata(
                run_id=active_writer.run_id,
                config=config,  # type: ignore[arg-type]
                started_at=started_at,
                ended_at=ended_at,
                observed_workers=observed_workers,
                repo_root=self._repo_root,
            )
            env_meta.update(
                {
                    "schema_version": 2,
                    "workload_sha256": resolved_workload_sha,
                    "schedule_sha256": schedule_sha256,
                    "measured_started_at": measured_started_at,
                    "measurement_window_ms": float(config.load.measurement_window_ms),
                    "drain_end_offset_ms": drain_end_offset_ms,
                    "final_workers": _sanitize_workers(final_workers),
                    "http_pool_limits": pool_limits,
                    "run_status": run_status if ended_at is not None else "running",
                    "failure": {"type": failure_type} if failure_type else None,
                }
            )
            env_meta.update(self._run_context)
            return env_meta

        active_writer.write_environment(_build_v2_env(None))

        inf_pool_cap = max(int(config.runner.max_in_flight) + 16, 64)
        ctrl_pool_cap = max(32, min(int(config.runner.max_in_flight), 64))

        async with GatewayHTTPClient(
            config.runner.gateway_url,
            timeout_s=float(config.runner.request_timeout_s),
            client=self._client,
            control_client=self._control_client,
            inference_max_connections=inf_pool_cap,
            control_max_connections=ctrl_pool_cap,
            control_timeout_s=max(10.0, self._trace_poll_timeout_s + 5.0),
        ) as gw:
            pool_limits = gw.pool_limits_metadata()
            try:
                # 1. Preflight health, topology, and model check
                await gw.check_health()
                observed_workers = await gw.list_workers()
                _verify_preflight_workers(config, observed_workers)
                active_writer.write_environment(_build_v2_env(None))

                # 2. Warm-up (sequential, excluded from schedule/requests/traces)
                warmup_pool = (
                    list(warmup_entries)
                    if warmup_entries is not None
                    else [item.entry for item in scheduled_entries]
                )
                selected_warmup = select_warmup_entries(warmup_pool, config.warmup.request_count)
                await self._run_warmup(gw, config, active_writer.run_id, selected_warmup)

                # 3. Wait for workers to return idle after warm-up
                idle_workers = await self._wait_for_workers_idle(gw, timeout_s=5.0)
                if any(w.get("active_requests", 0) != 0 for w in idle_workers):
                    raise RunnerFatalError(
                        "workers_not_idle",
                        "Workers did not return to 0 active requests after warm-up",
                    )

                # 4. Pre-measurement Prometheus baseline snapshot
                before_metrics = await gw.scrape_metrics()
                if hasattr(active_writer, "write_before_metrics"):
                    active_writer.write_before_metrics(before_metrics)
                else:
                    write_text_atomic(active_writer.run_dir / "metrics.before.prom", before_metrics)

                # 5. Monotonic t0 + open-loop dispatch + trace collection + drain
                scheduler = OpenLoopScheduler(
                    send_chat=lambda entry, req_id: gw.stream_chat_completion(
                        _build_chat_payload(config, entry),
                        req_id,
                    ),
                    fetch_trace=lambda req_id: gw.fetch_terminal_trace(
                        req_id,
                        timeout_s=self._trace_poll_timeout_s,
                        poll_interval_s=self._trace_poll_interval_s,
                    ),
                    validate_trace=_validate_trace_correlation,
                    on_trace=active_writer.append_trace,
                    on_request=active_writer.append_request,
                    clock=self._clock,
                    sleep_until=self._sleep_until,
                    request_timeout_s=float(config.runner.request_timeout_s),
                    generator_limited_drain_timeout_s=self._generator_limited_drain_timeout_s,
                    cancel_cleanup_timeout_s=self._cancel_cleanup_timeout_s,
                )
                measurement_started = True
                sched_result = await scheduler.execute(
                    run_id=active_writer.run_id,
                    scheduled_entries=scheduled_entries,
                    measurement_window_ms=float(config.load.measurement_window_ms),
                    max_in_flight=int(config.runner.max_in_flight),
                    max_dispatch_lag_ms=float(config.runner.max_dispatch_lag_ms),
                    cancel_event=cancel_event,
                )
                measured_started_at = sched_result.measured_started_at
                drain_end_offset_ms = sched_result.drain_end_offset_ms
                run_status = sched_result.run_status
                failure_type = sched_result.failure_type

                # 6. Verify observed routing policy on persisted traces
                persisted = read_traces_jsonl(active_writer.traces_path)
                observed_policies = {
                    t["routing"]["policy"]
                    for t in persisted
                    if isinstance(t.get("routing"), dict) and t["routing"].get("policy")
                }
                if observed_policies and observed_policies != {config.policy.name}:
                    raise RunnerFatalError(
                        "policy_mismatch",
                        f"Expected policy {config.policy.name!r}, "
                        f"observed {sorted(observed_policies)}",
                    )
                if run_status == "completed" and len(persisted) != sched_result.sent_count:
                    raise RunnerFatalError(
                        "missing_trace",
                        f"Expected {sched_result.sent_count} traces, persisted {len(persisted)}",
                    )
            except RunnerFatalError as exc:
                run_status = "failed"
                failure_type = exc.failure_type
            except asyncio.CancelledError:
                run_status = "failed"
                failure_type = "runner_cancelled"
            except Exception:
                run_status = "failed"
                failure_type = "runner_internal_error"
            finally:
                if not measurement_started:
                    # No measured inference was sent: record only genuine aborted slots.
                    from llm_serving_platform.benchmark.scheduler import make_not_dispatched_record

                    for item in scheduled_entries:
                        active_writer.append_request(
                            make_not_dispatched_record(
                                item.sequence, f"{active_writer.run_id}-req-{item.sequence:05d}"
                            )
                        )
                # 7. Check final worker active_requests == 0 (best-effort on unreachable gateway)
                try:
                    final_workers = await self._wait_for_workers_idle(gw, timeout_s=2.0)
                    if run_status == "completed" and any(
                        w.get("active_requests", 0) != 0 for w in final_workers
                    ):
                        run_status = "failed"
                        failure_type = "workers_not_idle"
                except RunnerFatalError:
                    if run_status == "completed":
                        run_status = "failed"
                        failure_type = "gateway_unreachable"

                # 8. Scrape final /metrics snapshot (best-effort)
                try:
                    metrics_text = await gw.scrape_metrics()
                    active_writer.write_metrics(metrics_text)
                except RunnerFatalError:
                    pass

                ended_at = utc_now_iso()
                active_writer.write_environment(_build_v2_env(ended_at))

                failure_obj = {"type": failure_type} if failure_type is not None else None
                if summary_builder is not None:
                    summary = summary_builder(
                        run_dir=active_writer.run_dir,
                        run_id=active_writer.run_id,
                        run_status=run_status,
                        config=config,
                        failure=failure_obj,
                    )
                    active_writer.write_summary(summary)
                    run_status = summary["run_status"]
                    failure_type = (summary.get("failure") or {}).get("type")
                    active_writer.write_environment(_build_v2_env(ended_at))
                else:
                    summary = {
                        "schema_version": 2,
                        "run_id": active_writer.run_id,
                        "run_status": run_status,
                        "policy": config.policy.name,
                        "requests": {
                            "planned": config.workload.request_count,
                            "sent": sched_result.sent_count if sched_result else 0,
                            "not_dispatched": (
                                sched_result.not_dispatched_count
                                if sched_result
                                else config.workload.request_count
                            ),
                        },
                        **({"failure": failure_obj} if failure_obj is not None else {}),
                    }

        return RunResult(
            run_id=active_writer.run_id,
            run_dir=active_writer.run_dir,
            succeeded=(run_status == "completed"),
            run_status=run_status,
            failure_type=failure_type,
            summary=summary,
        )

    async def _execute_measured(
        self,
        gw: GatewayHTTPClient,
        config: ExperimentConfig,
        writer: ArtifactWriter,
        measured_entries: list[WorkloadEntry],
    ) -> None:
        semaphore = asyncio.Semaphore(config.runner.concurrency)
        write_lock = asyncio.Lock()
        abort_event = asyncio.Event()
        first_fatal: RunnerFatalError | None = None

        async def _run_one(idx: int, entry: WorkloadEntry) -> None:
            nonlocal first_fatal
            async with semaphore:
                if abort_event.is_set():
                    return
                request_id = f"{writer.run_id}-req-{idx:05d}"
                payload = _build_chat_payload(config, entry)
                try:
                    call = await gw.stream_chat_completion(payload, request_id)
                    trace = await gw.fetch_terminal_trace(
                        request_id,
                        timeout_s=self._trace_poll_timeout_s,
                        poll_interval_s=self._trace_poll_interval_s,
                    )
                    _validate_trace_correlation(call, trace)
                except RunnerFatalError as exc:
                    if first_fatal is None:
                        first_fatal = exc
                    abort_event.set()
                    return
                async with write_lock:
                    writer.append_trace(trace)

        tasks = [
            asyncio.create_task(_run_one(idx, entry)) for idx, entry in enumerate(measured_entries)
        ]
        await asyncio.gather(*tasks)
        if first_fatal is not None:
            raise first_fatal
