"""Reproducible Week 7 experiment runner over the Gateway HTTP boundary."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from llm_serving_platform.benchmark.artifacts import (
    DEFAULT_RUNS_ROOT,
    ArtifactWriter,
    build_environment_metadata,
    utc_now_iso,
)
from llm_serving_platform.benchmark.client import (
    ChatCallResult,
    GatewayHTTPClient,
    RunnerFatalError,
)
from llm_serving_platform.benchmark.config import ExperimentConfig, load_manifest
from llm_serving_platform.benchmark.summary import (
    compute_summary_from_traces_file,
    read_traces_jsonl,
)
from llm_serving_platform.benchmark.workload import (
    WorkloadEntry,
    load_workload,
    select_measured_entries,
    select_warmup_entries,
)
from llm_serving_platform.common.config import REPO_ROOT


@dataclass(frozen=True)
class RunResult:
    """Outcome of a single ExperimentRunner execution."""

    run_id: str
    run_dir: Path
    succeeded: bool
    run_status: str
    failure_type: str | None
    summary: dict[str, Any]


def _build_chat_payload(config: ExperimentConfig, entry: WorkloadEntry) -> dict[str, Any]:
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


def _verify_preflight_workers(
    config: ExperimentConfig,
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
        repo_root: Path = REPO_ROOT,
        trace_poll_timeout_s: float = 5.0,
        trace_poll_interval_s: float = 0.02,
    ) -> None:
        self._runs_root = runs_root or DEFAULT_RUNS_ROOT
        self._client = client
        self._repo_root = repo_root
        self._trace_poll_timeout_s = trace_poll_timeout_s
        self._trace_poll_interval_s = trace_poll_interval_s

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
            asyncio.create_task(_run_one(idx, entry))
            for idx, entry in enumerate(measured_entries)
        ]
        await asyncio.gather(*tasks)
        if first_fatal is not None:
            raise first_fatal
