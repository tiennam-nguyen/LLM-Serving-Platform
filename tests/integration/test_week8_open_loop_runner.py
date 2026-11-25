"""Integration tests for Week 8 open-loop ExperimentRunner lifecycle and V1 regression."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest
from prometheus_client.parser import text_string_to_metric_families

from llm_serving_platform.benchmark.config import ExperimentConfig
from llm_serving_platform.benchmark.runner import ExperimentRunner
from llm_serving_platform.benchmark.workload import WorkloadEntry
from tests.week6_workers import mock_gateway

SECRET_PROMPT_SENTINEL = "W8_SECRET_PROMPT_SENTINEL_9913"


@dataclass(frozen=True)
class _ModelSpec:
    id: str


@dataclass(frozen=True)
class _TopologySpec:
    worker_count: int


@dataclass(frozen=True)
class _PolicySpec:
    name: str


@dataclass(frozen=True)
class _WorkloadSpec:
    path: str
    request_count: int
    profile: str = "steady"


@dataclass(frozen=True)
class _WarmupSpec:
    request_count: int


@dataclass(frozen=True)
class _GenerationSpec:
    temperature: float
    max_tokens: int
    top_p: float | None = 1.0


@dataclass(frozen=True)
class _RunnerSpec:
    gateway_url: str
    request_timeout_s: float
    max_in_flight: int
    max_dispatch_lag_ms: float


@dataclass(frozen=True)
class _LoadSpec:
    mode: str
    measurement_window_ms: float


@dataclass(frozen=True)
class _V2ConfigFixture:
    schema_version: int
    name: str
    model: _ModelSpec
    topology: _TopologySpec
    policy: _PolicySpec
    workload: _WorkloadSpec
    seed: int
    warmup: _WarmupSpec
    generation: _GenerationSpec
    runner: _RunnerSpec
    load: _LoadSpec

    def to_normalized_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "model": {"id": self.model.id},
            "topology": {"worker_count": self.topology.worker_count},
            "policy": {"name": self.policy.name},
            "workload": {
                "path": self.workload.path,
                "profile": self.workload.profile,
                "request_count": self.workload.request_count,
            },
            "seed": self.seed,
            "warmup": {"request_count": self.warmup.request_count},
            "generation": {
                "temperature": self.generation.temperature,
                "max_tokens": self.generation.max_tokens,
                "top_p": self.generation.top_p,
            },
            "runner": {
                "gateway_url": self.runner.gateway_url,
                "request_timeout_s": self.runner.request_timeout_s,
                "max_in_flight": self.runner.max_in_flight,
                "max_dispatch_lag_ms": self.runner.max_dispatch_lag_ms,
            },
            "load": {
                "mode": self.load.mode,
                "measurement_window_ms": self.load.measurement_window_ms,
            },
        }


@dataclass(frozen=True)
class _ScheduledFixture:
    sequence: int
    scheduled_offset_ms: float
    entry: WorkloadEntry


class _VirtualClock:
    def __init__(self) -> None:
        self.now = 5000.0

    def clock(self) -> float:
        return self.now

    async def sleep_until(self, deadline: float) -> None:
        for _ in range(5):
            await asyncio.sleep(0)
        if deadline > self.now:
            self.now = deadline
        for _ in range(3):
            await asyncio.sleep(0)


def _make_v2_config(
    tmp_path: Path,
    *,
    policy: str = "round_robin",
    request_count: int = 4,
    warmup_count: int = 2,
    request_timeout_s: float = 10.0,
    max_in_flight: int = 16,
    max_dispatch_lag_ms: float = 50.0,
    measurement_window_ms: float = 1000.0,
) -> tuple[_V2ConfigFixture, Path, str]:
    workload_path = tmp_path / "workload_v2.jsonl"
    lines = [
        json.dumps(
            {
                "id": f"q-{i:03d}",
                "sequence": i,
                "scheduled_offset_ms": i * 100.0,
                "messages": [{"role": "user", "content": f"{SECRET_PROMPT_SENTINEL} item {i}"}],
            }
        )
        for i in range(max(request_count, warmup_count, 4))
    ]
    workload_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    workload_path.write_bytes(workload_bytes)
    workload_sha = hashlib.sha256(workload_bytes).hexdigest()

    cfg = _V2ConfigFixture(
        schema_version=2,
        name=f"w8-test-{policy}",
        model=_ModelSpec(id="test-model"),
        topology=_TopologySpec(worker_count=2),
        policy=_PolicySpec(name=policy),
        workload=_WorkloadSpec(path=str(workload_path), request_count=request_count),
        seed=42,
        warmup=_WarmupSpec(request_count=warmup_count),
        generation=_GenerationSpec(temperature=0.0, max_tokens=16, top_p=1.0),
        runner=_RunnerSpec(
            gateway_url="http://mock-gateway",
            request_timeout_s=request_timeout_s,
            max_in_flight=max_in_flight,
            max_dispatch_lag_ms=max_dispatch_lag_ms,
        ),
        load=_LoadSpec(mode="open_loop", measurement_window_ms=measurement_window_ms),
    )
    return cfg, workload_path, workload_sha


def _make_schedule(offsets_ms: list[float]) -> list[_ScheduledFixture]:
    return [
        _ScheduledFixture(
            sequence=idx,
            scheduled_offset_ms=off,
            entry=WorkloadEntry(
                index=idx,
                entry_id=f"q-{idx:03d}",
                messages=({"role": "user", "content": f"{SECRET_PROMPT_SENTINEL} item {idx}"},),
            ),
        )
        for idx, off in enumerate(offsets_ms)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["round_robin", "least_loaded"])
async def test_open_loop_runner_lifecycle_warmup_isolation_and_artifacts(
    tmp_path: Path, policy: str
):
    """Full lifecycle with warm-up excluded, metrics.before.prom, schedule, requests, traces."""
    cfg, _, workload_sha = _make_v2_config(tmp_path, policy=policy, request_count=4, warmup_count=2)
    schedule = _make_schedule([0.0, 100.0, 200.0, 300.0])
    vclock = _VirtualClock()
    run_id = f"20261125T000000Z-lifecycle-{policy}"

    async with mock_gateway(policy, "test-model") as (_, client, controls):
        # Hold req 0 and req 1 across later dispatches, completing req 3 before req 1 (OOO).
        controls.hold(f"{run_id}-req-00000")
        controls.hold(f"{run_id}-req-00001")

        original_send = client.send
        req3_finished = False
        req0_active_at_req3_dispatch = False

        async def track_and_release(request: httpx.Request, **kwargs):
            nonlocal req3_finished, req0_active_at_req3_dispatch
            rid = request.headers.get("X-Request-ID", "")
            if request.url.path == "/v1/chat/completions" and rid == f"{run_id}-req-00003":
                req0_active_at_req3_dispatch = not controls.gates[f"{run_id}-req-00000"].is_set()
                resp = await original_send(request, **kwargs)
                req3_finished = True
                controls.finish(f"{run_id}-req-00000")
                controls.finish(f"{run_id}-req-00001")
                return resp
            return await original_send(request, **kwargs)

        client.send = track_and_release  # type: ignore[method-assign]

        runner = ExperimentRunner(
            runs_root=tmp_path / "runs",
            client=client,
            clock=vclock.clock,
            sleep_until=vclock.sleep_until,
        )
        result = await runner.run_open_loop(
            cfg,
            schedule,
            run_id=run_id,
            workload_sha256=workload_sha,
        )

        assert result.succeeded is True
        assert result.run_status == "completed"
        assert req0_active_at_req3_dispatch is True
        assert req3_finished is True

        expected_files = {
            "manifest.json",
            "environment.json",
            "schedule.jsonl",
            "requests.jsonl",
            "traces.jsonl",
            "metrics.before.prom",
            "metrics.prom",
        }
        assert expected_files.issubset({p.name for p in result.run_dir.iterdir()})

        # Warm-up ran (2 warmup + 4 measured = 6 worker calls), isolated from measured artifacts.
        assert sum(controls.calls.values()) == 6
        before_families = {
            f.name: f
            for f in text_string_to_metric_families(
                (result.run_dir / "metrics.before.prom").read_text(encoding="utf-8")
            )
        }
        before_total = sum(
            s.value
            for s in before_families["llm_gateway_requests"].samples
            if s.name == "llm_gateway_requests_total"
        )
        assert int(before_total) == 2

        after_families = {
            f.name: f
            for f in text_string_to_metric_families(
                (result.run_dir / "metrics.prom").read_text(encoding="utf-8")
            )
        }
        after_total = sum(
            s.value
            for s in after_families["llm_gateway_requests"].samples
            if s.name == "llm_gateway_requests_total"
        )
        assert int(after_total) == 6

        sched_rows = [
            json.loads(line)
            for line in (result.run_dir / "schedule.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        req_rows = sorted(
            (
                json.loads(line)
                for line in (result.run_dir / "requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            key=lambda row: row["sequence"],
        )
        trace_rows = [
            json.loads(line)
            for line in (result.run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        assert len(sched_rows) == 4
        assert len(req_rows) == 4
        assert len(trace_rows) == 4
        assert all("-warmup-" not in r["request_id"] for r in req_rows)
        assert all("-warmup-" not in t["request_id"] for t in trace_rows)
        assert {r["request_id"] for r in req_rows} == {t["request_id"] for t in trace_rows}
        assert [r["dispatch_offset_ms"] for r in req_rows] == pytest.approx(
            [0.0, 100.0, 200.0, 300.0], abs=0.01
        )
        # Req 3 completed before Req 0 and Req 1 (out-of-order completion).
        assert req_rows[3]["client_end_offset_ms"] <= req_rows[0]["client_end_offset_ms"]

        env_meta = json.loads((result.run_dir / "environment.json").read_text(encoding="utf-8"))
        assert env_meta["schema_version"] == 2
        assert env_meta["workload_sha256"] == workload_sha
        expected_sched_sha = hashlib.sha256(
            (result.run_dir / "schedule.jsonl").read_bytes()
        ).hexdigest()
        assert env_meta["schedule_sha256"] == expected_sched_sha
        assert env_meta["measurement_window_ms"] == 1000.0
        assert env_meta["measured_started_at"]
        assert env_meta["drain_end_offset_ms"] >= 0.0
        assert all(w["active_requests"] == 0 for w in env_meta["final_workers"])

        # Zero prompt leakage across all files in run_dir.
        for path in result.run_dir.iterdir():
            assert SECRET_PROMPT_SENTINEL not in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_open_loop_runner_policy_mismatch_and_gateway_503_handling(tmp_path: Path):
    """Policy mismatch is fatal; mid-run 503/SSE error continues schedule when trace exists."""
    cfg_err, _, wsha = _make_v2_config(
        tmp_path, policy="least_loaded", request_count=3, warmup_count=0
    )
    schedule = _make_schedule([0.0, 50.0, 100.0])
    vclock = _VirtualClock()
    run_id = "20261125T000001Z-sse-err"

    async with mock_gateway("least_loaded", "test-model") as (_, client, controls):
        controls.failures[f"{run_id}-req-00001"] = "pre"
        runner = ExperimentRunner(
            runs_root=tmp_path / "runs-err",
            client=client,
            clock=vclock.clock,
            sleep_until=vclock.sleep_until,
        )
        res = await runner.run_open_loop(cfg_err, schedule, run_id=run_id, workload_sha256=wsha)
        assert res.succeeded is True
        req_rows = sorted(
            (
                json.loads(line)
                for line in (res.run_dir / "requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            key=lambda row: row["sequence"],
        )
        assert req_rows[1]["client_status"] == "error"
        assert req_rows[1]["client_error_type"] == "sse_error"
        assert req_rows[1]["trace_observed"] is True
        assert req_rows[2]["client_status"] == "completed"

    # Policy mismatch must fail loudly and preserve partial traces/ledger.
    cfg_mis, _, wsha2 = _make_v2_config(
        tmp_path, policy="least_loaded", request_count=2, warmup_count=0
    )
    async with mock_gateway("round_robin", "test-model") as (_, client, _):
        runner2 = ExperimentRunner(
            runs_root=tmp_path / "runs-mis",
            client=client,
            clock=_VirtualClock().clock,
            sleep_until=_VirtualClock().sleep_until,
        )
        res2 = await runner2.run_open_loop(
            cfg_mis, _make_schedule([0.0, 50.0]), workload_sha256=wsha2
        )
        assert res2.succeeded is False
        assert res2.run_status == "failed"
        assert res2.failure_type == "policy_mismatch"
        assert (res2.run_dir / "traces.jsonl").is_file()
        assert (res2.run_dir / "requests.jsonl").is_file()


@pytest.mark.asyncio
async def test_open_loop_runner_client_timeout_and_missing_trace_against_mock_gateway(
    tmp_path: Path,
):
    """Client timeout with real gateway trace succeeds; missing trace fails cleanly."""
    cfg, _, wsha = _make_v2_config(
        tmp_path,
        policy="round_robin",
        request_count=2,
        warmup_count=0,
        request_timeout_s=0.25,
        max_dispatch_lag_ms=500.0,
    )
    schedule = _make_schedule([0.0, 10.0])
    run_id = "20261125T000002Z-timeout-ok"

    async with mock_gateway("round_robin", "test-model") as (_, client, controls):
        controls.hold(f"{run_id}-req-00000")
        runner = ExperimentRunner(
            runs_root=tmp_path / "runs-to",
            client=client,
            trace_poll_timeout_s=1.0,
            trace_poll_interval_s=0.01,
        )
        res = await runner.run_open_loop(cfg, schedule, run_id=run_id, workload_sha256=wsha)
        assert res.succeeded is True
        req_rows = sorted(
            (
                json.loads(line)
                for line in (res.run_dir / "requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            key=lambda row: row["sequence"],
        )
        assert req_rows[0]["client_status"] == "error"
        assert req_rows[0]["client_error_type"] == "client_timeout"
        assert req_rows[0]["trace_observed"] is True
        assert req_rows[1]["client_status"] == "completed"

        workers = (await client.get("/control/v1/workers")).json()
        assert all(w["active_requests"] == 0 for w in workers)

    # Missing trace after client timeout must fail the run and never synthesize a fake trace.
    cfg_miss, _, wsha_miss = _make_v2_config(
        tmp_path,
        policy="round_robin",
        request_count=2,
        warmup_count=0,
        request_timeout_s=0.25,
        max_dispatch_lag_ms=500.0,
    )
    async with mock_gateway("round_robin", "test-model") as (_, client2, controls2):
        run_miss_id = "20261125T000004Z-missing-trace"
        controls2.hold(f"{run_miss_id}-req-00000")
        orig_get = client2.get

        async def drop_req0_trace(url, *args, **kwargs):
            if isinstance(url, str) and url.endswith(f"{run_miss_id}-req-00000"):
                return httpx.Response(404, request=httpx.Request("GET", f"http://mock-gateway{url}"))
            return await orig_get(url, *args, **kwargs)

        client2.get = drop_req0_trace  # type: ignore[method-assign]
        runner2 = ExperimentRunner(
            runs_root=tmp_path / "runs-miss-trace",
            client=client2,
            trace_poll_timeout_s=0.05,
            trace_poll_interval_s=0.01,
        )
        res_miss = await runner2.run_open_loop(
            cfg_miss,
            _make_schedule([0.0, 10.0]),
            run_id=run_miss_id,
            workload_sha256=wsha_miss,
        )
        assert res_miss.succeeded is False
        assert res_miss.run_status == "failed"
        assert res_miss.failure_type == "missing_trace"
        assert (res_miss.run_dir / "requests.jsonl").is_file()
        assert (res_miss.run_dir / "traces.jsonl").is_file()


@pytest.mark.asyncio
async def test_open_loop_runner_controlled_cancellation_returns_workers_to_idle(tmp_path: Path):
    """Controlled cancel stops future sends, disconnects active stream, and drains workers to 0."""
    cfg, _, wsha = _make_v2_config(
        tmp_path,
        policy="round_robin",
        request_count=4,
        warmup_count=0,
        max_dispatch_lag_ms=500.0,
    )
    schedule = _make_schedule([0.0, 200.0, 400.0, 600.0])
    run_id = "20261125T000005Z-ctrl-cancel"
    cancel_event = asyncio.Event()

    async with mock_gateway("round_robin", "test-model") as (_, client, controls):
        controls.hold(f"{run_id}-req-00000")

        async def cancel_after_req0_starts() -> None:
            for _ in range(50):
                if controls.calls_for(f"{run_id}-req-00000") == 1:
                    cancel_event.set()
                    return
                await asyncio.sleep(0.01)

        canceller = asyncio.create_task(cancel_after_req0_starts())
        runner = ExperimentRunner(
            runs_root=tmp_path / "runs-cancel",
            client=client,
            trace_poll_timeout_s=1.0,
            trace_poll_interval_s=0.01,
        )
        res = await runner.run_open_loop(
            cfg,
            schedule,
            run_id=run_id,
            workload_sha256=wsha,
            cancel_event=cancel_event,
        )
        await canceller

        assert res.succeeded is False
        assert res.run_status == "failed"
        assert res.failure_type == "runner_cancelled"
        req_rows = sorted(
            (
                json.loads(line)
                for line in (res.run_dir / "requests.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ),
            key=lambda row: row["sequence"],
        )
        assert len(req_rows) == 4
        assert req_rows[0]["client_status"] == "cancelled"
        assert req_rows[0]["trace_observed"] is True
        assert all(r["dispatch_status"] == "not_dispatched" for r in req_rows[1:])

        workers = (await client.get("/control/v1/workers")).json()
        assert all(w["active_requests"] == 0 for w in workers)
        env_meta = json.loads((res.run_dir / "environment.json").read_text(encoding="utf-8"))
        assert all(w["active_requests"] == 0 for w in env_meta["final_workers"])


@pytest.mark.asyncio
async def test_v1_runner_retains_closed_concurrency_limited_semantics(tmp_path: Path):
    """Requirement 2.10: W7 v1 ExperimentRunner.run() stays concurrency-limited (not open-loop)."""
    workload_path = tmp_path / "w7_workload.jsonl"
    lines = [
        json.dumps({"messages": [{"role": "user", "content": f"v1 prompt {i}"}]})
        for i in range(4)
    ]
    workload_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cfg_v1 = ExperimentConfig.model_validate(
        {
            "schema_version": 1,
            "name": "w7-v1-regression",
            "model": {"id": "test-model"},
            "topology": {"worker_count": 2},
            "policy": {"name": "round_robin"},
            "workload": {"path": str(workload_path), "request_count": 2},
            "seed": 42,
            "warmup": {"request_count": 0},
            "generation": {"temperature": 0, "max_tokens": 16},
            "runner": {
                "concurrency": 1,
                "gateway_url": "http://mock-gateway",
                "request_timeout_s": 10,
            },
        }
    )
    entries = [
        WorkloadEntry(index=i, messages=({"role": "user", "content": f"v1 prompt {i}"},))
        for i in range(2)
    ]
    run_id = "20261125T000003Z-v1-closed"

    async with mock_gateway("round_robin", "test-model") as (_, client, controls):
        controls.hold(f"{run_id}-req-00000")
        runner = ExperimentRunner(runs_root=tmp_path / "runs-v1", client=client)
        task = asyncio.create_task(runner.run(cfg_v1, entries, run_id=run_id))

        # Wait until req-00000 is held; req-00001 MUST NOT dispatch yet (concurrency=1).
        for _ in range(20):
            await asyncio.sleep(0.01)
            if controls.calls_for(f"{run_id}-req-00000") == 1:
                break
        assert controls.calls_for(f"{run_id}-req-00000") == 1
        assert controls.calls_for(f"{run_id}-req-00001") == 0

        controls.finish(f"{run_id}-req-00000")
        result = await task
        assert result.succeeded is True
        assert controls.calls_for(f"{run_id}-req-00001") == 1
