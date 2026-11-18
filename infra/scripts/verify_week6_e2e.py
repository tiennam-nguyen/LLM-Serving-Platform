"""Exercise Gateway routing and retain per-request Week 6 evidence."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import partial
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm_serving_platform.common.config import GatewayConfig  # noqa: E402

POLICIES = ("round_robin", "least_loaded")
DEFAULT_LEVELS = (2, 4, 8)
WORKERS = "/control/v1/workers"
CHAT = "/v1/chat/completions"
SHORT = "Reply with exactly: OK"
LONG = "Explain continuous batching in a numbered list of 50 distinct points. Continue the list."
PROMPTS = {"short": SHORT, "long": LONG, "moderate": "Explain continuous batching in 5 sentences."}


class Inconclusive(Exception):
    """The intended workload state was not observed."""


class CheckFailed(Exception):
    """A Gateway invariant was violated."""


def require(condition, message):
    if not condition:
        raise CheckFailed(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def loads(state):
    return {row["worker_id"]: row["active_requests"] for row in state}


def safe_endpoint(url):
    parsed = urlsplit(url)
    host = parsed.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    authority = host + (f":{parsed.port}" if parsed.port else "")
    return f"{parsed.scheme}://{authority}"


def chat_payload(model, prompt_class, max_tokens):
    return {
        "model": model,
        "messages": [{"role": "user", "content": PROMPTS[prompt_class]}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }


class Phase:
    def __init__(self, client, model, policy, app=None, controls=None, set_worker_health=None):
        self.client = client
        self.model = model
        self.policy = policy
        self.app = app
        self.controls = controls
        self.set_worker_health = set_worker_health
        self.cursor = 0

    async def workers(self):
        response = await self.client.get(WORKERS)
        response.raise_for_status()
        state = response.json()
        require(len(state) == 2, "Expected exactly two registered workers")
        require(
            all(type(n) is int and n >= 0 for n in loads(state).values()),
            "Invalid active request count",
        )
        self.ids = sorted(loads(state))
        self.healthy = sorted(row["worker_id"] for row in state if row["status"] == "HEALTHY")
        return state

    async def trace(self, request_id):
        response = await self.client.get(f"/v1/traces/{quote(request_id, safe='')}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()

    async def check_fresh(self):
        state = await self.workers()
        require(self.healthy == self.ids, "Fresh phase requires two HEALTHY workers")
        require(not any(loads(state).values()), "Fresh phase has active requests")
        response = await self.client.get("/v1/traces")
        response.raise_for_status()
        require(not response.json()["traces"], "Fresh phase has existing traces")


class Scenario:
    def __init__(self, phase, scenario_id, purpose):
        self.phase = phase
        self.id = scenario_id
        self.result = {
            "scenario_id": scenario_id,
            "phase": phase.policy,
            "policy": phase.policy,
            "status": "PASS",
            "reason": "Observed workload and verified its invariants",
            "purpose": purpose,
            "observations": {"worker_snapshots": []},
            "requests": [],
            "final_worker_state": [],
            "assertions": [],
            "limitations": [],
        }
        self.tasks = []

    def check(self, condition, message):
        self.result["assertions"].append({"assertion": message, "passed": bool(condition)})
        require(condition, message)

    def observe(self, name, value):
        self.result["observations"][name] = value

    @staticmethod
    def summarize(state):
        return [
            {key: row[key] for key in ("worker_id", "status", "active_requests")} for row in state
        ]

    async def snapshot(self, label, predicate=lambda _: True, seconds=2):
        deadline = asyncio.get_running_loop().time() + seconds
        while True:
            state = self.summarize(await self.phase.workers())
            if predicate(state):
                self.record_snapshot(label, state)
                return state
            if asyncio.get_running_loop().time() >= deadline:
                self.record_snapshot(f"{label}_unmet", state)
                return None
            await asyncio.sleep(0.01 if self.phase.controls else 0.05)

    def record_snapshot(self, label, state):
        self.result["observations"]["worker_snapshots"].append(
            {"label": label, "timestamp_utc": now(), "workers": state}
        )

    def start(
        self,
        kind,
        *,
        prompt_class="short",
        max_tokens=8,
        hold=False,
        request_id=None,
        expected="completed",
        model=None,
        error_code=None,
    ):
        row = self.new_request(kind, prompt_class, max_tokens, request_id)
        if hold and self.phase.controls:
            self.phase.controls.hold(row["request_id"])
        task = asyncio.create_task(self.send(row, expected, model, error_code))
        self.tasks.append(task)
        return row, task

    def new_request(self, kind, prompt_class, max_tokens, request_id=None):
        rows = self.result["requests"]
        row = {
            "scenario_id": self.id,
            "phase": self.phase.policy,
            "policy": self.phase.policy,
            "request_id": request_id or f"w6-{self.phase.policy}-{self.id}-{len(rows) + 1}",
            "request_kind": kind,
            "prompt_class": prompt_class,
            "max_tokens": max_tokens,
            "request_start_timestamp": None,
            "request_terminal_timestamp": None,
            "http_status": None,
            "stream_terminal_status": None,
            "worker_id": None,
            "routing_policy": None,
            "candidate_loads": {},
            "selected_load_before": None,
            "selected_load_after": None,
            "routing_reason": None,
            "routing_ms": None,
            "ttft_ms": None,
            "e2e_ms": None,
            "error": None,
            "cancelled": False,
            "content_chunks": 0,
            "done": False,
        }
        rows.append(row)
        return row

    async def send(self, row, expected, model, error_code):
        phase = self.phase
        request_id = row["request_id"]
        healthy_at_submit = list(phase.healthy)
        payload = chat_payload(model or phase.model, row["prompt_class"], row["max_tokens"])
        row["request_start_timestamp"] = now()
        try:
            async with phase.client.stream(
                "POST", CHAT, headers={"X-Request-ID": request_id}, json=payload
            ) as response:
                row["http_status"] = response.status_code
                row["worker_id"] = response.headers.get("X-Worker-ID")
                self.check(response.headers.get("X-Request-ID") == request_id, "Request ID echoed")
                if response.status_code == 200:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        self.read_event(row, line.removeprefix("data:").strip())
                else:
                    await response.aread()
                    row["error"] = {"http_status": response.status_code}
        except Exception as exc:
            if row["error"] is None:
                row["error"] = {"client_error": type(exc).__name__}
            raise
        finally:
            row["request_terminal_timestamp"] = now()

        if expected == "duplicate":
            row["stream_terminal_status"] = "duplicate_rejected"
            self.check(
                row["http_status"] == 409 and row["worker_id"] == "none", "Duplicate HTTP 409"
            )
            return row

        if expected == "model_rejection" and row["http_status"] in (400, 404, 422):
            if await phase.trace(request_id) is None:
                row["stream_terminal_status"] = "rejected_before_routing"
                raise Inconclusive("Invalid model was rejected before a routing decision")

        await self.attach_trace(row, healthy_at_submit)
        if expected == "unavailable":
            self.check(
                row["http_status"] == 503
                and row["worker_id"] == "none"
                and row["stream_terminal_status"] == "error"
                and not row["candidate_loads"]
                and row["selected_load_after"] is None,
                "All-down HTTP 503 with empty candidates and no reservation",
            )
        else:
            terminal = "error" if expected == "model_rejection" else expected
            self.check(
                row["http_status"] == 200
                and row["done"]
                and row["stream_terminal_status"] == terminal,
                "SSE and trace terminated as expected",
            )
            if expected == "completed":
                self.check(row["content_chunks"] > 0 and row["error"] is None, "Useful SSE content")
            else:
                self.check(row["error"] is not None, "Error delivered as SSE")
                if error_code is not None:
                    self.check(row["error"] == {"sse_code": error_code}, "Expected SSE error code")
                if expected == "model_rejection":
                    self.check(row["error"]["sse_code"] in (400, 404, 422), "Worker rejected model")
        return row

    @staticmethod
    def read_event(row, data):
        if data == "[DONE]":
            row["done"] = True
        elif data:
            event = json.loads(data)
            if "error" in event:
                row["error"] = {"sse_code": event["error"].get("code")}
            choices = event.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content"):
                row["content_chunks"] += 1

    async def attach_trace(self, row, healthy_at_submit):
        trace = None
        for _ in range(40):
            trace = await self.phase.trace(row["request_id"])
            if trace and trace["status"] != "in_flight":
                break
            await asyncio.sleep(0.025)
        self.check(trace is not None and trace["status"] != "in_flight", "Terminal trace exists")
        routing = trace["routing"]
        row.update(
            routing_policy=routing["policy"],
            candidate_loads=routing["candidate_loads"],
            selected_load_before=routing["selected_load_before"],
            selected_load_after=routing["selected_load_after"],
            routing_reason=routing["reason"],
            stream_terminal_status=trace["status"],
            milestones=trace["milestones"],
            trace_token_chunks=trace["token_chunks"],
            **{key: trace["metrics_ms"][key] for key in ("routing_ms", "ttft_ms", "e2e_ms")},
        )
        self.check(
            trace["request_id"] == row["request_id"]
            and trace["worker_id"] == row["worker_id"]
            and row["routing_policy"] == self.phase.policy,
            "HTTP trace correlates to request, worker and policy",
        )
        if row["worker_id"] == "none":
            if row["error"] is None:
                row["error"] = {"trace_status": trace["status"]}
            return
        worker = row["worker_id"]
        candidates = row["candidate_loads"]
        before = row["selected_load_before"]
        after = row["selected_load_after"]
        self.check(
            worker == routing["selected_worker_id"]
            and worker in candidates
            and sorted(candidates) == healthy_at_submit,
            "Decision candidates match the observed HEALTHY pool and include the selected worker",
        )
        self.check(
            all(type(value) is int and value >= 0 for value in candidates.values())
            and type(before) is int
            and type(after) is int
            and before == candidates[worker]
            and after == before + 1,
            "Trace shows a single reservation on a nonnegative load snapshot",
        )
        self.check(
            all(
                trace["milestones"][name]
                for name in (
                    "arrival",
                    "routing_start",
                    "routing_end",
                    "worker_start",
                    "last_token",
                )
            )
            and type(row["routing_ms"]) in (float, int)
            and row["routing_ms"] >= 0,
            "Lifecycle milestones and routing duration are present",
        )
        if self.phase.policy == "least_loaded":
            self.check(
                before == min(candidates.values())
                and row["routing_reason"] == "min_active_requests",
                "Least-Loaded selected a minimum-load candidate",
            )
        else:
            # RR scenarios submit decisions in a controlled order. Check the cycle separately
            # from stream completion order, which may differ when the first request is held.
            self.check(row["routing_reason"] == "round_robin", "Round Robin decision reason")
        if trace["error"] and row["error"] is None:
            row["error"] = {"trace_status": trace["status"]}
        row["cancelled"] = trace["status"] == "cancelled"

    async def run(self, workload):
        try:
            initial = await self.snapshot("initial")
            self.check(not any(loads(initial).values()), "Scenario began drained")
            await workload(self)
        except Inconclusive as exc:
            self.result.update(status="INCONCLUSIVE", reason=str(exc))
            self.result["limitations"].append(str(exc))
        except Exception as exc:
            reason = str(exc) if isinstance(exc, CheckFailed) else type(exc).__name__
            self.result.update(status="FAIL", reason=reason)
        finally:
            if self.phase.controls:
                self.phase.controls.release_all()
            try:
                async with asyncio.timeout(100):
                    outcomes = await asyncio.gather(*self.tasks, return_exceptions=True)
                for outcome in outcomes:
                    if isinstance(outcome, Exception) and not isinstance(outcome, Inconclusive):
                        reason = (
                            str(outcome)
                            if isinstance(outcome, CheckFailed)
                            else (type(outcome).__name__)
                        )
                        if self.result["status"] != "FAIL":
                            self.result.update(status="FAIL", reason=reason)
                drained = await self.snapshot(
                    "drained", lambda state: not any(loads(state).values())
                )
                final = (
                    drained if drained is not None else self.summarize(await self.phase.workers())
                )
                self.result["final_worker_state"] = final
                self.check(not any(loads(final).values()), "All Gateway counters drained to zero")
            except Exception as exc:
                reason = str(exc) if isinstance(exc, CheckFailed) else type(exc).__name__
                self.result.update(status="FAIL", reason=reason)
            if self.phase.policy == "round_robin":
                try:
                    self.audit_round_robin()
                except CheckFailed as exc:
                    self.result.update(status="FAIL", reason=str(exc))
        return self.result

    def audit_round_robin(self):
        for row in self.result["requests"]:
            candidates = sorted(row["candidate_loads"])
            if not candidates:
                continue
            self.check(
                row["worker_id"] == candidates[self.phase.cursor % len(candidates)],
                "Round Robin followed the cyclic candidate order",
            )
            self.phase.cursor += 1


async def sequential_tie(s):
    ids = s.phase.ids
    start = s.phase.cursor % len(ids)
    sequence = []
    for index in range(8):
        before = await s.snapshot(
            f"before_{index + 1}", lambda state: loads(state) == dict.fromkeys(ids, 0)
        )
        s.check(before is not None, "Both workers idle before request")
        _, task = s.start("sequential")
        row = await task
        s.check(row["candidate_loads"] == dict.fromkeys(ids, 0), "Both candidates tied at zero")
        sequence.append(row["worker_id"])
    s.observe("sequence", sequence)
    s.check(sequence == (ids[start:] + ids[:start]) * 4, "Policy alternated across eight ties")


async def burst(s, concurrency):
    for repeat in range(3):
        rows = [
            s.start(f"burst_{repeat + 1}", prompt_class="moderate", max_tokens=64)
            for _ in range(concurrency)
        ]
        await asyncio.gather(*(task for _, task in rows))
        s.check(
            await s.snapshot(
                f"drained_repeat_{repeat + 1}", lambda state: not any(loads(state).values())
            )
            is not None,
            "Burst repetition drained",
        )
        if not any(any(row["candidate_loads"].values()) for row, _ in rows):
            s.result["limitations"].append(f"No overlap observed in repetition {repeat + 1}")
    if s.result["limitations"]:
        raise Inconclusive("A burst repetition provided no overlapping routing decision")


async def mixed(s):
    schedule = (
        ("long", 192),
        ("short", 8),
        ("long", 192),
        ("short", 8),
        ("short", 8),
        ("long", 192),
    )
    launched = []
    for index, (kind, tokens) in enumerate(schedule):
        if index == len(schedule) - 1:
            await launched[1][1]
        launched.append(s.start(kind, prompt_class=kind, max_tokens=tokens))
        await asyncio.sleep(0.015 if s.phase.controls else 0.05)
    finished = await asyncio.gather(*(task for _, task in launched))
    ordered = sorted(finished, key=lambda row: row["request_terminal_timestamp"])
    s.observe("completion_order", [row["request_id"] for row in ordered])
    first_long, first_short, *_, final_long = finished
    if not (
        first_short["request_terminal_timestamp"] < first_long["request_terminal_timestamp"]
        and any(first_short["candidate_loads"].values())
        and any(final_long["candidate_loads"].values())
    ):
        raise Inconclusive(
            "Did not observe a short completion while overlapping long work remained"
        )


async def divergence(s, attempt):
    a, b = s.phase.ids
    required = {a: 1, b: 0}
    original, long_task = s.start(
        "long",
        prompt_class="long",
        max_tokens=(256, 512, 1024)[attempt - 1],
        hold=True,
    )
    active = await s.snapshot("long_routed", lambda state: sum(loads(state).values()) == 1)
    if active is None:
        raise Inconclusive("The first long request ended before its reservation was observed")
    s.check(loads(active) == required, "First fresh decision reserved worker A")
    _, short_task = s.start("short")
    short = await short_task
    if long_task.done() or short["candidate_loads"] != required:
        raise Inconclusive(
            "Long request ended before the short request could route at unequal load"
        )
    s.check(short["worker_id"] == b, "Second request was routed to B")
    if not await s.snapshot("after_short", lambda state: loads(state) == required):
        raise Inconclusive("Could not observe A=1, B=0 after the short stream drained")
    _, probe_task = s.start("probe")
    probe = await probe_task
    s.observe("probe_request_id", probe["request_id"])
    if probe["candidate_loads"] != required:
        raise Inconclusive("The probe trace did not capture the controlled A=1, B=0 snapshot")
    s.observe("verified_unequal_snapshot", required)
    expected = a if s.phase.policy == "round_robin" else b
    s.check(probe["worker_id"] == expected, "Policy chose the expected worker at A=1, B=0")
    if s.phase.controls:
        s.phase.controls.finish(original["request_id"])
    await long_task
    s.check(original["worker_id"] == a, "First request was routed to A")


async def completion_rebalance(s):
    first, slow = s.start("overlap_long", prompt_class="long", max_tokens=512, hold=True)
    if not await s.snapshot("first_active", lambda state: sorted(loads(state).values()) == [0, 1]):
        raise Inconclusive("First stream ended before its reservation was observed")
    second, other = s.start("overlap_shorter", prompt_class="long", max_tokens=64, hold=True)
    if not await s.snapshot("both_active", lambda state: sorted(loads(state).values()) == [1, 1]):
        raise Inconclusive("Could not observe both workers active")
    if s.phase.controls:
        # Complete the first worker here, the opposite idle worker to the divergence probe.
        s.phase.controls.finish(first["request_id"])
    done, _ = await asyncio.wait((slow, other), return_when=asyncio.FIRST_COMPLETED)
    finished = await next(iter(done))
    remaining = await s.snapshot(
        "one_completed", lambda state: sorted(loads(state).values()) == [0, 1]
    )
    if not remaining:
        raise Inconclusive("Both requests drained before an idle-worker probe could be staged")
    active = loads(remaining)
    idle = next(worker_id for worker_id, count in active.items() if count == 0)
    s.check(finished["worker_id"] == idle, "Completed request released the idle worker")
    _, probe_task = s.start("probe")
    probe = await probe_task
    if probe["candidate_loads"] != active:
        raise Inconclusive("The remaining request ended before the probe decision")
    s.check(probe["worker_id"] == idle, "Least-Loaded selected the newly idle worker")
    s.observe("rebalanced_to", idle)
    s.observe("completed_before_probe", finished["request_id"])
    if s.phase.controls:
        s.phase.controls.finish(second["request_id"])
    await asyncio.gather(slow, other)


async def worker_failure(s, failure, code, expected_chunks):
    controls = s.phase.controls
    row, task = s.start(failure, hold=True, expected="error", error_code=code)
    request_id = row["request_id"]
    controls.failures[request_id] = failure
    reserved = await s.snapshot("reserved", lambda state: sum(loads(state).values()) == 1)
    s.check(reserved is not None, "Worker failure was triggered after reservation")
    controls.finish(request_id)
    await task
    s.check(row["content_chunks"] == expected_chunks, "Expected content before failure")
    s.check(controls.calls_for(request_id) == 1, "Exactly one worker invocation")


async def cancel(s):
    from tests.week6_workers import disconnect_chat

    row = s.new_request("client_disconnect", "long", 128)
    request_id = row["request_id"]
    s.phase.controls.hold(request_id)
    disconnect = asyncio.Event()

    async def on_message(message):
        if message["type"] == "http.response.start":
            headers = dict(message["headers"])
            row["http_status"] = message["status"]
            row["worker_id"] = headers[b"x-worker-id"].decode()
            s.check(headers[b"x-request-id"].decode() == request_id, "Request ID echoed")

    healthy_at_submit = list(s.phase.healthy)
    row["request_start_timestamp"] = now()
    task = asyncio.create_task(
        disconnect_chat(
            s.phase.app,
            request_id,
            chat_payload(s.phase.model, "long", 128),
            disconnect,
            on_message,
        )
    )
    s.tasks.append(task)
    try:
        reserved = await s.snapshot("reserved", lambda state: sum(loads(state).values()) == 1)
        s.check(reserved is not None, "Client disconnected while reservation was active")
    finally:
        disconnect.set()
    await asyncio.wait_for(task, timeout=3)
    row["request_terminal_timestamp"] = now()
    await s.attach_trace(row, healthy_at_submit)
    s.check(row["cancelled"], "ASGI disconnect ended the trace as cancelled")
    s.check(s.phase.controls.closed[request_id] == 1, "Worker stream was closed")


async def cancel_live(s):
    """Close a real HTTP stream after useful content while its reservation is active."""
    phase = s.phase
    row = s.new_request("client_disconnect", "long", 1024)
    request_id = row["request_id"]
    healthy = list(phase.healthy)
    row["request_start_timestamp"] = now()
    active = None
    try:
        async with phase.client.stream(
            "POST",
            CHAT,
            headers={"X-Request-ID": request_id},
            json=chat_payload(phase.model, "long", row["max_tokens"]),
        ) as response:
            row["http_status"] = response.status_code
            row["worker_id"] = response.headers.get("X-Worker-ID")
            s.check(response.status_code == 200, "Live cancellation opened an SSE response")
            s.check(response.headers.get("X-Request-ID") == request_id, "Request ID echoed")
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    s.read_event(row, line.removeprefix("data:").strip())
                if row["content_chunks"]:
                    active = await s.snapshot("before_http_close")
                    break
    finally:
        row["request_terminal_timestamp"] = now()

    if row["error"]:
        await s.attach_trace(row, healthy)
        raise CheckFailed("Worker error interrupted the live cancellation probe")
    if active is None or loads(active).get(row["worker_id"]) != 1:
        await s.attach_trace(row, healthy)
        raise Inconclusive("The live request drained before an active disconnect could be staged")

    drained = await s.snapshot(
        "after_http_close", lambda state: not any(loads(state).values()), seconds=12
    )
    s.check(drained is not None, "Live cancellation drained Gateway load within 12 seconds")
    await s.attach_trace(row, healthy)
    if row["stream_terminal_status"] == "completed":
        raise Inconclusive("The Gateway completed the stream before observing the HTTP close")
    s.check(row["cancelled"], "HTTP disconnect ended the live Gateway trace as cancelled")


async def duplicate(s):
    request_id = "duplicate-active-1"
    original, task = s.start("original", hold=True, request_id=request_id)
    active = await s.snapshot("original_active", lambda state: sum(loads(state).values()) == 1)
    s.check(active is not None, "Exactly one original reservation")
    before = await s.phase.trace(request_id)
    s.check(before is not None and before["status"] == "in_flight", "Original trace is active")
    s.observe("trace_before_duplicate", before)
    _, dup = s.start("duplicate", request_id=request_id, expected="duplicate")
    await dup
    after = await s.phase.trace(request_id)
    s.observe("trace_after_duplicate", after)
    s.check(after == before, "Duplicate preserved the entire original in-flight trace")
    after_rejection = await s.snapshot("after_duplicate")
    s.check(loads(after_rejection) == loads(active), "Duplicate preserved exact active counts")
    s.phase.controls.finish(request_id)
    await task
    s.check(s.phase.controls.calls_for(request_id) == 1, "Original alone reached a worker")
    s.check(original["stream_terminal_status"] == "completed", "Original completed normally")


async def health_change(s, healthy, *, all_workers=False):
    phase = s.phase
    targets = phase.ids if all_workers else phase.ids[:1]
    for worker_id in targets:
        if phase.controls:
            next(
                worker for worker in phase.controls.workers if worker.worker_id == worker_id
            ).healthy = healthy
        else:
            await phase.set_worker_health(worker_id, healthy)
    if phase.controls:
        await phase.app.state.health_monitor.check_once()

    target_status = "HEALTHY" if healthy else "UNHEALTHY"
    expected = {
        worker_id: target_status if worker_id in targets else "HEALTHY" for worker_id in phase.ids
    }
    result = await s.snapshot(
        "recovered" if healthy else "excluded",
        lambda state: {row["worker_id"]: row["status"] for row in state} == expected,
        seconds=35 if not phase.controls else 2,
    )
    s.check(result is not None, f"Observed expected health statuses {expected}")


async def health_recovery(s):
    a, b = s.phase.ids
    await health_change(s, False)
    try:
        for _ in range(3):
            _, task = s.start("failover")
            row = await task
            s.check(
                row["worker_id"] == b and set(row["candidate_loads"]) == {b},
                "Unhealthy worker excluded",
            )
    finally:
        await health_change(s, True)
    rows = []
    for _ in range(4):
        _, task = s.start("recovered")
        rows.append(await task)
    s.check(any(row["worker_id"] == a for row in rows), "Recovered worker served traffic")
    s.check(
        all(set(row["candidate_loads"]) == {a, b} for row in rows),
        "Recovered worker returned to the candidate set",
    )


async def all_down(s):
    _, warm = s.start("cursor_baseline")
    await warm
    await health_change(s, False, all_workers=True)
    try:
        _, task = s.start("all_down", expected="unavailable")
        await task
        state = await s.snapshot("after_503")
        s.check(not any(loads(state).values()), "503 did not reserve a worker")
    finally:
        await health_change(s, True, all_workers=True)
    _, task = s.start("cursor_after_503")
    await task


async def invalid_model(s):
    _, task = s.start(
        "invalid_model", model="week6-intentionally-unknown-model", expected="model_rejection"
    )
    await task


async def run_phase(phase, results, before_workloads, attempt, levels, *, fresh_ties=True):
    policy = phase.policy

    async def run(name, purpose, workload):
        result = await Scenario(phase, name, purpose).run(workload)
        results.append(result)
        print(f"  {policy} {name}: {result['status']} ({result['reason']})", flush=True)
        return result

    if before_workloads:
        return await run(
            f"controlled_divergence_{attempt}",
            "Probe placement at observed A=1, B=0",
            lambda s: divergence(s, attempt),
        )
    if policy == "round_robin":
        await run("rr_cycle", "Check eight cyclic Round Robin decisions", sequential_tie)
        if phase.controls:
            await run(
                "all_down", "Reject all-down traffic without consuming the RR cursor", all_down
            )
        return

    if fresh_ties:
        await run(
            "equal_load_tie", "Check eight strictly sequential equal-load ties", sequential_tie
        )
    for level in levels:
        await run(
            f"concurrency_{level}",
            "Observe overlapping homogeneous requests",
            lambda s, level=level: burst(s, level),
        )
    await run("mixed_duration", "Observe short completions amid ongoing long requests", mixed)
    await run(
        "completion_rebalance",
        "Route to the worker whose earlier stream drained",
        completion_rebalance,
    )
    if phase.controls:
        for name, failure, code, chunks in (
            ("pre_stream_error", "pre", 503, 0),
            ("mid_stream_error", "mid", 503, 1),
            ("timeout", "timeout", 504, 0),
            ("unexpected_exception", "unexpected", 500, 0),
        ):
            await run(
                name,
                "Release once on the selected worker after a terminal stream error",
                lambda s: worker_failure(s, failure, code, chunks),
            )
        await run("cancellation", "Disconnect an active ASGI client and verify release", cancel)
        await run("duplicate_active", "Reject a duplicate without changing the original", duplicate)
    else:
        await run(
            "invalid_model", "Verify release on request-scoped vLLM model rejection", invalid_model
        )
        await run("cancellation", "Close a live HTTP stream while reserved", cancel_live)
    if phase.controls or phase.set_worker_health:
        await run(
            "health_recovery",
            "Exclude an unhealthy worker and restore its eligibility",
            health_recovery,
        )


@asynccontextmanager
async def mock_phase(policy, model):
    from tests.week6_workers import mock_gateway

    async with mock_gateway(policy, model) as (app, client, controls):
        yield Phase(client, model, policy, app=app, controls=controls)


@asynccontextmanager
async def live_phase(policy, model, gateway_url, set_worker_health=None):
    async with httpx.AsyncClient(base_url=gateway_url, timeout=80, trust_env=False) as client:
        yield Phase(client, model, policy, set_worker_health=set_worker_health)


def summarize(results, no_gpu, policies, topology_verified, timestamp_status):
    verified = (
        "verified_no_gpu"
        if no_gpu
        else "verified_live_gpu"
        if topology_verified
        else "verified_attached_gateway"
    )
    by_name = {result["scenario_id"]: result["status"] for result in results}

    def status_from(states):
        states = list(states)
        if "FAIL" in states:
            return "failed"
        if "INCONCLUSIVE" in states:
            return "inconclusive"
        if states and all(state == "PASS" for state in states):
            return verified
        return "not_run"

    def status_for(*names):
        return status_from(by_name.get(name) for name in names)

    controlled = [
        result for result in results if result["scenario_id"].startswith("controlled_divergence_")
    ]
    last_attempts = {result["policy"]: result for result in controlled}
    pair = [last_attempts.get(policy) for policy in POLICIES]
    both = all(result and result["status"] == "PASS" for result in pair)
    comparisons = []
    if both:
        for phase in pair:
            obs = phase["observations"]
            probe = next(
                row for row in phase["requests"] if row["request_id"] == obs["probe_request_id"]
            )
            comparisons.append(
                {
                    "request_id": probe["request_id"],
                    "policy": phase["policy"],
                    "candidate_loads": probe["candidate_loads"],
                    "worker_id": probe["worker_id"],
                }
            )
    unequal = both and comparisons[0]["candidate_loads"] == comparisons[1]["candidate_loads"]
    diverged = unequal and comparisons[0]["worker_id"] != comparisons[1]["worker_id"]
    pair_status = status_from(result["status"] if result else None for result in pair)
    stable = [
        result
        for result in results
        if result not in controlled
        and not (
            not no_gpu
            and result["scenario_id"] == "cancellation"
            and result["status"] == "INCONCLUSIVE"
        )
    ]
    phase_status = status_from(
        [result["status"] for result in stable]
        + [
            last_attempts[policy]["status"] if policy in last_attempts else None
            for policy in policies
        ]
    )
    failure_names = ("pre_stream_error", "mid_stream_error", "timeout", "unexpected_exception")
    clean_drains = bool(results) and all(
        result["final_worker_state"] and not any(loads(result["final_worker_state"]).values())
        for result in results
    )
    ll_routes = [result for result in results if result["policy"] == "least_loaded"]

    # INCONCLUSIVE staging does not invalidate successfully audited routing decisions.
    def decision_status(rows, assertion):
        checks = [
            check["passed"]
            for result in rows
            for check in result["assertions"]
            if check["assertion"] == assertion
        ]
        return verified if checks and all(checks) else "failed" if checks else "not_run"

    return {
        "passed": phase_status == verified and (len(policies) == 1 or diverged),
        "test_mode": (
            "no_gpu" if no_gpu else "live_gpu" if topology_verified else "attached_gateway"
        ),
        "policies": list(policies),
        "controlled_probes": comparisons,
        "gates": {
            "person_a_timestamp_fix_verified": timestamp_status,
            "duplicate_trace_preservation_verified": status_for("duplicate_active"),
            "trace_schema_verified": decision_status(
                results, "Lifecycle milestones and routing duration are present"
            ),
            "reservation_cleanup_verified": status_for(
                *failure_names, "cancellation", "duplicate_active", "completion_rebalance"
            )
            if no_gpu
            else status_for("completion_rebalance", "invalid_model"),
            "duplicate_409_verified": status_for("duplicate_active"),
            "rr_cycle_verified": decision_status(
                results, "Round Robin followed the cyclic candidate order"
            ),
            "least_loaded_per_decision_oracle_verified": decision_status(
                ll_routes, "Least-Loaded selected a minimum-load candidate"
            ),
            "equal_load_tie_fairness_verified": status_for("equal_load_tie"),
            "concurrency_verified": status_for("concurrency_2", "concurrency_4", "concurrency_8"),
            "mixed_duration_verified": status_for("mixed_duration"),
            "completion_driven_rebalance_verified": status_for("completion_rebalance"),
            "controlled_unequal_snapshot_observed": verified
            if unequal
            else ("failed" if both else pair_status),
            "rr_vs_ll_policy_divergence_observed": verified
            if diverged
            else ("failed" if both else pair_status),
            "error_release_verified": status_for(*failure_names)
            if no_gpu
            else (status_for("invalid_model")),
            "timeout_release_verified": status_for("timeout"),
            "cancellation_release_verified": status_for("cancellation"),
            "health_exclusion_verified": status_for("health_recovery"),
            "all_down_verified": status_for("all_down"),
            "final_active_counts_zero": verified if clean_drains else "failed",
            "live_dual_t4_verified": (
                verified
                if topology_verified and diverged and phase_status == verified
                else "failed"
                if topology_verified and phase_status == "failed"
                else "inconclusive"
                if topology_verified
                else "not_run"
            ),
        },
    }


def verify_worker_timestamps():
    from llm_serving_platform.control.registry import WorkerRecord

    local = datetime(2026, 11, 10, 7, tzinfo=timezone(timedelta(hours=7)))
    utc = datetime(2026, 11, 10, tzinfo=timezone.utc)
    for field in ("last_checked_at", "last_load_change_at"):
        for supplied, expected in ((None, None), (utc, utc), (local, utc)):
            actual = getattr(WorkerRecord("a", "mock://a", "m", **{field: supplied}), field)
            require(actual == expected, f"Wrong UTC instant for {field}")
            if actual is not None:
                require(actual.tzinfo is timezone.utc, f"{field} retained a non-UTC offset")
        try:
            WorkerRecord("a", "mock://a", "m", **{field: utc.replace(tzinfo=None)})
        except ValueError:
            continue
        raise CheckFailed(f"{field} accepted a naive timestamp")


def command_output(args):
    try:
        return subprocess.check_output(
            args, cwd=ROOT, text=True, stderr=subprocess.DEVNULL, timeout=10
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def package_version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment(model, mode, endpoints, gateway_url, policies, hardware):
    patch = command_output(
        ["git", "diff", "--binary", "HEAD", "--", ".", ":(exclude)experiments/raw/week6"]
    )
    gpu_info = command_output(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    return {
        "timestamp_utc": now(),
        "git_revision": command_output(["git", "rev-parse", "HEAD"]),
        "branch": command_output(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "tracked_source_patch_sha256": hashlib.sha256(patch.encode()).hexdigest()
        if patch
        else None,
        "python": platform.python_version(),
        "pytorch": package_version("torch"),
        "vllm": package_version("vllm"),
        "cuda": None,
        "fastapi": package_version("fastapi"),
        "httpx": package_version("httpx"),
        "starlette": package_version("starlette"),
        "gpu_inventory": gpu_info.splitlines() if gpu_info else [],
        "model": model,
        "worker_endpoints": endpoints,
        "gateway_endpoint": safe_endpoint(gateway_url),
        "routing_policy": list(policies),
        "test_mode": mode,
        **(hardware or {}),
    }


def save_evidence(path, metadata, results, validation):
    path.mkdir(parents=True, exist_ok=True)
    files = {
        "environment": metadata,
        "scenario_matrix": results,
        "validation": validation,
        **{policy: [row for row in results if row["policy"] == policy] for policy in POLICIES},
    }
    # Store the matrix once. Policy files are indexes and keep exact probe evidence handy.
    for policy in POLICIES:
        files[policy] = {
            "scenario_ids": [row["scenario_id"] for row in files[policy]],
            "controlled_probes": [
                request
                for row in files[policy]
                for request in row["requests"]
                if request["request_kind"] == "probe"
                and row["scenario_id"].startswith("controlled_divergence_")
            ],
        }
    for name, value in files.items():
        dest = path / f"{name}.json"
        pending = dest.with_suffix(".json.tmp")
        pending.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")
        pending.replace(dest)


async def run_week6_e2e(
    *,
    no_gpu,
    model,
    gateway_url,
    policies=POLICIES,
    levels=DEFAULT_LEVELS,
    output_dir=None,
    phase_factory=None,
    hardware=None,
):
    results = []
    endpoints = {}
    managed = no_gpu or phase_factory is not None
    if no_gpu:
        gateway_url = "http://mock-gateway"
        phase_factory = partial(mock_phase, model=model)
    elif phase_factory is None:
        require(len(policies) == 1, "Use the Kaggle launcher to run both fresh live policies")
        phase_factory = partial(live_phase, model=model, gateway_url=gateway_url)
    problem = None
    timestamp_status = "not_run"
    try:
        if no_gpu:
            timestamp_status = "failed"
            verify_worker_timestamps()
            timestamp_status = "verified_no_gpu"
        attempts = 3 if managed and not no_gpu else 1
        for attempt in range(1, attempts + 1):
            decisions = {}
            for policy in policies:
                async with phase_factory(policy) as phase:
                    await phase.check_fresh()
                    endpoints.update(
                        {
                            row["worker_id"]: safe_endpoint(row["endpoint"])
                            for row in await phase.workers()
                        }
                    )
                    decisions[policy] = await run_phase(phase, results, True, attempt, levels)
            statuses = {decision["status"] for decision in decisions.values()}
            if "FAIL" in statuses or "INCONCLUSIVE" not in statuses:
                break

        for policy in policies:
            async with phase_factory(policy) as phase:
                if managed:
                    await phase.check_fresh()
                else:
                    # The attached process already served the controlled attempt.
                    await phase.workers()
                    if policy == "round_robin":
                        phase.cursor = sum(
                            bool(row["candidate_loads"]) for row in decisions[policy]["requests"]
                        )
                await run_phase(phase, results, False, attempt, levels, fresh_ties=managed)
    except Exception as exc:
        problem = str(exc) if isinstance(exc, CheckFailed) else type(exc).__name__

    topology_verified = bool(hardware and hardware.get("dual_t4_topology_verified"))
    validation = summarize(results, no_gpu, policies, topology_verified, timestamp_status)
    if problem:
        validation.update(passed=False, orchestration_error=problem)
    if output_dir is not None:
        metadata = environment(
            model, validation["test_mode"], endpoints, gateway_url, policies, hardware
        )
        save_evidence(output_dir, metadata, results, validation)
    return validation, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-gpu", action="store_true")
    parser.add_argument("--gateway-url", default="http://127.0.0.1:8080")
    parser.add_argument("--model", default=GatewayConfig().model_name)
    parser.add_argument("--save-evidence", action="store_true")
    parser.add_argument("--policy", choices=POLICIES)
    parser.add_argument("--concurrency-levels", nargs="+", type=int, default=DEFAULT_LEVELS)
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if not args.no_gpu and not args.policy:
        parser.error("Supply --policy for a fresh attached Gateway, or use the dual-T4 launcher")
    if any(level < 2 or level > 8 for level in args.concurrency_levels):
        parser.error("Concurrency levels must be between 2 and 8. --extended adds level 16.")
    levels = tuple(dict.fromkeys(args.concurrency_levels))
    if args.extended:
        levels += (16,)
    default_dir = ROOT / "experiments" / "raw" / "week6"
    if not args.no_gpu:
        default_dir /= "live"
    validation, _ = asyncio.run(
        run_week6_e2e(
            no_gpu=args.no_gpu,
            model=args.model,
            gateway_url=args.gateway_url,
            policies=(args.policy,) if args.policy else POLICIES,
            levels=levels,
            output_dir=(args.output_dir or default_dir) if args.save_evidence else None,
        )
    )
    print(json.dumps(validation, indent=2))
    raise SystemExit(0 if validation["passed"] else 1)


if __name__ == "__main__":
    main()
