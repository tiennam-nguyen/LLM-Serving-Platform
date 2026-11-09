"""Week 6 Person A unit tests: Registry load, RoundRobin, LeastLoaded & Policy Factory."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta, timezone

import pytest

from llm_serving_platform.common.config import GatewayConfig, normalize_routing_policy
from llm_serving_platform.control.registry import (
    DuplicateRequestReservationError,
    UnknownWorkerError,
    WorkerNotRoutableError,
    WorkerRecord,
    WorkerRegistry,
    WorkerRoutingCandidate,
    WorkerStatus,
)
from llm_serving_platform.routing.router import (
    LeastLoadedRouter,
    RoundRobinRouter,
    SingleWorkerRouter,
    create_routing_policy,
)
from llm_serving_platform.telemetry.trace import RequestTrace
from llm_serving_platform.workers.base import WorkerUnavailableError
from llm_serving_platform.workers.mock_adapter import MockInferenceWorker

T0 = datetime(2026, 11, 9, 10, 0, 0, tzinfo=timezone.utc)
T1 = T0 + timedelta(seconds=5)
T2 = T0 + timedelta(seconds=10)
T3 = T0 + timedelta(seconds=15)


# ==============================================================================
# Section 35: Local unit-test contract — WorkerRegistry Load & Reservations
# ==============================================================================
def test_registry_reserve_release_idempotency_and_timestamps():
    """Verify 0->1->2->1->0 transitions, idempotent release, and last_load_change_at updates."""
    registry = WorkerRegistry()
    worker_a = MockInferenceWorker(worker_id="worker-a")
    rec_init = registry.register(worker_a, "mock://a", "model-a")
    assert rec_init.active_requests == 0
    assert rec_init.last_load_change_at is None
    assert registry.active_request_count("worker-a") == 0

    registry.record_health("worker-a", True, checked_at=T0)

    # First reservation: 0 -> 1
    assert registry.reserve_request("worker-a", "req-1", changed_at=T1) == 1
    rec1 = registry.get_record("worker-a")
    assert rec1.active_requests == 1
    assert rec1.last_load_change_at == T1
    assert registry.active_request_count("worker-a") == 1

    # Second reservation: 1 -> 2
    assert registry.reserve_request("worker-a", "req-2", changed_at=T2) == 2
    rec2 = registry.get_record("worker-a")
    assert rec2.active_requests == 2
    assert rec2.last_load_change_at == T2
    assert registry.active_request_count("worker-a") == 2

    # First release of req-1: 2 -> 1
    assert registry.release_request("req-1", changed_at=T3) is True
    rec3 = registry.get_record("worker-a")
    assert rec3.active_requests == 1
    assert rec3.last_load_change_at == T3

    # Second release of same req-1: idempotent False, no change to counter or timestamp
    later = T3 + timedelta(seconds=30)
    assert registry.release_request("req-1", changed_at=later) is False
    rec3_again = registry.get_record("worker-a")
    assert rec3_again.active_requests == 1
    assert rec3_again.last_load_change_at == T3

    # Final release of req-2: 1 -> 0
    assert registry.release_request("req-2", changed_at=later) is True
    rec4 = registry.get_record("worker-a")
    assert rec4.active_requests == 0
    assert rec4.last_load_change_at == later
    assert registry.active_request_count("worker-a") == 0

    # Extra release when already at 0: never negative
    assert registry.release_request("req-2", changed_at=later) is False
    assert registry.release_request("never-existed", changed_at=later) is False
    assert registry.get_record("worker-a").active_requests == 0


def test_registry_reservation_preconditions_and_duplicate_request_id():
    """Verify unknown worker, unhealthy worker, and duplicate active request_id fail explicitly."""
    registry = WorkerRegistry()
    w_a = MockInferenceWorker(worker_id="worker-a")
    w_b = MockInferenceWorker(worker_id="worker-b")
    registry.register(w_a, "mock://a", "m")
    registry.register(w_b, "mock://b", "m")

    # Unknown worker reservation fails with UnknownWorkerError
    with pytest.raises(UnknownWorkerError):
        registry.reserve_request("worker-missing", "req-1")
    with pytest.raises(UnknownWorkerError):
        registry.active_request_count("worker-missing")

    # STARTING worker reservation fails with WorkerNotRoutableError
    with pytest.raises(WorkerNotRoutableError):
        registry.reserve_request("worker-a", "req-1")

    # Mark worker-a HEALTHY and worker-b UNHEALTHY
    registry.record_health("worker-a", True, checked_at=T0)
    registry.record_health("worker-b", False, checked_at=T0)

    # UNHEALTHY worker reservation fails with WorkerNotRoutableError
    with pytest.raises(WorkerNotRoutableError):
        registry.reserve_request("worker-b", "req-1")

    # Reserve req-1 on worker-a
    assert registry.reserve_request("worker-a", "req-1", changed_at=T1) == 1

    # Duplicate active request_id fails on same worker AND on another healthy worker
    registry.record_health("worker-b", True, checked_at=T1)
    with pytest.raises(DuplicateRequestReservationError):
        registry.reserve_request("worker-a", "req-1")
    with pytest.raises(DuplicateRequestReservationError):
        registry.reserve_request("worker-b", "req-1")

    # Load was not corrupted by rejected duplicate attempts
    assert registry.active_request_count("worker-a") == 1
    assert registry.active_request_count("worker-b") == 0

    # Once released, req-1 is no longer active and can be reserved again
    assert registry.release_request("req-1", changed_at=T2) is True
    assert registry.reserve_request("worker-b", "req-1", changed_at=T3) == 1
    assert registry.active_request_count("worker-b") == 1
    assert registry.release_request("req-1") is True


def test_health_transition_preserves_load_and_allows_release_when_unhealthy():
    """Verify health transitions do not reset active_requests and unhealthy workers can release."""
    registry = WorkerRegistry()
    w_a = MockInferenceWorker(worker_id="worker-a")
    w_b = MockInferenceWorker(worker_id="worker-b")
    registry.register(w_a, "mock://a", "m")
    registry.register(w_b, "mock://b", "m")
    registry.record_health("worker-a", True, checked_at=T0)
    registry.record_health("worker-b", True, checked_at=T0)

    registry.reserve_request("worker-a", "req-a1", changed_at=T1)
    registry.reserve_request("worker-a", "req-a2", changed_at=T1)
    registry.reserve_request("worker-b", "req-b1", changed_at=T1)

    # Transition worker-a to UNHEALTHY while it owns 2 active requests
    unhealthy_rec = registry.record_health("worker-a", False, checked_at=T2)
    assert unhealthy_rec.status == WorkerStatus.UNHEALTHY
    assert unhealthy_rec.active_requests == 2
    assert unhealthy_rec.last_load_change_at == T1

    # routing_candidates returns HEALTHY workers only, sorted by worker_id, with exact load snapshot
    candidates = registry.routing_candidates()
    assert len(candidates) == 1
    assert isinstance(candidates[0], WorkerRoutingCandidate)
    assert candidates[0].worker_id == "worker-b"
    assert candidates[0].active_requests == 1
    assert candidates[0].worker is w_b

    # Release succeeds even though worker-a is currently UNHEALTHY
    assert registry.release_request("req-a1", changed_at=T3) is True
    assert registry.get_record("worker-a").active_requests == 1
    assert registry.get_record("worker-a").status == WorkerStatus.UNHEALTHY
    assert registry.release_request("req-a2", changed_at=T3) is True
    assert registry.get_record("worker-a").active_requests == 0
    assert registry.release_request("req-b1", changed_at=T3) is True


def test_worker_record_invariants_and_serialization_safety():
    """Verify WorkerRecord validates its invariants and never exposes runtime objects."""
    with pytest.raises(ValueError):
        WorkerRecord(worker_id="a", endpoint="http://a", model="m", active_requests=-1)
    with pytest.raises(ValueError):
        WorkerRecord(
            worker_id="a",
            endpoint="http://a",
            model="m",
            active_requests=True,  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError):
        WorkerRecord(
            worker_id="a",
            endpoint="http://a",
            model="m",
            last_load_change_at=datetime(2026, 1, 1),
        )
    rec = WorkerRecord(worker_id="a", endpoint="http://a", model="m")
    assert "worker" not in asdict(rec)


# ==============================================================================
# Section 36: Local unit-test contract — RoundRobinRouter Week 6
# ==============================================================================
def test_round_robin_router_week6_reservations_and_trace_telemetry():
    """Verify RoundRobinRouter cycles A/B/A/B, commits reservations, and records load telemetry."""
    registry = WorkerRegistry()
    w_a = MockInferenceWorker(worker_id="worker-a")
    w_b = MockInferenceWorker(worker_id="worker-b")
    registry.register(w_a, "mock://a", "m")
    registry.register(w_b, "mock://b", "m")
    registry.record_health("worker-a", True, checked_at=T0)
    registry.record_health("worker-b", True, checked_at=T0)

    # Pre-seed worker-a with 3 active requests so loads are unequal (A=3, B=0).
    # Round Robin must still follow cyclic order regardless of load, while recording load snapshot.
    for idx in range(3):
        registry.reserve_request("worker-a", f"seed-{idx}", changed_at=T0)

    router = RoundRobinRouter(registry)

    t1 = RequestTrace(request_id="rr-1", model="m")
    t1.mark_arrival()
    chosen1 = router.select_worker(t1)
    assert chosen1 is w_a
    assert t1.worker_id == "worker-a"
    assert t1.routing_policy == "round_robin"
    assert t1.routing_candidate_loads == {"worker-a": 3, "worker-b": 0}
    assert t1.routing_selected_load_before == 3
    assert t1.routing_selected_load_after == 4
    assert t1.routing_reason == "round_robin"
    assert t1.routing_ms is not None and t1.routing_ms >= 0
    assert registry.active_request_count("worker-a") == 4

    t2 = RequestTrace(request_id="rr-2", model="m")
    t2.mark_arrival()
    chosen2 = router.select_worker(t2)
    assert chosen2 is w_b
    assert t2.worker_id == "worker-b"
    assert t2.routing_policy == "round_robin"
    assert t2.routing_candidate_loads == {"worker-a": 4, "worker-b": 0}
    assert t2.routing_selected_load_before == 0
    assert t2.routing_selected_load_after == 1
    assert t2.routing_reason == "round_robin"
    assert registry.active_request_count("worker-b") == 1

    # Duplicate active request_id fails and does NOT advance cursor
    dup_trace = RequestTrace(request_id="rr-2", model="m")
    with pytest.raises(DuplicateRequestReservationError):
        router.select_worker(dup_trace)

    # Next valid request still goes to worker-a (cursor was not advanced by failed reservation)
    t3 = RequestTrace(request_id="rr-3", model="m")
    assert router.select_worker(t3) is w_a


def test_round_robin_single_worker_and_cursor_preservation_on_empty_pool():
    """Verify 1-worker A/A/A behavior and cursor non-advancement when all workers are unhealthy."""
    registry = WorkerRegistry()
    w_a = MockInferenceWorker(worker_id="worker-a")
    registry.register(w_a, "mock://a", "m")
    registry.record_health("worker-a", True, checked_at=T0)

    router = RoundRobinRouter(registry)
    for i in range(3):
        tr = RequestTrace(request_id=f"single-rr-{i}", model="m")
        assert router.select_worker(tr) is w_a
        assert registry.release_request(f"single-rr-{i}") is True

    # Mark unhealthy -> raises 503 and does not advance cursor
    registry.record_health("worker-a", False, checked_at=T1)
    cursor_before = router._cursor
    with pytest.raises(WorkerUnavailableError) as exc_info:
        router.select_worker(RequestTrace(request_id="down-req", model="m"))
    assert exc_info.value.status_code == 503
    assert router._cursor == cursor_before


# ==============================================================================
# Section 37: Local unit-test contract — LeastLoadedRouter
# ==============================================================================
def test_least_loaded_selects_minimum_active_worker_and_filters_unhealthy():
    """Verify A=4,B=1 -> B; A=1,B=4 -> A; A unhealthy=0, B healthy=5 -> B; all unhealthy -> 503."""
    registry = WorkerRegistry()
    w_a = MockInferenceWorker(worker_id="worker-a")
    w_b = MockInferenceWorker(worker_id="worker-b")
    registry.register(w_a, "mock://a", "m")
    registry.register(w_b, "mock://b", "m")
    registry.record_health("worker-a", True, checked_at=T0)
    registry.record_health("worker-b", True, checked_at=T0)

    router = LeastLoadedRouter(registry)

    # Case 1: A=4, B=1 -> chooses B
    for i in range(4):
        registry.reserve_request("worker-a", f"a-load-{i}")
    registry.reserve_request("worker-b", "b-load-0")

    trace_b = RequestTrace(request_id="ll-choose-b", model="m")
    chosen_b = router.select_worker(trace_b)
    assert chosen_b is w_b
    assert trace_b.worker_id == "worker-b"
    assert trace_b.routing_policy == "least_loaded"
    assert trace_b.routing_candidate_loads == {"worker-a": 4, "worker-b": 1}
    assert trace_b.routing_selected_load_before == 1
    assert trace_b.routing_selected_load_after == 2
    assert trace_b.routing_reason == "min_active_requests"
    assert registry.active_request_count("worker-b") == 2

    # Clean up and set Case 2: A=1, B=4 -> chooses A
    for req_id in ("a-load-1", "a-load-2", "a-load-3"):
        registry.release_request(req_id)
    registry.reserve_request("worker-b", "b-load-2")
    registry.reserve_request("worker-b", "b-load-3")
    assert registry.active_request_count("worker-a") == 1
    assert registry.active_request_count("worker-b") == 4

    trace_a = RequestTrace(request_id="ll-choose-a", model="m")
    chosen_a = router.select_worker(trace_a)
    assert chosen_a is w_a
    assert trace_a.routing_candidate_loads == {"worker-a": 1, "worker-b": 4}
    assert trace_a.routing_selected_load_before == 1
    assert trace_a.routing_selected_load_after == 2
    assert trace_a.routing_reason == "min_active_requests"

    # Case 3: A UNHEALTHY with active=0, B HEALTHY with active=5 -> chooses B
    registry.release_request("a-load-0")
    registry.release_request("ll-choose-a")
    registry.record_health("worker-a", False, checked_at=T1)
    registry.reserve_request("worker-b", "b-load-4")
    assert registry.active_request_count("worker-a") == 0
    assert registry.active_request_count("worker-b") == 5

    trace_health_prio = RequestTrace(request_id="ll-health-prio", model="m")
    chosen_hp = router.select_worker(trace_health_prio)
    assert chosen_hp is w_b
    assert trace_health_prio.routing_candidate_loads == {"worker-b": 5}
    assert trace_health_prio.routing_selected_load_before == 5
    assert trace_health_prio.routing_selected_load_after == 6

    # Case 4: All UNHEALTHY -> 503 WorkerUnavailableError
    registry.record_health("worker-b", False, checked_at=T2)
    trace_503 = RequestTrace(request_id="ll-all-down", model="m")
    with pytest.raises(WorkerUnavailableError) as exc_info:
        router.select_worker(trace_503)
    assert exc_info.value.status_code == 503
    assert trace_503.worker_id == "none"


def test_least_loaded_equal_load_tie_break_is_deterministic_and_fair():
    """Verify equal A=0, B=0 tie-breaking alternates A, B, A, B and reservations shift next load."""
    registry = WorkerRegistry()
    w_a = MockInferenceWorker(worker_id="worker-a")
    w_b = MockInferenceWorker(worker_id="worker-b")
    registry.register(w_a, "mock://a", "m")
    registry.register(w_b, "mock://b", "m")
    registry.record_health("worker-a", True, checked_at=T0)
    registry.record_health("worker-b", True, checked_at=T0)

    router = LeastLoadedRouter(registry)

    # 1. Idle sequential requests (released after each): A=0, B=0 -> A, B, A, B
    idle_sequence: list[str] = []
    for i in range(4):
        req_id = f"idle-{i}"
        tr = RequestTrace(request_id=req_id, model="m")
        chosen = router.select_worker(tr)
        assert tr.routing_candidate_loads == {"worker-a": 0, "worker-b": 0}
        assert tr.routing_selected_load_before == 0
        assert tr.routing_selected_load_after == 1
        idle_sequence.append(chosen.worker_id)
        assert registry.release_request(req_id) is True

    assert idle_sequence == ["worker-a", "worker-b", "worker-a", "worker-b"]

    # 2. Overlapping requests (not released until all 4 are placed):
    # Req 0 sees 0/0 -> worker-a (now 1/0)
    # Req 1 sees 1/0 -> worker-b (now 1/1)
    # Req 2 sees 1/1 -> worker-a (now 2/1)
    # Req 3 sees 2/1 -> worker-b (now 2/2)
    active_sequence: list[str] = []
    observed_snapshots: list[dict[str, int]] = []
    for i in range(4):
        req_id = f"overlap-{i}"
        tr = RequestTrace(request_id=req_id, model="m")
        chosen = router.select_worker(tr)
        active_sequence.append(chosen.worker_id)
        observed_snapshots.append(dict(tr.routing_candidate_loads))

    assert active_sequence == ["worker-a", "worker-b", "worker-a", "worker-b"]
    assert observed_snapshots == [
        {"worker-a": 0, "worker-b": 0},
        {"worker-a": 1, "worker-b": 0},
        {"worker-a": 1, "worker-b": 1},
        {"worker-a": 2, "worker-b": 1},
    ]
    for i in range(4):
        assert registry.release_request(f"overlap-{i}") is True
    assert registry.active_request_count("worker-a") == 0
    assert registry.active_request_count("worker-b") == 0


def test_health_race_between_snapshot_and_reservation_refreshes_once():
    """Verify bounded 1-retry refresh when a worker becomes UNHEALTHY before reserve_request."""

    class RaceInjectingRegistry(WorkerRegistry):
        def __init__(self, fail_workers_on_reserve: list[str]) -> None:
            super().__init__()
            self._fail_workers = list(fail_workers_on_reserve)

        def reserve_request(
            self,
            worker_id: str,
            request_id: str,
            *,
            changed_at: datetime | None = None,
        ) -> int:
            if self._fail_workers and self._fail_workers[0] == worker_id:
                self._fail_workers.pop(0)
                self.record_health(worker_id, False, checked_at=T1)
            return super().reserve_request(worker_id, request_id, changed_at=changed_at)

    # Case A: worker-a turns UNHEALTHY before reserve_request -> refreshes once and picks worker-b
    reg1 = RaceInjectingRegistry(fail_workers_on_reserve=["worker-a"])
    w_a = MockInferenceWorker(worker_id="worker-a")
    w_b = MockInferenceWorker(worker_id="worker-b")
    reg1.register(w_a, "mock://a", "m")
    reg1.register(w_b, "mock://b", "m")
    reg1.record_health("worker-a", True, checked_at=T0)
    reg1.record_health("worker-b", True, checked_at=T0)

    ll_router = LeastLoadedRouter(reg1)
    tr1 = RequestTrace(request_id="race-recover-1", model="m")
    chosen = ll_router.select_worker(tr1)
    assert chosen is w_b
    assert tr1.worker_id == "worker-b"
    assert tr1.routing_candidate_loads == {"worker-b": 0}
    assert reg1.active_request_count("worker-b") == 1

    # Case B: Both workers become UNHEALTHY across initial attempt and single retry -> 503
    reg2 = RaceInjectingRegistry(fail_workers_on_reserve=["worker-a", "worker-b"])
    reg2.register(w_a, "mock://a", "m")
    reg2.register(w_b, "mock://b", "m")
    reg2.record_health("worker-a", True, checked_at=T0)
    reg2.record_health("worker-b", True, checked_at=T0)

    ll_router_fail = LeastLoadedRouter(reg2)
    tr2 = RequestTrace(request_id="race-503", model="m")
    with pytest.raises(WorkerUnavailableError) as exc_info:
        ll_router_fail.select_worker(tr2)
    assert exc_info.value.status_code == 503
    assert tr2.worker_id == "none"
    assert ll_router_fail._tie_cursor == 0


# ==============================================================================
# Section 27, 28 & 39: Policy Config Fail-Fast & Concurrent Routing Serialization
# ==============================================================================
def test_routing_policy_config_validation_and_factory_fail_fast(monkeypatch: pytest.MonkeyPatch):
    """Verify valid policies construct expected routers and unknown policies fail fast."""
    registry = WorkerRegistry()
    w0 = MockInferenceWorker(worker_id="worker-0")
    registry.register(w0, "mock://0", "m")

    cfg_rr = GatewayConfig(routing_policy="round_robin")
    assert isinstance(create_routing_policy(cfg_rr, registry, w0), RoundRobinRouter)

    cfg_ll = GatewayConfig(routing_policy="least_loaded")
    assert isinstance(create_routing_policy(cfg_ll, registry, w0), LeastLoadedRouter)

    cfg_single = GatewayConfig(routing_policy="single")
    assert isinstance(create_routing_policy(cfg_single, registry, w0), SingleWorkerRouter)

    cfg_legacy = GatewayConfig(routing_policy="single_worker")
    assert cfg_legacy.routing_policy == "single"
    assert isinstance(create_routing_policy(cfg_legacy, registry, w0), SingleWorkerRouter)

    with pytest.raises(ValueError):
        GatewayConfig(routing_policy="typo_policy")

    with pytest.raises(ValueError):
        normalize_routing_policy("random")

    monkeypatch.setenv("ROUTING_POLICY", "invalid_policy_name")
    with pytest.raises(ValueError):
        GatewayConfig()


def test_concurrent_least_loaded_selections_serialize_reservations():
    """Verify concurrent LeastLoadedRouter selections never observe stale pre-reservation loads."""
    registry = WorkerRegistry()
    w_a = MockInferenceWorker(worker_id="worker-a")
    w_b = MockInferenceWorker(worker_id="worker-b")
    registry.register(w_a, "mock://a", "m")
    registry.register(w_b, "mock://b", "m")
    registry.record_health("worker-a", True, checked_at=T0)
    registry.record_health("worker-b", True, checked_at=T0)

    router = LeastLoadedRouter(registry)
    total_requests = 20
    traces = [RequestTrace(request_id=f"conc-{i}", model="m") for i in range(total_requests)]

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(router.select_worker, tr) for tr in traces]
        results = [f.result() for f in futures]

    assert len(results) == total_requests
    # Because selections + reservations are serialized, exactly 10 requests land on each worker
    assert registry.active_request_count("worker-a") == 10
    assert registry.active_request_count("worker-b") == 10

    # Only the very first decision could have observed (0, 0)
    zero_zero_snapshots = [
        tr
        for tr in traces
        if tr.routing_candidate_loads == {"worker-a": 0, "worker-b": 0}
    ]
    assert len(zero_zero_snapshots) == 1

    # Oracle check (Section 51): every trace chose the minimum candidate load at its snapshot
    for tr in traces:
        assert tr.routing_selected_load_before == min(tr.routing_candidate_loads.values())
        assert tr.routing_selected_load_after == tr.routing_selected_load_before + 1

    # Release all reservations -> all counters return to 0
    for tr in traces:
        assert registry.release_request(tr.request_id) is True
    assert registry.active_request_count("worker-a") == 0
    assert registry.active_request_count("worker-b") == 0
