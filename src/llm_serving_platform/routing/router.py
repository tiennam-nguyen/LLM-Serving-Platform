"""Routing abstractions, atomic load reservation, and policy factory for Week 6."""

from __future__ import annotations

from abc import ABC, abstractmethod
from threading import Lock

from llm_serving_platform.common.config import GatewayConfig, normalize_routing_policy
from llm_serving_platform.control.registry import (
    WorkerNotRoutableError,
    WorkerRegistry,
    WorkerRoutingCandidate,
)
from llm_serving_platform.telemetry.trace import RequestTrace
from llm_serving_platform.workers.base import InferenceWorker, WorkerUnavailableError

_MAX_HEALTH_RACE_ATTEMPTS = 2


def _record_routing_decision(
    trace: RequestTrace,
    *,
    policy: str,
    candidates: tuple[WorkerRoutingCandidate, ...],
    selected_load_before: int | None,
    selected_load_after: int | None,
    reason: str | None,
) -> None:
    """Attach Week 6 routing telemetry attributes to RequestTrace."""
    candidate_loads = {c.worker_id: c.active_requests for c in candidates}
    trace.routing_policy = policy
    trace.routing_candidate_loads = candidate_loads
    trace.routing_selected_load_before = selected_load_before
    trace.routing_selected_load_after = selected_load_after
    trace.routing_reason = reason


class RoutingPolicy(ABC):
    """Abstract interface for worker selection strategies (Open/Closed & Dependency Inversion)."""

    @abstractmethod
    def select_worker(self, trace: RequestTrace) -> InferenceWorker:
        """Select a target inference worker, commit reservation, and record trace milestones."""
        ...


class SingleWorkerRouter(RoutingPolicy):
    """Week 3 baseline router that forwards requests to a single configured worker."""

    def __init__(self, worker: InferenceWorker) -> None:
        self._worker = worker

    @property
    def worker(self) -> InferenceWorker:
        return self._worker

    def select_worker(self, trace: RequestTrace) -> InferenceWorker:
        """Select the configured worker and record routing_start and routing_end."""
        trace.mark_routing_start()
        chosen = self._worker
        worker_id = getattr(chosen, "worker_id", "worker-vllm-0")
        _record_routing_decision(
            trace,
            policy="single",
            candidates=(),
            selected_load_before=None,
            selected_load_after=None,
            reason="single_worker",
        )
        trace.mark_routing_end(worker_id=worker_id)
        return chosen


class RoundRobinRouter(RoutingPolicy):
    """Registry-backed routing policy that cycles sequentially across HEALTHY workers."""

    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry
        self._lock = Lock()
        self._cursor = 0

    @property
    def registry(self) -> WorkerRegistry:
        return self._registry

    def select_worker(self, trace: RequestTrace) -> InferenceWorker:
        """Atomically snapshot HEALTHY workers, select in cyclic order, and reserve load."""
        with self._lock:
            trace.mark_routing_start()
            for attempt in range(_MAX_HEALTH_RACE_ATTEMPTS):
                candidates = self._registry.routing_candidates()
                if not candidates:
                    _record_routing_decision(
                        trace,
                        policy="round_robin",
                        candidates=(),
                        selected_load_before=None,
                        selected_load_after=None,
                        reason=None,
                    )
                    trace.mark_routing_end(worker_id="none")
                    raise WorkerUnavailableError(
                        "No healthy workers available in WorkerRegistry.",
                        status_code=503,
                    )

                chosen = candidates[self._cursor % len(candidates)]
                try:
                    load_after = self._registry.reserve_request(
                        chosen.worker_id,
                        trace.request_id,
                    )
                except WorkerNotRoutableError:
                    if attempt + 1 < _MAX_HEALTH_RACE_ATTEMPTS:
                        continue
                    _record_routing_decision(
                        trace,
                        policy="round_robin",
                        candidates=(),
                        selected_load_before=None,
                        selected_load_after=None,
                        reason=None,
                    )
                    trace.mark_routing_end(worker_id="none")
                    raise WorkerUnavailableError(
                        "No healthy workers available in WorkerRegistry.",
                        status_code=503,
                    ) from None
                except Exception:
                    trace.mark_routing_end(worker_id="none")
                    raise

                self._cursor += 1
                _record_routing_decision(
                    trace,
                    policy="round_robin",
                    candidates=candidates,
                    selected_load_before=chosen.active_requests,
                    selected_load_after=load_after,
                    reason="round_robin",
                )
                trace.mark_routing_end(worker_id=chosen.worker_id)
                return chosen.worker

            trace.mark_routing_end(worker_id="none")
            raise WorkerUnavailableError(
                "No healthy workers available in WorkerRegistry.",
                status_code=503,
            )


class LeastLoadedRouter(RoutingPolicy):
    """Week 6 routing policy selecting a HEALTHY worker with minimum active_requests."""

    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry
        self._lock = Lock()
        self._tie_cursor = 0

    @property
    def registry(self) -> WorkerRegistry:
        return self._registry

    def select_worker(self, trace: RequestTrace) -> InferenceWorker:
        """Atomically snapshot HEALTHY workers, select minimum active load, and reserve load."""
        with self._lock:
            trace.mark_routing_start()
            for attempt in range(_MAX_HEALTH_RACE_ATTEMPTS):
                candidates = self._registry.routing_candidates()
                if not candidates:
                    _record_routing_decision(
                        trace,
                        policy="least_loaded",
                        candidates=(),
                        selected_load_before=None,
                        selected_load_after=None,
                        reason=None,
                    )
                    trace.mark_routing_end(worker_id="none")
                    raise WorkerUnavailableError(
                        "No healthy workers available in WorkerRegistry.",
                        status_code=503,
                    )

                minimum = min(candidate.active_requests for candidate in candidates)
                ties = [
                    candidate
                    for candidate in candidates
                    if candidate.active_requests == minimum
                ]
                chosen = ties[self._tie_cursor % len(ties)]
                try:
                    load_after = self._registry.reserve_request(
                        chosen.worker_id,
                        trace.request_id,
                    )
                except WorkerNotRoutableError:
                    if attempt + 1 < _MAX_HEALTH_RACE_ATTEMPTS:
                        continue
                    _record_routing_decision(
                        trace,
                        policy="least_loaded",
                        candidates=(),
                        selected_load_before=None,
                        selected_load_after=None,
                        reason=None,
                    )
                    trace.mark_routing_end(worker_id="none")
                    raise WorkerUnavailableError(
                        "No healthy workers available in WorkerRegistry.",
                        status_code=503,
                    ) from None
                except Exception:
                    trace.mark_routing_end(worker_id="none")
                    raise

                self._tie_cursor += 1
                _record_routing_decision(
                    trace,
                    policy="least_loaded",
                    candidates=candidates,
                    selected_load_before=chosen.active_requests,
                    selected_load_after=load_after,
                    reason="min_active_requests",
                )
                trace.mark_routing_end(worker_id=chosen.worker_id)
                return chosen.worker

            trace.mark_routing_end(worker_id="none")
            raise WorkerUnavailableError(
                "No healthy workers available in WorkerRegistry.",
                status_code=503,
            )


def create_routing_policy(
    config: GatewayConfig,
    registry: WorkerRegistry,
    single_worker: InferenceWorker | None = None,
) -> RoutingPolicy:
    """Explicit routing policy factory that fails fast on unsupported policy names."""
    policy = normalize_routing_policy(config.routing_policy)
    if policy == "single":
        if single_worker is None:
            raise ValueError("single_worker must be provided when routing_policy is 'single'")
        return SingleWorkerRouter(single_worker)
    if policy == "round_robin":
        return RoundRobinRouter(registry)
    if policy == "least_loaded":
        return LeastLoadedRouter(registry)
    raise ValueError(f"Unsupported routing_policy: {config.routing_policy!r}")
