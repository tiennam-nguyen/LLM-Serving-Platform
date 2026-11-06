"""Routing abstractions for selecting an inference worker and recording routing timestamps."""

from abc import ABC, abstractmethod
from threading import Lock

from llm_serving_platform.control.registry import WorkerRegistry
from llm_serving_platform.telemetry.trace import RequestTrace
from llm_serving_platform.workers.base import InferenceWorker, WorkerUnavailableError


class RoutingPolicy(ABC):
    """Abstract interface for worker selection strategies (Open/Closed & Dependency Inversion)."""

    @abstractmethod
    def select_worker(self, trace: RequestTrace) -> InferenceWorker:
        """Select a target inference worker and record routing trace milestones."""
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
        trace.mark_routing_end(worker_id=getattr(chosen, "worker_id", "worker-vllm-0"))
        return chosen


class RoundRobinRouter(RoutingPolicy):
    """Week 4 routing policy that cycles sequentially across cached HEALTHY workers."""

    def __init__(self, registry: WorkerRegistry) -> None:
        self._registry = registry
        self._lock = Lock()
        self._cursor = 0

    @property
    def registry(self) -> WorkerRegistry:
        return self._registry

    def select_worker(self, trace: RequestTrace) -> InferenceWorker:
        """Select the next healthy worker in round-robin order and record routing milestones."""
        trace.mark_routing_start()
        candidates = self._registry.healthy_workers()
        if not candidates:
            trace.mark_routing_end(worker_id="none")
            raise WorkerUnavailableError(
                "No healthy workers available in WorkerRegistry.",
                status_code=503,
            )
        with self._lock:
            chosen = candidates[self._cursor % len(candidates)]
            self._cursor += 1
        trace.mark_routing_end(worker_id=getattr(chosen, "worker_id", "worker-unknown"))
        return chosen
