"""Routing abstractions for selecting an inference worker and recording routing timestamps."""

from abc import ABC, abstractmethod

from llm_serving_platform.telemetry.trace import RequestTrace
from llm_serving_platform.workers.base import InferenceWorker


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
