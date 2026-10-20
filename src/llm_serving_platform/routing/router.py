"""Routing abstraction for selecting an inference worker and recording routing timestamps."""

from llm_serving_platform.telemetry.trace import RequestTrace
from llm_serving_platform.workers.base import InferenceWorker


class SingleWorkerRouter:
    """Routes requests to the active worker while recording routing_start and routing_end."""

    def __init__(self, worker: InferenceWorker) -> None:
        self._worker = worker

    def select_worker(self, trace: RequestTrace) -> InferenceWorker:
        """Select a target worker and record routing trace milestones."""
        trace.mark_routing_start()
        chosen = self._worker
        trace.mark_routing_end(worker_id=getattr(chosen, "worker_id", "worker-vllm-0"))
        return chosen
