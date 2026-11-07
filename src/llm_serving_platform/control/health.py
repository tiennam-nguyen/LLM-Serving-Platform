"""Independent, concurrent health-probe cycles; scheduling belongs to integration."""

import asyncio
import logging
import math
from collections.abc import Callable
from datetime import datetime, timezone

from llm_serving_platform.control.registry import WorkerRegistry
from llm_serving_platform.workers.base import InferenceWorker

logger = logging.getLogger(__name__)


class HealthMonitor:
    def __init__(
        self,
        registry: WorkerRegistry,
        *,
        timeout_s: float = 5.0,
        clock: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Probe timeout must be finite and positive")
        self._registry = registry
        self._timeout_s = timeout_s
        self._clock = clock
        self._cycle_lock = asyncio.Lock()

    async def check_once(self) -> None:
        async with self._cycle_lock:
            await asyncio.gather(
                *(self._probe(key, worker) for key, worker in self._registry.probe_targets())
            )

    async def _probe(self, worker_id: str, worker: InferenceWorker) -> None:
        try:
            healthy = await asyncio.wait_for(worker.check_health(), timeout=self._timeout_s)
        except Exception:
            # Probe failures are health data; cancellation of the cycle still propagates.
            logger.warning("Health probe failed for worker %s", worker_id)
            healthy = False
        self._registry.record_health(worker_id, healthy is True, checked_at=self._clock())
