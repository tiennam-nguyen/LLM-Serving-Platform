"""In-memory worker metadata and cached health candidates, without network calls."""

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import Enum
from threading import RLock

from llm_serving_platform.workers.base import InferenceWorker


class WorkerStatus(str, Enum):
    STARTING = "STARTING"
    HEALTHY = "HEALTHY"
    UNHEALTHY = "UNHEALTHY"


class DuplicateWorkerError(ValueError):
    """A worker with this identity is already registered."""


class UnknownWorkerError(KeyError):
    """The requested worker is not registered."""


@dataclass(frozen=True)
class WorkerRecord:
    worker_id: str
    endpoint: str
    model: str
    status: WorkerStatus = WorkerStatus.STARTING
    last_checked_at: datetime | None = None


class WorkerRegistry:
    """Own metadata; runtime references are returned only to health/routing consumers."""

    def __init__(self) -> None:
        self._records: dict[str, WorkerRecord] = {}
        self._workers: dict[str, InferenceWorker] = {}
        self._lock = RLock()

    def register(self, worker: InferenceWorker, endpoint: str, model: str) -> WorkerRecord:
        worker_id = worker.worker_id
        if not all(isinstance(v, str) and v.strip() for v in (worker_id, endpoint, model)):
            raise ValueError("Worker ID, endpoint and model must be nonempty strings")
        with self._lock:
            if worker_id in self._records:
                raise DuplicateWorkerError(worker_id)
            record = WorkerRecord(worker_id=worker_id, endpoint=endpoint, model=model)
            self._workers[worker_id] = worker
            self._records[worker_id] = record
            return record

    def get_record(self, worker_id: str) -> WorkerRecord:
        with self._lock:
            try:
                return self._records[worker_id]
            except KeyError:
                raise UnknownWorkerError(worker_id) from None

    def list_workers(self) -> list[WorkerRecord]:
        with self._lock:
            return [self._records[key] for key in sorted(self._records)]

    def healthy_workers(self) -> list[InferenceWorker]:
        with self._lock:
            return [
                self._workers[key]
                for key in sorted(self._records)
                if self._records[key].status == WorkerStatus.HEALTHY
            ]

    def probe_targets(self) -> tuple[tuple[str, InferenceWorker], ...]:
        """Snapshot identities with runtime objects for the control-plane checker only."""
        with self._lock:
            return tuple((key, self._workers[key]) for key in sorted(self._workers))

    def record_health(
        self, worker_id: str, healthy: bool, *, checked_at: datetime | None = None
    ) -> WorkerRecord:
        if not isinstance(healthy, bool):
            raise ValueError("Health must be a boolean")
        timestamp = checked_at if checked_at is not None else datetime.now(timezone.utc)
        if timestamp.utcoffset() is None:
            raise ValueError("Health timestamps must be timezone-aware")
        timestamp = timestamp.astimezone(timezone.utc)
        with self._lock:
            previous = self.get_record(worker_id)
            record = replace(
                previous,
                status=WorkerStatus.HEALTHY if healthy else WorkerStatus.UNHEALTHY,
                last_checked_at=timestamp,
            )
            self._records[worker_id] = record
            return record
