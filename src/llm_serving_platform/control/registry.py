"""In-memory worker metadata, cached health candidates, and Gateway-owned request load."""

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


class DuplicateRequestReservationError(ValueError):
    """An active reservation already exists for this request ID."""


class WorkerNotRoutableError(RuntimeError):
    """The target worker is not currently HEALTHY and cannot accept a new reservation."""


def _validate_nonempty_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a nonempty string")
    return value


def _normalize_utc_timestamp(timestamp: datetime | None, label: str) -> datetime:
    resolved = timestamp if timestamp is not None else datetime.now(timezone.utc)
    if not isinstance(resolved, datetime) or resolved.utcoffset() is None:
        raise ValueError(f"{label} timestamps must be timezone-aware")
    return resolved.astimezone(timezone.utc)


@dataclass(frozen=True)
class WorkerRecord:
    worker_id: str
    endpoint: str
    model: str
    status: WorkerStatus = WorkerStatus.STARTING
    last_checked_at: datetime | None = None

    active_requests: int = 0
    last_load_change_at: datetime | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(v, str) and v.strip()
            for v in (self.worker_id, self.endpoint, self.model)
        ):
            raise ValueError("Worker ID, endpoint and model must be nonempty strings")
        if (
            isinstance(self.active_requests, bool)
            or not isinstance(self.active_requests, int)
            or self.active_requests < 0
        ):
            raise ValueError("active_requests must be an integer >= 0")
        for ts_field, ts_val in (
            ("last_checked_at", self.last_checked_at),
            ("last_load_change_at", self.last_load_change_at),
        ):
            if ts_val is not None and (
                not isinstance(ts_val, datetime) or ts_val.utcoffset() is None
            ):
                raise ValueError(f"{ts_field} must be timezone-aware when present")


@dataclass(frozen=True)
class WorkerRoutingCandidate:
    """Internal immutable point-in-time routing snapshot for a HEALTHY worker."""

    worker_id: str
    active_requests: int
    worker: InferenceWorker


class WorkerRegistry:
    """Own metadata and request-owned active load; runtime references stay internal."""

    def __init__(self) -> None:
        self._records: dict[str, WorkerRecord] = {}
        self._workers: dict[str, InferenceWorker] = {}
        self._active_reservations: dict[str, str] = {}
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

    def routing_candidates(self) -> tuple[WorkerRoutingCandidate, ...]:
        """Return a deterministic point-in-time snapshot of HEALTHY workers and their load."""
        with self._lock:
            return tuple(
                WorkerRoutingCandidate(
                    worker_id=key,
                    active_requests=self._records[key].active_requests,
                    worker=self._workers[key],
                )
                for key in sorted(self._records)
                if self._records[key].status == WorkerStatus.HEALTHY
            )

    def reserve_request(
        self,
        worker_id: str,
        request_id: str,
        *,
        changed_at: datetime | None = None,
    ) -> int:
        """Atomically reserve an active request slot on a HEALTHY worker."""
        _validate_nonempty_str(worker_id, "worker_id")
        _validate_nonempty_str(request_id, "request_id")
        timestamp = _normalize_utc_timestamp(changed_at, "Load")
        with self._lock:
            record = self.get_record(worker_id)
            if request_id in self._active_reservations:
                raise DuplicateRequestReservationError(request_id)
            if record.status != WorkerStatus.HEALTHY:
                raise WorkerNotRoutableError(
                    f"Worker {worker_id!r} is {record.status.value} and cannot accept reservations"
                )
            self._active_reservations[request_id] = worker_id
            new_active = record.active_requests + 1
            self._records[worker_id] = replace(
                record,
                active_requests=new_active,
                last_load_change_at=timestamp,
            )
            return new_active

    def release_request(
        self,
        request_id: str,
        *,
        changed_at: datetime | None = None,
    ) -> bool:
        """Idempotently release an active request reservation by request_id."""
        _validate_nonempty_str(request_id, "request_id")
        timestamp = _normalize_utc_timestamp(changed_at, "Load")
        with self._lock:
            worker_id = self._active_reservations.pop(request_id, None)
            if worker_id is None:
                return False
            record = self._records[worker_id]
            new_active = max(0, record.active_requests - 1)
            self._records[worker_id] = replace(
                record,
                active_requests=new_active,
                last_load_change_at=timestamp,
            )
            return True

    def active_request_count(self, worker_id: str) -> int:
        """Return the current Gateway-owned active request count for a registered worker."""
        with self._lock:
            return self.get_record(worker_id).active_requests

    def probe_targets(self) -> tuple[tuple[str, InferenceWorker], ...]:
        """Snapshot identities with runtime objects for the control-plane checker only."""
        with self._lock:
            return tuple((key, self._workers[key]) for key in sorted(self._workers))

    def record_health(
        self, worker_id: str, healthy: bool, *, checked_at: datetime | None = None
    ) -> WorkerRecord:
        if not isinstance(healthy, bool):
            raise ValueError("Health must be a boolean")
        timestamp = _normalize_utc_timestamp(checked_at, "Health")
        with self._lock:
            previous = self.get_record(worker_id)
            record = replace(
                previous,
                status=WorkerStatus.HEALTHY if healthy else WorkerStatus.UNHEALTHY,
                last_checked_at=timestamp,
            )
            self._records[worker_id] = record
            return record
