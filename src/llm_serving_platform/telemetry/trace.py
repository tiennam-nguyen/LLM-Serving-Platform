"""Per-request trace lifecycle collector and bounded in-memory store for Week 3."""

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Optional


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class RequestTrace:
    """Captures the 6 Week 3 lifecycle milestones and derived server latencies."""

    request_id: str
    model: str = ""
    worker_id: str = ""
    status: str = "in_flight"
    token_chunks: int = 0
    error: Optional[str] = None

    # ISO-8601 UTC wall-clock timestamps for each milestone
    arrival: Optional[str] = None
    routing_start: Optional[str] = None
    routing_end: Optional[str] = None
    worker_start: Optional[str] = None
    first_token: Optional[str] = None
    last_token: Optional[str] = None

    # Monotonic perf_counter timestamps (internal use for accurate durations)
    _arrival_mono: Optional[float] = field(default=None, repr=False)
    _routing_start_mono: Optional[float] = field(default=None, repr=False)
    _routing_end_mono: Optional[float] = field(default=None, repr=False)
    _worker_start_mono: Optional[float] = field(default=None, repr=False)
    _first_token_mono: Optional[float] = field(default=None, repr=False)
    _last_token_mono: Optional[float] = field(default=None, repr=False)

    def mark_arrival(self) -> None:
        self.arrival = _utc_now_iso()
        self._arrival_mono = time.perf_counter()

    def mark_routing_start(self) -> None:
        self.routing_start = _utc_now_iso()
        self._routing_start_mono = time.perf_counter()

    def mark_routing_end(self, worker_id: str) -> None:
        self.worker_id = worker_id
        self.routing_end = _utc_now_iso()
        self._routing_end_mono = time.perf_counter()

    def mark_worker_start(self) -> None:
        self.worker_start = _utc_now_iso()
        self._worker_start_mono = time.perf_counter()

    def mark_first_token(self) -> None:
        if self._first_token_mono is None:
            self.first_token = _utc_now_iso()
            self._first_token_mono = time.perf_counter()

    def mark_last_token(self, status: str = "completed", error: Optional[str] = None) -> None:
        self.last_token = _utc_now_iso()
        self._last_token_mono = time.perf_counter()
        self.status = status
        if error is not None:
            self.error = error

    @property
    def routing_ms(self) -> Optional[float]:
        if self._routing_start_mono is not None and self._routing_end_mono is not None:
            return round((self._routing_end_mono - self._routing_start_mono) * 1000.0, 3)
        return None

    @property
    def ttft_ms(self) -> Optional[float]:
        if self._arrival_mono is not None and self._first_token_mono is not None:
            return round((self._first_token_mono - self._arrival_mono) * 1000.0, 2)
        return None

    @property
    def e2e_ms(self) -> Optional[float]:
        if self._arrival_mono is not None and self._last_token_mono is not None:
            return round((self._last_token_mono - self._arrival_mono) * 1000.0, 2)
        return None

    @property
    def tpot_ms(self) -> Optional[float]:
        if (
            self._first_token_mono is not None
            and self._last_token_mono is not None
            and self.token_chunks > 1
        ):
            return round(
                ((self._last_token_mono - self._first_token_mono) / (self.token_chunks - 1))
                * 1000.0,
                2,
            )
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize trace milestones and computed durations for API/logging."""
        return {
            "request_id": self.request_id,
            "model": self.model,
            "worker_id": self.worker_id,
            "status": self.status,
            "token_chunks": self.token_chunks,
            "error": self.error,
            "milestones": {
                "arrival": self.arrival,
                "routing_start": self.routing_start,
                "routing_end": self.routing_end,
                "worker_start": self.worker_start,
                "first_token": self.first_token,
                "last_token": self.last_token,
            },
            "metrics_ms": {
                "routing_ms": self.routing_ms,
                "ttft_ms": self.ttft_ms,
                "tpot_ms": self.tpot_ms,
                "e2e_ms": self.e2e_ms,
            },
        }


class InMemoryTraceStore:
    """Bounded FIFO in-memory store for recent request traces."""

    def __init__(self, max_size: int = 500) -> None:
        self.max_size = max_size
        self._traces: OrderedDict[str, RequestTrace] = OrderedDict()

    def save(self, trace: RequestTrace) -> None:
        if trace.request_id in self._traces:
            self._traces.move_to_end(trace.request_id)
        self._traces[trace.request_id] = trace
        while len(self._traces) > self.max_size:
            self._traces.popitem(last=False)

    def get(self, request_id: str) -> Optional[RequestTrace]:
        return self._traces.get(request_id)

    def list_recent(self, limit: int = 50) -> list[Dict[str, Any]]:
        items = list(self._traces.values())[-limit:]
        items.reverse()
        return [t.to_dict() for t in items]
