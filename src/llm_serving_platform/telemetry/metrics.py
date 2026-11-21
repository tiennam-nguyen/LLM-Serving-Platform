"""Per-Gateway Prometheus metrics; no process-global registration or request labels."""

from collections import Counter as LoadCounter
from collections.abc import Iterator
from threading import Lock

from prometheus_client import CollectorRegistry, Counter, Histogram, generate_latest
from prometheus_client.core import GaugeMetricFamily

from llm_serving_platform.control.registry import WorkerRegistry, WorkerStatus
from llm_serving_platform.telemetry.trace import RequestTrace

TERMINAL_STATUSES = ("completed", "error", "cancelled")


class _WorkerCollector:
    """Read current cached health and reservations without probing workers during scrapes."""

    def __init__(self, workers: WorkerRegistry) -> None:
        self._workers = workers
        # SingleWorkerRouter has no registry reservations in Week 6.
        self._unreserved: LoadCounter[str] = LoadCounter()
        self._lock = Lock()

    def track_unreserved(self, worker_id: str, delta: int) -> None:
        with self._lock:
            self._unreserved[worker_id] += delta

    def collect(self) -> Iterator[GaugeMetricFamily]:
        active = GaugeMetricFamily(
            "llm_worker_active_requests", "Current Gateway-owned requests.", labels=["worker_id"]
        )
        health = GaugeMetricFamily(
            "llm_worker_health",
            "Cached worker health: HEALTHY=1, otherwise=0.",
            labels=["worker_id"],
        )
        records = self._workers.list_workers()
        with self._lock:
            loads = dict(self._unreserved)
        for record in records:
            loads[record.worker_id] = loads.get(record.worker_id, 0) + record.active_requests
            health.add_metric([record.worker_id], int(record.status == WorkerStatus.HEALTHY))
        for worker_id, count in sorted(loads.items()):
            active.add_metric([worker_id], count)
        yield active
        yield health


class GatewayMetrics:
    """Own one registry per app. The HTTP response owner records each terminal outcome once.

    Gateway count/E2E include completed, error and cancelled admitted requests. TTFT is
    observed only if content arrived. Router count includes successful selections;
    router duration also includes failed selections with measured routing milestones.
    HTTP validation failures and duplicate active IDs do not admit a new inference.
    """

    def __init__(self, workers: WorkerRegistry) -> None:
        self.registry = CollectorRegistry()
        self._requests = Counter(
            "llm_gateway_requests_total",
            "Terminal inference requests.",
            ["status"],
            registry=self.registry,
        )
        for status in TERMINAL_STATUSES:
            self._requests.labels(status=status)
        self._e2e = Histogram(
            "llm_gateway_request_e2e_seconds",
            "Arrival to terminal request milestone.",
            buckets=(0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300),
            registry=self.registry,
        )
        self._ttft = Histogram(
            "llm_gateway_ttft_seconds",
            "Arrival to first non-empty content chunk.",
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10, 30, 60),
            registry=self.registry,
        )
        self._routing = Histogram(
            "llm_router_decision_seconds",
            "Worker selection and reservation duration.",
            ["policy"],
            buckets=(0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1, 1),
            registry=self.registry,
        )
        self._routed = Counter(
            "llm_router_requests_total",
            "Successful worker selections.",
            ["policy", "worker_id"],
            registry=self.registry,
        )
        self._workers = _WorkerCollector(workers)
        self.registry.register(self._workers)

    def track_unreserved(self, worker_id: str, delta: int) -> None:
        self._workers.track_unreserved(worker_id, delta)

    def observe_routing(self, trace: RequestTrace) -> None:
        policy = trace.routing_policy
        if policy not in {"single", "round_robin", "least_loaded"}:
            return
        if trace.routing_ms is not None:
            self._routing.labels(policy=policy).observe(trace.routing_ms / 1000)
        if trace.worker_id and trace.worker_id != "none":
            self._routed.labels(policy=policy, worker_id=trace.worker_id).inc()

    def observe_request(self, trace: RequestTrace) -> None:
        if trace.status not in TERMINAL_STATUSES:
            return
        self._requests.labels(status=trace.status).inc()
        if trace.e2e_ms is not None:
            self._e2e.observe(trace.e2e_ms / 1000)
        if trace.ttft_ms is not None:
            self._ttft.observe(trace.ttft_ms / 1000)

    def render(self) -> bytes:
        return generate_latest(self.registry)
