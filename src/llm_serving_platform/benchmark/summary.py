"""Deterministic experiment summary computation strictly derived from traces.jsonl."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

PERCENTILE_METHOD = "linear_interpolation_r7"
TERMINAL_STATUSES = frozenset({"completed", "error", "cancelled"})


def compute_percentile(values: list[float], percentile: float) -> float | None:
    """Compute a deterministic percentile via linear interpolation (NIST R-7 / type 7).

    For sorted values x[0..n-1]:
      - n == 0 -> None (never fabricated as 0)
      - n == 1 -> x[0]
      - k = (p / 100) * (n - 1), lo = floor(k), hi = ceil(k)
      - result = x[lo] + (k - lo) * (x[hi] - x[lo])
    """
    if not (0.0 <= percentile <= 100.0):
        raise ValueError(f"percentile must be in [0, 100], got {percentile!r}")
    finite = sorted(float(v) for v in values if isinstance(v, (int, float)) and math.isfinite(v))
    count = len(finite)
    if count == 0:
        return None
    if count == 1:
        return round(finite[0], 3)
    rank = (percentile / 100.0) * (count - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    if lo == hi:
        return round(finite[lo], 3)
    interpolated = finite[lo] + (rank - lo) * (finite[hi] - finite[lo])
    return round(interpolated, 3)


def read_traces_jsonl(traces_path: Path | str) -> list[dict[str, Any]]:
    """Read terminal request traces line-by-line from a JSONL file."""
    path = Path(traces_path)
    if not path.is_file():
        return []
    traces: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: line {line_no}: malformed trace JSON ({exc.msg})") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}: line {line_no}: trace record must be a JSON object")
        traces.append(parsed)
    return traces


def _extract_finite_metric(trace: dict[str, Any], key: str) -> float | None:
    metrics_ms = trace.get("metrics_ms")
    if not isinstance(metrics_ms, dict):
        return None
    value = metrics_ms.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    num = float(value)
    return num if math.isfinite(num) else None


def compute_summary_from_traces_file(
    traces_path: Path | str,
    *,
    run_id: str,
    run_status: str,
    planned_requests: int,
    expected_policy: str,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate summary.json strictly by re-reading traces.jsonl from disk."""
    traces = read_traces_jsonl(traces_path)
    observed = len(traces)

    status_counts: Counter[str] = Counter()
    worker_counts: Counter[str] = Counter()
    observed_policies: set[str] = set()

    routing_samples: list[float] = []
    ttft_samples: list[float] = []
    e2e_samples: list[float] = []

    for trace in traces:
        status = trace.get("status")
        if isinstance(status, str) and status in TERMINAL_STATUSES:
            status_counts[status] += 1

        routing = trace.get("routing")
        if isinstance(routing, dict):
            pol = routing.get("policy")
            if isinstance(pol, str) and pol.strip():
                observed_policies.add(pol.strip())

        worker_id = trace.get("worker_id")
        if isinstance(worker_id, str) and worker_id.strip() and worker_id != "none":
            worker_counts[worker_id] += 1

        routing_ms = _extract_finite_metric(trace, "routing_ms")
        if routing_ms is not None:
            routing_samples.append(routing_ms)

        ttft_ms = _extract_finite_metric(trace, "ttft_ms")
        if ttft_ms is not None:
            ttft_samples.append(ttft_ms)

        e2e_ms = _extract_finite_metric(trace, "e2e_ms")
        if e2e_ms is not None:
            e2e_samples.append(e2e_ms)

    summary: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "run_status": run_status,
        "policy": expected_policy,
        "observed_policies": sorted(observed_policies),
        "percentile_method": PERCENTILE_METHOD,
        "requests": {
            "planned": planned_requests,
            "observed": observed,
            "completed": status_counts["completed"],
            "error": status_counts["error"],
            "cancelled": status_counts["cancelled"],
        },
        "latency_ms": {
            "routing_p50": compute_percentile(routing_samples, 50),
            "routing_p95": compute_percentile(routing_samples, 95),
            "ttft_p50": compute_percentile(ttft_samples, 50),
            "ttft_p95": compute_percentile(ttft_samples, 95),
            "e2e_p50": compute_percentile(e2e_samples, 50),
            "e2e_p95": compute_percentile(e2e_samples, 95),
        },
        "latency_sample_counts": {
            "routing": {
                "present": len(routing_samples),
                "missing": observed - len(routing_samples),
            },
            "ttft": {
                "present": len(ttft_samples),
                "missing": observed - len(ttft_samples),
            },
            "e2e": {
                "present": len(e2e_samples),
                "missing": observed - len(e2e_samples),
            },
        },
        "worker_distribution": {k: worker_counts[k] for k in sorted(worker_counts)},
    }
    if failure is not None:
        summary["failure"] = failure
    return summary
