"""Deterministic experiment summary computation strictly derived from traces.jsonl."""

from __future__ import annotations

import hashlib
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


def read_raw_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read raw records without accepting non-objects, duplicate keys or nonfinite JSON."""
    if not path.is_file():
        return []

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key in raw artifact")
            result[key] = value
        return result

    def invalid_constant(_value):
        raise ValueError("Nonfinite JSON number in raw artifact")

    result = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line, object_pairs_hook=pairs, parse_constant=invalid_constant)
        except ValueError as exc:
            raise ValueError(f"Invalid raw JSONL at {path.name}:{index}") from exc
        if not isinstance(row, dict):
            raise ValueError(f"Expected raw object at {path.name}:{index}")
        result.append(row)
    return result


def compute_summary_v2(
    *,
    run_dir: Path | str,
    run_id: str | None = None,
    run_status: str | None = None,
    config: Any = None,
    failure: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Recompute W8 from immutable raw inputs; never reads an old summary or changes a run.

    Incomplete but unambiguous runs produce a failed summary with explicit quality issues.
    Ambiguous/corrupt records raise ValueError rather than inventing a correlation.
    """
    directory = Path(run_dir)
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    env = json.loads((directory / "environment.json").read_text(encoding="utf-8"))
    # The historical pilot added this marker to its normalized manifest. Do not rewrite it.
    from llm_serving_platform.benchmark.config import validate_manifest_data

    manifest_data = dict(manifest)
    if manifest_data.get("pilot_only") is True:
        manifest_data.pop("pilot_only")
    cfg = validate_manifest_data(manifest_data)
    if cfg.schema_version != 2:
        raise ValueError("W8 summary requires manifest schema_version=2")
    if config is not None and cfg.to_normalized_dict() != config.to_normalized_dict():
        raise ValueError("Executed config differs from persisted manifest")
    identifier = env.get("run_id")
    if not isinstance(identifier, str) or not identifier or (run_id and run_id != identifier):
        raise ValueError("Raw environment run_id mismatch")
    policy = cfg.policy.name
    planned = cfg.workload.request_count
    window_ms = cfg.load.measurement_window_ms
    window_s = window_ms / 1000
    schedule = read_raw_jsonl(directory / "schedule.jsonl")
    requests = read_raw_jsonl(directory / "requests.jsonl")
    traces = read_raw_jsonl(directory / "traces.jsonl")
    issues: set[str] = set()
    required = (
        "manifest.json",
        "environment.json",
        "schedule.jsonl",
        "requests.jsonl",
        "traces.jsonl",
        "metrics.before.prom",
        "metrics.prom",
    )
    for name in required:
        if not (directory / name).is_file():
            issues.add("missing_" + name.replace(".", "_"))

    def number(value):
        return (
            not isinstance(value, bool)
            and isinstance(value, (float, int))
            and math.isfinite(value)
            and value >= 0
        )

    if len(schedule) != planned:
        issues.add("schedule_count_mismatch")
    last_offset = -1.0
    entry_ids = set()
    for seq, row in enumerate(schedule):
        if set(row) != {"sequence", "entry_id", "workload_index", "scheduled_offset_ms"}:
            raise ValueError("Invalid raw schedule keys")
        offset = row["scheduled_offset_ms"]
        if (
            type(row["sequence"]) is not int
            or row["sequence"] != seq
            or not number(offset)
            or offset < last_offset
            or offset >= window_ms
            or type(row["workload_index"]) is not int
            or row["workload_index"] < 0
            or not isinstance(row["entry_id"], str)
            or not row["entry_id"]
            or row["entry_id"] in entry_ids
        ):
            raise ValueError("Invalid raw schedule order/identity/timing")
        entry_ids.add(row["entry_id"])
        last_offset = offset
    schedule_path = directory / "schedule.jsonl"
    schedule_bytes = schedule_path.read_bytes() if schedule_path.exists() else b""
    schedule_file_hash = hashlib.sha256(schedule_bytes).hexdigest() if schedule_bytes else None
    # Git can check historical LF artifacts out as CRLF on Windows. Preserve those files;
    # verify only this explicitly reported newline transformation, never other content edits.
    schedule_hash = (
        hashlib.sha256(schedule_bytes.replace(b"\r\n", b"\n")).hexdigest()
        if schedule_bytes
        else None
    )
    if (
        env.get("schedule_sha256") not in {schedule_hash, schedule_file_hash}
        or schedule_hash is None
    ):
        issues.add("schedule_hash_mismatch")
    workload_hash = env.get("workload_sha256")
    if (
        not isinstance(workload_hash, str)
        or len(workload_hash) != 64
        or any(ch not in "0123456789abcdef" for ch in workload_hash)
    ):
        issues.add("missing_workload_hash")
    if env.get("measurement_window_ms") != window_ms:
        issues.add("window_mismatch")
    if not env.get("ended_at"):
        issues.add("unfinished_environment")
    if requests and not env.get("measured_started_at"):
        # Preflight failures legitimately have only not-dispatched records.
        if any(row.get("dispatch_status") == "sent" for row in requests):
            issues.add("missing_measurement_start")

    ledger_by_id = {}
    sequences = set()
    sent, aborted, lags, ends = [], [], [], []
    ledger_keys = {
        "sequence",
        "request_id",
        "dispatch_status",
        "not_dispatched_reason",
        "dispatch_offset_ms",
        "dispatch_lag_ms",
        "client_end_offset_ms",
        "http_status",
        "client_status",
        "client_error_type",
        "trace_observed",
    }
    error_types = {
        None,
        "client_timeout",
        "client_cancelled",
        "transport_error",
        "sse_error",
        "http_error",
        "incomplete_sse",
    }
    for row in requests:
        if set(row) != ledger_keys:
            raise ValueError("Invalid raw request ledger keys")
        seq, rid = row["sequence"], row["request_id"]
        if (
            type(seq) is not int
            or not 0 <= seq < planned
            or seq in sequences
            or not isinstance(rid, str)
            or rid != f"{identifier}-req-{seq:05d}"
            or rid in ledger_by_id
            or type(row["trace_observed"]) is not bool
        ):
            raise ValueError("Ambiguous raw request identity")
        sequences.add(seq)
        ledger_by_id[rid] = row
        if row["dispatch_status"] == "not_dispatched":
            if (
                row["not_dispatched_reason"] != "aborted"
                or row["trace_observed"]
                or any(
                    row[key] is not None
                    for key in (
                        "dispatch_offset_ms",
                        "dispatch_lag_ms",
                        "client_end_offset_ms",
                        "http_status",
                        "client_status",
                        "client_error_type",
                    )
                )
            ):
                raise ValueError("Not-dispatched record contains fabricated HTTP observations")
            aborted.append(row)
            continue
        if row["dispatch_status"] != "sent" or row["not_dispatched_reason"] is not None:
            raise ValueError("Invalid dispatch outcome")
        dispatch, lag, end = (
            row[k] for k in ("dispatch_offset_ms", "dispatch_lag_ms", "client_end_offset_ms")
        )
        if (
            not all(number(v) for v in (dispatch, lag, end))
            or end < dispatch - 0.002
            or seq >= len(schedule)
            or abs(dispatch - schedule[seq]["scheduled_offset_ms"] - lag) > 0.0021
        ):
            raise ValueError("Invalid request timing or dispatch lag")
        if (
            row["client_status"] not in TERMINAL_STATUSES
            or row["client_error_type"] not in error_types
        ):
            raise ValueError("Invalid client outcome vocabulary")
        code = row["http_status"]
        if code is not None and (type(code) is not int or not 100 <= code <= 599):
            raise ValueError("Invalid HTTP status")
        if row["client_status"] == "completed" and (
            code != 200 or row["client_error_type"] is not None
        ):
            raise ValueError("Inconsistent client completion")
        if dispatch >= window_ms or lag > cfg.runner.max_dispatch_lag_ms + 0.001:
            issues.add("generator_limited")
        sent.append(row)
        lags.append(lag)
        ends.append(end)
    if len(requests) != planned:
        issues.add("incomplete_request_ledger")

    trace_by_id, counts, distribution = {}, Counter(), Counter()
    samples = {"routing": [], "ttft": [], "e2e": []}
    observed_policies = set()
    no_worker = 0
    for trace in traces:
        rid = trace.get("request_id")
        if (
            not isinstance(rid, str)
            or rid in trace_by_id
            or rid not in ledger_by_id
            or ledger_by_id[rid]["dispatch_status"] != "sent"
            or trace.get("status") not in TERMINAL_STATUSES
        ):
            raise ValueError("Invalid/orphan/duplicate terminal trace")
        trace_by_id[rid] = trace
        status = trace["status"]
        counts[status] += 1
        if trace.get("model") != cfg.model.id:
            issues.add("model_mismatch")
        observed = trace.get("routing", {}).get("policy")
        if observed:
            observed_policies.add(observed)
        if observed != policy:
            issues.add("policy_mismatch")
        worker_id = trace.get("worker_id")
        if isinstance(worker_id, str) and worker_id and worker_id != "none":
            distribution[worker_id] += 1
        else:
            no_worker += 1
        for metric in samples:
            if metric != "routing" and status != "completed":
                continue
            value = _extract_finite_metric(trace, metric + "_ms")
            if value is not None:
                if value < 0:
                    raise ValueError("Negative server latency")
                samples[metric].append(value)
    missing = sum(row["request_id"] not in trace_by_id for row in sent)
    if missing:
        issues.add("missing_trace")
    if any(row["trace_observed"] != (row["request_id"] in trace_by_id) for row in sent):
        issues.add("trace_observation_mismatch")
    workers = env.get("final_workers", [])
    if (
        not isinstance(workers, list)
        or len(workers) != cfg.topology.worker_count
        or any(
            type(w.get("active_requests")) is not int or w["active_requests"] != 0 for w in workers
        )
    ):
        issues.add("workers_not_idle_or_unobserved")
    successful = [
        row
        for row in sent
        if row["client_status"] == "completed"
        and trace_by_id.get(row["request_id"], {}).get("status") == "completed"
    ]
    window_completed = sum(row["client_end_offset_ms"] < window_ms for row in successful)
    drain_s = max(window_s, max(ends) / 1000) if ends else window_s
    requested_status = run_status or env.get("run_status")
    if requested_status is None:
        requested_status = "completed" if len(sent) == planned and not issues else "failed"
    final_status = "failed" if issues or requested_status != "completed" else "completed"
    failure_obj = failure or env.get("failure")
    if final_status == "failed" and failure_obj is None:
        failure_obj = {"type": "generator_limited" if aborted else "artifact_incomplete"}
    quality = sorted(issues)
    eligible_reasons = list(quality)
    if final_status != "completed":
        eligible_reasons.append("run_failed")
    if env.get("git_dirty") is not False or not env.get("git_revision"):
        eligible_reasons.append("unverified_source_revision")
    if manifest.get("pilot_only"):
        eligible_reasons.append("historical_pilot_not_controlled_campaign")
    if env.get("test_mode") != "live_dual_t4":
        eligible_reasons.append("gpu_campaign_not_verified")
    if env.get("experiment_phase") == "calibration":
        eligible_reasons.append("calibration_not_main_campaign")
    if env.get("experiment_phase") == "conditioning":
        eligible_reasons.append("conditioning_not_main_campaign")
    cache = env.get("cache_control") or {}
    if not cache.get("workers_restarted_between_runs"):
        eligible_reasons.append("cache_procedure_not_verified")
    return {
        "schema_version": 2,
        "artifact_role": "benchmark_summary",
        "run_id": identifier,
        "run_status": final_status,
        "policy": policy,
        "comparison": cfg.comparison.model_dump(),
        "observed_policies": sorted(observed_policies),
        "percentile_method": PERCENTILE_METHOD,
        "measurement_window_ms": window_ms,
        "drain_span_s": round(drain_s, 6),
        "workload_sha256": workload_hash,
        "schedule_sha256": schedule_hash,
        "schedule_file_sha256": schedule_file_hash,
        "hash_normalization": "crlf_to_lf",
        "source_git_revision": env.get("git_revision"),
        "source_git_dirty": env.get("git_dirty"),
        "execution_mode": env.get(
            "test_mode", "historical_pilot" if manifest.get("pilot_only") else "unverified"
        ),
        "experiment_phase": env.get("experiment_phase", "main"),
        "requests": {
            "planned": planned,
            "sent": len(sent),
            "not_dispatched": len(aborted),
            "ledger_missing": planned - len(requests),
            "observed": len(traces),
            "completed": counts["completed"],
            "error": counts["error"],
            "cancelled": counts["cancelled"],
            "missing_trace": missing,
            "both_completed": len(successful),
        },
        "rates": {
            "offered_rps": round(planned / window_s, 6),
            "dispatched_rps": round(len(sent) / window_s, 6),
            "completed_window_rps": round(window_completed / window_s, 6),
            "completed_drain_rps": round(len(successful) / drain_s, 6),
            "gateway_error_rate": round(counts["error"] / len(traces), 6) if traces else None,
            "gateway_cancelled_rate": round(counts["cancelled"] / len(traces), 6)
            if traces
            else None,
            "noncompletion_rate": round(1 - len(successful) / planned, 6),
        },
        "latency_ms": {
            f"{name}_p{p}": compute_percentile(values, p)
            for name, values in samples.items()
            for p in (50, 95, 99)
        },
        "latency_sample_counts": {
            name: {
                "present": len(values),
                "missing": (len(traces) if name == "routing" else counts["completed"])
                - len(values),
            }
            for name, values in samples.items()
        },
        "dispatch_lag_ms": {
            "sample_count": len(lags),
            **{f"p{p}": compute_percentile(lags, p) for p in (50, 95, 99)},
            "max": round(max(lags), 3) if lags else None,
        },
        "worker_distribution": dict(sorted(distribution.items())),
        "no_worker_count": no_worker,
        "backlog_at_window_end": sum(row["client_end_offset_ms"] >= window_ms for row in sent),
        "validation": {
            "valid": not issues,
            "issues": quality,
            "comparison_eligible": not eligible_reasons,
            "ineligible_reasons": sorted(set(eligible_reasons)),
        },
        "p99_statistical_note": (
            f"TTFT/E2E use n={counts['completed']} completed traces; low-N P99 is descriptive."
        ),
        **({"failure": failure_obj} if failure_obj else {}),
    }
