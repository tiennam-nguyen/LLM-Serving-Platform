"""Pilot-only raw-artifact metrics recomputation, validation, and common load selection.

IMPORTANT: This module emits pilot_metrics.json (labeled pilot selection evidence)
and pilot_report.json / pilot_summary.csv for Member A's dual-T4 pilot.
Member B owns the final production Week 8 summary.py and plotting pipelines.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from prometheus_client.parser import text_string_to_metric_families

from llm_serving_platform.benchmark.summary import (
    PERCENTILE_METHOD,
    TERMINAL_STATUSES,
    compute_percentile,
    read_traces_jsonl,
)

PILOT_REQUIRED_ARTIFACTS = (
    "manifest.json",
    "environment.json",
    "schedule.jsonl",
    "requests.jsonl",
    "traces.jsonl",
    "metrics.before.prom",
    "metrics.prom",
)

DEFAULT_SELECTION_CRITERIA: dict[str, Any] = {
    "description": (
        "Pre-declared policy-neutral selection rules applied identically to paired RR and LL "
        "runs sharing the exact same ordered workload and schedule SHA-256."
    ),
    "max_allowed_noncompletion_rate_stable": 0.02,
    "min_drain_throughput_tracking_ratio": 0.90,
    "max_backlog_fraction_stable": 0.25,
    "max_tail_e2e_p95_ratio_vs_baseline": 2.50,
    "low_min_tracking_ratio": 0.95,
    "low_max_backlog_fraction": 0.15,
    "saturation_trigger_tracking_ratio": 0.90,
    "saturation_trigger_noncompletion_rate": 0.05,
    "saturation_trigger_backlog_fraction": 0.25,
    "saturation_trigger_e2e_p95_ratio": 2.50,
}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw.strip()
        if not stripped:
            continue
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise AssertionError(f"{path}:{line_no}: malformed JSON ({exc.msg})") from exc
        if not isinstance(parsed, dict):
            raise AssertionError(f"{path}:{line_no}: expected JSON object")
        rows.append(parsed)
    return rows


def _finite_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    num = float(value)
    return num if math.isfinite(num) else None


def _extract_trace_metric(trace: dict[str, Any], key: str) -> float | None:
    metrics_ms = trace.get("metrics_ms")
    if not isinstance(metrics_ms, dict):
        return None
    return _finite_number(metrics_ms.get(key))


def compute_pilot_metrics_from_dir(run_dir: Path | str) -> dict[str, Any]:
    """Recompute all Week 8 contract metrics strictly from raw files in run_dir."""
    path = Path(run_dir)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    schedule_rows = _read_jsonl(path / "schedule.jsonl")
    request_rows = _read_jsonl(path / "requests.jsonl")
    traces = read_traces_jsonl(path / "traces.jsonl")

    planned = int(manifest["workload"]["request_count"])
    window_ms = float(manifest["load"]["measurement_window_ms"])
    window_s = window_ms / 1000.0
    policy = str(manifest["policy"]["name"])

    traces_by_id: dict[str, dict[str, Any]] = {}
    for t in traces:
        rid = t.get("request_id")
        if not isinstance(rid, str) or not rid:
            raise AssertionError("Trace missing request_id")
        if rid in traces_by_id:
            raise AssertionError(f"Duplicate trace request_id: {rid}")
        traces_by_id[rid] = t

    req_by_id: dict[str, dict[str, Any]] = {}
    sent_rows: list[dict[str, Any]] = []
    not_dispatched_rows: list[dict[str, Any]] = []
    dispatch_lags: list[float] = []
    client_end_offsets_ms: list[float] = []

    for r in request_rows:
        rid = r.get("request_id")
        if not isinstance(rid, str) or not rid:
            raise AssertionError("Request ledger entry missing request_id")
        if rid in req_by_id:
            raise AssertionError(f"Duplicate request ledger request_id: {rid}")
        req_by_id[rid] = r
        status = r.get("dispatch_status")
        if status == "sent":
            sent_rows.append(r)
            lag = _finite_number(r.get("dispatch_lag_ms"))
            if lag is not None:
                dispatch_lags.append(lag)
            end_ms = _finite_number(r.get("client_end_offset_ms"))
            if end_ms is not None:
                client_end_offsets_ms.append(end_ms)
        elif status == "not_dispatched":
            not_dispatched_rows.append(r)
        else:
            raise AssertionError(f"Unknown dispatch_status {status!r}")

    status_counts: Counter[str] = Counter()
    worker_counts: Counter[str] = Counter()
    no_worker_count = 0
    observed_policies: set[str] = set()

    routing_samples: list[float] = []
    ttft_samples: list[float] = []
    e2e_samples: list[float] = []

    for trace in traces:
        t_status = trace.get("status")
        if isinstance(t_status, str) and t_status in TERMINAL_STATUSES:
            status_counts[t_status] += 1

        routing = trace.get("routing")
        if isinstance(routing, dict):
            pol = routing.get("policy")
            if isinstance(pol, str) and pol.strip():
                observed_policies.add(pol.strip())

        wid = trace.get("worker_id")
        if isinstance(wid, str) and wid.strip() and wid != "none":
            worker_counts[wid] += 1
        else:
            no_worker_count += 1

        routing_ms = _extract_trace_metric(trace, "routing_ms")
        if routing_ms is not None:
            routing_samples.append(routing_ms)

        if t_status == "completed":
            ttft_ms = _extract_trace_metric(trace, "ttft_ms")
            if ttft_ms is not None:
                ttft_samples.append(ttft_ms)
            e2e_ms = _extract_trace_metric(trace, "e2e_ms")
            if e2e_ms is not None:
                e2e_samples.append(e2e_ms)

    # Throughput success requires BOTH client completed AND correlated gateway trace completed.
    both_completed_total = 0
    both_completed_in_window = 0
    missing_trace_count = 0
    backlog_at_window_end = 0

    for r in sent_rows:
        rid = str(r["request_id"])
        trace = traces_by_id.get(rid)
        if trace is None or not bool(r.get("trace_observed")):
            if trace is None:
                missing_trace_count += 1
        end_ms = _finite_number(r.get("client_end_offset_ms"))
        if end_ms is None or end_ms >= window_ms:
            backlog_at_window_end += 1

        if (
            r.get("client_status") == "completed"
            and trace is not None
            and trace.get("status") == "completed"
        ):
            both_completed_total += 1
            if end_ms is not None and 0.0 <= end_ms < window_ms:
                both_completed_in_window += 1

    observed = len(traces)
    sent_count = len(sent_rows)
    not_dispatched_count = len(not_dispatched_rows)
    drain_span_s = (
        max(window_s, max(client_end_offsets_ms) / 1000.0) if client_end_offsets_ms else window_s
    )

    completed_gateway_count = status_counts["completed"]
    p99_note = (
        f"E2E/TTFT P99 computed from n={len(e2e_samples)} completed samples "
        f"(pilot low-N P99 has limited statistical precision; report sample counts explicitly)."
    )

    return {
        "schema_version": 2,
        "artifact_role": "pilot_selection_evidence_not_member_b_summary",
        "run_id": environment.get("run_id"),
        "policy": policy,
        "observed_policies": sorted(observed_policies),
        "comparison": manifest.get("comparison"),
        "percentile_method": PERCENTILE_METHOD,
        "measurement_window_ms": window_ms,
        "drain_span_s": round(drain_span_s, 6),
        "workload_sha256": environment.get("workload_sha256"),
        "schedule_sha256": environment.get("schedule_sha256"),
        "schedule_count": len(schedule_rows),
        "requests": {
            "planned": planned,
            "sent": sent_count,
            "not_dispatched": not_dispatched_count,
            "observed": observed,
            "completed": completed_gateway_count,
            "error": status_counts["error"],
            "cancelled": status_counts["cancelled"],
            "missing_trace": missing_trace_count,
            "both_completed": both_completed_total,
        },
        "rates": {
            "offered_rps": round(planned / window_s, 6),
            "dispatched_rps": round(sent_count / window_s, 6),
            "completed_window_rps": round(both_completed_in_window / window_s, 6),
            "completed_drain_rps": round(both_completed_total / drain_span_s, 6),
            "noncompletion_rate": round(1.0 - (both_completed_total / planned), 6),
            "gateway_error_rate": (
                round(status_counts["error"] / observed, 6) if observed > 0 else None
            ),
            "gateway_cancelled_rate": (
                round(status_counts["cancelled"] / observed, 6) if observed > 0 else None
            ),
        },
        "latency_ms": {
            "routing_p50": compute_percentile(routing_samples, 50),
            "routing_p95": compute_percentile(routing_samples, 95),
            "routing_p99": compute_percentile(routing_samples, 99),
            "ttft_p50": compute_percentile(ttft_samples, 50),
            "ttft_p95": compute_percentile(ttft_samples, 95),
            "ttft_p99": compute_percentile(ttft_samples, 99),
            "e2e_p50": compute_percentile(e2e_samples, 50),
            "e2e_p95": compute_percentile(e2e_samples, 95),
            "e2e_p99": compute_percentile(e2e_samples, 99),
        },
        "latency_sample_counts": {
            "routing": {
                "present": len(routing_samples),
                "missing": observed - len(routing_samples),
            },
            "ttft": {
                "present": len(ttft_samples),
                "missing": completed_gateway_count - len(ttft_samples),
            },
            "e2e": {
                "present": len(e2e_samples),
                "missing": completed_gateway_count - len(e2e_samples),
            },
        },
        "dispatch_lag_ms": {
            "sample_count": len(dispatch_lags),
            "p50": compute_percentile(dispatch_lags, 50),
            "p95": compute_percentile(dispatch_lags, 95),
            "p99": compute_percentile(dispatch_lags, 99),
            "max": round(max(dispatch_lags), 3) if dispatch_lags else None,
        },
        "backlog_at_window_end": backlog_at_window_end,
        "worker_distribution": {k: worker_counts[k] for k in sorted(worker_counts)},
        "no_worker_count": no_worker_count,
        "p99_statistical_note": p99_note,
    }


def validate_pilot_run_dir(
    run_dir: Path | str,
    *,
    expected_policy: str | None = None,
    expected_planned: int | None = None,
    expected_revision: str | None = None,
    require_clean_git: bool = False,
    forbidden_substrings: list[str] | None = None,
    max_dispatch_lag_ms: float | None = None,
    require_completed: bool = True,
) -> dict[str, Any]:
    """Independently verify all Phase 11 raw-artifact invariants on a pilot run directory."""
    path = Path(run_dir)
    if not path.is_dir():
        raise AssertionError(f"Run directory missing: {path}")

    for fname in PILOT_REQUIRED_ARTIFACTS:
        if not (path / fname).is_file():
            raise AssertionError(f"Missing required artifact {fname} in {path}")

    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    environment = json.loads((path / "environment.json").read_text(encoding="utf-8"))
    schedule_rows = _read_jsonl(path / "schedule.jsonl")
    request_rows = _read_jsonl(path / "requests.jsonl")
    traces = read_traces_jsonl(path / "traces.jsonl")
    metrics_prom = (path / "metrics.prom").read_text(encoding="utf-8")

    if manifest.get("schema_version") != 2:
        raise AssertionError(
            f"Expected manifest schema_version=2, got {manifest.get('schema_version')}"
        )
    if expected_revision is not None and environment.get("git_revision") != expected_revision:
        raise AssertionError(
            f"git_revision mismatch: {environment.get('git_revision')!r} != {expected_revision!r}"
        )
    if require_clean_git and environment.get("git_dirty") is not False:
        raise AssertionError(
            f"Expected clean git checkout, got git_dirty={environment.get('git_dirty')!r}"
        )

    expected_sched_sha = hashlib.sha256((path / "schedule.jsonl").read_bytes()).hexdigest()
    if environment.get("schedule_sha256") != expected_sched_sha:
        raise AssertionError("environment.json schedule_sha256 does not match schedule.jsonl bytes")

    planned = int(manifest["workload"]["request_count"])
    if expected_planned is not None and planned != expected_planned:
        raise AssertionError(f"planned={planned} != expected={expected_planned}")
    if len(schedule_rows) != planned:
        raise AssertionError(f"schedule.jsonl count {len(schedule_rows)} != planned {planned}")

    # Unique sequences & unique request IDs; planned == sent + not_dispatched
    seqs = [r.get("sequence") for r in request_rows]
    req_ids = [r.get("request_id") for r in request_rows]
    if sorted(seqs) != list(range(planned)):
        raise AssertionError(f"Expected sequences 0..{planned - 1}, got {seqs}")
    if len(set(req_ids)) != planned:
        raise AssertionError("Duplicate request_id in requests.jsonl")
    for row in request_rows:
        if "-warmup-" in str(row.get("request_id", "")):
            raise AssertionError(f"Warm-up leaked into requests.jsonl: {row['request_id']}")

    sent_ids = {r["request_id"] for r in request_rows if r.get("dispatch_status") == "sent"}
    not_disp_ids = {
        r["request_id"] for r in request_rows if r.get("dispatch_status") == "not_dispatched"
    }
    if len(sent_ids) + len(not_disp_ids) != planned:
        raise AssertionError("planned != sent + not_dispatched")

    trace_ids: list[str] = []
    for t in traces:
        rid = t.get("request_id")
        if not isinstance(rid, str) or "-warmup-" in rid:
            raise AssertionError(f"Invalid or warm-up trace in traces.jsonl: {rid!r}")
        if rid not in sent_ids:
            raise AssertionError(
                f"Trace {rid!r} does not match a sent request (possible fake/leak)"
            )
        if t.get("status") not in TERMINAL_STATUSES:
            raise AssertionError(f"Non-terminal trace in traces.jsonl: {t.get('status')!r}")
        trace_ids.append(rid)
    if len(trace_ids) != len(set(trace_ids)):
        raise AssertionError("Duplicate request_id in traces.jsonl")

    # Verify trace_observed matches real traces.jsonl presence
    trace_id_set = set(trace_ids)
    for row in request_rows:
        if row["dispatch_status"] == "sent":
            if bool(row.get("trace_observed")) != (row["request_id"] in trace_id_set):
                raise AssertionError(
                    f"trace_observed mismatch for {row['request_id']}: "
                    f"ledger={row.get('trace_observed')}, "
                    f"in_traces={row['request_id'] in trace_id_set}"
                )
        else:
            if row.get("trace_observed") is not False:
                raise AssertionError(f"not_dispatched row has trace_observed != False: {row}")

    # Verify final workers returned to active_requests == 0
    final_workers = environment.get("final_workers") or []
    if not final_workers or any(w.get("active_requests") != 0 for w in final_workers):
        raise AssertionError(f"final_workers not idle: {final_workers}")

    families = {f.name: f for f in text_string_to_metric_families(metrics_prom)}
    if "llm_worker_active_requests" not in families:
        raise AssertionError("metrics.prom missing llm_worker_active_requests")
    active_vals = [
        s.value
        for s in families["llm_worker_active_requests"].samples
        if s.name == "llm_worker_active_requests"
    ]
    if require_completed and any(v != 0 for v in active_vals):
        raise AssertionError(f"Non-zero active requests in metrics.prom: {active_vals}")

    # Prompt/secret leak check across all files
    if forbidden_substrings:
        for fname in (*PILOT_REQUIRED_ARTIFACTS, "pilot_metrics.json"):
            fpath = path / fname
            if fpath.is_file():
                content = fpath.read_text(encoding="utf-8")
                for secret in forbidden_substrings:
                    if secret and secret in content:
                        raise AssertionError(f"Forbidden text leaked into {fname}")

    recomputed = compute_pilot_metrics_from_dir(path)
    if expected_policy is not None:
        if recomputed["policy"] != expected_policy:
            raise AssertionError(f"policy={recomputed['policy']!r} != {expected_policy!r}")
        if traces and recomputed["observed_policies"] != [expected_policy]:
            raise AssertionError(
                f"observed_policies={recomputed['observed_policies']!r} != {[expected_policy]!r}"
            )

    if require_completed:
        if recomputed["requests"]["missing_trace"] != 0:
            raise AssertionError(f"missing_trace={recomputed['requests']['missing_trace']}")
        if recomputed["requests"]["sent"] != planned:
            raise AssertionError(
                f"Incomplete dispatch in completed run: "
                f"sent={recomputed['requests']['sent']} != {planned}"
            )
        if max_dispatch_lag_ms is not None:
            lag_max = recomputed["dispatch_lag_ms"]["max"]
            if lag_max is not None and lag_max > max_dispatch_lag_ms:
                raise AssertionError(
                    f"dispatch_lag_max_ms={lag_max} > limit={max_dispatch_lag_ms}"
                )

    pilot_metrics_path = path / "pilot_metrics.json"
    if pilot_metrics_path.is_file():
        on_disk = json.loads(pilot_metrics_path.read_text(encoding="utf-8"))
        if on_disk != recomputed:
            raise AssertionError(
                f"pilot_metrics.json differs from raw recomputation:\n{on_disk}\n!=\n{recomputed}"
            )

    return {"valid": True, "metrics": recomputed}


def _policy_is_stable(
    entry: dict[str, Any],
    *,
    offered_rps: float,
    baseline_p95: float | None,
    criteria: dict[str, Any],
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not entry.get("valid"):
        reasons.append("run_invalid")
        return False, reasons

    planned = max(int(entry["planned"]), 1)
    drain_rps = float(entry["completed_drain_rps"] or 0.0)
    noncomp = float(entry["noncompletion_rate"] or 0.0)
    backlog_frac = float(entry["backlog_at_window_end"] or 0) / planned
    p95 = entry.get("e2e_p95")

    if drain_rps < float(criteria["min_drain_throughput_tracking_ratio"]) * offered_rps:
        reasons.append(f"drain_rps={drain_rps:.3f} < 0.90*offered({offered_rps:.3f})")
    if noncomp > float(criteria["max_allowed_noncompletion_rate_stable"]):
        reasons.append(f"noncompletion={noncomp:.3f} > 0.02")
    if backlog_frac > float(criteria["max_backlog_fraction_stable"]):
        reasons.append(f"backlog_frac={backlog_frac:.3f} > 0.25")
    if (
        baseline_p95 is not None
        and baseline_p95 > 0
        and p95 is not None
        and float(p95) > float(criteria["max_tail_e2e_p95_ratio_vs_baseline"]) * baseline_p95
    ):
        reasons.append(f"e2e_p95={p95:.1f} > 2.5*baseline({baseline_p95:.1f})")

    return len(reasons) == 0, reasons


def select_common_load_points(
    pair_summaries: list[dict[str, Any]],
    *,
    criteria: dict[str, Any] = DEFAULT_SELECTION_CRITERIA,
) -> dict[str, Any]:
    """Select policy-neutral common low / medium / near_saturation points from paired runs."""
    ordered = sorted(pair_summaries, key=lambda p: float(p["offered_rps"]))
    valid_pairs = [p for p in ordered if p.get("pair_valid")]
    if not valid_pairs:
        return {
            "knee_reached": False,
            "reason": "No valid paired RR/LL runs available.",
            "selected": None,
            "evaluations": [],
        }

    # Baseline p95 from lowest valid tested rate (conservative upper bound across the two policies).
    first_pair = valid_pairs[0]
    p95_vals = [
        v
        for v in (
            first_pair["round_robin"].get("e2e_p95"),
            first_pair["least_loaded"].get("e2e_p95"),
        )
        if v is not None
    ]
    baseline_p95 = max(p95_vals) if p95_vals else None

    evaluations: list[dict[str, Any]] = []
    stable_pairs: list[dict[str, Any]] = []
    unstable_pairs: list[dict[str, Any]] = []

    for pair in ordered:
        rate = float(pair["offered_rps"])
        if not pair.get("pair_valid"):
            evaluations.append(
                {
                    "offered_rps": rate,
                    "pair_valid": False,
                    "both_stable": False,
                    "rr_reasons": ["pair_invalid"],
                    "ll_reasons": ["pair_invalid"],
                }
            )
            continue
        rr_ok, rr_reasons = _policy_is_stable(
            pair["round_robin"], offered_rps=rate, baseline_p95=baseline_p95, criteria=criteria
        )
        ll_ok, ll_reasons = _policy_is_stable(
            pair["least_loaded"], offered_rps=rate, baseline_p95=baseline_p95, criteria=criteria
        )
        both_stable = rr_ok and ll_ok
        evaluations.append(
            {
                "offered_rps": rate,
                "pair_valid": True,
                "both_stable": both_stable,
                "rr_reasons": rr_reasons,
                "ll_reasons": ll_reasons,
            }
        )
        if both_stable:
            stable_pairs.append(pair)
        else:
            unstable_pairs.append(pair)

    if not stable_pairs:
        return {
            "knee_reached": False,
            "reason": "Even the lowest tested rate was unstable or invalid; lower rates needed.",
            "selected": None,
            "evaluations": evaluations,
        }

    # Knee requires at least one tested rate (valid or cleanly saturated) above a stable rate
    # that exhibits leaving the stable operating envelope.
    highest_stable_rate = float(stable_pairs[-1]["offered_rps"])
    degraded_after_stable = [
        e
        for e in evaluations
        if float(e["offered_rps"]) > highest_stable_rate and not e["both_stable"]
    ]
    if not degraded_after_stable:
        return {
            "knee_reached": False,
            "reason": (
                f"All tested rates up to {highest_stable_rate} RPS remained inside the stable "
                "operating region; higher rates are needed to locate the knee."
            ),
            "selected": None,
            "evaluations": evaluations,
        }

    # Pick near_saturation = highest stable common valid rate before degradation.
    near_sat_pair = stable_pairs[-1]
    near_sat_rps = float(near_sat_pair["offered_rps"])

    # Pick low = lowest stable common rate meeting tighter tracking/backlog thresholds.
    low_candidates = [
        p
        for p in stable_pairs
        if float(p["offered_rps"]) < near_sat_rps
        and float(p["round_robin"]["completed_drain_rps"])
        >= float(criteria["low_min_tracking_ratio"]) * float(p["offered_rps"])
        and float(p["least_loaded"]["completed_drain_rps"])
        >= float(criteria["low_min_tracking_ratio"]) * float(p["offered_rps"])
        and float(p["round_robin"]["backlog_at_window_end"]) / int(p["round_robin"]["planned"])
        <= float(criteria["low_max_backlog_fraction"])
        and float(p["least_loaded"]["backlog_at_window_end"]) / int(p["least_loaded"]["planned"])
        <= float(criteria["low_max_backlog_fraction"])
    ]
    if not low_candidates:
        low_candidates = [p for p in stable_pairs if float(p["offered_rps"]) < near_sat_rps]
    if not low_candidates:
        return {
            "knee_reached": False,
            "reason": (
                "Only one stable rate exists below saturation; need finer lower/mid candidates."
            ),
            "selected": None,
            "evaluations": evaluations,
        }
    low_pair = low_candidates[0]
    low_rps = float(low_pair["offered_rps"])

    mid_candidates = [
        p for p in stable_pairs if low_rps < float(p["offered_rps"]) < near_sat_rps
    ]
    if not mid_candidates:
        return {
            "knee_reached": False,
            "reason": (
                f"No tested stable candidate strictly between low={low_rps} and "
                f"near_saturation={near_sat_rps}; add an intermediate rate."
            ),
            "selected": None,
            "evaluations": evaluations,
        }
    geo_mid = math.sqrt(low_rps * near_sat_rps)
    medium_pair = min(
        mid_candidates,
        key=lambda p: abs(math.log(float(p["offered_rps"])) - math.log(geo_mid)),
    )

    def _entry_summary(p: dict[str, Any]) -> dict[str, Any]:
        return {
            "offered_rps": p["offered_rps"],
            "schedule_sha256": p.get("schedule_sha256"),
            "round_robin_run_id": p["round_robin"]["run_id"],
            "least_loaded_run_id": p["least_loaded"]["run_id"],
        }

    return {
        "knee_reached": True,
        "reason": (
            f"Knee detected above {near_sat_rps} RPS "
            f"(first degraded rate: {degraded_after_stable[0]['offered_rps']} RPS); "
            f"geometric midpoint target={geo_mid:.3f} RPS."
        ),
        "selected": {
            "low": _entry_summary(low_pair),
            "medium": _entry_summary(medium_pair),
            "near_saturation": _entry_summary(near_sat_pair),
        },
        "evaluations": evaluations,
    }


CSV_COLUMNS = [
    "policy",
    "policy_order",
    "offered_rps",
    "sent",
    "planned",
    "completed",
    "both_completed",
    "completed_window_rps",
    "completed_drain_rps",
    "ttft_p50_ms",
    "ttft_p95_ms",
    "ttft_p99_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "e2e_p99_ms",
    "routing_p50_ms",
    "routing_p95_ms",
    "routing_p99_ms",
    "dispatch_lag_p95_ms",
    "dispatch_lag_p99_ms",
    "dispatch_lag_max_ms",
    "noncompletion_rate",
    "gateway_error_rate",
    "gateway_cancelled_rate",
    "backlog_at_window_end",
    "worker_distribution",
    "run_id",
    "schedule_sha256",
    "valid",
    "invalid_reason",
]


def write_pilot_reports(
    output_dir: Path,
    *,
    run_rows: list[dict[str, Any]],
    pair_summaries: list[dict[str, Any]],
    criteria: dict[str, Any],
    selection: dict[str, Any],
    environment: dict[str, Any],
) -> None:
    """Persist pilot_report.json and pilot_summary.csv for downstream inspection and plotting."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 2,
        "artifact_role": "pilot_selection_evidence_not_member_b_summary",
        "environment": environment,
        "criteria": criteria,
        "selection": selection,
        "pairs": pair_summaries,
        "runs": run_rows,
    }
    (output_dir / "pilot_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    with (output_dir / "pilot_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in run_rows:
            writer.writerow({k: row.get(k) for k in CSV_COLUMNS})
