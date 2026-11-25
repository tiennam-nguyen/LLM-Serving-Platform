"""Read-only run analysis, paired-input checks and between-run repeatability."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from llm_serving_platform.benchmark.artifacts import capture_git_info, write_json_atomic
from llm_serving_platform.benchmark.summary import compute_summary_v2

PRIMARY_METRICS = ("latency_ms.ttft_p50", "latency_ms.e2e_p50", "rates.completed_drain_rps")


def _metric(summary: dict[str, Any], name: str) -> float | None:
    section, key = name.split(".")
    return summary[section].get(key)


def repeatability(
    summaries: list[dict[str, Any]],
    *,
    minimum_repetitions: int = 3,
    tolerances: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Test each independent run against the median, never pool request samples across runs."""
    if type(minimum_repetitions) is not int or minimum_repetitions < 3:
        raise ValueError("At least three repetitions are required")
    if tolerances is not None and (
        set(tolerances) != set(PRIMARY_METRICS)
        or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in tolerances.values()
        )
    ):
        raise ValueError("Invalid predeclared repeatability tolerances")
    metrics = {}
    for name in PRIMARY_METRICS:
        values = [_metric(summary, name) for summary in summaries]
        present = [
            value
            for value in values
            if isinstance(value, (float, int))
            and not isinstance(value, bool)
            and math.isfinite(value)
        ]
        median = statistics.median(present) if present else None
        deviation = (
            max(abs(v - median) for v in present) / abs(median)
            if median
            else (0.0 if present and all(v == 0 for v in present) else None)
        )
        tolerance = (tolerances or {}).get(name)
        metrics[name] = {
            "values": values,
            "median": median,
            "max_relative_deviation": deviation,
            "tolerance": tolerance,
            "passed": (
                deviation <= tolerance
                if deviation is not None and tolerance is not None and len(present) == len(values)
                else None
            ),
        }
    repetitions = [s["comparison"]["repetition"] for s in summaries]
    enough = len(set(repetitions)) == len(repetitions) and len(repetitions) >= minimum_repetitions
    if not enough:
        status = "insufficient_repetitions"
    elif tolerances is None:
        status = "not_evaluated_no_predeclared_tolerances"
    else:
        status = "passed" if all(v["passed"] is True for v in metrics.values()) else "failed"
    return {"status": status, "repetitions": sorted(repetitions), "metrics": metrics}


def analyze_runs(
    run_dirs: list[Path],
    *,
    output_dir: Path,
    protocol_path: Path | None = None,
) -> dict[str, Any]:
    """Create new derived files; never modify source manifests, traces, metrics or summaries."""
    if not run_dirs:
        raise ValueError("At least one run directory is required")
    resolved = [p.resolve() for p in run_dirs]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Duplicate run directory")
    target = output_dir.resolve()
    if target.exists() or any(target == p or target.is_relative_to(p) for p in resolved):
        raise FileExistsError("Use a new analysis directory outside source runs")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8")) if protocol_path else None
    settings = (protocol or {}).get("repeatability", {})
    summaries, provenance, grouped = [], [], defaultdict(list)
    for path in resolved:
        summary = compute_summary_v2(run_dir=path)
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        environment = json.loads((path / "environment.json").read_text(encoding="utf-8"))
        summary["workload_profile"] = manifest["workload"]["profile"]
        summaries.append(summary)
        provenance.append(
            {
                "run_dir": str(path),
                "run_id": summary["run_id"],
                "git_revision": environment.get("git_revision"),
                "files": {
                    p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in path.iterdir()
                    if p.is_file()
                },
            }
        )
        grouped[(summary["comparison"]["id"], summary["policy"])].append(summary)
    if len({s["run_id"] for s in summaries}) != len(summaries):
        raise ValueError("Run IDs must be unique across input directories")
    pairs = defaultdict(lambda: defaultdict(list))
    group_conditions = defaultdict(set)
    manifest_names = []
    protocol_hashes = []
    for path, summary in zip(resolved, summaries):
        key = (summary["comparison"]["id"], summary["comparison"]["repetition"])
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        environment = json.loads((path / "environment.json").read_text(encoding="utf-8"))
        manifest_names.append(manifest["name"])
        protocol_hashes.append(environment.get("protocol_sha256"))
        invariant = {
            k: manifest[k]
            for k in ("model", "topology", "generation", "warmup", "seed", "runner", "load")
        }
        # Ports/hosts are transport addresses, not workload controls. Record topology separately.
        invariant["runner"] = {k: v for k, v in invariant["runner"].items() if k != "gateway_url"}
        invariant.update(
            workload_sha256=summary["workload_sha256"],
            schedule_sha256=summary["schedule_sha256"],
            git_revision=environment.get("git_revision"),
            packages=environment.get("packages"),
            gpu=environment.get("gpu"),
        )
        invariant["cache_control"] = environment.get("cache_control")
        invariant["worker_runtime_args"] = environment.get("worker_runtime_args")
        pairs[key][summary["policy"]].append(invariant)
        group_conditions[(summary["comparison"]["id"], summary["policy"])].add(
            json.dumps(invariant, sort_keys=True)
        )
    pair_checks = []
    for (comparison_id, repetition), policies in sorted(pairs.items()):
        complete = set(policies) == {"round_robin", "least_loaded"}
        unique = complete and all(len(v) == 1 for v in policies.values())
        matched = unique and policies["round_robin"] == policies["least_loaded"]
        pair_checks.append(
            {
                "comparison_id": comparison_id,
                "repetition": repetition,
                "matched": matched,
                "reason": (
                    None
                    if matched
                    else "missing_policy"
                    if not complete
                    else "duplicate_policy_repetition"
                    if not unique
                    else "different_experimental_conditions"
                ),
            }
        )
    repeated = [
        {
            "comparison_id": comparison_id,
            "policy": policy,
            **repeatability(
                rows,
                minimum_repetitions=settings.get("minimum_repetitions", 3),
                tolerances=settings.get("tolerances"),
            ),
        }
        for (comparison_id, policy), rows in sorted(grouped.items())
    ]
    for result in repeated:
        if len(group_conditions[(result["comparison_id"], result["policy"])]) > 1:
            result["status"] = "invalid_conditions_changed_between_runs"
    protocol_digest = (
        hashlib.sha256(protocol_path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if protocol_path
        else None
    )
    expected_names = [Path(row["manifest"]).stem for row in (protocol or {}).get("runs", [])]
    protocol_validation = {
        "coverage_complete": bool(expected_names)
        and sorted(expected_names) == sorted(manifest_names),
        "hash_matched": bool(protocol_digest)
        and all(h == protocol_digest for h in protocol_hashes),
        "predeclared": settings.get("status") == "predeclared",
    }
    report = {
        "schema_version": 1,
        "analysis_git": capture_git_info(),
        "runs": summaries,
        "provenance": provenance,
        "pair_checks": pair_checks,
        "repeatability": repeated,
        "protocol_validation": protocol_validation,
        "eligible_for_policy_comparison": (
            all(s["validation"]["comparison_eligible"] for s in summaries)
            and all(p["matched"] for p in pair_checks)
            and all(r["status"] == "passed" for r in repeated)
            and all(protocol_validation.values())
        ),
        "protocol_sha256": protocol_digest,
    }
    target.mkdir(parents=True)
    write_json_atomic(target / "analysis.json", report)
    with (target / "runs.csv").open("w", encoding="utf-8", newline="") as handle:
        columns = [
            "run_id",
            "policy",
            "profile",
            "load_level",
            "repetition",
            "valid",
            "offered_rps",
            "completed_window_rps",
            "completed_drain_rps",
            "ttft_p50",
            "ttft_p95",
            "ttft_p99",
            "e2e_p50",
            "e2e_p95",
            "e2e_p99",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for s in summaries:
            writer.writerow(
                {
                    "run_id": s["run_id"],
                    "policy": s["policy"],
                    "profile": s["workload_profile"],
                    "load_level": s["comparison"]["load_level"],
                    "repetition": s["comparison"]["repetition"],
                    "valid": s["validation"]["valid"],
                    **{k: s["rates"][k] for k in columns[6:9]},
                    **{k: s["latency_ms"][k] for k in columns[9:]},
                }
            )
    return report
