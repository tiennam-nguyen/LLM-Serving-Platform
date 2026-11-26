"""Run preregistered repeatability calibration, then a separate 54-run GPU campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

from infra.kaggle.week8_open_loop_pilot import run as pilot  # noqa: E402
from infra.scripts.run_week8_campaign import execute_campaign  # noqa: E402
from llm_serving_platform.benchmark.analysis import PRIMARY_METRICS, analyze_runs  # noqa: E402
from llm_serving_platform.benchmark.artifacts import (  # noqa: E402
    capture_git_info,
    write_json_atomic,
)
from llm_serving_platform.benchmark.campaign import prepare_campaign  # noqa: E402
from llm_serving_platform.benchmark.plots import plot_analysis  # noqa: E402
from llm_serving_platform.benchmark.scenarios import (  # noqa: E402
    REALISTIC_PROFILES,
    load_chat_token_counter,
)

# Fixed before any fresh GPU observation. High-noise calibration stops the main run.
CALIBRATION_RULE = {
    "profiles": ["steady", "burst", "heterogeneous"],
    "load_level": "medium",
    "repetitions": 3,
    "method": "twice_max_relative_deviation_across_policy_profile_groups",
    "multiplier": 2.0,
    "maximum_allowed_tolerance": 0.20,
    "primary_metrics": list(PRIMARY_METRICS),
}

PREPARATION_PLAN = {
    "model_cache": "prefetch_snapshot_then_offline_workers",
    "worker_startup": "parallel_for_every_conditioning_calibration_and_main_run",
    "conditioning_runs": 6,
    "conditioning_profile_load": "one_pair_per_profile_at_medium",
    "conditioning_not_used_to_derive_tolerances": True,
    "gpu_telemetry_interval_s": 1,
}


def prefetch_model(model: str) -> dict:
    """Executed only on the remote GPU host; no CUDA used during model download."""
    from huggingface_hub import snapshot_download

    path = Path(
        snapshot_download(
            repo_id=model,
            token=False,
            allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "*.tiktoken"],
        )
    )
    return {
        "model_id": model,
        "snapshot_revision": path.name,
        "snapshot_path": str(path),
        "prefetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def derive_tolerances(analysis: dict, *, rule: dict | None = None) -> dict[str, float]:
    """Use only independent calibration runs; never tune against main results."""
    rule = rule or CALIBRATION_RULE
    runs, repeats = analysis["runs"], analysis["repeatability"]
    groups = len(rule["profiles"]) * 2
    repetitions = rule["repetitions"]
    expected = {
        (f"w8-{profile}-medium", policy)
        for profile in rule["profiles"]
        for policy in ("round_robin", "least_loaded")
    }
    if (
        len(runs) != groups * repetitions
        or len(repeats) != groups
        or not all(
            run["validation"]["valid"]
            and run["run_status"] == "completed"
            and run["rates"]["noncompletion_rate"] == 0
            and not any(
                run.get("warning_counts", {}).get(kind)
                for kind in (
                    "dispatch_late",
                    "dispatch_after_window",
                    "in_flight_limit",
                    "missing_trace",
                    "workers_not_idle_or_unobserved",
                )
            )
            for run in runs
        )
    ):
        raise ValueError(f"Calibration requires {groups * repetitions} complete, error-free runs")
    if len(analysis["pair_checks"]) != groups * repetitions // 2 or not all(
        p["matched"] for p in analysis["pair_checks"]
    ):
        raise ValueError("Calibration policy pairs differ")
    if {(r["comparison_id"], r["policy"]) for r in repeats} != expected:
        raise ValueError("Calibration profile/policy coverage differs from preregistration")
    if any(
        r["repetitions"] != list(range(1, repetitions + 1))
        or r["status"] != "not_evaluated_no_predeclared_tolerances"
        for r in repeats
    ):
        raise ValueError("Calibration repetitions or conditions differ")
    tolerances = {}
    for metric in PRIMARY_METRICS:
        values = [row["metrics"][metric]["values"] for row in repeats]
        if any(
            len(group) != repetitions
            or any(
                isinstance(v, bool)
                or not isinstance(v, (int, float))
                or not math.isfinite(v)
                or v <= 0
                for v in group
            )
            for group in values
        ):
            raise ValueError("Every calibration metric needs three positive finite samples")
        deviations = [row["metrics"][metric]["max_relative_deviation"] for row in repeats]
        if any(
            isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) or x < 0
            for x in deviations
        ):
            raise ValueError("Calibration metric is missing or nonfinite")
        tolerance = rule["multiplier"] * max(deviations)
        if tolerance > rule["maximum_allowed_tolerance"]:
            raise ValueError("Calibration noise exceeds preregistered tolerance limit")
        tolerances[metric] = tolerance
    return tolerances


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _analysis(report: dict, protocol: Path, directory: Path) -> dict:
    sources = [Path(row["run_dir"]) for row in report["runs"]]
    result = analyze_runs(sources, output_dir=directory, protocol_path=protocol)
    plot_analysis(directory / "analysis.json", output_dir=directory / "figures")
    return result


class PolicyMismatchError(RuntimeError):
    """A benchmark assertion failure, distinct from slow or incomplete measurements."""


def run_study(output: Path, revision: str, *, suite: str = "legacy") -> dict:
    if output.exists():
        raise FileExistsError("Use a new study output directory")
    source = capture_git_info(ROOT)
    if source["git_revision"] != revision or source["git_dirty"] is not False:
        raise ValueError("Study requires the exact clean source revision")
    if suite not in {"legacy", "realistic", "stress"}:
        raise ValueError("Unknown benchmark suite")
    profiles = tuple(CALIBRATION_RULE["profiles"]) if suite == "legacy" else REALISTIC_PROFILES
    if suite == "stress":
        profiles += ("microburst_mixed",)
    rule = {**CALIBRATION_RULE, "profiles": list(profiles)}
    preparation = {**PREPARATION_PLAN, "conditioning_runs": len(profiles) * 2}
    output.mkdir(parents=True)
    preregistration = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "calibration_rule": rule,
        "calibration_runs": len(profiles) * 6,
        "main_runs": len(profiles) * 18,
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "window_ms": 20000,
        "main_analysis_may_fail_repeatability": True,
        "preparation_plan": preparation,
        "suite": suite,
        "execution_semantics": "observe_v1",
        "failure_policy": {
            "policy_mismatch": "failed",
            "slow_dispatch_or_inference": "measure_warn_continue",
            "in_flight_limit": "record_unsent_continue_without_client_queue",
            "unreliable_calibration": "continue_main_without_verified_tolerances",
            "unavailable_infrastructure": "blocked",
        },
    }
    write_json_atomic(output / "preregistration.json", preregistration)
    study = {
        "schema_version": 1,
        "source": source,
        "test_mode": "live_dual_t4",
        "started_at_utc": preregistration["created_at_utc"],
        "status": "running",
        "methodology_gate_passed": False,
        "warnings": [],
    }

    def record_execution(execution: dict, phase: str) -> None:
        if execution["run_status"] == "failed":
            raise PolicyMismatchError(f"Unexpected routing policy during {phase}")
        if execution["run_status"] != "completed":
            study["warnings"].append({"type": execution["run_status"], "phase": phase})
        if execution.get("warning_counts"):
            study["warnings"].append(
                {
                    "type": "measurement_warnings",
                    "phase": phase,
                    "counts": execution["warning_counts"],
                }
            )
        expected = (
            len(profiles) * 2
            if phase == "conditioning"
            else len(profiles) * 6
            if phase == "calibration"
            else len(profiles) * 18
        )
        if len(execution["runs"]) != expected:
            study["warnings"].append(
                {
                    "type": "incomplete_execution",
                    "phase": phase,
                    "observed_runs": len(execution["runs"]),
                    "planned_runs": expected,
                }
            )

    try:
        cache = prefetch_model(preregistration["model"])
        write_json_atomic(output / "model_cache.json", cache)
        study["model_cache"] = cache
        campaign_options = {"profiles": profiles}
        if suite != "legacy":
            campaign_options.update(
                token_counter=load_chat_token_counter(cache["snapshot_path"]),
                tokenizer_metadata={
                    "model_id": cache["model_id"],
                    "snapshot_revision": cache["snapshot_revision"],
                },
            )
        live_options = dict(
            no_gpu=False,
            revision=revision,
            prefetched_model=True,
            model_revision=cache["snapshot_revision"],
            gpu_telemetry=True,
        )
        conditioning = prepare_campaign(
            manifests_dir=output / "conditioning/manifests",
            workloads_dir=output / "conditioning/workloads",
            **campaign_options,
        )
        protocol = json.loads(conditioning.read_text())
        protocol["runs"] = [
            r
            for r in protocol["runs"]
            if r["comparison_id"].endswith("-medium") and r["repetition"] == 1
        ]
        protocol["name"] = "week8-unmeasured-conditioning"
        protocol["experiment_phase"] = "conditioning"
        protocol["repeatability"]["status"] = "calibration_only"
        protocol["preparation_plan"] = preparation
        protocol["execution_semantics"] = "observe_v1"
        write_json_atomic(conditioning, protocol)
        print(f"Starting {len(profiles) * 2} conditioning runs; excluded from cal/main", flush=True)
        execution = execute_campaign(
            conditioning,
            output_dir=output / "conditioning/execution",
            calibration=True,
            **live_options,
        )
        record_execution(execution, "conditioning")
        if execution["run_status"] == "cancelled":
            study["status"] = "cancelled"
            return study
        study["conditioning_runs"] = len(execution["runs"])
        study["preparation_complete"] = (
            execution["run_status"] == "completed" and len(execution["runs"]) == len(profiles) * 2
        )
        study["conditioning_run_ids"] = [r["run_id"] for r in execution["runs"]]
        calibration = prepare_campaign(
            manifests_dir=output / "calibration/manifests",
            workloads_dir=output / "calibration/workloads",
            **campaign_options,
        )
        protocol = json.loads(calibration.read_text())
        protocol["runs"] = [r for r in protocol["runs"] if r["comparison_id"].endswith("-medium")]
        protocol["name"] = "week8-repeatability-calibration"
        protocol["experiment_phase"] = "calibration"
        protocol["repeatability"]["status"] = "calibration_only"
        protocol["calibration_rule"] = rule
        protocol["execution_semantics"] = "observe_v1"
        write_json_atomic(calibration, protocol)
        print(f"Starting separate {len(profiles) * 6}-run calibration", flush=True)
        execution = execute_campaign(
            calibration,
            output_dir=output / "calibration/execution",
            calibration=True,
            **live_options,
        )
        record_execution(execution, "calibration")
        if execution["run_status"] == "cancelled":
            study["status"] = "cancelled"
            return study
        tolerances, frozen = None, None
        try:
            analysis = _analysis(execution, calibration, output / "calibration/analysis")
            tolerances = derive_tolerances(analysis, rule=rule)
        except (ValueError, KeyError) as exc:
            warning = {"type": "calibration_unreliable", "detail": str(exc)}
            study["warnings"].append(warning)
            print(json.dumps({"event": "study_warning", **warning}), flush=True)
        if tolerances is not None:
            frozen = {
                "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
                "rule": rule,
                "tolerances": tolerances,
                "preregistration_sha256": _hash(output / "preregistration.json"),
                "calibration_protocol_sha256": _hash(calibration),
                "calibration_analysis_sha256": _hash(output / "calibration/analysis/analysis.json"),
                "model_cache_sha256": _hash(output / "model_cache.json"),
                "conditioning_run_ids": study["conditioning_run_ids"],
                "calibration_run_ids": [r["run_id"] for r in analysis["runs"]],
            }
            write_json_atomic(output / "frozen_tolerances.json", frozen)
            print(json.dumps({"event": "calibration_frozen", "tolerances": tolerances}), flush=True)
            study["frozen_tolerances"] = frozen
        write_json_atomic(output / "study_report.json", study)
        main = prepare_campaign(
            manifests_dir=output / "main/manifests",
            workloads_dir=output / "main/workloads",
            tolerances=tolerances,
            **campaign_options,
        )
        protocol = json.loads(main.read_text())
        protocol["experiment_phase"] = "main"
        protocol["execution_semantics"] = "observe_v1"
        protocol["repeatability"]["calibration_evidence"] = frozen
        if frozen is None:
            protocol["repeatability"]["status"] = "calibration_unreliable"
        write_json_atomic(main, protocol)
        print(
            f"Starting independent {len(profiles) * 18}-run main; tolerances verified: "
            f"{frozen is not None}",
            flush=True,
        )
        execution = execute_campaign(main, output_dir=output / "main/execution", **live_options)
        record_execution(execution, "main")
        if execution["run_status"] == "cancelled":
            study["status"] = "cancelled"
            return study
        analysis = _analysis(execution, main, output / "main/analysis")
        study.update(
            status="blocked"
            if any(w["type"] == "blocked" for w in study["warnings"])
            else "completed",
            main_runs=len(analysis["runs"]),
            methodology_gate_passed=(
                study["preparation_complete"]
                and frozen is not None
                and analysis["eligible_for_policy_comparison"]
            ),
        )
        if not study["methodology_gate_passed"]:
            study["warnings"].append({"type": "methodology_gate_not_passed"})
            print(
                json.dumps({"event": "study_warning", "type": "methodology_gate_not_passed"}),
                flush=True,
            )
        return study
    except PolicyMismatchError as exc:
        study.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
        raise
    except Exception as exc:
        study.update(status="blocked", failure_type=type(exc).__name__, failure=str(exc))
        return study
    except BaseException:
        study["status"] = "cancelled"
        raise
    finally:
        study["ended_at_utc"] = datetime.now(timezone.utc).isoformat()
        write_json_atomic(output / "study_report.json", study)
        archive = output.parent / "week8-controlled-artifacts.zip"
        if archive.exists():
            raise FileExistsError("Refusing to overwrite an earlier study archive")
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(output.rglob("*")):
                if path.is_file():
                    bundle.write(path, path.relative_to(output))
        print(
            json.dumps({"event": "study_finished", "report": study, "archive": str(archive)}),
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dual-t4",
        action="store_true",
        required=True,
        help="Explicitly authorize this remote GPU study.",
    )
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--revision", required=True)
    parser.add_argument("--vllm-version", default="0.33.0")
    parser.add_argument("--suite", choices=("legacy", "realistic", "stress"), default="realistic")
    parser.add_argument("--output-dir", type=Path, default=Path("/kaggle/working/week8-controlled"))
    args = parser.parse_args()
    if sys.platform != "linux":
        raise ValueError("This launcher runs only on remote Linux GPU hosts")
    pilot.dual_t4.REPO = ROOT
    pilot.checked_inventory()
    if args.bootstrap:
        python = pilot.dual_t4.serving_python()
        pilot.dual_t4.install(python, vllm_requirement=f"vllm=={args.vllm_version}")
        subprocess.run(
            [python, "-m", "pip", "--isolated", "install", "-e", ".[analysis]"],
            cwd=ROOT,
            check=True,
            timeout=300,
        )
        subprocess.run(
            [
                python,
                str(Path(__file__).resolve()),
                "--dual-t4",
                "--revision",
                args.revision,
                "--output-dir",
                str(args.output_dir),
                "--suite",
                args.suite,
            ],
            cwd=ROOT,
            check=True,
            timeout=38000,
        )
        return
    print(
        json.dumps(run_study(args.output_dir.resolve(), args.revision, suite=args.suite), indent=2)
    )


if __name__ == "__main__":
    main()
