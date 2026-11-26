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


def derive_tolerances(analysis: dict) -> dict[str, float]:
    """Use only independent calibration runs; never tune against main results."""
    runs, repeats = analysis["runs"], analysis["repeatability"]
    expected = {
        (f"w8-{profile}-medium", policy)
        for profile in CALIBRATION_RULE["profiles"]
        for policy in ("round_robin", "least_loaded")
    }
    if (
        len(runs) != 18
        or len(repeats) != 6
        or not all(
            run["validation"]["valid"]
            and run["run_status"] == "completed"
            and run["rates"]["noncompletion_rate"] == 0
            for run in runs
        )
    ):
        raise ValueError("Calibration requires 18 complete, error-free runs")
    if len(analysis["pair_checks"]) != 9 or not all(p["matched"] for p in analysis["pair_checks"]):
        raise ValueError("Calibration policy pairs differ")
    if {(r["comparison_id"], r["policy"]) for r in repeats} != expected:
        raise ValueError("Calibration profile/policy coverage differs from preregistration")
    if any(
        r["repetitions"] != [1, 2, 3] or r["status"] != "not_evaluated_no_predeclared_tolerances"
        for r in repeats
    ):
        raise ValueError("Calibration repetitions or conditions differ")
    tolerances = {}
    for metric in PRIMARY_METRICS:
        values = [row["metrics"][metric]["values"] for row in repeats]
        if any(
            len(group) != 3
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
        tolerance = CALIBRATION_RULE["multiplier"] * max(deviations)
        if tolerance > CALIBRATION_RULE["maximum_allowed_tolerance"]:
            raise ValueError("Calibration noise exceeds preregistered limit; main run stopped")
        tolerances[metric] = tolerance
    return tolerances


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _analysis(report: dict, protocol: Path, directory: Path) -> dict:
    sources = [Path(row["run_dir"]) for row in report["runs"]]
    result = analyze_runs(sources, output_dir=directory, protocol_path=protocol)
    plot_analysis(directory / "analysis.json", output_dir=directory / "figures")
    return result


def run_study(output: Path, revision: str) -> dict:
    if output.exists():
        raise FileExistsError("Use a new study output directory")
    source = capture_git_info(ROOT)
    if source["git_revision"] != revision or source["git_dirty"] is not False:
        raise ValueError("Study requires the exact clean source revision")
    output.mkdir(parents=True)
    preregistration = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "calibration_rule": CALIBRATION_RULE,
        "calibration_runs": 18,
        "main_runs": 54,
        "model": "Qwen/Qwen2.5-3B-Instruct",
        "window_ms": 20000,
        "main_analysis_may_fail_repeatability": True,
        "preparation_plan": PREPARATION_PLAN,
    }
    write_json_atomic(output / "preregistration.json", preregistration)
    study = {
        "schema_version": 1,
        "source": source,
        "test_mode": "live_dual_t4",
        "started_at_utc": preregistration["created_at_utc"],
        "status": "running",
        "methodology_gate_passed": False,
    }
    try:
        cache = prefetch_model(preregistration["model"])
        write_json_atomic(output / "model_cache.json", cache)
        study["model_cache"] = cache
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
        protocol["preparation_plan"] = PREPARATION_PLAN
        write_json_atomic(conditioning, protocol)
        print("Starting six conditioning runs; excluded from calibration and main", flush=True)
        execution = execute_campaign(
            conditioning,
            output_dir=output / "conditioning/execution",
            calibration=True,
            **live_options,
        )
        if execution["run_status"] != "completed" or len(execution["runs"]) != 6:
            raise RuntimeError("Conditioning failed; calibration and main stopped")
        study["conditioning_runs"] = len(execution["runs"])
        study["conditioning_run_ids"] = [r["run_id"] for r in execution["runs"]]
        calibration = prepare_campaign(
            manifests_dir=output / "calibration/manifests",
            workloads_dir=output / "calibration/workloads",
        )
        protocol = json.loads(calibration.read_text())
        protocol["runs"] = [r for r in protocol["runs"] if r["comparison_id"].endswith("-medium")]
        protocol["name"] = "week8-repeatability-calibration"
        protocol["experiment_phase"] = "calibration"
        protocol["repeatability"]["status"] = "calibration_only"
        protocol["calibration_rule"] = CALIBRATION_RULE
        write_json_atomic(calibration, protocol)
        print("Starting separate 18-run calibration", flush=True)
        execution = execute_campaign(
            calibration,
            output_dir=output / "calibration/execution",
            calibration=True,
            **live_options,
        )
        if execution["run_status"] != "completed":
            raise RuntimeError("Calibration execution failed; partial raw preserved")
        analysis = _analysis(execution, calibration, output / "calibration/analysis")
        tolerances = derive_tolerances(analysis)
        frozen = {
            "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
            "rule": CALIBRATION_RULE,
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
        )
        protocol = json.loads(main.read_text())
        protocol["experiment_phase"] = "main"
        protocol["repeatability"]["calibration_evidence"] = frozen
        write_json_atomic(main, protocol)
        print("Calibration frozen; starting independent 54-run main campaign", flush=True)
        execution = execute_campaign(main, output_dir=output / "main/execution", **live_options)
        if execution["run_status"] != "completed":
            raise RuntimeError("Main execution failed; partial raw preserved")
        analysis = _analysis(execution, main, output / "main/analysis")
        study.update(
            status="completed",
            main_runs=len(analysis["runs"]),
            methodology_gate_passed=analysis["eligible_for_policy_comparison"],
        )
        return study
    except BaseException as exc:
        study.update(status="failed", failure_type=type(exc).__name__, failure=str(exc))
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
            ],
            cwd=ROOT,
            check=True,
            timeout=38000,
        )
        return
    print(json.dumps(run_study(args.output_dir.resolve(), args.revision), indent=2))


if __name__ == "__main__":
    main()
