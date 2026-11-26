"""GPU study orchestration tests using synthetic analysis, never local CUDA or Docker."""

import copy
import json
import zipfile
from pathlib import Path

import pytest

from infra.kaggle.week8_controlled_benchmark import run as study
from infra.kaggle.week8_open_loop_pilot import package_kernel as packager
from infra.scripts.run_week8_campaign import execute_campaign
from llm_serving_platform.benchmark.campaign import prepare_campaign
from llm_serving_platform.benchmark.summary import compute_summary_v2
from tests.week8_b_fixtures import raw_run


def calibration_analysis():
    return {
        "runs": [
            {
                "validation": {"valid": True},
                "run_status": "completed",
                "rates": {"noncompletion_rate": 0},
                "run_id": f"fixture-{i}",
            }
            for i in range(18)
        ],
        "pair_checks": [{"matched": True} for _ in range(9)],
        "repeatability": [
            {
                "comparison_id": f"w8-{profile}-medium",
                "policy": policy,
                "repetitions": [1, 2, 3],
                "status": "not_evaluated_no_predeclared_tolerances",
                "metrics": {
                    metric: {"values": [99, 100, 101], "max_relative_deviation": 0.01}
                    for metric in study.PRIMARY_METRICS
                },
            }
            for profile in ("steady", "burst", "heterogeneous")
            for policy in ("round_robin", "least_loaded")
        ],
    }


def test_calibration_uses_worst_independent_group_not_main_results():
    analysis = calibration_analysis()
    metric = study.PRIMARY_METRICS[0]
    analysis["repeatability"][4]["metrics"][metric] = {
        "values": [95, 100, 105],
        "max_relative_deviation": 0.05,
    }
    before = copy.deepcopy(analysis)
    result = study.derive_tolerances(analysis)
    assert result[metric] == pytest.approx(0.10)
    assert result[study.PRIMARY_METRICS[1]] == pytest.approx(0.02)
    assert analysis == before


@pytest.mark.parametrize("fault", ["noise", "missing", "error", "pair", "conditions", "coverage"])
def test_unreliable_calibration_is_rejected(fault):
    analysis = calibration_analysis()
    row = analysis["repeatability"][0]
    if fault == "noise":
        row["metrics"][study.PRIMARY_METRICS[0]]["max_relative_deviation"] = 0.101
    elif fault == "missing":
        row["metrics"][study.PRIMARY_METRICS[0]]["values"][1] = None
    elif fault == "error":
        analysis["runs"][0]["rates"]["noncompletion_rate"] = 0.01
    elif fault == "pair":
        analysis["pair_checks"][0]["matched"] = False
    elif fault == "conditions":
        row["status"] = "invalid_conditions_changed_between_runs"
    else:
        row["comparison_id"] = "unexpected-profile"
    with pytest.raises(ValueError):
        study.derive_tolerances(analysis)


def test_calibration_needs_flag_and_distinct_protocol(tmp_path):
    protocol = prepare_campaign(manifests_dir=tmp_path / "m", workloads_dir=tmp_path / "w")
    with pytest.raises(ValueError, match="explicit flag"):
        execute_campaign(protocol, output_dir=tmp_path / "out", no_gpu=True, calibration=True)
    assert not (tmp_path / "out").exists()


def test_calibration_raw_cannot_be_used_as_main_evidence(tmp_path):
    source = raw_run(tmp_path)
    env = json.loads((source / "environment.json").read_text())
    env["experiment_phase"] = "calibration"
    (source / "environment.json").write_text(json.dumps(env))
    summary = compute_summary_v2(run_dir=source)
    assert "calibration_not_main_campaign" in summary["validation"]["ineligible_reasons"]


@pytest.mark.parametrize("unstable", [False, True])
def test_study_freezes_before_main_and_archives_failures(tmp_path, monkeypatch, unstable):
    output = tmp_path / "study"
    monkeypatch.setattr(
        study,
        "capture_git_info",
        lambda _: {
            "git_revision": "synthetic-revision",
            "git_dirty": False,
        },
    )
    stages = []

    def execute(protocol_path, **kwargs):
        protocol = json.loads(protocol_path.read_text())
        phase = protocol["experiment_phase"]
        stages.append(phase)
        assert kwargs["no_gpu"] is False
        if phase == "main":
            assert (output / "frozen_tolerances.json").is_file()
            assert protocol["repeatability"]["status"] == "predeclared"
            assert len(protocol["runs"]) == 54
            assert protocol["repeatability"]["tolerances"][study.PRIMARY_METRICS[0]] == 0.02
        else:
            assert kwargs["calibration"] is True
            assert protocol["repeatability"]["tolerances"] is None
            assert len(protocol["runs"]) == 18
        return {"run_status": "completed", "runs": []}

    def analyze(_execution, protocol, directory):
        if "calibration" in str(protocol):
            result = calibration_analysis()
            if unstable:
                result["repeatability"][0]["metrics"][study.PRIMARY_METRICS[0]][
                    "max_relative_deviation"
                ] = 0.2
        else:
            result = {"runs": [{} for _ in range(54)], "eligible_for_policy_comparison": False}
        directory.mkdir(parents=True)
        (directory / "analysis.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(study, "execute_campaign", execute)
    monkeypatch.setattr(study, "_analysis", analyze)
    if unstable:
        with pytest.raises(ValueError, match="noise"):
            study.run_study(output, "synthetic-revision")
        assert stages == ["calibration"]
    else:
        result = study.run_study(output, "synthetic-revision")
        assert stages == ["calibration", "main"]
        assert result["status"] == "completed"
        assert result["methodology_gate_passed"] is False
    report = json.loads((output / "study_report.json").read_text())
    assert report["status"] == ("failed" if unstable else "completed")
    with zipfile.ZipFile(tmp_path / "week8-controlled-artifacts.zip") as archive:
        assert "study_report.json" in archive.namelist()
        assert "preregistration.json" in archive.namelist()
        assert ("frozen_tolerances.json" in archive.namelist()) is not unstable


@pytest.mark.parametrize("controlled", [False, True])
def test_private_source_bundle_preserves_old_and_new_launchers(tmp_path, monkeypatch, controlled):
    def git(*args):
        if args[:2] == ("status", "--porcelain"):
            return ""
        if args[:2] == ("branch", "--show-current"):
            return "feat/synthetic-study"
        if args[0] == "rev-parse":
            return "b" * 40 if args[1] == "HEAD" else "a" * 40
        if args[0] == "merge-base":
            return "a" * 40
        if args[:2] == ("bundle", "create"):
            Path(args[2]).write_bytes(b"synthetic-bundle-not-a-gpu-result")
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(packager, "git", git)
    options = (
        {
            "launcher_script": "infra/kaggle/week8_controlled_benchmark/run.py",
            "launcher_arguments": ["--bootstrap", "--revision", "b" * 40],
            "kernel_name": "synthetic-controlled-study",
            "timeout_seconds": 39600,
        }
        if controlled
        else {}
    )
    packager.package(
        tmp_path, "synthetic-owner", "synthetic-base", "synthetic-model", "0.33.0", **options
    )
    script = (tmp_path / "run.py").read_text()
    compile(script, "synthetic-package.py", "exec")
    metadata = json.loads((tmp_path / "kernel-metadata.json").read_text())
    assert metadata["is_private"] is True
    assert metadata["machine_shape"] == "NvidiaTeslaT4"
    assert metadata["id"].startswith("synthetic-owner/")
    if controlled:
        assert "infra/kaggle/week8_controlled_benchmark/run.py" in script
        assert "timeout=39600" in script
    else:
        assert "infra/kaggle/week8_open_loop_pilot/run.py" in script
        assert "timeout=3550" in script
