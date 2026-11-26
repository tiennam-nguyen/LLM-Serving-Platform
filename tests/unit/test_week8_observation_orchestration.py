"""The remote study's failure policy is exercised entirely with CPU mocks."""

import json
import sys
import zipfile

import pytest

from infra.kaggle.week8_controlled_benchmark import package_kernel
from infra.kaggle.week8_controlled_benchmark import run as study
from infra.scripts import run_week8_campaign as campaign
from llm_serving_platform.benchmark.campaign import prepare_campaign
from llm_serving_platform.benchmark.scenarios import REALISTIC_PROFILES
from tests.unit.test_week8_realistic_scenarios import count_fixture_tokens
from tests.unit.test_week8_study_orchestration import calibration_analysis


@pytest.fixture
def source(monkeypatch):
    monkeypatch.setattr(
        study,
        "capture_git_info",
        lambda _: {
            "git_revision": "synthetic-revision",
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(
        study,
        "prefetch_model",
        lambda model: {
            "model_id": model,
            "snapshot_revision": "f" * 40,
            "snapshot_path": "synthetic-cache",
        },
    )


def test_realistic_suite_uses_cached_tokenizer_and_all_phases(tmp_path, monkeypatch, source):
    monkeypatch.setattr(study, "load_chat_token_counter", lambda _: count_fixture_tokens)
    stages = []

    def execute(path, **options):
        protocol = json.loads(path.read_text())
        phase = protocol["experiment_phase"]
        stages.append((phase, len(protocol["runs"])))
        assert protocol["workload_design"]["token_counts_verified"]
        assert protocol["workload_design"]["tokenizer"]["snapshot_revision"] == "f" * 40
        if phase == "main":
            assert protocol["repeatability"]["status"] == "predeclared"
            assert (tmp_path / "study/frozen_tolerances.json").is_file()
        return {
            "run_status": "completed",
            "warning_counts": {"request_error": 1},
            "runs": [{"run_id": f"synthetic-{phase}-{i}"} for i in range(len(protocol["runs"]))],
        }

    def analyze(execution, path, directory):
        directory.mkdir(parents=True)
        result = (
            calibration_analysis(REALISTIC_PROFILES)
            if "calibration" in str(path)
            else {"runs": execution["runs"], "eligible_for_policy_comparison": False}
        )
        (directory / "analysis.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(study, "execute_campaign", execute)
    monkeypatch.setattr(study, "_analysis", analyze)
    result = study.run_study(tmp_path / "study", "synthetic-revision", suite="realistic")
    assert stages == [("conditioning", 8), ("calibration", 24), ("main", 72)]
    assert result["status"] == "completed"
    assert result["warnings"]
    assert not result["methodology_gate_passed"]


def test_policy_mismatch_is_a_failure_and_preserves_archive(tmp_path, monkeypatch, source):
    monkeypatch.setattr(
        study,
        "execute_campaign",
        lambda *a, **k: {
            "run_status": "failed",
            "runs": [],
        },
    )
    with pytest.raises(study.PolicyMismatchError, match="routing policy"):
        study.run_study(tmp_path / "study", "synthetic-revision")
    with zipfile.ZipFile(tmp_path / "week8-controlled-artifacts.zip") as archive:
        report = json.loads(archive.read("study_report.json"))
    assert report["status"] == "failed"
    assert report["failure_type"] == "PolicyMismatchError"


def test_unavailable_infrastructure_is_blocked_not_policy_failure(tmp_path, monkeypatch, source):
    def unavailable(_model):
        raise OSError("Synthetic unavailable model cache")

    monkeypatch.setattr(study, "prefetch_model", unavailable)
    result = study.run_study(tmp_path / "study", "synthetic-revision")
    assert result["status"] == "blocked"
    assert result["failure_type"] == "OSError"
    assert not result["methodology_gate_passed"]
    assert (tmp_path / "week8-controlled-artifacts.zip").is_file()


def test_cancelled_study_does_not_start_later_phases(tmp_path, monkeypatch, source):
    calls = []

    def cancel(path, **_options):
        calls.append(path)
        return {"run_status": "cancelled", "runs": []}

    monkeypatch.setattr(study, "execute_campaign", cancel)
    result = study.run_study(tmp_path / "study", "synthetic-revision")
    assert result["status"] == "cancelled"
    assert len(calls) == 1
    assert not (tmp_path / "study/main").exists()


def test_packaged_launcher_preserves_explicit_gpu_and_suite_options(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(package_kernel, "git", lambda *_: "a" * 40)
    monkeypatch.setattr(package_kernel, "package", lambda *a, **k: calls.append(k))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "package",
            "--owner",
            "synthetic-owner",
            "--base",
            "base",
            "--output-dir",
            str(tmp_path),
            "--suite",
            "stress",
        ],
    )
    package_kernel.main()
    arguments = calls[0]["launcher_arguments"]
    assert "--dual-t4" in arguments
    assert arguments[arguments.index("--suite") + 1] == "stress"


@pytest.mark.parametrize("fault", ["unverified", "missing_metadata", "revision", "model"])
def test_live_suite_rejects_unverified_tokenizer_before_hardware_probes(
    tmp_path, monkeypatch, fault
):
    path = prepare_campaign(
        manifests_dir=tmp_path / "manifests",
        workloads_dir=tmp_path / "workloads",
        profiles=("steady_mixed",),
        measurement_window_ms=1000,
        tolerances={name: 0.1 for name in study.PRIMARY_METRICS},
    )
    protocol = json.loads(path.read_text())
    design = protocol["workload_design"]
    design.update(
        token_counts_verified=True,
        tokenizer={
            "model_id": protocol["model"],
            "snapshot_revision": "f" * 40,
        },
    )
    if fault == "unverified":
        design["token_counts_verified"] = False
    elif fault == "missing_metadata":
        design["tokenizer"] = None
    elif fault == "revision":
        design["tokenizer"]["snapshot_revision"] = "b" * 40
    else:
        design["tokenizer"]["model_id"] = "wrong-model"
    path.write_text(json.dumps(protocol))
    monkeypatch.setattr(campaign.sys, "platform", "linux")
    monkeypatch.setattr(
        campaign,
        "capture_git_info",
        lambda _: {
            "git_revision": "synthetic",
            "git_dirty": False,
        },
    )
    monkeypatch.setattr(study.pilot, "checked_inventory", lambda: pytest.fail("GPU probe called"))
    with pytest.raises(ValueError, match="actual tokenizer"):
        campaign.execute_campaign(
            path,
            output_dir=tmp_path / "out",
            no_gpu=False,
            revision="synthetic",
            prefetched_model=True,
            model_revision="f" * 40,
        )
