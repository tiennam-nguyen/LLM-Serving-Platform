"""CPU-only orchestration tests: synthetic analysis and mocks, no CUDA/vLLM execution."""

import copy
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from infra.kaggle.week8_controlled_benchmark import run as study
from infra.kaggle.week8_open_loop_pilot import package_kernel as packager
from infra.scripts import run_week8_campaign as campaign
from infra.scripts.run_week8_campaign import execute_campaign
from llm_serving_platform.benchmark.analysis import analyze_runs
from llm_serving_platform.benchmark.campaign import prepare_campaign
from llm_serving_platform.benchmark.summary import compute_summary_v2
from tests.week8_b_fixtures import raw_run


def calibration_analysis(profiles=("steady", "burst", "heterogeneous")):
    return {
        "runs": [
            {
                "validation": {"valid": True},
                "run_status": "completed",
                "rates": {"noncompletion_rate": 0},
                "run_id": f"fixture-{i}",
            }
            for i in range(len(profiles) * 6)
        ],
        "pair_checks": [{"matched": True} for _ in range(len(profiles) * 3)],
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
            for profile in profiles
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


@pytest.mark.parametrize("phase", ["calibration", "conditioning"])
def test_calibration_raw_cannot_be_used_as_main_evidence(tmp_path, phase):
    source = raw_run(tmp_path)
    env = json.loads((source / "environment.json").read_text())
    env["experiment_phase"] = phase
    (source / "environment.json").write_text(json.dumps(env))
    summary = compute_summary_v2(run_dir=source)
    assert phase + "_not_main_campaign" in summary["validation"]["ineligible_reasons"]


@pytest.mark.parametrize("unstable", [False, True])
def test_study_freezes_or_warns_then_runs_main_and_preserves_raw(tmp_path, monkeypatch, unstable):
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
    monkeypatch.setattr(
        study,
        "prefetch_model",
        lambda model: {
            "model_id": model,
            "snapshot_revision": "f" * 40,
            "snapshot_path": "synthetic-cache",
        },
    )

    def execute(protocol_path, **kwargs):
        protocol = json.loads(protocol_path.read_text())
        phase = protocol["experiment_phase"]
        stages.append(phase)
        assert kwargs["no_gpu"] is False
        assert kwargs["prefetched_model"] is True
        assert kwargs["model_revision"] == "f" * 40
        assert kwargs["gpu_telemetry"] is True
        if phase == "main":
            assert len(protocol["runs"]) == 54
            if unstable:
                assert not (output / "frozen_tolerances.json").exists()
                assert protocol["repeatability"]["status"] == "calibration_unreliable"
                assert protocol["repeatability"]["tolerances"] is None
            else:
                assert (output / "frozen_tolerances.json").is_file()
                assert protocol["repeatability"]["status"] == "predeclared"
                assert protocol["repeatability"]["tolerances"][study.PRIMARY_METRICS[0]] == 0.02
        else:
            assert kwargs["calibration"] is True
            assert protocol["repeatability"]["tolerances"] is None
            assert len(protocol["runs"]) == (6 if phase == "conditioning" else 18)
        return {
            "run_status": "completed",
            "runs": [{"run_id": f"synthetic-{phase}-{i}"} for i in range(len(protocol["runs"]))],
        }

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
    result = study.run_study(output, "synthetic-revision")
    assert stages == ["conditioning", "calibration", "main"]
    assert result["status"] == "completed"
    assert result["methodology_gate_passed"] is False
    assert any(w["type"] == "calibration_unreliable" for w in result["warnings"]) is unstable
    report = json.loads((output / "study_report.json").read_text())
    assert report["status"] == "completed"
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


@pytest.mark.parametrize(
    "options", [{"gpu_telemetry": True}, {"prefetched_model": True, "model_revision": "f" * 40}]
)
def test_cpu_rejects_gpu_preparation_before_any_process(tmp_path, monkeypatch, options):
    protocol = prepare_campaign(manifests_dir=tmp_path / "m", workloads_dir=tmp_path / "w")
    monkeypatch.setattr(campaign, "GpuTelemetry", lambda _: pytest.fail("No GPU probe in CPU mode"))
    with pytest.raises(ValueError, match="explicit live"):
        execute_campaign(protocol, output_dir=tmp_path / "out", no_gpu=True, **options)
    assert not (tmp_path / "out").exists()


def test_sampler_stops_only_its_owned_process(tmp_path, monkeypatch):
    process = SimpleNamespace(
        poll=lambda: None,
        terminate=lambda: actions.append("terminate"),
        wait=lambda **_: actions.append("wait"),
    )
    actions = []
    calls = []
    monkeypatch.setattr(
        campaign.subprocess, "Popen", lambda args, **kwargs: calls.append(args) or process
    )
    sampler = campaign.GpuTelemetry(tmp_path / "synthetic-gpu.csv")
    sampler.stop()
    assert actions == ["terminate", "wait"]
    assert calls[0][0] == "nvidia-smi" and "--loop=1" in calls[0]
    assert sampler.output.closed


def test_prefetch_records_exact_snapshot_without_cached_account(tmp_path, monkeypatch):
    calls = []
    snapshot = tmp_path / ("f" * 40)
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=lambda **kwargs: calls.append(kwargs) or str(snapshot)),
    )
    result = study.prefetch_model("synthetic-model")
    assert result["snapshot_revision"] == "f" * 40
    assert calls[0]["token"] is False
    assert "*.safetensors" in calls[0]["allow_patterns"]


@pytest.mark.parametrize("prefetched", [False, True])
def test_parallel_startup_requires_prefetched_snapshot(tmp_path, monkeypatch, prefetched):
    protocol = prepare_campaign(
        manifests_dir=tmp_path / "m",
        workloads_dir=tmp_path / "w",
        tolerances={name: 0.1 for name in study.PRIMARY_METRICS},
    )
    raw = json.loads(protocol.read_text())
    raw["runs"] = raw["runs"][:2]
    protocol.write_text(json.dumps(raw))
    pilot = study.pilot
    monkeypatch.setattr(campaign.sys, "platform", "linux")
    monkeypatch.setattr(
        campaign, "capture_git_info", lambda _: dict(git_revision="synthetic", git_dirty=False)
    )
    monkeypatch.setattr(pilot, "checked_inventory", lambda: [])
    monkeypatch.setattr(pilot.dual_t4, "torch_info", lambda: dict(devices=["T4", "T4"]))
    monkeypatch.setattr(pilot, "_supports_no_prefix_caching_flag", lambda: True)
    service = SimpleNamespace(stop=lambda: None, process=SimpleNamespace(pid=123))
    monkeypatch.setattr(
        pilot, "_start_vllm_replica", lambda *a, **k: (service, {"args": ["synthetic"]})
    )
    monkeypatch.setattr(pilot, "_wait_both_replicas_ready", lambda *a: None)
    monkeypatch.setattr(pilot, "require_unused_port", lambda *a: None)
    monkeypatch.setattr(pilot, "_wait_gateway_ready", lambda *a: None)
    monkeypatch.setattr(pilot, "_wait_port_free", lambda *a: None)
    monkeypatch.setattr(pilot.dual_t4, "capture_processes", lambda *a: {})
    monkeypatch.setattr(pilot.dual_t4, "group_pids", lambda *a: set())
    monkeypatch.setattr(pilot.dual_t4, "Service", lambda *a: service)
    waits = []
    monkeypatch.setattr(pilot.dual_t4, "wait_for", lambda *a: waits.append(a))
    client = SimpleNamespace(__enter__=lambda _: None)

    class Client:
        def __enter__(self):
            return client

        def __exit__(self, *args):
            return False

    import httpx

    monkeypatch.setattr(httpx, "Client", lambda **_: Client())
    contexts = []

    class Runner:
        def __init__(self, **kwargs):
            contexts.append(kwargs["run_context"])

        async def run_from_manifest(self, *args, **kwargs):
            return SimpleNamespace(
                run_id="synthetic",
                run_dir=tmp_path / "raw",
                run_status="completed",
                failure_type=None,
                succeeded=True,
                summary={},
            )

    monkeypatch.setattr(campaign, "ExperimentRunner", Runner)
    execute_campaign(
        protocol,
        output_dir=tmp_path / "out",
        no_gpu=False,
        revision="synthetic",
        prefetched_model=prefetched,
        model_revision="f" * 40 if prefetched else None,
    )
    assert len(waits) == (0 if prefetched else 1)
    assert all(c["cache_control"]["model_cache_offline"] is prefetched for c in contexts)


def test_changed_model_snapshot_invalidates_repetition(tmp_path):
    runs = [
        raw_run(tmp_path, policy=policy, repetition=rep)
        for rep in (1, 2, 3)
        for policy in ("round_robin", "least_loaded")
    ]
    for path in runs:
        env = json.loads((path / "environment.json").read_text())
        manifest = json.loads((path / "manifest.json").read_text())
        env["model_revision"] = ("a" if manifest["comparison"]["repetition"] == 2 else "f") * 40
        (path / "environment.json").write_text(json.dumps(env))
    report = analyze_runs(runs, output_dir=tmp_path / "derived")
    assert all(p["matched"] for p in report["pair_checks"])
    assert all(
        r["status"] == "invalid_conditions_changed_between_runs" for r in report["repeatability"]
    )


def test_worker_pins_model_and_tokenizer_to_prefetched_revision(monkeypatch):
    pilot = study.pilot
    calls = []
    monkeypatch.setattr(pilot, "require_unused_port", lambda _: None)
    monkeypatch.setattr(pilot.dual_t4, "torch_info", lambda _: {"devices": ["T4"]})
    monkeypatch.setattr(pilot.dual_t4, "Service", lambda *args: calls.append(args) or "synthetic")
    service, launch = pilot._start_vllm_replica(
        0, {}, "synthetic-model", disable_prefix_caching=True, model_revision="f" * 40
    )
    assert service == "synthetic"
    args = calls[0][0]
    assert args[args.index("--revision") + 1] == "f" * 40
    assert args[args.index("--tokenizer-revision") + 1] == "f" * 40
    assert "--no-enable-prefix-caching" in args
    assert launch["CUDA_VISIBLE_DEVICES"] == "0"
