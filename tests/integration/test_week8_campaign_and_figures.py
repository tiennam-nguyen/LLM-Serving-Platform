"""Real loopback CLI/campaign smoke and portable scientific plotting, CPU only."""

import hashlib
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from infra.scripts.run_week8_campaign import execute_campaign
from llm_serving_platform.benchmark.__main__ import main
from llm_serving_platform.benchmark.analysis import PRIMARY_METRICS, analyze_runs
from llm_serving_platform.benchmark.campaign import prepare_campaign
from llm_serving_platform.benchmark.plots import plot_analysis
from tests.week8_b_fixtures import raw_run


def test_real_cpu_campaign_never_probes_or_launches_vllm(tmp_path):
    protocol = prepare_campaign(
        manifests_dir=tmp_path / "manifests",
        workloads_dir=tmp_path / "workloads",
        model="test-model",
        measurement_window_ms=1000,
    )
    with (
        patch(
            "infra.kaggle.week8_open_loop_pilot.run.checked_inventory",
            side_effect=AssertionError("CPU mode must not inspect GPUs"),
        ),
        patch(
            "infra.kaggle.week8_open_loop_pilot.run._supports_no_prefix_caching_flag",
            side_effect=AssertionError("CPU mode must not import vLLM"),
        ),
        patch(
            "infra.kaggle.week8_open_loop_pilot.run._start_vllm_replica",
            side_effect=AssertionError("CPU mode must not launch vLLM"),
        ),
    ):
        report = execute_campaign(protocol, output_dir=tmp_path / "executed", no_gpu=True, limit=2)
    assert report["run_status"] == "completed"
    assert report["test_mode"] == "cpu_mock"
    assert report["full_campaign"] is False
    assert len(report["runs"]) == 2
    for row in report["runs"]:
        environment = json.loads(Path(row["run_dir"]).joinpath("environment.json").read_text())
        assert environment["test_mode"] == "cpu_mock"
        assert environment["cache_control"]["workers_restarted_between_runs"] is False
        assert environment["protocol_sha256"] == report["protocol_sha256"]


def test_analysis_and_figures_are_recreated_from_raw_not_old_summaries(tmp_path):
    pytest.importorskip("matplotlib")
    sources = [
        raw_run(tmp_path / "raw", policy=policy, repetition=rep)
        for rep in (1, 2, 3)
        for policy in ("round_robin", "least_loaded")
    ]
    before = {str(p): p.read_bytes() for source in sources for p in source.iterdir()}
    report = analyze_runs(sources, output_dir=tmp_path / "analysis")
    assert len(report["runs"]) == 6
    assert all(p["matched"] for p in report["pair_checks"])
    assert not report["eligible_for_policy_comparison"]
    figures = plot_analysis(tmp_path / "analysis/analysis.json", output_dir=tmp_path / "figures")
    assert len(figures) == 6
    assert all(p.stat().st_size > 100 for p in figures)
    assert (
        next(p for p in figures if p.suffix == ".png").read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
    )
    assert {str(p): p.read_bytes() for source in sources for p in source.iterdir()} == before
    assert (
        main(
            [
                "analyze",
                "--runs-root",
                str(tmp_path / "raw"),
                "--output-dir",
                str(tmp_path / "cli-analysis"),
            ]
        )
        == 0
    )
    assert (tmp_path / "cli-analysis/runs.csv").is_file()


def test_equal_results_do_not_make_changed_conditions_repeatable(tmp_path):
    sources = [
        raw_run(tmp_path / "raw", policy=policy, repetition=rep)
        for rep in (1, 2, 3)
        for policy in ("round_robin", "least_loaded")
    ]
    for source in sources:
        manifest = json.loads((source / "manifest.json").read_text())
        if manifest["comparison"]["repetition"] == 2:
            manifest["generation"]["max_tokens"] = 16
            (source / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    protocol = tmp_path / "protocol.json"
    protocol.write_text(
        json.dumps(
            {
                "repeatability": {
                    "status": "predeclared",
                    "minimum_repetitions": 3,
                    "tolerances": {key: 0.05 for key in PRIMARY_METRICS},
                },
                "runs": [{"manifest": p.name + ".json"} for p in sources],
            }
        ),
        encoding="utf-8",
    )
    digest = hashlib.sha256(protocol.read_bytes()).hexdigest()
    for source in sources:
        environment = json.loads((source / "environment.json").read_text())
        environment["protocol_sha256"] = digest
        (source / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    report = analyze_runs(sources, output_dir=tmp_path / "analysis", protocol_path=protocol)
    assert all(p["matched"] for p in report["pair_checks"])
    assert all(
        r["status"] == "invalid_conditions_changed_between_runs" for r in report["repeatability"]
    )
    assert not report["eligible_for_policy_comparison"]
