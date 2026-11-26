"""Full production schema -> open-loop engine -> persisted summary, CPU workers only."""

import json

import pytest

from llm_serving_platform.benchmark.runner import ExperimentRunner
from llm_serving_platform.benchmark.summary import compute_summary_v2
from tests.week6_workers import mock_gateway
from tests.week8_b_fixtures import write_manifest


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["round_robin", "least_loaded"])
@pytest.mark.parametrize("profile", ["steady", "burst", "heterogeneous"])
async def test_production_v2_pipeline_all_profiles_and_policies(tmp_path, policy, profile):
    manifest, raw, _ = write_manifest(tmp_path, policy=policy, profile=profile)
    async with mock_gateway(policy, "test-model") as (_, client, controls):
        runner = ExperimentRunner(runs_root=tmp_path / "runs", client=client)
        result = await runner.run_from_manifest(manifest)
        assert result.succeeded, result.summary
        assert result.summary == compute_summary_v2(run_dir=result.run_dir)
        assert result.summary["requests"]["both_completed"] == 4
        assert result.summary["requests"]["missing_trace"] == 0
        assert result.summary["validation"]["valid"]
        assert result.summary["latency_ms"]["e2e_p99"] is not None
        assert len(list(result.run_dir.iterdir())) == 8
        assert sum(controls.calls.values()) == 4
        assert all(
            w["active_requests"] == 0
            for w in json.loads((result.run_dir / "environment.json").read_text())["final_workers"]
        )
        for path in result.run_dir.iterdir():
            assert "Read the following context" not in path.read_text(encoding="utf-8")


@pytest.mark.asyncio
async def test_capacity_limit_warns_and_preserves_all_planned_ledger_slots(tmp_path):
    manifest, raw, _ = write_manifest(tmp_path, profile="burst")
    raw["runner"]["max_in_flight"] = 1
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    async with mock_gateway("round_robin", "test-model") as (_, client, _):
        result = await ExperimentRunner(
            runs_root=tmp_path / "runs", client=client
        ).run_from_manifest(manifest)
    assert result.run_status == "completed"
    assert result.failure_type is None
    assert result.summary["warning_counts"]["in_flight_limit"] == 3
    assert result.summary["validation"]["valid"]
    assert not result.summary["validation"]["comparison_eligible"]
    assert result.summary["requests"]["planned"] == 4
    assert result.summary["requests"]["sent"] == 1
    assert result.summary["requests"]["not_dispatched"] == 3
    assert result.summary["requests"]["observed"] == 1
    assert len((result.run_dir / "requests.jsonl").read_text().splitlines()) == 4


@pytest.mark.asyncio
async def test_preflight_failure_has_no_fabricated_traces(tmp_path):
    manifest, raw, _ = write_manifest(tmp_path)
    raw["model"]["id"] = "wrong-model"
    manifest.write_text(json.dumps(raw), encoding="utf-8")
    async with mock_gateway("round_robin", "test-model") as (_, client, controls):
        result = await ExperimentRunner(
            runs_root=tmp_path / "runs", client=client
        ).run_from_manifest(manifest)
        assert not controls.calls
    assert result.failure_type == "model_mismatch"
    assert result.run_status == "blocked"
    assert result.summary["requests"]["sent"] == 0
    assert result.summary["requests"]["not_dispatched"] == 4
    assert (result.run_dir / "traces.jsonl").read_bytes() == b""
