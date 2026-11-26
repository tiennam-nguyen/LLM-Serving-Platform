"""Each realistic scenario traverses the production runner, HTTP gateway and raw analysis."""

import json

import pytest

from llm_serving_platform.benchmark.runner import ExperimentRunner
from llm_serving_platform.benchmark.scenarios import REALISTIC_PROFILES
from llm_serving_platform.benchmark.summary import compute_summary_v2
from tests.unit.test_week8_scheduler import FakeClock
from tests.week6_workers import mock_gateway
from tests.week8_b_fixtures import write_manifest


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", [*REALISTIC_PROFILES, "microburst_mixed"])
@pytest.mark.parametrize("policy", ["round_robin", "least_loaded"])
async def test_each_realistic_scenario_replays_through_http_and_recomputes(
    tmp_path, profile, policy
):
    manifest, _, rows = write_manifest(
        tmp_path, profile=profile, policy=policy, count=6, window=200
    )
    async with mock_gateway(policy, "test-model") as (_, client, controls):
        result = await ExperimentRunner(
            runs_root=tmp_path / "runs", client=client
        ).run_from_manifest(manifest)
        assert result.run_status == "completed"
        assert result.summary["policy_check"] == "passed"
        assert result.summary == compute_summary_v2(run_dir=result.run_dir)
        assert result.summary["requests"]["both_completed"] == 6
        assert sum(controls.calls.values()) == 6
        assert all(
            w["active_requests"] == 0 for w in (await client.get("/control/v1/workers")).json()
        )
    schedule = [
        json.loads(line) for line in (result.run_dir / "schedule.jsonl").read_text().splitlines()
    ]
    assert [row["scheduled_offset_ms"] for row in schedule] == [
        row["scheduled_offset_ms"] for row in rows
    ]
    assert {row["id"].split("-")[1] for row in rows} == {"short", "medium", "long"}


@pytest.mark.asyncio
@pytest.mark.parametrize("policy", ["round_robin", "least_loaded"])
async def test_delay_past_entire_window_warns_and_finishes_without_faking_comparison(
    tmp_path, policy
):
    manifest, raw, _ = write_manifest(tmp_path, policy=policy, count=4, window=100)
    raw["runner"]["max_dispatch_lag_ms"] = 250
    manifest.write_text(json.dumps(raw))
    clock = FakeClock()
    clock.lag_schedule_ms[0] = 1200
    async with mock_gateway(policy, "test-model") as (_, client, _):
        result = await ExperimentRunner(
            runs_root=tmp_path / "runs",
            client=client,
            clock=clock.clock,
            sleep_until=clock.sleep_until,
        ).run_from_manifest(manifest)
        assert all(
            w["active_requests"] == 0 for w in (await client.get("/control/v1/workers")).json()
        )
    assert result.run_status == "completed"
    assert result.summary["requests"]["both_completed"] == 4
    assert result.summary["warning_counts"]["dispatch_late"] == 4
    assert result.summary["warning_counts"]["dispatch_after_window"] == 4
    assert result.summary["policy_check"] == "passed"
    assert result.summary["validation"]["valid"]
    assert not result.summary["validation"]["comparison_eligible"]
    assert result.summary == compute_summary_v2(run_dir=result.run_dir)
