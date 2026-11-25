"""Production schema/workload/summary tests with independently expected outcomes."""

import hashlib
import json
from copy import deepcopy
from unittest.mock import AsyncMock, patch

import pytest

from llm_serving_platform.benchmark.analysis import analyze_runs, repeatability
from llm_serving_platform.benchmark.artifacts import ArtifactWriter
from llm_serving_platform.benchmark.campaign import prepare_campaign
from llm_serving_platform.benchmark.config import (
    ExperimentConfigError,
    load_manifest,
    validate_manifest_data,
)
from llm_serving_platform.benchmark.scheduler import OpenLoopScheduler
from llm_serving_platform.benchmark.summary import compute_summary_v2
from llm_serving_platform.benchmark.workload import (
    WorkloadError,
    load_workload,
    select_scheduled_entries,
)
from tests.week8_b_fixtures import raw_run, write_manifest


@pytest.mark.asyncio
async def test_early_os_timer_wakeup_cannot_trigger_early_dispatch():
    clock = iter([0.0, 0.4, 1.0])
    scheduler = OpenLoopScheduler(
        send_chat=AsyncMock(), fetch_trace=AsyncMock(), clock=lambda: next(clock)
    )
    with patch("asyncio.sleep", new_callable=AsyncMock) as sleep:
        await scheduler._default_sleep_until(1.0)
    assert [call.args[0] for call in sleep.await_args_list] == pytest.approx([1.0, 0.6])


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("runner", "concurrency", 2),
        ("runner", "max_in_flight", True),
        ("runner", "request_timeout_s", float("nan")),
        ("runner", "max_dispatch_lag_ms", 0),
        ("runner", "gateway_url", "http://user:password@host"),
        ("runner", "gateway_url", "http://host?secret=x"),
        ("workload", "profile", "unknown"),
        ("workload", "request_count", False),
        ("policy", "name", "single"),
        ("load", "mode", "closed_loop"),
        ("load", "measurement_window_ms", float("inf")),
        ("comparison", "repetition", 0),
        ("comparison", "id", "../escape"),
    ],
)
def test_v2_rejects_invalid_inputs_before_dispatch(tmp_path, section, key, value):
    _, raw, _ = write_manifest(tmp_path)
    raw[section][key] = value
    with pytest.raises(ExperimentConfigError):
        validate_manifest_data(raw)


@pytest.mark.parametrize("version", [True, False, "2", 2.0, 3, None])
def test_schema_version_never_coerces(tmp_path, version):
    _, raw, _ = write_manifest(tmp_path)
    raw["schema_version"] = version
    with pytest.raises(ExperimentConfigError):
        validate_manifest_data(raw)


def test_materialized_burst_preserves_order_and_rejects_bad_identity(tmp_path):
    path, _, rows = write_manifest(tmp_path, profile="burst")
    config, source = load_manifest(path)
    entries = load_workload(source, schema_version=2, expected_count=4, measurement_window_ms=200)
    assert [entry.sequence for entry in select_scheduled_entries(entries, 4)] == [0, 1, 2, 3]
    assert all(entry.scheduled_offset_ms == 0 for entry in entries)
    assert config.to_normalized_dict()["runner"].get("concurrency") is None
    assert config.to_normalized_dict()["comparison"]["repetition"] == 1
    rows[1]["id"] = rows[0]["id"]
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(WorkloadError, match="unique"):
        load_workload(source, schema_version=2)


@pytest.mark.parametrize("change", ["sequence", "outside", "decreasing", "boolean", "extra"])
def test_schedule_validation_is_fail_fast(tmp_path, change):
    _, raw, rows = write_manifest(tmp_path)
    if change == "sequence":
        rows[1]["sequence"] = 5
    if change == "outside":
        rows[1]["scheduled_offset_ms"] = 200
    if change == "decreasing":
        rows[2]["scheduled_offset_ms"] = 0
    if change == "boolean":
        rows[1]["scheduled_offset_ms"] = True
    if change == "extra":
        rows[1]["temperature"] = 1
    source = tmp_path / "bad.jsonl"
    source.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    with pytest.raises(WorkloadError):
        load_workload(source, schema_version=2, measurement_window_ms=200)


def test_writer_never_accepts_path_run_ids(tmp_path):
    path, _, _ = write_manifest(tmp_path)
    config, _ = load_manifest(path)
    for run_id in ("../escape", "nested/name", "C:\\escape", ".."):
        with pytest.raises(ValueError):
            ArtifactWriter.create(config=config, runs_root=tmp_path / "runs", run_id=run_id)
    assert not (tmp_path / "escape").exists()


def test_v2_percentiles_window_boundary_and_drain_denominators(tmp_path):
    path = raw_run(tmp_path)
    # A malicious/stale summary must never influence recomputation.
    (path / "summary.json").write_text('{"rates":{"offered_rps":99999}}', encoding="utf-8")
    summary = compute_summary_v2(run_dir=path)
    assert summary["validation"]["valid"] is True
    assert summary["latency_ms"]["ttft_p50"] == 20
    assert summary["latency_ms"]["ttft_p99"] == 29.8
    assert summary["latency_ms"]["e2e_p99"] == 991
    assert summary["rates"]["offered_rps"] == 2
    assert summary["rates"]["completed_window_rps"] == 1
    assert summary["rates"]["completed_drain_rps"] == pytest.approx(4 / 3, abs=1e-6)
    assert summary["worker_distribution"] == {"worker-mock-0": 1, "worker-mock-1": 1}
    rows = [json.loads(line) for line in (path / "requests.jsonl").read_text().splitlines()]
    rows[1]["client_end_offset_ms"] = 1000
    (path / "requests.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    assert compute_summary_v2(run_dir=path)["rates"]["completed_window_rps"] == 1


def test_missing_and_error_traces_are_not_zero_latency_or_success(tmp_path):
    path = raw_run(tmp_path)
    traces = [json.loads(line) for line in (path / "traces.jsonl").read_text().splitlines()]
    traces[0]["metrics_ms"]["ttft_ms"] = None
    traces[1]["status"] = "error"
    (path / "traces.jsonl").write_text(
        "\n".join(json.dumps(row) for row in traces), encoding="utf-8"
    )
    summary = compute_summary_v2(run_dir=path)
    assert summary["latency_ms"]["ttft_p99"] is None
    assert summary["latency_sample_counts"]["ttft"] == {"present": 0, "missing": 1}
    assert summary["rates"]["gateway_error_rate"] == 0.5
    assert summary["rates"]["noncompletion_rate"] == 0.5
    (path / "traces.jsonl").write_text(json.dumps(traces[0]) + "\n", encoding="utf-8")
    missing = compute_summary_v2(run_dir=path)
    assert missing["run_status"] == "failed"
    assert missing["requests"]["missing_trace"] == 1
    assert "missing_trace" in missing["validation"]["issues"]


def test_duplicate_or_orphan_traces_cannot_be_summarized(tmp_path):
    path = raw_run(tmp_path)
    original = (path / "traces.jsonl").read_text()
    with (path / "traces.jsonl").open("a") as handle:
        handle.write(original.splitlines()[0] + "\n")
    with pytest.raises(ValueError):
        compute_summary_v2(run_dir=path)


def test_crlf_checkout_is_reported_without_modifying_evidence(tmp_path):
    path = raw_run(tmp_path)
    source = path / "schedule.jsonl"
    original = source.read_bytes()
    source.write_bytes(original.replace(b"\n", b"\r\n"))
    summary = compute_summary_v2(run_dir=path)
    assert summary["validation"]["valid"] is True
    assert summary["schedule_sha256"] == hashlib.sha256(original).hexdigest()
    assert summary["schedule_file_sha256"] != summary["schedule_sha256"]
    assert source.read_bytes().count(b"\r\n") == 2


def test_repeatability_is_between_runs_and_requires_predeclared_thresholds(tmp_path):
    rows = [compute_summary_v2(run_dir=raw_run(tmp_path, repetition=i)) for i in range(1, 4)]
    assert repeatability(rows)["status"] == "not_evaluated_no_predeclared_tolerances"
    tolerances = {
        key: 0.05
        for key in ("latency_ms.ttft_p50", "latency_ms.e2e_p50", "rates.completed_drain_rps")
    }
    assert repeatability(rows, tolerances=tolerances)["status"] == "passed"
    altered = deepcopy(rows)
    altered[-1]["latency_ms"]["ttft_p50"] = 40
    assert repeatability(altered, tolerances=tolerances)["status"] == "failed"


def test_analysis_is_read_only_and_reports_unmatched_pairs(tmp_path):
    path = raw_run(tmp_path / "raw")
    before = {p.name: p.read_bytes() for p in path.iterdir()}
    report = analyze_runs([path], output_dir=tmp_path / "derived")
    assert report["pair_checks"][0]["matched"] is False
    assert not report["eligible_for_policy_comparison"]
    assert {p.name: p.read_bytes() for p in path.iterdir()} == before
    with pytest.raises(FileExistsError):
        analyze_runs([path], output_dir=path)


def test_campaign_has_all_pairs_profiles_and_repetitions_with_identical_inputs(tmp_path):
    campaign = prepare_campaign(
        manifests_dir=tmp_path / "manifests", workloads_dir=tmp_path / "workloads"
    )
    protocol = json.loads(campaign.read_text())
    assert len(protocol["runs"]) == 54
    assert protocol["load_points_rps"] == {"low": 1, "medium": 4, "near_saturation": 28}
    assert protocol["repeatability"]["tolerances"] is None
    grouped = {}
    for run in protocol["runs"]:
        cfg, source = load_manifest(campaign.parent / run["manifest"])
        entries = load_workload(
            source,
            schema_version=2,
            expected_count=cfg.workload.request_count,
            measurement_window_ms=cfg.load.measurement_window_ms,
        )
        assert len(entries) == cfg.workload.request_count
        grouped.setdefault(cfg.comparison.id, set()).add(
            hashlib.sha256(source.read_bytes()).hexdigest()
        )
    assert len(grouped) == 9 and all(len(hashes) == 1 for hashes in grouped.values())
    # Repetition two reverses the RR -> LL order used in repetition one.
    assert [row["policy"] for row in protocol["runs"][:4]] == [
        "round_robin",
        "least_loaded",
        "least_loaded",
        "round_robin",
    ]
    heterogeneous = (tmp_path / "workloads/heterogeneous_low.jsonl").read_text().splitlines()
    lengths = [len(json.loads(row)["messages"][0]["content"]) for row in heterogeneous]
    assert max(lengths) > min(lengths) * 5
    with pytest.raises(FileExistsError):
        prepare_campaign(manifests_dir=campaign.parent, workloads_dir=tmp_path / "workloads")
