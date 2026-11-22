"""Unit tests for Week 7 experiment manifest validation and JSONL workload loading."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_serving_platform.benchmark.config import (
    ExperimentConfig,
    ExperimentConfigError,
    load_manifest,
)
from llm_serving_platform.benchmark.workload import (
    WorkloadError,
    load_workload,
    select_measured_entries,
    select_warmup_entries,
)


def _valid_manifest_dict(workload_path: str) -> dict:
    return {
        "schema_version": 1,
        "name": "w7-smoke-least-loaded",
        "model": {"id": "Qwen/Qwen2.5-3B-Instruct"},
        "topology": {"worker_count": 2},
        "policy": {"name": "least_loaded"},
        "workload": {"path": workload_path, "request_count": 3},
        "seed": 42,
        "warmup": {"request_count": 1},
        "generation": {"temperature": 0, "max_tokens": 64},
        "runner": {
            "concurrency": 2,
            "gateway_url": "http://127.0.0.1:8080/",
            "request_timeout_s": 120,
        },
    }


def _write_workload(path: Path, count: int = 4) -> Path:
    lines = [
        json.dumps({"messages": [{"role": "user", "content": f"Prompt number {i}"}]})
        for i in range(count)
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_valid_manifest_loads_and_normalizes_deterministically(tmp_path: Path):
    workload_file = _write_workload(tmp_path / "smoke.jsonl", count=4)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(
        json.dumps(_valid_manifest_dict("smoke.jsonl")), encoding="utf-8"
    )

    cfg, resolved_workload = load_manifest(manifest_file, base_dir=tmp_path)
    assert isinstance(cfg, ExperimentConfig)
    assert cfg.schema_version == 1
    assert cfg.policy.name == "least_loaded"
    assert cfg.runner.gateway_url == "http://127.0.0.1:8080"
    assert resolved_workload == workload_file.resolve()

    normalized_a = json.dumps(cfg.to_normalized_dict(), sort_keys=False)
    normalized_b = json.dumps(
        ExperimentConfig.model_validate(cfg.to_normalized_dict()).to_normalized_dict(),
        sort_keys=False,
    )
    assert normalized_a == normalized_b


@pytest.mark.parametrize("bad_version", [0, 2, "1", True, None])
def test_rejects_unsupported_schema_version(tmp_path: Path, bad_version):
    _write_workload(tmp_path / "smoke.jsonl")
    data = _valid_manifest_dict("smoke.jsonl")
    data["schema_version"] = bad_version
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ExperimentConfigError):
        load_manifest(manifest_file, base_dir=tmp_path)


@pytest.mark.parametrize("bad_policy", ["random", "cache_aware", "single_worker", "", 123])
def test_rejects_invalid_policy(tmp_path: Path, bad_policy):
    _write_workload(tmp_path / "smoke.jsonl")
    data = _valid_manifest_dict("smoke.jsonl")
    data["policy"]["name"] = bad_policy
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ExperimentConfigError):
        load_manifest(manifest_file, base_dir=tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda d: d["topology"].update({"worker_count": 0}),
        lambda d: d["topology"].update({"worker_count": True}),
        lambda d: d["workload"].update({"request_count": 0}),
        lambda d: d["warmup"].update({"request_count": -1}),
        lambda d: d["runner"].update({"concurrency": 0}),
        lambda d: d["runner"].update({"request_timeout_s": 0}),
        lambda d: d["generation"].update({"max_tokens": 0}),
        lambda d: d["generation"].update({"temperature": -0.1}),
        lambda d: d["runner"].update({"gateway_url": "ftp://127.0.0.1:8080"}),
        lambda d: d.update({"unknown_top_level": "disallowed"}),
        lambda d: d["runner"].update({"extra_flag": True}),
        lambda d: d.update({"name": "../escape"}),
    ],
)
def test_rejects_invalid_counts_bounds_and_unknown_fields(tmp_path: Path, mutation):
    _write_workload(tmp_path / "smoke.jsonl")
    data = _valid_manifest_dict("smoke.jsonl")
    mutation(data)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(ExperimentConfigError):
        load_manifest(manifest_file, base_dir=tmp_path)


def test_warmup_zero_is_allowed(tmp_path: Path):
    _write_workload(tmp_path / "smoke.jsonl")
    data = _valid_manifest_dict("smoke.jsonl")
    data["warmup"]["request_count"] = 0
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(data), encoding="utf-8")

    cfg, _ = load_manifest(manifest_file, base_dir=tmp_path)
    assert cfg.warmup.request_count == 0


def test_missing_manifest_or_workload_or_traversal_rejected(tmp_path: Path):
    with pytest.raises(ExperimentConfigError):
        load_manifest(tmp_path / "missing.json", base_dir=tmp_path)

    # Missing workload file
    data = _valid_manifest_dict("does_not_exist.jsonl")
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ExperimentConfigError):
        load_manifest(manifest_file, base_dir=tmp_path)

    # Path traversal outside base_dir
    outside = tmp_path / "outside.jsonl"
    _write_workload(outside)
    sub = tmp_path / "sub"
    sub.mkdir()
    traversal_manifest = sub / "manifest.json"
    traversal_manifest.write_text(
        json.dumps(_valid_manifest_dict("../outside.jsonl")), encoding="utf-8"
    )
    with pytest.raises(ExperimentConfigError, match="escape|outside|traversal"):
        load_manifest(traversal_manifest, base_dir=sub)


def test_workload_loader_malformed_jsonl_reports_line_number(tmp_path: Path):
    bad_file = tmp_path / "bad.jsonl"
    bad_file.write_text(
        '{"messages": [{"role": "user", "content": "ok"}]}\n'
        '{not valid json}\n',
        encoding="utf-8",
    )
    with pytest.raises(WorkloadError, match="line 2"):
        load_workload(bad_file, min_entries=1)


@pytest.mark.parametrize(
    "bad_line",
    [
        '{"messages": []}',
        '{"messages": [{"role": "user", "content": "   "}]}',
        '{"messages": [{"role": "", "content": "hello"}]}',
        '{"prompt": "wrong schema"}',
        "[]",
    ],
)
def test_workload_loader_rejects_empty_or_invalid_messages(tmp_path: Path, bad_line: str):
    bad_file = tmp_path / "invalid.jsonl"
    bad_file.write_text(bad_line + "\n", encoding="utf-8")
    with pytest.raises(WorkloadError, match="line 1"):
        load_workload(bad_file, min_entries=1)


def test_workload_loader_rejects_insufficient_entries_and_orders_deterministically(tmp_path: Path):
    wl = _write_workload(tmp_path / "four.jsonl", count=4)
    with pytest.raises(WorkloadError, match="at least 5"):
        load_workload(wl, min_entries=5)

    entries = load_workload(wl, min_entries=3)
    assert len(entries) == 4
    order_a = [e.messages[0]["content"] for e in select_measured_entries(entries, 3, seed=42)]
    order_b = [e.messages[0]["content"] for e in select_measured_entries(entries, 3, seed=42)]
    assert order_a == order_b
    assert len(order_a) == 3

    warmup = select_warmup_entries(entries, 2)
    assert len(warmup) == 2
    assert warmup[0].messages == entries[0].messages
