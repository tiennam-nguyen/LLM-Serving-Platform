"""W8 CPU gate: six real HTTP runs, independent raw validation and exported figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from infra.scripts.run_week8_campaign import execute_campaign  # noqa: E402
from llm_serving_platform.benchmark.artifacts import (  # noqa: E402
    write_json_atomic,
    write_text_atomic,
)
from llm_serving_platform.benchmark.campaign import (  # noqa: E402
    generate_workload,
    prepare_campaign,
)
from llm_serving_platform.benchmark.summary import compute_summary_v2, read_raw_jsonl  # noqa: E402


def _verify(output: Path) -> dict:
    from prometheus_client.parser import text_string_to_metric_families

    protocol_path = prepare_campaign(
        manifests_dir=output / "manifests",
        workloads_dir=output / "workloads",
        model="cpu-mock-w8",
        measurement_window_ms=1000,
    )
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    # This shortened matrix is a functional check, not the calibrated GPU campaign.
    selected = []
    for row in protocol["runs"]:
        manifest_path = protocol_path.parent / row["manifest"]
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["comparison"]["load_level"] != "low" or row["repetition"] != 1:
            continue
        workload_path = output / "workloads" / (manifest["workload"]["profile"] + "-smoke.jsonl")
        entries = generate_workload(
            manifest["workload"]["profile"],
            request_count=4,
            measurement_window_ms=400,
            seed=42,
        )
        text = "".join(json.dumps(entry, separators=(",", ":")) + "\n" for entry in entries)
        write_text_atomic(workload_path, text)
        manifest["workload"].update(path=str(workload_path.resolve()), request_count=4)
        manifest["warmup"]["request_count"] = 1
        manifest["generation"]["max_tokens"] = 8
        manifest["load"]["measurement_window_ms"] = 400
        manifest["runner"].update(max_dispatch_lag_ms=1000, request_timeout_s=10)
        write_json_atomic(manifest_path, manifest)
        selected.append({**row, "workload_sha256": hashlib.sha256(text.encode()).hexdigest()})
    protocol.update(name="week8-cpu-functional-smoke", runs=selected)
    write_json_atomic(protocol_path, protocol)
    report = execute_campaign(protocol_path, output_dir=output / "executed", no_gpu=True)
    assert report["run_status"] == "completed" and len(report["runs"]) == 6
    snapshots = {}
    for run in report["runs"]:
        path = Path(run["run_dir"])
        summary = compute_summary_v2(run_dir=path)
        assert summary["run_status"] == "completed", summary
        assert summary["validation"]["valid"], summary
        assert summary["requests"]["both_completed"] == 4
        assert not summary["validation"]["comparison_eligible"]
        assert len(read_raw_jsonl(path / "requests.jsonl")) == 4
        assert len(read_raw_jsonl(path / "traces.jsonl")) == 4
        for name in ("metrics.before.prom", "metrics.prom"):
            assert list(text_string_to_metric_families((path / name).read_text()))
        for file in path.iterdir():
            if file.is_file():
                snapshots[file] = file.read_bytes()
    subprocess.run(
        [
            sys.executable,
            "-m",
            "llm_serving_platform.benchmark",
            "analyze",
            "--runs-root",
            str(output / "executed/runs"),
            "--output-dir",
            str(output / "analysis"),
            "--protocol",
            str(protocol_path),
            "--figures",
        ],
        cwd=ROOT,
        check=True,
    )
    analysis = json.loads((output / "analysis/analysis.json").read_text(encoding="utf-8"))
    assert all(pair["matched"] for pair in analysis["pair_checks"])
    assert not analysis["eligible_for_policy_comparison"]
    assert len(list((output / "analysis/figures").iterdir())) == 18
    assert all(file.read_bytes() == content for file, content in snapshots.items())
    validation = {
        "valid": True,
        "test_mode": "cpu_mock",
        "runs": 6,
        "measured_requests": 24,
        "figures": 18,
        "gpu_performance_verified": False,
    }
    write_json_atomic(output / "verification.json", validation)
    return validation


def run_week8_e2e_gate(output_dir: Path | None = None) -> dict:
    if output_dir is not None:
        if output_dir.exists():
            raise FileExistsError("Use a new verification output directory")
        return _verify(output_dir.resolve())
    with tempfile.TemporaryDirectory(prefix="week8-cpu-") as temporary:
        return _verify(Path(temporary))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(run_week8_e2e_gate(args.output_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
