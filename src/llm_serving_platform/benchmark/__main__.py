"""CLI for versioned experiment execution, controlled campaigns and raw analysis."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from llm_serving_platform.benchmark.config import ExperimentConfigError
from llm_serving_platform.benchmark.runner import ExperimentRunner
from llm_serving_platform.benchmark.workload import WorkloadError


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m llm_serving_platform.benchmark",
        description="Run and analyze versioned LLM serving experiments over the Gateway HTTP API.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser(
        "run",
        help="Execute an experiment manifest and write canonical run artifacts.",
    )
    run_parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="Path to a JSON experiment manifest (schema_version=1 or 2).",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Root directory where <run_id>/ artifacts are created (default: experiments/runs).",
    )
    run_parser.add_argument(
        "--gateway-url",
        default=None,
        help="Optional Gateway URL override (takes precedence over manifest runner.gateway_url).",
    )
    run_parser.add_argument(
        "--run-id",
        default=None,
        help="Optional explicit run identifier (must not already exist).",
    )
    prepare = subparsers.add_parser(
        "prepare", help="Materialize the controlled W8 campaign without running it."
    )
    prepare.add_argument("--manifests-dir", type=Path, required=True)
    prepare.add_argument("--workloads-dir", type=Path, required=True)
    prepare.add_argument("--window-seconds", type=float, default=20)
    prepare.add_argument("--repetitions", type=int, default=3)
    prepare.add_argument("--seed", type=int, default=42)
    prepare.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    prepare.add_argument("--suite", choices=("legacy", "realistic", "stress"), default="legacy")
    prepare.add_argument(
        "--tokenizer-path",
        type=Path,
        help="Optional cached model snapshot; requires Transformers, not CUDA.",
    )
    prepare.add_argument(
        "--tolerances", type=Path, help="Predeclared repeated-pilot tolerance JSON."
    )
    analysis = subparsers.add_parser(
        "analyze", help="Read raw W8 runs into a new derived directory."
    )
    inputs = analysis.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--run-dir", type=Path, action="append")
    inputs.add_argument("--runs-root", type=Path)
    analysis.add_argument("--output-dir", type=Path, required=True)
    analysis.add_argument("--protocol", type=Path)
    analysis.add_argument(
        "--figures", action="store_true", help="Export CPU plots; requires .[analysis]."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"prepare", "analyze"}:
        try:
            if args.command == "prepare":
                from llm_serving_platform.benchmark.campaign import prepare_campaign
                from llm_serving_platform.benchmark.scenarios import (
                    REALISTIC_PROFILES,
                    load_chat_token_counter,
                )

                options = {}
                if args.suite != "legacy":
                    options["profiles"] = REALISTIC_PROFILES + (
                        ("microburst_mixed",) if args.suite == "stress" else ()
                    )
                if args.tokenizer_path:
                    options["token_counter"] = load_chat_token_counter(str(args.tokenizer_path))
                    options["tokenizer_metadata"] = {
                        "model_id": args.model,
                        "snapshot_revision": args.tokenizer_path.name,
                    }

                path = prepare_campaign(
                    manifests_dir=args.manifests_dir,
                    workloads_dir=args.workloads_dir,
                    measurement_window_ms=args.window_seconds * 1000,
                    repetitions=args.repetitions,
                    seed=args.seed,
                    model=args.model,
                    tolerances=json.loads(args.tolerances.read_text(encoding="utf-8"))
                    if args.tolerances
                    else None,
                    **options,
                )
                print(json.dumps({"campaign": str(path)}))
            else:
                from llm_serving_platform.benchmark.analysis import analyze_runs

                sources = args.run_dir or sorted(
                    p.parent for p in args.runs_root.rglob("manifest.json")
                )
                report = analyze_runs(
                    sources, output_dir=args.output_dir, protocol_path=args.protocol
                )
                figures = []
                if args.figures:
                    from llm_serving_platform.benchmark.plots import plot_analysis

                    figures = plot_analysis(
                        args.output_dir / "analysis.json", output_dir=args.output_dir / "figures"
                    )
                print(
                    json.dumps(
                        {
                            "runs": len(report["runs"]),
                            "analysis": str(args.output_dir),
                            "figures": [str(p) for p in figures],
                            "eligible_for_policy_comparison": report[
                                "eligible_for_policy_comparison"
                            ],
                        }
                    )
                )
            return 0
        except ModuleNotFoundError:
            message = (
                "Tokenizer preparation requires Transformers and a cached model snapshot."
                if args.command == "prepare" and args.tokenizer_path
                else "Plotting requires: pip install -e '.[analysis]'"
            )
            print(f"[error] {message}", file=sys.stderr)
            return 2
        except (ValueError, OSError, KeyError) as exc:
            print(f"[error] {type(exc).__name__}: {exc}", file=sys.stderr)
            return 2

    if args.command == "run":
        runner = ExperimentRunner(runs_root=args.output_dir)
        try:
            result = asyncio.run(
                runner.run_from_manifest(
                    args.manifest,
                    gateway_url_override=args.gateway_url,
                    run_id=args.run_id,
                )
            )
        except (ExperimentConfigError, WorkloadError, ValueError) as exc:
            print(f"[error] Manifest/workload validation failed: {exc}", file=sys.stderr)
            return 2
        except FileExistsError as exc:
            print(f"[error] Run directory already exists: {exc}", file=sys.stderr)
            return 2

        output = {
            "run_id": result.run_id,
            "run_dir": str(result.run_dir),
            "run_status": result.run_status,
            "failure_type": result.failure_type,
            "requests": result.summary.get("requests"),
            "warning_counts": result.summary.get("warning_counts", {}),
        }
        print(json.dumps(output, indent=2))
        if result.summary.get("execution_semantics") == "observe_v1":
            return 1 if result.run_status == "failed" else 0
        return 0 if result.succeeded else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
