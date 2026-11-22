"""CLI entrypoint for reproducible Week 7 experiment execution."""

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
        description="Run reproducible Week 7 LLM serving experiments over the Gateway HTTP API.",
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
        help="Path to the JSON experiment manifest (schema_version=1).",
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
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

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
        except (ExperimentConfigError, WorkloadError) as exc:
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
        }
        print(json.dumps(output, indent=2))
        return 0 if result.succeeded else 1

    return 2


if __name__ == "__main__":
    sys.exit(main())
