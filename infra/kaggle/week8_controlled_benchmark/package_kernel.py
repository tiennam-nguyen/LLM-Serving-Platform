"""Package a clean source revision into a private, two-T4 Kaggle experiment."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from infra.kaggle.week8_open_loop_pilot.package_kernel import git, package  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-name", default="llm-serving-week8-controlled-benchmark")
    parser.add_argument("--vllm-version", default="0.33.0")
    args = parser.parse_args()
    if not all(re.fullmatch(r"[A-Za-z0-9-]+", x) for x in (args.owner, args.kernel_name)):
        parser.error("Expected Kaggle owner and kernel slugs")
    package(
        args.output_dir,
        args.owner,
        args.base,
        "Qwen/Qwen2.5-3B-Instruct",
        args.vllm_version,
        launcher_script="infra/kaggle/week8_controlled_benchmark/run.py",
        launcher_arguments=[
            "--bootstrap",
            "--revision",
            git("rev-parse", "HEAD"),
            "--vllm-version",
            args.vllm_version,
        ],
        kernel_name=args.kernel_name,
        title="LLM Serving W8 Calibration and Controlled Benchmark",
        timeout_seconds=39600,
    )


if __name__ == "__main__":
    main()
