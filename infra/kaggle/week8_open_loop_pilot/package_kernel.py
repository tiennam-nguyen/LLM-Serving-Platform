"""Package a clean Week 8 pilot revision as one private Kaggle script kernel."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
import subprocess
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ORIGIN = "https://github.com/tiennam-nguyen/LLM-Serving-Platform.git"
KERNEL_NAME = "llm-serving-week-8-open-loop-pilot"


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True, timeout=120).strip()


def package(
    output_dir: Path,
    owner: str,
    base: str,
    model: str,
    vllm_version: str | None,
    *,
    extra_launcher_args: list[str] | None = None,
    launcher_script: str = "infra/kaggle/week8_open_loop_pilot/run.py",
    launcher_arguments: list[str] | None = None,
    kernel_name: str = KERNEL_NAME,
    title: str = "LLM Serving Week 8 Open-Loop Pilot",
    timeout_seconds: int = 3550,
) -> None:
    if git("status", "--porcelain"):
        raise RuntimeError(
            "Package a clean checkout so the script matches its recorded Git revision"
        )
    branch = git("branch", "--show-current")
    if not branch or branch == "main":
        raise RuntimeError("Package the Week 8 feature branch (not main)")
    revision = git("rev-parse", "HEAD")
    base_rev = git("rev-parse", f"{base}^{{commit}}")
    if base_rev == revision or git("merge-base", base_rev, revision) != base_rev:
        raise RuntimeError("The public base must be a strict ancestor of the packaged revision")

    with tempfile.TemporaryDirectory() as temp:
        bundle = Path(temp) / "week8.bundle"
        git("bundle", "create", str(bundle), "HEAD", f"^{base_rev}")
        data = bundle.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    encoded = "\n".join(
        f"    {line!r}" for line in textwrap.wrap(base64.b64encode(data).decode(), 96)
    )
    launcher_args = ["--bootstrap", "--revision", revision, "--model", model]
    if vllm_version:
        launcher_args.extend(("--vllm-version", vllm_version))
    if extra_launcher_args:
        launcher_args.extend(extra_launcher_args)
    if launcher_arguments is not None:
        launcher_args = launcher_arguments

    script = f'''"""Restore the committed Week 8 pilot checkout and run its dual-T4 launcher."""

import base64
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ORIGIN = {ORIGIN!r}
BASE = {base_rev!r}
BRANCH = {branch!r}
REVISION = {revision!r}
BUNDLE_SHA256 = {digest!r}
LAUNCHER_ARGS = {launcher_args!r}
BUNDLE = (
{encoded}
)


def main():
    data = base64.b64decode(BUNDLE, validate=True)
    if hashlib.sha256(data).hexdigest() != BUNDLE_SHA256:
        raise RuntimeError("Kaggle source bundle failed its SHA-256 check")
    with tempfile.TemporaryDirectory(prefix="week8-kaggle-") as temp:
        root = Path(temp)
        bundle = root / "week8.bundle"
        bundle.write_bytes(data)
        checkout = root / "checkout"
        subprocess.run(["git", "init", "-q", str(checkout)], check=True, timeout=30)

        def git(*args):
            return subprocess.check_output(
                ("git", *args), cwd=checkout, text=True, timeout=120,
            ).strip()

        git("fetch", "--no-tags", ORIGIN, BASE)
        git("bundle", "verify", str(bundle))
        git("fetch", "--no-tags", str(bundle), "HEAD")
        git("checkout", "-b", BRANCH, "FETCH_HEAD")
        if git("rev-parse", "HEAD") != REVISION or git("status", "--porcelain"):
            raise RuntimeError("Restored source does not match the expected clean revision")
        print(f"Restored Week 8 revision {{REVISION}}", flush=True)
        subprocess.run(
            [
                sys.executable,
                {launcher_script!r},
                *LAUNCHER_ARGS,
            ],
            cwd=checkout,
            check=True,
            timeout={timeout_seconds!r},
        )


if __name__ == "__main__":
    main()
'''
    compile(script, "run.py", "exec")
    metadata = {
        "id": f"{owner}/{kernel_name}",
        "title": title,
        "code_file": "run.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "run.py").write_text(script, encoding="utf-8", newline="\n")
    (output_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Packaged {revision} for private kernel {metadata['id']}")
    print(f"Bundle SHA-256: {digest}")
    print(f"Submitted script SHA-256: {hashlib.sha256(script.encode()).hexdigest()}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, help="Authenticated Kaggle account slug")
    parser.add_argument("--base", required=True, help="Public ancestor commit reachable on origin")
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-version", default="0.33.0")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--coarse-rates", default=None)
    parser.add_argument("--window-ms", type=float, default=None)
    parser.add_argument("--max-in-flight", type=int, default=None)
    parser.add_argument("--max-dispatch-lag-ms", type=float, default=None)
    parser.add_argument("--no-refine", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]+", args.owner):
        parser.error("Expected a Kaggle account slug")
    extra: list[str] = []
    if args.coarse_rates:
        extra.extend(("--coarse-rates", args.coarse_rates))
    if args.window_ms is not None:
        extra.extend(("--window-ms", str(args.window_ms)))
    if args.max_in_flight is not None:
        extra.extend(("--max-in-flight", str(args.max_in_flight)))
    if args.max_dispatch_lag_ms is not None:
        extra.extend(("--max-dispatch-lag-ms", str(args.max_dispatch_lag_ms)))
    if args.no_refine:
        extra.append("--no-refine")
    package(
        args.output_dir,
        args.owner,
        args.base,
        args.model,
        args.vllm_version,
        extra_launcher_args=extra,
    )


if __name__ == "__main__":
    main()
