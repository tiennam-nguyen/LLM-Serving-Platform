"""Package an exact local Week 6 revision as one private Kaggle script kernel."""

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
KERNEL_NAME = "llm-serving-week6-routing-evidence"


def git(*args):
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True, timeout=120).strip()


def package(output_dir, owner, base, model, vllm_version, extended):
    if git("status", "--porcelain"):
        raise RuntimeError(
            "Package a clean checkout so the script matches its recorded Git revision"
        )
    branch = git("branch", "--show-current")
    if not branch or branch == "main":
        raise RuntimeError("Package the Week 6 experiment branch")
    revision = git("rev-parse", "HEAD")
    base = git("rev-parse", f"{base}^{{commit}}")
    if base == revision or git("merge-base", base, revision) != base:
        raise RuntimeError("The public base must be a strict ancestor of the packaged revision")

    with tempfile.TemporaryDirectory() as temp:
        bundle = Path(temp) / "week6.bundle"
        git("bundle", "create", str(bundle), "HEAD", f"^{base}")
        data = bundle.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    # Split the literal into short source lines accepted by Kaggle's script transport.
    encoded = "\n".join(
        f"    {line!r}" for line in textwrap.wrap(base64.b64encode(data).decode(), 96)
    )
    launcher_args = ["--bootstrap", "--revision", revision, "--model", model]
    if vllm_version:
        launcher_args.extend(("--vllm-version", vllm_version))
    if extended:
        launcher_args.append("--extended")
    script = f'''"""Restore the committed Week 6 checkout and launch its managed T4 gate."""

import base64
import hashlib
import subprocess
import sys
import tempfile
from pathlib import Path

ORIGIN = {ORIGIN!r}
BASE = {base!r}
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
    with tempfile.TemporaryDirectory(prefix="week6-kaggle-") as temp:
        root = Path(temp)
        bundle = root / "week6.bundle"
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
        print(f"Restored Week 6 revision {{REVISION}}", flush=True)
        subprocess.run(
            [
                sys.executable, "infra/kaggle/week6_routing_evidence/run.py",
                *LAUNCHER_ARGS,
            ],
            cwd=checkout, check=True, timeout=3450,
        )


if __name__ == "__main__":
    main()
'''
    compile(script, "run.py", "exec")
    metadata = {
        "id": f"{owner}/{KERNEL_NAME}",
        "title": "LLM Serving Week 6 Routing Evidence",
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
    print(f"Submitted script SHA-256: {hashlib.sha256(script.encode()).hexdigest()}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner", required=True, help="Authenticated Kaggle account slug")
    parser.add_argument(
        "--base", required=True, help="Ancestor commit reachable on the public repo"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--vllm-version")
    parser.add_argument("--extended", action="store_true")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9-]+", args.owner):
        parser.error("Expected a Kaggle account slug")
    package(args.output_dir, args.owner, args.base, args.model, args.vllm_version, args.extended)


if __name__ == "__main__":
    main()
