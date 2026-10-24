"""Unified Test Runner for Local GPU (default + -m), No-GPU (--no-gpu), and CI (--ci).

Usage:
  1. Default (Live GPU mode + 1 default prompt):
     python infra/scripts/run_test_suite.py

  2. Live GPU mode with custom prompt (-m):
     python infra/scripts/run_test_suite.py -m "Giải thích ngắn gọn PagedAttention trong 2 câu."

  3. No-GPU mode (runs unit/integration tests + mock E2E without GPU, no prompt needed):
     python infra/scripts/run_test_suite.py --no-gpu

  4. Full CI E2E mode (Backend --no-gpu + Frontend lint/typecheck/vitest/build/Playwright E2E):
     python infra/scripts/run_test_suite.py --no-gpu --with-frontend
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_PROMPT = "Giải thích ngắn gọn cơ chế Continuous Batching trong 2 câu."


def _run_cmd(cmd: list[str], cwd: Path = REPO_ROOT, title: str = "") -> None:
    print("\n" + "=" * 80)
    print(f"[*] {title or 'Running'}: {' '.join(cmd)}")
    print("=" * 80)
    result = subprocess.run(cmd, cwd=str(cwd), check=False)
    if result.returncode != 0:
        print(f"[-] FAILED (exit code {result.returncode}): {' '.join(cmd)}", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified Test Runner (GPU default with optional -m, or --no-gpu mock mode)."
    )
    parser.add_argument(
        "--no-gpu",
        "--mock",
        dest="no_gpu",
        action="store_true",
        help="Run in CPU/mock mode without requiring a live vLLM GPU server.",
    )
    parser.add_argument(
        "-m",
        "--prompt",
        default=None,
        help="Custom prompt string for Live GPU E2E verification (ignored when --no-gpu is set).",
    )
    parser.add_argument(
        "--gateway-url",
        default="http://127.0.0.1:8080",
        help="Gateway URL when running in live GPU mode (default: http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--with-frontend",
        "--ci",
        dest="with_frontend",
        action="store_true",
        help="Also run the full Frontend + Playwright browser E2E suite in apps/web.",
    )
    args = parser.parse_args()

    if args.no_gpu and args.prompt is not None:
        print(
            "[!] Note: -m/--prompt is ignored in --no-gpu mode because MockInferenceWorker "
            "uses deterministic canned chunks."
        )

    effective_prompt = args.prompt or DEFAULT_PROMPT
    py = sys.executable

    # 1. Ruff lint & format check
    _run_cmd([py, "-m", "ruff", "check", "."], title="Step 1/4: Ruff Linter")
    _run_cmd([py, "-m", "ruff", "format", "--check", "."], title="Step 2/4: Ruff Formatter Check")

    # 2. Pytest suite
    if args.no_gpu:
        pytest_cmd = [py, "-m", "pytest", "--no-gpu"]
    else:
        pytest_cmd = [py, "-m", "pytest", "--prompt", effective_prompt]
    mode_label = "NO-GPU" if args.no_gpu else "LIVE-GPU"
    _run_cmd(pytest_cmd, title=f"Step 3/4: Pytest Suite ({mode_label})")

    # 3. Week 3 E2E Trace & Stream Verification
    if args.no_gpu:
        e2e_cmd = [py, "infra/scripts/verify_week3_e2e.py", "--no-gpu"]
    else:
        e2e_cmd = [
            py,
            "infra/scripts/verify_week3_e2e.py",
            "-m",
            effective_prompt,
            "--gateway-url",
            args.gateway_url,
        ]
    _run_cmd(
        e2e_cmd,
        title=f"Step 4/4: Week 3 E2E Trace Correlation ({mode_label})",
    )

    # 4. Optional Frontend + Playwright E2E Suite
    if args.with_frontend:
        npm_bin = shutil.which("npm") or "npm"
        web_dir = REPO_ROOT / "apps" / "web"
        _run_cmd([npm_bin, "run", "lint"], cwd=web_dir, title="Frontend 1/5: ESLint")
        _run_cmd(
            [npm_bin, "run", "typecheck"],
            cwd=web_dir,
            title="Frontend 2/5: TypeScript Typecheck",
        )
        _run_cmd(
            [npm_bin, "test"], cwd=web_dir, title="Frontend 3/5: Vitest Unit & Component Suite"
        )
        _run_cmd(
            [npm_bin, "run", "build"], cwd=web_dir, title="Frontend 4/5: Next.js Production Build"
        )
        _run_cmd(
            [npm_bin, "run", "test:browser"],
            cwd=web_dir,
            title="Frontend 5/5: Playwright Browser E2E",
        )

    print("\n" + "#" * 80)
    if args.no_gpu:
        print(">>> ALL CHECKS PASSED SUCCESSFULLY (Mode: NO-GPU / CI) <<<")
    else:
        print(f">>> ALL CHECKS PASSED (Mode: LIVE GPU, Prompt: {effective_prompt!r}) <<<")
    print("#" * 80 + "\n")


if __name__ == "__main__":
    main()
