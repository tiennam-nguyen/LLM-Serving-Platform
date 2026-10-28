# ruff: noqa: E402, I001
"""Unified Open-Closed Verification & Cumulative Milestone Gate Runner.

Supports:
  1. Full E2E Verification (Default = Live GPU):
     python infra/scripts/verify.py

  2. Full E2E Verification without GPU (CI & CPU laptops):
     python infra/scripts/verify.py --no-gpu

  3. Partial / Component Verification (both GPU default and --no-gpu):
     python infra/scripts/verify.py --target lint [--no-gpu]
     python infra/scripts/verify.py --target unit [--no-gpu]
     python infra/scripts/verify.py --target integration [--no-gpu]
     python infra/scripts/verify.py --target w1 [--no-gpu]
     python infra/scripts/verify.py --target w2 [--no-gpu]
     python infra/scripts/verify.py --target w3 [--no-gpu]
     python infra/scripts/verify.py --target gates [--no-gpu]
     python infra/scripts/verify.py --target backend [--no-gpu]
     python infra/scripts/verify.py --target web [--no-gpu]

Open-Closed Principle (OCP) for Future Weeks (Week 4 -> Week 15):
  To add Week 4+ verification without deleting or breaking Week 1-3 gates,
  register a new GateSpec entry in MILESTONE_GATES below.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import importlib.util
import inspect
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from infra.scripts.measure_latency import stream_and_measure  # noqa: E402, I001
from infra.scripts.verify_week2_exit_gate import (  # noqa: E402, I001
    run_mock_verification,
    run_verification,
)
from infra.scripts.verify_week3_e2e import run_week3_e2e_check  # noqa: E402, I001
from llm_serving_platform.common.config import load_dotenv_file  # noqa: E402, I001

load_dotenv_file()

DEFAULT_PROMPT = "Giải thích ngắn gọn cơ chế Continuous Batching trong 2 câu."
_HAS_REAL_PYTEST = importlib.util.find_spec("pytest") is not None
_HAS_REAL_RUFF = importlib.util.find_spec("ruff") is not None


@dataclass(frozen=True)
class VerifyContext:
    """Execution parameters shared across all verification stages and milestone gates."""

    no_gpu: bool
    prompt: str
    gateway_url: str
    vllm_url: str
    model: str
    w2_requests: int = 10


@dataclass(frozen=True)
class GateSpec:
    """Declarative specification for a cumulative weekly milestone gate."""

    gate_id: str
    title: str
    runner: Callable[[VerifyContext], bool]


def _banner(title: str) -> None:
    print("\n" + "=" * 84)
    print(f"[*] {title}")
    print("=" * 84)


def _run_subprocess(
    cmd: list[str], cwd: Path = REPO_ROOT, env: dict[str, str] | None = None
) -> None:
    merged_env = os.environ.copy()
    py_path = f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'src'}"
    existing = merged_env.get("PYTHONPATH", "")
    merged_env["PYTHONPATH"] = f"{py_path}{os.pathsep}{existing}" if existing else py_path
    if env:
        merged_env.update(env)
    result = subprocess.run(cmd, cwd=str(cwd), env=merged_env, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed (exit {result.returncode}): {' '.join(cmd)}")


# ==============================================================================
# Stage 1: Linting & Static Verification
# ==============================================================================
def run_lint_stage(ctx: VerifyContext) -> bool:
    """Run Ruff linter (or AST bytecode compilation check if Ruff is absent)."""
    _banner("STAGE [lint]: Python Linter & Syntax Check")
    py = sys.executable
    if _HAS_REAL_RUFF:
        _run_subprocess([py, "-m", "ruff", "check", "."])
    else:
        print("[i] Ruff not installed in current Python env; running AST py_compile check...")
        py_files = (
            list((REPO_ROOT / "src").rglob("*.py"))
            + list((REPO_ROOT / "tests").rglob("*.py"))
            + list((REPO_ROOT / "infra" / "scripts").rglob("*.py"))
        )
        for fpath in py_files:
            py_compile.compile(str(fpath), doraise=True)
        print(f"[+] AST compilation succeeded for {len(py_files)} Python files.")
    return True


# ==============================================================================
# Stage 2 & 3: Unit & Integration Tests (Pytest + Zero-Dep Fallback Runner)
# ==============================================================================
def _install_minimal_pytest_shim() -> None:
    """Install a lightweight pytest module shim if running in a VM without pytest installed."""
    if "pytest" in sys.modules or _HAS_REAL_PYTEST:
        return

    shim = types.ModuleType("pytest")

    class _Skipped(Exception):
        pass

    class _Failed(AssertionError):
        pass

    class _RaisesContext:
        def __init__(self, expected_exc: type[BaseException]) -> None:
            self.expected_exc = expected_exc
            self.value: Any = None

        def __enter__(self) -> _RaisesContext:
            return self

        def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
            if exc_type is None:
                raise AssertionError(f"Expected {self.expected_exc.__name__} to be raised.")
            if issubclass(exc_type, self.expected_exc):
                self.value = exc_val
                return True
            return False

    class _MarkNamespace:
        def asyncio(self, fn: Any) -> Any:
            return fn

        def parametrize(self, argnames: str, argvalues: list[Any]) -> Callable[[Any], Any]:
            def decorator(fn: Any) -> Any:
                fn.__parametrize__ = (argnames, argvalues)
                return fn

            return decorator

        def skip(self, reason: str = "") -> Callable[[Any], Any]:
            def decorator(fn: Any) -> Any:
                fn.__skip_reason__ = reason
                return fn

            return decorator

    def fixture(fn: Any = None, **_kwargs: Any) -> Any:
        if fn is None:
            return lambda f: f
        fn.__is_fixture__ = True
        return fn

    def raises(expected_exc: type[BaseException]) -> _RaisesContext:
        return _RaisesContext(expected_exc)

    def skip(reason: str = "") -> None:
        raise _Skipped(reason)

    def fail(reason: str = "") -> None:
        raise _Failed(reason)

    class MonkeyPatch:
        def __init__(self) -> None:
            self._env_backup: dict[str, str | None] = {}

        def setenv(self, key: str, value: str) -> None:
            if key not in self._env_backup:
                self._env_backup[key] = os.environ.get(key)
            os.environ[key] = value

        def delenv(self, key: str, raising: bool = True) -> None:
            if key not in self._env_backup:
                self._env_backup[key] = os.environ.get(key)
            if key in os.environ:
                del os.environ[key]
            elif raising:
                raise KeyError(key)

        def undo(self) -> None:
            for k, v in self._env_backup.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

    class CaptureFixture:
        def __class_getitem__(cls, _item: Any) -> Any:
            return cls

    shim.mark = _MarkNamespace()  # type: ignore[attr-defined]
    shim.fixture = fixture  # type: ignore[attr-defined]
    shim.raises = raises  # type: ignore[attr-defined]
    shim.skip = skip  # type: ignore[attr-defined]
    shim.fail = fail  # type: ignore[attr-defined]
    shim.MonkeyPatch = MonkeyPatch  # type: ignore[attr-defined]
    shim.CaptureFixture = CaptureFixture  # type: ignore[attr-defined]
    shim.Parser = object  # type: ignore[attr-defined]
    shim.FixtureRequest = object  # type: ignore[attr-defined]
    shim.Config = object  # type: ignore[attr-defined]
    shim.Item = object  # type: ignore[attr-defined]
    shim._Skipped = _Skipped  # type: ignore[attr-defined]
    sys.modules["pytest"] = shim


def _run_pytest_or_fallback(subdir: str, ctx: VerifyContext) -> bool:
    """Execute tests in tests/<subdir> via pytest if available, or built-in runner otherwise."""
    mode_tag = "NO-GPU" if ctx.no_gpu else "LIVE-GPU"
    _banner(f"STAGE [{subdir}]: Running tests/{subdir} ({mode_tag})")
    py = sys.executable
    target_dir = REPO_ROOT / "tests" / subdir

    if _HAS_REAL_PYTEST:
        cmd = [py, "-m", "pytest", str(target_dir)]
        if ctx.no_gpu:
            cmd.append("--no-gpu")
        else:
            cmd.extend(["--prompt", ctx.prompt])
        env = {"REQUIRE_LIVE_GPU": "0" if ctx.no_gpu else "1"}
        _run_subprocess(cmd, env=env)
        return True

    _install_minimal_pytest_shim()
    import io
    from contextlib import redirect_stderr, redirect_stdout

    import pytest as _pt

    passed = 0
    skipped = 0
    for test_file in sorted(target_dir.glob("test_*.py")):
        spec = importlib.util.spec_from_file_location(test_file.stem, test_file)
        if spec is None or spec.loader is None:
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        fixtures: dict[str, Callable[..., Any]] = {}
        for name, obj in inspect.getmembers(mod, inspect.isfunction):
            if getattr(obj, "__is_fixture__", False):
                fixtures[name] = obj

        for name, func in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_") or getattr(func, "__is_fixture__", False):
                continue

            param_spec = getattr(func, "__parametrize__", None)
            call_variants: list[dict[str, Any]] = [{}]
            if param_spec is not None:
                argname, argvalues = param_spec
                call_variants = [{argname: val} for val in argvalues]

            for variant in call_variants:
                mp = _pt.MonkeyPatch()
                tmp_dir = Path(tempfile.mkdtemp(prefix="llm_test_"))
                out_buf = io.StringIO()
                err_buf = io.StringIO()

                class _Capsys:
                    def readouterr(self) -> Any:
                        return types.SimpleNamespace(
                            out=out_buf.getvalue(), err=err_buf.getvalue()
                        )

                resolved_cache: dict[str, Any] = {}

                def _resolve_arg(arg: str) -> Any:
                    if arg in resolved_cache:
                        return resolved_cache[arg]
                    if arg in variant:
                        return variant[arg]
                    if arg == "tmp_path":
                        return tmp_dir
                    if arg == "monkeypatch":
                        return mp
                    if arg == "capsys":
                        return _Capsys()
                    if arg == "no_gpu":
                        return ctx.no_gpu
                    if arg == "prompt_text":
                        return ctx.prompt
                    if arg in fixtures:
                        f_fn = fixtures[arg]
                        f_sig = inspect.signature(f_fn)
                        f_kwargs = {p: _resolve_arg(p) for p in f_sig.parameters}
                        val = f_fn(**f_kwargs)
                        resolved_cache[arg] = val
                        return val
                    raise KeyError(f"Unknown fixture: {arg}")

                sig = inspect.signature(func)
                kwargs = {p: _resolve_arg(p) for p in sig.parameters}
                try:
                    with redirect_stdout(out_buf), redirect_stderr(err_buf):
                        if inspect.iscoroutinefunction(func):
                            asyncio.run(func(**kwargs))
                        else:
                            func(**kwargs)
                    passed += 1
                    print(f"  [PASS] {test_file.name}::{name}")
                except _pt._Skipped as sk:  # type: ignore[attr-defined]
                    skipped += 1
                    print(f"  [SKIP] {test_file.name}::{name} ({sk})")
                finally:
                    mp.undo()
                    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"[+] Fallback test runner completed: {passed} passed, {skipped} skipped.")
    return True


# ==============================================================================
# Stage 4: Cumulative Milestone Gates (Week 1 -> Week 2 -> Week 3 -> Week 4+)
# ==============================================================================
def gate_week1_streaming_latency(ctx: VerifyContext) -> bool:
    """Week 1 Exit Gate: Single-request streaming lifecycle & TTFT/E2E measurement."""
    mode = "NO-GPU" if ctx.no_gpu else "LIVE-GPU"
    _banner(f"GATE [w1]: Week 1 Streaming & Latency Probe ({mode})")
    rc = stream_and_measure(
        prompt=ctx.prompt,
        no_gpu=ctx.no_gpu,
        base_url=ctx.gateway_url,
        model=ctx.model,
    )
    if rc != 0:
        raise RuntimeError(f"Week 1 Gate failed with exit code {rc}")
    return True


def gate_week2_gateway_exit_gate(ctx: VerifyContext) -> bool:
    """Week 2 Exit Gate: Data-Plane Gateway multi-request streaming & error boundary."""
    mode = "NO-GPU" if ctx.no_gpu else "LIVE-GPU"
    _banner(f"GATE [w2]: Week 2 Data-Plane Gateway Exit Gate ({mode}, {ctx.w2_requests} reqs)")
    if ctx.no_gpu:
        ok = asyncio.run(run_mock_verification(num_requests=ctx.w2_requests, prompt=ctx.prompt))
    else:
        ok = asyncio.run(
            run_verification(
                gateway_url=ctx.gateway_url,
                num_requests=ctx.w2_requests,
                model=ctx.model,
                concurrency=1,
                prompt=ctx.prompt,
            )
        )
    if not ok:
        raise RuntimeError("Week 2 Exit Gate failed.")
    return True


def gate_week3_e2e_trace_correlation(ctx: VerifyContext) -> bool:
    """Week 3 Exit Gate: Browser CORS, X-Request-ID, X-Worker-ID & 6-Milestone RequestTrace."""
    mode = "NO-GPU" if ctx.no_gpu else "LIVE-GPU"
    _banner(f"GATE [w3]: Week 3 E2E Trace & CORS Correlation ({mode})")
    evidence = asyncio.run(
        run_week3_e2e_check(
            gateway_url=ctx.gateway_url,
            model=ctx.model,
            prompt=ctx.prompt,
            no_gpu=ctx.no_gpu,
            output_path=None,
        )
    )
    print(
        f"  [w3] Request ID: {evidence['request_id']} | "
        f"Worker ID: {evidence['client_observed']['worker_id_header']} | "
        f"Server TTFT: {evidence['server_trace']['metrics_ms']['ttft_ms']} ms | "
        f"Server E2E: {evidence['server_trace']['metrics_ms']['e2e_ms']} ms"
    )
    if not evidence.get("passed"):
        raise RuntimeError("Week 3 Exit Gate failed.")
    return True


# ------------------------------------------------------------------------------
# OPEN-CLOSED MILESTONE REGISTRY:
# In Week 4+, append `"w4": GateSpec(...)` here without deleting w1, w2, or w3!
# ------------------------------------------------------------------------------
MILESTONE_GATES: dict[str, GateSpec] = {
    "w1": GateSpec(
        gate_id="w1",
        title="Week 1: Single-Worker Streaming & Latency Probe",
        runner=gate_week1_streaming_latency,
    ),
    "w2": GateSpec(
        gate_id="w2",
        title="Week 2: Data-Plane Gateway Multi-Request Exit Gate",
        runner=gate_week2_gateway_exit_gate,
    ),
    "w3": GateSpec(
        gate_id="w3",
        title="Week 3: E2E CORS, Worker ID & 6-Milestone Trace Correlation",
        runner=gate_week3_e2e_trace_correlation,
    ),
}


# ==============================================================================
# Stage 5: Frontend & Browser E2E Verification (apps/web)
# ==============================================================================
def run_web_stage(ctx: VerifyContext, skip_browser: bool = False) -> bool:
    """Run Frontend ESLint, TypeScript typecheck, Vitest unit tests, Next build, and Playwright."""
    mode = "NO-GPU" if ctx.no_gpu else "LIVE-GPU"
    _banner(f"STAGE [web]: Frontend & Browser E2E Suite ({mode})")
    npm_bin = shutil.which("npm") or "npm"
    web_dir = REPO_ROOT / "apps" / "web"

    node_ver_proc = subprocess.run(
        ["node", "-p", "process.versions.node.split('.')[0]"],
        capture_output=True,
        text=True,
        check=False,
    )
    node_major = int(node_ver_proc.stdout.strip() or "0") if node_ver_proc.returncode == 0 else 0
    if node_major < 20:
        print(
            f"[!] Local Node.js is v{node_major} (< 20 required by Next.js 16 / Vitest 5). "
            "Skipping npm execution in this shell (GitHub Actions CI uses Node 22)."
        )
        return True

    _run_subprocess([npm_bin, "run", "lint"], cwd=web_dir)
    _run_subprocess([npm_bin, "run", "typecheck"], cwd=web_dir)
    _run_subprocess([npm_bin, "test"], cwd=web_dir)
    _run_subprocess([npm_bin, "run", "build"], cwd=web_dir)
    if not skip_browser:
        _run_subprocess([npm_bin, "run", "test:browser"], cwd=web_dir)
    return True


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified Verification & Cumulative Gate Runner. "
            "Defaults to LIVE GPU mode; pass --no-gpu for CPU/Mock/CI mode."
        )
    )
    parser.add_argument(
        "--no-gpu",
        "--mock",
        dest="no_gpu",
        action="store_true",
        help="Run in No-GPU mode using MockInferenceWorker (CI & CPU hosts).",
    )
    target_choices = [
        "all",
        "backend",
        "lint",
        "unit",
        "integration",
        "gates",
        "web",
        *MILESTONE_GATES.keys(),
    ]
    parser.add_argument(
        "--target",
        choices=target_choices,
        default="all",
        help="Select full E2E verification ('all') or a specific component/milestone gate.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Custom prompt string for E2E verification.",
    )
    parser.add_argument(
        "--gateway-url",
        default=os.getenv("GATEWAY_URL", "http://127.0.0.1:8080"),
        help="Target Gateway URL in Live GPU mode (default: http://127.0.0.1:8080).",
    )
    parser.add_argument(
        "--vllm-url",
        default=os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000"),
        help="Target vLLM Worker URL in Live GPU mode (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct"),
        help="Model identifier (default: Qwen/Qwen2.5-0.5B-Instruct).",
    )
    parser.add_argument(
        "--w2-requests",
        type=int,
        default=10,
        help="Number of streaming requests for Week 2 Exit Gate (default: 10).",
    )
    parser.add_argument(
        "--skip-web",
        action="store_true",
        help="Skip apps/web checks when running --target all.",
    )
    parser.add_argument(
        "--skip-browser",
        action="store_true",
        help="Skip Playwright browser step inside Stage [web].",
    )
    parser.add_argument(
        "--with-frontend",
        "--ci",
        dest="with_frontend",
        action="store_true",
        help="Alias included for backward compatibility (web is included in --target all).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    ctx = VerifyContext(
        no_gpu=args.no_gpu,
        prompt=args.prompt,
        gateway_url=args.gateway_url,
        vllm_url=args.vllm_url,
        model=args.model,
        w2_requests=args.w2_requests,
    )

    mode_str = "NO-GPU (MockInferenceWorker)" if ctx.no_gpu else "LIVE-GPU (vLLM + Gateway)"
    print("#" * 84)
    print("# LLM SERVING PLATFORM - UNIFIED VERIFICATION PIPELINE")
    print(f"# Target : {args.target}")
    print(f"# Mode   : {mode_str}")
    print("#" * 84)

    target = args.target

    try:
        if target in {"all", "backend", "lint"}:
            run_lint_stage(ctx)

        if target in {"all", "backend", "unit"}:
            _run_pytest_or_fallback("unit", ctx)

        if target in {"all", "backend", "integration"}:
            _run_pytest_or_fallback("integration", ctx)

        if target in {"all", "backend", "gates"}:
            for gate_spec in MILESTONE_GATES.values():
                gate_spec.runner(ctx)
        elif target in MILESTONE_GATES:
            MILESTONE_GATES[target].runner(ctx)

        if target == "web" or (target == "all" and not args.skip_web):
            run_web_stage(ctx, skip_browser=args.skip_browser)

    except Exception as exc:
        msg = f"\n[-] VERIFICATION FAILED (target={target}, mode={mode_str}): {exc}"
        print(msg, file=sys.stderr)
        if not ctx.no_gpu:
            print(
                "    [Tip] Live GPU mode requires running vLLM (:8000) and Gateway (:8080).\n"
                "          If your machine has no GPU, add `--no-gpu` to test in CPU/mock mode.",
                file=sys.stderr,
            )
        return 1

    print("\n" + "#" * 84)
    print(f">>> ALL REQUESTED VERIFICATION CHECKS PASSED (target={target}, mode={mode_str}) <<<")
    print("#" * 84 + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
