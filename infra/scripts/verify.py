# ruff: noqa: E402, I001
"""Unified Open-Closed Verification & Cumulative Milestone Gate Runner.

Supports:
  1. Full E2E Verification (Default = CPU/Mock):
     python infra/scripts/verify.py

  2. Explicit CPU mode (backward compatible):
     python infra/scripts/verify.py --no-gpu

  Live GPU checks require --gpu and already running vLLM/Gateway services.

  3. Partial / Component Verification (CPU by default; optional --gpu):
     python infra/scripts/verify.py --target lint [--no-gpu]
     python infra/scripts/verify.py --target unit [--no-gpu]
     python infra/scripts/verify.py --target integration [--no-gpu]
     python infra/scripts/verify.py --target w1 [--no-gpu]
     python infra/scripts/verify.py --target w2 [--no-gpu]
     python infra/scripts/verify.py --target w3 [--no-gpu]
     python infra/scripts/verify.py --target w4 [--no-gpu]
     python infra/scripts/verify.py --target w6 --no-gpu
     python infra/scripts/verify.py --target w7-observability --no-gpu
     python infra/scripts/verify.py --target w7 --no-gpu
     python infra/scripts/verify.py --target w8 --no-gpu
     python infra/scripts/verify.py --target gates [--no-gpu]
     python infra/scripts/verify.py --target backend [--no-gpu]
     python infra/scripts/verify.py --target web [--no-gpu]

Open-Closed Principle (OCP) for Implemented Milestone Gates (W1, W2, W3, W4, W6, W7):
  Week 5 LAN/two-machine validation is intentionally separate from hosted CPU/mock CI.
  Register new weekly milestone gates in MILESTONE_GATES below without removing prior gates.
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
from infra.scripts.verify_week4_e2e import run_week4_e2e_check  # noqa: E402, I001
from infra.scripts.verify_week6_e2e import run_week6_e2e
from infra.scripts.verify_week7_e2e import run_week7_e2e_gate
from infra.scripts.verify_week8_e2e import run_week8_e2e_gate
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

    from datetime import datetime
    from fastapi.encoders import ENCODERS_BY_TYPE
    ENCODERS_BY_TYPE[datetime] = lambda dt: dt.isoformat().replace("+00:00", "Z")

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
            cmd.extend(["--gpu", "--prompt", ctx.prompt])
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


def gate_week4_registry_and_round_robin(ctx: VerifyContext) -> bool:
    """Week 4 Exit Gate: WorkerRegistry, HealthMonitor, Round-Robin & Unhealthy Exclusion."""
    mode = "NO-GPU" if ctx.no_gpu else "LIVE-GPU"
    _banner(f"GATE [w4]: Week 4 WorkerRegistry, Health Check & Round-Robin ({mode})")
    evidence = asyncio.run(
        run_week4_e2e_check(
            gateway_url=ctx.gateway_url,
            model=ctx.model,
            prompt=ctx.prompt,
            no_gpu=ctx.no_gpu,
            output_path=None,
        )
    )
    print(
        f"  [w4] Workers: {[w['worker_id'] for w in evidence['initial_workers']]} | "
        f"RR Sequence: {evidence['round_robin_sequence']} | "
        f"Failover: {evidence['failover_sequence']}"
    )
    if not evidence.get("passed"):
        raise RuntimeError("Week 4 Exit Gate failed.")
    return True


def gate_week6_routing_evidence(ctx: VerifyContext) -> bool:
    """Week 6 Exit Gate: routing decisions, concurrent load, and reservation cleanup."""
    _banner("GATE [w6]: Gateway Routing and Reservation Lifecycle")
    if not ctx.no_gpu:
        raise RuntimeError(
            "The Week 6 live gate needs two controlled replicas and fresh policy phases. "
            "Run infra/kaggle/week6_routing_evidence/run.py."
        )
    validation, _ = asyncio.run(
        run_week6_e2e(no_gpu=True, model=ctx.model, gateway_url=ctx.gateway_url)
    )
    if not validation["passed"]:
        raise RuntimeError(f"Week 6 no-GPU gate failed: {validation}")
    return True


def gate_week7_observability(ctx: VerifyContext) -> bool:
    """Member A's CPU contract checks for traces, metrics, and request lifecycle."""
    _banner("GATE [w7-observability]: Member A Trace, Metrics & Request Lifecycle")
    if not ctx.no_gpu:
        raise RuntimeError("The observability contract gate uses CPU workers; pass --no-gpu.")
    if not _HAS_REAL_PYTEST:
        raise RuntimeError("Install .[ci] to run the observability contract tests with pytest.")
    _run_subprocess([
        sys.executable, "-m", "pytest", "--no-gpu", "-p", "no:cacheprovider",
        "tests/integration/test_week7_observability.py",
        "tests/integration/test_week6_gateway_lifecycle.py",
        "tests/unit/test_week6_load_and_routing.py",
    ])
    return True


def gate_week7_experiment_pipeline(ctx: VerifyContext) -> bool:
    """Full Week 7 CPU/mock E2E experiment pipeline gate and Week 6 regression guard."""
    _banner("GATE [w7]: Week 7 Full Observability & Experiment Pipeline (CPU/Mock)")
    if not ctx.no_gpu:
        raise RuntimeError(
            "The Week 7 full live dual-T4 gate requires two controlled GPU replicas. "
            "Run infra/kaggle/week7_experiment_evidence/run.py, or pass --no-gpu for CPU/mock."
        )
    validation = run_week7_e2e_gate()
    if not validation.get("valid"):
        raise RuntimeError(f"Week 7 E2E experiment gate failed: {validation}")
    if _HAS_REAL_PYTEST:
        _run_subprocess([
            sys.executable, "-m", "pytest", "--no-gpu", "-p", "no:cacheprovider",
            "tests/unit/test_week6_load_and_routing.py",
            "tests/integration/test_week6_gateway_lifecycle.py",
        ])
    return True


def gate_week8_controlled_benchmark(ctx: VerifyContext) -> bool:
    """CPU functional coverage, not GPU performance acceptance."""
    _banner("GATE [w8]: Controlled open-loop campaign, raw analysis and figures (CPU/Mock)")
    if not ctx.no_gpu:
        raise RuntimeError(
            "W8 local gate requires --no-gpu; use the controlled launcher on dual-T4"
        )
    validation = run_week8_e2e_gate()
    if not validation.get("valid"):
        raise RuntimeError(f"Week 8 gate failed: {validation}")
    if _HAS_REAL_PYTEST:
        _run_subprocess([
            sys.executable, "-m", "pytest", "--no-gpu", "-p", "no:cacheprovider",
            "tests/unit/test_week8_b_contracts.py",
            "tests/unit/test_week8_scheduler.py",
            "tests/unit/test_week8_observation_scheduler.py",
            "tests/unit/test_week8_realistic_scenarios.py",
            "tests/unit/test_week8_study_orchestration.py",
            "tests/unit/test_week8_observation_orchestration.py",
            "tests/integration/test_week8_b_pipeline.py",
            "tests/integration/test_week8_open_loop_runner.py",
            "tests/integration/test_week8_observation_pipeline.py",
            "tests/integration/test_week8_campaign_and_figures.py",
        ])
    return True


# ------------------------------------------------------------------------------
# OPEN-CLOSED MILESTONE REGISTRY:
# Implemented milestone gates: W1, W2, W3, W4, W6, W7 (Week 5 LAN is separate).
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
    "w4": GateSpec(
        gate_id="w4",
        title="Week 4: WorkerRegistry, HealthMonitor & Round-Robin Failover",
        runner=gate_week4_registry_and_round_robin,
    ),
    "w6": GateSpec(
        gate_id="w6",
        title="Week 6: Routing Decisions, Concurrent Load & Reservation Lifecycle",
        runner=gate_week6_routing_evidence,
    ),
    "w7-observability": GateSpec(
        gate_id="w7-observability",
        title="Week 7 Member A: Trace, Metrics & Request Lifecycle (CPU)",
        runner=gate_week7_observability,
    ),
    "w7": GateSpec(
        gate_id="w7",
        title="Week 7 Full: Observability + Reproducible Experiment Pipeline (CPU)",
        runner=gate_week7_experiment_pipeline,
    ),
    "w8": GateSpec(
        gate_id="w8",
        title="Week 8: Controlled Workloads, Open-Loop Runner, Raw Analysis and Figures (CPU)",
        runner=gate_week8_controlled_benchmark,
    ),
}


# ==============================================================================
# Stage 5: Frontend & Browser E2E Verification (apps/web)
# ==============================================================================
def run_web_stage(ctx: VerifyContext, skip_browser: bool = False) -> bool:
    """Run Frontend ESLint, TypeScript typecheck, Vitest unit tests, Next build, and Playwright."""
    npm_bin = shutil.which("npm.cmd") or shutil.which("npm") or "npm"
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
            "Defaults to CPU/Mock mode; pass --gpu to opt in to live services."
        )
    )
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--no-gpu",
        "--mock",
        dest="no_gpu",
        action="store_true",
        default=True,
        help="Run in No-GPU mode using MockInferenceWorker (CI & CPU hosts).",
    )
    modes.add_argument(
        "--gpu", dest="no_gpu", action="store_false",
        help="Opt in to live GPU checks against already running vLLM/Gateway services.",
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
