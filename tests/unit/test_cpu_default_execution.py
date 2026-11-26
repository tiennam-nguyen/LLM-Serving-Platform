"""Verify CPU defaults and explicit GPU flags without contacting a GPU service."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from infra.kaggle.week8_controlled_benchmark import run as study
from infra.scripts import verify


@pytest.mark.parametrize("flags,expected", [([], True), (["--no-gpu"], True), (["--gpu"], False)])
def test_verifier_defaults_to_cpu(flags, expected):
    assert verify.build_arg_parser().parse_args(flags).no_gpu is expected


def test_verifier_rejects_conflicting_modes():
    with pytest.raises(SystemExit):
        verify.build_arg_parser().parse_args(["--gpu", "--no-gpu"])


def test_explicit_live_mode_is_forwarded_to_pytest(monkeypatch):
    calls = []
    monkeypatch.setattr(verify, "_HAS_REAL_PYTEST", True)
    monkeypatch.setattr(verify, "_run_subprocess", lambda cmd, **kw: calls.append((cmd, kw)))
    context = verify.VerifyContext(False, "synthetic", "http://unused", "http://unused", "fake")
    assert verify._run_pytest_or_fallback("unit", context)
    assert "--gpu" in calls[0][0]
    assert calls[0][1]["env"]["REQUIRE_LIVE_GPU"] == "1"


def test_remote_launcher_does_nothing_without_gpu_opt_in(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "--revision", "synthetic"])
    monkeypatch.setattr(
        study.pilot,
        "checked_inventory",
        lambda: pytest.fail("Must not probe GPU without explicit opt-in"),
    )
    with pytest.raises(SystemExit) as exc:
        study.main()
    assert exc.value.code == 2


@pytest.mark.parametrize(
    "flags,live,exit_code",
    [
        ([], False, 0),
        (["--no-gpu"], False, 0),
        (["--gpu"], True, 0),
        (["--gpu", "--no-gpu"], False, 4),
    ],
)
def test_pytest_mode_and_gpu_marker_are_opt_in(tmp_path, flags, live, exit_code):
    # Isolated child tests exercise actual pytest argument/marker handling. The marked
    # child is a sentinel with no GPU calls, so --gpu remains safe in this unit test.
    conftest = Path(__file__).resolve().parents[1] / "conftest.py"
    (tmp_path / "conftest.py").write_bytes(conftest.read_bytes())
    (tmp_path / "test_modes.py").write_text(
        "import pytest\n"
        f"def test_selected_mode(no_gpu):\n    assert no_gpu is {not live!r}\n"
        "@pytest.mark.gpu\ndef test_live_sentinel(no_gpu):\n    assert no_gpu is False\n",
        encoding="utf-8",
    )
    env = os.environ.copy()
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            *flags,
            "--basetemp",
            str(tmp_path / "scratch"),
            str(tmp_path),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == exit_code, result.stdout + result.stderr
    if exit_code == 0:
        assert ("2 passed" if live else "1 passed, 1 skipped") in result.stdout
