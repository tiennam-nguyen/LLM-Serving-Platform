"""Unit tests verifying package structure, submodule imports, and CLI smoke entrypoint."""

import importlib

import pytest


def test_root_package_import() -> None:
    """Verify that the root package can be imported and exports __version__."""
    import llm_serving_platform

    assert hasattr(llm_serving_platform, "__version__")
    assert isinstance(llm_serving_platform.__version__, str)
    assert len(llm_serving_platform.__version__) > 0


@pytest.mark.parametrize(
    "submodule_name",
    [
        "llm_serving_platform.gateway",
        "llm_serving_platform.control",
        "llm_serving_platform.benchmark",
        "llm_serving_platform.routing",
        "llm_serving_platform.workers",
        "llm_serving_platform.telemetry",
        "llm_serving_platform.common",
    ],
)
def test_submodule_imports(submodule_name: str) -> None:
    """Verify that all intended architectural submodules can be cleanly imported."""
    module = importlib.import_module(submodule_name)
    assert module is not None
    assert module.__doc__ is not None
    assert len(module.__doc__.strip()) > 0


def test_bootstrap_main_entrypoint(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify that the package __main__ entrypoint executes and prints expected smoke output."""
    from llm_serving_platform.__main__ import main

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "LLM Serving Platform bootstrap OK"
