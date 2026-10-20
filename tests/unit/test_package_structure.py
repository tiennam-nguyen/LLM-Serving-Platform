"""Unit tests verifying package structure, submodule imports, and CLI smoke entrypoint."""

import importlib

import pytest


def test_root_package_import() -> None:
    """Verify that the root package can be imported and exports __version__."""
    import src.llm_serving_platform

    assert hasattr(src.llm_serving_platform, "__version__")
    assert isinstance(src.llm_serving_platform.__version__, str)
    assert len(src.llm_serving_platform.__version__) > 0


@pytest.mark.parametrize(
    "submodule_name",
    [
        "src.llm_serving_platform.gateway",
        "src.llm_serving_platform.control",
        "src.llm_serving_platform.benchmark",
        "src.llm_serving_platform.routing",
        "src.llm_serving_platform.workers",
        "src.llm_serving_platform.telemetry",
        "src.llm_serving_platform.common",
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
    from src.llm_serving_platform.__main__ import main

    exit_code = main()
    captured = capsys.readouterr()

    assert exit_code == 0
    assert captured.out.strip() == "LLM Serving Platform bootstrap OK"
