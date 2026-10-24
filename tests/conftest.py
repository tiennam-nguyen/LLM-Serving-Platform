"""Pytest configuration supporting --no-gpu and --prompt flags for dual-mode testing."""

import pytest

DEFAULT_TEST_PROMPT = "Giải thích ngắn gọn cơ chế Continuous Batching trong 2 câu."


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options for GPU vs No-GPU execution and custom prompt injection."""
    parser.addoption(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Run tests in CPU/mock mode without requiring a live vLLM GPU server.",
    )
    parser.addoption(
        "--prompt",
        action="store",
        default=DEFAULT_TEST_PROMPT,
        help="Custom prompt string for E2E inference verification.",
    )


@pytest.fixture
def no_gpu(request: pytest.FixtureRequest) -> bool:
    """Return True if --no-gpu flag was passed to pytest."""
    return bool(request.config.getoption("--no-gpu"))


@pytest.fixture
def prompt_text(request: pytest.FixtureRequest) -> str:
    """Return the prompt string passed via --prompt (or the default 1-sentence prompt)."""
    return str(request.config.getoption("--prompt"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip tests explicitly marked @pytest.mark.gpu when --no-gpu is set."""
    if not config.getoption("--no-gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="Skipped live GPU test because --no-gpu was passed.")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
