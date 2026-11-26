"""CPU/mock tests by default; live GPU tests require an explicit --gpu opt-in."""

import pytest

DEFAULT_TEST_PROMPT = "Giải thích ngắn gọn cơ chế Continuous Batching trong 2 câu."


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register CLI options for GPU vs No-GPU execution and custom prompt injection."""
    parser.addoption(
        "--no-gpu",
        action="store_true",
        default=False,
        help="Explicitly select CPU/mock mode (the default).",
    )
    parser.addoption(
        "--gpu",
        action="store_true",
        default=False,
        help="Opt in to live GPU tests; requires an already running GPU service.",
    )
    parser.addoption(
        "--prompt",
        action="store",
        default=DEFAULT_TEST_PROMPT,
        help="Custom prompt string for E2E inference verification.",
    )


@pytest.fixture
def no_gpu(request: pytest.FixtureRequest) -> bool:
    """Use mock mode unless the caller deliberately selected live GPU execution."""
    return not request.config.getoption("--gpu")


def pytest_configure(config: pytest.Config) -> None:
    if config.getoption("--gpu") and config.getoption("--no-gpu"):
        raise pytest.UsageError("Choose either --gpu or --no-gpu, not both")


@pytest.fixture
def prompt_text(request: pytest.FixtureRequest) -> str:
    """Return the prompt string passed via --prompt (or the default 1-sentence prompt)."""
    return str(request.config.getoption("--prompt"))


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep real GPU tests out of ordinary CPU test runs."""
    if config.getoption("--gpu"):
        return
    skip_gpu = pytest.mark.skip(reason="Live GPU test requires explicit --gpu.")
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip_gpu)
