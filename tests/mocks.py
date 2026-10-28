"""Backward-compatible re-export of MockInferenceWorker from canonical workers package."""

from llm_serving_platform.workers.mock_adapter import MockInferenceWorker

__all__ = ["MockInferenceWorker"]
