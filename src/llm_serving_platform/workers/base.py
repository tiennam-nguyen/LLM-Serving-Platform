"""Worker base classes and exception definitions for backend inference engines."""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict


class WorkerException(Exception):
    """Base exception for all inference worker failures."""

    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class WorkerTimeoutError(WorkerException):
    """Raised when an inference worker times out while handling a request."""

    def __init__(self, message: str = "Worker timed out", status_code: int = 504) -> None:
        super().__init__(message, status_code=status_code)


class WorkerUnavailableError(WorkerException):
    """Raised when an inference worker is unreachable, disconnected, or returns non-200."""

    def __init__(self, message: str = "Worker unavailable", status_code: int = 503) -> None:
        super().__init__(message, status_code=status_code)


class InferenceWorker(ABC):
    """Abstract base class defining the contract for all inference runtime workers."""

    @abstractmethod
    async def stream_chat(
        self, payload: Dict[str, Any], request_id: str
    ) -> AsyncIterator[str]:
        """Stream chat completion tokens formatted as SSE events.

        Args:
            payload: OpenAI-compatible chat completion payload dictionary.
            request_id: Unique correlation ID for tracing the request.

        Yields:
            Raw SSE event lines (e.g. `data: {...}\\n\\n` or `data: [DONE]\\n\\n`).

        Raises:
            WorkerTimeoutError: If the worker fails to respond within the configured timeout.
            WorkerUnavailableError: If the worker is unreachable or returns an error status.
            WorkerException: For other worker-level failures.
        """
        ...

    @abstractmethod
    async def check_health(self) -> bool:
        """Check the health status of the worker.

        Returns:
            True if the worker backend is healthy and ready to serve requests, False otherwise.
        """
        ...
