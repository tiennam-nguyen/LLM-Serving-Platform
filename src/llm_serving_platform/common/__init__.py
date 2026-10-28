"""Common cross-cutting primitives module boundary.

Provides centralized 12-Factor environment configuration and structured JSON logging
shared across platform components.
"""

from llm_serving_platform.common.config import GatewayConfig, load_dotenv_file
from llm_serving_platform.common.logging import log_structured_event

__all__ = ["GatewayConfig", "load_dotenv_file", "log_structured_event"]
