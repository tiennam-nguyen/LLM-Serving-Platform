"""Gateway module boundary.

Acts as the data-plane HTTP entrypoint for client requests:
    client -> gateway -> worker adapter -> runtime

Handles protocol termination, request validation, and response streaming.
"""

from llm_serving_platform.gateway.main import app, create_app

__all__ = ["app", "create_app"]
