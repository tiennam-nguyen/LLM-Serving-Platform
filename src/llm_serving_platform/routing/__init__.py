"""Routing module boundary.

Defines worker selection abstractions and routing policies decoupled from HTTP transport.
"""

from llm_serving_platform.routing.router import RoutingPolicy, SingleWorkerRouter

__all__ = ["RoutingPolicy", "SingleWorkerRouter"]
