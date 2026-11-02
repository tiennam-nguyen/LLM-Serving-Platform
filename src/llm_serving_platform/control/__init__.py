"""In-memory worker registry, independent health probes, and read-only worker API.

Future routing may consume cached healthy runtime candidates. Probes, network
management, and persistence must remain outside the inference fast path.
"""
