"""Gateway module boundary.

Future responsibility:
    Acts as the data-plane HTTP entrypoint for client requests:
        client -> gateway -> router -> worker adapter -> runtime

    Handles protocol termination, request validation, and response streaming.
    No HTTP routing or inference proxy logic is implemented at bootstrap.
"""
