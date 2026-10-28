"""Test suite for the LLM Serving Platform.

Ensures `<repo>/src` is on `sys.path` so `from llm_serving_platform...` resolves
cleanly even when test modules are invoked outside pytest.
"""

import sys
from pathlib import Path

_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)
