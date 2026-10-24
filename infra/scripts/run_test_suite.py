"""Backward-compatible wrapper delegating to the unified infra/scripts/verify.py pipeline."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from infra.scripts.verify import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
