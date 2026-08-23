"""Bootstrap smoke entrypoint for the LLM Serving Platform package."""

import sys


def main() -> int:
    """Execute the bootstrap smoke entrypoint."""
    print("LLM Serving Platform bootstrap OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
