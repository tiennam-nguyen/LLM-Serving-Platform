"""Replayable arrival patterns and mixed input lengths, independent of serving speed."""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from typing import Any

REALISTIC_PROFILES = ("steady_mixed", "irregular_mixed", "burst_mixed", "ramp_mixed")
ARRIVAL_PATTERNS = dict(zip(REALISTIC_PROFILES, ("steady", "irregular", "spike", "ramp")))
ARRIVAL_PATTERNS["microburst_mixed"] = "microburst"
INPUT_TOKEN_BANDS = {"short": (96, 192), "medium": (384, 640), "long": (1024, 1664)}
TokenCounter = Callable[[list[dict[str, str]]], int]


def load_chat_token_counter(snapshot: str) -> TokenCounter:
    """Optional CPU tokenizer; never import Transformers during ordinary tests or replay."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        snapshot,
        local_files_only=True,
        trust_remote_code=False,
        token=False,
    )

    def count(messages: list[dict[str, str]]) -> int:
        return len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
            )
        )

    return count


def arrival_offsets(pattern: str, count: int, window_ms: float, seed: int) -> list[float]:
    """Generate exactly count arrivals; random schedules are frozen before either policy runs.

    Irregular arrivals are ordered uniform points conditioned on the fixed request count,
    not an unconditioned Poisson process. Spike arrivals have two finite high-rate windows;
    ramp arrivals have linearly increasing intensity. Microbursts are an explicit edge case.
    """
    if pattern not in {*ARRIVAL_PATTERNS.values()}:
        raise ValueError("Unknown arrival pattern")
    if type(count) is not int or count < 1:
        raise ValueError("Request count must be a positive integer")
    if isinstance(window_ms, bool) or not math.isfinite(window_ms) or window_ms <= 0:
        raise ValueError("Window must be positive and finite")
    if type(seed) is not int:
        raise ValueError("Seed must be an integer")
    rng = random.Random(f"{seed}:arrivals:{pattern}")
    if pattern == "steady":
        positions = [i / count for i in range(count)]
    elif pattern == "irregular":
        positions = sorted(rng.random() for _ in range(count))
    elif pattern == "microburst":
        groups = min(count, max(1, math.ceil(window_ms / 2000)))
        size = math.ceil(count / groups)
        positions = [(i // size) / groups for i in range(count)]
    else:
        quantiles = [(i + rng.random()) / count for i in range(count)]
        if pattern == "ramp":
            positions = [math.sqrt(q) for q in quantiles]
        else:
            segments = ((0.30, 1), (0.15, 6), (0.25, 1), (0.15, 6), (0.15, 1))
            total = sum(width * weight for width, weight in segments)
            positions = []
            for q in quantiles:
                mass, start = q * total, 0.0
                for width, weight in segments:
                    if mass < width * weight:
                        positions.append(start + mass / weight)
                        break
                    mass -= width * weight
                    start += width
    return [min(round(x * window_ms, 6), math.nextafter(window_ms, 0.0)) for x in positions]


def fit_messages(case: str, length: str, counter: TokenCounter) -> list[dict[str, str]]:
    """Fit natural synthetic context to a token band using the actual chat tokenizer."""
    low, high = INPUT_TOKEN_BANDS[length]
    sentence = (
        "An independent replica handles streaming requests of different lengths, "
        "retaining each request identity and releasing its resources after completion. "
    )

    def messages(repeats: int) -> list[dict[str, str]]:
        return [
            {
                "role": "user",
                "content": (
                    f"Case {case}. Read this context. "
                    + sentence * repeats
                    + "Explain its central issue briefly in at most sixty-four output tokens."
                ),
            }
        ]

    def tokens(repeats: int) -> int:
        value = counter(messages(repeats))
        if type(value) is not int or value <= 0:
            raise ValueError("Tokenizer must return a positive integer token count")
        return value

    upper = 1
    while tokens(upper) < low:
        upper *= 2
        if upper > 4096:
            raise ValueError("Tokenizer did not reach the requested input length")
    lower = 0
    while lower < upper:
        middle = (lower + upper) // 2
        if tokens(middle) < low:
            lower = middle + 1
        else:
            upper = middle
    result = messages(lower)
    if not low <= counter(result) <= high:
        raise ValueError("Tokenizer cannot fit the requested input token band")
    return result


def apply_scenario(
    rows: list[dict[str, Any]],
    profile: str,
    window_ms: float,
    seed: int,
    token_counter: TokenCounter | None = None,
) -> list[dict[str, Any]]:
    """Change arrival shape while preserving the same underlying mixed prompt order."""
    offsets = arrival_offsets(ARRIVAL_PATTERNS[profile], len(rows), window_ms, seed)
    for row, offset in zip(rows, offsets):
        length = row["id"].split("-")[1]
        row["id"] = f"{profile}-{length}-{row['sequence']:05d}"
        row["scheduled_offset_ms"] = offset
        if token_counter is not None:
            row["messages"] = fit_messages(f"{seed}-{row['sequence']:05d}", length, token_counter)
    return rows
