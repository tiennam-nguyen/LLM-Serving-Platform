"""Materialize paired, repeatable W8 workloads and manifests from recorded pilot choices."""

from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path
from typing import Any

from llm_serving_platform.benchmark.artifacts import write_json_atomic, write_text_atomic
from llm_serving_platform.benchmark.config import ExperimentConfigV2
from llm_serving_platform.benchmark.scenarios import (
    ARRIVAL_PATTERNS,
    INPUT_TOKEN_BANDS,
    TokenCounter,
    apply_scenario,
)
from llm_serving_platform.common.config import REPO_ROOT

PROFILES = ("steady", "burst", "heterogeneous")
LEVELS = ("low", "medium", "near_saturation")
POLICIES = ("round_robin", "least_loaded")
PILOT_SELECTION = REPO_ROOT / "experiments/raw/week8/pilot/combined_pilot_selection.json"


def generate_workload(
    profile: str,
    *,
    request_count: int,
    measurement_window_ms: float,
    seed: int,
    token_counter: TokenCounter | None = None,
) -> list[dict[str, Any]]:
    """Generate public synthetic input; token length is not inferred from character count."""
    if profile in ARRIVAL_PATTERNS:
        rows = generate_workload(
            "heterogeneous",
            request_count=request_count,
            measurement_window_ms=measurement_window_ms,
            seed=seed,
        )
        return apply_scenario(rows, profile, measurement_window_ms, seed, token_counter)
    if profile not in PROFILES:
        raise ValueError("Unknown workload profile")
    if type(request_count) is not int or request_count < 1:
        raise ValueError("request_count must be a positive integer")
    if (
        isinstance(measurement_window_ms, bool)
        or not math.isfinite(measurement_window_ms)
        or measurement_window_ms <= 0
    ):
        raise ValueError("measurement_window_ms must be finite and positive")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    rng = random.Random(f"{seed}:{profile}")
    topics = [
        "continuous batching",
        "load balancing",
        "streaming inference",
        "worker health",
        "request tracing",
        "bounded memory",
    ]
    classes = ["short", "medium", "long"] * math.ceil(request_count / 3)
    classes = classes[:request_count]
    rng.shuffle(classes)
    groups = max(
        1,
        math.ceil(
            request_count / max(2, round(request_count / (measurement_window_ms / 1000) * 2))
        ),
    )
    burst_size = math.ceil(request_count / groups)
    rows = []
    for sequence in range(request_count):
        length = classes[sequence] if profile == "heterogeneous" else "short"
        paragraphs = {"short": 1, "medium": 6, "long": 18}[length]
        topic = rng.choice(topics)
        context = " ".join(
            f"Section {part}: an independent replica receives streaming requests about {topic}; "
            "request lengths differ and each response must retain its identity."
            for part in range(paragraphs)
        )
        prompt = (
            f"Case {seed}-{sequence:05d}. Read the following context. {context} "
            "Explain the central issue briefly in at most sixty-four output tokens."
        )
        offset = (
            sequence * measurement_window_ms / request_count
            if profile != "burst"
            else (sequence // burst_size) * measurement_window_ms / groups
        )
        rows.append(
            {
                "id": f"{profile}-{length}-{sequence:05d}",
                "sequence": sequence,
                "scheduled_offset_ms": round(offset, 6),
                "messages": [{"role": "user", "content": prompt}],
            }
        )
    return rows


def prepare_campaign(
    *,
    manifests_dir: Path,
    workloads_dir: Path,
    pilot_selection: Path = PILOT_SELECTION,
    model: str = "Qwen/Qwen2.5-3B-Instruct",
    measurement_window_ms: float = 20000,
    repetitions: int = 3,
    seed: int = 42,
    gateway_url: str = "http://127.0.0.1:8080",
    max_in_flight: int = 350,
    max_dispatch_lag_ms: float = 250,
    tolerances: dict[str, float] | None = None,
    profiles: tuple[str, ...] = PROFILES,
    token_counter: TokenCounter | None = None,
    tokenizer_metadata: dict[str, Any] | None = None,
) -> Path:
    """Create a 54-cell default campaign without executing or overwriting any experiment.

    Pilot points were calibrated on steady input. Burst/heterogeneous stability must still
    be checked on the actual GPU topology. No statistical tolerance is invented here.
    """
    if type(repetitions) is not int or repetitions < 3:
        raise ValueError("A controlled campaign requires at least three repetitions")
    if (
        not profiles
        or len(set(profiles)) != len(profiles)
        or any(profile not in {*PROFILES, *ARRIVAL_PATTERNS} for profile in profiles)
    ):
        raise ValueError("Provide unique, supported workload profiles")
    if (
        isinstance(measurement_window_ms, bool)
        or not math.isfinite(measurement_window_ms)
        or measurement_window_ms <= 0
    ):
        raise ValueError("Invalid measurement window")
    selection_bytes = pilot_selection.read_bytes().replace(b"\r\n", b"\n")
    pilot = json.loads(selection_bytes)
    selected = pilot["combined_1to44rps_selection"]["selected"]
    rates = {level: selected[level]["offered_rps"] for level in LEVELS}
    if any(
        isinstance(rate, bool)
        or not isinstance(rate, (int, float))
        or not math.isfinite(rate)
        or rate <= 0
        for rate in rates.values()
    ):
        raise ValueError("Pilot load points must be finite positive numbers")
    if tolerances is not None:
        allowed = {"latency_ms.ttft_p50", "latency_ms.e2e_p50", "rates.completed_drain_rps"}
        if set(tolerances) != allowed or any(
            isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0
            for v in tolerances.values()
        ):
            raise ValueError(
                "Provide nonnegative relative tolerances for all three primary metrics"
            )
    # Precompute/validate everything before creating output directories.
    workloads, manifests, order = {}, {}, []
    for profile in profiles:
        for level in LEVELS:
            exact_count = rates[level] * measurement_window_ms / 1000
            count = round(exact_count)
            if count < 1 or not math.isclose(count, exact_count, abs_tol=1e-7):
                raise ValueError(
                    "Window must give an integer number of requests at every pilot rate"
                )
            workload_name = f"{profile}_{level}.jsonl"
            rows = generate_workload(
                profile,
                request_count=count,
                measurement_window_ms=measurement_window_ms,
                seed=seed,
                token_counter=token_counter,
            )
            text = "".join(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in rows
            )
            workloads[workload_name] = text
            source = (workloads_dir / workload_name).resolve()
            try:
                source_path = source.relative_to(REPO_ROOT).as_posix()
            except ValueError:
                source_path = str(source)
            for repetition in range(1, repetitions + 1):
                pair_order = POLICIES if repetition % 2 else tuple(reversed(POLICIES))
                for policy in pair_order:
                    name = f"w8-{profile}-{level}-{policy}-r{repetition}"
                    raw = {
                        "schema_version": 2,
                        "name": name,
                        "model": {"id": model},
                        "topology": {"worker_count": 2},
                        "policy": {"name": policy},
                        "workload": {
                            "path": source_path,
                            "profile": profile,
                            "request_count": count,
                        },
                        "seed": seed,
                        "warmup": {"request_count": 2},
                        "generation": {"temperature": 0, "max_tokens": 64, "top_p": 1},
                        "runner": {
                            "gateway_url": gateway_url,
                            "request_timeout_s": 120,
                            "max_in_flight": max_in_flight,
                            "max_dispatch_lag_ms": max_dispatch_lag_ms,
                        },
                        "load": {
                            "mode": "open_loop",
                            "measurement_window_ms": measurement_window_ms,
                        },
                        "comparison": {
                            "id": f"w8-{profile}-{level}",
                            "load_level": level,
                            "repetition": repetition,
                        },
                    }
                    normalized = ExperimentConfigV2.model_validate(raw).to_normalized_dict()
                    filename = name + ".json"
                    manifests[filename] = normalized
                    order.append(
                        {
                            "manifest": filename,
                            "comparison_id": raw["comparison"]["id"],
                            "policy": policy,
                            "repetition": repetition,
                            "workload_sha256": hashlib.sha256(text.encode()).hexdigest(),
                        }
                    )
    if manifests_dir.exists() or workloads_dir.exists():
        raise FileExistsError("Campaign output already exists; choose new output directories")
    manifests_dir.mkdir(parents=True)
    workloads_dir.mkdir(parents=True)
    for name, text in workloads.items():
        write_text_atomic(workloads_dir / name, text)
    for name, config in manifests.items():
        write_json_atomic(manifests_dir / name, config)
    protocol = {
        "schema_version": 1,
        "name": "week8-controlled-benchmark",
        "seed": seed,
        "repetitions": repetitions,
        "model": model,
        "load_points_rps": rates,
        "measurement_window_ms": measurement_window_ms,
        "calibration_source": {
            "path": pilot_selection.resolve().relative_to(REPO_ROOT).as_posix()
            if pilot_selection.resolve().is_relative_to(REPO_ROOT)
            else str(pilot_selection.resolve()),
            "sha256": hashlib.sha256(selection_bytes).hexdigest(),
            "selected": selected,
        },
        "cache_procedure": "restart_both_replicas_and_gateway_before_each_run",
        "repeatability": {
            "method": "max_relative_deviation_from_median",
            "minimum_repetitions": repetitions,
            "tolerances": tolerances,
            "status": "predeclared"
            if tolerances is not None
            else "thresholds_pending_repeated_pilot",
        },
        "runs": order,
    }
    if any(profile in ARRIVAL_PATTERNS for profile in profiles):
        protocol["workload_design"] = {
            "profiles": list(profiles),
            "arrival_patterns": {
                profile: ARRIVAL_PATTERNS.get(profile, "legacy") for profile in profiles
            },
            "length_mix": "short_medium_long",
            "input_token_bands": INPUT_TOKEN_BANDS if token_counter else None,
            "tokenizer": tokenizer_metadata if token_counter else None,
            "token_counts_verified": token_counter is not None,
            "workloads": {
                name: {
                    "input_tokens": [
                        token_counter(row["messages"])
                        for row in (json.loads(line) for line in text.splitlines())
                    ]
                }
                for name, text in workloads.items()
            }
            if token_counter
            else {},
        }
    path = manifests_dir / "campaign.json"
    write_json_atomic(path, protocol)
    return path
