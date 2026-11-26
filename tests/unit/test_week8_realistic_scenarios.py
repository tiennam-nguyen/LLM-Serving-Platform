"""CPU-only checks for replay schedules, input diversity, and tokenizer provenance."""

import json
import sys
from types import SimpleNamespace

import pytest

from llm_serving_platform.benchmark.campaign import generate_workload, prepare_campaign
from llm_serving_platform.benchmark.scenarios import (
    ARRIVAL_PATTERNS,
    INPUT_TOKEN_BANDS,
    REALISTIC_PROFILES,
    arrival_offsets,
    fit_messages,
    load_chat_token_counter,
)


@pytest.mark.parametrize("pattern", sorted(set(ARRIVAL_PATTERNS.values())))
def test_every_arrival_pattern_is_ordered_bounded_and_replayable(pattern):
    offsets = arrival_offsets(pattern, 60, 20000, 42)
    assert offsets == arrival_offsets(pattern, 60, 20000, 42)
    assert len(offsets) == 60
    assert offsets == sorted(offsets)
    assert all(0 <= value < 20000 for value in offsets)
    if pattern == "steady":
        assert offsets[1] - offsets[0] == pytest.approx(20000 / 60)
    elif pattern == "microburst":
        assert len(set(offsets)) < len(offsets)
    else:
        assert len(set(offsets)) == len(offsets)
        gaps = [b - a for a, b in zip(offsets, offsets[1:])]
        assert max(gaps) > min(gaps) * 1.1


def test_spike_has_finite_busy_intervals_and_ramp_increases_arrival_density():
    spike = arrival_offsets("spike", 100, 20000, 42)
    busy = sum(6000 <= value < 9000 or 14000 <= value < 17000 for value in spike)
    assert busy >= 65
    ramp = arrival_offsets("ramp", 100, 20000, 42)
    assert sum(value < 10000 for value in ramp) == 25
    assert sum(value >= 10000 for value in ramp) == 75


@pytest.mark.parametrize(
    "pattern,count,window,seed",
    [
        ("unknown", 4, 1000, 42),
        ("steady", True, 1000, 42),
        ("steady", 0, 1000, 42),
        ("steady", 4, float("nan"), 42),
        ("steady", 4, -1, 42),
        ("steady", 4, 1000, True),
    ],
)
def test_invalid_arrival_input_rejected(pattern, count, window, seed):
    with pytest.raises(ValueError):
        arrival_offsets(pattern, count, window, seed)


@pytest.mark.parametrize("profile", [*REALISTIC_PROFILES, "microburst_mixed"])
def test_each_scenario_mixes_lengths_and_keeps_the_same_inputs_across_arrival_shapes(profile):
    rows = generate_workload(profile, request_count=30, measurement_window_ms=20000, seed=42)
    assert {row["id"].split("-")[1] for row in rows} == {"short", "medium", "long"}
    baseline = generate_workload(
        "steady_mixed", request_count=30, measurement_window_ms=20000, seed=42
    )
    assert [row["messages"] for row in rows] == [row["messages"] for row in baseline]
    assert len({len(row["messages"][0]["content"]) for row in rows}) >= 3


def count_fixture_tokens(messages):
    # A synthetic tokenizer for unit tests; never presented as Qwen measurements.
    return len(messages[0]["content"].split()) + 8


@pytest.mark.parametrize("length", list(INPUT_TOKEN_BANDS))
def test_input_lengths_are_checked_with_tokenizer_not_character_count(length):
    messages = fit_messages("unit-fixture", length, count_fixture_tokens)
    low, high = INPUT_TOKEN_BANDS[length]
    assert low <= count_fixture_tokens(messages) <= high
    assert len(messages[0]["content"]) != count_fixture_tokens(messages)


def test_invalid_or_non_progressing_tokenizer_cannot_claim_verified_lengths():
    with pytest.raises(ValueError, match="positive integer"):
        fit_messages("fixture", "short", lambda _: True)
    with pytest.raises(ValueError, match="did not reach"):
        fit_messages("fixture", "short", lambda _: 1)


def test_cached_tokenizer_never_uses_account_or_downloads(monkeypatch):
    calls = []

    class Tokenizer:
        def apply_chat_template(self, messages, **options):
            # Transformers 5 returns a mapping by default; len(mapping) is not a token count.
            assert options["tokenize"] is True
            assert options["add_generation_prompt"] is True
            tokens = [1, 2, 3]
            if options.get("return_dict", True):
                return {"input_ids": tokens, "attention_mask": [1] * len(tokens)}
            return tokens

    def cached(path, **options):
        calls.append((path, options))
        return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=cached)),
    )
    counter = load_chat_token_counter("synthetic-snapshot")
    assert counter([{"role": "user", "content": "synthetic"}]) == 3
    assert calls == [
        (
            "synthetic-snapshot",
            {
                "local_files_only": True,
                "trust_remote_code": False,
                "token": False,
            },
        )
    ]


def test_realistic_campaign_materializes_both_policies_with_identical_token_verified_inputs(
    tmp_path,
):
    path = prepare_campaign(
        manifests_dir=tmp_path / "manifests",
        workloads_dir=tmp_path / "workloads",
        profiles=REALISTIC_PROFILES,
        token_counter=count_fixture_tokens,
        tokenizer_metadata={"model_id": "synthetic-tokenizer", "snapshot_revision": "fixture"},
    )
    protocol = json.loads(path.read_text())
    assert len(protocol["runs"]) == 72
    design = protocol["workload_design"]
    assert design["token_counts_verified"] is True
    assert len(design["workloads"]) == 12
    for name, values in design["workloads"].items():
        rows = [
            json.loads(line) for line in (tmp_path / "workloads" / name).read_text().splitlines()
        ]
        for row, tokens in zip(rows, values["input_tokens"]):
            low, high = INPUT_TOKEN_BANDS[row["id"].split("-")[1]]
            assert low <= tokens <= high
    for first, second in zip(protocol["runs"][::2], protocol["runs"][1::2]):
        assert first["workload_sha256"] == second["workload_sha256"]
        assert first["comparison_id"] == second["comparison_id"]
        assert first["policy"] != second["policy"]
