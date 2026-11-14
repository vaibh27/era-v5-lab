"""Small, tokenizer-independent tests for the candidate revalidation CLI."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import pytest
import revalidate_tokenizer_candidates as rtc

RESULTS = Path(__file__).resolve().parents[1] / "results" / "tokenizer_candidates"
SAMPLE_V0_RESULTS = RESULTS / "sample_v0"


def test_parse_candidate():
    candidate = rtc.parse_candidate("v0=/tmp/v0/tokenizer.json")
    assert candidate.name == "v0"
    assert candidate.sha256 == rtc.EXPECTED_SHA256["v0"]
    assert str(candidate.path).endswith("/tmp/v0/tokenizer.json")


def test_parse_future_candidate_with_explicit_hash():
    candidate = rtc.parse_candidate(f"future@{'a' * 64}=/tmp/future/tokenizer.json")
    assert candidate.name == "future"
    assert candidate.sha256 == "a" * 64


@pytest.mark.parametrize(
    "value",
    ["v0", "=/tmp/tokenizer.json", "v0=", "future=/tmp/tokenizer.json", "future@abc=/tmp/tokenizer.json"],
)
def test_parse_candidate_rejects_invalid_value(value):
    with pytest.raises(argparse.ArgumentTypeError):
        rtc.parse_candidate(value)


def test_committed_candidate_results_are_internally_consistent():
    report = json.loads((RESULTS / "summary.json").read_text(encoding="utf-8"))
    expected_order = ["v0", "v0.1", "v1-c1", "C1", "C2", "C3"]
    assert report["candidate_order"] == expected_order
    assert set(report["candidates"]) == set(expected_order)
    for name in expected_order:
        candidate = report["candidates"][name]
        assert candidate["tokenizer_sha256"] == rtc.EXPECTED_SHA256[name]
        assert candidate["vocab_size"] == 131_072
        assert candidate["max_raw_byte_length"] <= 32
        assert candidate["gate_zero_other_collision_groups"]
        for pos_dim in (32, 64):
            raw = candidate["collision_totals"][f"raw_P{pos_dim}"]
            assert raw["n_groups"] == raw["n_tokens_in_groups"] == raw["n_excess_tokens"] == 0

    sums = defaultdict(lambda: {"tokens": 0, "colliding": 0, "excess": 0})
    with (RESULTS / "collisions_by_script.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = row["candidate"], row["extraction"], int(row["P"])
            sums[key]["tokens"] += int(row["n_tokens"])
            sums[key]["colliding"] += int(row["n_colliding_tokens"])
            sums[key]["excess"] += int(row["n_excess_tokens"])
    assert len(sums) == len(expected_order) * 2 * 2
    for (name, extraction, pos_dim), totals in sums.items():
        expected = report["candidates"][name]["collision_totals"][f"{extraction}_P{pos_dim}"]
        assert totals["tokens"] == 131_072
        assert totals["colliding"] == expected["n_tokens_in_groups"]
        assert totals["excess"] == expected["n_excess_tokens"]

    group_sums = defaultdict(lambda: {"groups": 0, "tokens": 0, "excess": 0})
    with (RESULTS / "collision_groups.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            token_ids = [int(token_id) for token_id in row["token_ids"].split(";")]
            n_tokens = int(row["n_tokens"])
            assert len(token_ids) == n_tokens == len(set(token_ids))
            assert int(row["n_pairs"]) == n_tokens * (n_tokens - 1) // 2
            key = row["candidate"], row["extraction"], int(row["P"])
            group_sums[key]["groups"] += 1
            group_sums[key]["tokens"] += n_tokens
            group_sums[key]["excess"] += n_tokens - 1
    for (name, extraction, pos_dim), totals in group_sums.items():
        expected = report["candidates"][name]["collision_totals"][f"{extraction}_P{pos_dim}"]
        assert totals["groups"] == expected["n_groups"]
        assert totals["tokens"] == expected["n_tokens_in_groups"]
        assert totals["excess"] == expected["n_excess_tokens"]
    assert not any(extraction == "raw" for _, extraction, _ in group_sums)


def test_sample_v0_candidate_results_are_internally_consistent():
    """The separately generated sample-v0 report covers the delivered C1/C2/C3 artifacts."""
    report = json.loads((SAMPLE_V0_RESULTS / "summary.json").read_text(encoding="utf-8"))
    expected_order = ["C1-s0", "C2-s0", "C3-s0"]
    assert report["candidate_order"] == expected_order
    assert set(report["candidates"]) == set(expected_order)

    for name in expected_order:
        candidate = report["candidates"][name]
        assert candidate["tokenizer_sha256"] == rtc.EXPECTED_SHA256[name]
        assert candidate["vocab_size"] == 131_072
        assert candidate["max_raw_byte_length"] == 32
        assert candidate["gate_zero_other_collision_groups"]
        for pos_dim in (32, 64):
            raw = candidate["collision_totals"][f"raw_P{pos_dim}"]
            assert raw["n_groups"] == raw["n_tokens_in_groups"] == raw["n_excess_tokens"] == 0

    candidates_in_csv = set()
    totals = defaultdict(int)
    with (SAMPLE_V0_RESULTS / "collisions_by_script.csv").open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            candidates_in_csv.add(row["candidate"])
            totals[row["candidate"], row["extraction"], int(row["P"])] += int(row["n_tokens"])
    assert candidates_in_csv == set(expected_order)
    assert set(totals.values()) == {131_072}

    with (SAMPLE_V0_RESULTS / "collision_groups.csv").open(encoding="utf-8", newline="") as handle:
        groups = list(csv.DictReader(handle))
    assert groups
    assert {row["candidate"] for row in groups} == set(expected_order)
    assert {row["extraction"] for row in groups} == {"decode"}
    assert {row["cause"] for row in groups} == {"decode_collapse"}

    difference = report["differences_from_previous_candidate"]["C2-s0"]
    assert difference["reference"] == "C1-s0"
    assert difference["raw_changed"]["count"] == 8_704
    assert difference["decode_changed"]["count"] == 8_704

    difference = report["differences_from_previous_candidate"]["C3-s0"]
    assert difference["reference"] == "C2-s0"
    assert difference["raw_changed"]["count"] == 5_406
    assert difference["decode_changed"]["count"] == 5_406
