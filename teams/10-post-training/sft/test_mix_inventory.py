"""Tests for mix_inventory.py (G10 Package D3): the eligibility ledger and the mixture feasibility arithmetic.

    pytest teams/10-post-training/sft/test_mix_inventory.py -q

Standard library plus pytest. No network, no real data: every record below is a synthetic placeholder and the numbers
are made up for the test (they are not measurements of any set).
"""

import gzip
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mix_inventory as mi  # noqa: E402

TOKENIZER = "a" * 64


def unit(set_id, approval="tsai_reserves", coverage="full", kept=100, seq=1000, targets=800, tokenizer=TOKENIZER,
         measured=True, area="code", families=None):
    """A ledger record shaped like the ones build_units writes."""
    u = {"type": "set", "set_id": set_id, "area": area, "area_basis": "proposal_section_2", "group": None,
         "approval_state": approval, "coverage_class": coverage, "problems": [],
         "rows": {"stages": {"converted": kept + 5, "removed": 5, "kept": kept}},
         "families": {"unique": families, "basis": "test"}, "length": None,
         "tokens": {"status": "measured" if measured else "not_measured",
                    "sequence_tokens": seq if measured else None,
                    "shifted_supervised_targets": targets if measured else None,
                    "tokenizer_sha256": tokenizer if measured else None, "basis": "test"}}
    u["problems"] = mi.check_unit(u)
    return u


def by_id(*units):
    return {u["set_id"]: u for u in units}


def group(members, fraction=1.0, name="g"):
    return (name, fraction, list(members))


# --- required cases ---------------------------------------------------------------------------------------

def test_duplicate_family_across_sources_counts_once():
    sets = {"taco": {"f1", "f2", "f3"}, "apps": {"f3", "f4"}}
    ov = mi.family_overlap(sets)
    assert ov["sum_of_sources"] == 5
    assert ov["unique_families"] == 4          # f3 is one family although two sources hold it
    assert ov["duplicated"] == 1
    assert ov["pairs"] == {"apps+taco": 1}
    assert mi.family_overlap({"a": {"x"}, "b": {"y"}})["duplicated"] == 0


def test_rows_versus_tokens_mismatch_is_flagged():
    assert "rows_tokens_mismatch" in unit("s", kept=100, seq=60)["problems"]       # fewer tokens than rows
    assert "rows_tokens_mismatch" in unit("s", kept=0, seq=10)["problems"]         # tokens without rows
    assert unit("s", kept=100, seq=1000)["problems"] == []
    assert "targets_exceed_tokens" in unit("s", seq=1000, targets=1001)["problems"]


def test_absent_tokenizer_hash_makes_tokens_unusable():
    u = unit("s", tokenizer=None)
    assert "tokens_without_tokenizer_hash" in u["problems"]
    assert not mi.eligible(u, mi.LENS_RESERVED)
    cap = mi.group_capacity(group(["s"]), 1000, by_id(u), mi.LENS_RESERVED)
    assert cap["supply_tokens_measured"] == 0 and cap["status"]["1x"].startswith("INFEASIBLE")


def test_zero_assistant_targets_is_flagged_and_excluded_from_supply():
    u = unit("s", targets=0)
    assert "zero_assistant_targets" in u["problems"]
    cap = mi.group_capacity(group(["s"]), 1000, by_id(u), mi.LENS_RESERVED)
    assert cap["status"]["1x"].startswith("INFEASIBLE") and "zero_assistant_targets" in cap["status"]["1x"]
    text_only = unit("t", coverage="text-only", measured=False)
    text_only["tokens"]["shifted_supervised_targets"] = 0
    assert "zero_assistant_targets" in mi.check_unit(text_only)


def test_excluded_source_cannot_supply_a_requested_share():
    bad = unit("x", approval="excluded", coverage="excluded", kept=0, seq=0, targets=0, measured=False)
    cap = mi.group_capacity(group(["x"]), 5000, by_id(bad), mi.LENS_RESERVED)
    assert cap["status"]["4x"].startswith("INFEASIBLE") and "excluded" in cap["status"]["4x"]
    assert cap["supply_tokens_measured"] == 0 and cap["eligible_sets"] == []
    # a nonzero share is not rescued by a sibling that is excluded: only the allowed source counts
    ok = unit("ok", approval="allowed", seq=10_000, targets=9000)
    cap2 = mi.group_capacity(group(["x", "ok"]), 5000, by_id(bad, ok), mi.LENS_RESERVED)
    assert cap2["eligible_sets"] == ["ok"] and cap2["status"]["1x"] == "FEASIBLE"
    # zero request is not an error even with no source
    assert mi.group_capacity(group(["x"]), 0, by_id(bad), mi.LENS_RESERVED)["status"]["1x"] == "NO_REQUEST"


def test_impossible_area_effort_cell():
    requests = {("indic", "64k"): 500, ("code", "16k"): 100, ("code", "1k"): 0}
    supply = {("code", "16k"): 100, ("indic", "1k"): 10_000}
    out = mi.effort_cell_check(requests, supply, 1)
    assert [o["cell"] for o in out] == [["indic", "64k"]]
    assert "impossible cell" in out[0]["reason"]
    # a cell with supply but under-supplied at the cap
    out2 = mi.effort_cell_check({("code", "16k"): 450}, {("code", "16k"): 100}, 4)
    assert out2 and "exceeds 4x" in out2[0]["reason"]
    assert mi.effort_cell_check({("code", "16k"): 400}, {("code", "16k"): 100}, 4) == []


def test_shares_must_sum_to_one_and_be_known():
    ok = {"code": 0.35, "agentic": 0.25, "tool_use": 0.15, "indic": 0.15, "general": 0.10, "safety": 0.0}
    assert mi.check_shares(ok) == []
    assert any("not_1" in p for p in mi.check_shares(dict(ok, general=0.07)))      # 0.97: the 3% Safety was dropped
    assert any("not_1" in p for p in mi.check_shares(dict(ok, safety=0.03)))       # 1.03
    assert "negative_share" in mi.check_shares(dict(ok, safety=-0.1, general=0.2))
    assert any(p.startswith("unknown_area") for p in mi.check_shares({"code": 0.5, "vision": 0.5}))
    # a share with no composition group cannot be allocated
    assert any("no_composition_group" in p for p in mi.check_shares({"code": 1.0}, {"agentic": [group(["a"])]}))
    assert any("fractions_not_1" in p for p in mi.check_shares({"code": 1.0}, {"code": [group(["a"], 0.5)]}))


def test_unsatisfied_repeat_cap():
    u = unit("s", seq=1000, targets=900)
    cap = mi.group_capacity(group(["s"]), 3000, by_id(u), mi.LENS_RESERVED)   # needs 3x the supply
    assert cap["minimum_exposure"] == pytest.approx(3.0)
    assert cap["status"]["1x"].startswith("INFEASIBLE") and "exceeds 1x" in cap["status"]["1x"]
    assert cap["status"]["2x"].startswith("INFEASIBLE")
    assert cap["status"]["4x"] == "FEASIBLE"
    assert mi.group_capacity(group(["s"]), 1000, by_id(u), mi.LENS_RESERVED)["status"]["1x"] == "FEASIBLE"


def test_inventory_is_deterministic(tmp_path):
    team = mini_team(tmp_path)
    first = mi.dump_jsonl(mi.collect(team, None, None))
    second = mi.dump_jsonl(mi.collect(team, None, None))
    assert first == second
    assert mi.render_markdown(mi.collect(team, None, None)) == mi.render_markdown(mi.collect(team, None, None))
    assert "created" not in first and "timestamp" not in first      # no clock in the ledger


# --- supply is never guessed -----------------------------------------------------------------------------

def test_unmeasured_member_is_never_assumed_large_enough():
    a = unit("a", seq=1000, targets=900)
    b = unit("b", measured=False)
    cap = mi.group_capacity(group(["a", "b"]), 500, by_id(a, b), mi.LENS_RESERVED)
    assert cap["supply_complete"] is False and cap["unmeasured_sets"] == ["b"]
    assert cap["status"]["1x"] == "FEASIBLE_ON_MEASURED"       # the measured part alone covers it
    assert cap["exposure_note"] and "upper bound" in cap["exposure_note"]
    big = mi.group_capacity(group(["a", "b"]), 5000, by_id(a, b), mi.LENS_RESERVED)
    assert big["status"]["1x"] == "UNMEASURED"                 # cannot be called infeasible: b may add supply
    assert big["status"]["4x"] == "UNMEASURED"                 # 4 x 1,000 measured is still below 5,000
    only_b = mi.group_capacity(group(["b"]), 500, by_id(b), mi.LENS_RESERVED)
    assert only_b["status"]["1x"] == "UNMEASURED" and only_b["minimum_exposure"] is None


def test_lens_allowed_ignores_reserved_sets():
    r = unit("r", approval="tsai_reserves")
    a = unit("a", approval="allowed")
    assert mi.eligible(a, mi.LENS_ALLOWED) and not mi.eligible(r, mi.LENS_ALLOWED)
    assert mi.eligible(r, mi.LENS_RESERVED)
    assert not mi.eligible(unit("h", coverage="held", measured=False), mi.LENS_RESERVED)


def test_requested_and_minimum_exposure_arithmetic():
    assert mi.requested_tokens(100_000_000, 0.35) == pytest.approx(35_000_000)
    assert mi.minimum_exposure(35_000_000, 70_000_000) == pytest.approx(0.5)
    assert mi.minimum_exposure(1, 0) is None


def test_capacity_records_flag_safety_with_only_excluded_sources():
    units = [unit(s, approval="excluded", coverage="excluded", kept=0, seq=0, targets=0, measured=False)
             for s in mi.COMPOSITION["safety"][0][2]]
    shares = {"code": 0.0, "agentic": 0.0, "tool_use": 0.0, "indic": 0.0, "general": 0.97, "safety": 0.03}
    recs = mi.capacity_records(units, {"s": shares})
    safety = [r for r in recs if r["type"] == "capacity" and r["area"] == "safety" and r["lens"] == mi.LENS_RESERVED]
    assert safety and all(r["status"]["1x"].startswith("INFEASIBLE") for r in safety)
    assert {r["budget_tokens"] for r in safety} == set(mi.BUDGETS)
    assert safety[0]["requested_tokens"] == pytest.approx(mi.BUDGETS[0] * 0.03)
    total = next(r for r in recs if r["type"] == "capacity_total")
    assert total["share_problems"] == []        # 0.97 + 0.03 sums to 1; it is the source, not the share, that fails


def test_composition_fractions_and_known_sets_are_consistent():
    for area, groups in mi.COMPOSITION.items():
        assert sum(f for _, f, _ in groups) == pytest.approx(1.0), area
    assert mi.check_shares(mi.PLAN_SCENARIO, mi.COMPOSITION) == []
    assert "sft-code-self-oss-instruct" not in {s for g in mi.COMPOSITION.values() for _, _, ms in g for s in ms}


# --- parsers ---------------------------------------------------------------------------------------------

LICENCE = """# Licence table

### Stage 1: Supervised fine-tuning (SFT) instruction mixture

| Set | Source | Language | Generator model | Licence | Decision |
| --- | --- | --- | --- | --- | --- |
| sft-code-commitpackft | [bigcode/commitpackft](https://x/y) | Code | none (human) | mit (tag) | TSAI reserves |
| sft-code-opencodereasoning-2 | nvidia/OpenCodeReasoning-2 | Code | DeepSeek-R1 (MIT) \\| QwQ | cc-by-4.0 | TSAI reserves |
| sft-safety-wildjailbreak | allenai/wildjailbreak | English | GPT-4 closed | ODC-By | excluded |
| sft-dolly-15k | databricks/dolly | English | none | cc-by-sa-3.0 | allowed (CC-BY-SA, attribution needed) |

### Stage 2: Preference optimization and RLVR

| Set | Source |
| --- | --- |
| rl-x | y |
"""

DECONTAM = """# SFT decontamination report

- **set_sha256:** `""" + "b" * 64 + """`
- **Sets run:** 2. **Rows in:** 1,105. **Removed (flagged or canary):** 25 (2.26%). **Kept:** 1,080.

## 1. Result per set

| Set | Run on | Rows in | Flagged | Canary | Removed | Rows kept | Clean file sha256 |
|---|---|---:|---:|---:|---:|---:|---|
| `sft-code-commitpackft` | full set | 1,000 | 20 | 0 | 20 (2.00%) | 980 | `0cb076d440c8…`, `6c53baddfea0…` |
| `sft-code-opencodereasoning-2` | all 70 python shards, questions joined | 105 | 5 | 0 | 5 (4.76%) | 100 | — |
| **Total** | | **1,105** | **25** | **0** | **25** (2.26%) | **1,080** | |

## 2. Flagged documents by benchmark

## 6. Inputs (to reproduce a run)

| Set | Input file | Bytes | sha256 |
|---|---|---:|---|
| `sft-code-commitpackft` | `a.jsonl` | 1,234 | `""" + "c" * 64 + """` |

## 7. Sets not run (excluded)
"""

KEPT = """# Rows kept

| Set | Rows in | Removed | Removed % | Rows kept |
|---|---:|---:|---:|---:|
| `sft-code-commitpackft` | 1,000 | 20 | 2.00% | 980 |
| **1 sets** | **1,000** | **20** | **2.00%** | **980** |
"""

PROPOSAL = """# Proposal

## 2. Proposed SFT Mixture Shares by Area

| Area | Proposed Share (%) | Role | Datasets |
| :--- | :---: | :--- | :--- |
| **1. Code** *(Generation)* | **35%** | role | `sft-code-commitpackft`, `sft-code-opencodereasoning-2` |
| **6. Safety** *(English)* | **3% proposed** | role | `sft-safety-wildjailbreak` |
| **Total** | **100%** | | |

## 3. Rationale

## 7. Revision note (6 Nov): Indic quantity check

| Set | Rows | Notes |
| :--- | ---: | :--- |
| `sft-indic-aya-dataset` | 41,261 | Tamil 14,133, Telugu 8,439, **Hindi 1,153**, Kannada 334, **Odia 0**. Single turn |

## 8. Revision note
"""


def fake_report():
    outcomes = {"eligible_joined": 105, "held_source_unavailable": 7}
    return {
        "counts": {"totals": {"joined.raw_rows": 112, "joined.policy_excluded": 0, "joined.malformed_or_unresolved": 7,
                              "joined.eligible_joined": 105, "rendered.converter_rejected": 0, "rendered.rendered": 105,
                              "screened.rows_removed": 5, "screened.rows_canary": 0, "screened.rows_kept": 100,
                              "pack.admitted": 0, "pack.attempted": 100},
                   "outcomes": outcomes, "by_source": {"taco/train": {"eligible_joined": 105},
                                                       "open-r1/codeforces/train": {"held_source_unavailable": 7}},
                   "codeforces_train": {"rows": 7}, "tokens": {"rendered_tokens": 9, "rendered_trained_tokens": 8},
                   "states": {"complete": 1}, "total_identities": {}},
        "dedup": {"tokens": {"kept_tokens": 1000, "kept_trained_tokens": 900, "kept_positions": 40,
                             "kept_content_clusters_C08": 39},
                  "kept": {"P": {"by_source_positions": {"taco": 40}}}},
        "shards_loaded": [0], "sources": {"0": "results/shard-00000.json.gz"}}


def mini_team(tmp_path):
    team = tmp_path / "team"
    for rel, text in (("licences/licence_table.md", LICENCE), ("sft/decontam_report.md", DECONTAM),
                      ("sft/decontam_kept_rows.md", KEPT), ("sft/sft_mix_proposal.md", PROPOSAL)):
        path = team / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    out = team / "sft/results/ocr2_full_run.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fake_report()), encoding="utf-8")
    return team


def test_split_cells_keeps_escaped_pipes():
    assert mi.split_cells("| a | b \\| c | d |") == ["a", "b \\| c", "d"]


def test_parse_licence_stage1_stops_at_stage_2():
    lic = mi.parse_licence_stage1(LICENCE)
    assert sorted(lic) == ["sft-code-commitpackft", "sft-code-opencodereasoning-2", "sft-dolly-15k",
                           "sft-safety-wildjailbreak"]
    assert lic["sft-safety-wildjailbreak"]["decision"] == "excluded"
    assert lic["sft-dolly-15k"]["decision"] == "allowed"
    assert lic["sft-code-commitpackft"]["decision"] == "tsai_reserves"
    with pytest.raises(ValueError):
        mi.parse_licence_stage1(LICENCE.replace("| sft-dolly-15k | databricks/dolly | English | none | cc-by-sa-3.0 | "
                                                 "allowed (CC-BY-SA, attribution needed) |",
                                                 "| sft-code-commitpackft | a | b | c | d | excluded |"))


def test_parse_decontam_header_tolerates_a_percent_with_a_dot():
    rows, head = mi.parse_decontam_results(DECONTAM)
    assert head["sets_run"] == 2 and head["rows_in"] == 1105 and head["removed"] == 25 and head["kept"] == 1080
    assert [r["set_id"] for r in rows] == ["sft-code-commitpackft", "sft-code-opencodereasoning-2"]
    assert rows[0]["clean_sha256_prefixes"] == ["0cb076d440c8", "6c53baddfea0"]
    assert rows[1]["rows_in"] == 105 and rows[1]["kept"] == 100
    with pytest.raises(ValueError):
        mi.parse_decontam_results(DECONTAM.replace("**Sets run:**", "**Sets:**"))


def test_parse_proposal_and_aya_languages():
    prop = mi.parse_proposal(PROPOSAL)
    assert prop["areas"]["code"]["share"] == pytest.approx(0.35)
    assert prop["areas"]["safety"]["share"] == pytest.approx(0.03)
    assert prop["areas"]["code"]["sets"] == ["sft-code-commitpackft", "sft-code-opencodereasoning-2"]
    assert prop["aya_dataset_language_rows"] == {"Tamil": 14133, "Telugu": 8439, "Hindi": 1153, "Kannada": 334, "Odia": 0}
    cells = mi.indic_language_cells(prop["aya_dataset_language_rows"])
    assert cells[-1]["language"] == "Odia" and cells[-1]["flag"] == "zero rows"
    assert [c["flag"] for c in cells if c["language"] == "Kannada"] == ["under 1,000 rows"]


def test_parse_kept_rows_total_row_is_separate():
    rows, total = mi.parse_kept_rows(KEPT)
    assert rows == {"sft-code-commitpackft": {"rows_in": 1000, "removed": 20, "kept": 980}}
    assert total == {"rows_in": 1000, "removed": 20, "kept": 980}


def test_percentile_follows_the_pack_admission_rule():
    v = list(range(1, 101))
    assert mi.percentile(v, 0.5) == 51 and mi.percentile(v, 0.95) == 96 and mi.percentile(v, 1.0) == 100
    assert mi.percentile([], 0.5) is None
    assert mi.length_stats([5, 1, 3])["max"] == 5


# --- building units and contradictions --------------------------------------------------------------------

def build(tmp_path, kept_rows=None):
    team = mini_team(tmp_path)
    lic = mi.parse_licence_stage1(LICENCE)
    results, head = mi.parse_decontam_results(DECONTAM)
    inputs = mi.parse_decontam_inputs(DECONTAM)
    kept, _ = mi.parse_kept_rows(KEPT)
    if kept_rows is not None:
        kept = kept_rows
    prop = mi.parse_proposal(PROPOSAL)
    hashes = {"fingerprint_set_sha256": head["set_sha256"],
              "ocr2": mi.ocr2_hashes(team / "sft/results/ocr2_full_run.json", None)}
    return mi.build_units(lic, results, inputs, kept, prop, fake_report(), None, {}, hashes)


def test_build_units_classes_and_hashes(tmp_path):
    units, _ = build(tmp_path)
    u = {x["set_id"]: x for x in units}
    assert u["sft-safety-wildjailbreak"]["coverage_class"] == "excluded"
    assert u["sft-dolly-15k"]["coverage_class"] == "held"            # not excluded, never run
    assert u["sft-code-commitpackft"]["coverage_class"] == "full"
    assert u["sft-code-commitpackft"]["tokens"]["status"] == "not_measured"
    assert u["sft-code-commitpackft"]["hashes"]["input_files_sha256"] == ["c" * 64]
    assert u["sft-code-commitpackft"]["hashes"]["clean_sha256_prefixes"] == ["0cb076d440c8", "6c53baddfea0"]
    ocr = u["sft-code-opencodereasoning-2"]
    assert ocr["coverage_class"] == "partial"
    assert ocr["tokens"]["sequence_tokens"] == 1000 and ocr["rows"]["stages"]["kept"] == 100
    assert "tokens_without_tokenizer_hash" in ocr["problems"]       # the aggregate record has no tokenizer hash


def test_kept_rows_table_disagreement_is_a_contradiction(tmp_path):
    _, contra = build(tmp_path, kept_rows={"sft-code-commitpackft": {"rows_in": 1000, "removed": 20, "kept": 979}})
    kinds = {c["kind"] for c in contra}
    assert "kept_rows_table_differs" in kinds


def test_excluded_set_with_a_run_record_is_a_contradiction(tmp_path):
    team = mini_team(tmp_path)
    lic = mi.parse_licence_stage1(LICENCE.replace("| sft-code-commitpackft | [bigcode/commitpackft](https://x/y) | Code "
                                                  "| none (human) | mit (tag) | TSAI reserves |",
                                                  "| sft-code-commitpackft | [bigcode/commitpackft](https://x/y) | Code "
                                                  "| none (human) | mit (tag) | excluded |"))
    results, head = mi.parse_decontam_results(DECONTAM)
    kept, _ = mi.parse_kept_rows(KEPT)
    hashes = {"fingerprint_set_sha256": head["set_sha256"],
              "ocr2": mi.ocr2_hashes(team / "sft/results/ocr2_full_run.json", None)}
    _, contra = mi.build_units(lic, results, mi.parse_decontam_inputs(DECONTAM), kept, mi.parse_proposal(PROPOSAL),
                               fake_report(), None, {}, hashes)
    assert "excluded_set_has_run_record" in {c["kind"] for c in contra}


def test_token_record_must_match_kept_rows(tmp_path):
    team = mini_team(tmp_path)
    good = {"set_id": "sft-code-commitpackft", "rows": 980, "sequence_tokens": 5000, "shifted_supervised_targets": 4000,
            "tokenizer_sha256": TOKENIZER}
    bad = dict(good, rows=981)
    path = tmp_path / "tokens.jsonl"
    path.write_text(json.dumps(good) + "\n", encoding="utf-8")
    recs = mi.collect(team, None, path)
    unit_ = next(r for r in recs if r["type"] == "set" and r["set_id"] == "sft-code-commitpackft")
    assert unit_["tokens"]["status"] == "measured" and unit_["tokens"]["sequence_tokens"] == 5000
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    recs = mi.collect(team, None, path)
    unit_ = next(r for r in recs if r["type"] == "set" and r["set_id"] == "sft-code-commitpackft")
    assert unit_["tokens"]["status"] == "not_measured"
    assert "token_record_rows_differ" in {r["kind"] for r in recs if r["type"] == "contradiction"}


def test_ledger_lines_are_valid_json_with_known_types(tmp_path):
    lines = mi.dump_jsonl(mi.collect(mini_team(tmp_path), None, None)).splitlines()
    types = {json.loads(x)["type"] for x in lines}
    assert {"meta", "set", "capacity", "capacity_total", "contradiction", "language_cell"} <= types
    meta = json.loads(lines[0])
    assert meta["schema"] == mi.SCHEMA and meta["counts"]["approved_complete_training_packs"] == 0
    assert all(c in mi.COVERAGE for c in meta["counts"]["coverage"])


# --- OCR-2 shard records ------------------------------------------------------------------------------------

SCHEMA_COLS = ["joined_ordinal", "ocr_id", "ocr_question_id", "dataset", "split", "source_ordinal", "family_sha256",
               "norm_sha16", "rendered", "kept", "tokens", "trained_tokens", "rendered_text_sha16",
               "n_family_sources_in_index"]


def shard_row(i, dataset, family, kept, tokens, trained, rendered=1):
    return [i, f"id{dataset}{i}", "q", dataset, "train", i, family, "n", rendered, kept, tokens, trained, "t", 1]


def write_shard(folder, sid, rows, blockers=("tokenizer_not_approved",)):
    summary = {"rows_schema": SCHEMA_COLS, "rows": rows, "run_id": "run1", "sig_sha256": "s", "config_sha256": "c",
               "shard": {"counts": {"rendered": {"effort": {"1k": 1, "4k": 1, "16k": 0, "64k": 0}},
                                    "pack": {"blockers": list(blockers), "passed_rows": {"length_max": 9}}}}}
    path = folder / f"shard-{sid:05d}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as f:
        json.dump({"shard_id": sid, "summary": summary, "download": {"sha256": "d", "bytes": 1}}, f)
    return path.name


def test_ocr2_shard_reader_recomputes_kept_statistics(tmp_path):
    names = {0: write_shard(tmp_path, 0, [shard_row(0, "taco", "f1", 1, 100, 90), shard_row(1, "taco", "f1", 1, 300, 250),
                                         shard_row(2, "apps", "f2", 0, 999, 900)]),
             1: write_shard(tmp_path, 1, [shard_row(3, "taco", "f3", 1, 500, 400), shard_row(4, "apps", "f4", 1, 200, 150)])}
    rep = {"sources": {str(k): v for k, v in names.items()}}
    rows = mi.ocr2_from_shards(tmp_path, rep)
    assert rows["kept_rows"] == 4 and rows["kept_tokens"] == 1100 and rows["kept_trained_tokens"] == 890
    assert rows["rendered"] == {"rows": 5, "tokens": 2099, "trained": 1790}
    assert rows["per_source"]["taco"] == {"rows": 3, "tokens": 900, "trained": 740, "families": 2}
    assert rows["kept_families_by_position"] == 3 and rows["family_union_equals_sum"] is True   # f1 twice
    assert rows["kept_length"]["max"] == 500 and rows["kept_length"]["n"] == 4
    assert rows["kept_caps"]["1024"] == {"rows": 4, "tokens": 1100}
    assert rows["pack_blockers"] == ["tokenizer_not_approved"]
    assert rows["pack_blockers_identical_in_every_shard"] is True


def test_ocr2_shard_reader_rejects_a_duplicate_row_id(tmp_path):
    dup = shard_row(0, "taco", "f1", 1, 100, 90)
    names = {0: write_shard(tmp_path, 0, [dup]), 1: write_shard(tmp_path, 1, [dup])}
    with pytest.raises(ValueError):
        mi.ocr2_from_shards(tmp_path, {"sources": {str(k): v for k, v in names.items()}})


def test_ocr2_recompute_mismatch_is_reported(tmp_path):
    """The aggregate says 100 kept rows; the shard records hold 4: the ledger must say so, not pick one."""
    lic = mi.parse_licence_stage1(LICENCE)
    results, head = mi.parse_decontam_results(DECONTAM)
    names = {0: write_shard(tmp_path, 0, [shard_row(i, "taco", f"f{i}", 1, 100, 90) for i in range(4)])}
    rows = mi.ocr2_from_shards(tmp_path, {"sources": {"0": names[0]}})
    team = mini_team(tmp_path / "t")
    hashes = {"fingerprint_set_sha256": head["set_sha256"],
              "ocr2": mi.ocr2_hashes(team / "sft/results/ocr2_full_run.json", None)}
    kept, _ = mi.parse_kept_rows(KEPT)
    units, contra = mi.build_units(lic, results, mi.parse_decontam_inputs(DECONTAM), kept, mi.parse_proposal(PROPOSAL),
                                   fake_report(), rows, {}, hashes)
    msgs = [c["detail"] for c in contra if c["kind"] == "ocr2_recompute_mismatch"]
    assert any("kept rows" in m for m in msgs) and any("kept tokens" in m for m in msgs)
