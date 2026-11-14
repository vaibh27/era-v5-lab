"""Tests for eval/effort_suite.py (CPU, no model, no network).

    python -m pytest teams/10-post-training/eval/test_effort_suite.py -q -p no:cacheprovider

Completions below are code-made fixtures for the parser, not prompts or training text.
"""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest
from pytest import fixture  # bare decorators: pr-guard reads "+@a.b" as an email

parametrize = pytest.mark.parametrize

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("g10_effort_suite_test", HERE / "effort_suite.py")
es = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = es  # dataclasses look the module up by name
_spec.loader.exec_module(es)
T = es.T
MANIFEST = HERE / "effort_manifest.json"


def count(text: str) -> int:
    """Fixture tokenizer: one token per character."""
    return len(text)


class FakeBackend(es.Backend):
    """Prompt ids are the characters of `label|text`; generation replays a scripted completion."""

    name, template, model_id = "fake", "g10", "fake"

    def __init__(self, completion: str, stop: bool = True, context: int = 100_000):
        self.completion, self.stop, self.context = completion, stop, context
        self.calls: list[tuple] = []

    def prompt_ids(self, text, label):
        return [ord(c) for c in f"{label}|{text}"]

    def generate(self, ids, cap, seed, temperature):
        self.calls.append((len(ids), cap, seed))
        body = self.completion[:cap]
        stopped = self.stop and len(self.completion) < cap
        new = [ord(c) for c in body] + ([0] if stopped else [])
        return es.Generation(new, body, "stop:0" if stopped else "length")

    def count(self, text):
        return len(text)


@fixture(scope="module")
def manifest():
    return es.load_manifest(MANIFEST)


def prompts_for(manifest, checker="math"):
    return {s["slot"]: es.Prompt(s["slot"], s["category"], s["task_id"], f"problem {s['slot']}", checker,
                                 {"id": s["task_id"]})
            for s in manifest["slots"] if s["status"] == "resolved"}


def one_cell(manifest, backend, checker=None, why=None, stage="A"):
    prompts = prompts_for(manifest)
    cell = es.iter_matrix(manifest, stage)[0]
    caps = es.plan_caps(backend, prompts, [cell], stage, {}, 1024)
    return es.run_cell(backend, prompts[cell.slot], cell, caps[(cell.slot, cell.label)], 0.0, None, checker, why)


# --- empty completion ---------------------------------------------------------------------------------

def test_empty_completion(manifest):
    rec = one_cell(manifest, FakeBackend(""), checker=lambda task, text: {"reward": 0, "reason": "no answer found"})
    assert rec["ran"] and rec["stop_reason"] == "stop:0" and rec["generated_tokens"] == 1  # only the stop id
    assert rec["parse_status"] == "empty" and rec["trace_tokens"] is None and rec["raw_completion_tokens"] == 0
    assert rec["functional"]["success"] == 0 and not rec["censored"]
    s = es.summarise([rec], manifest, "A")["per_label"][rec["requested_effort"]]
    assert s["parse_ok"] == {"k": 0, "n": 1, "rate": 0.0} and s["empty"] == 1
    assert s["functional_success"] == {"k": 0, "n": 1, "rate": 0.0}


# --- over-cap censoring -------------------------------------------------------------------------------

def test_over_cap_is_censored_and_stays_in_denominator(manifest):
    long = "x" * 5000
    good = FakeBackend("\\boxed{1}")
    capped = FakeBackend(long, stop=False)
    ok = lambda task, text: {"reward": int("boxed" in text), "reason": ""}  # noqa: E731
    r_ok = one_cell(manifest, good, ok)
    r_cap = one_cell(manifest, capped, ok)
    assert capped.calls[0][1] == es.STAGE_CAP
    assert r_cap["stop_reason"] == "length" and r_cap["truncated"] and r_cap["censored"]
    assert r_cap["generated_tokens"] == es.STAGE_CAP and r_cap["enforced_cap"] == es.STAGE_CAP
    r_cap["requested_effort"] = r_ok["requested_effort"]
    s = es.summarise([r_ok, r_cap], manifest, "A")["per_label"][r_ok["requested_effort"]]
    assert s["functional_success"] == {"k": 1, "n": 2, "rate": 0.5}   # censored row kept as a failure
    assert s["functional_success_censored_in_denominator"] == 1
    assert s["censored_at_cap"] == {"k": 1, "n": 2, "rate": 0.5}


def test_requested_effort_is_separate_from_enforced_cap(manifest):
    backend = FakeBackend("done")
    prompts = prompts_for(manifest)
    cells = es.iter_matrix(manifest, "B")
    caps = es.plan_caps(backend, prompts, cells, "B", {}, 1024)
    recs = [es.run_cell(backend, prompts[c.slot], c, caps[(c.slot, c.label)], 0.0, None, None, "n/a")
            for c in cells[:8]]
    assert {r["enforced_cap"] for r in recs} == {1024}
    assert [r["requested_budget_tokens"] for r in recs[:4]] == [1024, 4096, 16384, 65536]


def test_context_lowers_common_cap_per_prompt_and_marks_unsupported(manifest):
    prompts = prompts_for(manifest)
    cells = [c for c in es.iter_matrix(manifest, "B") if c.slot == "code-00"]
    plen = max(len(FakeBackend("").prompt_ids(prompts["code-00"].text, lab)) for lab in es.LABELS)
    caps = es.plan_caps(FakeBackend("", context=plen + 300), prompts, cells, "B", {}, 1024)
    assert {v["cap"] for v in caps.values()} == {300} and all(v["supported"] for v in caps.values())
    assert all("lowered to 300" in v["note"] for v in caps.values())
    caps = es.plan_caps(FakeBackend("", context=plen), prompts, cells, "B", {}, 1024)
    assert not any(v["supported"] for v in caps.values())
    # stage C: a ceiling below the label budget cannot test that label
    caps = es.plan_caps(FakeBackend("", context=plen + 5000), prompts, cells, "C", {}, 1024)
    assert [caps[("code-00", lab)]["supported"] for lab in es.LABELS] == [True, True, False, False]
    assert caps[("code-00", "1k")]["cap"] == 1024 + 1024


# --- malformed traces ---------------------------------------------------------------------------------

@parametrize("text, why", [
    (f"{T['think']}\nstill thinking", "missing"),                                  # unclosed think
    (f"{T['think']}\nplan\n{T['think_end']}\n{T['think']}\nagain", "missing"),      # second span unclosed
    (f"{T['tool_call']}\n{{not json\n{T['tool_call_end']}\n", "not JSON"),          # bad tool JSON
    (f"answer {T['think']} late marker", ""),                                       # marker inside content
])
def test_malformed_trace_is_never_zero_length(text, why):
    a = es.analyse(text, count)
    assert a["parse_status"] == "malformed" and why in a["parse_error"]
    assert a["trace_tokens"] is None and a["raw_completion_tokens"] == len(text) > 0
    if T["tool_call"] in text:
        assert a["tool_call_valid"] is False


def test_well_formed_trace_counts_markers_like_the_converter():
    think = f"{T['think']}\nstep one\n{T['think_end']}"
    call = f"{T['tool_call']}\n{{\"name\": \"f\", \"arguments\": {{}}}}\n{T['tool_call_end']}"
    a = es.analyse(f"{think}\n{call}\nfinal \\boxed{{2}}", count)
    assert a["parse_status"] == "ok" and a["trace_tokens"] == len(think) + len(call)
    assert a["n_tool_calls"] == 1 and a["tool_call_valid"] is True
    plain = es.analyse("no trace at all", count)
    assert plain["parse_status"] == "ok" and plain["trace_tokens"] == 0 and plain["tool_call_valid"] is None


def test_censored_unclosed_trace_reports_parser_failure(manifest):
    rec = one_cell(manifest, FakeBackend(f"{T['think']}\n" + "y" * 3000, stop=False))
    assert rec["censored"] and rec["parse_status"] == "malformed" and rec["think_unclosed"]
    assert rec["trace_tokens"] is None and rec["raw_completion_tokens"] == es.STAGE_CAP


# --- manifest hash ------------------------------------------------------------------------------------

def test_committed_manifest_hash_verifies(manifest):
    raw = json.loads(MANIFEST.read_text(encoding="utf-8"))
    body = {k: v for k, v in raw.items() if k != "manifest_sha256"}
    canon = json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    assert hashlib.sha256(canon).hexdigest() == raw["manifest_sha256"] == manifest["manifest_sha256"]
    assert sum(manifest["composition"].values()) == 100 and len(manifest["slots"]) == 100


@parametrize("mutate", [
    lambda m: m["slots"][0].update(text_sha256="0" * 64),
    lambda m: m["slots"][0].update(row_index=m["slots"][0]["row_index"] + 1),
    lambda m: m["stages"]["A"]["slots"].pop(),
    lambda m: m["training_exclusion"]["task_ids"].pop(),
    lambda m: m.update(labels=["1k", "4k", "16k"]),
])
def test_tampered_manifest_is_rejected(mutate):
    bad = copy.deepcopy(json.loads(MANIFEST.read_text(encoding="utf-8")))
    mutate(bad)
    with pytest.raises(es.ManifestError, match="does not verify"):
        es.validate(bad)


@parametrize("mutate, msg", [
    (lambda m: m["stages"]["A"]["slots"].pop(), "exactly 10"),
    (lambda m: m["training_exclusion"]["task_ids"].pop(), "training_exclusion"),
    (lambda m: m.update(labels=["1k", "4k", "16k"]), "labels"),
    (lambda m: m["slots"].pop(), "composition"),
    (lambda m: m["slots"][-1].update(reason=""), "need a reason"),
])
def test_resealed_bad_manifest_fails_structure(mutate, msg):
    bad = copy.deepcopy(json.loads(MANIFEST.read_text(encoding="utf-8")))
    mutate(bad)
    with pytest.raises(es.ManifestError, match=msg):
        es.validate(es.seal(bad))


def test_manifest_records_sources_and_unavailable_slots(manifest):
    resolved = [s for s in manifest["slots"] if s["status"] == "resolved"]
    unavailable = [s for s in manifest["slots"] if s["status"] == "unavailable"]
    assert {s["category"] for s in resolved} <= {"code", "math"}
    assert all(s["licence"] and s["revision"] and s["file_sha256"] and s["generator"].startswith("none")
               for s in resolved)
    assert all(s["reason"] and "generated" in s["reason"] for s in unavailable)
    assert "text" not in json.dumps(resolved).replace("text_sha256", "")  # no prompt text stored
    assert len({s["group"] for s in resolved}) == len(resolved)


# --- matrix -------------------------------------------------------------------------------------------

def test_four_tier_matrix_is_deterministic(manifest):
    a, b = es.iter_matrix(manifest, "B"), es.iter_matrix(manifest, "B")
    assert a == b and len(a) == 4 * 55
    assert [c.label for c in a[:8]] == list(es.LABELS) * 2 == ["1k", "4k", "16k", "64k"] * 2
    for i in range(0, len(a), 4):  # slot-major, one seed per prompt shared by all labels
        assert len({c.slot for c in a[i:i + 4]}) == 1 and len({c.seed for c in a[i:i + 4]}) == 1
    assert len({c.seed for c in a}) == 55
    assert [c.slot for c in a[::4]] == [s["slot"] for s in manifest["slots"] if s["status"] == "resolved"]


def test_stages(manifest):
    a = es.iter_matrix(manifest, "A")
    assert len(a) == 40 and len({c.slot for c in a}) == 10
    assert {c.slot.split("-")[0] for c in a} == {"code", "math"}
    assert len(es.iter_matrix(manifest, "C")) == 32
    unavailable = {s["slot"] for s in manifest["slots"] if s["status"] == "unavailable"}
    assert not unavailable & {c.slot for c in es.iter_matrix(manifest, "B")}


def test_stage_c_is_off_by_default():
    parser = es.build_parser()
    assert parser.parse_args(["run", "--backend", "hf", "--out", "o"]).stage == "A"
    assert parser.parse_args(["plan"]).stage == "A"
    assert parser.parse_args(["run", "--backend", "hf", "--out", "o", "--stage", "C"]).stage == "C"


# --- prompts, checkers, outputs -----------------------------------------------------------------------

def test_prompt_is_raw_text_unless_team_instruction_given():
    assert es.apply_instruction("Solve it.", None) == "Solve it."
    assert es.apply_instruction("Solve it.", "Q: {problem}\nA:") == "Q: Solve it.\nA:"
    with pytest.raises(ValueError):
        es.apply_instruction("x", "no placeholder")


def test_missing_checker_gives_none_not_zero(manifest, tmp_path):
    fns, why, _ = es.load_checkers(tmp_path, execute_code=True)
    assert fns["math"] is None and "not found" in why["math"]
    rec = one_cell(manifest, FakeBackend("\\boxed{3}"), None, why["math"])
    assert rec["functional"]["success"] is None and "not found" in rec["functional"]["reason"]
    s = es.summarise([rec], manifest, "A")["per_label"][rec["requested_effort"]]
    assert s["functional_success"]["n"] == 0 and s["no_checker"] == 1
    _, why, _ = es.load_checkers(tmp_path, execute_code=False)
    assert why["stdio"] == "code execution disabled (--no-exec)"


def test_run_stage_outputs_are_marked_and_guard_catches_them(manifest, tmp_path):
    prompts = prompts_for(manifest)
    s = es.run_stage(manifest, "A", FakeBackend("\\boxed{1}"), prompts, tmp_path / "out",
                     {"math": lambda t, c: {"reward": 1, "reason": "ok"}}, {}, {"math": None})
    lines = [json.loads(x) for x in (tmp_path / "out" / "effort_stageA.jsonl").read_text().splitlines()]
    assert len(lines) == 40 and all(r["do_not_train"] and r["artifact"] == es.OUTPUT_ARTIFACT for r in lines)
    assert all(r["manifest_sha256"] == manifest["manifest_sha256"] for r in lines)
    assert s["do_not_train"] and len(s["cells_ran"]) == 40 and s["unavailable_slots"] == 45
    assert es.guard(manifest, [tmp_path / "out" / "effort_stageA.jsonl"])  # outputs are refused as training data
    train = tmp_path / "train.jsonl"
    train.write_text(json.dumps({"id": "unrelated", "text": "hello"}) + "\n")
    assert es.guard(manifest, [train]) == []
    train.write_text(json.dumps({"id": manifest["training_exclusion"]["task_ids"][0]}) + "\n")
    assert "eval task id" in es.guard(manifest, [train])[0]
    train.write_text(json.dumps({"messages": [{"content": "prefix problem code-00 suffix"}]}) + "\n")
    assert es.guard(manifest, [train], prompt_texts=["problem code-00"])


def test_compare_refuses_different_manifests(manifest):
    a = {"manifest_sha256": "a", "stage": "B", "cells_ran": [], "per_label": {}}
    with pytest.raises(ValueError):
        es.compare(a, {**a, "manifest_sha256": "b"})


# --- resolver against a tiny parquet ------------------------------------------------------------------

def test_resolver_checks_file_and_text_hashes(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    d = tmp_path / "src"
    d.mkdir()
    pq.write_table(pa.table({"problem": ["p0", "p1"], "solution": ["\\boxed{0}", "so \\boxed{7}"]}), d / "f.parquet")
    slot = {"slot": "math-00", "category": "math", "status": "resolved", "task_id": "t1", "group": "g",
            "source": "s", "dataset": "x", "revision": "r", "split": "train", "file": "f.parquet",
            "file_sha256": es.sha256_file(d / "f.parquet"), "row_index": 1, "field": "problem",
            "checker_fields": ["solution"], "text_sha256": es.sha256_text("p1"), "checker": "math",
            "answer_sha256": es.sha256_text("7"), "licence": "MIT", "licence_decision": "x", "generator": "none"}
    m = {"sources": {"s": {"dataset": "x", "revision": "r"}}, "slots": [slot]}
    lb = lambda s: s.rsplit("{", 1)[1].rstrip("}")  # noqa: E731  fixture extractor
    got = es.resolve_prompts(m, {"s": d}, last_boxed=lb)["math-00"]
    assert got.text == "p1" and got.task["answer"] == "7"
    for key, val, msg in [("file_sha256", "0" * 64, "pinned"), ("text_sha256", "0" * 64, "prompt text"),
                          ("answer_sha256", "0" * 64, "reference answer")]:
        bad = {**m, "slots": [{**slot, key: val}]}
        with pytest.raises(es.SourceError, match=msg):
            es.resolve_prompts(bad, {"s": d}, last_boxed=lb)
    with pytest.raises(es.SourceError, match="no --source-dir"):
        es.resolve_prompts(m, {})
