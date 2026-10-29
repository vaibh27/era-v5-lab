"""perfkit/report.py: compare() refuses a baseline pair that does not time the same committed code."""
from __future__ import annotations

import copy

from perfkit import report
from perfkit.schema import example_run


def _pair(**b_meta):
    a = example_run()
    b = copy.deepcopy(a)
    b["meta"].update(tag="b", **b_meta)
    return a, b


def test_same_clean_commit_passes():
    _, ok, problems = report.compare(*_pair())
    assert ok and problems == []


def test_same_commit_muon_implementation_variants_can_be_compared():
    a, b = _pair()
    a["config"]["optim"].update(name="muon", muon_impl="loop")
    b["config"]["optim"].update(name="muon", muon_impl="batched")
    _, ok, problems = report.compare(a, b)
    assert ok and problems == []


def test_same_commit_moe_dispatch_variants_can_be_compared():
    a, b = _pair()
    a["config"]["model"]["moe"] = {"dispatch": "loop", "num_experts": 8}
    b["config"]["model"]["moe"] = {"dispatch": "sorted", "num_experts": 8}
    _, ok, problems = report.compare(a, b)
    assert ok and problems == []
    assert "model.moe.dispatch: loop vs sorted" in report.ab_note(a, b)
    assert "A/B" in report.baseline_md(a, b, [], ok, problems)


def test_same_implementation_is_not_labelled_ab():
    a, b = _pair()
    assert report.ab_note(a, b) is None
    assert "A/B" not in report.baseline_md(a, b, [], True, [])


def test_moe_setting_other_than_dispatch_is_still_refused():
    a, b = _pair()
    a["config"]["model"]["moe"] = {"dispatch": "loop", "num_experts": 8}
    b["config"]["model"]["moe"] = {"dispatch": "sorted", "num_experts": 4}
    _, ok, problems = report.compare(a, b)
    assert not ok and "different model config" in problems


def test_different_commit_is_refused():
    _, ok, problems = report.compare(*_pair(commit="1" * 40))
    assert not ok and any("different commit" in p for p in problems)


def test_dirty_or_unknown_tree_is_refused():
    for dirty in (True, None):
        _, ok, problems = report.compare(*_pair(dirty=dirty))
        assert not ok and any(f"dirty={dirty}" in p for p in problems)


# ------------------------------------------------------------------------------ exact timing diagnostics in Markdown

def test_render_shows_per_stage_cuda_allocation_diagnostics():
    run = example_run()
    run["clean"]["events_created_during_steps"] = 0
    run["layer"] = {"events_created_during_steps": 2,
                    "inner_pass": {"events_created_during_steps": 4}}
    run["comm"] = {"moe_a2a": {"events_created_during_steps": 3}}
    run["warnings"] = ["9 CUDA timing events were created during measured steps"]
    md = report.render(run)
    assert "[OK] clean: events_created_during_steps=0" in md
    assert "[WARN] layer: events_created_during_steps=2" in md
    assert "[WARN] layer.inner_pass: events_created_during_steps=4" in md
    assert "[WARN] communication.max_rank: events_created_during_steps=3" in md
    assert "[warn] 9 CUDA timing events were created during measured steps" in md


def test_render_preserves_exact_warning_text_and_no_false_ke_claim():
    run = example_run()
    run["warnings"] = ["recorder overhead 3.2% > 2.0%: layer numbers are flagged"]
    md = report.render(run)
    assert "[warn] recorder overhead 3.2% > 2.0%: layer numbers are flagged" in md
    assert md.count("[warn] recorder overhead 3.2%") == 1


# Sunil PR24 review regression: distinct pool-counter diagnostics
def test_render_flags_inner_pool_allocation_when_timer_count_is_zero():
    run = example_run()
    run["layer"] = {"events_created_during_steps": 0,
                    "inner_pass": {"events_created_during_steps": 0,
                                   "events_created_while_measuring": 57}}
    run["warnings"] = ["57 recorder-pool CUDA events were created during layer.inner_pass measured steps"]
    md = report.render(run)
    assert "[OK] layer: events_created_during_steps=0" in md
    assert "[WARN] layer.inner_pass.recorder_pool: events_created_while_measuring=57" in md
    assert "[warn] 57 recorder-pool CUDA events" in md


def test_compare_gate_verdict_min_abs_ms():
    """report.compare uses the same rule as bench.quick_compare (report.gate_verdict)."""
    assert report.gate_verdict(10.0, 10.4, 5.0) == (report._pct_delta(10.0, 10.4), "ok")
    assert report.gate_verdict(0.031, 0.035, 5.0)[1] == "FAIL"               # default: no floor (TSAI not agreed)
    assert report.gate_verdict(0.031, 0.035, 5.0, min_abs_ms=0.05)[1] == "small"     # 4 us under the proposed floor
    assert report.gate_verdict(1.0, 1.06, 5.0, min_abs_ms=0.05)[1] == "FAIL"         # 60 us and 6%
    assert report.gate_verdict(0.0, 0.0, 5.0)[1] == "ok"
