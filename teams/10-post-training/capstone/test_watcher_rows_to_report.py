"""Tests of capstone/watcher_rows_to_report.py (no torch, no network): Team 9's eval_mini_v1 rows in points become the
fraction block forgetting.py accepts, and anything ambiguous is refused.

    python -m pytest capstone/test_watcher_rows_to_report.py -q
"""
from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import watcher_rows_to_report as w  # noqa: E402

_spec = importlib.util.spec_from_file_location("forgetting", HERE.parent / "sft" / "tests" / "forgetting.py")
fg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fg)


def record(task, value, stderr, n, dtype="bfloat16", rev="a" * 64):
    return {"model": "lab", "model_revision": rev, "checkpoint": {"run": "g10-dr2-full", "step": 243},
            "format": "bf16", "backend": "lab", "dtype": dtype, "task": task, "metric": "acc_norm", "filter": "none",
            "value": value, "stderr": stderr, "n_items": n, "seed": None,
            "harness": {"name": "lab", "version": "1"}, "hardware": "cuda", "run_date": "2026-11-26"}


def doc(**over):
    d = {"schema_version": "eval_mini_v1", "generated": "2026-11-26", "sources": [], "notes": [],
         "records": [record("arc_challenge", 47.35, 1.46, 1172), record("hellaswag", 62.0, 2.43, 400)]}
    d.update(over)
    return d


def test_points_become_fractions_and_the_dtype_and_checkpoint_are_kept():
    b = w.convert(doc(), source="x.json")["eval_mini"]
    assert b["returncode"] == 0 and b["dtype"] == "bfloat16"
    arc, hella = b["records"]
    assert (arc["task"], arc["value"], arc["stderr"], arc["n_items"]) == ("arc_challenge", 0.4735, 0.0146, 1172)
    assert (hella["value"], hella["stderr"]) == (0.62, 0.0243)
    assert b["source"]["model_revision"] == "a" * 64 and b["source"]["checkpoint"]["step"] == 243
    assert set(arc) <= set(w.KEEP)


def test_a_missing_stderr_stays_missing():
    d = doc()
    d["records"][0]["stderr"] = None            # the smoke scorer writes no stderr
    assert w.convert(d)["eval_mini"]["records"][0]["stderr"] is None


def test_forgetting_py_accepts_the_block_and_checks_its_dtype():
    block = {"eval_mini": w.convert(doc())["eval_mini"], "compute_dtype": "bfloat16"}
    recs = fg._records(block, "after")
    assert recs["arc_challenge"]["value"] == 0.4735
    fg._check_benchmark_dtype(block, "after")                          # same dtype: fine
    with pytest.raises(fg.ReportError, match="same dtype"):
        fg._check_benchmark_dtype({**block, "compute_dtype": "float32 (TF32 off)"}, "after")


def test_points_given_to_forgetting_py_unconverted_are_refused_there_and_converted_here():
    raw = {"eval_mini": {"returncode": 0, "records": [dict(r, task=r["task"]) for r in doc()["records"]]}}
    with pytest.raises(fg.ReportError, match="outside"):
        fg._records(raw, "after")
    fg._records({"eval_mini": w.convert(doc())["eval_mini"]}, "after")


@pytest.mark.parametrize("mutate,match", [
    (lambda d: d.update(schema_version="eval_mini_v2"), "not an eval_mini_v1"),
    (lambda d: d["records"].pop(), "no record for hellaswag"),
    (lambda d: d["records"].append(dict(d["records"][0], format="fp32")), "appear more than once"),
    (lambda d: d["records"][0].update(dtype="float32"), "different dtypes"),
    (lambda d: d["records"][0].update(dtype="int8"), "not float32 or bfloat16"),
    (lambda d: d["records"][0].update(model_revision="b" * 64), "different checkpoints"),
    (lambda d: d["records"][0].update(value=147.0), "outside 0..100"),
    (lambda d: d["records"][1].update(stderr=-1.0), "outside 0..100"),
    (lambda d: d["records"][0].update(value="47"), "not a number"),
])
def test_ambiguous_documents_are_refused(mutate, match):
    d = copy.deepcopy(doc())
    mutate(d)
    with pytest.raises(w.WatcherRowsError, match=match):
        w.convert(d)


def test_cli_writes_the_block_or_merges_it_into_an_evaluate_report(tmp_path, capsys):
    src = tmp_path / "eval_mini_v1_lab.json"
    src.write_text(json.dumps(doc()))
    out = tmp_path / "mini.json"
    assert w.main(["--watcher-doc", str(src), "--out", str(out)]) == 0
    assert "2 tasks, dtype bfloat16" in capsys.readouterr().out
    block = json.loads(out.read_text())
    assert "sha256" in block["eval_mini"]["source"]["file"]

    rep = tmp_path / "forget.json"
    rep.write_text(json.dumps({"forgetting": {"bpb_all": 1.1, "heldout_sha256": "x"}, "compute_dtype": "bfloat16"}))
    merged = tmp_path / "after.json"
    assert w.main(["--watcher-doc", str(src), "--report", str(rep), "--out", str(merged)]) == 0
    m = json.loads(merged.read_text())
    assert m["forgetting"]["bpb_all"] == 1.1 and m["eval_mini"]["dtype"] == "bfloat16"

    rep.write_text(json.dumps({"forgetting": {}, "compute_dtype": "float32 (TF32 off)"}))   # the T4 case
    assert w.main(["--watcher-doc", str(src), "--report", str(rep), "--out", str(tmp_path / "x.json")]) == 1
    assert "same dtype" in capsys.readouterr().err
    assert not (tmp_path / "x.json").exists()


def test_cli_refuses_a_missing_or_broken_file(tmp_path, capsys):
    assert w.main(["--watcher-doc", str(tmp_path / "nope.json"), "--out", str(tmp_path / "o.json")]) == 1
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert w.main(["--watcher-doc", str(bad), "--out", str(tmp_path / "o.json")]) == 1
    assert "refused" in capsys.readouterr().err


def test_a_real_run_lab_ckpt_document_goes_through(tmp_path, monkeypatch):
    """The document Team 9's own build_records / build_doc write, with the two benchmark tasks scored."""
    torch = pytest.importorskip("torch")
    # run_lab_ckpt puts Team 5's harness first on sys.path; left there, a later `import train` finds the wrong train.py
    monkeypatch.setattr(sys, "path", list(sys.path))
    spec = importlib.util.spec_from_file_location("run_lab_ckpt", HERE.parents[1] / "09-evaluation" / "harness_lab" / "run_lab_ckpt.py")
    rl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(rl)
    if rl.version_key(torch.__version__) < rl.version_key(rl.MIN_TORCH):
        pytest.skip(f"run_lab_ckpt.py needs torch {rl.MIN_TORCH}")
    facts = {"model": "lab", "model_revision": "c" * 64, "checkpoint": {"run": "r", "step": 1, "epoch": 0},
             "source": {"path": "r/manifest.json", "schema_version": "v5-ckpt/2"}, "manifest_sha256": "d" * 64}
    scored = [{"task": t, "metric": "acc_norm", "filter": "none", "primary": True, "value": v, "stderr": se, "n_items": 100,
               "item_set": "full", "num_fewshot": 0, "decoding": "loglikelihood"}
              for t, v, se in (("arc_challenge", 25.0, 1.0), ("hellaswag", 30.0, 2.0))]
    records = rl.build_records(facts, scored, "bf16", "bfloat16", "lab", None, "cpu", "lab", "2026-11-26")
    convert = rl.load_converter()
    d, _ = rl.build_doc(facts, records, "2026-11-26", convert)
    b = w.convert(d)["eval_mini"]
    assert b["dtype"] == "bfloat16" and [r["value"] for r in b["records"]] == [0.25, 0.3]
    fg._records({"eval_mini": b}, "after")


def _team9_rows(arc_primary="acc_norm"):
    """The two benchmark tasks as Team 9's own records_for writes them: an acc and an acc_norm record each."""
    spec = importlib.util.spec_from_file_location("lm_eval_lab_for_rows",
                                                  HERE.parents[1] / "09-evaluation" / "harness_lab" / "lm_eval_lab.py")
    lab = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lab)
    common = {"model_revision": "a" * 64, "dtype": "bfloat16", "format": "bf16"}
    group = {"items": "limit", "profile": "mini", "seed": {"random": 42}}
    label = {"diagnostic": "diagnostic", "freeze": "freeze"}
    rows = []
    for task, base in (("arc_challenge", 0.20), ("hellaswag", 0.30)):
        res = {"results": {task: {"acc,none": base, "acc_stderr,none": 0.01, "acc_norm,none": base + 0.05,
                                  "acc_norm_stderr,none": 0.02}},
               "n": {task: {"effective": 400, "original": 1000}}}
        entry = {"task": task, "num_fewshot": 0, "limit": 400, "role": "diagnostic", "primary": arc_primary}
        rows += lab.records_for(entry, res, group, "s" * 64, label, None, [], common)
    return rows


@pytest.mark.parametrize("primary,values", [("acc_norm", [0.25, 0.35]), ("acc", [0.2, 0.3])])
def test_team9s_two_metrics_per_task_are_read_from_the_primary_one(primary, values):
    pytest.importorskip("torch")
    rows = _team9_rows(primary)
    assert len(rows) == 4
    for ordered in (rows, rows[::-1]):
        got = {r["task"]: r for r in w.convert(doc(records=ordered))["eval_mini"]["records"]}
        assert [got[t]["value"] for t in w.DEFAULT_TASKS] == values
        assert {r["metric"] for r in got.values()} == {primary}


def test_two_primary_records_or_none_among_several_are_refused():
    pytest.importorskip("torch")
    for mark in (True, False):
        rows = [dict(r, primary=mark) for r in _team9_rows()]
        with pytest.raises(w.WatcherRowsError, match="not exactly one record is marked primary"):
            w.convert(doc(records=rows))
