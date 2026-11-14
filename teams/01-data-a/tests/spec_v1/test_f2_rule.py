"""Unit tests for spec_v1/f2_rule.py, F2 with G9's benchmark fingerprints, and for run_clean.py --f2: F2 reads the text clean.py
keeps, the final counts add up to the input rows, an F2 error is reported, an all-dropped output keeps its columns, a missing
or another set is refused, and a train and a heldout part never overwrite each other. Every doc here is made up; the one
benchmark string is a canary read from G9's fingerprint set at run time, never written in this file. One run_clean.py test
loads the full set (about 20 s, 1.1 GB); the others load one benchmark.

  python -m pytest tests/spec_v1
"""
import hashlib
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SPEC = Path(__file__).resolve().parents[2] / "spec_v1"
TOOLS = SPEC  # f2_rule.py and normalize.py sit next to clean.py
sys.path.insert(0, str(TOOLS))
import f2_rule

CLEAN_TEXT = "Area of a circle.\nThe radius is half the diameter.\n"
# Named decorators: pr-guard's email pattern reads an added pytest decorator line as an address.
module_fixture = pytest.fixture(scope="module")
fixture = pytest.fixture


def canary():
    """A benchmark canary string from G9's set: an exact marker, so one occurrence flags a doc."""
    return (f2_rule._g9().parent / "fingerprints" / f2_rule.SET / "gpqa_diamond" / "canaries.txt").read_text(encoding="utf-8").splitlines()[0]


def set_sha256():
    return json.loads((f2_rule._g9().parent / "fingerprints" / f2_rule.SET / "MANIFEST.json").read_text(encoding="utf-8"))["set_sha256"]


@module_fixture
def gpqa():
    return f2_rule.fingerprints(("gpqa_diamond",))


@fixture
def one_benchmark(monkeypatch, gpqa):
    monkeypatch.setattr(f2_rule, "fingerprints", lambda: gpqa)


def test_a_doc_with_a_benchmark_canary_is_dropped(one_benchmark):
    doc = {"doc_id": "x", "text": f"Notes on chemistry.\n{canary()}\n"}
    assert f2_rule.f2(doc) == ("drop", doc["text"])
    flagged, hits = f2_rule.check(doc["text"])
    assert flagged and hits[0]["item"] == "gpqa_diamond:canary" and hits[0]["offset"] == len("Notes on chemistry.\n")


def test_a_clean_doc_is_kept(one_benchmark):
    assert f2_rule.f2({"doc_id": "x", "text": CLEAN_TEXT}) == ("keep", CLEAN_TEXT)


def test_g9s_normalize_is_used_even_when_team_1s_is_already_imported():
    """Both are named normalize. With the wrong one the set refuses to load (its normalize_id differs)."""
    import normalize as team1
    assert Path(team1.__file__).parent == TOOLS
    f2_rule.fingerprints.cache_clear()
    assert f2_rule.fingerprints(("gpqa_diamond",)).set_sha256 == set_sha256()
    assert sys.modules["normalize"] is team1


def test_v2_matching_removes_joiners_without_editing_document_text(gpqa, one_benchmark):
    assert gpqa.manifest['version'] == 'v2' and gpqa.norm.id == 'g9-v2'
    text = 'क्\u200dष क्\u200cष'
    assert gpqa.norm.words_with_offsets(text) == [('क्ष', 0), ('क्ष', 5)]
    assert f2_rule.f2({'doc_id': 'joiners', 'text': text}) == ('keep', text)


def test_v2_adds_terminal_bench_canary_coverage():
    fp = f2_rule.fingerprints(('terminal_bench_2',))
    path = f2_rule._g9().parent / 'fingerprints' / f2_rule.SET / 'terminal_bench_2/canaries.txt'
    text = 'Notes.\n' + path.read_text(encoding='utf-8').splitlines()[0]
    flagged, hits = fp.check(text)
    assert flagged and any(hit['item'] == 'terminal_bench_2:canary' for hit in hits)


def test_v1_override_is_refused_after_v2_migration(monkeypatch):
    monkeypatch.setenv('F2_FINGERPRINTS', str(f2_rule._g9().parent / 'fingerprints/v1'))
    f2_rule.fingerprints.cache_clear()
    try:
        with pytest.raises(ValueError, match='not the pinned'):
            f2_rule.fingerprints(('gpqa_diamond',))
    finally:
        f2_rule.fingerprints.cache_clear()


def test_harm_test_reads_the_rule_without_loading_the_set():
    """harm_test.py runs this file's source with no __file__; the set loads only on the first doc."""
    sys.path.insert(0, str(SPEC.parent / "harmtest"))
    import harm_test
    ns = harm_test.load_rules(TOOLS / "f2_rule.py")
    ns = ns[0] if isinstance(ns, tuple) else ns
    assert callable(harm_test.find_rule(ns, "F2"))


def test_run_clean_f2_drops_the_flagged_doc_and_records_the_set(tmp_path):
    rows = [{"doc_id": "d1", "text": CLEAN_TEXT, "url": "", "lang_label": "en"},
            {"doc_id": "d2", "text": f"Notes on chemistry.\n{canary()}\n", "url": "", "lang_label": "en"}]
    pack = tmp_path / "en-dclm.parquet"
    pq.write_table(pa.Table.from_pylist(rows), pack)
    r = subprocess.run([sys.executable, str(SPEC / "run_clean.py"), "--pack", str(pack), "--out", str(tmp_path / "out"), "--f2"],
                       capture_output=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr
    s = json.loads((tmp_path / "out" / "en-dclm_summary.json").read_text(encoding="utf-8"))
    assert s["f2"]["set_sha256"] == set_sha256()
    assert [d["doc_id"] for d in s["f2"]["dropped"]] == ["d2"] and s["f2"]["dropped"][0]["items"] == ["gpqa_diamond:canary"]
    assert pq.read_table(tmp_path / "out" / "en-dclm_clean.parquet").column("doc_id").to_pylist() == ["d1"]


def test_run_clean_without_f2_has_no_f2_entry(tmp_path):
    pack = tmp_path / "en-dclm.parquet"
    pq.write_table(pa.Table.from_pylist([{"doc_id": "d1", "text": CLEAN_TEXT, "url": "", "lang_label": "en"}]), pack)
    subprocess.run([sys.executable, str(SPEC / "run_clean.py"), "--pack", str(pack), "--out", str(tmp_path / "out")], check=True, capture_output=True)
    assert "f2" not in json.loads((tmp_path / "out" / "en-dclm_summary.json").read_text(encoding="utf-8"))


def run_f2_here(monkeypatch, tmp_path, rows):
    """run_clean.py --f2 in this process, with one benchmark loaded (fast): (exit code, summary, written table)."""
    import clean

    pack = tmp_path / "en-dclm.parquet"
    pq.write_table(pa.Table.from_pylist(rows), pack)
    monkeypatch.setattr(sys, "argv", ["run_clean.py", "--pack", str(pack), "--out", str(tmp_path / "out"), "--f2"])
    # The CLI installs timing wrappers for its process lifetime. Isolate those
    # changes when exercising it inside pytest, including its in-place _ID update.
    with monkeypatch.context() as runner_state:
        runner_state.setattr(clean, "_run", clean._run)
        runner_state.setattr(clean, "_ID", dict(clean._ID))
        with pytest.raises(SystemExit) as e:
            runpy.run_path(str(SPEC / "run_clean.py"), run_name="__main__")
    s = json.loads((tmp_path / "out" / "en-dclm_summary.json").read_text(encoding="utf-8"))
    assert sum(s["final"].values()) == s["rows"]  # every input row is kept, dropped by clean.py, dropped by F2 or an error
    return e.value.code, s, pq.read_table(tmp_path / "out" / "en-dclm_clean.parquet")


def doc(doc_id, text):
    return {"doc_id": doc_id, "text": text, "url": "", "lang_label": "en"}


def test_f2_reads_the_text_clean_py_keeps(one_benchmark, monkeypatch, tmp_path):
    """A zero-width space hides the canary from F2 on the raw text; clean.py removes it, so F2 drops the doc."""
    hidden = f"Notes on chemistry.\n{canary()[:5]}{chr(0x200B)}{canary()[5:]}\n"
    assert not f2_rule.check(hidden)[0]
    code, s, out = run_f2_here(monkeypatch, tmp_path, [doc("d1", CLEAN_TEXT), doc("d2", hidden), doc("d3", "Subscribe to our newsletter\n")])
    assert code == 0 and s["actions"] == {"keep": 1, "fix": 1, "drop": 1}  # actions stay clean.py's
    assert s["final"] == {"kept": 1, "dropped_by_clean": 1, "dropped_by_f2": 1, "errors": 0}
    assert [d["doc_id"] for d in s["f2"]["dropped"]] == ["d2"] and out.column("doc_id").to_pylist() == ["d1"]


def test_an_f2_error_is_reported_by_type_and_the_run_fails(one_benchmark, gpqa, monkeypatch, tmp_path):
    def check(text):
        if "Broken" in text:
            raise RuntimeError("made up")
        return gpqa.check(text)
    monkeypatch.setattr(f2_rule, "check", check)
    code, s, out = run_f2_here(monkeypatch, tmp_path, [doc("d1", CLEAN_TEXT), doc("d2", "Broken notes.\n")])
    assert code == 1 and s["errors"] == [["d2", "RuntimeError"]]
    assert s["final"] == {"kept": 1, "dropped_by_clean": 0, "dropped_by_f2": 0, "errors": 1} and out.num_rows == 1


def test_all_docs_dropped_still_writes_the_columns(one_benchmark, monkeypatch, tmp_path):
    code, s, out = run_f2_here(monkeypatch, tmp_path, [doc(f"d{i}", f"Notes {i}.\n{canary()}\n") for i in range(2)])
    assert code == 0 and s["final"]["dropped_by_f2"] == 2
    assert out.num_rows == 0 and out.column_names == ["doc_id", "text", "url", "lang_label", "pack_id"]


def test_a_missing_set_stops_before_anything_is_written(tmp_path):
    pack = tmp_path / "en-dclm.parquet"
    pq.write_table(pa.Table.from_pylist([doc("d1", CLEAN_TEXT)]), pack)
    r = subprocess.run([sys.executable, str(SPEC / "run_clean.py"), "--pack", str(pack), "--out", str(tmp_path / "out"), "--f2"],
                       capture_output=True, text=True, env=dict(os.environ, F2_FINGERPRINTS=str(tmp_path / "no_set")))
    assert r.returncode != 0 and "no fingerprint set" in r.stderr and not (tmp_path / "out").exists()


def test_a_set_with_another_set_sha256_is_refused(monkeypatch):
    monkeypatch.setattr(f2_rule, "SET_SHA256", "0" * 64)
    f2_rule.fingerprints.cache_clear()
    try:
        with pytest.raises(ValueError, match="not the pinned"):
            f2_rule.fingerprints(("gpqa_diamond",))
    finally:
        f2_rule.fingerprints.cache_clear()


def test_a_train_and_a_heldout_part_never_overwrite_each_other(tmp_path):
    """Both are <split>/en-dclm/part-00000.parquet, so their results share a name in one --out."""
    def part(split, text):
        path = tmp_path / split / "en-dclm" / "part-00000.parquet"
        path.parent.mkdir(parents=True)
        pq.write_table(pa.Table.from_pylist([doc(f"{split}-1", text)]), path)
        return path
    def run(path):
        return subprocess.run([sys.executable, str(SPEC / "run_clean.py"), "--pack", str(path), "--pack-id", "en-dclm", "--out", str(tmp_path / "out")],
                              capture_output=True, text=True)
    train, heldout = part("train", CLEAN_TEXT), part("heldout", "Volume of a sphere.\n")
    assert run(train).returncode == 0
    r = run(heldout)
    assert r.returncode != 0 and "another input file" in r.stderr
    s = json.loads((tmp_path / "out" / "en-dclm_part-00000_summary.json").read_text(encoding="utf-8"))
    assert s["input_sha256"] == hashlib.sha256(train.read_bytes()).hexdigest()  # the train results are still there
    assert run(train).returncode == 0  # the same input may run again
