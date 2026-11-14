"""Unit tests for run_clean.py, the runner that cleans one pack, or one part of a pack, with clean.py: it stops on a pack
clean.py has no rules for, before reading or writing anything; --pack-id names the pack of a file with another name (a
tokenizer sample part); and its summary records which input and which clean.py ran. Every text here is made up; none
comes from a pack. pyarrow and ftfy come from spec_v1/requirements.txt: without them these tests fail, they do not skip.

  python -m pytest tests/spec_v1
"""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

SPEC = Path(__file__).resolve().parents[2] / "spec_v1"
RUN = SPEC / "run_clean.py"
sys.path.insert(0, str(SPEC))
import clean as c

ZWSP = chr(0x200B)  # written as a number: a raw zero-width space in code is invisible
DOCS = [("d1", "Area of a circle.\n"), ("d2", f"Area{ZWSP} of a circle.\n"), ("d3", "Subscribe to our newsletter\n")]


def pack_file(folder, name, pack_id=None):
    """A made-up English pack, <name>.parquet, with a pack_id column only when pack_id is given."""
    rows = [{"doc_id": d, "text": t, "url": "", "lang_label": "en", **({} if pack_id is None else {"pack_id": pack_id})} for d, t in DOCS]
    path = folder / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path)
    return path


def cleaned(pack):
    """(doc_id, action, text) for each made-up doc, from clean.clean() itself."""
    return [(d, *c.clean({"doc_id": d, "text": t, "url": "", "lang_label": "en", "pack_id": pack})) for d, t in DOCS]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(pack, out, *options):
    return subprocess.run([sys.executable, "-I", str(RUN), "--pack", str(pack), "--out", str(out), *options], capture_output=True, text=True)


def test_a_known_pack_is_cleaned_and_the_summary_names_its_clean_py(tmp_path):
    expected = cleaned("en-dclm")
    assert [a for _, a, _ in expected] == ["keep", "fix", "drop"]  # the made-up docs take every path of the runner
    pack = pack_file(tmp_path, "en-dclm")
    r = run(pack, tmp_path / "out")
    assert r.returncode == 0, r.stderr
    raw = (tmp_path / "out" / "en-dclm_summary.json").read_text(encoding="utf-8")
    summary = json.loads(raw)
    assert (summary["pack"], summary["pack_id"], summary["family"], summary["rows"]) == ("en-dclm.parquet", "en-dclm", "english", 3)
    assert (summary["actions"], summary["dropped"], summary["errors"]) == ({"keep": 1, "fix": 1, "drop": 1}, ["d3"], [])
    assert summary["input_sha256"] == sha256(pack)
    assert summary["clean_py_sha256"] == sha256(SPEC / "clean.py")
    assert "circle" not in raw and "newsletter" not in raw  # IDs and counts only, never text
    profile = summary["rule_seconds"]  # the per-rule time profile: only rules that ran, slowest first, named by rule ID
    assert profile and set(profile) <= set(c._RULE)
    assert all(v["calls"] >= 1 and v["seconds"] >= 0 for v in profile.values())
    assert max(v["calls"] for v in profile.values()) <= 3  # no rule runs on more documents than the pack has
    assert [v["seconds"] for v in profile.values()] == sorted((v["seconds"] for v in profile.values()), reverse=True)
    kept = pq.read_table(tmp_path / "out" / "en-dclm_clean.parquet").to_pylist()
    assert [(row["doc_id"], row["text"], row["pack_id"]) for row in kept] == [(d, t, "en-dclm") for d, a, t in expected if a != "drop"]


@pytest.mark.parametrize("pack_id", [None, "en-fineweb-edu-v2", "en-fineweb-edu"])
def test_an_unknown_pack_stops_before_anything_is_written(tmp_path, pack_id):
    # A renamed copy of a known pack: without a pack_id column, with the new name in it, or with the old name in it.
    r = run(pack_file(tmp_path, "en-fineweb-edu-v2", pack_id), tmp_path / "out")
    assert r.returncode == 1
    assert f"'en-fineweb-edu-v2' is not one of the {len(c._FAMILY)} packs clean.py has rules for" in r.stderr
    assert all(p in r.stderr for p in c._FAMILY)  # the message lists the names that work
    assert not (tmp_path / "out").exists()


def test_the_pack_name_is_checked_before_the_file_is_read(tmp_path):
    bad = tmp_path / "en-fineweb-edu-v2.parquet"
    bad.write_bytes(b"not a parquet file")
    r = run(bad, tmp_path / "out")
    assert r.returncode == 1 and "is not one of the" in r.stderr and "Traceback" not in r.stderr


def test_a_pack_id_that_does_not_match_the_file_name_stops_the_run(tmp_path):
    r = run(pack_file(tmp_path, "en-dclm", "en-fineweb-edu"), tmp_path / "out")
    assert r.returncode == 1 and "has pack_id 'en-fineweb-edu'" in r.stderr
    assert not (tmp_path / "out").exists()


# --pack-id: a tokenizer sample part is <split>/<pack_id>/part-NNNNN.parquet, and a download may be renamed.

@pytest.mark.parametrize("name, out_name", [("train/en-dclm/part-00000", "en-dclm_part-00000"),
                                            ("part-00000 (12)", "en-dclm_part-00000 (12)"),
                                            ("en-dclm", "en-dclm")])
def test_pack_id_cleans_a_file_of_any_name(tmp_path, name, out_name):
    part = pack_file(tmp_path, name)
    r = run(part, tmp_path / "out", "--pack-id", "en-dclm")
    assert r.returncode == 0, r.stderr
    summary = json.loads((tmp_path / "out" / f"{out_name}_summary.json").read_text(encoding="utf-8"))
    assert (summary["pack"], summary["pack_id"], summary["family"]) == (part.name, "en-dclm", "english")
    assert summary["input_sha256"] == sha256(part)  # the part can be matched against the sample's SHA256SUMS
    kept = pq.read_table(tmp_path / "out" / f"{out_name}_clean.parquet").to_pylist()
    assert [(row["doc_id"], row["text"], row["pack_id"]) for row in kept] == [(d, t, "en-dclm") for d, a, t in cleaned("en-dclm") if a != "drop"]


def test_parts_of_two_packs_do_not_overwrite_each_other(tmp_path):
    for pack in ("en-dclm", "en-fineweb-edu"):
        r = run(pack_file(tmp_path, f"train/{pack}/part-00000"), tmp_path / "out", "--pack-id", pack)
        assert r.returncode == 0, r.stderr
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == [
        "en-dclm_part-00000_clean.parquet", "en-dclm_part-00000_summary.json",
        "en-fineweb-edu_part-00000_clean.parquet", "en-fineweb-edu_part-00000_summary.json"]


@pytest.mark.parametrize("pack_id", ["en-fineweb-edu-v2", "ind-hi-example"])  # a renamed pack; an Indic pack, which no rule of ours covers
def test_an_unknown_pack_id_stops_before_anything_is_written(tmp_path, pack_id):
    r = run(pack_file(tmp_path, "train/x/part-00000"), tmp_path / "out", "--pack-id", pack_id)
    assert r.returncode == 1 and f"'{pack_id}' is not one of the" in r.stderr and "--pack-id" in r.stderr
    assert not (tmp_path / "out").exists()


def test_pack_id_must_agree_with_a_pack_id_column(tmp_path):
    r = run(pack_file(tmp_path, "train/en-dclm/part-00000", "en-fineweb-edu"), tmp_path / "out", "--pack-id", "en-dclm")
    assert r.returncode == 1 and "has pack_id 'en-fineweb-edu', not 'en-dclm'" in r.stderr
    assert not (tmp_path / "out").exists()
