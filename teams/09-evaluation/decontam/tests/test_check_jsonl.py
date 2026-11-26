"""Tests for check_jsonl.py (G9).

    pytest teams/09-evaluation/decontam/tests/test_check_jsonl.py -q

The cut into parts is tested on its own. The other tests run the real decontam_check.py and canary_scan.py on a
small fingerprint set, so they check the whole chain: planted leaks are found with the right doc_id, the parts give the
same result as one run on the file, and no copy of the text is left behind.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

import check_jsonl as cj

HERE = Path(cj.__file__).resolve().parent
ITEM = ("the old stone bridge over the quiet river was built by farmers who carried every rock "
        "from the hills above the village")
FILLER = "numbers letters and other plain words in a clean training document number"


@pytest.fixture
def fp_dir(tmp_path):
    import fpset

    d = tmp_path / "set"
    fpset.write_benchmark(d, "bench", [("bench:test:0", {"prose": [ITEM]})], {"prose": 13, "code": 20},
                          {"name": "Bench", "tags": ["english"]})
    fpset.write_index(d)
    return d


def write_jsonl(path, records, blank_after=()):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for i, r in enumerate(records):
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if i in blank_after:
                f.write("\n")
    return path


def planted(n=40):
    """n documents; a benchmark text in d5 and in a line with no id (line 30), the lab canary in d20."""
    from lab_canary import LAB_CANARY

    recs = [{"id": f"d{i}", "text": f"{FILLER} {i}"} for i in range(n)]
    recs[5]["text"] = f"{FILLER} five. A story: {ITEM}. The end."
    recs[20]["text"] = f"{FILLER} twenty {LAB_CANARY} more words"
    recs[30] = {"text": f"no id here. {ITEM}"}
    return recs


def run_main(tmp_path, files, work="w", *extra):
    return cj.main([str(f) for f in files] + ["--work", str(tmp_path / work)] + list(extra))


def test_parts_are_whole_lines_and_give_back_the_file(tmp_path):
    lines = [b'{"id": 1, "text": "a\\r\\nb"}\n', b"\n", b'{"id": 2, "text": "' + b"x" * 50 + b'"}\n',
             b"  \r\n", b'{"id": 3, "text": "c\rd"}\n', b'{"id": 4, "text": "last, no newline"}']
    src = tmp_path / "t.jsonl"
    src.write_bytes(b"".join(lines))
    out = tmp_path / "text"
    out.mkdir()
    for part_bytes in (1, 7, 40, 10_000):
        for p in out.iterdir():
            p.unlink()
        sha, size, docs, parts = cj.split_jsonl(src, 0, part_bytes, out)
        assert size == src.stat().st_size and docs == 4
        data = [p.read_bytes() for p, _ in parts]
        assert b"".join(data) == src.read_bytes() + b"\n"  # the last line gets its newline
        assert all(d.endswith(b"\n") for d in data)
        # each part's first-line index is the number of b"\n" before it in the file
        joined = b""
        for (p, first), d in zip(parts, data):
            assert first == joined.count(b"\n")
            joined += d
        if part_bytes == 1:
            assert len(parts) == 6  # one line per part; a lone \r inside a line never cuts it


def test_parts_cut_at_chunk_edges(tmp_path, monkeypatch):
    monkeypatch.setattr(cj, "CHUNK", 5)  # lines longer than a read: the carry joins them
    src = write_jsonl(tmp_path / "t.jsonl", [{"id": i, "text": "word " * i} for i in range(30)])
    out = tmp_path / "text"
    out.mkdir()
    _, _, docs, parts = cj.split_jsonl(src, 0, 64, out)
    assert docs == 30 and len(parts) > 3
    assert b"".join(p.read_bytes() for p, _ in parts) == src.read_bytes()


def test_planted_leaks_are_caught_through_the_parts(tmp_path, fp_dir):
    src = write_jsonl(tmp_path / "train.jsonl", planted(), blank_after=(2,))
    code = run_main(tmp_path, [src], "w", "--fingerprints", str(fp_dir), "--part-bytes", "600")
    assert code == 1  # hits make the exit code 1
    r = json.loads((tmp_path / "w" / "results" / "train.jsonl.json").read_text())
    assert r["parts"] > 3 and r["docs"] == 40
    # the line with no id is line index 31 of the file: 30, plus the blank line after d2
    assert r["flagged"] == 3 and sorted(r["flagged_doc_ids"]) == ["d20", "d5", "train.jsonl:31"]
    assert r["canary_hits"] == 1 and r["canary_hit_list"][0]["doc_id"] == "d20"
    assert r["set_sha256"] == json.loads((fp_dir / "MANIFEST.json").read_text())["set_sha256"]
    assert list((tmp_path / "w" / "text").iterdir()) == []


def test_parts_give_the_same_result_as_one_run_on_the_file(tmp_path, fp_dir):
    src = write_jsonl(tmp_path / "train.jsonl", planted(), blank_after=(2, 11))
    p = subprocess.run([sys.executable, str(HERE / "decontam_check.py"), str(src), "--id-field", "id",
                        "--fingerprints", str(fp_dir)], capture_output=True, text=True, check=True)
    whole = sorted(json.loads(line)["doc_id"] for line in p.stdout.splitlines() if line.strip())
    run_main(tmp_path, [src], "w", "--fingerprints", str(fp_dir), "--part-bytes", "300")
    r = json.loads((tmp_path / "w" / "results" / "train.jsonl.json").read_text())
    assert sorted(r["flagged_doc_ids"]) == whole


def test_a_clean_file_passes(tmp_path, fp_dir, capsys):
    src = write_jsonl(tmp_path / "train.jsonl", [{"id": f"d{i}", "text": f"{FILLER} {i}"} for i in range(25)])
    assert run_main(tmp_path, [src], "w", "--fingerprints", str(fp_dir), "--part-bytes", "500") == 0
    out = capsys.readouterr().out
    assert "| 1 | `train.jsonl` |" in out and "| 25 | 0 | 0 |" in out
    assert "Total: 1 file(s), 25 docs" in out


def test_a_bad_line_fails_the_file_and_leaves_no_text(tmp_path, fp_dir, capsys):
    src = tmp_path / "train.jsonl"
    src.write_text('{"id": "a", "text": "fine"}\nnot json\n', encoding="utf-8")
    assert run_main(tmp_path, [src], "w", "--fingerprints", str(fp_dir)) == 1
    assert "train.jsonl: FAILED: decontam_check.py failed" in capsys.readouterr().err
    assert not (tmp_path / "w" / "results" / "train.jsonl.json").exists()
    assert list((tmp_path / "w" / "text").iterdir()) == []


def test_parts_left_by_a_stopped_run_are_deleted(tmp_path, fp_dir):
    (tmp_path / "w" / "text").mkdir(parents=True)
    (tmp_path / "w" / "text" / "f000-old.jsonl.part0000.jsonl").write_text("text of a stopped run\n")
    src = write_jsonl(tmp_path / "train.jsonl", [{"id": "a", "text": FILLER}])
    assert run_main(tmp_path, [src], "w", "--fingerprints", str(fp_dir)) == 0
    assert list((tmp_path / "w" / "text").iterdir()) == []


def test_usage_errors_exit_2(tmp_path):
    a = write_jsonl(tmp_path / "a.jsonl", [{"text": "x"}])
    (tmp_path / "sub").mkdir()
    b = write_jsonl(tmp_path / "sub" / "a.jsonl", [{"text": "y"}])
    txt = tmp_path / "a.txt"
    txt.write_text("x")
    for files in ([tmp_path / "missing.jsonl"], [txt], [a, b]):
        with pytest.raises(SystemExit) as e:
            run_main(tmp_path, files)
        assert e.value.code == 2


def test_a_second_run_on_the_same_work_folder_stops(tmp_path, fp_dir):
    src = write_jsonl(tmp_path / "a.jsonl", [{"text": "x"}])
    (tmp_path / "w").mkdir()
    import check_shards

    held = check_shards.lock_work(tmp_path / "w")
    try:
        with pytest.raises(SystemExit) as e:
            run_main(tmp_path, [src], "w", "--fingerprints", str(fp_dir))
        assert e.value.code == 2
    finally:
        held.close()


def test_runs_as_a_script_in_isolated_mode(tmp_path, fp_dir):
    """python -I leaves the script's folder out of sys.path: the imports and the default set must still work."""
    src = write_jsonl(tmp_path / "train.jsonl", planted())
    p = subprocess.run([sys.executable, "-I", str(HERE / "check_jsonl.py"), str(src), "--work", str(tmp_path / "w"),
                        "--fingerprints", str(fp_dir), "--part-bytes", "800"], capture_output=True, text=True)
    assert p.returncode == 1, p.stderr
    assert "| 1 | `train.jsonl` |" in p.stdout and "3 flagged, 1 canary hits" in p.stdout


def test_default_part_size_is_one_part_per_worker_up_to_the_cap():
    assert cj.auto_part_bytes(100, 10) == 10
    assert cj.auto_part_bytes(101, 10) == 11
    assert cj.auto_part_bytes(5, 10) == 1
    assert cj.auto_part_bytes(10 ** 10, 10) == cj.MAX_PART_BYTES


def test_by_default_a_small_file_uses_every_worker(tmp_path, fp_dir):
    src = write_jsonl(tmp_path / "train.jsonl", planted())
    assert run_main(tmp_path, [src], "w", "--fingerprints", str(fp_dir), "--jobs", "4") == 1
    r = json.loads((tmp_path / "w" / "results" / "train.jsonl.json").read_text())
    assert r["parts"] in (4, 5)  # whole lines: the last part can hold the rest of a cut
    assert sorted(r["flagged_doc_ids"]) == ["d20", "d5", "train.jsonl:30"]


@pytest.mark.parametrize("data", [b"", b"\n\n  \n", b"\r\n"])
def test_a_file_with_no_documents_fails(tmp_path, fp_dir, capsys, data):
    src = tmp_path / "train.jsonl"
    src.write_bytes(data)
    assert run_main(tmp_path, [src], "w", "--fingerprints", str(fp_dir)) == 1
    assert "train.jsonl: FAILED: the file has no documents" in capsys.readouterr().err
    assert list((tmp_path / "w" / "text").iterdir()) == []


def test_invalid_utf8_fails_the_file(tmp_path, fp_dir, capsys):
    src = tmp_path / "train.jsonl"
    src.write_bytes(b'{"id": "a", "text": "fine"}\n{"id": "b", "text": "bad \xff byte"}\n')
    assert run_main(tmp_path, [src], "w", "--fingerprints", str(fp_dir)) == 1
    assert "train.jsonl: FAILED" in capsys.readouterr().err
    assert list((tmp_path / "w" / "text").iterdir()) == []


def test_many_files_with_few_workers(tmp_path, fp_dir, capsys):
    files = [write_jsonl(tmp_path / f"t{i}.jsonl", [{"id": f"{i}-{j}", "text": f"{FILLER} {j}"} for j in range(3)])
             for i in range(12)]
    files[7] = write_jsonl(tmp_path / "t7.jsonl", [{"text": f"x {ITEM}"}, {"id": "ok", "text": FILLER}])
    assert run_main(tmp_path, files, "w", "--fingerprints", str(fp_dir), "--jobs", "2") == 1
    rs = {p.name: json.loads(p.read_text()) for p in (tmp_path / "w" / "results").glob("*.json")}
    assert len(rs) == 12 and sum(r["flagged"] for r in rs.values()) == 1
    assert rs["t7.jsonl.json"]["flagged_doc_ids"] == ["t7.jsonl:0"]
    assert "Total: 12 file(s), 35 docs" in capsys.readouterr().out


def test_ids_map_back_past_9999_parts():
    first = {("f000-train.jsonl", 10000): 123}
    assert cj.global_id("f000-train.jsonl.part10000.jsonl:4", first, "train.jsonl") == "train.jsonl:127"
    assert cj.global_id("/w/text/f000-train.jsonl.part10000.jsonl:4", first, "train.jsonl") == "train.jsonl:127"
    assert cj.global_id("real-id-7", first, "train.jsonl") == "real-id-7"
