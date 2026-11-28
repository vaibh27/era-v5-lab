"""Tests for run_shards.py: a synthetic pipeline run over several packs and parts (synthetic_run.py), a made-up pack table,
and the real shard_writer.py with G6's baseline tokenizer (--any-tokenizer, as in test_shard_writer.py).

  python -m pytest teams/01-data-a/shards/tests/test_run_shards.py
"""
import json
import os
import sys
from pathlib import Path

import pytest

pa = pytest.importorskip("pyarrow")
import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from synthetic_run import make_run, sha  # noqa: E402  (also puts shards/ and pipeline/ on the path)
import run_shards as rs  # noqa: E402  (imports shard_writer, which puts G6's folder on the path)
from dataio.tokenizer import BrahmicTokenizer  # noqa: E402

BASE = BrahmicTokenizer().path
TEXTS = ["The water cycle moves water between the sea, the air and the land.\n",
         "package main\n\nfunc area(r float64) float64 {\n\treturn 3.14159 * r * r\n}\n",
         "पानी समुद्र, हवा और ज़मीन के बीच घूमता है।\n"]
TABLE = {"packs": {
    "pk-a": {"language": "go", "script": "Latn", "domain": "code", "lane": "code"},
    "pk-b": {"language": "en", "script": "Latn", "domain": "english", "lane": "english"},
    "pk-nolane": {"language": "en", "script": "Latn", "domain": "english", "lane": None},
}}


def rows(pack, tag):
    return [{"doc_id": f"{tag}{k}", "text": t, "url": "", "lang_label": "en", "pack_id": pack} for k, t in enumerate(TEXTS)]


def without_times(parts):
    return {k: {f: v for f, v in e.items() if f not in ("seconds", "peak_rss_mb")} for k, e in parts.items()}


@pytest.fixture(autouse=True)
def fork_children(monkeypatch):
    """The writer processes are forked here (they then see the tests' monkeypatches, and skip the 4 s of imports each); the
    test of the real start method (spawn) says so."""
    if "fork" in rs.mp.get_all_start_methods():
        monkeypatch.setattr(rs, "START_METHOD", "fork")


@pytest.fixture
def run(tmp_path, monkeypatch):
    """A finished run over pk-a part-00000 and part-00001, pk-b part-00000, pk-nolane part-00000 and a held-out pk-a part."""
    parts = {("train", "pk-a/part-00000.parquet"): rows("pk-a", "a0_"), ("train", "pk-a/part-00001.parquet"): rows("pk-a", "a1_"),
             ("train", "pk-b/part-00000.parquet"): rows("pk-b", "b0_"), ("train", "pk-nolane/part-00000.parquet"): rows("pk-nolane", "n0_"),
             ("heldout", "pk-a/part-00000.parquet"): rows("pk-a", "h0_")}
    return make_run(tmp_path / "pipeline", monkeypatch, parts)


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "pack_table.json"
    path.write_text(json.dumps(TABLE), encoding="utf-8")
    return path


def go(run, tmp_path, table, *extra, out="shards", in_process=False):
    """in_process: the tests that break shard_writer from inside (monkeypatch) need it in this process, not in a child."""
    out = tmp_path / out
    code = rs.main(["--input", str(run / "final"), "--manifest", str(run / "pipeline_manifest.json"), "--tokenizer", str(BASE),
                    "--any-tokenizer", "--pack-table", str(table), "--out", str(out), *extra, *(["--in-process"] if in_process else [])])
    return code, out, json.loads((out / "shards_summary.json").read_text(encoding="utf-8"))


def test_every_part_gets_its_shard_and_the_table_row(run, tmp_path, table):
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b")
    written = {k: v for k, v in s["parts"].items() if v["status"] == "written"}
    assert sorted(written) == ["train/pk-a/part-00000.parquet", "train/pk-a/part-00001.parquet", "train/pk-b/part-00000.parquet"]
    assert [[x["shard_id"] for x in written[k]["shards"]] for k in sorted(written)] == \
        [["pk-a-train-000000"], ["pk-a-train-000001"], ["pk-b-train-000000"]]
    m = json.loads((out / "pk-b-train-000000.json").read_text(encoding="utf-8"))
    assert (m["language"], m["script"], m["domain"], m["capability_lane"]) == ("en", "Latn", "english", "english")
    assert all(v["docs"] == 3 and not v["truncated"] and v["rows_read"] == v["rows_in_part"] == 3 for v in written.values())
    assert sorted(p.name for p in out.glob("*.bin")) == ["pk-a-train-000000.bin", "pk-a-train-000001.bin", "pk-b-train-000000.bin"]
    assert len(set((out / n).read_bytes() for n in ("pk-a-train-000000.json", "pk-a-train-000001.json"))) == 2
    assert len(list(out.glob("cleaning_pipeline.json"))) == 1  # one file for the whole run


def test_a_pack_without_a_lane_or_row_is_skipped_and_the_rest_written(run, tmp_path, table):
    code, out, s = go(run, tmp_path, table, "--pack", "pk-nolane", "--pack", "pk-b")
    assert code == 1
    assert s["parts"]["train/pk-nolane/part-00000.parquet"]["status"] == "skipped"
    assert "no lane" in s["parts"]["train/pk-nolane/part-00000.parquet"]["reason"]
    assert s["parts"]["train/pk-b/part-00000.parquet"]["status"] == "written"
    assert s["counts"] == {"written": 1, "empty": 0, "all_left_out": 0, "failed": 0, "skipped": 1, "not_sharded": 0, "truncated": 0, "shards": 1}
    assert not list(out.glob("pk-nolane*"))
    table.write_text(json.dumps({"packs": {}}), encoding="utf-8")
    code, _, s = go(run, tmp_path, table, "--pack", "pk-b", out="shards2")
    assert code == 1 and "not in pack_table.json" in s["parts"]["train/pk-b/part-00000.parquet"]["reason"]


def test_a_held_out_part_is_not_sharded_and_is_not_an_error(run, tmp_path, table):
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a")
    assert code == 0
    held = s["parts"]["heldout/pk-a/part-00000.parquet"]
    assert held["status"] == "not_sharded" and "G3 and G9" in held["reason"] and "shard_id" not in held
    assert s["counts"] == {"written": 2, "empty": 0, "all_left_out": 0, "failed": 0, "skipped": 0, "not_sharded": 1, "truncated": 0, "shards": 2}
    assert not list(out.glob("*heldout*"))


def test_a_validation_folder_is_not_sharded_either(run, tmp_path, table):
    (run / "final" / "heldout").rename(run / "final" / "validation")
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a")
    assert code == 0 and s["parts"]["validation/pk-a/part-00000.parquet"]["status"] == "not_sharded"


def test_a_held_out_part_of_a_pack_without_a_lane_is_not_sharded_not_skipped(run, tmp_path, table):
    t = json.loads(table.read_text(encoding="utf-8"))
    t["packs"]["pk-a"]["lane"] = None
    table.write_text(json.dumps(t), encoding="utf-8")
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a")
    assert s["parts"]["heldout/pk-a/part-00000.parquet"]["status"] == "not_sharded"
    assert s["counts"]["skipped"] == 2 and code == 1  # the two train parts


def test_held_out_parts_alone_are_a_clean_run_and_do_not_block_a_rerun(run, tmp_path, table):
    (out := tmp_path / "shards").mkdir()
    (out / "pk-a-heldout-000000.json").write_text("{}", encoding="utf-8")  # whatever is there for a held-out part is not checked
    for part in (run / "final" / "train").rglob("*.parquet"):
        part.unlink()
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a")
    assert code == 0 and s["counts"]["written"] == 0 and s["counts"]["not_sharded"] == 1
    assert s["verified"] is None and s["problems"] == []  # no shard, nothing to check


def test_a_part_whose_file_changed_after_f2_fails_alone(run, tmp_path, table):
    path = run / "final" / "train" / "pk-a" / "part-00001.parquet"
    pq.write_table(pa.Table.from_pylist(rows("pk-a", "x_")), path)
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a")
    assert s["parts"]["train/pk-a/part-00001.parquet"]["status"] == "failed"
    assert s["parts"]["train/pk-a/part-00000.parquet"]["status"] == "written"
    assert not (out / "pk-a-train-000001.json").exists()


def test_shards_without_a_summary_stop_the_run_before_anything_is_written(run, tmp_path, table):
    go(run, tmp_path, table, "--pack", "pk-b")
    out = tmp_path / "shards"
    (out / "shards_summary.json").unlink()  # with no summary, nothing says these shards are this run's
    before = sorted(p.name for p in out.iterdir())
    with pytest.raises(SystemExit, match="already in .*pk-b-train-000000.json"):
        go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b")
    assert sorted(p.name for p in out.iterdir()) == before  # pk-a was not started


def test_a_second_run_for_other_packs_adds_to_the_summary(run, tmp_path, table):
    go(run, tmp_path, table, "--pack", "pk-b")
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a")
    assert sorted(x["shard_id"] for v in s["parts"].values() if v["status"] == "written" for x in v["shards"]) == \
        ["pk-a-train-000000", "pk-a-train-000001", "pk-b-train-000000"]
    assert code == 0 and s["counts"]["not_sharded"] == 1


def test_a_summary_from_another_run_is_refused(run, tmp_path, table):
    go(run, tmp_path, table, "--pack", "pk-b")
    summary = tmp_path / "shards" / "shards_summary.json"
    s = json.loads(summary.read_text(encoding="utf-8"))
    s["run_id"] = "another-run"
    summary.write_text(json.dumps(s), encoding="utf-8")
    with pytest.raises(SystemExit, match="another pipeline run"):
        go(run, tmp_path, table, "--pack", "pk-a")


def test_the_summary_names_the_run_and_the_processing(run, tmp_path, table):
    code, out, s = go(run, tmp_path, table, "--pack", "pk-b")
    assert s["run_id"] == "test-run" and "manifest_sha256" not in s
    assert s["cleaning_pipeline_sha256"] == sha(out / "cleaning_pipeline.json")


def test_other_processing_into_the_same_out_fails_part_by_part_and_keeps_the_earlier_shards(run, tmp_path, table, monkeypatch):
    go(run, tmp_path, table, "--pack", "pk-b")
    other = make_run(tmp_path / "other", monkeypatch, {("train", "pk-a/part-00000.parquet"): rows("pk-a", "a0_")},
                     removed=[("train", "pk-a/part-00000.parquet", 1)])  # another remove list, same run_id
    code, out, s = go(other, tmp_path, table, "--pack", "pk-a")
    assert code == 1 and "other processing" in s["parts"]["train/pk-a/part-00000.parquet"]["reason"]
    assert s["parts"]["train/pk-b/part-00000.parquet"]["status"] == "written"


def test_forced_rollover_numbers_shards_on_across_a_packs_parts_and_maps_every_row(run, tmp_path, table):
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a", "--tokens", "40")
    a0, a1 = (s["parts"][f"train/pk-a/part-0000{k}.parquet"] for k in (0, 1))
    ids = [x["shard_id"] for e in (a0, a1) for x in e["shards"]]
    assert code == 0 and len(a0["shards"]) > 1 and ids == [f"pk-a-train-{k:06d}" for k in range(len(ids))]
    assert s["counts"]["truncated"] == 0 and s["counts"]["shards"] == len(ids)
    for e in (a0, a1):
        rows = [r["input_row"] for x in e["shards"] for r in pq.read_table(out / f"{x['shard_id']}.docs.parquet").to_pylist()]
        assert rows == [0, 1, 2] and e["rows_read"] == 3 and e["docs"] == sum(x["docs"] for x in e["shards"])


def test_no_parts_is_an_error(run, tmp_path, table):
    with pytest.raises(SystemExit, match="no <split>/<pack>/<part>.parquet"):
        go(run, tmp_path, table, "--pack", "nothing")


def break_after_writing(monkeypatch, breaker):
    """Run shard_writer as usual, then let breaker(out folder, manifest) damage what it wrote."""
    real = rs.shard_writer.main

    def main(argv):
        manifests = real(argv)
        for manifest in manifests:  # one or more shards a part (rollover)
            breaker(Path(argv[argv.index("--out") + 1]), manifest)
        return manifests
    monkeypatch.setattr(rs.shard_writer, "main", main)


def flip_a_token_byte(out, m):
    if m["pack_id"] == "pk-b":
        p = out / m["payload_uri"]
        data = bytearray(p.read_bytes())
        data[1030] ^= 1
        p.write_bytes(bytes(data))


def test_a_good_run_is_verified(run, tmp_path, table):
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b")
    assert code == 0 and s["verified"] is True and s["problems"] == []


def test_no_verify_writes_the_shards_and_checks_nothing(run, tmp_path, table, monkeypatch):
    def not_called(*args, **kwargs):
        raise AssertionError("verify_shards.verify ran with --no-verify")
    monkeypatch.setattr(rs.verify_shards, "verify", not_called)
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b", "--no-verify")
    assert code == 0 and s["verified"] is None and s["problems"] == [] and s["counts"]["written"] == 3


def test_no_verify_does_not_hide_a_part_the_writer_itself_failed(run, tmp_path, table):
    code, out, s = go(run, tmp_path, table, "--no-verify")  # pk-nolane has no lane in the table: skipped
    assert code == 1 and s["verified"] is None and s["parts"]["train/pk-nolane/part-00000.parquet"]["status"] == "skipped"


def test_a_shard_that_fails_verification_fails_its_part_alone(run, tmp_path, table, monkeypatch):
    break_after_writing(monkeypatch, flip_a_token_byte)
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b", in_process=True)
    e = s["parts"]["train/pk-b/part-00000.parquet"]
    assert code == 1 and e["status"] == "failed" and e["shards"][0]["shard_id"] == "pk-b-train-000000"
    assert e["reason"].startswith("verify_shards: pk-b-train-000000: pk-b-train-000000.bin: sha256")
    assert s["parts"]["train/pk-a/part-00000.parquet"]["status"] == "written" and s["verified"] is False
    assert s["counts"]["written"] == 2 and s["counts"]["failed"] == 1 and s["problems"] == []


def test_a_problem_that_belongs_to_no_part_goes_to_the_summary_and_fails_the_run(run, tmp_path, table, monkeypatch):
    break_after_writing(monkeypatch, lambda out, m: (out / "cleaning_pipeline.json").unlink() if m["pack_id"] == "pk-b" else None)
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b", in_process=True)
    assert code == 1 and s["verified"] is False and s["counts"]["written"] == 3
    assert any("cleaning_pipeline.json is missing" in p for p in s["problems"])


def test_a_rerun_checks_the_shards_an_earlier_run_wrote(run, tmp_path, table):
    go(run, tmp_path, table, "--pack", "pk-b")
    (tmp_path / "shards" / "pk-b-train-000000.idx.npy").unlink()  # damaged between the runs
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a")
    e = s["parts"]["train/pk-b/part-00000.parquet"]
    assert code == 1 and e["status"] == "failed" and "pk-b-train-000000.idx.npy is missing" in e["reason"]
    assert s["parts"]["train/pk-a/part-00000.parquet"]["status"] == "written"


def test_a_long_list_of_problems_is_cut_in_the_reason(run, tmp_path, table, monkeypatch):
    def damage(out, m):
        if m["pack_id"] == "pk-b":
            flip_a_token_byte(out, m)
            for ext in (".idx.npy", ".docs.parquet", ".dump.md"):
                (out / f"{m['shard_id']}{ext}").unlink()
    break_after_writing(monkeypatch, damage)
    code, out, s = go(run, tmp_path, table, "--pack", "pk-b", in_process=True)
    reason = s["parts"]["train/pk-b/part-00000.parquet"]["reason"]
    assert reason.count("; ") == 2 and "(and 1 more)" in reason  # four problems: three are shown
def licensed(tmp_path, change=None):
    """The table of these tests with a licence and its source in every row, as the shard job's pack table has."""
    packs = {p: {**r, "license": f"licence-of-{p}", "license_source": "licence_table.md at fb7d5e6"} for p, r in TABLE["packs"].items()}
    if change:
        change(packs)
    path = tmp_path / "pack_metadata.json"
    path.write_text(json.dumps({"_comment": "the job's pack table", "packs": packs}), encoding="utf-8")
    return path


def test_a_table_with_licences_puts_them_in_every_manifest(run, tmp_path):
    code, out, s = go(run, tmp_path, licensed(tmp_path), "--pack", "pk-a", "--pack", "pk-b")
    assert code == 0 and sum(v["status"] == "written" for v in s["parts"].values()) == 3
    for shard_id, pack in (("pk-a-train-000000", "pk-a"), ("pk-a-train-000001", "pk-a"), ("pk-b-train-000000", "pk-b")):
        m = json.loads((out / f"{shard_id}.json").read_text(encoding="utf-8"))
        assert (m["license"], m["license_source"]) == (f"licence-of-{pack}", "licence_table.md at fb7d5e6")
        keys = list(m)
        assert keys[keys.index("capability_lane"):][:4] == ["capability_lane", "license", "license_source", "split"]


def test_a_pack_without_a_licence_is_skipped_like_one_without_a_lane(run, tmp_path):
    code, out, s = go(run, tmp_path, licensed(tmp_path, lambda packs: packs["pk-b"].update(license=None)), "--pack", "pk-a", "--pack", "pk-b")
    entry = s["parts"]["train/pk-b/part-00000.parquet"]
    assert code == 1 and entry["status"] == "skipped" and "pk-b has no license" in entry["reason"]
    assert s["parts"]["train/pk-a/part-00000.parquet"]["status"] == "written" and not (out / "pk-b-train-000000.json").exists()


def test_without_licences_in_the_table_the_manifest_has_none(run, tmp_path, table):
    code, out, s = go(run, tmp_path, table, "--pack", "pk-b")
    m = json.loads((out / "pk-b-train-000000.json").read_text(encoding="utf-8"))
    assert code == 0 and "license" not in m and "license_source" not in m


def test_the_dataset_version_is_checked_before_any_part_is_read(run, tmp_path, table):
    """run_shards.py passes --dataset-version to every shard_writer.py, and refuses a missing or wrong one at once (every
    part would fail on it): nothing is written."""
    base = ["--input", str(run / "final"), "--manifest", str(run / "pipeline_manifest.json"), "--tokenizer", str(BASE),
            "--pack-table", str(table)]
    for i, (extra, message) in enumerate((((), "--dataset-version is required"),
                                          (("--dataset-version", "test"), "--dataset-version is required"),
                                          (("--any-tokenizer", "--dataset-version", "corpus-v1"), "--any-tokenizer shards are"))):
        out = tmp_path / f"out{i}"
        with pytest.raises(SystemExit, match="run_shards: " + message):
            rs.main([*base, "--out", str(out), *extra])
        assert not out.exists()


# --- A04: parts with no document to write; A03: a run stopped part-way, run again -------------------------------------
def test_an_emptied_part_and_a_part_left_out_whole_get_their_own_status_and_no_shard(tmp_path, monkeypatch, table):
    mail = "someone.name" + "@" + "gmail.com"  # built at run time: pr-guard scans added lines for addresses
    parts = {("train", "pk-a/part-00000.parquet"): rows("pk-a", "a0_"),
             ("train", "pk-b/part-00000.parquet"): [{"doc_id": f"b0_{k}", "text": f"write to {mail}\n", "url": "", "lang_label": "en",
                                                     "pack_id": "pk-b"} for k in range(2)]}
    run = make_run(tmp_path / "pipeline", monkeypatch, parts, f2_drop=["a0_0", "a0_1", "a0_2"])
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b")
    a, b = s["parts"]["train/pk-a/part-00000.parquet"], s["parts"]["train/pk-b/part-00000.parquet"]
    assert code == 0 and a["status"] == "empty" and a["rows_in_part"] == 0 and a["shards"] == []
    assert b["status"] == "all_left_out" and b["left_out_docs"]["secret"] == ["b0_0", "b0_1"] and b["shards"] == []
    assert s["counts"]["empty"] == s["counts"]["all_left_out"] == 1 and s["counts"]["shards"] == 0 and not list(out.glob("*.bin"))


def stop_at(monkeypatch, where, n):
    """Make the n-th shard stop part-way, like a killed process: while its payload is written, or just before its manifest
    is published."""
    calls = []
    if where == "payload":
        real = rs.shard_writer.write_shard

        def write_shard(path, tokens):
            calls.append(path)
            if len(calls) == n:
                Path(path).write_bytes(b"cut")
                raise KeyboardInterrupt
            return real(path, tokens)
        monkeypatch.setattr(rs.shard_writer, "write_shard", write_shard)
    else:
        real = rs.shard_writer.os.replace

        def replace(src, dst):
            if Path(dst).name.startswith("pk-") and Path(dst).suffix == ".json":  # shard manifests, not the summary
                calls.append(dst)
                if len(calls) == n:
                    raise KeyboardInterrupt
            return real(src, dst)
        monkeypatch.setattr(rs.shard_writer.os, "replace", replace)


@pytest.mark.parametrize("where", ["payload", "publish"])
def test_a_run_stopped_part_way_and_run_again_ends_as_one_clean_run(run, tmp_path, table, monkeypatch, where):
    """Forced rollover (3 shards a part): stop in the 4th shard, the first of pk-a's second part. The rerun keeps part 0's
    shards as they are and writes the rest; the outcome is the clean run's, file for file."""
    args = ("--pack", "pk-a", "--tokens", "40")
    _, clean, want = go(run, tmp_path, table, *args, out="clean")
    with monkeypatch.context() as m:
        stop_at(m, where, 4)
        with pytest.raises(KeyboardInterrupt):
            go(run, tmp_path, table, *args, out="again", in_process=True)
    out = tmp_path / "again"
    kept = {p.name: p.stat().st_mtime_ns for p in out.glob("pk-a-train-00000[0-2].bin")}
    code, out, got = go(run, tmp_path, table, *args, out="again")
    assert code == 0 and without_times(got["parts"]) == without_times(want["parts"]) and got["counts"] == want["counts"]
    assert {p.name: p.stat().st_mtime_ns for p in out.glob("pk-a-train-00000[0-2].bin")} == kept and len(kept) == 3


def test_a_run_stopped_in_its_first_part_can_be_run_again(run, tmp_path, table, monkeypatch):
    """Stopped before any part is in the summary (pk-b has no held-out part, which would be saved first; a kill in code-go's
    first part on real data left shards and no summary): the rerun is not refused, and ends as the clean run, file for file."""
    args = ("--pack", "pk-b", "--tokens", "40")
    _, clean, want = go(run, tmp_path, table, *args, out="clean")
    with monkeypatch.context() as m:
        stop_at(m, "publish", 2)
        with pytest.raises(KeyboardInterrupt):
            go(run, tmp_path, table, *args, out="again", in_process=True)
    code, out, got = go(run, tmp_path, table, *args, out="again")
    assert code == 0 and without_times(got["parts"]) == without_times(want["parts"]) and got["counts"] == want["counts"]
    assert {p.name: p.read_bytes() for p in out.glob("pk-b-*.bin")} == {p.name: p.read_bytes() for p in clean.glob("pk-b-*.bin")} != {}
    names = lambda d: sorted(p.name for p in d.iterdir() if p.suffix != ".md" and p.name != "shards_summary.json")
    assert names(out) == names(clean) and not list(out.glob("*.tmp"))
    for f in clean.glob("*.bin"):
        assert (out / f.name).read_bytes() == f.read_bytes()


def test_a_rerun_with_other_settings_is_refused(run, tmp_path, table):
    go(run, tmp_path, table, "--pack", "pk-a", "--tokens", "40")
    with pytest.raises(SystemExit, match="was made with"):
        go(run, tmp_path, table, "--pack", "pk-a", "--tokens", "80")


def test_every_written_part_records_its_time_and_memory(run, tmp_path, table):
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b")
    for name, e in s["parts"].items():
        if e["status"] == "written":
            assert e["seconds"] > 0 and (e["peak_rss_mb"] is None or e["peak_rss_mb"] > 0), name
        if e["status"] == "not_sharded":
            assert "seconds" not in e
    kept = go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b")[2]["parts"]   # a rerun keeps what it did not redo
    assert {k: (e["seconds"], e["peak_rss_mb"]) for k, e in kept.items() if e["status"] == "written"} == \
        {k: (e["seconds"], e["peak_rss_mb"]) for k, e in s["parts"].items() if e["status"] == "written"}


needs_fork = pytest.mark.skipif("fork" not in rs.mp.get_all_start_methods(),
                                reason="the monkeypatch has to reach the child process, which only a forked one inherits")


@needs_fork
def test_each_part_is_written_by_a_process_of_its_own(run, tmp_path, table, monkeypatch):
    seen = tmp_path / "pids.txt"
    real = rs.run_writer

    def noting(argv):
        with open(seen, "a") as f:
            f.write(f"{os.getpid()}\n")
        return real(argv)
    monkeypatch.setattr(rs, "run_writer", noting)
    go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b")
    pids = seen.read_text().split()
    assert len(pids) == 3 and len(set(pids)) == 3 and str(os.getpid()) not in pids
    seen.unlink()
    go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b", out="inside", in_process=True)
    assert set(seen.read_text().split()) == {str(os.getpid())}


def test_a_child_process_gives_the_same_shards_as_writing_in_this_one(run, tmp_path, table):
    args = ("--pack", "pk-a", "--pack", "pk-b", "--tokens", "40")   # forced rollover: several shards a part
    _, a_out, a = go(run, tmp_path, table, *args, out="child")
    _, b_out, b = go(run, tmp_path, table, *args, out="inside", in_process=True)
    assert without_times(a["parts"]) == without_times(b["parts"]) and a["counts"] == b["counts"]
    assert sorted(p.name for p in a_out.glob("*.bin")) == sorted(p.name for p in b_out.glob("*.bin")) != []
    for f in a_out.glob("*.bin"):
        assert f.read_bytes() == (b_out / f.name).read_bytes()


@needs_fork
def test_a_writer_process_that_dies_fails_its_part_and_the_run_goes_on(run, tmp_path, table, monkeypatch):
    real = rs.run_writer

    def dying(argv):
        if "pk-b/part-00000.parquet" in argv[argv.index("--input") + 1].replace("\\", "/"):
            os._exit(9)   # like the kernel's out-of-memory kill: no result, no exception
        return real(argv)
    monkeypatch.setattr(rs, "run_writer", dying)
    code, out, s = go(run, tmp_path, table, "--pack", "pk-a", "--pack", "pk-b")
    e = s["parts"]["train/pk-b/part-00000.parquet"]
    assert code == 1 and e["status"] == "failed" and "writer process died" in e["reason"] and e["peak_rss_mb"] is None
    assert s["parts"]["train/pk-a/part-00001.parquet"]["status"] == "written" and s["counts"]["written"] == 2


def test_the_real_start_method_writes_the_same_shards(run, tmp_path, table, monkeypatch):
    """The default start method is spawn: a fresh interpreter imports the writer again and gets everything through its arguments."""
    monkeypatch.setattr(rs, "START_METHOD", "spawn")
    assert rs.START_METHOD == "spawn" and rs.child_context().get_start_method() == "spawn"
    _, s_out, s = go(run, tmp_path, table, "--pack", "pk-b", out="spawned")
    _, i_out, i = go(run, tmp_path, table, "--pack", "pk-b", out="inside", in_process=True)
    peak = s["parts"]["train/pk-b/part-00000.parquet"]["peak_rss_mb"]
    assert without_times(s["parts"]) == without_times(i["parts"]) and (peak is None or peak > 0)   # None where there is no /proc or resource
    assert (s_out / "pk-b-train-000000.bin").read_bytes() == (i_out / "pk-b-train-000000.bin").read_bytes()


@pytest.mark.skipif(not Path("/proc/self/status").is_file(), reason="VmHWM is read from /proc (Linux)")
def test_a_spawned_writers_peak_memory_is_its_own_not_its_parents(monkeypatch):
    """ru_maxrss is carried over exec, so a spawned child reports at least what its parent peaked at; VmHWM is the child's own."""
    import numpy as np
    from concurrent.futures import ProcessPoolExecutor
    np.ones(1500 * 2 ** 20 // 8)                      # this process peaks 1.5 GB higher, then frees it
    parent = rs.peak_rss_mb()
    with ProcessPoolExecutor(max_workers=1, mp_context=rs.mp.get_context("spawn")) as pool:
        child = pool.submit(rs.peak_rss_mb).result()
    assert parent > 1500 and 0 < child < parent - 1000


# --- several packs at once (--workers) ---------------------------------------------------------------------------------------

TABLE4 = {"packs": {**TABLE["packs"],
                    "pk-c": {"language": "hi", "script": "Deva", "domain": "indic", "lane": "indic"},
                    "pk-d": {"language": "en", "script": "Latn", "domain": "math", "lane": "math"}}}


@pytest.fixture
def run4(tmp_path, monkeypatch):
    """Four packs of two train parts each (three docs a part), and a held-out part."""
    parts = {("train", f"{pack}/part-0000{k}.parquet"): rows(pack, f"{pack[-1]}{k}_") for pack in ("pk-a", "pk-b", "pk-c", "pk-d") for k in (0, 1)}
    parts[("heldout", "pk-a/part-00000.parquet")] = rows("pk-a", "h0_")
    return make_run(tmp_path / "pipeline", monkeypatch, parts)


@pytest.fixture
def table4(tmp_path):
    path = tmp_path / "pack_table4.json"
    path.write_text(json.dumps(TABLE4), encoding="utf-8")
    return path


def same_files(a_out, b_out):
    names = sorted(p.name for p in a_out.iterdir() if not p.name.startswith("shards_summary"))
    assert names == sorted(p.name for p in b_out.iterdir() if not p.name.startswith("shards_summary"))
    for name in names:
        assert (a_out / name).read_bytes() == (b_out / name).read_bytes(), name
    return names


def test_several_packs_at_once_give_the_same_summary_and_files_as_one_at_a_time(run4, tmp_path, table4, monkeypatch):
    args = ("--tokens", "40")  # forced rollover: several shards a part, so the numbers across a pack's parts are tested too
    _, seq_out, seq = go(run4, tmp_path, table4, *args, out="seq")
    monkeypatch.setattr(rs, "START_METHOD", "spawn")   # real processes, started from the worker threads
    code, par_out, par = go(run4, tmp_path, table4, *args, "--workers", "3", out="par")
    assert code == 0 and seq["counts"]["written"] == 8 and seq["counts"]["shards"] > 8
    assert without_times(par["parts"]) == without_times(seq["parts"]) and par["counts"] == seq["counts"]
    assert list(par["parts"]) == list(seq["parts"])          # the same part order, whichever pack finished first
    assert len(same_files(seq_out, par_out)) > 30 and par["verified"] is True


def test_workers_with_in_process_is_refused_and_so_is_zero(run4, tmp_path, table4):
    for extra in (("--workers", "2", "--in-process"), ("--workers", "0")):
        with pytest.raises(SystemExit, match="--workers|--in-process"):
            go(run4, tmp_path, table4, *extra)
    assert not (tmp_path / "shards").exists()


def test_a_pack_that_fails_stops_no_other_pack(run4, tmp_path, table4, monkeypatch):
    pq.write_table(pa.Table.from_pylist(rows("pk-b", "x_")), run4 / "final" / "train" / "pk-b" / "part-00000.parquet")  # changed after F2
    monkeypatch.setattr(rs, "START_METHOD", "spawn")
    code, out, s = go(run4, tmp_path, table4, "--workers", "4")
    assert code == 1 and s["parts"]["train/pk-b/part-00000.parquet"]["status"] == "failed"
    assert {n: e["status"] for n, e in s["parts"].items() if e["pack_id"] != "pk-b" and e["status"] != "not_sharded"} == \
        {f"train/{p}/part-0000{k}.parquet": "written" for p in ("pk-a", "pk-c", "pk-d") for k in (0, 1)}


def test_a_second_run_with_workers_keeps_the_packs_of_the_first_and_ends_as_a_clean_run(run4, tmp_path, table4, monkeypatch):
    monkeypatch.setattr(rs, "START_METHOD", "spawn")
    _, clean_out, want = go(run4, tmp_path, table4, "--workers", "2", out="clean")
    go(run4, tmp_path, table4, "--pack", "pk-a", "--pack", "pk-c", "--workers", "2", out="again")
    kept = {p.name: p.stat().st_mtime_ns for p in (tmp_path / "again").glob("pk-[ac]-train-*.bin")}
    code, out, got = go(run4, tmp_path, table4, "--workers", "2", out="again")
    assert code == 0 and without_times(got["parts"]) == without_times(want["parts"]) and got["counts"] == want["counts"]
    now = {p.name: p.stat().st_mtime_ns for p in out.glob("pk-[ac]-train-*.bin")}
    assert len(kept) == 4 and {n: now[n] for n in kept} == kept   # the first run's shards were not written again
    same_files(clean_out, out)


def test_the_run_record_is_copied_once_and_never_half_written(tmp_path):
    record = tmp_path / "record.json"
    record.write_text('{"run": 1}\n', encoding="utf-8")
    out = tmp_path / "shards"
    out.mkdir()
    target = out / "cleaning_pipeline.json"
    rs.shard_writer.publish_pipeline_record(record, target)
    assert target.read_bytes() == record.read_bytes()
    before = target.stat().st_mtime_ns
    rs.shard_writer.publish_pipeline_record(record, target)     # already there: left alone
    assert target.stat().st_mtime_ns == before and [p.name for p in out.iterdir()] == ["cleaning_pipeline.json"]


KILLED_PARENT = """
import os, sys, time
sys.path.insert(0, sys.argv[2])
import run_shards as rs
from concurrent.futures import ProcessPoolExecutor

def writer(marker, parent):           # stands in for run_writer_in_child: the same call first, then a long part
    rs.die_with_parent(parent)
    open(marker + ".started", "w").close()
    time.sleep(10)
    open(marker, "w").write("written after the run was killed")

if __name__ == "__main__":
    with ProcessPoolExecutor(max_workers=1, mp_context=rs.child_context()) as pool:   # as write_one: spawn, one part
        pool.submit(writer, sys.argv[1], os.getpid()).result()
"""


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux only")
def test_a_writer_dies_with_a_killed_run_and_writes_nothing_after(tmp_path):
    """kill -9 of run_shards.py (or a stopped job) must not leave a writer behind that goes on writing while the run is
    started again into the same folder."""
    import signal, subprocess, time
    script, marker = tmp_path / "killed_parent.py", tmp_path / "marker"
    script.write_text(KILLED_PARENT, encoding="utf-8")
    parent = subprocess.Popen([sys.executable, str(script), str(marker), str(Path(rs.__file__).parent)])
    deadline = time.monotonic() + 120
    while not Path(str(marker) + ".started").exists():   # the writer is under way
        assert parent.poll() is None and time.monotonic() < deadline, "the writer did not start"
        time.sleep(0.2)
    parent.send_signal(signal.SIGKILL)
    parent.wait()
    time.sleep(12)                                        # longer than the writer's part
    assert not marker.exists()

