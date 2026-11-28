"""capstone/forget_bpb_kept.py on Team 5's real LanguageModel (the t2 arm shrunk to width 64, CPU, byte-level stand-in
tokenizer), with kept folders written by keep_weights through Team 7's Checkpointer.

    python -m pytest capstone/test_forget_bpb_kept.py -q

The float32 number must equal sft_g5.forgetting_bpb (the SFT job's own), and the reports must go through
watcher_rows_to_report.py into forgetting.py without any hand edit. Not covered: a real adapter config and a CUDA run.
"""
from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest
from pytest import fixture  # bare decorator: pr-guard reads a dotted decorator as an e-mail address

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("safetensors")
HERE = Path(__file__).resolve().parent
TEAMS = HERE.parents[1]
H5 = TEAMS / "05-architecture" / "harness"
sys.path.insert(0, str(HERE))
sys.path.append(str(H5))  # at the end: Team 5's harness has a train.py that must not shadow another `train`
import forget_bpb_kept as fb  # noqa: E402
import keep_weights as kw  # noqa: E402
import sft_g5  # noqa: E402
import watcher_rows_to_report as w  # noqa: E402
from week2 import config as C  # noqa: E402
from week2.model import LanguageModel  # noqa: E402
from week2.scoring import INDIC  # noqa: E402


def _by_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fg = _by_path("forgetting_for_bpb_test", HERE.parent / "sft" / "tests" / "forgetting.py")


@fixture
def setup(tmp_path, monkeypatch):
    """(bundle holding the base model, kept folder of a changed copy, held-out file, sources file, changed model)."""
    monkeypatch.setenv("V5_EVENTS_DIR", str(tmp_path / "ev"))
    spec = copy.deepcopy(C.load_runtime(str(TEAMS / "05-architecture" / "configs" / "arms" / "t2.yaml"))["model"])
    spec.update(vocab_size=256, block_size=32, n_layer=2, n_head=2, n_kv_head=1, n_embd=64)
    spec["moe"] = dict(spec["moe"], num_experts=4, top_k=2, num_shared=1, ffn_hidden=32)
    rows = torch.randint(0, 255, (256, 32), generator=torch.Generator().manual_seed(1)).to(torch.uint8)
    lengths = torch.randint(1, 5, (256,), generator=torch.Generator().manual_seed(2))
    torch.manual_seed(0)
    base = LanguageModel(spec, rows, lengths)
    changed = copy.deepcopy(base)
    gen = torch.Generator().manual_seed(9)
    with torch.no_grad():
        for p in changed.parameters():
            p.add_(0.05 * torch.randn(p.shape, generator=gen))
    info = kw.save_run_folder(changed, tmp_path / "keep", "g10-dr2-full", step=3, probe_fn=kw.team5_probe)
    artifacts = types.SimpleNamespace(encode=lambda text: list(text.encode("utf-8")), lengths=np.ones(256, dtype=np.int16),
                                      pad_id=0, bos_id=1, tokenizer_sha="t" * 64)
    bundle = types.SimpleNamespace(model=base, spec=base.spec, artifacts=artifacts,
                                   manifest={"model_hash_g5": "b" * 64, "model_spec_sha256": "s" * 64})
    held, src = tmp_path / "heldout.jsonl", tmp_path / "sources.jsonl"
    docs = [{"id": f"d{i}", "text": f"{name} document number {i}. " * (i % 4 + 1)}
            for i, name in enumerate(("en", "code", *INDIC))]
    held.write_text("".join(json.dumps(d) + "\n" for d in docs))
    src.write_text("".join(json.dumps({"id": d["id"], "slice": name,
                                       "text_sha256": hashlib.sha256(d["text"].encode()).hexdigest()}) + "\n"
                           for d, name in zip(docs, ("en", "code", *INDIC))))
    return types.SimpleNamespace(bundle=bundle, kept=Path(info["run_dir"]), info=info, held=held, src=src, changed=changed,
                                 base=copy.deepcopy(base))


def _read(path):
    return json.loads(Path(path).read_text())


def test_float32_bpb_equals_the_sft_jobs_own_number_for_base_and_kept(setup, tmp_path):
    s = setup
    out = fb.run(s.bundle, s.held, s.src, [s.kept], "cpu", "float32", tmp_path / "out")
    base, arm = _read(out["base"]), _read(out["g10-dr2-full"])
    for report, model in ((base, s.base), (arm, s.changed)):
        job = sft_g5.forgetting_bpb(types.SimpleNamespace(model=model, spec=model.spec, artifacts=s.bundle.artifacts),
                                    s.held, s.src, "cpu")
        assert report["forgetting"]["bpb_all"] == pytest.approx(job["val/bpb"], abs=1e-12)
        assert report["forgetting"]["groups"]["val/groups/code_bpb"] == pytest.approx(job["val/groups/code_bpb"], abs=1e-12)
    assert base["forgetting"]["bpb_all"] != arm["forgetting"]["bpb_all"]
    assert base["compute_dtype"].startswith("float32") and base["model_revisions"] == ["b" * 64]
    assert arm["model_revisions"] == [s.info["manifest_sha256"]] and arm["kept"]["step"] == 3
    assert arm["forgetting"]["heldout_sha256"] == hashlib.sha256(s.held.read_bytes()).hexdigest()
    assert set(arm["forgetting"]["slices"]) == {"en", "code", *INDIC}


def test_the_kept_revision_is_the_one_team9s_reader_records(setup, monkeypatch):
    # run_lab_ckpt.py puts folders at the front of sys.path (Team 5's train.py among them): undone after this test
    monkeypatch.setattr(sys, "path", list(sys.path))
    rl = _by_path("run_lab_ckpt_for_bpb_test", TEAMS / "09-evaluation" / "harness_lab" / "run_lab_ckpt.py")
    facts = rl.checkpoint_facts(setup.kept, "keep", rl.read_pointer(setup.kept, None), None)
    assert facts["model_revision"] == setup.info["manifest_sha256"]
    assert fb.load_kept(setup.bundle, setup.kept)["manifest_sha256"] == facts["model_revision"]


def _rows(path, rev, value, dtype="bfloat16"):
    records = [{"model_revision": rev, "dtype": dtype, "task": t, "metric": "acc_norm", "filter": "none",
                "num_fewshot": 0, "value": value, "stderr": 1.5, "n_items": 400} for t in ("arc_challenge", "hellaswag")]
    path.write_text(json.dumps({"schema_version": "eval_mini_v1", "records": records}))
    return path


def test_bf16_reports_and_team9_rows_reach_a_verdict_and_swapped_rows_are_refused(setup, tmp_path, capsys):
    s = setup
    out = fb.run(s.bundle, s.held, s.src, [s.kept], "cpu", "bfloat16", tmp_path / "out")
    assert _read(out["g10-dr2-full"])["compute_dtype"] == "bfloat16"
    base_rows = _rows(tmp_path / "base_rows.json", "b" * 64, 50.0)
    arm_rows = _rows(tmp_path / "arm_rows.json", s.info["manifest_sha256"], 40.0)
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    assert w.main(["--watcher-doc", str(base_rows), "--report", str(out["base"]), "--out", str(before)]) == 0
    assert w.main(["--watcher-doc", str(arm_rows), "--report", str(out["g10-dr2-full"]), "--out", str(after)]) == 0
    # exit 1 is forgetting.py's "verdict: forgetting" (the rows fall by 0.10 here), not a failure; a refusal is exit 2
    assert fg.main(["--before", str(before), "--after", str(after), "--out", str(tmp_path / "forgetting.md")]) == 1
    assert "Verdict: FORGETTING" in (tmp_path / "forgetting.md").read_text()
    capsys.readouterr()
    bad = tmp_path / "swapped.json"
    assert w.main(["--watcher-doc", str(base_rows), "--report", str(out["g10-dr2-full"]), "--out", str(bad)]) == 1
    assert "cannot be paired" in capsys.readouterr().err and not bad.exists()


def test_a_folder_saved_for_another_model_config_is_refused(setup):
    setup.bundle.spec = {**setup.bundle.spec, "n_layer": 3}
    with pytest.raises(fb.KeptError, match="differs"):
        fb.load_kept(setup.bundle, setup.kept)


def test_an_uncommitted_folder_and_two_folders_with_one_name_are_refused(setup, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(fb.KeptError, match="latest.json"):
        fb.read_pointer(empty)
    with pytest.raises(fb.KeptError, match="share a name"):
        fb.run(setup.bundle, setup.held, setup.src, [setup.kept, setup.kept], "cpu", "float32", tmp_path / "o")


def test_the_base_takes_a_store_record_only_for_its_own_weights(setup, tmp_path):
    s = setup
    record = tmp_path / "G10_FROM_STORE.json"
    record.write_text(json.dumps({"model_hash_g5": "b" * 64, "manifest_sha256": "c" * 64}))
    out = fb.run(s.bundle, s.held, s.src, [], "cpu", "float32", tmp_path / "out", base_store_record=record)
    assert _read(out["base"])["model_revisions"] == ["b" * 64, "c" * 64]
    rows = _rows(tmp_path / "rows.json", "c" * 64, 50.0, "float32")
    assert w.main(["--watcher-doc", str(rows), "--report", str(out["base"]), "--out", str(tmp_path / "base.json")]) == 0
    # a run folder rebuilt from the store carries the record itself
    adapter = tmp_path / "adapter.yaml"
    adapter.write_text(json.dumps({"run_folder": {"path": str(tmp_path)}}))
    assert fb.base_revisions(s.bundle, adapter) == ["b" * 64, "c" * 64]
    record.write_text(json.dumps({"model_hash_g5": "d" * 64, "manifest_sha256": "c" * 64}))
    with pytest.raises(fb.KeptError, match="is not the base's"):
        fb.base_revisions(s.bundle, None, record)
    with pytest.raises(fb.KeptError, match="not found"):
        fb.base_revisions(s.bundle, None, tmp_path / "missing.json")


def test_a_folder_with_other_frozen_tables_is_refused(setup, tmp_path):
    other = copy.deepcopy(setup.changed)
    other.state_dict()["embedding.rows"].add_(1)
    info = kw.save_run_folder(other, tmp_path / "keep", "other-table", step=3, probe_fn=kw.team5_probe)
    with pytest.raises(fb.KeptError, match="frozen tables differ.*embedding.rows"):
        fb.load_kept(setup.bundle, Path(info["run_dir"]))


def test_one_bad_folder_stops_the_run_before_anything_is_measured(setup, tmp_path):
    s = setup
    with pytest.raises(fb.KeptError, match="missing"):
        fb.run(s.bundle, s.held, s.src, [s.kept, tmp_path / "typo"], "cpu", "float32", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_damaged_checkpoint_file_is_a_refusal_not_a_traceback(setup):
    step = json.loads((setup.kept / "latest.json").read_text())["path"]
    files = json.loads((setup.kept / step / "MANIFEST.json").read_text())["files"]
    victim = setup.kept / step / max(files, key=lambda n: files[n].get("bytes", 0))
    victim.write_bytes(victim.read_bytes() + b"x")
    with pytest.raises(fb.KeptError, match="IntegrityError"):
        fb.load_kept(setup.bundle, setup.kept)


def test_the_report_says_whether_the_probe_bits_were_compared(setup):
    assert fb.load_kept(setup.bundle, setup.kept)["probe"]["result"] == "compared"
    step = json.loads((setup.kept / "latest.json").read_text())["path"]
    path = setup.kept / step / "MANIFEST.json"
    manifest = json.loads(path.read_text())
    manifest["probe_loss"] = {"cuda:Other GPU": "0" * 16}
    path.write_text(json.dumps(manifest))
    probe = fb.load_kept(setup.bundle, setup.kept)["probe"]
    assert probe["result"] == "skipped" and probe["saved_device_classes"] == ["cuda:Other GPU"]


def test_bf16_on_a_device_without_that_autocast_is_refused(setup, tmp_path):
    s = setup
    with pytest.raises(fb.KeptError, match="not applied"):
        fb.run(s.bundle, s.held, s.src, [], "mps", "bfloat16", tmp_path / "out")
    assert not (tmp_path / "out").exists()


def _step_dir(kept):
    return kept / json.loads((kept / "latest.json").read_text())["path"]


@fixture(params=["missing file", "garbled pointer", "no step", "no commits", "missing key"])
def damaged(request, setup):
    kept, step = setup.kept, _step_dir(setup.kept)
    manifest = json.loads((step / "MANIFEST.json").read_text())
    if request.param == "missing file":
        (step / sorted(manifest["files"])[0]).unlink()
    elif request.param == "garbled pointer":
        (kept / "latest.json").write_text("{")
    elif request.param == "no step":
        (step / "MANIFEST.json").write_text("{}")
    elif request.param == "no commits":
        (kept / "commits.jsonl").unlink()
    else:  # a folder saved without one of the model's tensors
        manifest["keys"].pop(next(k for k in manifest["keys"] if k.startswith("model.") and "router_bias" in k))
        (step / "MANIFEST.json").write_text(json.dumps(manifest))
    return setup


def test_every_kind_of_damaged_folder_is_a_refusal(damaged):
    with pytest.raises(fb.KeptError):
        fb.load_kept(damaged.bundle, damaged.kept)


def test_a_later_bad_folder_stops_before_measuring_and_a_rerun_leaves_no_old_report(setup, tmp_path):
    s, out = setup, tmp_path / "out"
    other = copy.deepcopy(s.changed)
    other.state_dict()["embedding.rows"].add_(1)
    bad = Path(kw.save_run_folder(other, tmp_path / "keep", "other-table", step=3, probe_fn=kw.team5_probe)["run_dir"])
    with pytest.raises(fb.KeptError, match="frozen tables differ"):
        fb.run(s.bundle, s.held, s.src, [s.kept, bad], "cpu", "float32", out)
    assert not out.exists()
    with pytest.raises(fb.KeptError, match="held-out files refused"):
        fb.run(s.bundle, s.held, tmp_path / "no_sources.jsonl", [s.kept], "cpu", "float32", out)
    assert not out.exists()
    out.mkdir()
    (out / "forget_g10-dr2-full.json").write_text("old")
    real = fb.heldout_bpb
    calls = []

    def fails_on_the_arm(*a, **k):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("out of memory")
        return real(*a, **k)

    fb.heldout_bpb = fails_on_the_arm
    try:
        with pytest.raises(RuntimeError, match="out of memory"):
            fb.run(s.bundle, s.held, s.src, [s.kept], "cpu", "float32", out)
    finally:
        fb.heldout_bpb = real
    assert sorted(p.name for p in out.iterdir()) == ["forget_base.json"]
