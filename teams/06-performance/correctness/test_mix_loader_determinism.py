"""Determinism: same config and shards give the same bytes across runs, processes, access order, shard layout
and resume; golden hashes pin the output (a mismatch means bump ALGO and re-pin). CPU, synthetic shards."""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from dataio import mix_loader as ml
from test_mix_loader import Corpus

HERE = Path(__file__).resolve().parent
PKG = HERE.parent                      # teams/06-performance
N = 40                                 # batches per check; the small packs pass several epochs in this many


class ShaRng:
    """integers(lo, hi) from sha256 of a counter: fixed across numpy versions and platforms."""

    def __init__(self, tag="golden"):
        self.tag, self.i = tag, 0

    def integers(self, lo, hi):
        h = hashlib.sha256(f"{self.tag}:{self.i}".encode()).digest()
        self.i += 1
        return lo + int.from_bytes(h[:8], "little") % (hi - lo)


def hashes(loader, ns):
    return [ml.batch_sha256(loader.global_batch(n)) for n in ns]


def run_next(config, n=N, log=None):
    L = ml.MixLoader(config, log_path=log)
    return [ml.batch_sha256(L.next()) for _ in range(n)]


def digest(items) -> str:
    return hashlib.sha256("\n".join(items).encode()).hexdigest()


@pytest.fixture
def corpus(tmp_path):
    return Corpus(tmp_path / "c")


# ---- same process ----------------------------------------------------------------------------------------------------

def test_repeat_runs_and_fresh_loaders_give_the_same_bytes(corpus):
    cfg = corpus.config()
    first = run_next(cfg)
    assert len(set(first)) == N                                   # batches differ from each other
    assert run_next(cfg) == first
    L = ml.MixLoader(cfg)
    assert hashes(L, range(N)) == first                           # global_batch(n) == n-th next()
    assert hashes(L, range(N)) == first                           # warm caches give the same bytes


def test_access_order_does_not_matter(corpus):
    cfg = corpus.config()
    ref = run_next(cfg)
    order = list(range(N))
    random.Random(5).shuffle(order)
    order += [N - 1, 0, N - 1, 3, 3, N // 2, 1]                   # repeats and jumps across epochs
    L = ml.MixLoader(cfg)
    for n in order:
        assert ml.batch_sha256(L.global_batch(n)) == ref[n], n


def test_rank_slices_are_deterministic_in_any_order(corpus):
    cfg = corpus.config()
    ref = [ml.MixLoader(cfg).global_batch(n) for n in range(8)]
    ranks = [ml.MixLoader(cfg, dp_rank=r, dp_world=4) for r in range(4)]
    for n in [7, 2, 7, 0, 5]:
        for r in [3, 0, 2, 1]:
            assert torch.equal(ranks[r].batch(n), ref[n][r::4])


def test_composition_log_is_byte_identical(corpus, tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    assert run_next(corpus.config(), log=a) == run_next(corpus.config(), log=b)
    assert a.read_bytes() == b.read_bytes()


def test_global_rng_state_is_not_used(corpus):
    cfg = corpus.config()
    ref = run_next(cfg)
    np.random.seed(999)
    torch.manual_seed(999)
    random.seed(999)
    L = ml.MixLoader(cfg)
    got = []
    for _ in range(N):
        np.random.random(17), torch.rand(3), random.random()       # other code drawing from the global RNGs
        got.append(ml.batch_sha256(L.next()))
    assert got == ref


# ---- shard files ---------------------------------------------------------------------------------------------------

def test_shard_write_order_and_location_do_not_matter(corpus, tmp_path):
    cfg = corpus.config()
    ref, ref_manifest = run_next(cfg), ml.MixLoader(cfg).manifest
    files = sorted(p for p in corpus.root.iterdir() if p.name != corpus.csv.name)
    one, two = tmp_path / "x" / "deep" / "one", tmp_path / "y" / "two"
    one.mkdir(parents=True)
    two.mkdir(parents=True)
    groups = {}
    for p in files:
        groups.setdefault(p.name.split(".")[0], []).append(p)
    for i, sid in enumerate(sorted(groups, reverse=True)):        # written in reverse order, split over two roots
        for p in reversed(groups[sid]):
            shutil.copyfile(p, (one if i % 2 else two) / p.name)  # new mtimes, new inodes
    moved = json.loads(json.dumps(cfg))
    moved["shards"]["roots"] = [str(two), str(one.parent)]       # other root order, one root a parent dir
    L = ml.MixLoader(moved)
    assert L.manifest == ref_manifest
    assert hashes(L, range(N)) == ref


# ---- separate processes --------------------------------------------------------------------------------------------

CHILD = r"""
import json, random, sys
import numpy as np, torch
from dataio import mix_loader as ml
cfg_path, out_path, mode, k = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
np.random.seed(k); torch.manual_seed(k); random.seed(k); torch.set_num_threads(1 + k % 3)
cfg = json.load(open(cfg_path))
L = ml.MixLoader(cfg, log_path=out_path + ".log")
if mode == "resume":
    L.load_state_dict(json.load(open(out_path + ".state_in")))
hs = [ml.batch_sha256(L.next()) for _ in range(int(sys.argv[5]))]
json.dump({"hashes": hs, "state": L.state_dict(), "hash_of_str": hash("probe")}, open(out_path, "w"))
"""


def child(tmp_path, cfg, name, k, n, mode="run", state=None):
    cfg_path, out = tmp_path / f"{name}.cfg.json", tmp_path / f"{name}.json"
    cfg_path.write_text(json.dumps(cfg))
    if state is not None:
        Path(str(out) + ".state_in").write_text(json.dumps(state))
    env = dict(os.environ, PYTHONHASHSEED=str(k), PYTHONPATH=str(PKG))
    subprocess.run([sys.executable, "-c", CHILD, str(cfg_path), str(out), mode, str(k), str(n)],
                   cwd=tmp_path, env=env, check=True, timeout=300)
    return json.loads(out.read_text()), Path(str(out) + ".log").read_bytes()


def test_separate_processes_agree(corpus, tmp_path):
    cfg = corpus.config()
    ref = run_next(cfg)
    a, log_a = child(tmp_path, cfg, "a", 1, N)
    b, log_b = child(tmp_path, cfg, "b", 2, N)
    assert a["hash_of_str"] != b["hash_of_str"]                   # the hash seed really differed
    assert a["hashes"] == b["hashes"] == ref
    assert log_a == log_b


def test_resume_in_another_process(corpus, tmp_path):
    cfg = corpus.config()
    ref = run_next(cfg)
    first, _ = child(tmp_path, cfg, "first", 3, N // 2)
    assert first["state"]["position"] == N // 2
    second, _ = child(tmp_path, cfg, "second", 4, N - N // 2, mode="resume", state=first["state"])
    assert first["hashes"] + second["hashes"] == ref


# ---- golden hashes -------------------------------------------------------------------------------------------------

# Pinned on 28 Nov 2026 (numpy 2.5.4, torch 2.14.1). mix and batches change only together with ALGO; log also
# changes when log fields are added (re-pinned 28 Nov: sources, weights_file, source_passes, source_epochs; then sources_file,
# source_passes_mean, per-pack cap names; then the source note without the config map).
GOLDEN = {
    "manifest": "39dab7158dbdeeea2cb18c4f0adb7bf936f6ba7c64f66654911f8116325d1f4c",
    "best_fit": {"mix": "ec360c953becb472fce3b734e5f7ec3cb47834bcc4882e0b6203a745f4a5107f",
                 "batches": "d204ec9a29c03920ed9325075a3593845d96e00dee183bb44978ff4c7150e648",
                 "log": "6b89fd0ef731d9faa670e264314db66af15941cfb65fa5082d7b789db5a91444"},
}


def golden_values(tmp_path):
    c = Corpus(tmp_path / "g", rng=ShaRng())
    got = {"manifest": ml.MixLoader(c.config()).manifest}
    for packing in ["best_fit"]:
        cfg = c.config()
        log = tmp_path / f"g-{packing}.jsonl"
        hs = run_next(cfg, log=log)
        got[packing] = {"mix": ml.MixLoader(cfg).mix, "batches": digest(hs),
                        "log": hashlib.sha256(log.read_bytes()).hexdigest()}
    return got


def test_golden_hashes(tmp_path):
    got = golden_values(tmp_path)
    assert golden_values(tmp_path / "again") == got
    if GOLDEN["manifest"] is None:
        pytest.fail("GOLDEN not pinned; values from this run: " + json.dumps(got))
    assert got == GOLDEN
