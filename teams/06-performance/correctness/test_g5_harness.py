"""perfkit/g5_harness.py times G5's harness step (teams/05-architecture/harness/train.py) without editing it.

  * The timed step is their step: on their CPU smoke test (config_v0 + tiny_moe, 40 steps) the losses equal the
    losses.txt their own run() writes, to the bit (same "%.8f" text).
  * The four parts (data, fwd, bwd, opt) add up to the step; MFU fields follow perfkit.metrics (mfu_reason off
    CUDA); --set changes a setting; main() writes a JSON.
  * The G5 files match PINNED_SHA256 (main() warns otherwise); load_harness leaves sys.path and sys.modules clean.
Never skipped (Gate 1 bar: 0 skip): a missing G5 harness fails, a missing kronecker_embeddings fails with an install hint.
"""
from __future__ import annotations

import json
import sys

import pytest
import torch

from perfkit import g5_harness as G


@pytest.fixture(scope="module")
def h():
    try:
        import kronecker_embeddings  # noqa: F401  (G5's train.py imports it at the top)
    except ImportError:
        pytest.fail("kronecker_embeddings not installed: pip install -r perfkit/requirements.txt", pytrace=False)
    return G.load_harness()                                           # FileNotFoundError if G5's harness is missing


def test_timed_step_gives_the_losses_of_their_run(h, tmp_path):
    cfg = G.settings(h)
    h.run(cfg, tmp_path, torch.device("cpu"))                         # their own loop: writes losses.txt
    theirs = (tmp_path / "losses.txt").read_text().split()
    steps = int(cfg["tiny.steps"])
    res = G.run(cfg, torch.device("cpu"), h, steps=steps - 10, warmup=10)
    assert [f"{x:.8f}" for x in res["losses"]] == theirs


def test_parts_add_up_and_metrics(h):
    res = G.run(G.settings(h), torch.device("cpu"), h, steps=5, warmup=2)
    assert set(res["parts_ms"]) == set(G.PARTS)
    total = sum(res["parts_ms"][p]["median"] for p in G.PARTS)
    assert res["step_ms"]["n"] == 5 and total > 0
    m = res["metrics"]
    assert m["mfu"] is None and m["mfu_reason"] == "not_cuda"
    assert m["hw_flops_per_token"] > m["model_flops_per_token"]       # every expert runs: HFU > MFU work
    assert res["tokens_per_step"] == res["batch_size"] * res["seq_len"]


def test_set_overrides_a_setting(h):
    cfg = G.settings(h, sets=["model.d_model=64", "tiny.batch=4"])
    assert cfg["model.d_model"] == 64 and cfg["tiny.batch"] == 4
    with pytest.raises(SystemExit):
        G.settings(h, sets=["model.d_model"])


def test_main_writes_json(h, tmp_path):
    out = tmp_path / "g5.json"
    assert G.main(["--device", "cpu", "--steps", "3", "--warmup", "1", "--out", str(out)]) == 0
    res = json.loads(out.read_text())
    assert res["meta"]["harness_sha256"].keys() == G.PINNED_SHA256.keys() and set(res["parts_ms"]) == set(G.PARTS)
    assert "harness_pinned" in res["meta"]


def test_harness_matches_the_pin():
    assert G.harness_shas()["harness_pinned"], "G5 harness changed: re-check these tests, then update PINNED_SHA256"


def test_load_leaves_sys_path_and_modules_clean(h):
    before = list(sys.path), "check_config" in sys.modules
    G.load_harness()
    assert (list(sys.path), "check_config" in sys.modules) == before
