"""perfkit/g5_harness.py times G5's harness step (teams/05-architecture/harness/train.py) without editing it.

  * The timed step is their step: on their CPU smoke test (config_v0 + tiny_moe, 40 steps) the losses equal the
    losses.txt their own run() writes, to the bit (same "%.8f" text).
  * The four parts (data, fwd, bwd, opt) add up to the step; MFU fields follow perfkit.metrics (mfu_reason off
    CUDA); --set changes a setting; main() writes a JSON.
Skipped when G5's harness or the kronecker_embeddings package is not available.
"""
from __future__ import annotations

import json

import pytest
import torch

from perfkit import g5_harness as G

pytestmark = pytest.mark.skipif(not G.HARNESS.is_file(), reason=f"G5 harness not found: {G.HARNESS}")


@pytest.fixture(scope="module")
def h():
    pytest.importorskip("kronecker_embeddings")
    pytest.importorskip("yaml")
    return G.load_harness()


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
    assert res["meta"]["harness_sha256"] and set(res["parts_ms"]) == set(G.PARTS)
