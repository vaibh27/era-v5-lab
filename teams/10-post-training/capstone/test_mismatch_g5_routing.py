"""capstone/mismatch_g5.py RouteRecorder: the expert sets it records are the ones G5's own MoE.forward chooses.

G5's loss-free balancer (`moe.bias_gamma`, 0.001 in dry run 2's arm) adds `router_bias` to the router scores for the
expert choice. The recorder re-derives the choice in a pre-hook, so it must apply the same rule. The reference is the
model's own `expert_load` counter, which counts the choice its forward actually made.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import mismatch_g5  # noqa: E402

IDS = torch.tensor([[3, 7, 9, 11, 13, 5, 6, 8], [4, 10, 12, 1, 2, 3, 0, 0]])
VALID = torch.tensor([[True] * 8, [True] * 6 + [False] * 2])


def tiny_g5():
    """The tiny Team 5 LanguageModel of dt/tests/test_mask.py (model code imported from this checkout)."""
    spec = importlib.util.spec_from_file_location(
        "g5_model_for_mismatch_test", HERE.parents[1] / "05-architecture" / "harness" / "week2" / "model.py")
    g5 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(g5)
    rows = np.zeros((64, 4), dtype=np.uint8)
    rows[:, 0] = np.arange(64)
    lengths = np.ones(64, dtype=np.int16)
    m = {"vocab_size": 64, "n_embd": 16, "n_head": 2, "n_kv_head": 1, "n_layer": 2, "rope_theta": 10000.0,
         "norm_eps": 1e-5, "mtp": False, "moe": {"num_experts": 4, "top_k": 2, "num_shared": 1, "ffn_hidden": 32,
                                                  "bias": False}}
    torch.manual_seed(0)
    return g5.LanguageModel(m, rows, lengths)


def model_counts(lm):
    """Valid tokens per (block, expert) as G5's MoE.forward routes them (public start/finish_expert_load counter,
    which counts in train mode only; routing does not depend on the mode)."""
    was = lm.training
    lm.train()
    try:
        lm.start_expert_load()
        with torch.no_grad():
            lm(IDS, VALID)
        return lm.finish_expert_load()
    finally:
        lm.train(was)
        for b in lm.blocks:
            if hasattr(b.moe, "_balance_load"):
                b.moe._balance_load = None


def recorded_counts(lm):
    """The same counts from the expert sets RouteRecorder records."""
    rec = mismatch_g5.RouteRecorder(lm, type(lm.blocks[0].moe))
    try:
        rec.start()
        with torch.no_grad():
            lm(IDS, VALID)
        choices = rec.stop()
    finally:
        rec.close()
    n_exp = len(lm.blocks[0].moe.experts)
    return [[int((c[VALID] == e).any(-1).sum()) for e in range(n_exp)] for c in choices]


def test_recorder_counts_as_many_rows_per_expert_as_the_model():
    lm = tiny_g5()
    assert recorded_counts(lm) == model_counts(lm)


def test_recorder_follows_the_loss_free_balancer_router_bias():
    lm = tiny_g5()
    if not hasattr(lm.blocks[0].moe, "router_bias"):
        pytest.skip("this checkout's G5 model.py predates the loss-free balancer (router_bias); "
                    "the test runs once main's teams/05-architecture is merged")
    unbiased = model_counts(lm)
    for b in lm.blocks:  # a moderate bias: it reorders experts, it does not just force one
        b.moe.router_bias.copy_(torch.tensor([0.05, -0.02, 0.01, -0.04]))
    assert model_counts(lm) == unbiased and recorded_counts(lm) == unbiased  # gamma 0: G5 ignores the bias
    for b in lm.blocks:
        b.moe.balance_gamma = 1e-3
    want = model_counts(lm)
    assert want != unbiased  # the bias moved the choice
    assert recorded_counts(lm) == want
