"""perfkit.metrics.flops_per_token for the G5 config_v0 options: the MTP head, dense MoE dispatch, sparse KE.

MFU counts what the model needs (top_k experts, a dense KE projection, both heads); HFU what the code runs.
Each test changes one option on a small model and checks the change in both counts, from the formulas in the
perfkit/metrics.py docstring.
"""
from __future__ import annotations

from models.config import ModelConfig
from models.gpt import GPT
from models.ke1 import token_bytes_random
from perfkit.metrics import flops_per_token

V, D_MODEL, L, T, P = 512, 64, 2, 32, 32


def _flops(**kw):
    cfg = ModelConfig(vocab_size=V, block_size=T, n_layer=L, n_head=4, n_embd=D_MODEL, embedding="ke1",
                      ke_max_pos=P, ffn="moe", **kw)
    return flops_per_token(GPT(cfg, token_bytes_random(V, P)), T)


def _expert_params():
    h = 4 * D_MODEL
    return 2 * D_MODEL * h                    # c_fc + c_proj weights


def test_mtp_head_is_counted_in_mfu_and_recomputed_when_chunked():
    base, mtp = _flops(), _flops(mtp=True)
    assert mtp["params_active"] - base["params_active"] == V * D_MODEL
    assert mtp["model_flops_per_token"] - base["model_flops_per_token"] == 6 * V * D_MODEL
    chunked, chunked_mtp = _flops(ce_impl="chunked"), _flops(ce_impl="chunked", mtp=True)
    # chunked recomputes every head in backward: 2 * head FLOPs extra, now for both heads
    assert chunked_mtp["hw_flops_per_token"] - chunked["hw_flops_per_token"] == 8 * V * D_MODEL


def test_dense_dispatch_changes_hfu_not_mfu():
    moe = dict(num_experts=8, top_k=2, num_shared=0, bias_gamma=0.01)
    loop, dense = _flops(moe={**moe, "dispatch": "loop"}), _flops(moe={**moe, "dispatch": "dense"})
    idle = L * (8 - 2) * _expert_params()
    assert dense["model_flops_per_token"] == loop["model_flops_per_token"]
    assert dense["hw_flops_per_token"] - loop["hw_flops_per_token"] == 6 * idle
    rl = _flops(moe={**moe, "dispatch": "loop"}, reversible=True)
    rd = _flops(moe={**moe, "dispatch": "dense"}, reversible=True)
    assert rd["hw_flops_per_token"] - rl["hw_flops_per_token"] == 6 * idle + 2 * 2 * idle   # + invert, recompute


def test_sparse_ke_changes_hfu_not_mfu():
    dyn, sparse = _flops(ke_mode="dynamic"), _flops(ke_mode="sparse")
    D = 256 * P
    assert sparse["model_flops_per_token"] == dyn["model_flops_per_token"]
    assert dyn["hw_flops_per_token"] - sparse["hw_flops_per_token"] == 6 * (D - P) * D_MODEL


def test_hfu_is_below_mfu_only_with_sparse_ke():
    """docs/METRICS.md: sparse KE runs P * d of the D * d projection MFU counts, so HFU < MFU there (not the PaLM /
    Megatron HFU >= MFU); every other option only adds to HFU."""
    sparse = _flops(ke_mode="sparse")
    assert sparse["hw_flops_per_token"] < sparse["model_flops_per_token"]
    for kw in (dict(ke_mode="dynamic"), dict(ke_mode="cached"), dict(ce_impl="chunked"), dict(reversible=True),
               dict(moe=dict(num_experts=8, top_k=2, num_shared=0, bias_gamma=0.01, dispatch="dense"))):
        f = _flops(**kw)
        assert f["hw_flops_per_token"] >= f["model_flops_per_token"], kw
