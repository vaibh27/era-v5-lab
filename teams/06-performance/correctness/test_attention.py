"""Attention: models.gpt.GQAttention against a naive fp64 attention.

Reference (written from the definition, never by calling the fast code): from the module's c_attn / c_proj weights,
q, k, v (GQA: k, v have n_kv_head heads; query head h uses kv head h // (n_head / n_kv_head)) ->
softmax(q k^T / sqrt(d_head) + causal mask) v -> c_proj. One loop iteration per query head, an explicit mask and
an explicit softmax: no F.scaled_dot_product_attention and no repeat_interleave.

Pairs (fast fp32 vs reference fp64; outputs, the input gradient and all parameter gradients):
  attention.causal              n_kv_head = n_head / 2, SDPA enable_gqa path
  attention.causal.mha          n_kv_head = n_head      (plain multi-head)
  attention.causal.mqa          n_kv_head = 1, SDPA enable_gqa path
  attention.causal.lab          n_kv_head = n_head / 4 (lab_small: 8 query heads share 2 kv heads), SDPA enable_gqa
  attention.causal.fallback     n_kv_head = n_head / 2, repeat_interleave path (models.gpt._SDPA_GQA = False)
  attention.causal.mqa_fallback n_kv_head = 1, repeat_interleave path
  attention.causal.lab_fallback n_kv_head = n_head / 4, repeat_interleave path
  attention.causal.eager        n_kv_head = n_head / 2, attn_impl="eager" (G5 harness: scores, mask, softmax)
  attention.causal.eager_mha    n_kv_head = n_head,     attn_impl="eager"
  attention.causal.eager_mqa    n_kv_head = 1,          attn_impl="eager"
T = 37 is not a power of 2. Weights and biases are random and large enough that the softmax is not uniform, so
the bias terms and the mask matter.

The 1.01 drill for every pair runs in test_framework.py (test_every_registered_pair_scaled_fails).

Property tests (no pair): causality on both GQA code paths (a change in the future leaves the past unchanged),
and the two GQA code paths agree with each other.

The gradient of the k part of c_attn.bias is exactly 0: adding q . b_k to every score of a query shifts its
softmax row by a constant. That is why errors are taken over whole tensors; a k-bias-only error would be 0 / 0.
"""
from __future__ import annotations

import math

import pytest
import torch

import models.gpt as gpt
from correctness.framework import PAIRS, Case, check, pair, pairs_in, rel_err, to64
from models.config import ModelConfig
from models.gpt import GQAttention


N_HEAD, N_EMBD, B, T = 8, 64, 2, 37
WEIGHT_STD, BIAS_STD = 0.15, 0.1


def naive_attention(x, w_attn, b_attn, w_proj, b_proj, n_head, n_kv_head):
    """Reference attention from the definition. x: [B, T, C]; all tensors fp64."""
    Bsz, Tn, C = x.shape
    hd = C // n_head
    group = n_head // n_kv_head
    qkv = x @ w_attn.T + b_attn
    q = qkv[..., :C]
    k = qkv[..., C:C + n_kv_head * hd]
    v = qkv[..., C + n_kv_head * hd:]
    future = torch.triu(torch.ones(Tn, Tn, dtype=torch.bool, device=x.device), diagonal=1)
    heads = []
    for h in range(n_head):
        g = h // group                                   # the kv head this query head reads
        qh, kh, vh = q[..., h * hd:(h + 1) * hd], k[..., g * hd:(g + 1) * hd], v[..., g * hd:(g + 1) * hd]
        scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(hd)
        scores = scores.masked_fill(future, float("-inf"))
        heads.append(torch.softmax(scores, dim=-1) @ vh)
    return torch.cat(heads, dim=-1) @ w_proj.T + b_proj


def _module(n_kv_head, seed=0, attn_impl="sdpa"):
    cfg = ModelConfig(n_head=N_HEAD, n_kv_head=n_kv_head, n_embd=N_EMBD, attn_impl=attn_impl)
    mod = GQAttention(cfg, "attn")
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in mod.named_parameters():
            p.copy_(torch.randn(p.shape, generator=g) * (WEIGHT_STD if name.endswith("weight") else BIAS_STD))
    return mod


def _with_sdpa_gqa(flag, fn):
    """Run fn() with models.gpt._SDPA_GQA forced to flag (restored afterwards)."""
    old = gpt._SDPA_GQA
    gpt._SDPA_GQA = flag
    try:
        return fn()
    finally:
        gpt._SDPA_GQA = old


def _register(name, n_kv_head, sdpa_gqa, attn_impl="sdpa"):
    @pair(name, tol=1e-5, grad_tol=1e-4)
    def build(device):
        if attn_impl == "sdpa" and sdpa_gqa and n_kv_head != N_HEAD and not gpt._SDPA_GQA:
            pytest.skip("SDPA enable_gqa needs torch >= 2.5")
        fast_mod = _module(n_kv_head, attn_impl=attn_impl).to(device)
        ref_mod = to64(fast_mod)
        x = torch.randn(B, T, N_EMBD, device=device,
                        generator=torch.Generator(device=device).manual_seed(1)).requires_grad_(True)

        def fast(inp):
            return _with_sdpa_gqa(sdpa_gqa, lambda: fast_mod(inp))

        def ref(inp):
            return naive_attention(inp, ref_mod.c_attn.weight, ref_mod.c_attn.bias,
                                   ref_mod.c_proj.weight, ref_mod.c_proj.bias, N_HEAD, n_kv_head)

        params = lambda m: [m.c_attn.weight, m.c_attn.bias, m.c_proj.weight, m.c_proj.bias]
        return Case(ref=ref, fast=fast, inputs=[x], params_ref=params(ref_mod), params_fast=params(fast_mod))
    return build


_register("attention.causal", N_HEAD // 2, sdpa_gqa=True)
_register("attention.causal.mha", N_HEAD, sdpa_gqa=True)
_register("attention.causal.mqa", 1, sdpa_gqa=True)
_register("attention.causal.lab", N_HEAD // 4, sdpa_gqa=True)
_register("attention.causal.fallback", N_HEAD // 2, sdpa_gqa=False)
_register("attention.causal.mqa_fallback", 1, sdpa_gqa=False)
_register("attention.causal.lab_fallback", N_HEAD // 4, sdpa_gqa=False)
_register("attention.causal.eager", N_HEAD // 2, sdpa_gqa=True, attn_impl="eager")
_register("attention.causal.eager_mha", N_HEAD, sdpa_gqa=True, attn_impl="eager")
_register("attention.causal.eager_mqa", 1, sdpa_gqa=True, attn_impl="eager")


@pair("attention.causal.bf16", dtype=torch.bfloat16)
def _bf16_case(device):
    fast_mod = _module(N_HEAD // 2).to(device)
    ref_mod = to64(fast_mod)
    x = torch.randn(B, T, N_EMBD, device=device,
                    generator=torch.Generator(device=device).manual_seed(1)).requires_grad_(True)

    def fast(inp):
        with torch.autocast(device.type, dtype=torch.bfloat16):
            return fast_mod(inp)

    def ref(inp):
        return naive_attention(inp, ref_mod.c_attn.weight, ref_mod.c_attn.bias,
                               ref_mod.c_proj.weight, ref_mod.c_proj.bias, N_HEAD, N_HEAD // 2)

    params = lambda m: [m.c_attn.weight, m.c_attn.bias, m.c_proj.weight, m.c_proj.bias]
    return Case(ref=ref, fast=fast, inputs=[x], params_ref=params(ref_mod), params_fast=params(fast_mod))


@pytest.mark.parametrize("name", pairs_in(__name__))
def test_pair(name, device):
    errors = check(PAIRS[name], device)
    assert "out[0]" in errors and "grad_in[0]" in errors, f"{name}: output and input gradient must be compared"
    assert len([k for k in errors if k.startswith("grad_param")]) == 4, f"{name}: all 4 param gradients must be compared"


@pytest.mark.parametrize("sdpa_gqa", [True, False], ids=["sdpa_gqa", "repeat_interleave"])
@pytest.mark.parametrize("n_kv_head", [N_HEAD, N_HEAD // 2, N_HEAD // 4, 1])
def test_causal(n_kv_head, sdpa_gqa, device):
    if sdpa_gqa and n_kv_head != N_HEAD and not gpt._SDPA_GQA:
        pytest.skip("SDPA enable_gqa needs torch >= 2.5")
    mod = _module(n_kv_head).to(device)
    g = torch.Generator(device=device).manual_seed(2)
    x = torch.randn(B, T, N_EMBD, device=device, generator=g)
    cut = 20
    x2 = x.clone()
    x2[:, cut:] += torch.randn(x2[:, cut:].shape, device=device, generator=g)
    with torch.no_grad():
        y, y2 = _with_sdpa_gqa(sdpa_gqa, lambda: (mod(x), mod(x2)))
    assert torch.equal(y[:, :cut], y2[:, :cut]) or (y[:, :cut] - y2[:, :cut]).abs().max() < 1e-6, \
        "output before the change moved: attention is not causal"
    assert (y[:, cut:] - y2[:, cut:]).abs().max() > 1e-3, "the change had no effect after the cut"


@pytest.mark.parametrize("n_kv_head", [N_HEAD // 2, N_HEAD // 4, 1])
def test_gqa_paths_agree(n_kv_head, device):
    if not gpt._SDPA_GQA:
        pytest.skip("SDPA enable_gqa needs torch >= 2.5")
    mod = _module(n_kv_head).to(device)
    x = torch.randn(B, T, N_EMBD, device=device, generator=torch.Generator(device=device).manual_seed(3))
    with torch.no_grad():
        a = _with_sdpa_gqa(True, lambda: mod(x))
        b = _with_sdpa_gqa(False, lambda: mod(x))
    assert rel_err(a, b) < 1e-5
