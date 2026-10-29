"""The whole model: models.gpt.GPT against an fp64 model written from the definitions, on real Brahmic tokens.

The parts have their own pairs (test_attention.py, test_ke.py, test_moe.py, test_reversible.py). This file checks
that they fit together the way bench runs them: the real configs at full width and depth, the real
BrahmicTokenizer-131K vocabulary and KE byte table, real text (English, Hindi, Tamil, Bengali, code), the reversible
custom backward, the tied or untied 131072-row head and the fp32 cross-entropy. T is short (64) so it runs on a CPU.
Weights: the model's own init, except the head x HEAD_SCALE. At init the softmax over 131072 tokens is nearly
uniform, and a loss-side bug (label smoothing 0.01) then moves the loss by only 8e-6; with x10 by 5.8e-5.

Reference (never calls the model's forward; reads only its weights): KE v1 codec built byte by byte; LayerNorm,
GELU and cross-entropy by formula; naive GQA attention (test_attention.naive_attention); MoE computed densely
(every expert on every token, gate 0 when not chosen); the stack by its recurrence (midpoint p_next = p_prev +
2h f(p_curr), or plain x + f(x)) with ordinary autograd.

Pairs (outputs logits and loss; every parameter gradient):
  model.lab_small        configs/lab_small.yaml, fp32. The reference routes by itself, so the router is checked too.
  model.base             configs/base.yaml (table embedding with tied head, dense MLP, plain stack), fp32.
  model.lab_small.chunked  lab_small with model.ce_impl=chunked, fp32: loss only (no logits), every gradient.
  model.lab_small.mtp    lab_small with the t+2 head (model.mtp), fp32; .mtp.chunked the same with ce_impl=chunked.
  model.g5_v0            configs/g5_v0.yaml (G5 config_v0 settings: eager attention, MTP, dense MoE, sparse KE), fp32.
  model.lab_small.bf16   lab_small under bf16 autocast, TOLERANCES[bfloat16]. In bf16 a few tokens go to other
                         experts than in fp64 (4 of 128 here), and a flipped token makes the expert gradients
                         noise (13-26%). So this reference REPLAYS the fast routing (the expert choice only, from
                         models.reversible._forward_recording); gates and experts are still computed in fp64.
The fp32 builder asserts that no routing decision is a near tie (margin > 1e-6): a near tie could flip between
fp32 and fp64 and fail the pair for no bug. If that fires after a model change, change SEED.
"""
from __future__ import annotations

import functools
import math
from pathlib import Path

import pytest
import torch

from correctness.framework import PAIRS, Case, check, pair, pairs_in, to64
from models.config import load_config, model_config
# Bare name, as pytest and test_framework.py import test files: correctness.test_attention would load the file a
# second time and register its pairs twice.
from test_attention import naive_attention
from models.gpt import GPT

cache = functools.cache

ROOT = Path(__file__).resolve().parents[1]
B, T, SEED = 2, 64, 0
HEAD_SCALE = 10       # head weights x10: the softmax is peaked (max p ~0.4) instead of uniform over 131072
MIN_MARGIN = 1e-6
TEXT = (
    "The lab model reads Indian languages and English with one tokenizer of 131072 pieces. "
    "भारत में सैकड़ों भाषाएँ बोली जाती हैं, और हिंदी उनमें सबसे अधिक बोली जाने वाली भाषा है। "
    "தமிழ் உலகின் மிகப் பழமையான மொழிகளில் ஒன்று, இது இந்தியாவிலும் இலங்கையிலும் பேசப்படுகிறது. "
    "বাংলা ভাষা দক্ষিণ এশিয়ার একটি প্রধান ভাষা। "
    "def step(model, x, y):\n    loss = model(x, y)[1]\n    loss.backward()\n    return loss.item()\n"
    "Training runs at 512 tokens per sequence; this test uses 64 so that it runs on a laptop in seconds. "
    "मशीन लर्निंग मॉडल को हर लिपि के लिए सही टोकन चाहिए, नहीं तो एक शब्द कई टुकड़ों में बँट जाता है। "
)


@cache
def _tokenizer():
    from dataio.tokenizer import BrahmicTokenizer
    return BrahmicTokenizer()


@cache
def _token_bytes(max_pos):
    return _tokenizer().token_bytes(max_pos)


def _batch(device):
    ids = torch.tensor(_tokenizer().encode(TEXT), dtype=torch.int64)
    assert ids.numel() >= B * (T + 1), f"TEXT gives {ids.numel()} tokens, need {B * (T + 1)}"
    buf = ids[:B * (T + 1)].view(B, T + 1).to(device)
    return buf[:, :-1].contiguous(), buf[:, 1:].contiguous()


def _model(config, device, overrides=()):
    mcfg = model_config(load_config(ROOT / "configs" / config, overrides))
    torch.manual_seed(SEED)
    tb = _token_bytes(mcfg.ke_max_pos) if mcfg.embedding == "ke1" else None
    model = GPT(mcfg, tb).to(device).train()
    with torch.no_grad():
        model.head.weight.mul_(HEAD_SCALE)          # tied in base: the embedding table too
        if model.mtp_head is not None:
            model.mtp_head.weight.mul_(HEAD_SCALE)
    return model, mcfg


# ------------------------------------------------------------------ fp64 reference
def ref_ke(idx, byte_table, lens, P):
    """kappa(t) = 1/sqrt(L) sum_p onehot256(b_p) (x) onehot_P(p), z-normalised (Bessel std, + 1e-6)."""
    rows = {}
    for t in idx.unique().tolist():
        L = min(int(lens[t]), P)
        v = torch.zeros(256 * P, dtype=torch.float64, device=idx.device)
        for p in range(L):
            v[int(byte_table[t, p]) * P + p] += 1.0 / math.sqrt(L)
        mean = v.sum() / v.numel()
        std = torch.sqrt(((v - mean) ** 2).sum() / (v.numel() - 1))
        rows[t] = (v - mean) / (std + 1e-6)
    return torch.stack([rows[t] for t in idx.reshape(-1).tolist()]).view(*idx.shape, 256 * P)


def ref_ln(x, ln):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    return (x - mu) / torch.sqrt(var + ln.eps) * ln.weight + ln.bias


def ref_mlp(m, x):
    h = x @ m.c_fc.weight.T + m.c_fc.bias
    return (0.5 * h * (1.0 + torch.erf(h / math.sqrt(2.0)))) @ m.c_proj.weight.T + m.c_proj.bias


def ref_moe(moe, x, forced=None, margins=None):
    """Dense MoE: softmax router scores; top-k on scores + router_bias (or the forced choice); gates = chosen
    scores renormalised to 1; out = sum gate * expert + shared experts."""
    shp = x.shape
    x = x.reshape(-1, shp[-1])
    scores = torch.softmax(x @ moe.router.weight.T + moe.router.bias, dim=-1)
    biased = (scores + moe.router_bias).detach()
    order = biased.argsort(dim=1, descending=True)
    if margins is not None:
        srt = biased.gather(1, order)
        margins.append(srt[:, moe.top_k - 1] - srt[:, moe.top_k])
    chosen = order[:, :moe.top_k] if forced is None else forced.long()
    gates = scores * torch.zeros_like(scores).scatter(1, chosen, 1.0)
    gates = gates / gates.sum(1, keepdim=True)
    out = sum(gates[:, e:e + 1] * ref_mlp(moe.experts[e], x) for e in range(moe.num_experts))
    for s in moe.shared:
        out = out + ref_mlp(s, x)
    return out.reshape(shp)


def ref_forward(rm, cfg, idx, targets, routes=None, margins=None):
    if cfg.embedding == "ke1":
        x = ref_ke(idx, rm.tok.byte_table, rm.tok.lens, cfg.ke_max_pos) @ rm.tok.proj.weight.T
    else:
        x = rm.tok.weight[idx]
    x = x + rm.pos.weight[:idx.shape[1]]

    def f(i, blk, p):
        a = naive_attention(ref_ln(p, blk.ln1), blk.attn.c_attn.weight, blk.attn.c_attn.bias,
                            blk.attn.c_proj.weight, blk.attn.c_proj.bias, cfg.n_head, cfg.n_kv_head)
        h = ref_ln(p + a, blk.ln2)
        if cfg.ffn == "moe":
            return a + ref_moe(blk.ffn, h, None if routes is None else routes[i], margins)
        return a + ref_mlp(blk.ffn, h)

    if cfg.reversible:
        a, b = x, x
        for i, blk in enumerate(rm.blocks):
            a, b = b, a + 2 * cfg.rev_h * f(i, blk, b)
        x = b
    else:
        for i, blk in enumerate(rm.blocks):
            x = x + f(i, blk, x)
    h = ref_ln(x, rm.ln_f)
    logits = h @ rm.head.weight.T
    loss = ref_ce(logits, targets)
    if cfg.mtp:                                       # position t predicts token t+2 = targets[t + 1]
        loss = loss + ref_ce(h[:, :-1] @ rm.mtp_head.weight.T, targets[:, 1:])
    return logits, loss


def ref_ce(logits, targets):
    lg = logits.reshape(-1, logits.shape[-1])
    return (torch.logsumexp(lg, -1) - lg.gather(1, targets.reshape(-1, 1)).squeeze(1)).mean()


def _params(m):
    return [p for _, p in m.named_parameters()]


# ------------------------------------------------------------------ pairs
def _fp32_pair(name, config, overrides=(), grad_tol=None):
    @pair(name, grad_tol=grad_tol)
    def build(device):
        fast, mcfg = _model(config, device, overrides)
        ref = to64(fast)
        idx, targets = _batch(device)
        if mcfg.ffn == "moe":
            margins = []
            with torch.no_grad():
                ref_forward(ref, mcfg, idx, targets, margins=margins)
            low = torch.cat(margins).min().item()
            assert low > MIN_MARGIN, f"{name}: routing near tie (margin {low:.1e}); change SEED"
        if mcfg.ce_impl == "chunked":                 # chunked returns no logits with targets: compare the loss
            def fast_fn(i, t):
                logits, loss = fast(i, t)
                assert logits is None, "ce_impl=chunked must not build the logits"
                return loss
            return Case(ref=lambda i, t: ref_forward(ref, mcfg, i, t)[1], fast=fast_fn,
                        inputs=[idx, targets], params_ref=_params(ref), params_fast=_params(fast))
        return Case(ref=lambda i, t: ref_forward(ref, mcfg, i, t), fast=lambda i, t: fast(i, t),
                    inputs=[idx, targets], params_ref=_params(ref), params_fast=_params(fast))
    return build


_fp32_pair("model.lab_small", "lab_small.yaml")
_fp32_pair("model.base", "base.yaml")
# The out-of-memory fallback (README, bench): head + cross-entropy in ce_chunks checkpointed pieces. Its loss and
# every gradient against the same fp64 reference; the head is recomputed in backward, so the head gradient too.
# grad_tol 1e-4, measured: the ln_f.weight gradient is 6.2e-5 off fp64 (ln_f.bias 3.8e-5), and the SAME 6.2e-5
# without chunking when the head runs as a 2-D matmul. The gradient reaching ln_f is (softmax - onehot) @ W, an
# fp32 sum over 131072 vocabulary rows, and the ln_f.weight gradient then cancels to a third of its terms.
# model.lab_small passes 5e-5 only because its 3-D head matmul takes a BLAS kernel that rounds differently.
# Still 100x below the 1.01 drill (1e-2).
_fp32_pair("model.lab_small.chunked", "lab_small.yaml", ["model.ce_impl=chunked"], grad_tol=1e-4)
# G5 config_v0 pieces on the lab model: the t+2 head (plain and chunked loss), and G5's eager attention with
# every expert run densely and the sparse KE path. Same reference (naive attention is what eager computes).
_fp32_pair("model.lab_small.mtp", "lab_small.yaml", ["model.mtp=true"])
_fp32_pair("model.lab_small.mtp.chunked", "lab_small.yaml", ["model.mtp=true", "model.ce_impl=chunked"],
           grad_tol=1e-4)
# g5_v0 runs KE in G5's "cached" mode (a 4 GiB [V, D] table); here the sparse mode, which test_ke.py ties to it.
# g5_v0 uses ce_impl=chunked, so grad_tol 1e-4 as model.lab_small.chunked (ln_f gradient, measured 6.6e-5).
_fp32_pair("model.g5_v0", "g5_v0.yaml", ["model.ke_mode=sparse"], grad_tol=1e-4)


@pair("model.lab_small.bf16", dtype=torch.bfloat16)
def _lab_bf16(device):
    from models.reversible import _forward_recording
    fast, mcfg = _model("lab_small.yaml", device)
    ref = to64(fast)
    idx, targets = _batch(device)

    def amp():
        return torch.autocast(device.type, dtype=torch.bfloat16)

    with torch.no_grad(), amp():                      # the bf16 expert choice, layer by layer
        emb = fast.tok(idx) + fast.pos(torch.arange(idx.shape[1], device=device))
        routes = [r[0] for r in _forward_recording(fast.blocks, emb)[2]]

    def fast_fn(i, t):
        with amp():
            return fast(i, t)

    return Case(ref=lambda i, t: ref_forward(ref, mcfg, i, t, routes=routes), fast=fast_fn,
                inputs=[idx, targets], params_ref=_params(ref), params_fast=_params(fast))


@pytest.mark.parametrize("name", pairs_in(__name__))
def test_pair(name, device):
    errors = check(PAIRS[name], device)
    assert len([k for k in errors if k.startswith("grad_param")]) > 100, f"{name}: too few gradients compared"
