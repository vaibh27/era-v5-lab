"""Chunked and vocabulary-tiled cross-entropy against the plain path and a CE definition.

GPT(ce_impl="chunked" or "fused") must match ce_impl="torch" for masked targets and MTP. Chunked loss tiles
positions; fused loss tiles vocabulary and recomputes in backward. Fused checks always cover CPU and cover CUDA
when available. No tokenizer is needed: the model is tiny and uses a plain embedding table.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from models.config import ModelConfig
from models.gpt import GPT, IGNORE_INDEX
from models.linear_ce import linear_cross_entropy, linear_cross_entropy_cached
from correctness.framework import Case, PAIRS, check, pair, pairs_in

VOCAB = 97
BLOCK = 16


def _models(device, chunks, ce_impl="chunked", mtp=False):
    cfg = dict(vocab_size=VOCAB, block_size=BLOCK, n_layer=1, n_head=2, n_embd=32)
    torch.manual_seed(0)
    plain = GPT(ModelConfig(**cfg, ce_impl="torch", mtp=mtp)).to(device).train()
    candidate = GPT(ModelConfig(**cfg, ce_impl=ce_impl, ce_chunks=chunks, ce_vocab_chunk=13,
                                ce_token_chunk=5, mtp=mtp)).to(device).train()
    candidate.load_state_dict(plain.state_dict())
    return plain, candidate


def _data(device, B, T, masked=()):
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, VOCAB, (B, T), generator=g).to(device)
    y = torch.randint(0, VOCAB, (B, T), generator=g).to(device)
    for b, t in masked:
        y[b, t] = IGNORE_INDEX
    return idx, y


def _mean_ce(logits, y):
    """Mean of -log p(target) over the non-ignored positions, from the definition, in fp64."""
    lp = torch.log_softmax(logits.double().reshape(-1, logits.shape[-1]), dim=-1)
    y = y.reshape(-1)
    keep = y != IGNORE_INDEX
    return -lp[keep, y[keep]].mean()


def _grads(model):
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _loss_and_grads(model, idx, y):
    model.zero_grad(set_to_none=True)
    loss = model(idx, y)[1]
    loss.backward()
    return loss.detach(), _grads(model)


def _same(a, b, what):
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6, msg=lambda m: f"{what}: {m}")


def _linear_ce_pair(device, impl, dtype=torch.float32):
    torch.manual_seed(140)
    x = torch.randn(5, 7, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(23, 7, device=device, dtype=dtype, requires_grad=True)
    targets = torch.tensor([2, -100, 21, -100, 0], device=device)

    def reference(x, w, y):
        return F.cross_entropy(F.linear(x, w), y, ignore_index=-100)

    def fast(x, w, y):
        if impl == "fused":
            return linear_cross_entropy(x, w, y, vocab_chunk=8, ignore_index=-100)
        return linear_cross_entropy_cached(x, w, y, token_chunk=2, ignore_index=-100)

    return Case(ref=reference, fast=fast, inputs=[x, weight, targets])


@pair("ce.fused", tol=1e-5, grad_tol=1e-4)
def _fused_pair(device):
    """The established vocabulary-tiled implementation remains a separately checked option."""
    return _linear_ce_pair(device, "fused")


@pair("ce.fused.bf16", dtype=torch.bfloat16)
def _fused_bf16_pair(device):
    return _linear_ce_pair(device, "fused", torch.bfloat16)


@pair("ce.fused_cached", tol=1e-5, grad_tol=1e-4)
def _fused_cached_pair(device):
    return _linear_ce_pair(device, "fused_cached")


@pair("ce.fused_cached.bf16", dtype=torch.bfloat16)
def _fused_cached_bf16_pair(device):
    return _linear_ce_pair(device, "fused_cached", torch.bfloat16)


MASKS = {
    "none": (),
    "scattered": ((0, 0), (0, 5), (1, 3), (1, 15)),
    "half": tuple((b, t) for b in range(2) for t in range(8)),
    "one_row": tuple((0, t) for t in range(16)),
    "one_left": tuple((b, t) for b in range(2) for t in range(16) if (b, t) != (1, 7)),
}


@pytest.mark.parametrize("mask", MASKS)
def test_chunked_matches_plain_and_definition(device, mask):
    plain, chunked = _models(device, chunks=4)
    idx, y = _data(device, 2, 16, MASKS[mask])
    with torch.no_grad():
        reference = _mean_ce(plain(idx)[0], y)         # logits of the plain model, loss from the definition
    lp, gp = _loss_and_grads(plain, idx, y)
    lc, gc = _loss_and_grads(chunked, idx, y)
    _same(lp.double(), reference, f"plain vs definition ({mask})")
    _same(lc.double(), reference, f"chunked vs definition ({mask})")
    assert gp.keys() == gc.keys()
    for name in gp:
        _same(gc[name], gp[name], f"gradient of {name} ({mask})")


@pytest.mark.parametrize("B, T, chunks", [(3, 7, 4), (1, 5, 8), (2, 16, 3), (2, 16, 32)])
def test_chunked_when_chunks_do_not_divide_the_token_count(device, B, T, chunks):
    plain, chunked = _models(device, chunks=chunks)
    idx, y = _data(device, B, T, masked=((0, 1), (B - 1, T - 1)))
    lp, gp = _loss_and_grads(plain, idx, y)
    lc, gc = _loss_and_grads(chunked, idx, y)
    _same(lc, lp, f"loss B={B} T={T} chunks={chunks}")
    for name in gp:
        _same(gc[name], gp[name], f"gradient of {name} B={B} T={T} chunks={chunks}")


def test_chunked_loss_is_zero_when_every_target_is_ignored(device):
    _, chunked = _models(device, chunks=4)
    idx, _ = _data(device, 2, 16)
    y = torch.full((2, 16), IGNORE_INDEX, device=device)
    loss, grads = _loss_and_grads(chunked, idx, y)
    assert loss.item() == 0.0, "no target counts, so the loss is 0 (the plain path gives NaN here)"
    assert all(torch.isfinite(g).all() and not g.any() for g in grads.values()), "no target, so no gradient"


def _available_test_devices():
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    return devices


@pytest.mark.parametrize("device", _available_test_devices(), ids=lambda d: d.type)
@pytest.mark.parametrize("mtp", [False, True], ids=["ntp", "mtp"])
@pytest.mark.parametrize("mask", ["none", "scattered", "one_row", "one_left"])
def test_fused_matches_plain_loss_and_all_gradients(device, mtp, mask):
    plain, fused = _models(device, chunks=4, ce_impl="fused", mtp=mtp)
    idx, y = _data(device, 2, 16, masked=MASKS[mask])

    plain_logits, _ = plain(idx)
    fused_logits, _ = fused(idx)
    assert fused_logits.shape == plain_logits.shape
    lp, gp = _loss_and_grads(plain, idx, y)
    lf, gf = _loss_and_grads(fused, idx, y)

    _same(lf, lp, f"fused loss ({device}, mtp={mtp}, mask={mask})")
    assert gp.keys() == gf.keys()
    for name in gp:
        _same(gf[name], gp[name], f"fused gradient of {name} ({device}, mtp={mtp}, mask={mask})")


@pytest.mark.parametrize("device", _available_test_devices(), ids=lambda d: d.type)
@pytest.mark.parametrize("mask", ["none", "scattered", "one_row", "one_left"])
@pytest.mark.parametrize("mtp", [False, True], ids=["ntp", "mtp"])
def test_fused_cached_matches_plain_loss_and_all_gradients(device, mask, mtp):
    plain, cached = _models(device, chunks=4, ce_impl="fused_cached", mtp=mtp)
    idx, y = _data(device, 2, 16, masked=MASKS[mask])
    lp, gp = _loss_and_grads(plain, idx, y)
    lc, gc = _loss_and_grads(cached, idx, y)
    _same(lc, lp, f"fused_cached loss ({device}, mask={mask}, mtp={mtp})")
    assert gp.keys() == gc.keys()
    for name in gp:
        _same(gc[name], gp[name], f"fused_cached gradient of {name} ({device}, mask={mask}, mtp={mtp})")


@pytest.mark.parametrize("device", _available_test_devices(), ids=lambda d: d.type)
@pytest.mark.parametrize("n, token_chunk", [(11, 4), (5, 8), (17, 6), (7, 1)])
def test_fused_cached_token_chunks_need_not_divide_token_count(device, n, token_chunk):
    torch.manual_seed(29)
    x = torch.randn(n, 9, device=device, requires_grad=True)
    weight = torch.randn(31, 9, device=device, requires_grad=True)
    targets = torch.randint(0, 31, (n,), device=device)
    targets[1::4] = IGNORE_INDEX
    expected = F.cross_entropy(F.linear(x, weight), targets, ignore_index=IGNORE_INDEX)
    actual = linear_cross_entropy_cached(x, weight, targets, token_chunk=token_chunk,
                                         ignore_index=IGNORE_INDEX)
    expected_grads = torch.autograd.grad(expected, (x, weight))
    actual_grads = torch.autograd.grad(actual, (x, weight))
    _same(actual, expected, f"cached token chunk loss n={n}, chunk={token_chunk} ({device})")
    for name, got, want in zip(("x", "weight"), actual_grads, expected_grads):
        _same(got, want, f"cached token chunk {name} gradient n={n}, chunk={token_chunk} ({device})")


@pytest.mark.parametrize("device", _available_test_devices(), ids=lambda d: d.type)
def test_fused_cached_all_targets_ignored_has_zero_loss_and_gradients(device):
    x = torch.randn(5, 9, device=device, requires_grad=True)
    weight = torch.randn(31, 9, device=device, requires_grad=True)
    targets = torch.full((5,), IGNORE_INDEX, device=device)
    loss = linear_cross_entropy_cached(x, weight, targets, token_chunk=2, ignore_index=IGNORE_INDEX)
    grads = torch.autograd.grad(loss, (x, weight))
    assert loss.item() == 0.0
    assert all(torch.isfinite(g).all() and not g.any() for g in grads)


@pytest.mark.parametrize("device", _available_test_devices(), ids=lambda d: d.type)
def test_fused_returns_loss_without_logits_and_handles_all_ignored_targets(device):
    _, fused = _models(device, chunks=4, ce_impl="fused")
    idx, _ = _data(device, 2, 16)
    y = torch.full((2, 16), IGNORE_INDEX, device=device)
    logits, loss = fused(idx, y)
    assert logits is None
    assert loss.item() == 0.0
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and not p.grad.any()
               for p in fused.parameters() if p.requires_grad)


@pytest.mark.parametrize("device", _available_test_devices(), ids=lambda d: d.type)
def test_fused_linear_cross_entropy_supports_a_bias(device):
    torch.manual_seed(4)
    x = torch.randn(7, 11, device=device, requires_grad=True)
    weight = torch.randn(37, 11, device=device, requires_grad=True)
    bias = torch.randn(37, device=device, requires_grad=True)
    targets = torch.randint(0, 37, (7,), device=device)

    reference = F.cross_entropy(F.linear(x, weight, bias), targets)
    actual = linear_cross_entropy(x, weight, targets, bias=bias, vocab_chunk=8)
    reference_grads = torch.autograd.grad(reference, (x, weight, bias))
    actual_grads = torch.autograd.grad(actual, (x, weight, bias))

    _same(actual, reference, f"biased fused CE ({device})")
    for name, got, expected in zip(("x", "weight", "bias"), actual_grads, reference_grads):
        _same(got, expected, f"biased fused CE gradient {name} ({device})")


@pytest.mark.parametrize("device", _available_test_devices(), ids=lambda d: d.type)
@pytest.mark.parametrize("vocab_chunk", [1, 2, 4, 8, 16, 32, 64, 128])
def test_fused_linear_cross_entropy_power_of_two_vocab_chunks(device, vocab_chunk):
    """Power-of-two tiles must match CE even when the final tile is partial or oversized."""
    torch.manual_seed(9)
    x = torch.randn(5, 11, device=device, requires_grad=True)
    weight = torch.randn(37, 11, device=device, requires_grad=True)
    targets = torch.tensor([0, 7, 19, 36, -100], device=device)

    reference = F.cross_entropy(F.linear(x, weight), targets, ignore_index=-100)
    actual = linear_cross_entropy(x, weight, targets, vocab_chunk=vocab_chunk)
    reference_grads = torch.autograd.grad(reference, (x, weight))
    actual_grads = torch.autograd.grad(actual, (x, weight))

    _same(actual, reference, f"fused CE loss ({device}, vocab_chunk={vocab_chunk})")
    for name, got, expected in zip(("x", "weight"), actual_grads, reference_grads):
        _same(got, expected, f"fused CE gradient {name} ({device}, vocab_chunk={vocab_chunk})")


@pytest.mark.parametrize("name", pairs_in(__name__))
@pytest.mark.parametrize("device", _available_test_devices(), ids=lambda d: d.type)
def test_pair(name, device):
    check(PAIRS[name], device)
