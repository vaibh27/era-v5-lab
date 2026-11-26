"""completion_logprobs for Team 5's policy: the output head runs on the completion positions only (change 1), in
chunks of HEAD_CHUNK tokens that are checkpointed under autograd (change 2). Both must give the values of the old path
(full forward sliced, log_softmax + gather) to float rounding, in every caller (job RLM, 24 Nov 2026)."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import policy as P  # noqa: E402
import train  # noqa: E402
from test_g5_sample_many import adapter_yaml, bundle, needs_standin  # noqa: E402,F401  Team 5 run folder (stand-in)


def _old_lp(pol, seq, plen):
    """The code before job RLM, verbatim."""
    logits = pol.logits(seq)[:, plen - 1:-1, :].float()
    return torch.log_softmax(logits, -1).gather(-1, seq[:, plen:, None]).squeeze(-1)


# --- change 1: head on completion positions only --------------------------------------------------------------------

class _HeadSpy:
    """Tokens per head call (all leading dims flattened)."""

    def __init__(self, model):
        self.calls = []
        self.handle = model.head.register_forward_pre_hook(
            lambda mod, args: self.calls.append(args[0].numel() // args[0].shape[-1]))

    def close(self):
        self.handle.remove()


def _seq(V, rows, plen, new, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(16, min(V, 30000), (rows, plen + new), generator=g)


@needs_standin
def test_team5_policy_runs_the_head_on_completion_positions_only_in_every_caller(bundle):
    """No caller may fall back to the full-sequence head for Team 5's policy: the update, old_lp (chunked), the
    identical-weights gate and kl_to_start all go through completion_logprobs. Fails if the policy loses hidden/head
    (the head then sees plen + new positions) or if a caller bypasses the split path."""
    pol = bundle.policy()
    assert pol._hidden is not None and pol._head is not None
    plen, new = 7, 5
    seq = _seq(bundle.spec["vocab_size"], 3, plen, new)
    cmask = torch.ones(3, new, dtype=torch.bool)
    spy = _HeadSpy(bundle.model)
    try:
        with torch.no_grad():
            old = train.chunked_logprobs(pol, seq, plen, 2)
        n_tok = int(cmask.sum())
        train.backward_objective_v1(pol, (seq, cmask, torch.tensor([1.0, -1.0, 0.5]), old, None, plen, None), 0.2,
                                    n_tok, 1)
        gate = train.logprob_gate(pol, bundle.model, [(seq, cmask, plen, None)], "eval", 0.2, 2)
        roll = type("R", (), {"seq": seq, "cmask": cmask})()

        class _Start:
            def swap(self):
                pass
        train.kl_to_start(pol, _Start(), [roll], [[0] * plen])
    finally:
        spy.close()
    # rows per call are 1 to 3, so the split path gives multiples of `new` (5, 10, 15) and the old full-sequence head
    # gives multiples of plen + new (12, 24, 36): disjoint
    assert spy.calls and set(spy.calls) <= {new, 2 * new, 3 * new}, spy.calls
    assert gate["ok"]


@needs_standin
def test_team5_split_path_equals_the_old_full_forward(bundle):
    pol = bundle.policy()
    plen, new = 6, 9
    seq = _seq(bundle.spec["vocab_size"], 2, plen, new, seed=4)
    with torch.no_grad():
        new_lp = P.completion_logprobs(pol, seq, plen).double()
        old_lp = _old_lp(pol, seq, plen).double()
        logits64 = pol.logits(seq)[:, plen - 1:-1, :].double()
        ref = torch.log_softmax(logits64, -1).gather(-1, seq[:, plen:, None]).squeeze(-1)
    assert (new_lp - old_lp).abs().max() < 5e-5
    assert (new_lp - ref).abs().max() <= (old_lp - ref).abs().max() + 1e-6


def test_split_head_gradients_equal_the_old_path():
    """A per-position head on a toy trunk: loss and gradients of the split path equal the old one (fp32)."""
    torch.manual_seed(5)

    class Toy(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb, self.mix, self.head = nn.Embedding(50, 8), nn.Linear(8, 8), nn.Linear(8, 50)

        def hidden(self, ids, valid):
            x = self.emb(ids)
            return torch.tanh(self.mix(x.cumsum(1)))  # causal mixing over positions

        def forward(self, ids, valid):
            return self.head(self.hidden(ids, valid))

    m = Toy()
    seq = torch.randint(0, 50, (2, 11))
    plen = 4
    grads = []
    for pol, fn in ((P.module_policy(m), _old_lp), (P.module_policy(m, split_head=True), P.completion_logprobs)):
        m.zero_grad()
        lp = fn(pol, seq, plen)
        lp.sum().backward()
        grads.append(([p.grad.clone() for p in m.parameters()], lp.detach()))
    (g0, lp0), (g1, lp1) = grads
    assert (lp0 - lp1).abs().max() < 1e-5
    assert max(float((a - b).abs().max()) for a, b in zip(g0, g1)) < 1e-6 * max(float(a.abs().max()) for a in g0) + 1e-7


def test_policy_needs_hidden_and_head_together():
    import pytest
    with pytest.raises(ValueError):
        P.Policy(nn.Linear(2, 2), lambda m, i, k: m(i), hidden=lambda m, i, k: i)


@needs_standin
def test_head_chunks_are_checkpointed_and_change_nothing(bundle, monkeypatch):
    """Chunked + checkpointed head (change 2) vs one head call: same log-probs and gradients; under autograd the
    head runs twice per chunk (forward, then the rebuild in backward), under no_grad once, and no call is larger
    than HEAD_CHUNK tokens. Fails if chunks are not checkpointed (one call per chunk under autograd)."""
    pol = bundle.policy()
    pol.head_checkpoint = True  # the memory mode (train.head_checkpoint / rl_job_v1 --head-checkpoint)
    plen, new = 5, 13
    seq = _seq(bundle.spec["vocab_size"], 2, plen, new, seed=6)
    params = [p for p in bundle.model.parameters() if p.requires_grad]
    out = {}
    for chunk in (10**9, 4):
        monkeypatch.setattr(P, "HEAD_CHUNK", chunk)
        bundle.model.zero_grad(set_to_none=True)
        spy = _HeadSpy(bundle.model)
        try:
            lp = P.completion_logprobs(pol, seq, plen)
            (lp * torch.linspace(-1, 1, lp.numel()).reshape(lp.shape)).sum().backward()
            calls_grad = list(spy.calls)
            spy.calls.clear()
            with torch.no_grad():
                P.completion_logprobs(pol, seq, plen)
            calls_nograd = list(spy.calls)
        finally:
            spy.close()
        out[chunk] = (lp.detach(), [None if p.grad is None else p.grad.clone() for p in params], calls_grad, calls_nograd)
    (lp0, g0, cg0, cn0), (lp1, g1, cg1, cn1) = out[10**9], out[4]
    n_chunks = -(-2 * new // 4)
    assert len(cn1) == n_chunks and max(cn1) <= 4 and len(cg1) == 2 * n_chunks
    assert (lp0 - lp1).abs().max() < 1e-5
    gmax = max(float(a.abs().max()) for a in g0 if a is not None)
    assert max(float((a - b).abs().max()) for a, b in zip(g0, g1) if a is not None) < 1e-5 * gmax


@needs_standin
def test_default_is_one_head_call_without_checkpoint_and_bit_identical(bundle, monkeypatch):
    """Default (head_checkpoint off, job RLM2): under autograd ONE head call over all completion tokens, no checkpoint,
    log-probs bit-identical to the old full forward and gradients equal. Fails if the default checkpoints."""
    pol = bundle.policy()
    assert pol.head_checkpoint is False
    monkeypatch.setattr(P, "HEAD_CHUNK", 4)  # chunks would be visible if the default used them
    n_ckpt = []
    real = P.checkpoint
    monkeypatch.setattr(P, "checkpoint", lambda *a, **k: (n_ckpt.append(1), real(*a, **k))[1])
    plen, new = 5, 13
    seq = _seq(bundle.spec["vocab_size"], 2, plen, new, seed=7)
    params = [p for p in bundle.model.parameters() if p.requires_grad]
    w = torch.linspace(-1, 1, 2 * new).reshape(2, new)
    res = []
    for fn in (_old_lp, P.completion_logprobs):
        bundle.model.zero_grad(set_to_none=True)
        spy = _HeadSpy(bundle.model)
        try:
            lp = fn(pol, seq, plen)
            (lp * w).sum().backward()
        finally:
            spy.close()
        res.append((lp.detach(), [None if p.grad is None else p.grad.clone() for p in params], list(spy.calls)))
    (lp0, g0, c0), (lp1, g1, c1) = res
    assert not n_ckpt and c1 == [2 * new] and c0 == [2 * (plen + new)]
    assert torch.equal(lp0, lp1)
    # gradients are not bit-identical: the head's weight gradient sums over fewer positions (the prompt rows of the old
    # path contribute exact zeros) in a different order. Measured 1.8e-6 of the max on the CPU stand-in, 3.9e-7 on a T4.
    gmax = max(float(a.abs().max()) for a in g0 if a is not None)
    assert max(float((a - b).abs().max()) for a, b in zip(g0, g1) if a is not None) <= 1e-5 * gmax


def test_set_head_checkpoint_reads_the_train_config_and_refuses_policies_without_a_split_head():
    import pytest
    pol = P.module_policy(nn.Linear(2, 2), split_head=False)
    assert train.set_head_checkpoint(pol, {}) is False and pol.head_checkpoint is False
    with pytest.raises(ValueError, match="hidden/head"):
        train.set_head_checkpoint(pol, {"head_checkpoint": True})

    class M(nn.Module):
        def __init__(self):
            super().__init__()
            self.head = nn.Linear(2, 3)

        def hidden(self, ids, valid):
            return torch.zeros(*ids.shape, 2)
    split = P.module_policy(M(), split_head=True)
    assert train.set_head_checkpoint(split, {"head_checkpoint": True}) is True and split.head_checkpoint is True
    assert train.set_head_checkpoint(split, {"head_checkpoint": False}) is False and split.head_checkpoint is False
