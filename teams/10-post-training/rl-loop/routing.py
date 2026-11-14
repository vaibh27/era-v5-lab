"""Routing stability: which experts each token picks, how evenly the load spreads, and how often picks change.

A frozen router keeps its weights, but the experts a token picks can still change, because the router's input (the
hidden state) depends on the attention weights we train. Gate 4 asks for a frozen router; this module measures
whether routing actually stays stable under RL:

    routing(policy, seq) [L, N, E]  dense routing weights: the top-k gate weights scattered over all E experts
    expert_load(w)       [L, E]  share of (token, slot) picks that go to each expert, per layer (rows sum to 1)
    load_ratio(load)     [L]     max share / mean share per layer (1.0 = perfectly even; 2.0 = one expert gets 2x)
    flip_rate(a, b)      share of tokens whose top-k expert SET differs in at least one layer, and per layer
    top1_flip_rate(a, b) share of (token, layer) pairs whose highest-weight expert differs
    mass_shift(a, b)     mean over (token, layer) of 0.5 * L1 distance between normalised routing weights
                         (0 = same experts with the same weights, 1 = completely different experts)

Why three change measures (A2, 5 Nov): the router scores in bf16, so the 8th and 9th expert are often nearly
tied, and a tiny weight change swaps them. The set flip rate counts each such swap in any of 16 layers, so it
is high even when almost no routing weight moves. mass_shift and top1_flip_rate say how much actually moved.

The trainer uses flip_rate in two ways: before vs after each optimiser step on the same tokens (drift per step),
and step 0 vs now on a fixed probe batch (drift over the run). Routing is read with forward hooks on the router
modules (OLMoE: model.layers.N.mlp.gate, which returns (logits, scores, top-k indices)). Models without such a
module (Team 5's) return None, and the trainer skips the log.

Not measured here: the routing difference between vLLM and the training forward (R3, arXiv 2510.11370), because
vLLM does not hand back its expert choices in this setup.
"""
from __future__ import annotations

import re
from contextlib import contextmanager

import torch
from torch import no_grad  # decorators without a dot: pr-guard reads '+@a.b' as an email

ROUTER_MODULE = re.compile(r"(^|\.)mlp\.gate$")


def router_modules(model) -> list:
    return [m for n, m in model.named_modules() if ROUTER_MODULE.search(n)]


@contextmanager
def record(model):
    """Collects dense routing weights [N, E] per layer, in layer order, for every forward inside the block."""
    store: list[torch.Tensor] = []

    def hook(mod, _inp, out):
        if isinstance(out, tuple) and len(out) >= 3:  # OLMoE: (logits, top-k weights, top-k ids)
            logits, w, idx = out[0], out[1], out[2]
        else:  # a plain linear router: softmax, then top-k
            logits = out[0] if isinstance(out, tuple) else out
            w, idx = torch.softmax(logits.float(), -1).topk(mod.top_k, -1)
        n_exp = logits.shape[-1]
        w = w.detach().float().reshape(-1, idx.shape[-1])
        dense = torch.zeros(w.shape[0], n_exp, device=w.device).scatter_(1, idx.reshape(w.shape[0], -1), w)
        store.append(dense.cpu())

    handles = [m.register_forward_hook(hook) for m in router_modules(model)]
    try:
        yield store
    finally:
        for h in handles:
            h.remove()


@no_grad()
def routing(policy, seq: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor | None:
    """Dense routing weights [L, N, E] for the tokens where mask is True (all tokens if mask is None)."""
    if not router_modules(policy.model):
        return None
    was_training = policy.model.training
    policy.model.eval()
    with record(policy.model) as store:
        policy.logits(seq)
    policy.model.train(was_training)
    w = torch.stack(store)
    if mask is not None:
        w = w[:, mask.reshape(-1).cpu()]
    return w


def completion_positions(prompt_len: int, cmask: torch.Tensor) -> torch.Tensor:
    """Full-sequence mask [k, P + T] that selects the completion tokens only."""
    return torch.cat([torch.zeros(cmask.shape[0], prompt_len, dtype=torch.bool, device=cmask.device), cmask], 1)


def expert_load(w: torch.Tensor, n_experts: int | None = None) -> torch.Tensor:
    picks = (w > 0).float().sum(1)  # [L, E]
    return picks / picks.sum(-1, keepdim=True).clamp_min(1)


def load_ratio(load: torch.Tensor) -> torch.Tensor:
    return load.max(-1).values / load.mean(-1)


def flip_rate(a: torch.Tensor, b: torch.Tensor) -> tuple[float, list[float]]:
    """(share of tokens with a different expert set in >= 1 layer, per-layer share). a, b: [L, N, E] weights."""
    diff = ((a > 0) != (b > 0)).any(-1)  # [L, N]
    return float(diff.any(0).float().mean()), [round(float(x), 4) for x in diff.float().mean(-1)]


def top1_flip_rate(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a.argmax(-1) != b.argmax(-1)).float().mean())


def mass_shift(a: torch.Tensor, b: torch.Tensor) -> float:
    na = a / a.sum(-1, keepdim=True).clamp_min(1e-12)
    nb = b / b.sum(-1, keepdim=True).clamp_min(1e-12)
    return float(0.5 * (na - nb).abs().sum(-1).mean())
