"""Frozen MoE router: freeze the router weights, checksum them, and prove they did not change.

Gate 4 (TSAI plan, week 4): "Run the RL loop on the small MoE model with frozen router weights. Check that the router
checksum stays the same after RL." Evidence for freezing: in the MiMo-V2.6 report (arXiv 2610.11959) a trainable
router collapsed expert load within 20 RL steps, and freezing it fixed that.

Router parameter names (regex, matched against `model.named_parameters()`):
    OLMoE (Hugging Face)       model.layers.N.mlp.gate.weight      (not experts.gate_up_proj)
    Team 5 (harness/week2)     blocks.N.moe.router.weight
"""
from __future__ import annotations

import hashlib
import re

import torch
from torch import nn

ROUTER_PATTERNS = (r"(^|\.)mlp\.gate\.weight$", r"(^|\.)moe\.router\.weight$")


def router_params(model: nn.Module, patterns=ROUTER_PATTERNS) -> list[tuple[str, nn.Parameter]]:
    rx = [re.compile(p) for p in patterns]
    return [(n, p) for n, p in model.named_parameters() if any(r.search(n) for r in rx)]


def freeze_router(model: nn.Module, patterns=ROUTER_PATTERNS) -> list[str]:
    """Set requires_grad=False on every router weight. Raises if the model has none (wrong patterns)."""
    found = router_params(model, patterns)
    if not found:
        raise ValueError(f"no router parameters match {patterns}; is this an MoE model?")
    for _, p in found:
        p.requires_grad_(False)
    return [n for n, _ in found]


def router_checksum(model: nn.Module, patterns=ROUTER_PATTERNS) -> str:
    """SHA-256 over the router tensors' names, shapes, dtypes and exact bytes (in name order)."""
    h = hashlib.sha256()
    for name, p in sorted(router_params(model, patterns), key=lambda x: x[0]):
        t = p.detach().to("cpu").contiguous()
        h.update(f"{name}|{tuple(t.shape)}|{t.dtype}|".encode())
        h.update(t.view(torch.uint8).numpy().tobytes())  # raw bytes; works for bf16 too
    return h.hexdigest()


def assert_router_frozen(model: nn.Module, patterns=ROUTER_PATTERNS) -> None:
    """Fails if any router weight is trainable."""
    trainable = [n for n, p in router_params(model, patterns) if p.requires_grad]
    if trainable:
        raise AssertionError(f"router weights are trainable: {trainable}")
