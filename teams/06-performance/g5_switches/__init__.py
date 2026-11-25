"""G6 speed switches for the G5 trainer, applied at run time. No file under teams/05-architecture changes.

`train.py` runs G5's `harness/train_week2.py` unchanged, with these switches patched in first:

  attn_impl=sdpa   global (causal) attention layers use torch's fused scaled_dot_product_attention (Flash on
                   SM80+) instead of eager fp32 scores. Local (windowed) layers stay eager. No new parameters.
                   Changes the numeric trace (fused softmax, bf16 scores).

The default (`attn_impl=eager`) patches nothing, so the run is G5's code as is.

Safety:
  * G5_GQA_FORWARD_SHA256 pins the G5 attention code the patch was checked against. If G5 changes
    `GQA.forward`, `install` stops the run until this patch is re-checked.
  * The chosen switches and this package's source hashes go into the resolved config under
    `provenance.g6_switches`, which G5 hashes into the run identity and writes to the manifest: a run with
    switches cannot resume a checkpoint made without them, or the other way round.
"""
from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import torch
from torch.nn import functional as F

HERE = Path(__file__).resolve().parent
SWITCHES = {"attn_impl": ("eager", "sdpa")}
DEFAULTS = {"attn_impl": "eager"}

# sha256 of the source of G5's GQA.forward (g5_method_source) on the G5 code this patch was checked against.
G5_GQA_FORWARD_SHA256 = "729b0093164710d9b9654ba298e5b6b2401a25d87ca7e5645308a4e50e610329"


def parse(pairs):
    """['attn_impl=sdpa', ...] -> {'attn_impl': 'sdpa'}, checked against SWITCHES, defaults filled in."""
    out = dict(DEFAULTS)
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or key not in SWITCHES:
            raise ValueError(f"unknown G6 switch {pair!r}; known: {sorted(SWITCHES)}")
        if value not in SWITCHES[key]:
            raise ValueError(f"{key} must be one of {SWITCHES[key]}, got {value!r}")
        out[key] = value
    return out


def g5_method_source(path, cls, name) -> str:
    """Exact source lines of method `cls.name` in the file at `path` (read from disk, no import)."""
    text = Path(path).read_text(encoding="utf8")
    lines = text.splitlines(keepends=True)
    for node in ast.parse(text).body:
        if isinstance(node, ast.ClassDef) and node.name == cls:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == name:
                    return "".join(lines[item.lineno - 1:item.end_lineno])
    raise LookupError(f"{cls}.{name} not found in {path}")


def g5_forward_sha256(model_module) -> str:
    return hashlib.sha256(g5_method_source(model_module.__file__, "GQA", "forward").encode()).hexdigest()


def provenance(switches) -> dict:
    """What goes into the G5 run identity: the switches and the sha256 of every file in this package."""
    sources = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(HERE.glob("*.py"))}
    return {"switches": dict(sorted(switches.items())), "sources": sources}


def _sdpa_forward(self, x, residual=None):
    """G5's GQA.forward with the score/softmax/value part replaced by fused SDPA (global layers only)."""
    b, t, w = x.shape
    q = self.rotary(self.q(x).view(b, t, self.nh, self.d).transpose(1, 2))
    k = self.rotary(self.k(x).view(b, t, self.nkv, self.d).transpose(1, 2))
    v = self.v(x).view(b, t, self.nkv, self.d).transpose(1, 2)
    # Fused causal attention on the shared K/V heads, without a T x T score matrix.
    heads = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=self.nh != self.nkv)
    out = heads.transpose(1, 2).reshape(b, t, w)
    # Apply the channel gate before mixing heads with the output projection (as in G5).
    if self.gate is not None:
        gate_input = x if self.gate_input == "normalized" else residual
        if gate_input is None:
            raise ValueError("Residual gate input requires the pre-norm residual")
        out = out * torch.sigmoid(self.gate(gate_input))
    return self.o(out)


def install(switches, model_module=None, check_source=True):
    """Patch G5's classes for the chosen switches. Idempotent; eager patches nothing."""
    if model_module is None:
        from week2 import model as model_module
    gqa = model_module.GQA
    if switches.get("attn_impl", "eager") == "sdpa":
        original = getattr(gqa, "_g5_forward", gqa.forward)
        if check_source and g5_forward_sha256(model_module) != G5_GQA_FORWARD_SHA256:
            raise RuntimeError(
                "G5 changed GQA.forward since the G6 SDPA patch was checked; re-check g5_switches and update "
                "G5_GQA_FORWARD_SHA256 before running with attn_impl=sdpa"
            )

        def forward(self, x, residual=None):
            if self.local:   # windowed layers keep G5's eager path (window mask and sink)
                return original(self, x, residual)
            return _sdpa_forward(self, x, residual)

        gqa._g5_forward = original
        gqa.forward = forward
    return gqa


def uninstall(model_module=None):
    """Undo `install` (tests)."""
    if model_module is None:
        from week2 import model as model_module
    gqa = model_module.GQA
    if hasattr(gqa, "_g5_forward"):
        gqa.forward = gqa._g5_forward
        del gqa._g5_forward
