"""Model config and YAML loading.  [CONTRACT]

Changing a field name or meaning changes a shared interface: bench, roofline, report and tests read these.

Rule: no number about the model is hard-coded anywhere else. Everything reads ModelConfig.
G5's configs become new configs/g5_*.yaml files. If they need a field we do not have,
add it here with a default, so every old config still loads.

    cfg = load_config("configs/lab_small.yaml", ["model.n_layer=4", "run.batch_size=8"])
    mcfg = model_config(cfg)          # ModelConfig; unknown keys are an error, not ignored
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import yaml

MOE_KEYS = {"num_experts", "top_k", "num_shared", "bias_gamma", "dispatch"}   # SparseMoE keyword arguments
CE_IMPLS = ("torch", "chunked")
ATTN_IMPLS = ("sdpa", "eager")
KE_MODES = ("dynamic", "cached", "sparse")


@dataclass
class ModelConfig:
    vocab_size: int = 131072         # exact BrahmicTokenizer-131K vocabulary, no padding
    block_size: int = 512            # sequence length T
    n_layer: int = 9
    n_head: int = 8                  # query heads
    n_kv_head: int | None = None     # GQA: key/value heads, divides n_head; None = n_head (plain multi-head)
    attn_impl: str = "sdpa"          # sdpa (F.scaled_dot_product_attention) | eager (scores, mask, softmax: G5 harness)
    n_embd: int = 256                # d_model
    mlp_ratio: int = 4               # MLP / expert hidden = mlp_ratio * n_embd
    embedding: str = "table"         # table (tied head) | ke1 (Kronecker v1, untied head)
    ke_max_pos: int = 32             # KE v1 byte positions: codec dim = 256 * ke_max_pos
    ke_mode: str = "dynamic"         # KE v1: dynamic (codec every step) | cached (V x D table) | sparse (no codec)
    ffn: str = "mlp"                 # mlp | moe
    moe: dict = field(default_factory=lambda: dict(num_experts=8, top_k=2, num_shared=0,
                                                   bias_gamma=0.01, dispatch="loop"))
    reversible: bool = False         # True: midpoint reversible stack
    rev_h: float = 1.0               # midpoint step size
    ce_impl: str = "torch"           # torch: full fp32 logits (baseline) | chunked: ce_chunks pieces, checkpointed
    ce_chunks: int = 8
    mtp: bool = False                # True: a second untied head predicts token t+2 (G5 config_v0 loss.train_on);
                                     # the loss is ntp + mtp, GPT.last_losses keeps both (ntp is the logged loss)

    def __post_init__(self):
        assert self.embedding in ("table", "ke1"), f"embedding={self.embedding!r}"
        assert self.ffn in ("mlp", "moe"), f"ffn={self.ffn!r}"
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        assert self.ce_impl in CE_IMPLS, f"ce_impl={self.ce_impl!r} not in {CE_IMPLS}"
        assert self.attn_impl in ATTN_IMPLS, f"attn_impl={self.attn_impl!r} not in {ATTN_IMPLS}"
        assert self.ke_mode in KE_MODES, f"ke_mode={self.ke_mode!r} not in {KE_MODES}"
        assert not (self.mtp and self.block_size < 3), "mtp needs block_size >= 3"
        unknown = sorted(set(self.moe) - MOE_KEYS)
        assert not unknown, f"unknown moe keys {unknown}; known: {sorted(MOE_KEYS)}"
        if self.n_kv_head is None:
            self.n_kv_head = self.n_head
        assert self.n_kv_head > 0 and self.n_head % self.n_kv_head == 0, "n_kv_head must be > 0 and divide n_head"

    def to_dict(self):
        return asdict(self)


def _merge(base, over):
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path, overrides=()):
    """YAML with `inherit: <parent>.yaml`, then `a.b.c=value` overrides (values parsed as YAML)."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _merge(load_config(path.parent / parent), cfg)
    for o in overrides:
        key, sep, val = o.partition("=")
        if not sep:
            raise ValueError(f"override {o!r} must look like key=value")
        node = cfg
        *head, last = key.split(".")
        for h in head:
            node = node.setdefault(h, {})
        node[last] = yaml.safe_load(val)
    return cfg


def model_config(cfg: dict) -> ModelConfig:
    m = dict(cfg.get("model", cfg))
    known = {f.name for f in fields(ModelConfig)}
    unknown = sorted(set(m) - known)
    if unknown:
        raise KeyError(f"unknown model keys {unknown}: add them to ModelConfig (with a default) first")
    return ModelConfig(**m)
