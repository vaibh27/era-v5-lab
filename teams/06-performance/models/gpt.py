"""NanoGPT, driven only by ModelConfig.  [implemented: dense core; MoE / KE v1 / reversible plug in]

Parts and where they come from:
  * GQAttention (grouped-query attention): hd = d / n_head; c_attn = Linear(d, (n_head + 2 * n_kv_head) * hd) gives
    q, k, v; F.scaled_dot_product_attention(is_causal=True, enable_gqa=True) shares each k/v head with
    n_head / n_kv_head query heads without copying (torch >= 2.5; older torch: repeat_interleave copies).
    c_proj = Linear(d, d). Both Linears have bias.
    Attributes n_head, n_kv_head. n_kv_head == n_head is plain multi-head. The attention test reads these names.
    cfg.attn_impl "eager" instead materialises q k^T / sqrt(hd), a causal -inf mask and the softmax (k, v heads
    repeated): what G5's harness runs so two seeds give the same loss. Same result, O(T^2) memory per head.
  * MLP: c_fc = Linear(d, mlp_ratio*d), GELU, c_proj = Linear(mlp_ratio*d, d).
  * Block i: pre-LN. ffn = MLP, or models.moe.SparseMoE when cfg.ffn == "moe".
        f(p) = a + FFN(LN2(p + a)),  a = Attn(LN1(p))
    Pre-LN only: Post-LN, LN(x + f(x)), normalises the residual stream, and a LayerNorm cannot be
    inverted, so the midpoint stack could not rebuild its inputs. The plain stack uses the same f:
    forward(x) = x + f(x).
    For the reversible stack the block also has update() and invert() (models/reversible.py):
        update(prev, curr) = prev + 2h * f(curr)          invert(prev_out, curr_out) = curr_out - 2h * f(prev_out)
  * GPT: token embedding (nn.Embedding tied with the head, or models.ke1.KroneckerEmbedding with an
    untied head: the codec dim is not d_model), learned position table, blocks (plain loop or
    models.reversible.ReversibleStack), ln_f, head, fp32 cross-entropy.
  * Init (NanoGPT): weights normal(0, 0.02), biases zero, residual projections (c_proj)
    normal(0, 0.02 / sqrt(2 * n_layer)); MoE experts get the same (proj_std). KE initialises its projection.
  * Loss: cfg.ce_impl "torch" materialises fp32 logits [B*T, V] (~8 GB with grads at B=32, T=512);
    "chunked" runs head + cross-entropy in cfg.ce_chunks pieces under activation checkpointing and
    returns logits=None when targets are given. Both paths average over the targets that are not IGNORE_INDEX.
  * MTP (cfg.mtp, G5 config_v0): mtp_head, a second untied Linear(d, V), predicts token t+2 from position t
    (targets[:, 1:] for positions :-1). loss = ntp + mtp; GPT.last_losses = {"ntp", "mtp"} (detached) after each
    forward with targets. Both heads go through the same CE path (torch or chunked).

Timing markers (roofline and report read these names; keep them)
  region("embed", 2)                       around tok + pos embedding (KE adds "embed.ke" at level 3 inside)
  region("stack", 1)                       around all blocks
  region(f"block{i}", 2)                   around each block (plain stack; the reversible stack sets its own)
  region(f"block{i}.ln1" / ".ln2", 3)      each LayerNorm
  region(f"block{i}.attn.qkv" / ".attn.sdpa" / ".attn.proj", 3)   (.attn.sdpa is the eager core too)
  region(f"block{i}.mlp", 3)               dense FFN (MoE regions are set inside models/moe.py)
  region("head+loss", 2)                   ln_f + head + cross-entropy (+ mtp_head and its loss)
  mark(x, "stack", 1)                      on the embedding output (backward of the stack ends here)
  mark(x, f"block{i}", 2)                  on each block input (plain stack only)
  mark(x, "head+loss", 1)                  on the stack output
  How bench turns these points into bwd:<name> segments (bwd:stack = the whole stack): perfkit/bench.py step 6.

Follow-up: fused linear-cross-entropy kernel; aux / z loss for non-reversible MoE.
"""
from __future__ import annotations

import math

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig
from perfkit.regions import mark, region


_SDPA_GQA = tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2, 5)

IGNORE_INDEX = -100              # a target with this value is left out of the loss and of its divisor (loss masks)


def _init_linear(lin: nn.Linear, std: float = 0.02):
    nn.init.normal_(lin.weight, mean=0.0, std=std)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


def _residual_std(cfg: ModelConfig) -> float:
    return 0.02 / math.sqrt(2 * cfg.n_layer)


def _eager_attention(q, k, v):
    """q, k, v [B, H, T, hd] (k, v already repeated to H heads) -> [B, H, T, hd]."""
    T = q.size(-2)
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
    future = torch.ones(T, T, dtype=torch.bool, device=q.device).triu(1)
    return torch.softmax(scores.masked_fill(future, float("-inf")), dim=-1) @ v


class GQAttention(nn.Module):
    def __init__(self, cfg: ModelConfig, name: str):
        super().__init__()
        d = cfg.n_embd
        self.n_head, self.n_kv_head, self.name = cfg.n_head, cfg.n_kv_head, name
        self.eager = cfg.attn_impl == "eager"
        self.c_attn = nn.Linear(d, (cfg.n_head + 2 * cfg.n_kv_head) * (d // cfg.n_head))
        self.c_proj = nn.Linear(d, d)
        _init_linear(self.c_attn)
        _init_linear(self.c_proj, _residual_std(cfg))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        hd, kv = C // self.n_head, self.n_kv_head
        with region(f"{self.name}.qkv", 3):
            q, k, v = self.c_attn(x).split([C, kv * hd, kv * hd], dim=2)
            q = q.view(B, T, self.n_head, hd).transpose(1, 2)
            k = k.view(B, T, kv, hd).transpose(1, 2)
            v = v.view(B, T, kv, hd).transpose(1, 2)
            gqa = kv != self.n_head
            if gqa and (self.eager or not _SDPA_GQA):
                k = k.repeat_interleave(self.n_head // kv, dim=1)
                v = v.repeat_interleave(self.n_head // kv, dim=1)
        with region(f"{self.name}.sdpa", 3):
            if self.eager:
                y = _eager_attention(q, k, v)
            elif gqa and _SDPA_GQA:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        with region(f"{self.name}.proj", 3):
            return self.c_proj(y.transpose(1, 2).contiguous().view(B, T, C))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig, name: str):
        super().__init__()
        d, h = cfg.n_embd, cfg.mlp_ratio * cfg.n_embd
        self.name = name
        self.c_fc = nn.Linear(d, h)
        self.c_proj = nn.Linear(h, d)
        _init_linear(self.c_fc)
        _init_linear(self.c_proj, _residual_std(cfg))

    def forward(self, x):
        with region(self.name, 3):
            return self.c_proj(F.gelu(self.c_fc(x)))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, i: int):
        super().__init__()
        d = cfg.n_embd
        self.name, self.h = f"block{i}", cfg.rev_h
        self.ln1 = nn.LayerNorm(d)
        self.attn = GQAttention(cfg, f"{self.name}.attn")
        self.ln2 = nn.LayerNorm(d)
        if cfg.ffn == "moe":
            from models.moe import SparseMoE
            self.ffn = SparseMoE(d, expert_hidden=cfg.mlp_ratio * d, name=f"{self.name}.moe",
                                 proj_std=_residual_std(cfg), **cfg.moe)
        else:
            self.ffn = MLP(cfg, f"{self.name}.mlp")

    def _ln(self, ln, x, tag):
        with region(f"{self.name}.{tag}", 3):
            return ln(x)

    def f(self, p):
        """Residual function: both stacks add (a multiple of) it to the stream."""
        a = self.attn(self._ln(self.ln1, p, "ln1"))
        return a + self.ffn(self._ln(self.ln2, p + a, "ln2"))

    def update(self, prev, curr):
        return prev + 2 * self.h * self.f(curr)

    def invert(self, prev_out, curr_out):
        return curr_out - 2 * self.h * self.f(prev_out)

    def forward(self, x):
        """Plain residual block: x + f(x)."""
        return x + self.f(x)


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig, token_bytes=None):
        """token_bytes: (bytes [V, P] uint8, lens [V]) from models.ke1, only when cfg.embedding == 'ke1'."""
        super().__init__()
        self.cfg = cfg
        d = cfg.n_embd
        if cfg.embedding == "ke1":
            if token_bytes is None:
                raise ValueError("embedding=ke1 needs the tokenizer's byte table (token_bytes)")
            if token_bytes[0].size(0) != cfg.vocab_size or token_bytes[1].size(0) != cfg.vocab_size:
                raise ValueError("KE byte table must match model.vocab_size")
            from models.ke1 import KroneckerEmbedding
            self.tok = KroneckerEmbedding(token_bytes, d, cfg.ke_max_pos, cfg.ke_mode)
        else:
            self.tok = nn.Embedding(cfg.vocab_size, d)
            nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        self.pos = nn.Embedding(cfg.block_size, d)
        nn.init.normal_(self.pos.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, cfg.vocab_size, bias=False)
        if cfg.embedding == "table":
            self.head.weight = self.tok.weight          # tied
        else:
            nn.init.normal_(self.head.weight, mean=0.0, std=0.02)
        self.mtp_head = None
        if cfg.mtp:                                     # never tied: a second output table
            self.mtp_head = nn.Linear(d, cfg.vocab_size, bias=False)
            nn.init.normal_(self.mtp_head.weight, mean=0.0, std=0.02)
        self.last_losses = None

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx, targets: [B, T] int64. Returns (logits, loss); loss is None without targets."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        with region("embed", 2):
            x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))
        x = mark(x, "stack", 1)
        with region("stack", 1):
            if self.cfg.reversible:
                from models.reversible import ReversibleStack
                x = ReversibleStack.apply(x, self.blocks)
            else:
                for i, blk in enumerate(self.blocks):
                    x = mark(x, f"block{i}", 2)
                    with region(f"block{i}", 2):
                        x = blk(x)
        x = mark(x, "head+loss", 1)
        with region("head+loss", 2):
            x = self.ln_f(x)
            if targets is not None and self.cfg.ce_impl == "chunked":
                return None, self._with_mtp(self._chunked_ce(x, targets, self.head), x, targets)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = self._with_mtp(self._plain_ce(logits, targets), x, targets)
        return logits, loss

    @staticmethod
    def _plain_ce(logits, y):
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), ignore_index=IGNORE_INDEX)

    def _with_mtp(self, ntp, x, targets):
        """ntp + the t+2 loss when cfg.mtp (position t predicts targets[t + 1]); records last_losses."""
        mtp = None
        if self.mtp_head is not None:
            xm, ym = x[:, :-1], targets[:, 1:]
            if self.cfg.ce_impl == "chunked":
                mtp = self._chunked_ce(xm, ym, self.mtp_head)
            else:
                mtp = self._plain_ce(self.mtp_head(xm), ym)
        self.last_losses = {"ntp": ntp.detach(), "mtp": None if mtp is None else mtp.detach()}
        return ntp if mtp is None else ntp + mtp

    def _chunked_ce(self, x, y, head):
        """Compute mean CE in checkpointed position chunks to bound saved logits memory.

        Same value as the plain path: the mean is over the targets that are not IGNORE_INDEX, so masked
        positions (SFT loss masks) do not count in the divisor. If every target is ignored the loss is 0
        (the plain path gives NaN there). x [..., d], y [...]: flattened here."""
        x, y = x.reshape(-1, x.size(-1)), y.reshape(-1)

        def piece(xc, yc):
            return F.cross_entropy(head(xc).float(), yc, reduction="sum", ignore_index=IGNORE_INDEX)
        total = sum(torch.utils.checkpoint.checkpoint(piece, xc, yc, use_reentrant=False)
                    for xc, yc in zip(x.chunk(self.cfg.ce_chunks), y.chunk(self.cfg.ce_chunks)))
        return total / (y != IGNORE_INDEX).sum().clamp_min(1)

    def param_counts(self):
        """(total, active). active = matmul weights one token multiplies with: attention, router,
        top_k + shared experts, KE projection, LM head (and the MTP head). Not biases, norms, lookup tables. Used for MFU.
        Counted by module, not by parameter name: a tied head shares its tensor with the embedding
        table and would otherwise drop out of the count."""
        total = sum(p.numel() for p in self.parameters())
        active = self.head.weight.numel()
        if self.mtp_head is not None:
            active += self.mtp_head.weight.numel()
        if self.cfg.embedding == "ke1":
            active += self.tok.proj.weight.numel()
        for blk in self.blocks:
            active += blk.attn.c_attn.weight.numel() + blk.attn.c_proj.weight.numel()
            ffn = blk.ffn
            if isinstance(ffn, MLP):
                active += ffn.c_fc.weight.numel() + ffn.c_proj.weight.numel()
            else:
                per_expert = ffn.experts[0].c_fc.weight.numel() + ffn.experts[0].c_proj.weight.numel()
                active += ffn.router.weight.numel() + per_expert * (ffn.top_k + len(ffn.shared))
        return total, active

    def moe_layers(self):
        """All SparseMoE modules (the bench calls update_bias() on each after every step)."""
        if self.cfg.ffn != "moe":
            return []
        return [blk.ffn for blk in self.blocks]
