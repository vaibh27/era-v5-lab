"""NanoGPT, driven only by ModelConfig.  [implemented: dense core; MoE / KE v1 / reversible plug in]

Parts and where they come from:
  * GQAttention (grouped-query attention): hd = d / n_head; c_attn = Linear(d, (n_head + 2 * n_kv_head) * hd) gives
    q, k, v; F.scaled_dot_product_attention(is_causal=True, enable_gqa=True) shares each k/v head with
    n_head / n_kv_head query heads without copying (torch >= 2.5; older torch: repeat_interleave copies).
    c_proj = Linear(d, d). Both Linears have bias.
    Attributes n_head, n_kv_head. n_kv_head == n_head is plain multi-head. The attention test reads these names.
    cfg.attn_impl "eager" (G5 config_v0: fast attention backward does not repeat, so two seeds would not match):
    repeat_interleave k, v to n_head heads, then softmax(q k^T / sqrt(hd) + causal mask) v by explicit matmuls.
    Same region names as SDPA (".sdpa" times the attention core either way).
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
  * MTP (cfg.mtp, G5 config_v0): a second untied head, mtp_head = Linear(d, V, bias=False), reads the same ln_f
    output and predicts token t+2 (targets shifted by one more; the last position has none). The returned loss is
    ntp + mtp, the sum G5 backpropagates; model.loss_parts holds both, detached, for logging (G5 logs ntp as loss).
  * Loss: cfg.ce_impl "torch" materialises logits [B*T, V]; "chunked" checkpoints position chunks;
    "fused" tiles over the vocabulary and recomputes the linear operation in custom backward. The two
    memory-bounded paths return logits=None when targets are given. Loss averages non-ignored targets.

Timing markers (roofline and report read these names; keep them)
  region("embed", 2)                       around tok + pos embedding (KE adds "embed.ke" at level 3 inside)
  region("stack", 1)                       around all blocks
  region(f"block{i}", 2)                   around each block (plain stack; the reversible stack sets its own)
  region(f"block{i}.ln1" / ".ln2", 3)      each LayerNorm
  region(f"block{i}.attn.qkv" / ".attn.sdpa" / ".attn.proj", 3)
  region(f"block{i}.mlp", 3)               dense FFN (MoE regions are set inside models/moe.py)
  region("head+loss", 2)                   ln_f + head + cross-entropy (+ MTP head and its loss)
  region("head+loss.mtp", 3)               MTP head + its cross-entropy (cfg.mtp only)
  mark(x, "stack", 1)                      on the embedding output (backward of the stack ends here)
  mark(x, f"block{i}", 2)                  on each block input (plain stack only)
  mark(x, "head+loss", 1)                  on the stack output
  How bench turns these points into bwd:<name> segments (bwd:stack = the whole stack): perfkit/bench.py step 6.

The fused CE path is a portable PyTorch custom-autograd implementation; a single-kernel Triton/CUDA
backend and aux / z loss for non-reversible MoE remain follow-ups.
"""
from __future__ import annotations

import math

import torch
import torch.utils.checkpoint
import torch.nn as nn
import torch.nn.functional as F

from models.config import ModelConfig
from models.linear_ce import linear_cross_entropy
from perfkit.regions import mark, region


_SDPA_GQA = tuple(int(v) for v in torch.__version__.split(".")[:2]) >= (2, 5)

IGNORE_INDEX = -100              # a target with this value is left out of the loss and of its divisor (loss masks)


def _init_linear(lin: nn.Linear, std: float = 0.02):
    nn.init.normal_(lin.weight, mean=0.0, std=std)
    if lin.bias is not None:
        nn.init.zeros_(lin.bias)


def _residual_std(cfg: ModelConfig) -> float:
    return 0.02 / math.sqrt(2 * cfg.n_layer)


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
                future = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
                att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
                y = F.softmax(att.masked_fill(future, float("-inf")), dim=-1) @ v
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
        elif cfg.embedding == "ke2":
            from models.ke2 import KE2Embedding
            self.tok = KE2Embedding(cfg.vocab_size, d, cfg.ke_max_pos, token_bytes)   # G4's raw table by default
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
        if cfg.mtp:
            self.mtp_head = nn.Linear(d, cfg.vocab_size, bias=False)
            nn.init.normal_(self.mtp_head.weight, mean=0.0, std=0.02)
        self.loss_parts = None

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx, targets: [B, T] int64. Returns (logits, loss); loss is None without targets."""
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.cfg.block_size}")
        if self.cfg.mtp and targets is not None and T < 2:
            raise ValueError("mtp needs T >= 2: the t+2 target of the last position does not exist")
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
            if targets is not None and self.cfg.ce_impl in ("chunked", "fused"):
                loss = self._head_ce(self.head, x.reshape(B * T, -1), targets.reshape(B * T))
                return None, self._add_mtp(loss, x, targets)
            logits = self.head(x)
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(B * T, -1).float(), targets.reshape(B * T),
                                       ignore_index=IGNORE_INDEX)
                loss = self._add_mtp(loss, x, targets)
        return logits, loss

    def _add_mtp(self, ntp, x, targets):
        """ntp + the t+2 loss of mtp_head (cfg.mtp); ntp unchanged otherwise. x: ln_f output [B, T, d]."""
        if not self.cfg.mtp:
            return ntp
        with region("head+loss.mtp", 3):
            xm, ym = x[:, :-1].reshape(-1, x.shape[-1]), targets[:, 1:].reshape(-1)
            if self.cfg.ce_impl in ("chunked", "fused"):
                mtp = self._head_ce(self.mtp_head, xm, ym)
            else:
                mtp = F.cross_entropy(self.mtp_head(xm).float(), ym, ignore_index=IGNORE_INDEX)
        self.loss_parts = {"ntp": ntp.detach(), "mtp": mtp.detach()}
        return ntp + mtp

    def _chunked_ce(self, head, x, y):
        """Compute mean CE of head(x) in checkpointed position chunks to bound saved logits memory.

        Same value as the plain path: the mean is over the targets that are not IGNORE_INDEX, so masked
        positions (SFT loss masks) do not count in the divisor. If every target is ignored the loss is 0
        (the plain path gives NaN there)."""
        def piece(xc, yc):
            return F.cross_entropy(head(xc).float(), yc, reduction="sum", ignore_index=IGNORE_INDEX)
        total = sum(torch.utils.checkpoint.checkpoint(piece, xc, yc, use_reentrant=False)
                    for xc, yc in zip(x.chunk(self.cfg.ce_chunks), y.chunk(self.cfg.ce_chunks)))
        return total / (y != IGNORE_INDEX).sum().clamp_min(1)

    def _head_ce(self, head, x, y):
        if self.cfg.ce_impl == "fused":
            return linear_cross_entropy(
                x, head.weight, y, bias=head.bias,
                vocab_chunk=self.cfg.ce_vocab_chunk, ignore_index=IGNORE_INDEX,
            )
        return self._chunked_ce(head, x, y)

    def param_counts(self):
        """(total, active). active = matmul weights one token multiplies with: attention, router,
        top_k + shared experts, KE projection, LM head (and MTP head). Not biases, norms, lookup tables. Used for MFU.
        Counted by module, not by parameter name: a tied head shares its tensor with the embedding
        table and would otherwise drop out of the count."""
        total = sum(p.numel() for p in self.parameters())
        active = self.head.weight.numel() + (self.mtp_head.weight.numel() if self.cfg.mtp else 0)
        if self.cfg.embedding in ("ke1", "ke2"):
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
