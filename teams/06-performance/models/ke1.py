"""Kronecker Embedding v1 (input side), drop-in for nn.Embedding.  [implemented]

Ported from the reference implementation https://github.com/theschoolofai/kronecker-embeddings
(src/kronecker_embeddings/codec.py and embedding.py, Apache-2.0, Copyright 2026 The School of AI).
Same maths, same defaults; only the constructor takes our byte-table tuple instead of an HF tokenizer.

Definition:
    kappa(t) = (1/sqrt(L)) * sum_p onehot256(b_p) (x) onehot_P(p)        D = 256 * P (P = 32: 8192)
    index of byte b at slot p = b * P + p;  L = number of bytes (truncated to P)
    kappa is then z-normalised per token: (kappa - mean) / (std + 1e-6)  (std with Bessel correction)
    embedding(t) = W @ kappa(t),  W learned [d_model, D], no bias, init normal(0, 1/sqrt(D)).
    The codec itself is fixed (no parameters).

Token byte table (the data contract, used by tests too): a tuple
    (bytes [V, P] uint8, lens [V] int64); bytes past lens[t] are ignored; lens >= 1.

Modes (dynamic and cached are the upstream names):
    dynamic  keep only the [V, P] byte table, recompute the codec every forward (default)
    cached   precompute the [V, D] codec table once, forward is an index_select
    sparse   never build the D-dim codec (closed form below); G5's harness uses upstream "cached"
All three give the same output. Memory at B=32, T=512: dynamic builds a [16384, 8192] fp32 codec (512 MB,
plus ~512 MB of z-norm temporaries; 256 MB bf16 saved for backward); cached keeps a [V, D] fp32 table
(4 GiB for V = 131072). sparse keeps only the byte table and gathers L <= P columns of W per token.

Sparse closed form (G4's Lemma 1: every code has two values). kappa has n = min(L, P) entries of a = 1/sqrt(L) at
idx_p = b_p * P + p and zeros elsewhere, so mean m = n * a / D, var = (n * a^2 - D * m^2) / (D - 1) (Bessel), and
    W @ z = (a * sum_p W[:, idx_p] - m * W.sum(1)) / (std + eps)
Per token that is 2 * P * d FLOPs forward instead of 2 * D * d (perfkit/metrics.py counts it in HFU).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from perfkit.regions import region

N_BYTES = 256
MODES = ("dynamic", "cached", "sparse")


def token_bytes_random(vocab_size: int, max_pos: int = 32, seed: int = 0):
    """Synthetic byte table for tests and offline runs (lengths 1..8, like real BPE tokens)."""
    g = torch.Generator().manual_seed(seed)
    tb = torch.randint(0, N_BYTES, (vocab_size, max_pos), generator=g, dtype=torch.uint8)
    lens = torch.randint(1, 9, (vocab_size,), generator=g)
    return tb, lens


def kronecker_codec(byte_seqs: torch.Tensor, lens: torch.Tensor, pos_dim: int = 32,
                    length_normalize: bool = True, z_normalize: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """byte_seqs [N, pos_dim] uint8/long, lens [N] -> [N, 256 * pos_dim] fp32 (upstream kronecker_codec)."""
    N, dev = byte_seqs.size(0), byte_seqs.device
    pos = torch.arange(pos_dim, device=dev).unsqueeze(0).expand(N, -1)
    lin_idx = byte_seqs.long() * pos_dim + pos
    lens = lens.long()
    src = (pos < lens.unsqueeze(1)).float()
    if length_normalize:
        src = src * torch.rsqrt(lens.clamp_min(1).float()).unsqueeze(1)
    out = torch.zeros((N, N_BYTES * pos_dim), device=dev, dtype=torch.float32)
    out.scatter_add_(1, lin_idx, src)
    if z_normalize:                                   # in place: same values as upstream, 1 temporary less
        out.sub_(out.mean(dim=-1, keepdim=True))
        out.div_(out.std(dim=-1, keepdim=True) + eps)
    return out


class KroneckerEmbedding(nn.Module):
    def __init__(self, token_bytes, d_model: int, max_pos: int = 32, mode: str = "dynamic"):
        super().__init__()
        assert mode in MODES, f"KE mode {mode!r} not in {MODES}"
        tb, lens = token_bytes
        assert tb.dim() == 2 and tb.size(1) == max_pos, f"byte table must be [V, {max_pos}], got {tuple(tb.shape)}"
        self.max_pos, self.mode, self.D = max_pos, mode, N_BYTES * max_pos
        self.register_buffer("byte_table", tb.to(torch.uint8), persistent=False)
        self.register_buffer("lens", lens.to(torch.int64), persistent=False)
        self.proj = nn.Linear(self.D, d_model, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(self.D))
        if mode == "cached":
            with torch.no_grad():
                self.register_buffer("codec_table", kronecker_codec(self.byte_table, self.lens, max_pos),
                                     persistent=False)

    def codec(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [...] -> kappa [..., D] fp32."""
        flat = idx.reshape(-1)
        if self.mode == "cached":
            out = self.codec_table.index_select(0, flat)
        else:
            out = kronecker_codec(self.byte_table.index_select(0, flat), self.lens.index_select(0, flat),
                                  self.max_pos)
        return out.view(*idx.shape, self.D)

    def sparse_embed(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [...] -> W @ kappa [..., d_model] without the D-dim codec (closed form in the module docstring)."""
        flat = idx.reshape(-1)
        P, W = self.max_pos, self.proj.weight
        tb = self.byte_table.index_select(0, flat).long()
        lens = self.lens.index_select(0, flat)
        pos = torch.arange(P, device=flat.device)
        cols = tb * P + pos                                             # [N, P] codec index of each byte
        valid = (pos < lens.unsqueeze(1)).float()                       # bytes past lens[t] weigh 0
        # embedding_bag sums the chosen rows of W^T without a [N, P, d] gather; its backward scatters into W.
        s = F.embedding_bag(cols, W.t(), per_sample_weights=valid.to(W.dtype), mode="sum").float()
        n = valid.sum(1)
        a = torch.rsqrt(lens.clamp_min(1).float())                      # kronecker_codec normalises by lens
        m = n * a / self.D
        std = ((n * a * a - self.D * m * m) / (self.D - 1)).sqrt()
        out = (a.unsqueeze(1) * s - m.unsqueeze(1) * W.float().sum(1)) / (std + 1e-6).unsqueeze(1)
        return out.to(W.dtype).view(*idx.shape, -1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        with region("embed.ke", 3):
            if self.mode == "sparse":
                return self.sparse_embed(idx)
            return self.proj(self.codec(idx).to(self.proj.weight.dtype))
