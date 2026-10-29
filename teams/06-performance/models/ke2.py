"""Kronecker Embedding v2 (input side), drop-in for nn.Embedding.  [implemented]

Built on G4's KE2 code in teams/04-ke2/src (the ke2 package), not a copy of it:
    * the byte table is G4's committed raw table, teams/04-ke2/results/tables/raw_table_P{P}.npz, loaded with
      ke2.tables.ByteTable.load, which re-checks its content hash and that it was built for our pinned
      tokenizer.json (dataio.tokenizer.SHA256). Raw = exact token bytes, so no two tokens share a row.
    * the code values come from ke2.codec.code_values (G4 proof.md, Lemma 1): after z-normalisation a token of
      L bytes has only two values, high[L] on its L support cells and low[L] on the other D - L cells.

Shape (G4 output_shape.md): token IDs [B, T] -> fixed code [B, T, D], D = 256 * P -> Linear(D, d_model, bias=False).
    kappa(t)     = high[L] on cells b_p * P + p (p < L), low[L] everywhere else
    embedding(t) = W @ kappa(t) = (high[L] - low[L]) * sum_{p<L} W[:, b_p * P + p] + low[L] * W.sum(1)
The second form is what forward computes: one embedding_bag over the L support columns of W plus one row sum,
so the [N, D] code (512 MB fp32 at B=32, T=512) is never built. W is proj.weight [d_model, D], init
normal(0, 1/sqrt(D)); it is an embedding parameter, so the optimizer gives it AdamW.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from perfkit.regions import region

N_BYTES = 256
G4_SRC = Path(__file__).resolve().parents[2] / "04-ke2" / "src"


def load_g4_table(vocab_size: int, max_pos: int = 32):
    """G4's raw byte table as (bytes [V, P] uint8, lens [V] int64), verified against our tokenizer."""
    if str(G4_SRC) not in sys.path:
        sys.path.insert(0, str(G4_SRC))
    from dataio.tokenizer import SHA256
    from ke2.config import TABLES_DIR
    from ke2.tables import ByteTable

    path = TABLES_DIR / f"raw_table_P{max_pos}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"{path}: G4 has no raw table for P={max_pos} (teams/04-ke2/src/build_tables.py)")
    t = ByteTable.load(path, tokenizer_sha256=SHA256)
    if t.vocab_size != vocab_size:
        raise ValueError(f"G4 table has {t.vocab_size} rows, model.vocab_size is {vocab_size}")
    if t.n_empty_rows:
        raise ValueError(f"G4 table has {t.n_empty_rows} empty rows: they would all share the zero code")
    return torch.from_numpy(t.bytes.copy()), torch.from_numpy(t.lengths.astype("int64"))


def code_values(max_pos: int):
    """(high [P + 1], low [P + 1]) fp32, indexed by token length L, from G4's ke2.codec.code_values (fp64)."""
    if str(G4_SRC) not in sys.path:
        sys.path.insert(0, str(G4_SRC))
    import numpy as np
    from ke2.codec import code_values as g4_code_values

    high, low = g4_code_values(np.arange(max_pos + 1), max_pos)
    return torch.from_numpy(high).float(), torch.from_numpy(low).float()


class KE2Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, max_pos: int = 32, token_bytes=None):
        """token_bytes: optional (bytes [V, P], lens [V]) for tests; default is G4's raw table."""
        super().__init__()
        tb, lens = token_bytes if token_bytes is not None else load_g4_table(vocab_size, max_pos)
        assert tb.dim() == 2 and tb.size(1) == max_pos, f"byte table must be [V, {max_pos}], got {tuple(tb.shape)}"
        assert int(lens.min()) >= 1 and int(lens.max()) <= max_pos, f"token lengths must be in 1..{max_pos}"
        self.max_pos, self.D = max_pos, N_BYTES * max_pos
        pos = torch.arange(max_pos)
        # Column of W for each (token, byte slot); slots past L get weight 0 below, so their column does not matter.
        self.register_buffer("cols", (tb.long() * max_pos + pos).to(torch.int32), persistent=False)
        self.register_buffer("lens", lens.to(torch.int64), persistent=False)
        high, low = code_values(max_pos)
        self.register_buffer("high", high, persistent=False)
        self.register_buffer("low", low, persistent=False)
        self.register_buffer("pos", pos, persistent=False)
        self.proj = nn.Linear(self.D, d_model, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1.0 / math.sqrt(self.D))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx [...] -> [..., d_model] fp32."""
        with region("embed.ke", 3):
            flat = idx.reshape(-1)
            L = self.lens.index_select(0, flat)
            cols = self.cols.index_select(0, flat).long()
            w = (self.pos < L.unsqueeze(1)).float() * (self.high - self.low).index_select(0, L).unsqueeze(1)
            W = self.proj.weight.float()
            out = F.embedding_bag(cols, W.t(), per_sample_weights=w, mode="sum")
            out = out + self.low.index_select(0, L).unsqueeze(1) * W.sum(1)
            return out.view(*idx.shape, -1)
