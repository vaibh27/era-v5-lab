"""KE2 decode in torch: KE code vectors -> token ID, best cosine and decode margin.  [implemented: first fast version]

The fast side of the KE2 pair. The reference is G4's teams/04-ke2/src/decode_ref.py (numpy, float64, CPU,
exhaustive; contract in teams/04-ke2/decode_ref.md). This computes the same thing in torch, on any device and
in fp32 or fp64, so it can run next to the model on the GPU. correctness/test_ke2_decode.py checks it against
decode_ref on G4's committed raw_table_P32.npz.

Contract (G4 decode_ref.md, 29 Oct):
    input   q [N, 256 * P] or [256 * P]: unprojected KE1 forward-codec vectors (models.ke1.kronecker_codec)
    table   the byte table the encoder used, models.ke1 contract (bytes [V, P] uint8, lens [V]); row = token ID
    output  ids [N] int64, scores [N] best cosine, margins [N] best minus second-best cosine (>= 0)
    ties    the lowest ID wins and the margin is 0 (two tokens with the same code, e.g. U+FFFD rows)
    errors  wrong width, nonfinite values or a zero query raise ValueError (decode_ref raises DecodeInputError)

Scoring, G4's Lemma 1: every code has two values, high on its n = min(L, P) support cells b_p * P + p and low
elsewhere, so

    q . z_t = low_t * sum(q) + (high_t - low_t) * sum_{p < n_t} q[b_p * P + p]

done as P gathers of [N, V], one per byte slot, in slot order (a pad slot reads an appended zero column), never a
[V, D] matrix (8.6 GB at P = 32 in fp64). Per query that is P * V reads instead of D * V multiply-adds (1 / 256).
Every sum runs in a fixed order with no atomics, so two runs give identical bits on the same device.

Memory: chunk_rows queries at a time; temporaries ~4 * chunk_rows * V * itemsize (32 rows, V = 131072, fp32: 64 MiB).
Week 2: one fused kernel (gather, scale, top-2) instead of P gathers and a [chunk, V] score matrix.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from models.ke1 import N_BYTES


@dataclass
class Decoded:
    ids: torch.Tensor          # int64 [N]
    scores: torch.Tensor       # [N] best cosine
    margins: torch.Tensor      # [N] best minus second-best cosine


class KE2Decoder:
    def __init__(self, token_bytes, pos_dim: int = 32, device="cpu", dtype=torch.float32, eps: float = 1e-6,
                 chunk_rows: int = 32):
        tb, lens = token_bytes
        assert tb.dim() == 2 and tb.size(1) == pos_dim, f"byte table must be [V, {pos_dim}], got {tuple(tb.shape)}"
        assert tb.size(0) >= 2, "a margin needs at least two tokens"
        V, P = tb.shape
        self.P, self.D, self.V, self.dtype, self.chunk_rows = P, N_BYTES * P, V, dtype, chunk_rows
        lens = lens.to(torch.float64)
        n = lens.clamp(max=P)                                           # support cells inside P
        a = torch.where(lens > 0, lens.clamp_min(1).rsqrt(), torch.zeros_like(lens))
        # code values in fp64, as models.ke1.kronecker_codec (and G4's ke2.codec.code_values) define them
        m = n * a / self.D
        std = ((n * (a - m) ** 2 + (self.D - n) * m ** 2) / (self.D - 1)).sqrt() + eps
        high, low = (a - m) / std, -m / std
        norms = (n * high ** 2 + (self.D - n) * low ** 2).sqrt()
        pos = torch.arange(P)
        cells = tb.long() * P + pos                                     # [V, P]
        cells = torch.where(pos < n.long().unsqueeze(1), cells, self.D + pos)    # pad slots read zeros
        self.cells = cells.t().contiguous().to(device)                  # [P, V]: slot-major, one gather per slot
        self.low = low.to(device, dtype)
        self.step = (high - low).to(device, dtype)
        self.norms = norms.to(device, dtype)

    def scores(self, q: torch.Tensor) -> torch.Tensor:
        """Cosine of every query [n, D] to every code: [n, V]. The empty token (zero code) scores 0."""
        qp = torch.cat([q, q.new_zeros(q.size(0), self.P)], dim=1)     # the zero columns pad slots read
        on_support = qp[:, self.cells[0]]
        for p in range(1, self.P):
            on_support = on_support + qp[:, self.cells[p]]
        # sum(q) and |q| reduce over D = 8192 values: in fp32 |q| was 1.7e-5 off (relative), the whole fp32 score
        # error. In fp64 they cost n * D, nothing next to the n * V * P gathers.
        q64 = q.double()
        dots = self.low * q64.sum(1, keepdim=True).to(q.dtype) + self.step * on_support
        denom = q64.norm(dim=1, keepdim=True).to(q.dtype) * self.norms
        return torch.where(denom > 0, dots / denom, torch.zeros_like(dots))

    @torch.no_grad()
    def decode(self, q: torch.Tensor) -> Decoded:
        single = q.dim() == 1
        q = q.reshape(1, -1) if single else q
        if q.dim() != 2 or q.size(1) != self.D:
            raise ValueError(f"queries must be [N, {self.D}], got {tuple(q.shape)}")
        q = q.to(self.low.device, self.dtype)
        if not torch.isfinite(q).all():
            raise ValueError("queries must be finite")
        if (q.abs().amax(1) == 0).any():
            raise ValueError("a zero query has no direction")
        ids, best, margins = [], [], []
        for chunk in q.split(self.chunk_rows):
            s = self.scores(chunk)
            i = s.argmax(1, keepdim=True)                               # first maximum: the lowest ID on a tie
            b = s.gather(1, i)
            second = s.scatter(1, i, float("-inf")).amax(1, keepdim=True)
            ids.append(i.squeeze(1))
            best.append(b.squeeze(1))
            margins.append((b - second).squeeze(1))
        out = Decoded(torch.cat(ids), torch.cat(best), torch.cat(margins))
        if single:
            out = Decoded(out.ids[0], out.scores[0], out.margins[0])
        return out
