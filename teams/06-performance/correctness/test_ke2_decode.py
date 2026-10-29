"""KE2 decode: models.ke2_decode.KE2Decoder against an fp64 dense reference and against G4's decode_ref.py.

The decoder has no gradient, so these are tests, not framework pairs. IDs must match exactly; scores and margins
within a tolerance per dtype (MARGIN_TOL below, our proposal to G4, who left the tolerance to G6):

  * dense: on a small random table, every code built byte by byte in fp64, the full cosine matrix, argmax and
    second best by sorting. Noisy queries, so the margins are not all the same.
  * ties: duplicate table rows give margin 0 and the lowest ID, like decode_ref.
  * G4: their committed raw_table_P32.npz (all 131072 BrahmicTokenizer-131K rows), 256 of their sample IDs
    (decode_ref.sample_token_ids, seed 0) encoded with our fp32 kronecker_codec. Our fp64 decoder must give
    their IDs and their scores and margins to 1e-12; our fp32 decoder their IDs and margins to MARGIN_TOL.
    Their decode table's U+FFFD tie (IDs 77 and 173) must decode to 77 with margin 0.
    Skipped when teams/04-ke2 or its dependencies are missing.
  * the byte table: G4's raw table equals ours from dataio/tokenizer.py, row for row (same tokenizer sha256).
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

from models.ke1 import N_BYTES, kronecker_codec, token_bytes_random
from models.ke2_decode import KE2Decoder

P, V = 32, 300
MARGIN_TOL = {torch.float64: 1e-12, torch.float32: 1e-5}     # |score / margin - decode_ref|, IDs exact
G4_SRC = Path(__file__).resolve().parents[2] / "04-ke2" / "src"
G4_RAW = G4_SRC.parent / "results" / "tables" / "raw_table_P32.npz"
G4_DECODE = G4_SRC.parent / "results" / "tables" / "decode_table_P32.npz"


def dense_codes(tb, lens, pos_dim=P, eps=1e-6):
    """fp64 codes [V, D], one byte at a time (the models/ke1.py definition)."""
    rows = []
    for t in range(tb.size(0)):
        n = min(int(lens[t]), pos_dim)
        k = torch.zeros(N_BYTES * pos_dim, dtype=torch.float64)
        for p in range(n):
            k[int(tb[t, p]) * pos_dim + p] += 1.0 / math.sqrt(int(lens[t]))
        rows.append((k - k.mean()) / (k.std() + eps))
    return torch.stack(rows)


def dense_decode(codes, q):
    c = codes / codes.norm(dim=1, keepdim=True)
    s = (q.double() / q.double().norm(dim=1, keepdim=True)) @ c.T
    top = s.sort(dim=1, descending=True, stable=True)
    return top.indices[:, 0], top.values[:, 0], top.values[:, 0] - top.values[:, 1]


def _queries(tb, lens, ids, noise=0.3, seed=0):
    q = kronecker_codec(tb[ids], lens[ids], P)
    g = torch.Generator().manual_seed(seed)
    return q + noise * torch.randn(q.shape, generator=g)


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_matches_dense_reference(dtype, device):
    tb, lens = token_bytes_random(V, P)
    ids = torch.arange(0, V, 7)
    q = _queries(tb, lens, ids)
    ref_ids, ref_s, ref_m = dense_decode(dense_codes(tb, lens), q)
    out = KE2Decoder((tb, lens), P, device=device, dtype=dtype).decode(q.to(device))
    assert torch.equal(out.ids.cpu(), ref_ids)
    assert (out.scores.cpu().double() - ref_s).abs().max() < MARGIN_TOL[dtype]
    assert (out.margins.cpu().double() - ref_m).abs().max() < MARGIN_TOL[dtype]


def _first_with_same_code(tb, lens):
    """For each token, the lowest ID with the same bytes (the random table has repeated short tokens)."""
    first, out = {}, []
    for t in range(tb.size(0)):
        key = bytes(tb[t, :int(lens[t])].tolist())
        out.append(first.setdefault(key, t))
    return torch.tensor(out)


def test_round_trip_and_chunks_agree(device):
    tb, lens = token_bytes_random(V, P)
    want = _first_with_same_code(tb, lens)
    q = kronecker_codec(tb, lens, P)
    a = KE2Decoder((tb, lens), P, device=device, chunk_rows=7).decode(q.to(device))
    b = KE2Decoder((tb, lens), P, device=device, chunk_rows=300).decode(q.to(device))
    assert torch.equal(a.ids.cpu(), want)
    assert torch.equal(a.margins, b.margins) and torch.equal(a.scores, b.scores)
    unique = torch.bincount(want, minlength=V)[want] == 1             # no other token has this code
    assert (a.margins.cpu()[unique] > 0).all() and (a.margins.cpu()[~unique] == 0).all()


def test_tie_goes_to_lowest_id_with_zero_margin(device):
    tb, lens = token_bytes_random(V, P)
    tb[200], lens[200] = tb[50], lens[50]                        # tokens 50 and 200 share a code
    q = kronecker_codec(tb[[200, 50]], lens[[200, 50]], P)
    out = KE2Decoder((tb, lens), P, device=device, dtype=torch.float64).decode(q.to(device))
    assert out.ids.tolist() == [50, 50]
    assert out.margins.tolist() == [0.0, 0.0]


def test_single_query_and_bad_inputs():
    tb, lens = token_bytes_random(V, P)
    dec = KE2Decoder((tb, lens), P)
    one = dec.decode(kronecker_codec(tb[3:4], lens[3:4], P)[0])
    assert one.ids.dim() == 0 and int(one.ids) == 3
    with pytest.raises(ValueError, match="queries must be"):
        dec.decode(torch.ones(2, 256 * 16))
    with pytest.raises(ValueError, match="finite"):
        dec.decode(torch.full((1, 256 * P), float("nan")))
    with pytest.raises(ValueError, match="zero query"):
        dec.decode(torch.zeros(1, 256 * P))


# ------------------------------------------------------------------------------ G4's decode_ref

@pytest.fixture(scope="module")
def g4():
    if not G4_RAW.is_file():
        pytest.skip(f"G4 table not found: {G4_RAW}")
    pytest.importorskip("numpy")
    sys.path.insert(0, str(G4_SRC))
    try:
        import decode_ref
        from ke2.config import TOKENIZER_SHA256
        from ke2.tables import ByteTable
    except ImportError as e:
        pytest.skip(f"G4 decode_ref not importable: {e}")
    return decode_ref, ByteTable, TOKENIZER_SHA256


def _ours(table):
    return torch.from_numpy(table.bytes.copy()), torch.from_numpy(table.lengths.astype("int64"))


@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
def test_matches_g4_decode_ref(g4, dtype, device):
    decode_ref, ByteTable, sha = g4
    table = ByteTable.load(G4_RAW, tokenizer_sha256=sha)
    tb, lens = _ours(table)
    ids = torch.from_numpy(decode_ref.sample_token_ids(table.vocab_size, n=256, seed=0))
    q = kronecker_codec(tb[ids], lens[ids], P)                       # fp32, like G4's upstream encoder
    ref = decode_ref.decode(q.double().numpy(), decode_ref.CodeTable.from_byte_table(table))
    out = KE2Decoder((tb, lens), P, device=device, dtype=dtype).decode(q.to(device))
    assert torch.equal(out.ids.cpu(), torch.from_numpy(ref.token_ids)), "token IDs differ from decode_ref"
    assert torch.equal(out.ids.cpu(), ids), "raw table must recover every sampled ID"
    tol = MARGIN_TOL[dtype]
    assert (out.scores.cpu().double() - torch.from_numpy(ref.scores)).abs().max() < tol
    assert (out.margins.cpu().double() - torch.from_numpy(ref.margins)).abs().max() < tol


def test_g4_decode_table_tie(g4, device):
    decode_ref, ByteTable, sha = g4
    table = ByteTable.load(G4_DECODE, tokenizer_sha256=sha)
    tb, lens = _ours(table)
    q = kronecker_codec(tb[[173, 77]], lens[[173, 77]], P)
    ref = decode_ref.decode(q.double().numpy(), decode_ref.CodeTable.from_byte_table(table))
    out = KE2Decoder((tb, lens), P, device=device, dtype=torch.float64).decode(q.to(device))
    assert ref.token_ids.tolist() == [77, 77] and ref.margins.tolist() == [0.0, 0.0]
    assert out.ids.tolist() == [77, 77] and out.margins.tolist() == [0.0, 0.0]


def test_g4_raw_table_is_our_byte_table(g4):
    _, ByteTable, sha = g4
    from dataio.tokenizer import SHA256, BrahmicTokenizer
    assert sha == SHA256, "G4 built its tables on another tokenizer than ours"
    try:
        tok = BrahmicTokenizer()
    except OSError as e:
        pytest.skip(f"tokenizer not available offline: {e}")
    tb, lens = tok.token_bytes(P)
    g4_tb, g4_lens = _ours(ByteTable.load(G4_RAW, tokenizer_sha256=sha))
    assert torch.equal(lens, g4_lens) and torch.equal(tb, g4_tb)
