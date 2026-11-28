"""How full the training rows are under three ways of filling them, from shard document lengths.

Reads only `.idx.npy` document offsets (G1 shard format: one int64 offset per document boundary;
every stored document ends in one EOS). No token ids or text are read.

For each window length L it reports, for one pass over the documents:

- current:    what week2/data.py does today. Each row holds one window of one document; the last
              window of a document is padded to L. Rows = sum(ceil(len / L)).
- bucketed:   the same windows, but batches are formed from windows of similar length (sorted inside
              shuffled pools) and each batch is padded only to its longest window, not to L.
- packed:     documents placed end to end, each followed by its EOS, filling rows of exactly L;
              a document that does not fit continues in the next row. With a document mask (no
              attention across documents) the fill is the same; only the attention changes.

    python teams/05-architecture/harness/packing_fill.py --index a.idx.npy b.idx.npy --windows 2048 4096
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def document_lengths(paths):
    """Content tokens per document: offset differences minus the terminal EOS (as #825 counts them)."""
    out = []
    for p in paths:
        offsets = np.load(p, allow_pickle=False)
        if offsets.ndim != 1 or offsets.size < 2 or np.any(np.diff(offsets) < 1):
            raise SystemExit(f"{p}: not a G1 offset index")
        out.append(np.diff(offsets).astype(np.int64) - 1)
    return np.concatenate(out)


def windows(lengths, L):
    """Window lengths as the current loader cuts them: full windows of L, then the remainder."""
    full, rest = np.divmod(lengths, L)
    return np.concatenate([np.full(int(full.sum()), L, dtype=np.int64), rest[rest > 0]])


def current(lengths, L):
    rows = int(np.ceil(lengths / L).sum())
    return rows * L, rows


def bucketed(lengths, L, rows_per_batch, pool_batches=100, seed=1234):
    w = windows(lengths, L)
    rng = np.random.default_rng(seed)
    w = w[rng.permutation(w.size)]
    pool = rows_per_batch * pool_batches
    positions = 0
    for start in range(0, w.size, pool):
        chunk = np.sort(w[start:start + pool])[::-1]
        for b in range(0, chunk.size, rows_per_batch):
            batch = chunk[b:b + rows_per_batch]
            positions += int(batch[0]) * len(batch)  # padded to the longest window; a short last batch has fewer rows
    return positions, int(np.ceil(w.size / rows_per_batch))


def packed(lengths, L):
    stream = int(lengths.sum() + lengths.size)  # each document followed by its EOS
    rows = -(-stream // L)
    return rows * L, rows


def report(lengths, Ls, tokens_per_step):
    content = int(lengths.sum())
    lines = [f"Documents: {lengths.size:,}; content tokens: {content:,}; median length {int(np.median(lengths)):,}.", "",
             "| Window | Option | Positions per pass | Fill | Positions vs current |",
             "|---:|---|---:|---:|---:|"]
    for L in Ls:
        cur, _ = current(lengths, L)
        rows_per_batch = max(1, tokens_per_step // L)
        for name, (pos, _) in [("current", (cur, None)),
                               ("bucketed", bucketed(lengths, L, rows_per_batch)),
                               ("packed", packed(lengths, L))]:
            lines.append(f"| {L:,} | {name} | {pos:,} | {content / pos:.1%} | {pos / cur:.2f}x |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--index", type=Path, nargs="+", required=True, help=".idx.npy files")
    ap.add_argument("--windows", type=int, nargs="+", default=[2048, 4096])
    ap.add_argument("--tokens-per-step", type=int, default=16384,
                    help="input positions per optimizer step (dry run 1 and 2: 8 x 2,048)")
    args = ap.parse_args()
    print(report(document_lengths(args.index), args.windows, args.tokens_per_step))


if __name__ == "__main__":
    main()
