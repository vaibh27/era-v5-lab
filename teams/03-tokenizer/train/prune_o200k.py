#!/usr/bin/env python3
"""
prune_o200k.py
==============

o200k pruning by usage for v2 (#163, v2_spec.md step 1). Writes o200k_v2.json:

  prune   v1 base tokens that are never used, in the order they may be removed.
          Usage = the v1 base tokenizer on held-out English (usage/En.txt), held-out
          code (usage/Code.txt) and the full Indic training corpus. A token is a
          candidate only if it is used 0 times, has more than one byte, contains no
          Brahmic character, and no kept base token is merged from it. Order:
          non-Latin scripts first (Cyrillic, CJK, Arabic, ...), then Common, then
          Latin; within a group the rarest o200k rank first. Every prefix is a valid
          prune set: the base stays closed under its merges.
  ranked  o200k tokens past the v1 base cut, ranked separately by tokens saved on
          held-out English and on held-out code:
              gain_d(t) = count_d(t) * (pieces(t) - 1)
          with burst-robust counts (at most --cap per --chunk-kb chunk, at least
          --min-chunks chunks). Each token brings its o200k merge parents first.

slot_search.py chooses how many tokens to swap (K) and writes `tokens`;
train_bpe.py removes the first K `prune` tokens and puts the K `tokens` at their
ids. Every other base token keeps its v1 id. FLORES is never read.

Usage
-----
    python prune_o200k.py --heldout /path/to/heldout --corpus /path/to/corpus --out o200k_v2.json
"""
import argparse
import hashlib
import heapq
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_corpus import LANGS  # noqa: E402
from train_bpe import INDIC_RANGES, build_base, o200k_split, scripts_of, to_unicode, tokenizer_json  # noqa: E402

V1_BASE = 114332          # tokens in the v1 base (ids 0-114332, id 36 is a special)
BRAHMIC = set(INDIC_RANGES) - {"Arabic", "Sinhala"}


def pieces(token, base, ranks):
    """Tokens `token` splits into using only base tokens (lowest-rank merge first, as tiktoken)."""
    parts = [bytes([x]) for x in token]
    while len(parts) > 1:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and parts[i] + parts[i + 1] in base and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
    return len(parts)


def usage_counts(enc, path, chunk_kb, cap):
    """Capped o200k token counts (by rank), chunk counts, and the file size in MB."""
    text = Path(path).read_text(encoding="utf-8")
    counts, chunks = Counter(), Counter()
    step = chunk_kb * 1000
    for i in range(0, len(text), step):
        c = Counter(enc.encode_ordinary(text[i:i + step]))
        counts.update({r: min(n, cap) for r, n in c.items()})
        chunks.update(c.keys())
    return counts, chunks, len(text.encode("utf-8")) / 1e6


def base_usage(base, merges, pattern, paths):
    """How often each v1 base token is used when the files are encoded with the base alone."""
    from tokenizers import Tokenizer
    j = tokenizer_json({to_unicode(b): i for b, i in base.items()},
                       [(to_unicode(a), to_unicode(b)) for a, b in merges], [], pattern)
    tok = Tokenizer.from_str(json.dumps(j))
    counts = Counter()
    for p in paths:
        lines = Path(p).read_text(encoding="utf-8").split("\n")
        chunks = ["\n".join(lines[i:i + 2000]) for i in range(0, len(lines), 2000)]
        for e in tok.encode_batch(chunks, add_special_tokens=False):
            counts.update(e.ids)
    return counts


def prune_order(base, merges, used, ranks):
    """Unused base tokens in an order where every prefix leaves the base closed under its merges."""
    children = Counter()
    for a, b in merges:
        children[a] += 1
        children[b] += 1
    parents = {a + b: (a, b) for a, b in merges}

    def group(t):
        s = scripts_of(t)
        return 2 if "Latin" in s else (1 if not s else 0)

    def eligible(t):
        return len(t) > 1 and used[base[t]] == 0 and not scripts_of(t) & BRAHMIC and children[t] == 0

    heap = [(group(t), -ranks[t], t) for t in base if eligible(t)]
    heapq.heapify(heap)
    order, done = [], set()
    while heap:
        _, _, t = heapq.heappop(heap)
        if t in done:
            continue
        done.add(t)
        order.append(t)
        for p in parents.get(t, ()):
            children[p] -= 1
            if p not in done and eligible(p):
                heapq.heappush(heap, (group(p), -ranks[p], p))
    return order


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--heldout", required=True, help="fetch_heldout.py output (uses usage/En.txt, usage/Code.txt)")
    ap.add_argument("--corpus", required=True, help="Indic training corpus (fetch_corpus.py), as for v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max", type=int, default=16384, help="length of each ranked list")
    ap.add_argument("--chunk-kb", type=int, default=20)
    ap.add_argument("--cap", type=int, default=4, help="max count of a token per chunk")
    ap.add_argument("--min-chunks", type=int, default=3)
    args = ap.parse_args()

    import tiktoken
    t0 = time.time()
    enc = tiktoken.get_encoding("o200k_base")
    ranks = enc._mergeable_ranks
    by_rank = {r: t for t, r in ranks.items()}
    base, base_merges, _ = build_base(V1_BASE)
    cut = max(ranks[t] for t in base)

    # 1. Prune list.
    usage_files = [Path(args.heldout) / "usage" / "En.txt", Path(args.heldout) / "usage" / "Code.txt"]
    indic_files = [Path(args.corpus) / f"{lang}.txt" for lang in LANGS]
    used = base_usage(base, base_merges, enc._pat_str, usage_files + indic_files)
    prune = prune_order(base, base_merges, used, ranks)
    by_script = Counter(next(iter(scripts_of(t)), "Common") for t in prune)
    print(f"prune: {len(prune)} unused base tokens {dict(by_script.most_common())} ({time.time() - t0:.0f}s)", flush=True)

    # 2. Extension rankings.
    def rank(gain):
        chosen, have = [], set(base)
        def add(t):
            if t in have:
                return
            for part in o200k_split(ranks, t, ranks[t]):
                add(part)
            have.add(t)
            chosen.append(t)
        for t in sorted(gain, key=lambda t: (-gain[t], ranks[t])):
            mark = len(chosen)
            add(t)
            if len(chosen) > args.max:          # ancestors overflowed the budget: undo this candidate
                del chosen[mark:]
                break
        return chosen

    usage, ranked = {}, {}
    for name, path in zip(("En", "Code"), usage_files):
        counts, chunks, mb = usage_counts(enc, path, args.chunk_kb, args.cap)
        gain = {by_rank[r]: c * (pieces(by_rank[r], base, ranks) - 1) for r, c in counts.items()
                if r > cut and len(by_rank[r]) <= 32 and len(scripts_of(by_rank[r])) <= 1
                and chunks[r] >= args.min_chunks}
        ranked[name] = rank(gain)
        usage[name] = {"mb": round(mb, 3), "capped_tokens": sum(counts.values()), "candidates": len(gain)}
        print(f"{name}: {len(gain)} candidates; first {[t.decode('utf-8', 'replace') for t in ranked[name][:12]]}",
              flush=True)

    out = {
        "_comment": "v2 (#163). `prune`: unused v1 base tokens, in removal order. `ranked`: o200k tokens past the "
                    "v1 base cut, best first, for English and for code. `tokens`: the extension slot_search.py chose; "
                    "train_bpe.py removes the first len(tokens) `prune` tokens and puts `tokens` at their ids. "
                    "Made by prune_o200k.py and slot_search.py.",
        "base_tokens": V1_BASE, "base_cut_rank": cut,
        "usage_sources": [str(p.relative_to(p.parent.parent)) for p in usage_files] + ["Indic training corpus (11)"],
        "counting": {"chunk_kb": args.chunk_kb, "cap": args.cap, "min_chunks": args.min_chunks},
        "usage": usage,
        "prune_by_script": dict(by_script.most_common()),
        "prune_sha256": hashlib.sha256("\n".join(t.hex() for t in prune).encode()).hexdigest(),
        "prune": [t.hex() for t in prune],
        "ranked": {name: [t.hex() for t in ts] for name, ts in ranked.items()},
    }
    Path(args.out).write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")
    print(f"cut rank {cut}; wrote {args.out} ({time.time() - t0:.0f}s)")


def combine(ranked, n_code, n_en):
    """The first n_code code tokens, then the first n_en English tokens (hex), without repeats.
    Both prefixes are closed under merges, so the result is too."""
    out, seen = [], set()
    for h in ranked["Code"][:n_code] + ranked["En"][:n_en]:
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def with_tokens(data, n_code, n_en):
    """`data` (an o200k_v2 json) with the chosen combination as `tokens` and its sha256."""
    tokens = combine(data["ranked"], n_code, n_en)
    if len(tokens) > len(data["prune"]):
        sys.exit(f"ERROR: {len(tokens)} extension tokens but only {len(data['prune'])} prunable base tokens.")
    return {**data, "chosen": {"Code": n_code, "En": n_en}, "tokens": tokens,
            "tokens_sha256": hashlib.sha256("\n".join(tokens).encode()).hexdigest()}


if __name__ == "__main__":
    main()
