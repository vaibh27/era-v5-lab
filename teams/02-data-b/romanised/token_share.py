#!/usr/bin/env python3
"""token_share.py - romanised Indic share in tokens, per language, for the cap in romanised_policy_v1.md.

TSAI's rules (#213, part 2): the cap is measured in tokens, and a document counts as romanised when most of its words
are romanised (doc_detector.decide, word-list classifier, chosen in eval_v2.md).
Per language it reports
  cap share      tokens of the romanised documents / all tokens          (what the cap limits)
  segment share  tokens of romanised segments in any document / all tokens (for scale only: romanised headings and
                 comments inside native documents, which the cap does not touch)

The tokenizer is a stand-in until the project tokenizer is final: BrahmicTokenizer-131K, the baseline team
03-tokenizer measures its candidates against (github.com/theschoolofai/BrahmicTokenizer-131K, commit 7dfcf398,
tokenizer.json, sha256 01d968a7...). Pass --tokenizer to use another tokenizer.json.

Before counting, the script checks its inputs and stops on a mismatch: every pack against packs_SHA256SUMS (the 22
ind-* sample packs, which must all be present), and the tokenizer against --tokenizer-sha256 when it is given.

Usage
  python token_share.py --packs-dir packs --tokenizer tokenizer/tokenizer.json --out romanised_token_share_v1.csv
  python token_share.py --tokenizer <frozen tokenizer.json> --tokenizer-sha256 <its sha256> \
                        --out romanised_token_share_v2.csv
"""
import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq
from tokenizers import Tokenizer

import doc_detector

HERE = Path(__file__).resolve().parent


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_inputs(packs_dir, sums, tokenizer, tokenizer_sha):
    """Stop unless every pack in sums is in packs_dir with its sha256, no other pack is there, and the tokenizer
    matches tokenizer_sha (when given). Returns the tokenizer's sha256."""
    want = dict(reversed(line.split()) for line in sums.read_text().splitlines() if line.strip())
    have = {p.name for p in packs_dir.glob("ind-*.parquet")}
    errors = [f"missing pack {n}" for n in sorted(set(want) - have)]
    errors += [f"pack not in {sums.name}: {n}" for n in sorted(have - set(want))]
    errors += [f"sha256 mismatch: {n}" for n in sorted(set(want) & have) if sha256(packs_dir / n) != want[n]]
    tok_sha = sha256(tokenizer)
    if tokenizer_sha and tok_sha != tokenizer_sha:
        errors.append(f"tokenizer sha256 is {tok_sha}, expected {tokenizer_sha}")
    if errors:
        raise SystemExit("input check failed:\n  " + "\n  ".join(errors))
    print(f"inputs ok: {len(want)} packs match {sums.name}; tokenizer {tokenizer} sha256 {tok_sha}")
    return tok_sha


def main(argv=None):
    ap = argparse.ArgumentParser(description="Romanised Indic share in tokens, per language.")
    ap.add_argument("--packs-dir", default="packs", type=Path)
    ap.add_argument("--packs-sums", default=HERE / "packs_SHA256SUMS", type=Path)
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.json", type=Path)
    ap.add_argument("--tokenizer-sha256", default=None, help="expected sha256 of the tokenizer.json (checked)")
    ap.add_argument("--out", default="romanised_token_share_v1.csv", type=Path)
    a = ap.parse_args(argv)

    check_inputs(a.packs_dir, a.packs_sums, a.tokenizer, a.tokenizer_sha256)
    tok = Tokenizer.from_file(str(a.tokenizer))
    n = lambda texts: sum(len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)) if texts else 0
    agg = defaultdict(lambda: defaultdict(int))
    top = []
    for path in sorted(a.packs_dir.glob("ind-*.parquet")):
        lang = path.stem.split("-")[1]
        docs = pq.read_table(path, columns=["doc_id", "text"]).to_pylist()
        texts = [d["text"] or "" for d in docs]
        doc_tokens = [len(e.ids) for e in tok.encode_batch(texts, add_special_tokens=False)]
        s = agg[lang]
        s["packs"] += 1
        for d, text, t in zip(docs, texts, doc_tokens):
            segs = doc_detector.segments(text)
            flagged = [g for g in segs if doc_detector.wordlist_segment(g) and doc_detector.latin_words(g)]
            dec = doc_detector.decide(text)
            s["docs"] += 1
            s["tokens"] += t
            s["seg_tokens"] += n(flagged)
            s["contains_docs"] += bool(flagged)
            if dec["romanised"]:
                s["rom_docs"] += 1
                s["rom_tokens"] += t
            top.append((dec["share"], path.stem, d["doc_id"]))

    fields = ["lang", "packs", "docs", "tokens", "romanised_docs", "romanised_doc_tokens", "cap_share",
              "docs_with_romanised_segments", "romanised_segment_tokens", "segment_share"]
    rows = []
    for lang in sorted(agg, key=lambda l: -agg[l]["seg_tokens"] / agg[l]["tokens"]) + ["all"]:
        s = agg[lang] if lang != "all" else {k: sum(agg[l][k] for l in agg) for k in
                                             ("packs", "docs", "tokens", "rom_docs", "rom_tokens", "contains_docs",
                                              "seg_tokens")}
        rows.append({"lang": lang, "packs": s["packs"], "docs": s["docs"], "tokens": s["tokens"],
                     "romanised_docs": s["rom_docs"], "romanised_doc_tokens": s["rom_tokens"],
                     "cap_share": f"{s['rom_tokens'] / s['tokens']:.4%}",
                     "docs_with_romanised_segments": s["contains_docs"], "romanised_segment_tokens": s["seg_tokens"],
                     "segment_share": f"{s['seg_tokens'] / s['tokens']:.4%}"})
    with open(a.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"tokenizer: {a.tokenizer}")
    print(" | ".join(fields))
    for r in rows:
        print(" | ".join(str(r[k]) for k in fields))
    print("highest romanised word share in any document:",
          ", ".join(f"{sh:.0%} {p} {d}" for sh, p, d in sorted(top, reverse=True)[:5]))


if __name__ == "__main__":
    main()
