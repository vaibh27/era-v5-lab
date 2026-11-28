#!/usr/bin/env python3
"""Rewrite the frozen tokenizer's vocabulary so llama.cpp (Ollama), which can only apply NFC, gives the training ids
for Gurmukhi text (Bengali measured, not fixable this way: see SCRIPTS).

    python remap_vocab.py --tokenizer tokenizer.json --out tokenizer_gguf.json

Why. p1536's normalizer is NFC followed by 15 replacements. Nine of them (Bengali ড় ঢ় য়, Gurmukhi ਲ਼ ਸ਼ ਖ਼ ਗ਼ ਜ਼ ਫ਼)
recompose letters that NFC splits into letter + nukta (Unicode composition exclusions). A GGUF carries only the NFC
flag, so in llama.cpp these letters stay split and tokenize to other ids than in training.

What this does instead: the token **strings** move to the split form, so the text llama.cpp sees after NFC matches the
vocabulary. For each of the nine letters, in byte-level (GPT-2) form:
  1. every vocab token containing the composed letter is renamed to the same string with the split form;
  2. every merge is rewritten the same way; the merge that built the composed letter from its bytes is replaced by
     one that builds it from the two split characters (letter, nukta), placed right after the merges that make those;
  3. tokens that already contained the split form (they can never come out of the training tokenizer, whose
     normalizer removes it) are renamed to an unreachable string (`<|unused_split_<id>|>`) and their merges dropped,
     so no two ids share a string.
Token **ids never change**, so the model's embedding and head rows stay where they are. The six Malayalam chillu
replacements are left alone: renaming those tokens would break users who type the atomic chillu.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

SCRIPTS = {  # composed: split (letter + nukta); composition exclusions that p1536 recomposes after NFC
    "gurmukhi": {"\u0a33": "\u0a32\u0a3c", "\u0a36": "\u0a38\u0a3c", "\u0a59": "\u0a16\u0a3c",
                 "\u0a5a": "\u0a17\u0a3c", "\u0a5b": "\u0a1c\u0a3c", "\u0a5e": "\u0a2b\u0a3c"},
    # Bengali is measured, not shipped: byte-level merges such as ('\u00e0\u00a6\u00bf', '\u00e0\u00a6') join a
    # vowel sign with the lead bytes of the next letter before YA is complete, so a split YA+NUKTA never reaches the
    # new (YA, NUKTA) merge, and reordering merges would change the training ids of every plain YA. Wikipedia test
    # (932 lines): 77.7% -> 72.2% of lines still differ, with more tokens than before. Gurmukhi: 57.1% -> 10.7%.
    "bengali": {"\u09dc": "\u09a1\u09bc", "\u09dd": "\u09a2\u09bc", "\u09df": "\u09af\u09bc"},
}


def byte_map() -> dict[int, str]:
    bs = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    cs, n = bs[:], 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def remap(tok: dict, scripts=("gurmukhi",)) -> tuple[dict, dict]:
    PAIRS = {c: d for name in scripts for c, d in SCRIPTS[name].items()}
    b2u = byte_map()
    enc = lambda s: "".join(b2u[b] for b in s.encode("utf-8"))
    model = tok["model"]
    vocab: dict[str, int] = model["vocab"]
    merges = [m if isinstance(m, list) else m.split(" ") for m in model["merges"]]
    comp = {enc(c): enc(s) for c, s in PAIRS.items()}             # byte-level composed -> split
    split_parts = {enc(c): (enc(s[0]), enc(s[1])) for c, s in PAIRS.items()}

    def sub(s: str) -> str:
        for c, s2 in comp.items():
            s = s.replace(c, s2)
        return s

    # 3. tokens that already hold a split form: unreachable in training, move them out of the way
    retired = {v: i for v, i in vocab.items() if any(s2 in v for s2 in comp.values())}
    new_vocab = {}
    for v, i in vocab.items():
        new_vocab[f"<|unused_split_{i}|>" if v in retired else sub(v)] = i
    if len(new_vocab) != len(vocab):
        raise SystemExit("remap would merge two tokens into one string")

    # 2. merges
    out, dropped, rebuilt, lost, rank_of = [], 0, 0, [], {}
    for a, b in merges:
        if a + b in retired or a in retired or b in retired:
            dropped += 1
            continue
        if a + b in comp:                                         # the byte merge that built a composed letter
            continue                                              # replaced below by (letter, nukta)
        sa, sb, s = sub(a), sub(b), sub(a + b)
        if sa + sb == s:
            out.append([sa, sb])
            continue
        # a composed letter straddles the two halves (e.g. ('Ġà§', 'Ł')): build the renamed token from two
        # split-form pieces that are both tokens, preferring the cut just before or after a nukta
        cuts = [k for k in range(1, len(s)) if s[:k] in new_vocab and s[k:] in new_vocab]
        nuktas = {p[1] for p in split_parts.values()}
        cuts.sort(key=lambda k: (not (s[k:] in nuktas or s[:k][-len(next(iter(nuktas))):] in nuktas), -k))
        if cuts:
            out.append([s[:cuts[0]], s[cuts[0]:]]); rebuilt += 1
        else:
            lost.append(vocab[a + b]); dropped += 1
    for i, (a, b) in enumerate(out):
        rank_of[a + b] = i
    for letter, nukta in split_parts.values():
        for part in (letter, nukta):
            if part not in new_vocab:
                raise SystemExit(f"split part {part!r} is not a token")
        # right after the merges that produce the two parts (single characters are built from bytes early)
        pos = max(rank_of.get(letter, -1), rank_of.get(nukta, -1)) + 1
        out.insert(pos, [letter, nukta])
        rank_of = {a + b: i for i, (a, b) in enumerate(out)}
    model["vocab"], model["merges"] = new_vocab, out
    renamed = sum(1 for v in vocab if v not in retired and sub(v) != v)
    bad = [m for m in out if m[0] not in new_vocab or m[1] not in new_vocab or m[0] + m[1] not in new_vocab]
    if bad:
        raise SystemExit(f"{len(bad)} merges reference strings that are not tokens, e.g. {bad[:3]}")
    return tok, {"renamed": renamed, "retired": sorted(retired.values()), "merges_dropped": dropped,
                 "merges_rebuilt": rebuilt, "unbuildable_ids": lost, "merges": len(out), "pairs": len(PAIRS)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--scripts", nargs="+", choices=sorted(SCRIPTS), default=["gurmukhi"],
                    help="default gurmukhi; bengali only for measurement (see SCRIPTS)")
    a = ap.parse_args()
    tok, report = remap(json.loads(a.tokenizer.read_text(encoding="utf-8")), tuple(a.scripts))
    report["scripts"] = a.scripts
    a.out.write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
    a.out.with_suffix(".report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (v if k != "retired" else len(v)) for k, v in report.items()}))


if __name__ == "__main__":
    main()
