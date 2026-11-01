#!/usr/bin/env python3
"""
train_bpe.py
============

Builds a 131,072-id byte-level BPE tokenizer with a fixed id slot per Brahmic
script, in the BrahmicTokenizer-131K format (HF tokenizer.json, o200k_base
pre-tokenizer, GPT-2 ByteLevel).

Id layout (from slots.json)
---------------------------
    [0, base)                    o200k_base tokens in rank order (English, code, ...)
                                 v2 (#163): unused ones swapped in place for o200k tokens
                                 past the cut (o200k_v2.json); all other ids unchanged
    [base, base + slots_1)       new merges for script 1 (e.g. Devanagari)
    ...                          one contiguous block per script, in slots.json order
    [vocab - special_slots, vocab)  special tokens: named ones, then <|reserved_N|>

With --special-layout v0 the special ids are tokenizer v0's instead (#107):
all 356 special entries are copied verbatim from the v0 (or v0.1, #132) tokenizer.json, at
their exact ids (including the 9 outside the top block, e.g. 36 and 128512).
Normal tokens then fill the remaining 130,716 ids in order, skipping those
ids, so a script block may contain a special id; the manifest lists them.

Method
------
1. Base. Walk o200k_base in rank order. Keep a token only if it is <= 32 bytes,
   spans at most one script, and both halves of its merge were kept. Stop at
   `base` tokens. The kept set is closed under its merges, so it is a valid BPE.
   With "o200k_swap" in slots.json (step 1b, v2): the unused base tokens listed
   in o200k_v2.json are removed and o200k tokens past the cut take their ids,
   each merged from its own o200k halves, in o200k rank order. Every other base
   id stays the same; "base" in slots.json pins the base size.
2. Per script. Pre-tokenize the script's corpus with the o200k regex, keep only
   pre-tokens whose non-Common characters are all in that script, encode them
   with the base, and continue BPE training on those words for exactly `slots`
   merges. A merge is allowed only if the result is <= 32 bytes, spans one
   script, and is not already in the vocab.
3. Specials fill the top `special_slots` ids, or v0's ids (--special-layout v0).

Steps 1 and 2 enforce the four verify-script properties by construction:
no cross-script merges, no normal token > 32 bytes. With --brahmic-dir the
script runs the four verify scripts on the output as a final gate. The
structural-identity script compares the output with --compare-to (default:
the BrahmicTokenizer-131K tokenizer.json), not with itself; it reports the
vocab / merge / special deltas and does not fail on a difference.

Usage
-----
    python train_bpe.py --corpus /path/to/corpus --out /path/to/out \\
        --brahmic-dir /path/to/BrahmicTokenizer-131K

    # Same merges, but tokenizer v0's special ids (build v0 with v0/build_v0.py)
    python train_bpe.py --corpus /path/to/corpus --out /path/to/out \\
        --special-layout v0 --v0-tokenizer /path/to/v0_tokenizer.json \\
        --brahmic-dir /path/to/BrahmicTokenizer-131K

    # v1 retrain (#128): v0 ids, #134 normalizer applied before counting words
    python train_bpe.py --corpus /path/to/corpus --out /path/to/out --slots slots_v1.json \\
        --special-layout v0 --v0-tokenizer /path/to/v0_tokenizer.json --normalizer v1 \\
        --brahmic-dir /path/to/BrahmicTokenizer-131K

    # v2 (#163): o200k pruning by usage (unused base ids swapped in place), slots chosen on held-out text
    python train_bpe.py --corpus /path/to/corpus --out /path/to/out --slots slots_v2.json \\
        --special-layout v0 --v0-tokenizer /path/to/v0_tokenizer.json --normalizer v1 \\
        --brahmic-dir /path/to/BrahmicTokenizer-131K
    python check_ids.py /path/to/v1/tokenizer.json /path/to/out/tokenizer.json

    # Week-2 candidates (C1 -> C2 -> C3 each add one change; v0.1 specials from v0/build_v0_1.py)
    python train_bpe.py ... --slots slots_c1.json --special-layout v0 --v0-tokenizer v0_1_tokenizer.json   # C1: o200k swap only
    python train_bpe.py ... --slots slots_v2.json --special-layout v0 --v0-tokenizer v0_1_tokenizer.json   # C2: + slots_v1
    python train_bpe.py ... --slots slots_v2.json --special-layout v0 --v0-tokenizer v0_1_tokenizer.json \\
        --normalizer v1                                                                                    # C3: + normalizer (v2)

Writes <out>/tokenizer.json and <out>/manifest.json (sha256, slot table,
corpus hashes, verify results). Exit 1 if a verify script fails.
"""
import argparse
import hashlib
import heapq
import json
import subprocess
import sys
import time
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

MAX_BYTES = 32

# Script ranges and name prefixes: the union of the classifiers in
# verify_no_cross_script_merges.py and verify_kronecker_constraints_unified.py,
# so a token we accept is accepted by both.
INDIC_RANGES = {
    "Devanagari": (0x0900, 0x097F), "Bengali": (0x0980, 0x09FF), "Gurmukhi": (0x0A00, 0x0A7F),
    "Gujarati": (0x0A80, 0x0AFF), "Oriya": (0x0B00, 0x0B7F), "Tamil": (0x0B80, 0x0BFF),
    "Telugu": (0x0C00, 0x0C7F), "Kannada": (0x0C80, 0x0CFF), "Malayalam": (0x0D00, 0x0D7F),
    "Sinhala": (0x0D80, 0x0DFF), "Arabic": (0x0600, 0x06FF),
}
NAME_PREFIXES = [("LATIN", "Latin"), ("CJK", "CJK"), ("HIRAGANA", "CJK"), ("KATAKANA", "CJK"),
                 ("HANGUL", "CJK"), ("CYRILLIC", "Cyrillic"), ("THAI", "Thai"), ("ARABIC", "Arabic"),
                 ("SINHALA", "Sinhala"), ("HEBREW", "Hebrew"), ("GREEK", "Greek")]


def script_of_char(ch):
    cp = ord(ch)
    for s, (lo, hi) in INDIC_RANGES.items():
        if lo <= cp <= hi:
            return s
    name = unicodedata.name(ch, "")
    for prefix, s in NAME_PREFIXES:
        if name.startswith(prefix):
            return s
    return None


def scripts_of(b):
    """Non-Common scripts in a byte string, decoded the way the verify scripts decode it."""
    return {s for ch in b.decode("utf-8", errors="replace") if (s := script_of_char(ch)) is not None}


def gpt2_byte_map():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


B2U = gpt2_byte_map()
U2B = {u: b for b, u in B2U.items()}


def to_unicode(b):
    return "".join(B2U[x] for x in b)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def o200k_split(ranks, token, rank):
    """The two parts o200k_base merges into `token`: BPE on its bytes using ranks < rank."""
    parts = [bytes([x]) for x in token]
    while len(parts) > 2:
        best = None
        for i in range(len(parts) - 1):
            r = ranks.get(parts[i] + parts[i + 1])
            if r is not None and r < rank and (best is None or r < best[0]):
                best = (r, i)
        if best is None:
            break
        i = best[1]
        parts = parts[:i] + [parts[i] + parts[i + 1]] + parts[i + 2:]
    return parts


def build_base(size):
    """Step 1: the first `size` admissible o200k_base tokens and their merges."""
    import tiktoken
    ranks = tiktoken.get_encoding("o200k_base")._mergeable_ranks
    vocab, merges = {}, []          # bytes -> id, [(left bytes, right bytes)]
    dropped = Counter()
    for token, rank in sorted(ranks.items(), key=lambda kv: kv[1]):
        if len(vocab) == size:
            break
        if len(token) == 1:
            vocab[token] = len(vocab)
            continue
        if len(token) > MAX_BYTES:
            dropped["over_32_bytes"] += 1
            continue
        if len(scripts_of(token)) > 1:
            dropped["cross_script"] += 1
            continue
        parts = o200k_split(ranks, token, rank)
        if len(parts) != 2 or parts[0] not in vocab or parts[1] not in vocab:
            dropped["parent_dropped"] += 1
            continue
        vocab[token] = len(vocab)
        merges.append((parts[0], parts[1]))
    if len(vocab) != size or sum(len(t) == 1 for t in vocab) != 256:
        sys.exit(f"ERROR: base has {len(vocab)} tokens, wanted {size} including all 256 bytes.")
    return vocab, merges, dict(dropped)


def load_swap(path):
    """o200k_v2.json (prune_o200k.py + slot_search.py): the extension tokens and the prune list, as bytes."""
    data = json.loads(Path(path).read_text())
    for key in ("tokens", "prune"):
        if hashlib.sha256("\n".join(data[key]).encode()).hexdigest() != data[f"{key}_sha256"]:
            sys.exit(f"ERROR: {path}: {key} do not match {key}_sha256.")
    return [bytes.fromhex(h) for h in data["tokens"]], [bytes.fromhex(h) for h in data["prune"]], data


def choose_prune(vocab, prune, tokens):
    """The first len(tokens) entries of `prune`, skipping base ancestors of the extension tokens
    (an extension token is merged from its o200k halves, so those must stay)."""
    import tiktoken
    ranks = tiktoken.get_encoding("o200k_base")._mergeable_ranks
    keep = set()
    def protect(t):
        if t in keep or len(t) == 1:
            return
        if t in vocab:
            keep.add(t)
        for part in o200k_split(ranks, t, ranks[t]):
            protect(part)
    for t in tokens:
        protect(t)
    out = [t for t in prune if t not in keep][:len(tokens)]
    if len(out) < len(tokens):
        sys.exit(f"ERROR: only {len(out)} prunable base tokens for {len(tokens)} extension tokens.")
    return out


def swap_base(vocab, merges, prune, tokens):
    """Step 1b (v2, #163): o200k pruning by usage. Remove the `prune` tokens from the base and give
    their ids to `tokens` (o200k tokens past the base cut), in o200k rank order. Every other base
    token keeps its id. Each new token is merged from its own o200k halves, so merge priority stays
    o200k's; its merges go after the base merges. Returns the new merges for the extension."""
    import tiktoken
    ranks = tiktoken.get_encoding("o200k_base")._mergeable_ranks
    gone = set(prune)
    if len(gone) != len(prune) or any(t not in vocab or len(t) == 1 for t in gone):
        sys.exit("ERROR: prune list has repeats, single bytes or tokens outside the base.")
    if any(a in gone or b in gone for a, b in merges if a + b not in gone):
        sys.exit("ERROR: a kept base token is merged from a pruned one; the base would not be closed.")
    free = sorted(vocab.pop(t) for t in prune)
    merges[:] = [(a, b) for a, b in merges if a + b not in gone]
    new = []
    for token, i in zip(sorted(tokens, key=ranks.__getitem__), free):
        if token in vocab or len(token) > MAX_BYTES or len(scripts_of(token)) > 1:
            sys.exit(f"ERROR: o200k extension token {token!r} is a duplicate or not admissible.")
        parts = o200k_split(ranks, token, ranks[token])
        if len(parts) != 2 or parts[0] not in vocab or parts[1] not in vocab:
            sys.exit(f"ERROR: o200k extension token {token!r}: merge parents {parts} missing.")
        vocab[token] = i
        new.append((parts[0], parts[1]))
    return new


def tokenizer_json(vocab, merges, specials, pattern, normalizer=None):
    """vocab: unicode-string -> id; merges: [(a, b)] unicode strings;
    specials: [(id, content)] or full added_tokens entries (dicts, copied as-is)."""
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [s if isinstance(s, dict) else
                         {"id": s[0], "content": s[1], "single_word": False, "lstrip": False, "rstrip": False,
                          "normalized": False, "special": True} for s in specials],
        "normalizer": normalizer,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": pattern}, "behavior": "Isolated", "invert": False},
            {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}]},
        "post_processor": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True},
        "decoder": {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True},
        "model": {"type": "BPE", "dropout": None, "unk_token": None, "continuing_subword_prefix": None,
                  "end_of_word_suffix": None, "fuse_unk": False, "byte_fallback": False, "ignore_merges": True,
                  "vocab": vocab, "merges": [[a, b] for a, b in merges]},
    }


def load_normalizer(name):
    """The v1 normalizer from #134 (NFC + atomic chillu + Bn/Pa nukta re-join), or None."""
    if name == "none":
        return None
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "v1"))
    from normalizer_experiment_v2 import VARIANTS
    return VARIANTS["nfc+chillu+nukta-BnPa"]


def normalizer_json(norm):
    from tokenizers import Tokenizer, models
    t = Tokenizer(models.BPE())
    t.normalizer = norm
    return json.loads(t.to_str())["normalizer"]


def word_counts(paths, script, pattern, norm=None):
    """Pre-token byte strings from the corpus whose non-Common characters are all in `script`."""
    from tokenizers import Regex, pre_tokenizers
    split = pre_tokenizers.Split(Regex(pattern), behavior="isolated")
    counts = Counter()
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                if norm is not None:
                    line = norm.normalize_str(line)  # count words as the tokenizer will see them
                counts.update(piece for piece, _ in split.pre_tokenize_str(line))
    keep = Counter()
    for piece, c in counts.items():
        if {script_of_char(ch) for ch in piece} - {None} == {script}:
            keep[piece.encode("utf-8")] = c
    return keep


def learn_merges(words, freqs, id_bytes, vocab_bytes, script, n_merges, first_id):
    """Step 2: continue BPE on `words` (lists of ids) for exactly n_merges new tokens."""
    pair_count = defaultdict(int)
    pair_words = defaultdict(set)
    for wi, w in enumerate(words):
        for a, b in zip(w, w[1:]):
            pair_count[(a, b)] += freqs[wi]
            pair_words[(a, b)].add(wi)
    heap = [(-c, a, b) for (a, b), c in pair_count.items()]
    heapq.heapify(heap)
    banned = set()
    merges = []
    next_id = first_id
    while len(merges) < n_merges:
        if not heap:
            sys.exit(f"ERROR: {script} ran out of pairs after {len(merges)} merges; "
                     f"lower its slots or add corpus.")
        negc, a, b = heapq.heappop(heap)
        if -negc != pair_count.get((a, b), 0) or (a, b) in banned or -negc <= 0:
            continue
        new = id_bytes[a] + id_bytes[b]
        if len(new) > MAX_BYTES or new in vocab_bytes or scripts_of(new) - {script}:
            banned.add((a, b))
            continue
        nid = next_id
        next_id += 1
        id_bytes[nid] = new
        vocab_bytes.add(new)
        merges.append((a, b, nid, -negc))           # count = corpus frequency when merged
        touched = set()
        for wi in pair_words.pop((a, b)):
            w = words[wi]
            if len(w) < 2:
                continue
            out, i, hit = [], 0, False
            while i < len(w):
                if i + 1 < len(w) and w[i] == a and w[i + 1] == b:
                    out.append(nid)
                    i += 2
                    hit = True
                else:
                    out.append(w[i])
                    i += 1
            if not hit:
                continue
            f = freqs[wi]
            for p in zip(w, w[1:]):
                pair_count[p] -= f
                touched.add(p)
            for p in zip(out, out[1:]):
                pair_count[p] += f
                pair_words[p].add(wi)
                touched.add(p)
            words[wi] = out
        pair_count.pop((a, b), None)
        for p in touched:
            c = pair_count.get(p, 0)
            if c > 0:
                heapq.heappush(heap, (-c, p[0], p[1]))
    return merges


def run_verify(brahmic_dir, tok_path, compare_to):
    """Run the four reviewer scripts; the structural diff is against `compare_to`, not the output itself."""
    cmds = [
        ["verify_no_cross_script_merges.py", tok_path],
        ["verify_max_byte_length.py", tok_path],
        ["verify_kronecker_constraints_unified.py", tok_path],
        ["verify_sarvam_m_structural_identity.py", tok_path, compare_to],
    ]
    results = {}
    for cmd in cmds:
        p = subprocess.run([sys.executable, str(Path(brahmic_dir) / cmd[0]), *map(str, cmd[1:])],
                           capture_output=True, text=True)
        out = p.stdout + p.stderr
        # verify_kronecker exits 0 even on a failed constraint, so read its RESULT line too.
        ok = p.returncode == 0 and "RESULT: FAIL" not in out and "FAIL:" not in out
        results[cmd[0]] = {"exit": p.returncode, "pass": ok,
                           "summary": [ln.strip() for ln in out.splitlines()
                                       if any(k in ln for k in ("PASS", "FAIL", "RESULT", " identical:"))]}
        label = "REPORTED" if ok and cmd[0].startswith("verify_sarvam") else ("PASS" if ok else "FAIL")
        print(f"  {cmd[0]:<42} exit={p.returncode}  {label}", flush=True)
        if not ok:
            print(out[-2000:], flush=True)
    return results


def load_v0_specials(path):
    """Tokenizer v0's or v0.1's added_tokens, after checking the file against v0/SHA256SUMS (#107, #132)."""
    sha = sha256_file(path)
    sums = Path(__file__).resolve().parent.parent / "v0" / "SHA256SUMS"
    want = {ln.split()[1].split("/")[0]: ln.split()[0] for ln in sums.read_text().splitlines()
            if ln.endswith(("v0/tokenizer.json", "v0.1/tokenizer.json"))} if sums.exists() else {}
    if not want:
        sys.exit(f"ERROR: no v0/tokenizer.json line in {sums}; cannot check the v0 file.")
    version = next((v for v, s in want.items() if s == sha), None)
    if version is None:
        sys.exit(f"ERROR: {path} sha256 {sha} is not tokenizer v0 or v0.1 ({want}). "
                 "Build it with v0/build_v0.py (then v0/build_v0_1.py).")
    t = json.loads(Path(path).read_text(encoding="utf-8"))
    return t["added_tokens"], sha, version


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--corpus", required=True, help="directory with <lang>.txt from fetch_corpus.py")
    ap.add_argument("--out", required=True, help="output directory (outside the repo; tokenizer.json > 5 MB)")
    ap.add_argument("--slots", default=str(Path(__file__).with_name("slots.json")))
    ap.add_argument("--brahmic-dir", help="clone of BrahmicTokenizer-131K; runs its four verify scripts on the output")
    ap.add_argument("--compare-to", help="tokenizer.json for the structural diff (default: <brahmic-dir>/tokenizer.json)")
    ap.add_argument("--normalizer", choices=["none", "v1"], default="none",
                    help="v1: #134 normalizer (v1/normalizer_experiment_v2.py, nfc+chillu+nukta-BnPa), "
                         "applied before counting words and written into tokenizer.json")
    ap.add_argument("--special-layout", choices=["own", "v0"], default="own",
                    help="own: specials in the top special_slots ids (slots.json); v0: tokenizer v0's exact special ids")
    ap.add_argument("--v0-tokenizer", help="tokenizer v0 tokenizer.json (from v0/build_v0.py), for --special-layout v0")
    args = ap.parse_args()

    import tiktoken
    t_start = time.time()
    cfg = json.loads(Path(args.slots).read_text())
    corpus = Path(args.corpus)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    pattern = tiktoken.get_encoding("o200k_base")._pat_str
    norm = load_normalizer(args.normalizer)

    vocab_size, n_special = cfg["vocab_size"], cfg["special_slots"]
    scripts = cfg["scripts"]
    if args.special_layout == "v0":
        if not args.v0_tokenizer:
            sys.exit("ERROR: --special-layout v0 needs --v0-tokenizer.")
        specials, v0_sha, v0_version = load_v0_specials(args.v0_tokenizer)
        n_special = len(specials)
    else:
        names = cfg["named_specials"] + [f"<|reserved_{i}|>" for i in range(n_special - len(cfg["named_specials"]))]
        specials = [(vocab_size - n_special + i, c) for i, c in enumerate(names)]
    special_ids = {s["id"] if isinstance(s, dict) else s[0] for s in specials}
    # The build works on dense ids 0..n_normal-1; real_id maps them past the special ids.
    real_id = [i for i in range(vocab_size) if i not in special_ids]
    base_size = len(real_id) - sum(s["slots"] for s in scripts.values())
    if "base" in cfg and cfg["base"] != base_size:
        sys.exit(f"ERROR: slots give base={base_size}, but {args.slots} fixes base={cfg['base']}.")
    print(f"layout: base={base_size}  scripts={sum(s['slots'] for s in scripts.values())}  "
          f"specials={n_special} ({args.special_layout})  total={vocab_size}", flush=True)

    t0 = time.time()
    base_vocab, base_merges, dropped = build_base(base_size)
    print(f"base: {len(base_vocab)} o200k tokens, {len(base_merges)} merges, dropped {dropped} "
          f"({time.time() - t0:.0f}s)", flush=True)
    swap = None
    if cfg.get("o200k_swap"):
        ext_tokens, prune_list, swap_data = load_swap(Path(args.slots).parent / cfg["o200k_swap"])
        pruned = choose_prune(base_vocab, prune_list, ext_tokens)
        swapped_ids = sorted(base_vocab[t] for t in pruned)
        base_merges += swap_base(base_vocab, base_merges, pruned, ext_tokens)
        swap = {"file": cfg["o200k_swap"], "tokens_sha256": swap_data["tokens_sha256"],
                "prune_sha256": swap_data["prune_sha256"], "swapped": len(pruned)}
        print(f"o200k swap: {len(pruned)} unused base tokens replaced in place", flush=True)

    # Encoder for the base alone, used to segment each script's words before step 2.
    from tokenizers import Tokenizer
    base_json = tokenizer_json({to_unicode(b): i for b, i in base_vocab.items()},
                               [(to_unicode(a), to_unicode(b)) for a, b in base_merges], [], pattern)
    base_json["pre_tokenizer"] = {"type": "ByteLevel", "add_prefix_space": False, "trim_offsets": True, "use_regex": False}
    base_tok = Tokenizer.from_str(json.dumps(base_json))

    id_bytes = {i: b for b, i in base_vocab.items()}
    vocab_bytes = set(base_vocab)
    all_merges = [(base_vocab[a], base_vocab[b]) for a, b in base_merges]
    def block(lo, hi):
        """Real id range of dense ids [lo, hi), plus any special ids that fall inside it."""
        b = [real_id[lo], real_id[hi - 1] + 1]
        inside = sorted(i for i in special_ids if b[0] <= i < b[1])
        return b + [{"special_ids_inside": inside}] if inside else b

    layout = {"base": block(0, base_size)}
    if swap:
        swap["swapped_ids"] = [real_id[i] for i in swapped_ids]
    last_count = {}
    corpus_manifest_path = corpus / "corpus_manifest.json"
    corpus_manifest = json.loads(corpus_manifest_path.read_text()) if corpus_manifest_path.exists() else {}
    next_id = base_size
    for script, spec in scripts.items():
        t0 = time.time()
        paths = [corpus / f"{lang}.txt" for lang in spec["langs"]]
        missing = [str(p) for p in paths if not p.exists()]
        if missing:
            sys.exit(f"ERROR: missing corpus files {missing}. Run fetch_corpus.py first.")
        counts = word_counts(paths, script, pattern, norm)
        items = sorted(counts.items())                      # fixed order -> deterministic ties
        encs = base_tok.encode_batch([b.decode("utf-8") for b, _ in items], add_special_tokens=False)
        words = [e.ids for e in encs]
        freqs = [c for _, c in items]
        merges = learn_merges(words, freqs, id_bytes, vocab_bytes, script, spec["slots"], next_id)
        all_merges.extend((a, b) for a, b, *_ in merges)
        last_count[script] = {"last_merge_count": merges[-1][3], "words": sum(freqs)}
        layout[script] = block(next_id, next_id + spec["slots"])
        print(f"{script}: {len(items)} unique words, {sum(freqs)} occurrences -> ids "
              f"[{layout[script][0]}, {layout[script][1]})  ({time.time() - t0:.0f}s)", flush=True)
        next_id += spec["slots"]

    assert next_id == len(real_id)
    if args.special_layout == "v0":
        top = vocab_size - sum(1 for i in special_ids if i >= real_id[-1])
        layout["specials"] = {"from": "tokenizer v0", "count": n_special,
                              "ids_outside_top_block": sorted(i for i in special_ids if i < top)}
    else:
        layout["specials"] = [next_id, next_id + n_special]

    vocab = {to_unicode(b): real_id[i] for i, b in id_bytes.items()}
    for s in specials:
        c, i = (s["content"], s["id"]) if isinstance(s, dict) else (s[1], s[0])
        if c in vocab:
            sys.exit(f"ERROR: special {c!r} is also a normal token string.")
        vocab[c] = i
    assert len(vocab) == vocab_size and sorted(vocab.values()) == list(range(vocab_size))
    merges_u = [(to_unicode(id_bytes[a]), to_unicode(id_bytes[b])) for a, b in all_merges]
    tok_path = out / "tokenizer.json"
    tok_json = tokenizer_json(vocab, merges_u, specials, pattern, normalizer_json(norm) if norm else None)
    tok_path.write_text(json.dumps(tok_json, ensure_ascii=False), encoding="utf-8")

    # Load back through the real library and round-trip a mixed sample.
    tok = Tokenizer.from_file(str(tok_path))
    sample = ["भारत एक देश है", "আমার সোনার বাংলা", "def f(x):\n    return x ** 2  # ok", "ଓଡ଼ିଆ ଭାଷା", "Hello, world!"]
    for lang in ("Hi", "Ta", "Or"):
        with open(corpus / f"{lang}.txt", encoding="utf-8") as f:
            sample.append(f.readline())
    for s in sample:
        if tok.decode(tok.encode(s).ids) != (norm.normalize_str(s) if norm else s):
            sys.exit(f"ERROR: round-trip failed on {s[:60]!r}")
    assert tok.get_vocab_size() == vocab_size

    sha = sha256_file(tok_path)
    manifest = {
        "tokenizer": "tokenizer.json",
        "sha256": sha,
        "vocab_size": vocab_size,
        "merges": len(merges_u),
        "layout": layout,
        "slots": {s: v["slots"] for s, v in scripts.items()},
        "special_layout": args.special_layout,
        **({"special_tokens": names[:len(cfg["named_specials"])],
            "reserved_special_slots": n_special - len(cfg["named_specials"])}
           if args.special_layout == "own" else {"v0_tokenizer": v0_version, "v0_tokenizer_sha256": v0_sha,
                                                          "special_tokens": n_special}),
        "base": {"source": "tiktoken o200k_base", "kept": base_size, "dropped": dropped},
        **({"o200k_swap": swap} if swap else {}),
        "corpus": {lang: corpus_manifest.get(lang) for spec in scripts.values() for lang in spec["langs"]},
        "normalizer": args.normalizer,
        "slots_json_sha256": sha256_file(args.slots),
        "slot_saturation": last_count,
        "build_seconds": round(time.time() - t_start),
    }
    print(f"wrote {tok_path}  sha256={sha}", flush=True)

    ok = True
    if args.brahmic_dir:
        print("verify scripts:", flush=True)
        compare_to = args.compare_to or str(Path(args.brahmic_dir) / "tokenizer.json")
        manifest["verify"] = run_verify(args.brahmic_dir, tok_path, compare_to)
        manifest["structural_diff_against"] = args.compare_to or "BrahmicTokenizer-131K/tokenizer.json"
        ok = all(v["pass"] for v in manifest["verify"].values())
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    print(f"wrote {out / 'manifest.json'}  ({manifest['build_seconds']}s)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
