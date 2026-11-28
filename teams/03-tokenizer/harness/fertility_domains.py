"""Tokens per word and bytes per token of the frozen p1536 tokenizer by Stage 1 pack, by domain and by Indic language,
with o200k_base (200,019 ids) as the outside tokenizer with a bigger vocabulary (G3, 28 Nov: Tasks C and D, design task 1).

    python teams/03-tokenizer/harness/fertility_domains.py --tokenizer release/tokenizer.json --sample <sample v0> \
        --heldout <fetch_heldout.py dir> --out teams/03-tokenizer/results/fertility_domains

Text (every file checked against its pinned sha256: the sample's SHA256SUMS, train/heldout_manifest.json):
  packs     the heldout/ part of each of the 38 sample-v0 packs, the Stage 1 source universe (TSAI, 22 Nov).
  Indic     heldout/dev/<lang>.txt for the 11 languages (FineWeb-2 test split, never in tokenizer training; counted per
            line), and Sanskrit from the sample's ind-sa-fw2 heldout part plus the first --scan docs of its train part.
            p1536 has no Sanskrit in its training text, so both are unseen.
  medicine, biology, chemistry: documents of sci-finepdfs, en-fineweb-edu, en-dclm and en-nemotroncc (heldout parts, then
            the first --scan docs of each train part; p1536 never trained on these packs) put in a group by the keyword
            rule in SUBJECT below. The rule is a heuristic, written down so anyone can rerun it.

Word rule (one rule for every language): a word is a maximal run of non-whitespace characters of the raw text
(Python str.split()). tokens/word = tokens of the whole document / its words; bytes/token = UTF-8 bytes / tokens.
Tokens are add_special_tokens=False, no bos or eos. Lossless: decode(encode(x)) == normalizer(x) for p1536 (its
normalizer is NFC plus the v1 fixes), decode(encode(x)) == x for o200k.

Most-split words: per group, the distinct words (punctuation stripped from the ends; letters and marks only, ZWJ and
ZWNJ allowed; 2 to 40 characters; seen at least twice) that give the most p1536 tokens when encoded with a leading
space, as they appear in running text. Shown with the pieces.

Encoder speed: tokens per second on one core (RAYON_NUM_THREADS=1, set before tokenizers loads) over all text here.
Writes <out>.json and <out>.md.
"""
import os

os.environ.setdefault("RAYON_NUM_THREADS", "1")           # one core, before tokenizers is imported
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

import pyarrow.parquet as pq  # noqa: E402

P1536 = "e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b"
TEAM = Path(__file__).resolve().parents[1]
INDIC = ("Hi", "Mr", "Bn", "As", "Ta", "Te", "Kn", "Ml", "Gu", "Pa", "Or")
SUBJECT_PACKS = ("sci-finepdfs", "en-fineweb-edu", "en-dclm", "en-nemotroncc")
SUBJECT = {
    "medicine": "patient patients clinical disease diseases treatment therapy diagnosis symptoms symptom drug drugs dose "
                "doses hospital surgery surgical physician medical medicine infection chronic syndrome cancer tumor "
                "pharmacological placebo prognosis dosage mortality",
    "biology": "cell cells protein proteins gene genes genetic species organism organisms dna rna enzyme enzymes "
               "evolution membrane tissue chromosome bacteria bacterial ecology mitochondria genome mutation "
               "metabolism photosynthesis biodiversity",
    "chemistry": "molecule molecules reaction reactions compound compounds acid acids chemical chemistry synthesis "
                 "bond bonds catalyst oxidation electron electrons solvent polymer ion ions molar solution "
                 "spectroscopy atom atoms isotope reagent",
}
SUBJECT = {k: set(v.split()) for k, v in SUBJECT.items()}
SUBJECT_RULE = ("a document joins the subject whose keywords (lower case, whole words, lists in SUBJECT) are densest, "
                "if that density is at least 8 per 1,000 words and at least 1.5 times the next subject's")
JOINERS = {"‌", "‍"}
EDGE = re.compile(r"^[\W_]+|[\W_]+$", re.UNICODE)


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def clean_word(w):
    w = EDGE.sub("", w)
    if not 2 <= len(w) <= 40:
        return None
    if all(unicodedata.category(c)[0] in "LM" or c in JOINERS for c in w):
        return w
    return None


class Meter:
    """Counts for one group under both tokenizers."""

    def __init__(self):
        self.docs = self.bytes = self.words = self.p = self.o = self.lossless_p = self.lossless_o = 0
        self.words_seen = Counter()

    def add(self, text, tok, enc, sec):
        if not text.strip():
            return
        t0 = time.perf_counter()
        e = tok.encode(text, add_special_tokens=False)
        sec["p1536"] += time.perf_counter() - t0
        t0 = time.perf_counter()
        o = enc.encode_ordinary(text)
        sec["o200k"] += time.perf_counter() - t0
        self.docs += 1
        self.bytes += len(text.encode("utf-8"))
        ws = text.split()
        self.words += len(ws)
        self.p += len(e.ids)
        self.o += len(o)
        self.lossless_p += tok.decode(e.ids, skip_special_tokens=False) == tok.normalizer.normalize_str(text)
        self.lossless_o += enc.decode(o) == text
        self.words_seen.update(filter(None, map(clean_word, ws)))

    def row(self):
        return {"docs": self.docs, "bytes": self.bytes, "words": self.words, "p1536_tokens": self.p,
                "o200k_tokens": self.o,
                "p1536_tokens_per_word": round(self.p / self.words, 3) if self.words else None,
                "o200k_tokens_per_word": round(self.o / self.words, 3) if self.words else None,
                "p1536_bytes_per_token": round(self.bytes / self.p, 3) if self.p else None,
                "o200k_bytes_per_token": round(self.bytes / self.o, 3) if self.o else None,
                "p1536_vs_o200k_tokens": round(self.p / self.o - 1, 4) if self.o else None,
                "lossless_p1536": f"{self.lossless_p}/{self.docs}", "lossless_o200k": f"{self.lossless_o}/{self.docs}"}

    def most_split(self, tok, n=20):
        cand = [(w, c) for w, c in self.words_seen.items() if c >= 2]
        scored = []
        for w, c in cand:
            e = tok.encode(" " + w, add_special_tokens=False)
            scored.append((len(e.ids), c, w, [(" " + w)[a:b] for a, b in e.offsets]))
        scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
        return [{"word": w, "pieces": k, "count": c, "split": "·".join(p.strip() or "␠" for p in parts)}
                for k, c, w, parts in scored[:n]]


def parquet_texts(path, limit=None):
    n = 0
    for batch in pq.ParquetFile(path).iter_batches(batch_size=512, columns=["text"]):
        for t in batch.column("text").to_pylist():
            yield t or ""
            n += 1
            if limit and n >= limit:
                return


def subject_of(text):
    ws = [w.lower().strip(".,;:()[]\"'") for w in text.split()]
    if len(ws) < 50:
        return None
    dens = sorted(((sum(w in kw for w in ws) * 1000 / len(ws), s) for s, kw in SUBJECT.items()), reverse=True)
    (d1, s1), (d2, _) = dens[0], dens[1]
    return s1 if d1 >= 8 and d1 >= 1.5 * d2 else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--sample", type=Path, required=True, help="tokenizer sample v0 (SHA256SUMS, train/, heldout/)")
    ap.add_argument("--heldout", type=Path, required=True, help="fetch_heldout.py output (dev/<lang>.txt)")
    ap.add_argument("--scan", type=int, default=6000, help="train docs per subject pack to scan for medicine/biology/chemistry")
    ap.add_argument("--subject-docs", type=int, default=600, help="docs kept per subject")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    import tiktoken
    import tokenizers
    from tokenizers import Tokenizer

    if sha(a.tokenizer) != P1536:
        sys.exit("not the frozen p1536 tokenizer")
    tok = Tokenizer.from_file(a.tokenizer)
    enc = tiktoken.get_encoding("o200k_base")
    sums = {n: d for d, n in (l.split(maxsplit=1) for l in (a.sample / "SHA256SUMS").read_text().splitlines())}
    held = json.loads((TEAM / "train/heldout_manifest.json").read_text(encoding="utf-8"))
    sec = {"p1536": 0.0, "o200k": 0.0}
    t_start = time.time()

    def checked(rel):
        p = a.sample / rel
        if sha(p) != sums[rel]:
            sys.exit(f"{rel}: sha256 does not match the sample's SHA256SUMS")
        return p

    # 1. every Stage 1 pack, heldout part
    packs = {}
    for rel in sorted(n for n in sums if n.startswith("heldout/")):
        pack = rel.split("/")[1]
        m = packs.setdefault(pack, Meter())
        for t in parquet_texts(checked(rel)):
            m.add(t, tok, enc, sec)
        print(f"{pack}: {m.docs} docs", flush=True)

    # 2. Indic languages (FineWeb-2 test split) and Sanskrit
    groups = {}
    for lang in INDIC:
        f = a.heldout / "dev" / f"{lang}.txt"
        want = held[lang]["files"][f"dev/{lang}.txt"]["sha256"]
        if sha(f) != want:
            sys.exit(f"{f}: sha256 does not match heldout_manifest.json")
        m = groups.setdefault(lang, Meter())
        for line in f.read_text(encoding="utf-8").split("\n"):
            m.add(line, tok, enc, sec)
    sa = groups.setdefault("Sa", Meter())                  # p1536 has no Sanskrit in its training text: train is unseen too
    for rel, limit in (("heldout/ind-sa-fw2/part-00000.parquet", None), ("train/ind-sa-fw2/part-00000.parquet", a.scan)):
        for t in parquet_texts(checked(rel), limit):
            sa.add(t, tok, enc, sec)

    # 3. medicine / biology / chemistry from English and science packs
    subj = {s: Meter() for s in SUBJECT}
    scanned = Counter()
    for pack in SUBJECT_PACKS:
        for rel, limit in ((f"heldout/{pack}/part-00000.parquet", None), (f"train/{pack}/part-00000.parquet", a.scan)):
            if rel not in sums:
                continue
            for t in parquet_texts(checked(rel), limit):
                scanned[pack] += 1
                s = subject_of(t)
                if s and subj[s].docs < a.subject_docs:
                    subj[s].add(t, tok, enc, sec)
    for s, m in subj.items():
        groups[s] = m

    tokens_total = sum(m.p for m in packs.values()) + sum(groups[g].p for g in INDIC) + sum(subj[s].p for s in subj)
    o_total = sum(m.o for m in packs.values()) + sum(groups[g].o for g in INDIC) + sum(subj[s].o for s in subj)
    report = {
        "tokenizer_sha256": P1536, "outside_tokenizer": "tiktoken o200k_base (200,019 ids)",
        "word_rule": "maximal runs of non-whitespace (str.split())", "subject_rule": SUBJECT_RULE,
        "subject_keywords": {k: sorted(v) for k, v in SUBJECT.items()}, "subject_docs_scanned": dict(scanned),
        "packs": {k: m.row() for k, m in packs.items()},
        "groups": {k: {**m.row(), "most_split": m.most_split(tok)} for k, m in groups.items()},
        "encoder_tokens_per_second_one_core": {"p1536": round(tokens_total / sec["p1536"]),
                                               "o200k": round(o_total / sec["o200k"])},
        "inputs": {"sample_SHA256SUMS": sha(a.sample / "SHA256SUMS"),
                   "heldout_manifest": sha(TEAM / "train/heldout_manifest.json")},
        "environment": {"python": platform.python_version(), "tokenizers": tokenizers.__version__,
                        "tiktoken": tiktoken.__version__, "platform": platform.platform(),
                        "script_sha256": sha(__file__)},
        "seconds": round(time.time() - t_start),
    }
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(report, a.out.with_suffix(".md"))
    return 0


def write_md(r, path):
    def line(name, x):
        return (f"| {name} | {x['docs']:,} | {x['bytes'] / 1e6:.2f} | {x['words']:,} | **{x['p1536_tokens_per_word']}** | "
                f"{x['o200k_tokens_per_word']} | {x['p1536_bytes_per_token']} | {x['o200k_bytes_per_token']} | "
                f"{x['p1536_vs_o200k_tokens'] * 100:+.1f}% | {x['lossless_p1536']} |")
    head = ["| {} | Docs | MB | Words | p1536 tok/word | o200k tok/word | p1536 bytes/tok | o200k bytes/tok | "
            "p1536 tokens vs o200k | Lossless (p1536) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    sp = r["encoder_tokens_per_second_one_core"]
    md = [f"# Tokens per word with the frozen tokenizer p1536 `{P1536[:8]}…`", "",
          "Made by `harness/fertility_domains.py`. Outside tokenizer with a bigger vocabulary: o200k_base (200,019 ids).", "",
          f"- **Word rule (all languages):** {r['word_rule']}. tokens/word = tokens / words of the same text; no bos or eos.",
          f"- **Encoder speed, one core:** p1536 {sp['p1536']:,} tokens/s, o200k {sp['o200k']:,} tokens/s "
          f"(`tokenizers` {r['environment']['tokenizers']}, {r['environment']['platform']}).",
          f"- **Lossless:** decode(encode(x)) == normalizer(x) for every document in every table below.", "",
          "## 1. Tokens per word by group (Task D)", "",
          "Indic: FineWeb-2 test split (`heldout/dev`, never in tokenizer training; Docs counts lines there). "
          "Sanskrit: sample-v0 `ind-sa-fw2`, heldout part plus the first 6,000 train docs (p1536 has no Sanskrit training text). "
          f"Medicine, biology, chemistry: English and science documents grouped by keyword ({r['subject_rule']}).", ""]
    md += [head[0].format("Group"), head[1]]
    md += [line(k, v) for k, v in r["groups"].items()]
    md += ["", "## 2. Every Stage 1 pack, heldout part (Task C)", "", head[0].format("Pack"), head[1]]
    md += [line(k, v) for k, v in r["packs"].items()]
    md += ["", "## 3. The 20 words that split into the most p1536 pieces, per group", "",
           "Distinct words (2–40 letters and marks, seen at least twice), encoded with a leading space as in running text.", ""]
    for g, v in r["groups"].items():
        md += [f"### {g}", "", "| Word | Pieces | Count | Split |", "|---|---:|---:|---|"]
        md += [f"| {w['word']} | {w['pieces']} | {w['count']} | {w['split']} |" for w in v["most_split"]]
        md.append("")
    md += ["## Reproduce", "", "```",
           "python teams/03-tokenizer/harness/fertility_domains.py --tokenizer release/tokenizer.json \\",
           "    --sample <tokenizer sample v0> --heldout <fetch_heldout.py dir> --out teams/03-tokenizer/results/fertility_domains",
           "```"]
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:40]))


if __name__ == "__main__":
    sys.exit(main())
