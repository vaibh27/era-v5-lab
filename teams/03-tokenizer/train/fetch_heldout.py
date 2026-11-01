#!/usr/bin/env python3
"""
fetch_heldout.py
================

Writes the held-out sets for v2 (#163, see v2_spec.md). None of them is used to
train merges, and FLORES / HumanEval / MBPP / GSM8K are never used here.

    <out>/dev/<lang>.txt     FineWeb-2 `test`, first --indic-mb per language  -> slot selection
    <out>/check/<lang>.txt   FineWeb-2 `test`, next --indic-mb               -> reporting only
    <out>/usage/{En,Code}.txt  FineWeb sample-10BT / codeparrot-clean-valid  -> ranking o200k tokens
    <out>/dev/{En,Code}.txt    the next part of the same streams             -> choosing the extension size
    <out>/check/{En,Code}.txt  the part after that                           -> reporting only

Each document is written followed by "\\n" (newlines inside it are kept; the file is
measured as a whole). A held-out document is dropped if half or more of its long
lines (40+ chars) also occur in the training corpus (--train-corpus). A code file
is dropped if it shares a 13-gram (whitespace tokens) with HumanEval or MBPP.

Usage
-----
    python fetch_heldout.py --out /path/to/heldout --train-corpus /path/to/corpus

Do not commit the output.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_corpus import LANGS  # noqa: E402

LONG = 40


def line_hashes(text):
    return {hashlib.blake2b(s.strip().encode(), digest_size=8).digest()
            for s in text.split("\n") if len(s.strip()) >= LONG}


def train_hashes(path):
    """Hashes of every long line of a training file."""
    with open(path, encoding="utf-8") as f:
        return line_hashes(f.read())


def ngrams(text, n=13):
    w = text.split()
    return {" ".join(w[i:i + n]) for i in range(len(w) - n + 1)}


def eval_code_ngrams():
    """13-grams of HumanEval (prompt + solution) and MBPP-sanitized (all splits)."""
    from datasets import load_dataset
    grams = set()
    for r in load_dataset("openai/openai_humaneval", split="test"):
        grams |= ngrams(r["prompt"] + r["canonical_solution"])
    for split in ("train", "validation", "test", "prompt"):
        for r in load_dataset("google-research-datasets/mbpp", "sanitized", split=split):
            grams |= ngrams(r["prompt"] + "\n" + r["code"])
    return grams


def stream(name, config, split, field):
    from datasets import load_dataset
    for row in load_dataset(name, config, split=split, streaming=True):
        text = row[field].strip()
        if text:
            yield text


def write_parts(docs, parts, keep):
    """Fill each (path, budget bytes) in order from `docs`; `keep(text)` filters. Returns manifest entries."""
    out, dropped = {}, 0
    it = iter(docs)
    for path, budget in parts:
        h = hashlib.sha256()
        n_docs = n_bytes = 0
        with path.open("w", encoding="utf-8") as f:
            for text in it:
                if not keep(text):
                    dropped += 1
                    continue
                b = (text + "\n").encode("utf-8")
                f.write(text + "\n")
                h.update(b)
                n_docs += 1
                n_bytes += len(b)
                if n_bytes >= budget:
                    break
        if n_bytes < budget:
            print(f"  WARNING: {path} has only {n_bytes / 1e6:.2f} MB (wanted {budget / 1e6:.2f})", flush=True)
        out[str(path.relative_to(path.parent.parent))] = {"docs": n_docs, "bytes": n_bytes, "sha256": h.hexdigest()}
    return out, dropped


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="output directory (outside the repo)")
    ap.add_argument("--train-corpus", required=True, help="fetch_corpus.py output used for training")
    ap.add_argument("--indic-mb", type=float, default=2)
    ap.add_argument("--en-mb", default="30,5,5", help="usage,dev,check MB for English")
    ap.add_argument("--code-mb", default="20,3,3", help="usage,dev,check MB for code")
    ap.add_argument("--langs", default="all")
    args = ap.parse_args()

    out = Path(args.out)
    for d in ("dev", "check", "usage"):
        (out / d).mkdir(parents=True, exist_ok=True)
    manifest_path = out / "heldout_manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def mb(x):
        return int(float(x) * 1_000_000)

    langs = list(LANGS) if args.langs == "all" else args.langs.split(",")

    for lang in [l for l in langs if l in LANGS]:
        t0 = time.time()
        seen = train_hashes(Path(args.train_corpus) / f"{lang}.txt")
        def keep(text, seen=seen):
            hs = line_hashes(text)
            return not hs or len(hs & seen) * 2 < len(hs)
        parts = [(out / "dev" / f"{lang}.txt", mb(args.indic_mb)), (out / "check" / f"{lang}.txt", mb(args.indic_mb))]
        files, dropped = write_parts(stream("HuggingFaceFW/fineweb-2", LANGS[lang], "test", "text"), parts, keep)
        manifest[lang] = {"source": f"HuggingFaceFW/fineweb-2 {LANGS[lang]} test", "dropped_overlap_with_train": dropped,
                          "files": files}
        print(f"{lang}: {files}  dropped {dropped}  ({time.time() - t0:.0f}s)", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    if "En" in langs or args.langs == "all":
        t0 = time.time()
        u, d, c = map(mb, args.en_mb.split(","))
        parts = [(out / "usage" / "En.txt", u), (out / "dev" / "En.txt", d), (out / "check" / "En.txt", c)]
        files, dropped = write_parts(stream("HuggingFaceFW/fineweb", "sample-10BT", "train", "text"), parts, lambda t: True)
        manifest["En"] = {"source": "HuggingFaceFW/fineweb sample-10BT train, in stream order", "files": files}
        print(f"En: {files}  ({time.time() - t0:.0f}s)", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))

    if "Code" in langs or args.langs == "all":
        t0 = time.time()
        grams = eval_code_ngrams()
        u, d, c = map(mb, args.code_mb.split(","))
        parts = [(out / "usage" / "Code.txt", u), (out / "dev" / "Code.txt", d), (out / "check" / "Code.txt", c)]
        files, dropped = write_parts(stream("codeparrot/codeparrot-clean-valid", None, "train", "content"), parts,
                                     lambda t: not (ngrams(t) & grams))
        manifest["Code"] = {"source": "codeparrot/codeparrot-clean-valid (Python), in stream order",
                            "dropped_13gram_overlap_with_humaneval_mbpp": dropped, "files": files}
        print(f"Code: {files}  dropped {dropped}  ({time.time() - t0:.0f}s)", flush=True)
        manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"wrote {manifest_path}")


if __name__ == "__main__":
    main()
