#!/usr/bin/env python3
"""run_harm.py - harm-test medicine.md on each pack's own gold set, one pack at a time.

The D1 filter needs to know the pack's script, which harm_test_indic.py cannot pass to a rule. This wrapper sets
D1_SCRIPT for each pack and calls the tool once per pack (TSAI: each pack on its own gold set, never pooled).
It writes harm_<pack>.md for each pack. A pack with no gold set gets "gold set empty, not testable" and needs no pack file.

Usage (from teams/02-data-b/filters):
  python run_harm.py --packs-dir PATH_TO_PARQUETS [--packs ind-kn-fw2 ...]
Defaults: all 22 packs of ../gold/gold_coverage.csv, medicine.md, gold in ../gold, output in this folder.
Exit codes: 0 all tested packs pass, 1 a pack fails, 2 a pack file is missing or the tool could not run.
One pack by hand, without this wrapper (set D1_SCRIPT to the pack's script first):
  D1_SCRIPT=Kannada python ../tools/harm_test_indic.py --medicine medicine.md --pack ind-kn-fw2.parquet \
      --gold-dir ../gold --md-out harm_ind-kn-fw2.md
"""
import argparse
import csv
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
import harm_test_indic as ht  # noqa: E402

# language code in the pack id (ind-<lang>-<source>) -> script of its text
LANG_SCRIPT = {
    "hi": "Devanagari", "mr": "Devanagari", "sa": "Devanagari",
    "bn": "Bengali", "gu": "Gujarati", "or": "Odia", "ta": "Tamil",
    "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
}
EMPTY_TEXT = "## {pack}\n4. **Harm test:** gold set empty, not testable.\n"


def script_of_pack(pack):
    parts = pack.split("-")
    if len(parts) < 3 or parts[1] not in LANG_SCRIPT:
        raise ValueError(f"{pack}: cannot tell the script from the pack id")
    return LANG_SCRIPT[parts[1]]


def all_packs(gold_dir):
    with open(gold_dir / "gold_coverage.csv", encoding="utf-8-sig", newline="") as f:
        return [r["pack_id"] for r in csv.DictReader(f)]


def find_pack_file(packs_dir, pack):
    for ext in (".parquet", ".jsonl"):
        p = packs_dir / (pack + ext)
        if p.exists():
            return p
    return None


def read_all_docs(pack_file):
    """[(pack, doc_id, text)] for every document of a pack (the harm tool reads only the gold ids)."""
    if pack_file.suffix == ".jsonl":
        import json
        rows = [json.loads(l) for l in pack_file.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        import pyarrow.parquet as pq
        rows = pq.read_table(pack_file, columns=["doc_id", "text"]).to_pylist()
    return [(pack_file.stem, r["doc_id"], r["text"]) for r in rows]


def joiner_report(medicine, docs):
    """One markdown line for each filter: ZWJ and ZWNJ in the gold documents before and after the filter.
    A dropped document leaves the set whole, so its joiners are listed apart. Each document the filter keeps or
    fixes must hold the same ZWJ and ZWNJ as before. Only counts are written, never text."""
    rules, _ = ht.load_rules(medicine)
    lines = []
    for code, fn in rules:
        if fn is None:
            continue
        zb = nb = za = na = zd = nd = ndrop = 0  # ZWJ and ZWNJ: before, after, in dropped documents
        same = True
        for _, _, text in docs:
            verdict, new = ht.classify(text, fn(text))
            j, n = text.count(ht.ZWJ), text.count(ht.ZWNJ)
            zb, nb = zb + j, nb + n
            if verdict == "drop":
                zd, nd, ndrop = zd + j, nd + n, ndrop + 1
            else:
                za, na = za + new.count(ht.ZWJ), na + new.count(ht.ZWNJ)
                same = same and (new.count(ht.ZWJ), new.count(ht.ZWNJ)) == (j, n)
        lines.append(f"   - ZWJ/ZWNJ before and after {code}: ZWJ {zb} to {za}, ZWNJ {nb} to {na} "
                     f"({ndrop} dropped documents held ZWJ {zd} and ZWNJ {nd}). "
                     f"**{'PASS' if same else 'FAIL'}**: every kept document keeps its joiners.")
    return lines

def proxy_report(medicine, docs):
    """Information only, never a harm result: what each filter does to ALL documents of a pack that has no gold set.
    Counts only (changed, dropped, documents whose ZWJ or ZWNJ count differs, documents that left NFC)."""
    rules, _ = ht.load_rules(medicine)
    lines = []
    for code, fn in rules:
        if fn is None:
            continue
        res = ht.run_rule(code, [fn], docs, 5.0)
        n = len(docs)
        lines.append(f"- {code}: changed {len(res['changed'])} of {n}, dropped {len(res['dropped'])} of {n}; "
                     f"documents with a different ZWJ or ZWNJ count: {len(res['joiners'])}; "
                     f"documents that left NFC or lost a compatibility character: {len(res['form'])}; "
                     f"errors: {len(res['errors'])}.")
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--packs-dir", type=Path, default=Path("packs"), help="folder of <pack_id>.parquet files")
    ap.add_argument("--gold-dir", type=Path, default=HERE.parent / "gold")
    ap.add_argument("--medicine", type=Path, default=HERE / "medicine.md")
    ap.add_argument("--out-dir", type=Path, default=HERE)
    ap.add_argument("--packs", nargs="*", help="pack ids (default: every pack in gold_coverage.csv)")
    a = ap.parse_args(argv)
    packs = a.packs or all_packs(a.gold_dir)
    worst = 0
    for pack in packs:
        out = a.out_dir / f"harm_{pack}.md"
        gold = a.gold_dir / f"{pack}_gold.csv"
        ids = ht.read_gold(gold) if gold.exists() else []
        if not ids:
            text = EMPTY_TEXT.format(pack=pack)
            pack_file = find_pack_file(a.packs_dir, pack)
            if pack_file is not None:  # the pack is here: say what the filters do to it, as information only
                os.environ["D1_SCRIPT"] = script_of_pack(pack)
                rows = read_all_docs(pack_file)
                info = proxy_report(a.medicine, rows)
                text += ("\n## Information only, not a harm result\n"
                         f"All {len(rows)} documents of the pack, because it has no gold set. Counts only.\n" + "\n".join(info) + "\n")
                print(f"{pack}: gold set empty, not testable\n" + "\n".join(info))
            else:
                print(f"{pack}: gold set empty, not testable")
            out.write_text(text, encoding="utf-8")
            continue
        pack_file = find_pack_file(a.packs_dir, pack)
        if pack_file is None:
            print(f"{pack}: pack file not found in {a.packs_dir} (needs {pack}.parquet)", file=sys.stderr)
            worst = max(worst, 2)
            continue
        os.environ["D1_SCRIPT"] = script_of_pack(pack)
        print(f"== {pack} (D1_SCRIPT={os.environ['D1_SCRIPT']})")
        code = ht.main(["--medicine", str(a.medicine), "--pack", str(pack_file),
                        "--gold-dir", str(a.gold_dir), "--md-out", str(out)])
        if out.exists():  # the before-and-after joiner count of each filter, on this pack's gold documents
            lines = joiner_report(a.medicine, ht.read_docs(pack_file, ids))
            print("\n".join(l.strip() for l in lines))
            out.write_text(out.read_text(encoding="utf-8").rstrip("\n") + "\n\n## ZWJ and ZWNJ before and after\n"
                           + "\n".join(l.strip() for l in lines) + "\n", encoding="utf-8")
        worst = max(worst, code)
    os.environ.pop("D1_SCRIPT", None)
    return worst


if __name__ == "__main__":
    sys.exit(main())
