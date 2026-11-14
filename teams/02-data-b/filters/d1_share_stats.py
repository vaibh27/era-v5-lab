#!/usr/bin/env python3
"""d1_share_stats.py - the numbers behind d1_proposal_v1.md: the share of the pack's script in each document.

Prints counts only (never document text): for each pack, the lowest share, the documents under 0.7 / 0.5 / 0.3,
the gold documents and their lowest share, and for each document under 0.5 the largest other script.
The share is the one `medicine_d1` uses (letters and combining signs only, from medicine.md section D1).

  python d1_share_stats.py --packs-dir ../../../packs --gold-dir ../../../gold_set
"""
import argparse
import csv
import re
import sys
import textwrap
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools"))
sys.path.insert(0, str(HERE))
import harm_test_indic as ht  # noqa: E402
import run_harm  # noqa: E402


def d1_namespace(medicine=HERE / "medicine.md"):
    src = medicine.read_text(encoding="utf-8")
    marks = list(ht.SECTION.finditer(src))
    ns = {"re": re, "__name__": "medicine"}
    for fm in ht.FENCE.finditer(src, marks[0].start(), marks[1].start()):
        exec(compile(textwrap.dedent(fm.group(2)), "medicine.md D1", "exec"), ns)
    return ns


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--packs-dir", default=str(HERE.parent.parent.parent.parent / "packs"))
    ap.add_argument("--gold-dir", default=str(HERE.parent.parent.parent.parent / "gold_set"))
    a = ap.parse_args()
    ns = d1_namespace()
    counts, min_letters = ns["_script_counts"], ns["_MIN_LETTERS"]
    print("pack | script | docs | lowest | <0.7 | <0.5 | <0.3 | gold docs | lowest gold | docs under 0.5: largest other script")
    for f in sorted(Path(a.packs_dir).glob("*.parquet")):
        pid, script = f.stem, run_harm.script_of_pack(f.stem)
        gold_file = Path(a.gold_dir) / (pid + "_gold.csv")
        gold = set()
        if gold_file.exists():
            with open(gold_file, encoding="utf-8", newline="") as fh:
                gold = {str(r["doc_id"]) for r in csv.DictReader(fh)}
        t = pq.read_table(f, columns=["doc_id", "text"]).to_pydict()
        shares, gold_shares, low = [], [], []
        for d, x in zip(t["doc_id"], t["text"]):
            c = counts(x)
            n = sum(c.values())
            if n < min_letters:
                continue
            s = c.get(script, 0) / n
            shares.append(s)
            if str(d) in gold:
                gold_shares.append(s)
            if s < 0.5:
                other = max(((v, k) for k, v in c.items() if k != script), default=(0, "-"))
                low.append("%s %.0f%%" % (other[1], 100 * other[0] / n))
        print("%s | %s | %d | %.2f | %d | %d | %d | %d | %s | %s" % (
            pid, script, len(t["text"]), min(shares), sum(s < .7 for s in shares), sum(s < .5 for s in shares),
            sum(s < .3 for s in shares), len(gold_shares), ("%.2f" % min(gold_shares)) if gold_shares else "-",
            ", ".join(low) or "-"))


if __name__ == "__main__":
    main()
