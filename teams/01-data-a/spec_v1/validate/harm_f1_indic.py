"""Harm test of the shared F1 on the Indic packs: F1 alone and the whole clean(), on each pack's own gold set.

  python -I spec_v1/validate/harm_f1_indic.py --packs <folder with the ind-*.parquet packs> [--gold ../02-data-b/gold]

Run from teams/01-data-a/. Uses the team's harm_test.py (teams/01-data-a/harmtest/) as it is, and the limit
max(1, floor(1% of gold)). A pack's documents get pack_id from the file name when the Parquet has no such column.
Writes results/spec_v1/cleaner/f1_indic_harm.json and prints one line per pack. IDs, counts and digests only.
Exit 0: all supported Indic packs pass; 1: a pack fails or errors; 2: required inputs are missing or gold is empty.
"""
import argparse, hashlib, json, sys
from fractions import Fraction
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
TEAM = SPEC.parent
sys.path.insert(0, str(TEAM / "harmtest")); import harm_test as ht
sys.path.insert(0, str(SPEC)); import clean as c

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--packs", type=Path, required=True)
ap.add_argument("--gold", type=Path, default=TEAM.parent / "02-data-b" / "gold")
ap.add_argument("--out", type=Path, default=SPEC.parent / "results" / "spec_v1" / "cleaner" / "f1_indic_harm.json")
args = ap.parse_args()


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


report, bad, ran = {"f1_rule_sha256": sha(SPEC / "build" / "f1_rule.py"), "clean_sha256": sha(SPEC / "clean.py"),
                     "harm_test_sha256": sha(TEAM / "harmtest" / "harm_test.py"), "packs": {}}, False, 0
missing = False
for pack in sorted(p for p, f in c._FAMILY.items() if f == "indic"):
    path, gold = args.packs / f"{pack}.parquet", args.gold / f"{pack}_gold.csv"
    if not path.exists() or not gold.exists():
        report["packs"][pack] = {"result": "NOT RUN", "why": "pack not found" if not path.exists() else "no gold file"}
        missing = True
        print(pack, "NOT RUN"); continue
    try:
        ids = ht.read_gold(gold)
    except ht.NotTestable:
        report["packs"][pack] = {"result": "NOT TESTABLE", "why": "Gold set empty, not testable"}
        missing = True
        print(pack, "NOT TESTABLE"); continue
    docs = ht.read_pack_docs(path, ids)
    for d in docs.values():
        d.setdefault("pack_id", pack)
    entry = {"gold": len(ids), "limit": ht.allowed_documents(len(ids), Fraction(1)), "pack_sha256": sha(path), "gold_sha256": sha(gold)}
    for name, fn in (("f1", c.f1), ("clean", c.clean)):
        changed, dropped, errors = ht.run_rule(fn, docs, 5)
        ok = not errors and ht.harm_percent(len(changed), len(dropped), len(ids), Fraction(1))[1]
        entry[name] = {"changed": sorted(changed), "dropped": sorted(dropped), "errors": errors, "result": "PASS" if ok else "FAIL"}
        bad |= not ok
    report["packs"][pack] = entry; ran += 1
    print(f"{pack}: gold {len(ids)} (limit {entry['limit']}) | F1 {len(entry['f1']['changed'])}+{len(entry['f1']['dropped'])} "
          f"{entry['f1']['result']} | clean() {len(entry['clean']['changed'])}+{len(entry['clean']['dropped'])} {entry['clean']['result']}")
args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
sys.exit(1 if bad else 2 if missing or not ran else 0)
