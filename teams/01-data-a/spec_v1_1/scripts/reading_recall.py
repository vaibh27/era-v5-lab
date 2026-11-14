#!/usr/bin/env python3
"""How many of the readers' docs each rule of a medicine sheet treats, on the cleaned text the readers read.

Loads the sheet the way tools/harm_test.py does, and runs each rule on the cleaned docs of the reading rows that
carry its reading code. A doc counts as treated when the rule changes or drops it. This gives the
"reading rows the rule treats / reading rows with its code" figure on each card. The harm test on gold
is separate.

Only doc_ids and counts are printed, never text.

From spec_v1_1/scripts/:
    python3 -I reading_recall.py --rules ../sheets/medicine_math.md --domain math \\
        --parts <folder>/math-finemath/part-00000.parquet <folder>/math-megamath/part-00000.parquet \\
        --code DA-C9-003=NEW-template-pages

A rule's reading code is its CODE unless --code says otherwise. Needs pyarrow (only to read the parts).
"""
import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent  # teams/01-data-a/spec_v1_1/scripts
TOOLS = HERE.parents[1] / "tools"
READING = HERE.parent / "results" / "reading"
DOMAINS = ("english", "code", "math", "science")
FIELDS = ("doc_id", "text", "url", "lang_label", "pack_id")


def parse_codes(values):
    """['DA-C9-003=NEW-template-pages'] -> {'DA-C9-003': 'NEW-template-pages'}."""
    out = {}
    for v in values or ():
        rule, sep, code = v.partition("=")
        if not sep or not rule.strip() or not code.strip():
            raise ValueError(f"--code wants RULE=READING_CODE, got {v!r}")
        out[rule.strip()] = code.strip()
    return out


def reading_docs(path: Path, domain: str, codes):
    """{reading code: {doc_id: pack}} for the rows of this domain whose code is wanted (kind new only)."""
    found = defaultdict(dict)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["domain"] == domain and r["kind"] == "new" and r["code"] in codes:
                found[r["code"]].setdefault(r["doc_id"], r["pack"])
    return found


def load_docs(parts, doc_ids):
    """doc_id -> {doc_id, text, url, lang_label, pack_id} with the columns the parts have."""
    import pyarrow.parquet as pq
    docs, wanted = {}, sorted(doc_ids)
    for part in parts:
        cols = [c for c in FIELDS if c in pq.read_schema(part).names]
        rows = pq.read_table(part, columns=cols, filters=[("doc_id", "in", wanted)]).to_pylist()
        for row in rows:
            docs.setdefault(row["doc_id"], row)
    return docs


def tally(rule, classify, docs_by_id, wanted):
    """Run the rule on each wanted doc: {'fix': [...], 'drop': [...], 'keep': [...], 'missing': [...], 'error': [...]}."""
    out = {k: [] for k in ("fix", "drop", "keep", "missing", "error")}
    for doc_id, pack in sorted(wanted.items()):
        d = docs_by_id.get(doc_id)
        if d is None:
            out["missing"].append(doc_id)
            continue
        d = dict(d)
        d.setdefault("pack_id", pack)
        d.setdefault("url", "")
        d.setdefault("lang_label", None)
        try:
            out[classify(d["text"], rule(d))].append(doc_id)
        except Exception as e:  # a rule that raises is a rule bug; its type only
            out["error"].append(f"{doc_id} ({type(e).__name__})")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--rules", required=True, type=Path, help="a medicine sheet (sheets/medicine_<domain>.md)")
    ap.add_argument("--domain", required=True, choices=DOMAINS)
    ap.add_argument("--parts", nargs="+", required=True, type=Path, help="cleaned sample Parquet parts (Train)")
    ap.add_argument("--code", action="append", metavar="RULE=CODE", help="the reading code of a rule; repeatable")
    ap.add_argument("--reading", type=Path, default=READING, help="the collector's output folder")
    args = ap.parse_args(argv)
    try:
        overrides = parse_codes(args.code)
    except ValueError as e:
        ap.error(str(e))

    sys.path.insert(0, str(TOOLS))
    import harm_test  # only its loader and classify are used; the file is not changed
    ns, _ = harm_test.load_rules(args.rules)
    rules = ns.get("RULES_BY_ID") or {}
    codes = {rid: overrides.get(rid, fn.__globals__.get("CODE", "")) for rid, fn in rules.items()}
    unknown = sorted(set(overrides) - set(rules))
    if unknown:
        ap.error("--code names rules that are not in the sheet: " + ", ".join(unknown))

    wanted = reading_docs(args.reading / "findings_all.csv", args.domain, set(codes.values()))
    docs = load_docs(args.parts, {d for ids in wanted.values() for d in ids})
    if docs and not any("lang_label" in d for d in docs.values()):
        print("note: the parts have no lang_label column; rules that check LANGS will keep every doc", file=sys.stderr)

    print(f"{args.rules.name}, {args.domain}: {len(rules)} rules, {len(docs)} docs read from {len(args.parts)} parts")
    for rid, code in codes.items():
        ids = wanted.get(code, {})
        t = tally(rules[rid], harm_test.classify, docs, ids)
        treated = len(t["fix"]) + len(t["drop"])
        print(f"{rid}  {code}: treats {treated}/{len(ids)} reading docs "
              f"(fix {len(t['fix'])}, drop {len(t['drop'])})")
        for what in ("keep", "missing", "error"):
            if t[what]:
                print(f"  {what}: " + ", ".join(t[what]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
