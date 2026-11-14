"""Run clean.py on one pack, or on one part of a pack. Reads only that file, writes only to --out, needs no network.

  python run_clean.py --pack code-go.parquet --out out
  python run_clean.py --pack train/code-go/part-00000.parquet --pack-id code-go --out out

The pack is --pack-id when given, else the file name (<pack_id>.parquet). It must be one of the registered packs clean.py has
rules for: any other stops the run with exit code 1 before the file is read or anything is written, since clean.py would
pass every doc through unchanged. A pack_id column in the file must name the same pack.
Writes out/<name>_clean.parquet (the kept docs, with their cleaned text) and out/<name>_summary.json (counts, dropped
doc_ids, and the SHA-256 of the input file and of the clean.py that ran; no text). <name> is the file name without
.parquet, after the pack when --pack-id names another (code-go_part-00000), so parts of two packs never overwrite each other.

--f2 then runs F2, G9's benchmark fingerprints (f2_rule.py), on each doc clean.py keeps, and drops the ones it flags.
The summary also gets "rule_seconds": for each rule that ran, the seconds it took in total, how many documents it ran on ("calls")
and the milliseconds per call, slowest first. A rule SCOPE skips on this pack, or one never reached because an earlier rule dropped
the document, has no entry. It is the per-rule time profile for Gate 2: it shows which rules are slow at scale.
The summary gets "f2": the set_sha256, and each dropped doc_id with up to 5 benchmark items and the offset in the cleaned text,
and "final": kept, dropped by clean.py, dropped by F2 and errors, which add up to the input rows ("actions" stays clean.py's).

A run stops, before writing, if --out already holds results of another input file with the same name, such as a train and a
heldout part of one pack: give each split its own --out.
"""
import argparse, hashlib, json, sys, time
from collections import Counter
from pathlib import Path
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import clean as c


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--pack", type=Path, required=True, help="the Parquet file: a pack, or a part of one")
ap.add_argument("--pack-id", help="the pack the file belongs to, for a file not named <pack_id>.parquet (a tokenizer sample part)")
ap.add_argument("--out", type=Path, default=Path("out"))
ap.add_argument("--f2", action="store_true", help="after clean.py, drop the docs G9's benchmark fingerprints flag (about 1.1 GB, 20 s to load)")
args = ap.parse_args()
pack = args.pack_id or args.pack.stem
name = args.pack.stem if args.pack.stem == pack else f"{pack}_{args.pack.stem}"
if pack not in c._FAMILY:  # no rule applies to it, so a run would "succeed" without cleaning anything
    sys.exit(f"{args.pack.name}: {pack!r} is not one of the {len(c._FAMILY)} packs clean.py has rules for, so nothing would be cleaned. "
             f"Name the file <pack_id>.parquet or pass --pack-id, with one of: {', '.join(sorted(c._FAMILY))}")
previous = args.out / f"{name}_summary.json"  # a train and a heldout part can share a name: never overwrite another input's results
if previous.exists() and json.loads(previous.read_text(encoding="utf-8")).get("input_sha256") != sha256(args.pack):
    sys.exit(f"{previous} holds the results of another input file with the same name (a train and a heldout part?). Use another --out.")
table = pq.read_table(args.pack)
rows = table.to_pylist()
for row in rows:  # clean.py picks the rules by pack_id: a file without the column takes the pack above
    if row.setdefault("pack_id", pack) != pack:
        sys.exit(f"{args.pack.name}: doc_id {row['doc_id']} has pack_id {row['pack_id']!r}, not {pack!r}")
f2 = None
if args.f2:
    import f2_rule as f2
    try:
        f2.fingerprints()  # load and check the set before anything is written
    except (OSError, ValueError) as e:
        sys.exit(f"--f2: {e}")
rule_time = {}  # rule id -> [seconds, calls]


def _timed(rid, make):
    """The rule make() builds, with the clock running only while it runs: clean.py is not changed, and a rule _run skips costs nothing."""
    def factory():
        rule = make()

        def timed(doc):
            start = time.perf_counter()
            try:
                return rule(doc)
            finally:  # a rule that raises still used the time
                entry = rule_time.setdefault(rid, [0.0, 0])
                entry[0] += time.perf_counter() - start
                entry[1] += 1
        return timed
    return factory


_timed_make = {make: _timed(rid, make) for rid, make in c._RULE.items()}
c._ID.update({timed: c._ID[make] for make, timed in _timed_make.items()})  # _run looks the rule's SCOPE up by its factory
_untimed_run = c._run
c._run = lambda doc, *factories: _untimed_run(doc, *(_timed_make[make] for make in factories))  # every a1()..clean() calls _run
args.out.mkdir(parents=True, exist_ok=True)
t0, actions, dropped, errors, kept, f2_dropped = time.time(), Counter(), [], [], [], []
print(f"cleaning {len(rows)} docs of {pack} ({c._FAMILY[pack]} family)" + (", then F2" if f2 else ""), file=sys.stderr)
for i, row in enumerate(rows, 1):
    try:
        action, text = c.clean(row)
        flagged, hits = f2.check(text) if f2 and action != "drop" else (False, [])
    except Exception as e:  # a rule that crashes keeps the doc out and is reported by type, never by text
        errors.append([row["doc_id"], type(e).__name__])
        if len(errors) == 1: print(f"first error: doc_id {row['doc_id']}: {type(e).__name__}", file=sys.stderr)
    else:
        actions[action] += 1
        if action == "drop": dropped.append(row["doc_id"])
        elif flagged: f2_dropped.append({"doc_id": row["doc_id"], "items": [h["item"] for h in hits[:5]], "offset": hits[0]["offset"]})
        else: kept.append(dict(row, text=text))
    if i % 500 == 0 or i == len(rows):  # progress goes to stderr: stdout keeps the one summary line
        print(f"{i}/{len(rows)} {dict(actions)} errors={len(errors)}" + (f" F2 dropped={len(f2_dropped)}" if f2 else "") + f" {i / max(time.time() - t0, 1e-9):.0f} docs/s", file=sys.stderr, flush=True)
final = {"kept": len(kept), "dropped_by_clean": len(dropped), "dropped_by_f2": len(f2_dropped), "errors": len(errors)}
if sum(final.values()) != len(rows):
    sys.exit(f"counts do not add up to the {len(rows)} input rows: {final}")
schema = table.schema if "pack_id" in table.schema.names else table.schema.append(pa.field("pack_id", pa.string()))
pq.write_table(pa.Table.from_pylist(kept) if kept else schema.empty_table(), args.out / f"{name}_clean.parquet")  # all dropped: keep the columns
summary = {"pack": args.pack.name, "input_sha256": sha256(args.pack), "pack_id": pack, "family": c._FAMILY[pack], "rows": len(rows), "actions": dict(actions), "errors": errors, "dropped": dropped, "seconds": round(time.time() - t0, 1), "docs_per_second": round(len(rows) / max(time.time() - t0, 1e-9), 1),
           "clean_py_sha256": sha256(Path(c.__file__)),
           "rule_seconds": {rid: {"seconds": round(t, 3), "calls": n, "ms_per_call": round(t / n * 1000, 3)}
                            for rid, (t, n) in sorted(rule_time.items(), key=lambda kv: -kv[1][0])}}  # which rules ran: the same hash means the same clean.py
if f2:  # actions are clean.py's; final is what was written, after F2
    summary["f2"] = {"set_sha256": f2.fingerprints().set_sha256, "dropped": f2_dropped}
    summary["final"] = final
(args.out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
profile = summary["rule_seconds"]
rules_total = sum(v["seconds"] for v in profile.values()) or 1e-9
print(f"\nslowest rules of {len(profile)} that ran (rules took {rules_total:.1f} s of {summary['seconds']} s):", file=sys.stderr)
print(f"  {'rule':<14}{'seconds':>10}{'share':>8}{'calls':>9}{'ms/call':>10}", file=sys.stderr)
for rid, v in list(profile.items())[:10]:
    print(f"  {rid:<14}{v['seconds']:>10.1f}{v['seconds'] / rules_total:>8.1%}{v['calls']:>9}{v['ms_per_call']:>10.3f}", file=sys.stderr)
print({k: v for k, v in summary.items() if k not in ("dropped", "f2", "rule_seconds")}, "dropped:", len(dropped), *(("F2 dropped:", len(f2_dropped)) if f2 else ()))
sys.exit(1 if errors else 0)
