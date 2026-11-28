"""Check the two medicines of math-audit-Aneesha_Das.md (DA-C1-012, DA-C2-007): speed on full packs, and the audit docs
each one treats.

    python -I teams/01-data-a/audit/medicine_math_checks.py speed <pack.parquet> [<pack.parquet> ...] [--json <out.json>]
    python -I teams/01-data-a/audit/medicine_math_checks.py treated --post-dir <folder>

speed: for each file, the wall time of each rule over every doc, before (the frozen copy below, as it was before the
speed-up) and after (the card in the report), the docs it changes, the docs where the two differ and the docs a second
run changes (both must be 0), then both rules in order, C1 first. A file is a Cap1 pack part
(<pack>/<Part_1|Heldout>/part-00000.parquet) or a 500-doc sample pack (<pack>.parquet). Exit code 1 when a doc differs
or a second run changes one.
treated: on the cleaned packs the audit read (<folder>/<pack_id>/part-00000.parquet), the audit docs labelled C1 or C2,
how many are in the cleaned files, how many of those each rule changes, and the audit docs without the code it changes.

The loop over docs is this script calling a rule once per doc, as clean.py does; the rules have no Python loop over a
doc's lines or passages. Prints doc_ids, counts and timings only, never document text. Needs pyarrow.
"""
import argparse
import csv
import json
import re
import sys
import textwrap
import time
from collections import Counter
from pathlib import Path

T0 = time.perf_counter()
REPO = Path(__file__).resolve().parents[3]
AUDIT = REPO / "teams/01-data-a/audit"
REPORT = AUDIT / "math-audit-Aneesha_Das.md"
PACKS = ("math-finemath", "math-megamath")


def card(rule_id, report=REPORT):
    """The card's python block run on its own, in a fresh namespace."""
    src = Path(report).read_text(encoding="utf-8")
    body = re.search(rf"^## Medicine \d+ — math / \w+ / `{rule_id}`.*?^```python\n(.*?)^```", src, re.S | re.M).group(1)
    ns = {"__name__": "rules"}
    exec(textwrap.dedent(body), ns)
    return ns


C1, C2 = card("DA-C1-012"), card("DA-C2-007")


# ---------------------------------------------------------------- frozen copies, as before the speed-up. Keep unchanged.
# They use the regexes of their card, so only the speed-ups are compared.

def old_c1(text, ns=C1):
    # no gates: every regex runs on every doc
    text = text.replace("{[ promptMessage ]}", "")
    return ns["C1_CHROME"].sub("", ns["C1_EDIT"].sub("", text))


def _old_c2_passages(text, ns):
    parts, seen, out = ns["C2_PARA_SPLIT"].split(text), set(), []
    for k, part in enumerate(parts):
        q = part.strip()
        if k % 2 == 0 and len(q) >= 150 and not re.match(r"\s*[•*-] ", q):
            ql = [x for x in q.splitlines() if x.strip()]
            options = sum(bool(ns["C2_OPTION"].match(x)) for x in ql) >= max(2, len(ql) // 2)
            code = sum(bool(re.search(r"[;{}]\s*$", x)) for x in ql) >= max(1, len(ql) // 3)
            if not options and not code:
                if q in seen:
                    if out and k > 0:
                        out.pop()
                    continue
                seen.add(q)
        out.append(part)
    return "".join(out)


def old_c2(text, ns=C2):
    lines = text.split("\n")
    counts = Counter(map(str.strip, lines))
    cut = {s for s, n in counts.items()
           if n >= 2 and len(s) >= 15 and not ns["C2_CODE"].search(s)
           and ((n >= 5 and not ns["C2_MATH"].search(s) and not ns["C2_TURN"].match(s)) or ns["C2_CAPTION"].match(s))}
    if cut:
        seen, out = set(), []
        for l in lines:
            s = l.strip()
            if s in cut:
                if s in seen:
                    continue
                seen.add(s)
            out.append(l)
        text = "\n".join(out)
    return _old_c2_passages(text, ns)


# ---------------------------------------------------------------- speed

def pack_of(path):
    return path.parent.parent.name if path.parent.name in ("Part_1", "Part_2", "Heldout") else path.stem


def timed(fn, items):
    s = time.perf_counter()
    out = list(map(fn, items))
    return out, time.perf_counter() - s


def speed(paths):
    import pyarrow.parquet as pq
    c1, c2 = C1["medicine"], C2["medicine"]
    rows, total = [], {"docs": 0, "chars": 0}
    print(f"rules loaded: {time.perf_counter() - T0:.2f} s after start")
    for path in paths:
        s = time.perf_counter()
        table = pq.read_table(path, columns=["doc_id", "text"])
        ids, texts = table["doc_id"].to_pylist(), table["text"].to_pylist()
        read_s = time.perf_counter() - s
        pack = pack_of(path)
        docs = [{"doc_id": i, "text": t, "lang_label": "en", "pack_id": pack} for i, t in zip(ids, texts)]
        chars = sum(map(len, texts))
        total["docs"] += len(docs)
        total["chars"] += chars
        print(f"\n{pack} {path.parent.name}/{path.name}: {len(docs):,} docs, {chars / 1e6:.1f} M chars, read {read_s:.2f} s")
        print(f"  {'rule':<14}{'before s':>10}{'after s':>10}{'speedup':>9}{'docs/s after':>14}{'changed':>9}{'differ':>8}"
              f"{'not idem':>10}")
        for rule_id, before_fn, after_fn in (("DA-C1-012", old_c1, c1), ("DA-C2-007", old_c2, c2)):
            before, before_s = timed(before_fn, texts)
            after_out, after_s = timed(after_fn, docs)
            after = [t for _, t in after_out]
            changed = sum(x != t for x, t in zip(after, texts))
            differ = [i for i, x, y in zip(ids, after, before) if x != y]
            # idempotence, outside the timed part: a second run on the output must change nothing
            not_idem = [d["doc_id"] for d, x in zip(docs, after) if after_fn({**d, "text": x})[1] != x]
            print(f"  {rule_id:<14}{before_s:>10.2f}{after_s:>10.2f}{before_s / after_s:>8.1f}x"
                  f"{len(docs) / after_s:>14,.0f}{changed:>9,}{len(differ):>8}{len(not_idem):>10}")
            if differ:
                print(f"    differ (first 5): {', '.join(differ[:5])}")
            if not_idem:
                print(f"    not idempotent (first 5): {', '.join(not_idem[:5])}")
            rows.append({"file": path.name, "pack": pack, "part": path.parent.name, "rule": rule_id, "docs": len(docs),
                         "chars": chars, "before_s": round(before_s, 3), "after_s": round(after_s, 3),
                         "changed": changed, "differ": len(differ), "differ_ids": differ[:20],
                         "not_idempotent": len(not_idem), "not_idempotent_ids": not_idem[:20]})
        s = time.perf_counter()
        both = [c2({**d, "text": c1(d)[1]})[1] for d in docs]
        both_s = time.perf_counter() - s
        print(f"  {'both, in order':<14}{'':>10}{both_s:>10.2f}{'':>9}{len(docs) / both_s:>14,.0f}"
              f"{sum(x != t for x, t in zip(both, texts)):>9,}")
        rows.append({"file": path.name, "pack": pack, "part": path.parent.name, "rule": "both",
                     "after_s": round(both_s, 3)})
    wall = time.perf_counter() - T0
    bad = sum(r.get("differ", 0) for r in rows)
    not_idem = sum(r.get("not_idempotent", 0) for r in rows)
    print(f"\nALL: {total['docs']:,} docs, {total['chars'] / 1e6:.1f} M chars; wall time {wall:.1f} s from start; "
          f"docs that differ from the frozen copies: {bad}; docs a second run changes: {not_idem}")
    return {"wall_s": round(wall, 2), "total": total, "rows": rows}, bool(bad or not_idem)


# ---------------------------------------------------------------- treated

def treated(post_dir):
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    labels, texts = {}, {}
    for pack in PACKS:
        with open(AUDIT / f"{pack}-Part_1-Aneesha_Das.csv", encoding="utf-8", newline="") as f:
            for row in csv.DictReader(f):
                labels[row["doc_id"]] = (pack, set(filter(None, row["diseases"].split(";"))))
        ids = [i for i, (p, _) in labels.items() if p == pack]
        t = pq.read_table(post_dir / pack / "part-00000.parquet", columns=["doc_id", "text"])
        t = t.filter(pc.is_in(t["doc_id"], value_set=pa.array(ids, type=t["doc_id"].type)))
        texts.update(zip(t["doc_id"].to_pylist(), t["text"].to_pylist()))
    print(f"audit docs: {len(labels)}; in the cleaned files: {len(texts)}")
    results = {}
    for code, rule_id, ns in (("C1", "DA-C1-012", C1), ("C2", "DA-C2-007", C2)):
        changed = {i for i, t in texts.items()
                   if ns["medicine"]({"doc_id": i, "text": t, "lang_label": "en", "pack_id": labels[i][0]})[1] != t}
        with_code = [i for i, (_, codes) in labels.items() if code in codes]
        present = [i for i in with_code if i in texts]
        hit = [i for i in present if i in changed]
        missed = [i for i in present if i not in changed]
        extra = sorted(i for i in changed if code not in labels[i][1])
        print(f"\n{rule_id} ({code}): {len(with_code)} audit docs labelled {code}, {len(present)} in the cleaned files "
              f"({len(with_code) - len(present)} dropped by the cleaner); the rule changes {len(hit)} of them")
        print(f"  changed: {', '.join(hit) or '-'}")
        print(f"  not changed: {', '.join(missed) or '-'}")
        print(f"  also changes {len(extra)} audit docs not labelled {code}: {', '.join(extra) or '-'}")
        results[rule_id] = {"labelled": len(with_code), "in_cleaned": len(present), "changed": hit, "not_changed": missed,
                            "changed_unlabelled": extra}
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="check", required=True)
    sp = sub.add_parser("speed", help="time the rules on full packs")
    sp.add_argument("packs", nargs="+", type=Path)
    sp.add_argument("--json", type=Path, help="also write the results here")
    tr = sub.add_parser("treated", help="count the audit docs each rule changes")
    tr.add_argument("--post-dir", required=True, type=Path)
    tr.add_argument("--json", type=Path, help="also write the results here")
    a = ap.parse_args(argv)
    if a.check == "speed":
        results, failed = speed(a.packs)
    else:
        results, failed = treated(a.post_dir), False
    if a.json:
        a.json.write_text(json.dumps(results, indent=1) + "\n")
        print(f"results: {a.json.name}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
