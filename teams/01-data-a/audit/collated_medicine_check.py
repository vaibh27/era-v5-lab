"""Run the current cleaner with the cards of collated_medicine.md in place, and compare it with clean() as it is.

    python -I teams/01-data-a/audit/collated_medicine_check.py docs  --raw <tokenizer_sample_v0/train> --out <csv>
    python -I teams/01-data-a/audit/collated_medicine_check.py packs --raw <tokenizer_sample_v0/train> --out <csv> [--pack en-dclm ...]
    python -I teams/01-data-a/audit/collated_medicine_check.py gold  --raw <folder of the 500-document sample packs>

docs   the 313 labelled documents of collation_docs.csv: what the cleaner does now, what it does with the cards, whether the
       output differs, the rules that acted, and the number of <PHONE> masks now and with the cards.
packs  whole packs: documents dropped now and with the cards, recovered, newly dropped, output changed, changed again by a
       second run; and for each card the documents where it does something else than the rule it replaces, on the same text.
gold   the gold documents of each pack that the whole pipeline changes or drops, now and with the cards, and those whose
       output the cards change. Exit code 1 if there is any of the last.

A card with REPLACES runs in the place of that rule. On a pack without that rule, and for a new card, RUNS_AFTER says where:
after that rule, or ("family") after the pack's last family rule, before the shared F1 rule. The sheet is run as code: read it before you run it.
The outputs hold ids, rule ids and counts only: no document text.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import csv
import re
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I does not put the script folder on the path
import collation_replay as replay  # noqa: E402

SHEET = HERE / "collated_medicine.md"
GOLD = HERE.parent / "gold"
PACKS = replay.SCOPE_PACKS + ["sci-finepdfs"]
cards = []  # the namespace of each card, loaded once in each process


def load():
    replay.load()
    cards[:] = []
    for block in re.findall(r"(?ms)^```python\n(.*?)^```", SHEET.read_text(encoding="utf-8")):
        ns = {"__name__": "rules"}
        exec(compile(block, SHEET.name, "exec"), ns)
        cards.append(ns)


def packs_of(card):
    return card.get("SPEC_PACKS") or card["PACKS"]


def with_cards(pack):
    """[(rule id, rule)] of this pack with the cards in place, and {card id: the rule it replaces}"""
    rules, old = replay.rules(pack), {}
    for card in (x for x in cards if pack in packs_of(x)):
        rid, new = card["RULE_ID"], (card["RULE_ID"], card["medicine"])
        ids = [r for r, _ in rules]
        at, after = card.get("REPLACES"), card.get("RUNS_AFTER")
        if at in ids:
            old[rid] = rules[ids.index(at)][1]
            rules[ids.index(at)] = new
            continue
        if not after:
            raise SystemExit(f"{rid}: {at} does not run on {pack}")
        last = len(rules) - (ids[-1] == "f1_rule.py")  # the shared F1 rule stays last
        rules.insert(last if after == "family" else ids.index(after) + 1, new)
    return rules, old


def run(doc, rules):
    """(the rules that acted, the text or None for a drop)"""
    return replay.trace(doc, rules)


def action(doc, text):
    return "drop" if text is None else "keep" if text == doc["text"] else "fix"


def read(path, pack, **kw):
    for doc in pq.read_table(path, columns=replay.COLUMNS, **kw).to_pylist():
        doc["pack_id"] = pack
        yield doc


def docs(args):
    load()
    labels = list(csv.DictReader(open(args.labels, encoding="utf-8", newline="")))
    found = {}
    for pack in sorted({r["pack"] for r in labels}):
        ids = sorted({r["doc_id"] for r in labels if r["pack"] == pack})
        now, (new, _) = replay.rules(pack), with_cards(pack)
        for part in sorted((args.raw / pack).glob("*.parquet")):
            for doc in read(part, pack, filters=[("doc_id", "in", ids)]):
                (_, a), (steps, b) = run(doc, now), run(doc, new)
                found[pack, doc["doc_id"]] = [
                    action(doc, a), action(doc, b), "yes" if a != b else "no", (a or "").count("<PHONE>"), (b or "").count("<PHONE>"),
                    "; ".join(f"{rid} {act}{'' if act == 'drop' else f' {n:+d}'}" for rid, act, n in steps)]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["pack", "doc_id", "audit", "verdict", "now", "with_cards", "output_differs", "phone_masks_now",
                      "phone_masks_with_cards", "rules_with_cards"])
        for r in labels:
            out.writerow([r[k] for k in ("pack", "doc_id", "audit", "verdict")] + found[r["pack"], r["doc_id"]])
    changed = [(r["verdict"], *found[r["pack"], r["doc_id"]][:2]) for r in labels if found[r["pack"], r["doc_id"]][2] == "yes"]
    print(f"{len(labels)} labels, output differs on {len(changed)}:", dict(Counter(changed)))


def one_group(job):
    part, pack, group = job
    now, (new, old) = replay.rules(pack), with_cards(pack)
    n, effect = Counter(), Counter()
    ids = {card["RULE_ID"] for card in cards}
    for doc in pq.ParquetFile(part).read_row_group(group, columns=replay.COLUMNS).to_pylist():
        doc["pack_id"] = pack
        _, a = run(doc, now)
        b, d = doc["text"], dict(doc)
        for rid, rule in new:  # as replay.trace, with each card compared with the rule it replaces on the same text
            act, text = rule(d)
            if rid in ids:
                was, was_text = old[rid](d) if rid in old else ("keep", d["text"])
                if (act == "drop") != (was == "drop") or (act != "drop" and text != was_text):
                    effect[rid] += 1
            if act == "drop":
                b = None
                break
            b = d["text"] = text
        n["docs"] += 1
        n["dropped_now"] += a is None
        n["dropped_with_cards"] += b is None
        n["recovered"] += a is None and b is not None
        n["newly_dropped"] += a is not None and b is None
        n["output_changed"] += a != b
        if b is not None and run(dict(doc, text=b), new)[1] != b:
            n["second_run_changes"] += 1
            n["of_them_also_now"] += a is not None and run(dict(doc, text=a), now)[1] != a
    return pack, n, effect


def packs(args):
    load()
    jobs = []
    for pack in args.pack or PACKS:
        for part in sorted((args.raw / pack).glob("*.parquet")):
            jobs += [(part, pack, g) for g in range(pq.ParquetFile(part).num_row_groups)]
    totals, effects = defaultdict(Counter), defaultdict(Counter)
    with Pool(args.workers, initializer=load) as pool:
        for pack, n, effect in pool.imap_unordered(one_group, jobs):
            totals[pack].update(n)
            effects[pack].update(effect)
    keys = ["docs", "dropped_now", "dropped_with_cards", "recovered", "newly_dropped", "output_changed", "second_run_changes",
            "of_them_also_now"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["pack", "what", "documents"])
        for pack in args.pack or PACKS:
            out.writerows([pack, k, totals[pack][k]] for k in keys)
            out.writerows([pack, f"{c['RULE_ID']} differs from the rule it replaces" if c.get("REPLACES") else
                           f"{c['RULE_ID']} changes", effects[pack][c["RULE_ID"]]] for c in cards if pack in packs_of(c))
            print(pack, dict(totals[pack]), dict(effects[pack]), flush=True)


def gold(args):
    load()
    bad = 0
    for pack in PACKS:
        ids = [r["doc_id"] for r in csv.DictReader(open(GOLD / f"{pack}_gold.csv", encoding="utf-8-sig", newline=""))]
        now, (new, _) = replay.rules(pack), with_cards(pack)
        n = Counter()
        for doc in read(args.raw / f"{pack}.parquet", pack, filters=[("doc_id", "in", ids)]):
            n["found"] += 1
            a, b = run(doc, now)[1], run(doc, new)[1]
            n[f"{action(doc, a)} now"] += 1
            n[f"{action(doc, b)} with cards"] += 1
            n["differ"] += a != b
        bad += n["differ"] > 0 or n["found"] != len(ids)
        print(f"{pack}: {n['found']} of {len(ids)} gold documents; changed / dropped now {n['fix now']} / {n['drop now']}, "
              f"with the cards {n['fix with cards']} / {n['drop with cards']}; output differs on {n['differ']}")
    sys.exit(bool(bad))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in (("docs", docs), ("packs", packs), ("gold", gold)):
        p = sub.add_parser(name)
        p.set_defaults(run=fn)
        p.add_argument("--raw", type=Path, required=True)
        if name != "gold":
            p.add_argument("--out", type=Path, required=True)
        if name == "docs":
            p.add_argument("--labels", type=Path, default=HERE / "collation_docs.csv")
        if name == "packs":
            p.add_argument("--pack", nargs="*")
            p.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    t = time.perf_counter()
    args.run(args)
    print(f"wall time {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()
