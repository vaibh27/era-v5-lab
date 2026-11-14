"""Rule report: run every rule of one pack, and compare the rules for each code.

Usage:
    python -I rule_report.py --pack-id <pack_id> --pack-file <pack.parquet> [--gold <pack_gold.csv>]

    --pack-id    the pack, for example code-rust. The rules are the active ones in rules/registry.json
    --pack-file  the pack Parquet file (all of its documents are read)
    --gold       the gold CSV. Without it the harm test columns say n/a and no rule is ruled out by the harm test
    --limit      the failing limit in percent of the gold set (default 1), as in harm_test.py
    --out        the report file (default: ../rules/<pack_id>/REPORT.md)

For each rule the report gives the harm test result (the same numbers as harm_test.py) and how the rule does on
the label sheets of the pack (diagnosis/submissions/<pack_id>/*/labels.csv, all annotators together):
    sick rows  the rows labelled with the rule's code (verdict fix or drop)
    hit        sick rows that the rule changes or drops
    extra      rows labelled with something else (or clean) that the rule changes or drops
    unlabelled rows that nobody has labelled yet that the rule changes or drops
    missed     sick rows that the rule leaves alone
A hit means "the rule touches a row the annotators called sick". It does not check that the new text is right.

Where two or more rules have the same code it advises which to keep. A rule that does nothing also passes the
harm test, so the advice is not "the smallest change". The rules that pass the harm test are taken from the most
sick rows fixed downwards. A rule is kept when it fixes sick rows that the rules kept before it do not fix. So
two rules that fix different cases are both kept. This is advice for a person to check, not a decision.

The report names doc counts only: no document text and no doc_ids.

The rule code is run, as in harm_test.py: it is the annotators' code, so read it first.
"""
import argparse
import csv
import json
import sys
from fractions import Fraction
from pathlib import Path

import pyarrow.parquet as pq

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I does not put the script folder on the path
import harm_test  # noqa: E402

REPO = HERE.parents[2]
DEFAULT_SUBMISSIONS = REPO / "diagnosis" / "submissions"
MIN_CASES = 5  # DISEASES.md: a medicine is needed for a disease found more than five times


# --- reading -----------------------------------------------------------------------------------------------------

def read_labels(pack_dir: Path):
    """Return ({code: set of sick doc_ids}, {handle: number of label rows}, set of all labelled doc_ids)."""
    sick, rows, labelled = {}, {}, set()
    for f in sorted(pack_dir.glob("*/labels.csv")):
        try:
            with open(f, newline="", encoding="utf-8") as fh:
                reader = list(csv.DictReader(fh))
        except (OSError, UnicodeDecodeError) as e:
            print(f"warning: cannot read {f}: {e}", file=sys.stderr)
            continue
        rows[f.parent.name] = len(reader)
        for r in reader:
            labelled.add((r.get("doc_id") or "").strip())
            if (r.get("verdict") or "").strip() == "clean":
                continue
            for code in (r.get("diseases") or "").split(";"):
                if code.strip():
                    sick.setdefault(code.strip().upper(), set()).add((r.get("doc_id") or "").strip())
    return sick, rows, labelled


def read_all_texts(path: Path) -> dict:
    try:
        table = pq.read_table(path, columns=["doc_id", "text"])
    except (OSError, KeyError, ValueError) as e:
        raise harm_test.Fatal(f"cannot read pack {path}: {e}")
    return {d: (t or "") for d, t in zip(table["doc_id"].to_pylist(), table["text"].to_pylist())}


# --- measuring ---------------------------------------------------------------------------------------------------

def observed_action(changed: set, dropped: set) -> str:
    return "fix+drop" if changed and dropped else "drop" if dropped else "fix" if changed else "none"


def measure(entry: dict, rule, texts: dict, gold: list, sick: set, labelled: set, limit: Fraction, timeout: float) -> dict:
    """Run one rule on the whole pack. `rule` is None (with entry['load_error'] set) if it could not be loaded."""
    m = dict(entry, harm="n/a", pct=None, gold_changed=None, gold_dropped=None, sick=len(sick),
             hits=set(), extra=0, unlabelled=0, missed=len(sick), errors=0, observed="none")
    if rule is None:
        m.update(errors=len(texts), harm="ERROR")
        return m
    changed, dropped, errors = harm_test.run_rule(rule, texts, timeout)
    changed, dropped = set(changed), set(dropped)
    touched = changed | dropped
    m.update(observed=observed_action(changed, dropped), errors=len(errors), hits=touched & sick)
    m.update(extra=len((touched & labelled) - sick), unlabelled=len(touched - labelled), missed=len(sick - touched))
    if gold:
        g = set(gold)
        m.update(gold_changed=len(changed & g), gold_dropped=len(dropped & g))
        m["pct"], passed = harm_test.harm_percent(m["gold_changed"], m["gold_dropped"], len(g), limit)
        m["harm"] = "ERROR" if any(d in g for d, _ in errors) else "PASS" if passed else "FAIL"
    return m


def recommend(rules: list, n_sick: int) -> dict:
    """Advice for the rules of one code: {rule id: (KEEP or DROP, reason)}. Empty if there is nothing to compare.

    Each rule is a dict with id, harm (PASS, FAIL, ERROR or n/a), errors, hits (a set) and pct.
    """
    if n_sick == 0:
        return {}
    advice, kept, covered = {}, [], set()
    eligible = []
    for r in rules:
        if r["harm"] in ("FAIL", "ERROR") or r["errors"]:
            why = (f"fails the harm test ({float(r['pct']):.2f}%)" if r["harm"] == "FAIL"
                   else "the rule raised errors on some documents")
            advice[r["id"]] = ("DROP", why)
        else:
            eligible.append(r)
    no_gold = " (harm test not run)" if any(r["harm"] == "n/a" for r in eligible) else ""
    for r in sorted(eligible, key=lambda r: (-len(r["hits"]), r["id"])):
        new = r["hits"] - covered
        if not r["hits"]:
            advice[r["id"]] = ("DROP", f"touches none of the {n_sick} sick rows: a rule that does nothing also passes" + no_gold)
        elif not kept:
            advice[r["id"]] = ("KEEP", f"passes and fixes the most sick rows ({len(r['hits'])} of {n_sick})" + no_gold)
        elif new:
            advice[r["id"]] = ("KEEP", f"fixes {len(new)} sick rows that {', '.join(kept)} does not" + no_gold)
        else:
            advice[r["id"]] = ("DROP", f"its {len(r['hits'])} sick rows are all fixed by {', '.join(kept)}" + no_gold)
        if advice[r["id"]][0] == "KEEP":
            kept.append(r["id"])
            covered |= r["hits"]
    return advice


# --- the report --------------------------------------------------------------------------------------------------

def cell(value, fmt=str):
    return "-" if value is None else fmt(value)


def render(pack: str, pack_file: Path, n_docs: int, gold_file, n_gold: int, limit, label_rows: dict, ms: list,
           advice: dict, flags: list, withdrawn: int) -> str:
    gold_line = f"{gold_file.name} ({n_gold} documents), limit {limit}%" if gold_file else "none: the harm test was not run"
    labels = ", ".join(f"{h} ({n} rows)" for h, n in label_rows.items()) or "none found"
    out = [f"# Rule report: {pack}", "",
           f"- Pack file: {pack_file.name} ({n_docs} documents)",
           f"- Gold set: {gold_line}",
           f"- Label sheets: {labels}",
           f"- Rules in this report: {len(ms)} active" + (f", {withdrawn} withdrawn (not run)" if withdrawn else ""), "",
           "## Rules", "",
           "| ID | Code | Annotator | Action: written / seen | Gold changed | Gold dropped | % of gold | Harm test "
           "| Sick rows | Hit | Extra | Unlabelled | Missed | Errors |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in ms:
        out.append(f"| {m['id']} | {m['code']} | {m['handle']} | {m['action'] or '?'} / {m['observed']} "
                   f"| {cell(m['gold_changed'])} | {cell(m['gold_dropped'])} | {cell(m['pct'], lambda p: f'{float(p):.2f}%')} "
                   f"| {m['harm']} | {m['sick']} | {len(m['hits'])} | {m['extra']} | {m['unlabelled']} | {m['missed']} | {m['errors']} |")
    out += ["", "Sick rows: labelled with the code (verdict fix or drop), by any annotator. Hit: sick rows the rule changes or "
            "drops. Extra: rows labelled with something else that it changes or drops. Unlabelled: rows nobody has labelled "
            "that it changes or drops. Missed: sick rows it leaves alone. A hit is not a check that the new text is right."]
    out += ["", "## Rules with the same code", ""]
    multi = sorted({m["code"] for m in ms if sum(x["code"] == m["code"] for x in ms) > 1})
    if not multi:
        out.append("No code has more than one rule.")
    for code in multi:
        out += [f"### {code}", ""]
        if not any(m["id"] in advice for m in ms if m["code"] == code):
            out += [f"No row in the label sheets carries {code}, so the rules cannot be compared.", ""]
            continue
        out += ["| ID | Advice | Why |", "|---|---|---|"]
        for m in (m for m in ms if m["code"] == code):
            a, why = advice[m["id"]]
            out.append(f"| {m['id']} | {a} | {why} |")
        out.append("")
    out += ["", "## Flags", ""] + ([f"- {f}" for f in flags] or ["None."])
    return "\n".join(out) + "\n"


def find_flags(ms: list, sick: dict, submissions_pack: Path, label_rows: dict) -> list:
    flags = []
    for m in ms:
        if m.get("load_error"):
            flags.append(f"{m['id']}: {m['load_error']}")
        elif m["errors"]:
            flags.append(f"{m['id']}: the rule raised an error on {m['errors']} documents")
        if m["observed"] == "none" and not m["errors"]:
            flags.append(f"{m['id']}: changes no document of the pack at all")
        elif m["sick"] and not m["hits"] and not m["errors"]:
            flags.append(f"{m['id']}: touches none of the {m['sick']} rows labelled {m['code']}")
        elif (m["action"] == "fix" and "drop" in m["observed"]) or (m["action"] == "drop" and "fix" in m["observed"]):
            flags.append(f"{m['id']}: the medicine.md says {m['action']}, but the rule does {m['observed']}")
        elif m["action"] is None:
            flags.append(f"{m['id']}: the medicine.md has no clear '**Action:**' line (fix or drop)")
    for code in sorted({m["code"] for m in ms}):
        actions = {m["action"] for m in ms if m["code"] == code and m["action"]}
        if len(actions) > 1:
            flags.append(f"{code}: the rules disagree on the action ({', '.join(sorted(actions))})")
    handles = sorted({m["handle"] for m in ms})
    if len(handles) > 1:
        for code in sorted({m["code"] for m in ms}):
            who = sorted({m["handle"] for m in ms if m["code"] == code})
            if len(who) == 1:
                flags.append(f"{code}: only {who[0]} has a medicine for it")
    for d in sorted(submissions_pack.glob("*/medicine.md")):
        if d.parent.name not in handles:
            flags.append(f"{d.parent.name}: has a medicine.md but no rule in the registry. Run medicine_extract.py and read its notes")
    for code, ids in sorted(sick.items()):
        if len(ids) > MIN_CASES and code not in {m["code"] for m in ms}:
            flags.append(f"{code}: {len(ids)} sick rows in the labels and no rule for it")
    return flags


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run every rule of one pack and compare the rules for each code.")
    ap.add_argument("--pack-id", required=True)
    ap.add_argument("--pack-file", required=True, type=Path)
    ap.add_argument("--gold", type=Path)
    ap.add_argument("--limit", default="1", help="failing limit in percent of the gold set (default 1)")
    ap.add_argument("--registry", type=Path, default=harm_test.DEFAULT_REGISTRY)
    ap.add_argument("--submissions", type=Path, default=DEFAULT_SUBMISSIONS)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds per document (default 5)")
    args = ap.parse_args(argv)

    try:
        limit = Fraction(args.limit)
        if limit < 0:
            raise ValueError
    except ValueError:
        print(f"error: --limit must be a number of percent, not {args.limit!r}", file=sys.stderr)
        return 2
    try:
        registry = json.loads(args.registry.read_text(encoding="utf-8"))["rules"]
        entries = [e for e in registry if e["pack"] == args.pack_id]
        active = sorted((e for e in entries if not e.get("withdrawn")), key=lambda e: e["id"])
        if not active:
            raise harm_test.Fatal(f"no active rules for {args.pack_id} in {args.registry}. Run medicine_extract.py first")
        texts = read_all_texts(args.pack_file)
        gold = harm_test.read_gold(args.gold) if args.gold else []
        missing = [i for i in gold if i not in texts]
        if missing:
            raise harm_test.Fatal(f"{len(missing)} gold doc_ids are not in the pack, for example {missing[0]}")
    except (OSError, ValueError, KeyError) as e:
        print(f"error: cannot read the registry {args.registry}: {e}", file=sys.stderr)
        return 2
    except harm_test.Fatal as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    pack_dir = args.submissions / args.pack_id
    sick, label_rows, labelled = read_labels(pack_dir)
    print("warning: running the rule code of the annotators", file=sys.stderr)

    modules, ms = {}, []
    for e in active:
        path = args.registry.parent / e["file"]
        if path not in modules:
            try:
                modules[path] = harm_test.load_rules(path)[0]
            except harm_test.Fatal as err:
                modules[path] = str(err)
        ns, rule, e = modules[path], None, dict(e)
        if isinstance(ns, dict):
            try:
                rule = harm_test.find_rule(ns, e["id"])
            except harm_test.Fatal as err:
                e["load_error"] = str(err)
        else:
            e["load_error"] = ns
        ms.append(measure(e, rule, texts, gold, sick.get(e["code"], set()), labelled, limit, args.timeout))

    advice = {}
    for code in {m["code"] for m in ms}:
        group = [m for m in ms if m["code"] == code]
        if len(group) > 1:
            advice.update(recommend(group, len(sick.get(code, ()))))
    flags = find_flags(ms, sick, pack_dir, label_rows)
    text = render(args.pack_id, args.pack_file, len(texts), args.gold, len(gold), args.limit, label_rows, ms, advice,
                  flags, len(entries) - len(active))
    out = args.out or args.registry.parent / args.pack_id / "REPORT.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")

    print(f"{args.pack_id}: {len(ms)} rules, " + ", ".join(
        f"{k} {sum(m['harm'] == k for m in ms)}" for k in ("PASS", "FAIL", "ERROR", "n/a")) + f", {len(flags)} flags")
    for rid, (a, _) in sorted((m["id"], advice[m["id"]]) for m in ms if m["id"] in advice):
        print(f"  {rid}: {a}")
    print(f"report: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
