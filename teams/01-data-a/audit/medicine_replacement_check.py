"""Reproduce the harm, speed and per-document checks of the medicines proposed in an audit report.

A report adds a medicine as a card with a python block that has two lines, `REPLACES = "<old rule id>"` and `RULE_ID = "<new rule id>"`
(the form of PR #71, #75, #76 and the code-go report). This script finds those cards in the report, finds the rule each replaces in the
spec sheet, and runs the project's own checks on "the sheet as committed" against "the sheet with the cards swapped in". Nothing is
written into the repository: everything goes to --work.

Steps, in this order (all need --report):
  prepare   copy the sheet, swap each card's code block for the report's (with the old rule's ID and STATUS, so the diff shows only
            the real change), build clean.py before and after, write <work>/specv1_1_changes.diff, and check that "before" is
            byte for byte the committed clean.py.
  harm      harm_test.py on the gold set of every pack in each card's scope (ORIGIN, or SCOPE, or SPEC_PACKS), loading the card
            straight from the report. Needs --packs.
  docs      run the whole cleaner before and after on those packs; list the documents that differ and whether the only change is
            deleted lines. Needs --packs.
  stress    time the old and the new rule on worst-case texts: a built-in set (one huge line, many short lines, thousands of blocks)
            and the cases of --stress-file, if given.
  speed     the project's speed_compare.sh with the right options: whole-cleaner gold harm before against after, documents changed,
            per-rule profile and docs/s. About 10 minutes. Run it alone, in a plain terminal: it pins to CPU 0 and timings vary by
            about 5%. Needs --packs.

  python -I medicine_replacement_check.py prepare --report <report.md>
  python -I medicine_replacement_check.py harm    --report <report.md> --packs <folder of <pack_id>.parquet>
  ... docs / stress / speed the same way

Options: --sheet (default spec_v1_1/sheets/specv1_1.md), --builder (default spec_v1_1/build/build_clean.py), --committed-clean (default
spec_v1_1/clean.py), --work (default: ../scratchpad/medicine-replacement next to the repository), --include-all (speed: every pack in
--packs, not only the original 13), --stress-file.

--stress-file is a python file with `CASES = [("<old rule id or *>", "<name>", "<text, or a function that returns it>"), ...]`.

In `speed`, the output sha256 differs on the packs where a medicine changes documents. That is expected: speed_compare.sh tests for a
pure speed-up and prints RESULT: FAIL. What must hold is that the sha256 of every other pack and every gold result is the same.

The script runs the code of the cards in the report (so does harm_test.py): read the report first.
"""
import argparse
import difflib
import importlib.util
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent
REPO = TEAM.parents[1]
FENCE = re.compile(r"^```python\n(.*?)^```", re.S | re.M)
ORIGINAL_13 = ["code-cpp", "code-go", "code-java", "code-js-ts", "code-python", "code-rust", "en-dclm", "en-fineweb-edu",
               "en-nemotroncc", "en-nemotroncc-synth", "math-finemath", "math-megamath", "sci-finepdfs"]
GOLD_DIRS = [TEAM / "gold", REPO / "teams" / "02-data-b" / "gold"]


class Card:
    def __init__(self, new_id, old_id, code):
        self.new_id, self.old_id, self.code = new_id, old_id, code
        ns = {}
        exec(compile(code, new_id, "exec"), ns)
        self.ns = ns
        self.packs = scope_packs(ns, old_id)
        langs = sorted(ns.get("LANGS") or [])
        self.lang = langs[0] if langs else "en"


def scope_packs(ns, old_id):
    """The packs a card runs on: its ORIGIN, else its SCOPE (for the old rule), else SPEC_PACKS."""
    origin = ns.get("ORIGIN")
    if isinstance(origin, str) and origin.strip():
        return [p.strip() for p in origin.split(",") if p.strip()]
    scope = ns.get("SCOPE")
    if isinstance(scope, dict) and scope:
        values = [scope[old_id]] if old_id in scope else list(scope.values())
        return sorted(set().union(*[set(v) for v in values]))
    return sorted(ns.get("SPEC_PACKS") or [])


def value(block, name):
    m = re.search(rf'^{name} = "([^"]*)"', block, re.M)
    return m.group(1) if m else None


def report_cards(report):
    cards = []
    for m in FENCE.finditer(Path(report).read_text(encoding="utf-8")):
        old, new = value(m.group(1), "REPLACES"), value(m.group(1), "RULE_ID")
        if old and new:
            cards.append(Card(new, old, m.group(1)))
    if not cards:
        sys.exit(f"no card with REPLACES and RULE_ID lines in {report}")
    return cards


def sheet_block(sheet, rule_id):
    """The one python block of the sheet whose RULE_ID is rule_id: its regex match."""
    hits = [m for m in FENCE.finditer(sheet) if re.search(rf'^RULE_ID = "{re.escape(rule_id)}"$', m.group(1), re.M)]
    if len(hits) != 1:
        sys.exit(f"{rule_id}: {len(hits)} blocks with this RULE_ID in the sheet, expected 1")
    return hits[0]


def in_place(card, old_block):
    """The card's code with the replaced rule's ID and STATUS, so the sheet diff shows only the change."""
    code = re.sub(rf'^REPLACES = "{re.escape(card.old_id)}"\n', "", card.code, flags=re.M)
    code = code.replace(f'RULE_ID = "{card.new_id}"', f'RULE_ID = "{card.old_id}"', 1)
    status = value(old_block, "STATUS")
    if status and value(code, "STATUS"):
        code = re.sub(r'^STATUS = "[^"]*"', f'STATUS = "{status}"', code, count=1, flags=re.M)
    # the sheet builder reads SPEC_PACKS as a literal set: write the value of a computed one (SPEC_PACKS = SCOPE["..."]) out
    if re.search(r"^SPEC_PACKS = (?!\{)", code, re.M) and card.ns.get("SPEC_PACKS"):
        code = re.sub(r"^SPEC_PACKS = .*$", "SPEC_PACKS = " + repr(set(card.ns["SPEC_PACKS"])), code, count=1, flags=re.M)
    return code


def swapped_sheet(sheet, cards):
    for card in cards:
        m = sheet_block(sheet, card.old_id)
        sheet = sheet[:m.start(1)] + in_place(card, m.group(1)) + sheet[m.end(1):]
    return sheet


def run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], check=False, **kw)


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def packs_of(cards):
    seen = []
    for c in cards:
        seen += [p for p in c.packs if p not in seen]
    return seen


def prepare(a, cards):
    sheet = Path(a.sheet).read_text(encoding="utf-8")
    for side in ("before", "after"):
        (a.work / side).mkdir(parents=True, exist_ok=True)
    after = swapped_sheet(sheet, cards)
    # both copies keep the sheet's file name: it is written into clean.py
    (a.work / "before" / Path(a.sheet).name).write_text(sheet, encoding="utf-8", newline="\n")
    (a.work / "after" / Path(a.sheet).name).write_text(after, encoding="utf-8", newline="\n")
    diff = "".join(difflib.unified_diff(sheet.splitlines(True), after.splitlines(True), "sheet (committed)",
                                        "sheet (with the cards)", n=3))
    (a.work / "sheet_changes.diff").write_text(diff, encoding="utf-8", newline="\n")
    for side in ("before", "after"):
        r = run([sys.executable, "-I", a.builder, "--sheet", a.work / side / Path(a.sheet).name, "--out", a.work / f"clean_{side}.py"],
                capture_output=True, text=True)
        if r.returncode:
            sys.exit("the sheet builder rejected the sheet " + ("with the cards swapped in" if side == "after" else "as committed") + ":\n"
                     + "\n".join((r.stdout + r.stderr).strip().splitlines()[-3:])
                     + "\nA card that is narrower in scope than the rule it replaces, or that the builder cannot read, cannot be built "
                     "in place. harm and stress do not need the build and still work.")
    same = (a.work / "clean_before.py").read_bytes() == Path(a.committed_clean).read_bytes()
    changed = sum(1 for l in diff.splitlines() if l[:1] in "+-" and l[:3] not in ("+++", "---"))
    print(f"cards: " + ", ".join(f"{c.new_id} replaces {c.old_id} (packs: {' '.join(c.packs) or 'none found'})" for c in cards))
    print(f"sheet diff: {changed} changed lines, {len(re.findall(r'^@@', diff, re.M))} hunks -> {a.work / 'sheet_changes.diff'}")
    print(f"clean_before.py is the committed {Path(a.committed_clean).name}: {'yes' if same else 'NO'}")
    print(f"clean_after.py: {a.work / 'clean_after.py'}")
    return 0 if same else 1


def gold_file(pack):
    for d in GOLD_DIRS:
        if (d / f"{pack}_gold.csv").exists():
            return d / f"{pack}_gold.csv"


def harm(a, cards):
    bad = 0
    for card in cards:
        for pack in card.packs:
            gold, parquet = gold_file(pack), Path(a.packs) / f"{pack}.parquet"
            if not gold or not parquet.exists():
                print(f"{pack:22} {card.new_id}: skipped ({'no gold file' if not gold else 'no ' + parquet.name + ' in --packs'})")
                continue
            r = run([sys.executable, "-I", TEAM / "harmtest" / "harm_test.py", "--pack", parquet, "--gold", gold,
                     "--rules", a.report, "--rule", card.new_id], capture_output=True, text=True)
            lines = {l.split(":")[0]: l.split(":", 1)[1].strip() for l in r.stdout.splitlines() if ":" in l}
            print(f"{pack:22} {card.new_id}: changed {lines.get('documents changed')}, dropped {lines.get('documents dropped')}, "
                  f"{lines.get('limit')} -> {lines.get('result', 'ERROR ' + r.stderr.strip()[-100:])}")
            bad += r.returncode != 0
    return 1 if bad else 0


def docs(a, cards):
    import pyarrow.parquet as pq
    before, after = load(a.work / "clean_before.py", "clean_before"), load(a.work / "clean_after.py", "clean_after")
    for pack in packs_of(cards):
        parquet = Path(a.packs) / f"{pack}.parquet"
        if not parquet.exists():
            print(f"{pack}: skipped, no {parquet.name} in --packs")
            continue
        t = pq.read_table(parquet, columns=["doc_id", "text", "url", "lang_label", "pack_id"]).to_pydict()
        rows = []
        for i in range(len(t["doc_id"])):
            doc = {k: t[k][i] for k in t}
            b, n = before.clean(dict(doc)), after.clean(dict(doc))
            if b != n:
                bl, nl = b[1].split("\n"), n[1].split("\n")
                ops = [o for o in difflib.SequenceMatcher(None, bl, nl, autojunk=False).get_opcodes() if o[0] != "equal"]
                kind = "only lines deleted" if all(o[0] == "delete" for o in ops) and b[0] != "drop" and n[0] != "drop" \
                    else "text changed in place" if b[0] == n[0] == "fix" or b[0] == n[0] == "keep" else "action changed"
                rows.append((doc["doc_id"], b[0], n[0], kind))
        print(f"{pack}: {len(t['doc_id'])} docs, {len(rows)} differ")
        for r in rows:
            print(f"    {r[0]}  {r[1]} -> {r[2]}  {r[3]}")
    return 0


def builtin_cases():
    return [("*", "one 2 MB line, no newline", "word " * 400000),
            ("*", "200,000 short lines", "x = 1\n" * 200000),
            ("*", "20,000 comment blocks separated by blank lines", "".join(f"// note {i}\n\n" for i in range(20000))),
            ("*", "20,000 short paragraphs", "".join(f"Paragraph {i}. A few words.\n\n" for i in range(20000))),
            ("*", "2 MB of one repeated character", "a" * 2000000)]


def stress(a, cards):
    cases = builtin_cases()
    if a.stress_file:
        cases += list(load(a.stress_file, "stress_cases").CASES)
    sheet = Path(a.sheet).read_text(encoding="utf-8")

    def best(code, text, card, n=5):
        ns = {}
        exec(compile(code, "card", "exec"), ns)
        doc = {"doc_id": "x", "text": text, "url": "src/file", "lang_label": card.lang, "pack_id": (card.packs or ["x"])[0]}
        times = []
        for _ in range(n):
            s = time.perf_counter()
            ns["medicine"](doc)
            times.append(time.perf_counter() - s)
        return min(times) * 1000
    print(f"{'rule':11} {'case':58} {'old ms':>9} {'new ms':>9}")
    for card in cards:
        old = sheet_block(sheet, card.old_id).group(1)
        for rule, name, text in cases:
            if rule not in ("*", card.old_id):
                continue
            text = text() if callable(text) else text
            o, n = best(old, text, card), best(card.code, text, card)
            print(f"{card.old_id:11} {name:58} {o:9.2f} {n:9.2f}" + ("   <-- more than 5x slower" if n > 100 and n > 5 * o else ""))
    return 0


def speed(a, cards):
    folder = a.work / "packs"
    folder.mkdir(parents=True, exist_ok=True)
    available = sorted(p.stem for p in Path(a.packs).glob("*.parquet"))
    wanted = available if a.include_all else [p for p in ORIGINAL_13 if p in available]
    scope = [p for p in packs_of(cards) if p in available]
    if not scope:
        sys.exit("none of the packs in the cards' scope is in --packs")
    for p in wanted:
        link = folder / f"{p}.parquet"
        if not link.exists():
            link.symlink_to((Path(a.packs) / f"{p}.parquet").resolve())
    base = [TEAM / "oneoff" / "speed_compare.sh", "--packs", folder, "--work", a.work / "speed",
            "--rules", ",".join(c.old_id for c in cards), "--before-clean", a.work / "clean_before.py",
            "--after-clean", a.work / "clean_after.py", "--sheet", a.work / "after" / Path(a.sheet).name,
            "--profile-packs", " ".join(scope[:2]), "--count-packs", " ".join(scope)]
    if a.include_all:
        base.append("--include-all")
    status = 0
    for stage in (["trees"], ["harm", "count"], ["profile", "speed"]):   # timing steps last, and alone
        status |= run(base + stage).returncode != 0   # FAIL is expected (see the top of this file): run every step anyway
    return status


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["prepare", "harm", "docs", "stress", "speed"])
    ap.add_argument("--report", type=Path, required=True, help="the audit report that holds the cards")
    ap.add_argument("--sheet", type=Path, default=TEAM / "spec_v1_1" / "sheets" / "specv1_1.md")
    ap.add_argument("--builder", type=Path, default=TEAM / "spec_v1_1" / "build" / "build_clean.py")
    ap.add_argument("--committed-clean", type=Path, default=TEAM / "spec_v1_1" / "clean.py")
    ap.add_argument("--packs", type=Path, default=REPO.parent / "week1-packs", help="folder of <pack_id>.parquet (default ../week1-packs)")
    ap.add_argument("--work", type=Path, default=REPO.parent / "scratchpad" / "medicine-replacement")
    ap.add_argument("--include-all", action="store_true", help="speed: every pack in --packs (Indic gold must be copied in, see speed_handoff.md)")
    ap.add_argument("--stress-file", type=Path, help="python file with CASES = [(old rule id or '*', name, text or function), ...]")
    a = ap.parse_args(argv)
    a.work.mkdir(parents=True, exist_ok=True)
    cards = report_cards(a.report)
    if a.step in ("docs", "speed") and not (a.work / "clean_after.py").exists():
        sys.exit("run the prepare step first")
    return {"prepare": prepare, "harm": harm, "docs": docs, "stress": stress, "speed": speed}[a.step](a, cards)


if __name__ == "__main__":
    sys.exit(main())
