"""Medicine check: does each gold file have enough medicines, and do the annotators' medicines clash?

Usage:
    python -I medicine_conflicts.py [--packs en-] [--medicines <folder>] [--gold-dir <dir>] [--out <file.md>]

    --medicines  the medicine sheets, as <pack_id>/<handle>/medicine.md (default: ../medicines/submissions),
                 or as <pack_id>-medicine/<handle>-medicine.md
    --gold-dir   the gold files, <pack_id>_gold.csv (default: ../gold). The report has a section for each one
    --packs      only the packs whose id starts with this, for example en-
    --registry   the rule registry (default: ../rules/registry.json)
    --sheet      take the rules from a standard sheet instead (one RULE_ID block per rule, with ORIGIN and AUTHOR),
                 for example ../spec_v1/sheets/medicine_english.md
    --out        the report (default: ../rules/MEDICINE_CHECK.md)

Nothing is run: the sheets and the rules files are only read (with ast), so this is safe before anyone has read
the annotators' code. The harm test of each rule is harm_test.py's job, and the choice between rules that pass
it is rule_report.py's.

For each gold file:
    NO MEDICINE     no medicine sheet, or only blank templates
    NOT ENOUGH      no rule that can run: none was extracted, or each one needs a name that its code never defines
    ONE ANNOTATOR   only one annotator wrote medicines, so there is nothing to compare them with
    NOT EXTRACTED   a sheet has medicine sections but no rule in the registry (run medicine_extract.py)
    CANNOT RUN      a rule needs names that its code never defines; it fails with NameError
    NO RULE         a code that a sheet says was found more than five times has no rule that can run
    BLANK SHEET     an annotator's sheet is the blank template (when other medicines cover the pack)
A sheet rule whose ORIGIN lists several packs (a rule written for all of them) counts for each of those packs.
For each code with two or more medicines in one pack:
    ACTIONS DIFFER  the medicines disagree on the action (fix, drop, keep)
    RULE VS HAND    one annotator wrote a rule, another edits named rows by hand
    CHOOSE          two or more rules can run: rule_report.py says which to keep (guideline 2)
The medicines of each such code are listed side by side: a clash in what they do (cut a cut-off text, or join it
to the next chunk) cannot be found by a program, so a person compares them.

The last section lists the codes that several packs of one language treat (en-, code-, ...), because a rule that
runs on every pack of the language must suit all of them.
"""
import argparse
import ast
import builtins
import csv
import json
import re
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I does not put the script folder on the path
import medicine_extract  # noqa: E402

DEFAULT_MEDICINES = HERE.parent / "medicines" / "submissions"
DEFAULT_GOLD = HERE.parent / "gold"
DEFAULT_OUT = HERE.parent / "rules" / "MEDICINE_CHECK.md"
MIN_CASES = 5                                       # DISEASES.md: a medicine for each disease found more than 5 times
TEMPLATE_MARK = "count in your pack, three examples"  # a line of the blank medicine template
CODE = re.compile(r"\b([A-H][1-9])\b")
MEDICINE_LINE = re.compile(r"\*\*Medicine[^*\n]*\*\*:?[ \t]*")
PARAGRAPH_END = re.compile(r"\n[ \t]*(?:\n|```|\d+\.[ \t]+\*\*)")  # a blank line, a code fence, or the next item
FIRST_NUMBER = re.compile(r"\b(\d+)\b")
HAND_LINE = re.compile(r"^[ \t]*(?:#{1,6}|[-*])[ \t]+(.*\brows?\b.*)$", re.M | re.I)  # a hand edit names its rows
QUOTE = re.compile(r"`[^`]*`")


def shorten(text: str, n: int = 110) -> str:
    text = " ".join(QUOTE.sub("`…`", text).split())  # quoted characters may come from a document: leave them out
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


# --- reading the sheets ------------------------------------------------------------------------------------------

def find_sheets(folder: Path) -> dict:
    """{pack_id: {handle: path}} for both layouts."""
    found = {}
    for p in folder.glob("*/*/medicine.md"):
        found.setdefault(p.parent.parent.name, {})[p.parent.name] = p
    for p in folder.glob("*-medicine/*-medicine.md"):
        found.setdefault(p.parent.name[: -len("-medicine")], {})[p.stem[: -len("-medicine")]] = p
    return found


def registry_handles(sheets: dict, team_folder: Path) -> dict:
    """{(pack, handle): the registry's handle}. A sheet in another folder may be saved under another name (dsaha
    for deepjyotisaha); if its text is the same as a sheet in the team's submissions folder, that name is used."""
    team = {}
    for pack, by_handle in find_sheets(team_folder).items():
        for handle, path in by_handle.items():
            team[(pack, path.read_bytes())] = handle
    return {(pack, h): team.get((pack, p.read_bytes()), h) for pack, by_handle in sheets.items() for h, p in by_handle.items()}


def read_sheet(path: Path) -> dict:
    """{blank, medicines, hand}. medicines: the **Disease:** sections. hand: edits of named rows (a sheet without
    sections), one for each heading or bullet line that names a code."""
    src = path.read_text(encoding="utf-8")
    sections, blocks = medicine_extract.read_medicine(src)
    medicines, seen = [], {}
    for start, end, code, action in sections:
        seen[code] = seen.get(code, 0) + 1
        disease = src[start: src.find("\n", start) if "\n" in src[start:] else len(src)]
        count = FIRST_NUMBER.search(disease[disease.find(code) + 2:])
        label = MEDICINE_LINE.search(src, start, end)
        stop = PARAGRAPH_END.search(src, label.end(), end) if label else None
        summary = src[label.end(): stop.start() if stop else end] if label else ""
        medicines.append({"code": code, "n": seen[code], "action": action, "count": int(count.group(1)) if count else None,
                          "summary": shorten(summary)})
    blank = not sections and not blocks and TEMPLATE_MARK in src
    hand = []
    if not sections and not blank:
        for m in HAND_LINE.finditer(src):
            for code in dict.fromkeys(CODE.findall(m.group(1))):
                hand.append({"code": code, "action": "drop" if re.search(r"\bdrop", m.group(1), re.I) else "fix",
                             "line": src.count("\n", 0, m.start()) + 1, "summary": shorten(m.group(1))})
    return {"blank": blank, "medicines": medicines, "hand": hand}


# --- can a rule run? (static: the code is never run) -------------------------------------------------------------

def _bound(node) -> set:
    names = set()
    for n in ast.walk(node):
        if isinstance(n, ast.Name) and not isinstance(n.ctx, ast.Load):
            names.add(n.id)
        elif isinstance(n, ast.arg):
            names.add(n.arg)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and n is not node:
            names.add(n.name)
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            names.update((a.asname or a.name).split(".")[0] for a in n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            names.add(n.name)
    return names


def _free(node) -> set:
    """The names that a function or a constant's value uses but does not bind itself."""
    used = {n.id for n in ast.walk(node) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return used - _bound(node)


class RulesFile:
    """A generated rules file (or one block of a standard sheet), read with ast: which names each rule needs that
    the code never defines."""

    def __init__(self, path: Path, text: str = None):
        self.top, self.defined, self.load_missing, self.by_id = {}, set(dir(builtins)), set(), {}
        try:
            tree = ast.parse(text if text is not None else path.read_text(encoding="utf-8"))
        except (OSError, SyntaxError) as e:
            self.error = f"cannot read {path.name}: {e}"
            return
        self.error = None
        self._collect(tree.body, guarded=False)
        for node, guarded in self._statements:
            if not guarded and not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.load_missing |= _free(node) - self.defined   # this runs when the file loads
        for node in tree.body:
            if isinstance(node, ast.Assign) and [t.id for t in node.targets if isinstance(t, ast.Name)] == ["RULES_BY_ID"]:
                for k, v in zip(node.value.keys, node.value.values):
                    self.by_id[k.value] = v.id if isinstance(v, ast.Name) else None

    def _collect(self, body, guarded, out=None):
        self._statements = [] if out is None else out
        for node in body:
            self._statements.append((node, guarded))
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                self.top[node.name] = node
            elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
                for t in getattr(node, "targets", [getattr(node, "target", None)]):
                    for n in ast.walk(t):  # A, B = 1, 2 binds both
                        if isinstance(n, ast.Name):
                            self.top[n.id] = node
            elif isinstance(node, ast.Try):
                for part in (node.body, node.orelse, node.finalbody, *[h.body for h in node.handlers]):
                    self._collect(part, True, self._statements)
            elif isinstance(node, ast.If):
                for part in (node.body, node.orelse):
                    self._collect(part, guarded, self._statements)
            self.defined |= _bound(node) if isinstance(node, (ast.Import, ast.ImportFrom)) else set()
        self.defined |= set(self.top)

    def missing(self, rule_id: str, function: str = None):
        """Sorted names the rule needs and the file never defines, or None if the rule cannot be found.
        function: the rule's function, if it is not given by RULES_BY_ID (a sheet block's medicine)."""
        function = function or self.by_id.get(rule_id)
        if self.error or not function or function not in self.top:
            return None
        need, seen, missing = [function], set(), set(self.load_missing)
        while need:
            name = need.pop()
            if name in seen:
                continue
            seen.add(name)
            if name in self.top:
                need.extend(_free(self.top[name]))
            elif name not in self.defined:
                missing.add(name)
        return sorted(missing)


def read_sheet_rules(path: Path) -> list:
    """The rules of a standard sheet (such as spec_v1/sheets/medicine_english.md): one ```python block per rule with
    RULE_ID, CODE, ACTION, ORIGIN (the pack) and AUTHOR (the annotator), and medicine(doc). Each block is read
    with ast, on its own, as harm_test.py runs it. Returned in the form of registry entries, with "missing" set."""
    rules, seen = [], {}
    for m in medicine_extract.FENCE.finditer(path.read_text(encoding="utf-8")):
        text = textwrap.dedent(m.group(2))
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        consts = {}
        for node in tree.body:
            if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                try:
                    consts[node.targets[0].id] = ast.literal_eval(node.value)
                except ValueError:
                    pass
        if not consts.get("RULE_ID"):
            continue
        key = (consts.get("ORIGIN"), consts.get("AUTHOR"), consts.get("CODE"))
        seen[key] = seen.get(key, 0) + 1
        rules.append({"id": consts["RULE_ID"], "pack": consts.get("ORIGIN"), "handle": consts.get("AUTHOR"),
                      "code": consts.get("CODE"), "n": seen[key], "action": consts.get("ACTION"), "file": path.name,
                      "missing": RulesFile(path, text).missing(consts["RULE_ID"], "medicine")})
    return rules


# --- the check ---------------------------------------------------------------------------------------------------

def read_gold_info(gold_dir: Path, pack: str) -> dict:
    gold = gold_dir / f"{pack}_gold.csv"
    info = {"file": gold.name, "n": None, "status": None}
    if gold.exists():
        with open(gold, newline="", encoding="utf-8") as f:
            info["n"] = len({r["doc_id"].strip() for r in csv.DictReader(f) if (r.get("doc_id") or "").strip()})
    cov = gold_dir / "gold_coverage.csv"
    if cov.exists():
        with open(cov, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("pack_id") == pack:
                    info["status"] = f"{r.get('gold_docs')} of the target {r.get('target')} ({r.get('status')})"
    return info


def packs_of(entry) -> list:
    """The packs a rule is for: its "pack" (a sheet's ORIGIN may list several, separated by commas)."""
    return [p.strip() for p in (entry.get("pack") or "").split(",") if p.strip()]


def check_pack(pack: str, sheets: dict, registry: list, registry_dir: Path, gold: dict, names: dict = None,
               rules_from: str = "the registry") -> dict:
    """All the facts and flags of one pack. sheets: {handle: path}. names: {(pack, handle): registry handle}."""
    files, rows, flags = {}, [], []
    for handle, path in sorted(sheets.items()):
        sheet = read_sheet(path)
        reg_handle = (names or {}).get((pack, handle), handle)
        rules = sorted((e for e in registry if e["pack"] == pack and e["handle"] == reg_handle and not e.get("withdrawn")),
                       key=lambda e: e["id"])
        for e in rules:
            if "missing" in e:  # a sheet rule: checked when the sheet was read
                continue
            if e["file"] not in files:
                files[e["file"]] = RulesFile(registry_dir / e["file"])
            e["missing"] = files[e["file"]].missing(e["id"])
        by_key = {(e["code"], e.get("n", 1)): e for e in rules}
        for m in sheet["medicines"]:
            e = by_key.get((m["code"], m["n"]))
            rows.append(dict(m, handle=handle, kind="rule" if e else "no rule", rule=e))
        for h in sheet["hand"]:
            rows.append(dict(h, handle=handle, kind=f"hand edit (line {h['line']})", rule=None, count=None))
        status = ("blank template" if sheet["blank"] else f"{len(sheet['medicines'])} medicines" if sheet["medicines"]
                  else "hand edits" if sheet["hand"] else "no medicine sections")
        sheets[handle] = {"status": status, "rules": rules}
        if sheet["medicines"] and not rules:
            flags.append(("NOT EXTRACTED", f"{handle}: the sheet has {len(sheet['medicines'])} medicine sections "
                                           f"but no rule in {rules_from}"))
        for e in rules:
            if e["missing"]:
                flags.append(("CANNOT RUN", f"{e['id']} ({e['code']}, {handle}) needs {', '.join(e['missing'])}, "
                                            "which its code never defines"))

    runnable = [r["rule"] for r in rows if r["rule"] and r["rule"]["missing"] == []]
    # rules written for several packs at once (a sheet rule whose ORIGIN lists this pack among others)
    shared = [e for e in registry if pack in packs_of(e) and len(packs_of(e)) > 1 and e.get("missing") == []]
    writers = sorted({r["handle"] for r in rows})
    if not writers and not shared:
        flags.insert(0, ("NO MEDICINE", "no medicine sheet" if not sheets else "no sheet has a medicine: "
                         + ", ".join(f"{h} ({s['status']})" for h, s in sheets.items())))
    elif not runnable and not shared:
        flags.insert(0, ("NOT ENOUGH", "no rule that can run, so nothing can be tested on this gold set"))
    for h, s in sheets.items():
        if s["status"] == "blank template" and (writers or shared):
            flags.append(("BLANK SHEET", f"{h}: the medicine sheet is the blank template"))
    if len(writers) == 1:
        flags.append(("ONE ANNOTATOR", f"only {writers[0]} wrote medicines"))
    covered = {e["code"] for e in runnable + shared}
    for code in sorted({r["code"] for r in rows if (r["count"] or 0) > MIN_CASES} - covered):
        flags.append(("NO RULE", f"{code}: found more than {MIN_CASES} times "
                                 f"({', '.join(str(r['count']) for r in rows if r['code'] == code and r['count'])}) "
                                 "and no rule that can run"))
    if gold["n"] is not None and gold["n"] < 200:
        flags.append(("SMALL GOLD SET", f"{gold['n']} documents: the limit is 1 document (max(1, 1%)), "
                                        "so a second touched document fails a rule"))

    clashes = []
    for code in sorted({r["code"] for r in rows}):
        group = [r for r in rows if r["code"] == code]
        rules = [r for r in group if r["rule"] and r["rule"]["missing"] == []]
        if len({r["handle"] for r in group}) < 2 and len(rules) < 2:
            continue
        kinds = []
        if len({r["action"] for r in group if r["action"]}) > 1:
            kinds.append("ACTIONS DIFFER")
        if any(r["rule"] for r in group) and any(r["kind"].startswith("hand") for r in group):
            kinds.append("RULE VS HAND")
        if len(rules) > 1:
            kinds.append("CHOOSE")
        clashes.append((code, kinds or ["COMPARE"], group))
    return {"pack": pack, "gold": gold, "sheets": sheets, "rows": rows, "runnable": runnable + shared, "shared": shared,
            "flags": flags,
            "clashes": clashes}


# --- the report --------------------------------------------------------------------------------------------------

def rule_cell(r) -> str:
    e = r["rule"]
    if not e:
        return "-"
    can = "can run" if e["missing"] == [] else "cannot run" if e["missing"] else "not checked"
    return f"{e['id']} ({can})"


def render(results: list, medicines: Path, gold_dir: Path, by_language: dict, rules_from: str = "the registry") -> str:
    out = ["# Medicine check", "",
           f"Medicine sheets: `{medicines}`. Gold files: `{gold_dir}`. Rules: {rules_from}. Made by "
           "`medicine_conflicts.py`; no code was run.",
           "\"Can run\" means that the rule's code defines every name it needs. It says nothing about the harm test: "
           "run `harm_test.py` for that.", "",
           "## Summary", "",
           "| Gold file | Gold docs | Sheets | Rules | Rules that can run | Flags |", "|---|---|---|---|---|---|"]
    for res in results:
        g, sheets = res["gold"], res["sheets"]
        n_rules = sum(len(s["rules"]) for s in sheets.values()) + len(res.get("shared", []))
        out.append(f"| {g['file']} | {g['n'] if g['n'] is not None else 'missing'} | {len(sheets)} "
                   f"| {n_rules} | {len(res['runnable'])} | {', '.join(dict.fromkeys(f for f, _ in res['flags'])) or '-'} |")
    for res in results:
        g = res["gold"]
        out += ["", f"## {res['pack']}", "",
                f"- Gold set: {g['file']}, " + (f"{g['n']} documents (1% allows {g['n'] // 100})" if g["n"] is not None
                                                else "not found")
                + (f"; {g['status']}" if g["status"] else ""), ""]
        if res["sheets"]:
            out += ["| Annotator | Sheet | Rules | Can run |", "|---|---|---|---|"]
            for handle, s in res["sheets"].items():
                out.append(f"| {handle} | {s['status']} | {len(s['rules'])} | {sum(e['missing'] == [] for e in s['rules'])} |")
        if res.get("shared"):
            out += ["", f"Rules written for several packs that also run here ({len(res['shared'])}): "
                    + ", ".join(f"`{e['id']}` ({e['code']})" for e in res["shared"]) + "."]
        out += ["", "### Flags", ""] + ([f"- **{f}**: {why}" for f, why in res["flags"]] or ["None."])
        out += ["", "### Medicines with the same code", ""]
        if not res["clashes"]:
            out.append("No code has more than one medicine.")
        for code, kinds, group in res["clashes"]:
            out += [f"#### {code}: {', '.join(kinds)}", "", "| Annotator | Kind | Action | Rule | Medicine |",
                    "|---|---|---|---|---|"]
            out += [f"| {r['handle']} | {r['kind']} | {r['action'] or '?'} | {rule_cell(r)} | {r['summary']} |" for r in group]
            out.append("")
    out += ["", "## Across the packs of one language", ""]
    if not by_language:
        out.append("No code is treated in more than one pack of a language.")
    for (lang, code), group in sorted(by_language.items()):
        actions = {r["action"] for _, r in group if r["action"]}
        out += [f"### {lang}-* {code}" + (": ACTIONS DIFFER" if len(actions) > 1 else ""), "",
                "| Pack | Annotator | Action | Rule | Medicine |", "|---|---|---|---|---|"]
        out += [f"| {p} | {r['handle']} | {r['action'] or '?'} | {rule_cell(r)} | {r['summary']} |" for p, r in group]
        out.append("")
    return "\n".join(out).rstrip() + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Flag gold files without enough medicines, and medicines that clash.")
    ap.add_argument("--medicines", type=Path, default=DEFAULT_MEDICINES)
    ap.add_argument("--gold-dir", type=Path, default=DEFAULT_GOLD)
    ap.add_argument("--packs", default="", help="only packs whose id starts with this, for example en-")
    ap.add_argument("--registry", type=Path, default=medicine_extract.DEFAULT_REGISTRY)
    ap.add_argument("--sheet", type=Path, help="take the rules from this standard sheet instead of the registry")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    try:
        registry = (read_sheet_rules(args.sheet) if args.sheet
                    else json.loads(args.registry.read_text(encoding="utf-8"))["rules"])
    except (OSError, ValueError, KeyError) as e:
        print(f"error: cannot read the rules {args.sheet or args.registry}: {e}", file=sys.stderr)
        return 2
    sheets = find_sheets(args.medicines)
    packs = sorted({g.stem[: -len("_gold")] for g in args.gold_dir.glob("*_gold.csv")} | set(sheets))
    packs = [p for p in packs if p.startswith(args.packs)]
    if not packs:
        print(f"error: no gold file or medicine sheet for packs starting with {args.packs!r}", file=sys.stderr)
        return 2

    names = registry_handles(sheets, DEFAULT_MEDICINES)
    rules_from = f"`{args.sheet.name}`" if args.sheet else "the registry"
    results = [check_pack(p, dict(sheets.get(p, {})), registry, args.registry.parent, read_gold_info(args.gold_dir, p), names,
                          rules_from) for p in packs]
    by_language = {}
    for res in results:
        for r in res["rows"]:
            by_language.setdefault((res["pack"].split("-")[0], r["code"]), []).append((res["pack"], r))
    by_language = {k: v for k, v in by_language.items() if len({p for p, _ in v}) > 1}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(render(results, args.medicines, args.gold_dir, by_language, rules_from), encoding="utf-8")
    for res in results:
        print(f"{res['pack']}: {len(res['runnable'])} rules can run, {len(res['clashes'])} codes with several medicines")
        for f, why in res["flags"]:
            print(f"  {f}: {why}")
    print(f"report: {args.out}")
    return 1 if any(f in ("NO MEDICINE", "NOT ENOUGH") for res in results for f, _ in res["flags"]) else 0


if __name__ == "__main__":
    sys.exit(main())
