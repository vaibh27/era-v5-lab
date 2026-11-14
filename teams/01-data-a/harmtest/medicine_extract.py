"""Medicine extractor: turn one medicine.md into a rules file that harm_test.py can run.

Usage:
    python -I medicine_extract.py <medicine.md> [--out <rules.py>]
    python -I medicine_extract.py --dir <folder>

--dir takes a folder with <pack_id>/<handle>/medicine.md files (for example the folder that fetch_medicines.py
fills) and does all of them in one run, so the registry is set up with one command. The pack and the handle come
from the folder names, so the folder can have any name. The rules files go next to the registry, in
rules/<pack_id>/<handle>.py (--registry moves both). A
medicine.md that gives no rule is listed at the end; the others are still done.

By default the output goes to teams/01-data-a/rules/<pack_id>/<handle>.py, taken from the path
diagnosis/submissions/<pack_id>/<handle>/medicine.md. (Do not write into the submissions folder: pr-guard allows
only labels.csv and medicine.md there.) For any other path, give --out.

The medicine.md is only read, never run. The steps:
  1. Each "**Disease:** <code>" line starts a medicine section. A section ends at the next one, or at an
     "Appendix" heading.
  2. Every ```python block in the file is read with ast:
       code      defines a function. It goes into the rules file as it is.
       support   only imports and UPPER_CASE constants. It goes in too, inside try/except.
       fragment  anything else (a code line that needs a variable from elsewhere, a syntax error). Left out.
  3. For each disease code the rule is the first of:
       - a function named medicine_<code> (for example medicine_b3), or cure_<code>
       - the only public function (no leading underscore) in that section
     Anything else is reported, and nothing is guessed.
  4. The rules file ends with RULES = {"B3": <function>, ...} and RULES_BY_ID = {"DB-B3-001": <function>, ...}.
     A RULES dict written inside a medicine.md is not read: the file always gets its own.
  5. Each rule gets an ID: <team>-<code>-<number>. The team is DA for the Data A packs (English, code, math,
     science) and DB for every other pack (Indic, cadj-*). The number counts up for each team and code. The ID
     goes into rules/registry.json with the pack, the handle and the file. When you run the extractor again, a
     (pack, handle, code) that is already in the registry keeps its ID. A code that has gone from the
     medicine.md is marked "withdrawn" and its ID is never used again.
     A second section for the same code is a second rule with its own ID (the registry keeps its number n).
     The registry also keeps the action the annotator wrote on the "**Action:**" line (fix or drop).
  Then harm_test.py can find the rule by its ID alone:
       python -I harm_test.py --pack <pack.parquet> --gold <gold.csv> --rule DB-B3-001
  A path outside diagnosis/submissions/<pack_id>/<handle>/ has no pack or handle, so it gets no IDs.

Two medicines for one code: the name medicine_<code> (or cure_<code>) can only point to one function, so it is
used for the first section of a code. A later section of the same code needs the only-function rule: put its
function in that section, and give it a name that no earlier rule uses.
"""
import argparse
import ast
import copy
import hashlib
import json
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import NamedTuple

HERE = Path(__file__).resolve().parent

# A fenced python block, possibly indented inside a list item. The closing fence must have the same indent.
FENCE = re.compile(r"^([ \t]*)```(?:python|py)[ \t]*\n(.*?)^\1```[ \t]*$", re.S | re.M)
DISEASE = re.compile(r"\*\*Disease:?\*\*:?[ \t]*`?([A-H][1-9])\b")
APPENDIX = re.compile(r"^#{1,6}[^\n]*appendix", re.I | re.M)
CONST_NAME = re.compile(r"^[A-Z][A-Z0-9_]*$")
ACTION = re.compile(r"\*\*Action:?\*\*:?[ \t]*([^\n]*)")
ACTION_WORD = re.compile(r"\b(fix|drop)\b", re.I)
DEFAULT_REGISTRY = HERE.parent / "rules" / "registry.json"
# The packs Data A owns (English, code, math, science). Data B owns the rest (Indic and cadj-*).
DATA_A_PACKS = frozenset({
    "code-cpp", "code-go", "code-java", "code-js-ts", "code-python", "code-rust",
    "en-dclm", "en-fineweb-edu", "en-nemotroncc", "en-nemotroncc-synth",
    "math-finemath", "math-megamath", "sci-finepdfs",
})


class Block:
    def __init__(self, index, text, pos):
        self.index, self.text, self.pos = index, text, pos
        self.section = None   # index into the sections list, or None
        self.kind = "fragment"
        self.funcs = []       # names of functions defined at the top level of the block
        self._analyse()

    def _analyse(self):
        try:
            tree = ast.parse(self.text)
        except SyntaxError:
            self.kind = "syntax error"
            return
        defining, support_only = False, True
        for node in tree.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.funcs.append(node.name)
                defining = True
            elif isinstance(node, ast.ClassDef):
                defining = True
            elif isinstance(node, ast.Assign) and all(isinstance(t, ast.Name) for t in node.targets):
                names = [t.id for t in node.targets]
                if isinstance(node.value, ast.Lambda):
                    self.funcs.extend(names)
                    defining = True
                elif not all(CONST_NAME.match(n) for n in names):
                    support_only = False
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                pass
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                pass  # a docstring
            else:
                support_only = False
        if defining:
            self.kind = "code"
        elif support_only and tree.body:
            self.kind = "support"


def read_medicine(src: str):
    """Return (sections, blocks). A section is (start, end, code, action); action is "fix", "drop" or None."""
    starts = [(m.start(), m.group(1)) for m in DISEASE.finditer(src)]
    appendix = APPENDIX.search(src)
    cut = appendix.start() if appendix else len(src)
    sections = []
    for i, (pos, code) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(src)
        end = min(end, cut) if pos < cut else end
        line = ACTION.search(src, pos, end)
        word = ACTION_WORD.search(line.group(1)) if line else None  # the first one: "fix the text. No document is dropped."
        sections.append((pos, end, code, word.group(1).lower() if word else None))
    blocks = [Block(i, textwrap.dedent(m.group(2)), m.start()) for i, m in enumerate(FENCE.finditer(src), 1)]
    for b in blocks:
        for si, (start, end, _, _) in enumerate(sections):
            if start <= b.pos < end:
                b.section = si
    return sections, blocks


def resolve(sections, blocks):
    """Return ([rule], [notes]). A rule is {code, n, expr, how, action}; n counts the sections of one code.

    expr is Python that names the rule's function. The name medicine_<code> (or cure_<code>) serves the first
    section of a code only. A later section needs a function of its own, not already used by an earlier rule.
    """
    code_blocks = [b for b in blocks if b.kind == "code"]
    funcs = {n.lower(): n for b in code_blocks for n in b.funcs}
    chosen, notes, seen, claimed = [], [], {}, set()
    for si, (_, _, code, action) in enumerate(sections):
        n = seen[code] = seen.get(code, 0) + 1
        label = code if n == 1 else f"{code} (medicine {n})"
        expr = how = None
        named = [f"{p}_{code.lower()}" for p in ("medicine", "cure") if n == 1 and f"{p}_{code.lower()}" in funcs]
        if named:
            expr, how = funcs[named[0]], f"function {named[0]}"
        else:
            publics = sorted({f for b in code_blocks if b.section == si for f in b.funcs if not f.startswith("_")} - claimed)
            if len(publics) == 1:
                expr, how = publics[0], "the only function in the section"
            elif publics:
                notes.append(f"{label}: several functions in the section ({', '.join(publics)}); "
                             f"name the rule medicine_{code.lower()}")
            else:
                notes.append(f"{label}: no function found for this section")
        if expr:
            chosen.append({"code": code, "n": n, "expr": expr, "how": how, "action": action})
            claimed.add(expr)
    return chosen, notes


class Extracted(NamedTuple):
    sections: list   # (start, end, code, action) for each "**Disease:**" section
    blocks: list     # every python block, as a Block
    chosen: list     # the rules found: {code, n, expr, how, action}
    notes: list      # why a section gave no rule
    clashes: list    # names that two blocks define with different code (the last one wins for every rule)


def _bound_names(node):
    """The names a top-level statement defines: a function, a class, an assignment or an import."""
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        return [t.id for t in node.targets if isinstance(t, ast.Name)]
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return [(a.asname or a.name).split(".")[0] for a in node.names]
    return []


def find_clashes(sections, blocks):
    """Messages for each name that two or more blocks define with different code.

    All blocks end up in one module, so a name means its last definition for every rule. The same helper copied into
    several sections is fine; a name with different code in different sections silently changes the earlier rules.
    """
    versions = {}  # name -> [(block, dump of the code)]
    for b in blocks:
        if b.kind not in ("code", "support"):
            continue
        here = {}
        for node in ast.parse(b.text).body:
            for name in _bound_names(node):
                here[name] = ast.dump(node)  # no line numbers: formatting and comments do not count
        for name, dump in here.items():
            versions.setdefault(name, []).append((b, dump))
    messages = []
    for name, found in versions.items():
        if len({dump for _, dump in found}) > 1:
            where = ", ".join(f"block {b.index} ({sections[b.section][2] if b.section is not None else 'outside the sections'})"
                              for b, _ in found)
            messages.append(f"{name} is defined with different code in {where}; the last one is used by every rule. "
                            f"Rename one of them")
    return messages


def write_text_lf(path: Path, text: str):
    """Write UTF-8 text with LF line endings on every platform. Path.write_text writes CRLF on Windows, so the
    registry and the rules files made there would differ, line by line, from the ones made on Linux."""
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def read_rules(src: str) -> Extracted:
    """Read a medicine.md and pick the rule of each section. Nothing is run."""
    sections, blocks = read_medicine(src)
    chosen, notes = resolve(sections, blocks)
    return Extracted(sections, blocks, chosen, notes, find_clashes(sections, blocks))


def fingerprint(code: str) -> str:
    """sha256 of the code's syntax tree. Comments, blank lines, spacing and the header docstring do not change it;
    changing what the code does, or the order of the blocks, does."""
    tree = ast.parse(code)
    if tree.body and isinstance(tree.body[0], ast.Expr) and isinstance(getattr(tree.body[0], "value", None), ast.Constant):
        tree.body = tree.body[1:]  # the header says which file it came from
    return hashlib.sha256(ast.dump(tree).encode("utf-8")).hexdigest()


def rules_from_markdown(src: str, name: str = "medicine.md"):
    """Return (Extracted, code): the rules found in a medicine.md, and the text of the rules file for them.

    The code has RULES only (no IDs, so one function for each code). This is the one place that reads a
    medicine.md: the extractor writes the code to a file, and harm_test.py can run it in memory.
    """
    found = read_rules(src)
    return found, render(name, found.blocks, found.chosen)


def origin(path: Path):
    """(pack_id, handle) for diagnosis/submissions/<pack_id>/<handle>/medicine.md, else None."""
    path = path.resolve()
    if path.parent.parent.parent.name == "submissions":
        return path.parent.parent.name, path.parent.name
    return None


def load_registry(path: Path):
    if not path.exists():
        return []
    try:
        entries = json.loads(path.read_text(encoding="utf-8"))["rules"]
    except (OSError, ValueError, KeyError) as e:
        raise SystemExit(f"error: cannot read the registry {path}: {e}")
    for e in entries:  # a registry written on Windows before the as_posix fix has backslash paths
        if "file" in e:
            e["file"] = e["file"].replace("\\", "/")
    return entries


SHEET_ID = re.compile(r'(?m)^RULE_ID\s*=\s*["\'](D[AB]-[A-H][1-9]-\d{3})["\']')


def sheet_ids(team_dir: Path) -> frozenset:
    """Rule IDs that the team's medicine sheets (<team>/*-medicines/*.md) use. New registry IDs skip them, so a
    sheet's hand-numbered rule and a newly extracted rule never share an ID."""
    ids = set()
    for sheet in sorted(team_dir.glob("*-medicines/*.md")):
        try:
            ids.update(SHEET_ID.findall(sheet.read_text(encoding="utf-8")))
        except (OSError, UnicodeDecodeError):
            pass
    return frozenset(ids)


def save_registry(path: Path, entries):
    path.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(entries, key=lambda e: e["id"])
    write_text_lf(path, json.dumps({"rules": entries}, indent=2) + "\n")


def assign_ids(entries, pack: str, handle: str, rules, file_rel: str, reserved=frozenset(), code_sha256=None):
    """Return the rule ID of each rule in `rules`, in order. New entries are added to the registry list.

    A (pack, handle, code, n) that is in the registry keeps its ID. A new one takes the next number for its
    team and code, skipping the IDs in `reserved` (used by the medicine sheets). Rules of this pack and handle
    that are no longer in the medicine.md are marked withdrawn. `code_sha256` is the fingerprint of the
    medicine.md's code (see fingerprint); every rule of the file keeps it.
    """
    prefix = "DA" if pack in DATA_A_PACKS else "DB"
    mine = {(e["code"], e.get("n", 1)): e for e in entries if e["pack"] == pack and e["handle"] == handle}
    now = {(r["code"], r["n"]) for r in rules}
    for key, e in mine.items():
        e["withdrawn"] = key not in now
    ids = []
    for r in rules:
        e = mine.get((r["code"], r["n"]))
        if e is None:
            head = f"{prefix}-{r['code']}-"
            used = [int(i.rsplit("-", 1)[1]) for i in [x["id"] for x in entries] + list(reserved) if i.startswith(head)]
            e = {"id": f"{prefix}-{r['code']}-{max(used, default=0) + 1:03d}", "pack": pack, "handle": handle,
                 "code": r["code"], "n": r["n"]}
            entries.append(e)
        e["n"], e["file"], e["action"], e["withdrawn"] = r["n"], file_rel, r["action"], False
        if code_sha256:
            e["code_sha256"] = code_sha256
        ids.append(e["id"])
    return ids


def render(source_name: str, blocks, chosen, ids=None) -> str:
    out = [f'"""Generated by medicine_extract.py from {source_name}. Do not edit by hand: '
           f'change the medicine.md and run the extractor again."""',
           "import re  # most rules use regexes, so every rules file has it"]
    for b in blocks:
        text = b.text.rstrip() + "\n"
        if b.kind == "code":
            out.append(f"# --- block {b.index} ---\n{text}")
        elif b.kind == "support":
            out.append(f"# --- block {b.index} (imports and constants; ignored if it does not run) ---\n"
                       f"try:\n{textwrap.indent(text, '    ')}except Exception:\n    pass\n")
    first = {}
    for r in chosen:
        first.setdefault(r["code"], r["expr"])  # RULES has one function for each code; the others are only in RULES_BY_ID
    out.append("RULES = {\n" + "".join(f'    "{code}": {expr},\n' for code, expr in first.items()) + "}\n")
    if ids:
        out.append("RULES_BY_ID = {\n" + "".join(f'    "{rid}": {r["expr"]},\n' for rid, r in zip(ids, chosen)) + "}\n")
    return "\n\n".join(out)


def rules_path(pack: str, handle: str, base: Path = DEFAULT_REGISTRY.parent) -> Path:
    """Where the rules file of a pack and handle goes: <base>/<pack_id>/<handle>.py. The base is the registry's folder."""
    return base / pack / f"{handle}.py"


def default_out(path: Path, base: Path = DEFAULT_REGISTRY.parent):
    found = origin(path)
    return rules_path(found[0], found[1], base) if found else None


def remove_generated(out: Path):
    """Delete the rules file `out` if this tool wrote it (it starts with the "Generated by medicine_extract.py"
    line). A file written by hand is never deleted."""
    try:
        if out.is_file() and out.read_text(encoding="utf-8").startswith('"""Generated by medicine_extract.py'):
            out.unlink()
            print(f"  removed the old rules file {out}")
    except OSError as e:
        print(f"warning: cannot remove the old rules file {out}: {e.strerror or e}", file=sys.stderr)


def withdraw_missing(entries, seen, base: Path, pack_dirs=()):
    """--dir: withdraw the rules of a (pack, handle) whose medicine.md is not in the folder any more.

    Only packs that are in this run are checked: a pack with at least one medicine.md, or a pack folder in `pack_dirs`
    (it may be empty: its last medicine.md was deleted). So a folder that holds one pack (a fetch with --pack-id)
    never withdraws the rules of the other packs. Returns {(pack, handle): [IDs]}."""
    packs = {pack for pack, _ in seen} | set(pack_dirs)
    gone = {}
    for e in entries:
        key = (e["pack"], e["handle"])
        if e["pack"] in packs and key not in seen and not e.get("withdrawn"):
            e["withdrawn"] = True
            gone.setdefault(key, []).append(e["id"])
    for (pack, handle), ids in sorted(gone.items()):
        print(f"\n== {pack}/{handle} ==\n  no medicine.md in this run: withdrawn {', '.join(ids)}")
        remove_generated(rules_path(pack, handle, base))
    return gone


def withdraw_stale(found, entries, out: Path):
    """A medicine.md that now gives no rule: withdraw every registry entry of its (pack, handle) and remove the
    rules file an earlier run generated for it, so no stale rule stays active. Only a file this tool wrote
    (it starts with the "Generated by medicine_extract.py" line) is removed. Returns the withdrawn IDs."""
    ids = []
    if found and entries is not None:
        for e in entries:
            if e["pack"] == found[0] and e["handle"] == found[1] and not e.get("withdrawn"):
                e["withdrawn"] = True
                ids.append(e["id"])
    remove_generated(out)
    if ids:
        print(f"  withdrawn (the medicine.md no longer gives these rules): {', '.join(ids)}")
    return ids


def registry_changes(before, after) -> dict:
    """Compare the registry entries before and after a run. Returns {kind: [entry]} for new, withdrawn, reinstated and
    changed (an active rule whose code, action or file changed). Done on the two lists, so no path can miss a change.
    A field the old entry did not have (code_sha256 on a registry from before it was added) is filled in, not changed."""
    old = {e["id"]: e for e in before}
    found = {"new": [], "withdrawn": [], "reinstated": [], "changed": []}
    for e in sorted(after, key=lambda e: e["id"]):
        o = old.get(e["id"])
        if o is None:
            found["new"].append(e)
        elif e.get("withdrawn") and not o.get("withdrawn"):
            found["withdrawn"].append(e)
        elif o.get("withdrawn") and not e.get("withdrawn"):
            found["reinstated"].append(e)
        elif not e.get("withdrawn") and any(k in o and e.get(k) != o[k] for k in ("code_sha256", "action", "file")):
            found["changed"].append(e)
    return found


def print_changes(before, after):
    """The run summary: for each kind, one line per pack/handle with its rule IDs."""
    found = registry_changes(before, after)
    rule = "=" * 78
    counts = ", ".join(f"{len(v)} {k}" for k, v in found.items() if v) or "none"
    print(f"\n{rule}\nREGISTRY CHANGES: {counts}\n{rule}")
    if not any(found.values()):
        return
    for kind, entries in found.items():
        if not entries:
            continue
        by_handle = {}
        for e in entries:
            by_handle.setdefault(f"{e['pack']}/{e['handle']}", []).append(e["id"])
        width = max(map(len, by_handle))
        print(f"\n{kind} ({len(entries)})")
        for who, ids in sorted(by_handle.items()):
            print(textwrap.fill(", ".join(ids), width=100, initial_indent=f"  {who:<{width}}  ",
                                subsequent_indent=" " * (width + 4)))


def extract_one(medicine: Path, found, out: Path, entries, registry: Path, clashed=None) -> int:
    """Turn one medicine.md into the rules file `out`. Return 0 if done, 1 if no rule could be made, 2 if unreadable.

    `found` is (pack_id, handle), or None. `entries` is the registry list (changed in place, saved by the caller),
    or None: the rules then get no IDs. `clashed`, if given, gets `found` added when two blocks define a name
    with different code.
    """
    try:
        src = medicine.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        print(f"error: cannot read {medicine}: {getattr(e, 'strerror', None) or e}", file=sys.stderr)
        return 2

    sections, blocks, chosen, notes, clashes = read_rules(src)
    if not sections:
        print("error: no '**Disease:** <code>' line found, so there is no medicine section", file=sys.stderr)
        withdraw_stale(found, entries, out)
        return 1

    print(f"{medicine.name}: {len(sections)} sections, {len(blocks)} python blocks "
          f"({sum(b.kind == 'code' for b in blocks)} code, {sum(b.kind == 'support' for b in blocks)} support)")
    if not chosen:
        for n in notes:
            print(f"  note: {n}")
        print("error: no rule could be matched to a disease code; nothing written", file=sys.stderr)
        withdraw_stale(found, entries, out)
        return 1

    ids = None
    if found and entries is not None:
        # as_posix: the registry is shared, so its paths use "/" on every platform (Windows gives "\\")
        try:
            file_rel = Path(os.path.relpath(out.resolve(), registry.resolve().parent)).as_posix()
        except ValueError:  # Windows: the two files are on different drives, so there is no relative path
            file_rel = out.resolve().as_posix()
        ids = assign_ids(entries, found[0], found[1], chosen, file_rel,
                         reserved=sheet_ids(registry.resolve().parent.parent),
                         code_sha256=fingerprint(render(medicine.name, blocks, chosen)))
    for i, r in enumerate(chosen):
        print(f"  {r['code']}: {r['expr']}   (from {r['how']}, action {r['action'] or 'not stated'})"
              + (f"   -> {ids[i]}" if ids else ""))
    for b in blocks:
        if b.kind in ("fragment", "syntax error"):
            print(f"  left out: block {b.index}, {b.kind}")
    for n in notes:
        print(f"  note: {n}")
    for c in clashes:
        print(f"  warning: {c}")
    if clashes and clashed is not None:
        clashed.append(found)
    if ids is None:
        print("  note: the path has no pack and handle, so the rules get no IDs and the registry is not changed")
        if any(r["n"] > 1 for r in chosen):
            print("  note: without IDs only the first rule of each code is in the file")

    out.parent.mkdir(parents=True, exist_ok=True)
    write_text_lf(out, render(medicine.name, blocks, chosen, ids))
    print(f"written: {out}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Turn medicine.md files into rules files for harm_test.py.")
    ap.add_argument("medicine", type=Path, nargs="?", help="one medicine.md")
    ap.add_argument("--dir", type=Path, help="a folder with <pack_id>/<handle>/medicine.md: extract all of them")
    ap.add_argument("--out", type=Path, help="the rules file to write (one medicine.md only)")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY, help="the rule ID registry (default: ../rules/registry.json)")
    args = ap.parse_args(argv)
    if (args.medicine is None) == (args.dir is None):
        print("error: give one medicine.md, or --dir <folder>", file=sys.stderr)
        return 2
    if args.dir and args.out:
        print("error: --out is for one medicine.md; with --dir each rules file goes next to the registry, in <pack_id>/<handle>.py", file=sys.stderr)
        return 2

    if args.medicine:
        found = origin(args.medicine)
        out = args.out or default_out(args.medicine, args.registry.parent)
        if out is None:
            print("error: this path is not diagnosis/submissions/<pack_id>/<handle>/medicine.md; give --out", file=sys.stderr)
            return 2
        entries = load_registry(args.registry) if found else None
        before = copy.deepcopy(entries)
        rc = extract_one(args.medicine, found, out, entries, args.registry)
        if rc in (0, 1) and entries is not None:  # 1: no rule now, but stale IDs may have been withdrawn
            save_registry(args.registry, entries)
            print(f"registry: {args.registry}")
            print_changes(before, entries)
        return rc

    if not args.dir.is_dir():
        print(f"error: {args.dir} is not a folder", file=sys.stderr)
        return 2
    files = sorted(args.dir.glob("*/*/medicine.md"))
    pack_dirs = {d.name for d in args.dir.iterdir() if d.is_dir()}   # an empty pack folder still counts as in this run
    if not files and not pack_dirs:
        print(f"error: no <pack_id>/<handle>/medicine.md under {args.dir}", file=sys.stderr)
        return 1
    entries = load_registry(args.registry)
    before = copy.deepcopy(entries)
    status, clashed = {}, []
    for path in files:
        pack, handle = path.parent.parent.name, path.parent.name
        print(f"\n== {pack}/{handle} ==")
        status[(pack, handle)] = extract_one(path, (pack, handle), rules_path(pack, handle, args.registry.parent), entries, args.registry, clashed)
    gone = withdraw_missing(entries, set(status), args.registry.parent, pack_dirs)
    if not files and not gone:
        print(f"error: no <pack_id>/<handle>/medicine.md under {args.dir}", file=sys.stderr)
        return 1
    save_registry(args.registry, entries)
    failed = [k for k, rc in status.items() if rc]
    print(f"\n{len(status) - len(failed)} of {len(status)} medicine.md files done. Registry: {args.registry}")
    for pack, handle in failed:
        print(f"  no rules: {pack}/{handle}")
    for pack, handle in sorted(gone):
        print(f"  medicine.md gone, rules withdrawn: {pack}/{handle}")
    for pack, handle in clashed:
        print(f"  same name, different code in two sections (see the warning above): {pack}/{handle}")
    print_changes(before, entries)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
