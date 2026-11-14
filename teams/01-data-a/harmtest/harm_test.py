"""Harm test: run one cleaning rule on a gold set and count the damage.

Usage:
    python -I harm_test.py --pack <pack.parquet> --gold <pack_gold.csv> --rule <RULE_ID>
    python -I harm_test.py --pack <pack.parquet> --gold <pack_gold.csv> --rules <rules.py|medicine.md> --rule <CODE>

    --pack      the pack Parquet file (columns doc_id, text are read)
    --gold      the gold CSV (doc_id, reader_1, reader_2; distinct nonempty reviewers required)
    --rule      a rule ID such as DA-B3-001 (DA = Data A, DB = Data B, then the disease code and a number),
                or a disease code such as B3 when --rules is given
    --rules     a .py file, or a medicine.md file whose ```python blocks hold the code. Not needed for a rule ID:
                the registry says which file medicine_extract.py wrote the rule to
    --registry  the registry file (default: ../rules/registry.json)
    --limit     the failing limit in percent of the gold set (default 1); at least 1 document is always allowed
    --show N    also list the first N changed or dropped doc_ids (ids only, never text)
    --timeout   seconds allowed for one rule call on one document (default 5)

A plain medicine.md (one "**Disease:** <code>" section for each code) is read by medicine_extract.py, the same code
that writes rules/<pack_id>/<handle>.py, so the rules that run here are the rules the team runs. A fragment (a block
that does not run on its own) or a block with a syntax error is an error here: fix or remove it. (The extractor leaves
such a block out and reports it.) Two sections for one code, a name that two blocks define with different code, or
no section at all is an error too. The rule of a section is medicine_<code>, cure_<code> or its only public function.

Inside a rules .py file the rule is found in one of three ways:
    RULES_BY_ID = {"DA-B3-001": fn}   written by medicine_extract.py (used for a rule ID)
    RULES = {"B3": fn}                fn(text) -> ("keep" | "fix" | "drop", new_text)   (the shared format)
    def cure_b3(text): ...            returns the new text as a str

A medicine sheet (for example code-medicines/medicine_code.md) may hold one rule per ```python block, each with its
own RULE_ID = "DA-B3-001" and a function medicine(doc). Each such block runs in its own namespace, so blocks may
reuse names, and the rule is found by its RULE_ID. --rule all then tests every rule of the sheet alone, and the
whole sheet in file order (a drop ends a document, a fix passes the new text to the next rule).

A rule may also return True (drop) or False (keep). A rule whose first parameter is named doc gets a dict with
doc_id, text, url and lang_label (url and lang_label are read when the pack has them; required metadata cannot be missing), not the text.

A document is "changed" when the rule returns text that differs from the input (compared byte for byte, no
normalisation). It is "dropped" when the rule returns the action "drop". Each document counts once.
PASS requires no errors and (changed + dropped) at most max(1, floor(--limit percent of n)) documents, where n is
the number of rows in the gold set of the rule's own pack (TSAI decision for small gold sets). With the default 1%
this is 1 document for every gold set under 200 rows, 2 for 200 to 299 rows, and so on. An empty gold set is
reported as "result: NOT TESTABLE" with exit code 2.

Exit code: 0 = PASS, 1 = FAIL, 2 = the test could not be run (bad input, empty gold set, or the rule raised an error).

The rules file is run as code: read it before you run it. A per-document timeout stops runaway regexes. On Linux and
macOS it uses SIGALRM: the document is reported as an error and the test goes on. Windows has no SIGALRM, so there
the faulthandler watchdog is used: a rule call that runs past --timeout ends the whole test (exit code 1) and prints
where it was stuck (a stack trace, never document text).
"""
import argparse
import ast
import copy
import csv
import faulthandler
import hashlib
import inspect
import json
import math
import os
import re
import signal
import sys
import textwrap
from fractions import Fraction
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # python -I does not put the script folder on the path
import medicine_extract  # noqa: E402  (reads a plain medicine.md: the same code the team's extractor uses)

ACTIONS = ("keep", "fix", "drop")
RULE_ID = re.compile(r"^D[AB]-[A-H][1-9]-\d{3}$")
RULE_ID_RE = RULE_ID
DEFAULT_REGISTRY = Path(__file__).resolve().parent.parent / "rules" / "registry.json"
REPO_ROOT = Path(__file__).resolve().parents[3]


def repo_relative(path: Path) -> str:
    """Serialize rule sources relative to the repository root."""
    try:
        return Path(os.path.relpath(Path(path).resolve(), REPO_ROOT)).as_posix()
    except ValueError:  # Windows: the file is on another drive than the repository, so there is no relative path
        return Path(path).resolve().as_posix()

FENCE = medicine_extract.FENCE  # a fenced python block, possibly indented inside a list item
HAS_ALARM = hasattr(signal, "setitimer")
# faulthandler writes to a real file descriptor, so it uses the process's own stderr (None under pythonw: no watchdog)
WATCHDOG_OUT = sys.__stderr__
DEBUG = False  # set by --debug


def debug(message: str):
    """A progress line on stderr when --debug is given. Ids, counts and offsets only, never document text."""
    if DEBUG:
        print("debug: " + message, file=sys.stderr)


class Fatal(Exception):
    """The test cannot be run. The message is printed and the exit code is 2."""


class NotTestable(Fatal):
    """The gold set is empty, so the pack cannot be tested. Reported as "result: NOT TESTABLE", exit code 2."""


class RuleTimeout(Exception):
    pass


class RuleMutationError(Exception):
    pass


def _on_alarm(signum, frame):
    raise RuleTimeout("per-document time limit exceeded")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



def _code_assignment_names(code: str):
    """Inspect top-level bindings and reject duplicate literal rule keys.

    Duplicate keys inside a Python dictionary are normally overwritten before
    runtime inspection, which would make --rule all undercount submitted rules.
    """
    tree = ast.parse(code)
    names = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        seen = set()
        for key in node.keys:
            if not isinstance(key, ast.Constant) or not isinstance(key.value, str):
                continue
            if key.value in seen:
                raise ValueError(f'duplicate dictionary key {key.value!r}')
            seen.add(key.value)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for target in targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
    return names


def _is_sheet(blocks) -> bool:
    """True if a block sets RULE_ID or defines medicine(): a medicine sheet, one rule in each block."""
    for block in blocks:
        try:
            tree = ast.parse(block)
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "medicine":
                return True
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
            if any(isinstance(t, ast.Name) and t.id == "RULE_ID" for t in targets):
                return True
    return False


def load_medicine_md(path: Path, src: str):
    """A plain medicine.md (one "**Disease:**" section for each code): the rules are found by medicine_extract.py.

    A block that does not run on its own (a fragment) or has a syntax error stops the test: the annotator must fix
    or remove it, so a PASS is for the whole medicine.md. (medicine_extract.py leaves such a block out and
    reports it.) The rules run from the same code that the extractor writes to rules/<pack_id>/<handle>.py
    (without the IDs).
    """
    found, code = medicine_extract.rules_from_markdown(src, path.name)
    for b in found.blocks:
        debug(f"{path.name} block {b.index}: {b.kind}")
        if b.kind in ("fragment", "syntax error"):
            what = "has a syntax error" if b.kind == "syntax error" else "is a fragment that does not run on its own"
            raise Fatal(f"{path.name} block {b.index} {what}. Make it complete, or remove it from the medicine.md")
    if not found.sections:
        raise Fatal(f"{path.name}: no '**Disease:** <code>' line found, so there is no medicine section")
    for note in found.notes:
        print(f"warning: {note}", file=sys.stderr)
    if found.clashes:
        raise Fatal(f"{path.name}: " + "; ".join(found.clashes))
    if not found.chosen:
        raise Fatal(f"{path.name}: no rule could be matched to a disease code")
    for r in found.chosen:
        debug(f"section {r['code']}: rule {r['expr']} (from {r['how']})")
    ns = {"__name__": "rules"}
    try:
        exec(compile(code, path.name, "exec"), ns)
    except Exception as e:
        raise Fatal(f"{path.name}: {type(e).__name__}: {e}") from e
    return ns, medicine_extract.fingerprint(code)  # the registry keeps the same value as code_sha256


def load_rules(path: Path):
    """Run all executable rule blocks; reject skipped/overwritten definitions."""
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        raise Fatal(f"cannot read rules file {path}: {e}") from e
    is_md = path.suffix.lower() == ".md"
    blocks = [textwrap.dedent(m.group(2)) for m in FENCE.finditer(src)] if is_md else [src]
    if not blocks:
        raise Fatal(f"{path}: no executable python blocks")
    if is_md and not _is_sheet(blocks):
        return load_medicine_md(path, src)
    ns = {"__name__": "rules"}
    declared = {}
    for i, block in enumerate(blocks, 1):
        label = f"{path.name} block {i}" if is_md else path.name
        try:
            assignments = _code_assignment_names(block)
            tree = ast.parse(block)
            has_medicine = "medicine" in assignments or any(
                isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == "medicine"
                for n in tree.body
            )
            if is_md and has_medicine and "RULE_ID" not in assignments:
                raise ValueError("medicine() defined without RULE_ID; rule would be skipped")
            if is_md and "RULE_ID" in assignments:
                own = {"__name__": "rules"}
                exec(compile(block, label, "exec"), own)
                rid, medicine = own.get("RULE_ID"), own.get("medicine")
                if not isinstance(rid, str) or not RULE_ID.fullmatch(rid):
                    raise ValueError("RULE_ID is absent or invalid")
                if not callable(medicine):
                    raise ValueError(f"{rid}: medicine must be callable")
                if rid in declared or rid in (ns.get("RULES_BY_ID") or {}):
                    raise ValueError(f"duplicate RULE_ID {rid} (previous block {declared.get(rid, 'registry mapping')})")
                declared[rid] = i
                ns.setdefault("RULES_BY_ID", {})[rid] = medicine
                required = own.get("REQUIRED_FIELDS", ())
                if isinstance(required, str):
                    raise ValueError(f"{rid}: REQUIRED_FIELDS must be a collection of field names")
                requirements = set(required)
                if own.get("LANGS") is not None:
                    requirements.add("lang_label")
                # Detect metadata use even through doc.get() with a default.
                for node in ast.walk(tree):
                    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name) and node.value.id == "doc":
                        if isinstance(node.slice, ast.Constant) and node.slice.value in ("url", "lang_label"):
                            requirements.add(node.slice.value)
                    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                            and isinstance(node.func.value, ast.Name) and node.func.value.id == "doc"
                            and node.func.attr == "get" and node.args
                            and isinstance(node.args[0], ast.Constant)
                            and node.args[0].value in ("url", "lang_label")):
                        requirements.add(node.args[0].value)
                if not all(isinstance(s, str) and s in ("doc_id", "text", "url", "lang_label") for s in requirements):
                    raise ValueError(f"{rid}: invalid REQUIRED_FIELDS")
                setattr(medicine, "_g1_required_fields", frozenset(requirements))
            else:
                # An incomplete example or overwritten mapping must never produce PASS.
                prior = dict(ns.get("RULES_BY_ID") or {})
                prior_short = dict(ns.get("RULES") or {})
                exec(compile(block, label, "exec"), ns)
                updated, updated_short = ns.get("RULES_BY_ID") or {}, ns.get("RULES") or {}
                if not isinstance(updated, dict) or not isinstance(updated_short, dict):
                    raise ValueError("RULES_BY_ID and RULES must be dictionaries")
                if any(rid not in updated or updated[rid] is not fn for rid, fn in prior.items()):
                    raise ValueError("overwrote an already registered RULE_ID")
                if any(rid not in updated_short or updated_short[rid] is not fn for rid, fn in prior_short.items()):
                    raise ValueError("overwrote an already registered RULES entry")
        except Exception as e:
            raise Fatal(f"{label}: {type(e).__name__}: {e}") from e
    by_id, by_code = ns.get("RULES_BY_ID") or {}, ns.get("RULES") or {}
    if not isinstance(by_id, dict) or not all(callable(fn) for fn in by_id.values()):
        raise Fatal("RULES_BY_ID must map IDs to callable rules")
    if not isinstance(by_code, dict) or not all(isinstance(k, str) and callable(fn) for k, fn in by_code.items()):
        raise Fatal("RULES must map disease codes to callable rules")
    if not all(isinstance(rid, str) and RULE_ID.fullmatch(rid) for rid in by_id):
        raise Fatal("RULES_BY_ID includes invalid rule ID")
    return ns, hashlib.sha256("\n".join(blocks).encode("utf-8")).hexdigest()

def check_one_section(path: Path, code: str):
    """A disease code names one rule only if the medicine.md has one section for it."""
    if path.suffix.lower() != ".md":
        return
    try:
        src = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as e:
        raise Fatal(f"cannot read rules file {path}: {e}") from e
    n = sum(1 for s in medicine_extract.read_rules(src).sections if s[2] == code)
    if n > 1:
        raise Fatal(f"{n} sections for {code} in {path.name}, so I can't tell which to test. "
                    f"Keep one section for each disease code.")


def registry_file_for(rule_id: str, registry: Path) -> Path:
    """Find the rule file, rejecting duplicates and unsafe/withdrawn paths."""
    try:
        entries = json.loads(registry.read_text(encoding="utf-8"))["rules"]
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Fatal(f"cannot read the registry {registry}: {e}") from e
    if not isinstance(entries, list) or any(not isinstance(e, dict) for e in entries):
        raise Fatal(f"{registry}: rules must be a list of records")
    matches = [e for e in entries if e.get("id") == rule_id]
    if not matches:
        raise Fatal(f"{rule_id} is not in the registry {registry} (expected one registry entry, found 0)")
    if len(matches) != 1:
        raise Fatal(f"{rule_id}: expected one registry entry, found {len(matches)}")
    e = matches[0]
    if e.get("withdrawn"):
        raise Fatal(f"{rule_id}: withdrawn rule cannot be approved")
    filename = e.get("file")
    if not isinstance(filename, str) or not filename.strip():
        raise Fatal(f"{rule_id}: missing rule file path in registry")
    resolved = (registry.parent / filename).resolve()
    trusted_root = (Path(__file__).resolve().parents[3] if registry.resolve() == DEFAULT_REGISTRY.resolve()
                    else registry.parent.resolve())
    if not resolved.is_relative_to(trusted_root):
        raise Fatal(f"{rule_id}: rule file must remain within trusted registry directory/repository")
    return resolved

def find_rule(ns: dict, code: str):
    if RULE_ID.match(code):
        by_id = ns.get("RULES_BY_ID")
        if isinstance(by_id, dict) and code in by_id:
            return by_id[code]
        raise Fatal(f"rule {code} is not in the rules file (found: {', '.join(sorted(by_id or {})) or 'none'})")
    rules = ns.get("RULES")
    if isinstance(rules, dict) and code in rules:
        return rules[code]
    fn = ns.get("cure_" + code.lower())
    if callable(fn):
        return fn
    found = sorted(rules) if isinstance(rules, dict) else sorted(k[5:].upper() for k in ns if k.startswith("cure_"))
    raise Fatal(f"no rule for {code} in the rules file (found: {', '.join(found) or 'none'})")


def classify(text: str, out) -> str:
    """Return 'keep', 'fix' or 'drop' for one rule result. 'fix' means the text really differs."""
    if isinstance(out, bool):  # a detector: True means drop the document
        return "drop" if out else "keep"
    if isinstance(out, str):
        return "fix" if out != text else "keep"
    if isinstance(out, tuple) and len(out) == 2 and out[0] in ACTIONS:
        if out[0] == "drop":
            return "drop"
        if not isinstance(out[1], str):
            raise TypeError("new_text is not a str")
        # Judge by the text: a rule that says "keep" but returns different text still changed the document.
        return "fix" if out[1] != text else "keep"
    raise TypeError("rule must return a str, (action, new_text), True or False")


def read_gold(path: Path):
    """Gold membership must have two *different* nonempty named readers.

    This validates provenance structure, NOT the underlying diagnosis votes;
    verify those against the source submissions when building the gold set.
    """
    required = {'doc_id', 'reader_1', 'reader_2'}
    ids = []
    seen = set()
    try:
        with path.open(newline='', encoding='utf-8-sig') as stream:
            reader = csv.DictReader(stream)
            if not reader.fieldnames or not required.issubset(reader.fieldnames):
                raise Fatal(f'{path.name}: required columns are doc_id,reader_1,reader_2')
            if len(reader.fieldnames) != len(set(reader.fieldnames)):
                raise Fatal(f'{path.name}: duplicate column heading')
            for number, row in enumerate(reader, 2):
                if None in row:
                    raise Fatal(f'{path.name} line {number}: extra CSV fields')
                id_, a, b = ((row.get(key) or '').strip() for key in ('doc_id', 'reader_1', 'reader_2'))
                if not id_ or not a or not b:
                    raise Fatal(f'{path.name} line {number}: missing ID or clean reader')
                if a.casefold() == b.casefold():
                    raise Fatal(f'{path.name} line {number}: same reader recorded twice')
                if id_ in seen:
                    raise Fatal(f'{path.name} line {number}: duplicate doc_id {id_}')
                seen.add(id_)
                ids.append(id_)
    except (OSError, UnicodeError, csv.Error) as exc:
        raise Fatal(f'cannot read gold file {path}: {exc}') from exc
    if not ids:
        raise NotTestable(f'{path.name}: the gold set is empty')
    return ids

def read_pack_docs(path: Path, ids):
    """Return validated gold documents; preserve missing optional metadata as missing."""
    try:
        import pyarrow as pa
        import pyarrow.compute as pc
        import pyarrow.parquet as pq
    except ImportError as e:
        raise Fatal("pyarrow is required to read Parquet; pip install pyarrow") from e
    try:
        names = pq.read_schema(path).names
        if not {"doc_id", "text"}.issubset(names):
            raise Fatal(f"{path.name}: pack requires doc_id and text columns")
        cols = ["doc_id", "text"] + [c for c in ("url", "lang_label", "pack_id") if c in names]
        table = pq.read_table(path, columns=cols)
        table = table.filter(pc.is_in(table["doc_id"], value_set=pa.array(ids, type=table["doc_id"].type)))
        rows = table.to_pylist()
    except Fatal:
        raise
    except Exception as e:
        raise Fatal(f"cannot read pack {path}: {type(e).__name__}: {e}") from e
    docs = {}
    for row in rows:
        doc_id = row["doc_id"]
        if not isinstance(doc_id, str) or doc_id in docs:
            raise Fatal(f"{path.name}: invalid or duplicate doc_id {doc_id!r}")
        if not isinstance(row["text"], str):
            raise Fatal(f"{path.name}: doc_id {doc_id} has non-text value")
        if "pack_id" in row and row["pack_id"] != path.stem:
            raise Fatal(f"{path.name}: doc_id {doc_id} belongs to pack {row['pack_id']!r}")
        docs[doc_id] = dict(row)
    missing = [id_ for id_ in ids if id_ not in docs]
    if missing:
        raise Fatal(f"{path.name}: {len(missing)} gold IDs are not in the pack, e.g. {', '.join(missing[:3])}")
    return {id_: docs[id_] for id_ in ids}

def read_pack_texts(path: Path, ids):
    """Return {doc_id: text} for the gold ids only."""
    return {i: d["text"] for i, d in read_pack_docs(path, ids).items()}


def takes_doc(rule) -> bool:
    """True if the rule's first parameter is named doc: it gets the document dict, not the text."""
    try:
        params = list(inspect.signature(rule).parameters)
    except (TypeError, ValueError):
        return False
    return bool(params) and params[0] == "doc"


def allowed_documents(n: int, limit: Fraction) -> int:
    """How many of the n gold documents a rule may change or drop: max(1, floor(limit% of n)).

    TSAI decision for small gold sets: at 1% a set under 100 rows would allow no change at all, so at least one
    document is always allowed. With the default 1% the limit is 1 document for every gold set under 200 rows.
    """
    return max(1, math.floor(limit * n / 100))


def harm_percent(changed: int, dropped: int, n: int, limit: Fraction):
    """Return (percent of the gold set changed or dropped, passed). The one place the limit is applied."""
    if n <= 0:
        raise NotTestable('the gold set is empty')
    percent = Fraction((changed + dropped) * 100, n)
    return percent, changed + dropped <= allowed_documents(n, limit)

def limit_line(n: int, limit: Fraction, shown: str) -> str:
    allowed = allowed_documents(n, limit)
    return (f"limit: {allowed} document{'s' if allowed != 1 else ''} changed or dropped "
            f"(max(1, {shown}% of {n}))")


def _replacement(text, out):
    return out if isinstance(out, str) else out[1]



def _invoke(rule, doc, timeout):
    """Classify a rule without ever exposing shared inputs to mutation."""
    original = copy.deepcopy(doc)
    required = set(getattr(rule, '_g1_required_fields', ()))
    missing = sorted(k for k in required if k not in original or original[k] is None or (k == 'lang_label' and not str(original[k]).strip()))
    if missing:
        raise ValueError('required metadata missing: ' + ', '.join(missing))
    working = copy.deepcopy(original)
    if HAS_ALARM:
        signal.setitimer(signal.ITIMER_REAL, timeout)
    elif WATCHDOG_OUT is not None:  # no SIGALRM (Windows): faulthandler's own thread ends the run if the call hangs
        debug(f"calling the rule on {doc.get('doc_id', '?')}")  # the last line before a timeout names the document
        faulthandler.dump_traceback_later(timeout, exit=True, file=WATCHDOG_OUT)
    try:
        out = rule(working) if takes_doc(rule) else rule(working['text'])
        if working != original:
            raise RuleMutationError('medicine mutated input document; return a new value instead')
        verdict = classify(original['text'], out)
        return verdict, (_replacement(original['text'], out) if verdict == 'fix' else original['text'])
    finally:
        if HAS_ALARM:
            signal.setitimer(signal.ITIMER_REAL, 0)
        elif WATCHDOG_OUT is not None:
            faulthandler.cancel_dump_traceback_later()



def _normalized(doc_id, value):
    if isinstance(value, dict):
        return value
    return {'doc_id': doc_id, 'text': value}


def debug_not_passing(name: str, detail: dict):
    """With --debug: one line for each gold document that the rule changed, dropped or crashed on."""
    if not DEBUG:
        return
    debug(f"{name}: " + (f"{len(detail)} gold documents did not pass" if detail else "every gold document passed"))
    for doc_id, what in detail.items():
        debug(f"  {doc_id}  {what}")


def run_rule(rule, texts: dict, timeout: float, detail: dict = None):
    """texts maps doc_id to the text, or to the document dict of read_pack_docs.

    If `detail` is a dict, it gets a line for each document that did not pass: doc_id -> a short description
    (the kind of change, lengths and the first offset that differs, or the error type). Never the text itself.
    """
    changed, dropped, errors = [], [], []
    if HAS_ALARM:
        signal.signal(signal.SIGALRM, _on_alarm)
    for doc_id, doc in texts.items():
        if not isinstance(doc, dict):
            doc = {"doc_id": doc_id, "text": doc}
        try:
            verdict, new = _invoke(rule, doc, timeout)
        except Exception as e:  # a rule that crashes is a rule bug; report the type only, never document text
            errors.append((doc_id, type(e).__name__))
            if detail is not None:
                detail[doc_id] = f"error {type(e).__name__}"
            continue
        if verdict == "fix":
            changed.append(doc_id)
            if detail is not None:
                old = doc["text"]
                detail[doc_id] = (f"changed  {len(old)} -> {len(new)} characters, "
                                  f"first difference at offset {len(os.path.commonprefix([old, new]))}")
        elif verdict == "drop":
            dropped.append(doc_id)
            if detail is not None:
                detail[doc_id] = f"dropped  ({len(doc['text'])} characters)"
    return changed, dropped, errors


def run_sheet(rules, texts: dict, timeout: float):
    """Run the rules in order on each document: a drop ends it, a fix passes the new text on. Same counts as run_rule."""
    changed, dropped, errors = [], [], []
    if HAS_ALARM:
        signal.signal(signal.SIGALRM, _on_alarm)
    for doc_id, doc in texts.items():
        d = copy.deepcopy(doc) if isinstance(doc, dict) else {"doc_id": doc_id, "text": doc}
        start, verdict = d["text"], "keep"
        for rid, rule in rules:
            try:
                step, newtext = _invoke(rule, d, timeout)
            except Exception as e:
                errors.append((doc_id, f"{rid}: {type(e).__name__}"))
                verdict = "error"
                break
            if step == "drop":
                verdict = "drop"
                break
            if step == "fix":
                d["text"] = newtext
        if verdict == "drop":
            dropped.append(doc_id)
        elif verdict == "keep" and d["text"] != start:
            changed.append(doc_id)
    return changed, dropped, errors
def _result(name, rows, n, limit):
    changed, dropped, errors = rows
    percent, passed = harm_percent(len(changed), len(dropped), n, limit)
    status = 'ERROR' if errors else 'PASS' if passed else 'FAIL'
    return {'rule': name, 'changed': len(changed), 'dropped': len(dropped),
            'affected_unique': len(set(changed) | set(dropped)),
            'errors': len(errors), 'error_types': sorted(set(error for _, error in errors)),
            'percent': f'{float(percent):.6f}', 'result': status,
            'changed_ids': changed, 'dropped_ids': dropped,
            'affected_ids': list(dict.fromkeys(changed + dropped))}



def _write_report(args, rules_path, digest, count, limit, status, results):
    if not args.report:
        return
    report = {"pack": args.pack.name, "pack_sha256": sha256_file(args.pack),
              "gold": args.gold.name, "gold_sha256": sha256_file(args.gold),
              "gold_count": count, "rules": repo_relative(rules_path),
              "rules_file_sha256": sha256_file(rules_path), "code_sha256": digest,
              "limit_percent": str(limit), "limit_documents": allowed_documents(count, limit),
              "status": status, "results": results}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    medicine_extract.write_text_lf(args.report, json.dumps(report, indent=2) + "\n")


def main_all(args, limit) -> int:
    """--rule all: every rule of the rules file alone, then all of them in file order."""
    ids = read_gold(args.gold)
    debug(f"gold: {len(ids)} doc_ids from {args.gold.name}")
    texts = read_pack_docs(args.pack, ids)
    debug(f"pack: {len(texts)} documents read from {args.pack.name}")
    ns, digest = load_rules(args.rules)
    debug(f"rules file: {args.rules} (code sha256 {digest[:12]})")
    rules = list((ns.get("RULES_BY_ID") or {}).items()) or list((ns.get("RULES") or {}).items())
    if not rules:
        raise Fatal(f"{args.rules.name}: no callable rules for --rule all")
    n = len(texts)
    print(f"pack: {args.pack.name}   gold: {args.gold.name} ({n} documents)   rules: {len(rules)}")
    print(f"rules file: {args.rules.name}   code sha256: {digest}")
    print(limit_line(n, limit, args.limit))
    print(f"{'rule':<18}{'changed':>8}{'dropped':>8}{'percent':>9}  result")
    worst = 0
    results = []
    details = {rid: {} for rid, _ in rules}
    for name, rows in [(rid, run_rule(rule, texts, args.timeout, details[rid] if DEBUG else None)) for rid, rule in rules] + [
            ("ALL IN ORDER", run_sheet(rules, texts, args.timeout))]:
        changed, dropped, errors = rows
        result = _result(name, rows, n, limit)
        results.append(result)
        percent, passed = harm_percent(len(changed), len(dropped), n, limit)
        status = result["result"]
        worst = max(worst, 2 if errors else 0 if passed else 1)
        print(f"{name:<18}{len(changed):>8}{len(dropped):>8}{float(percent):>8.2f}%  {status}")
        if args.show:
            print("  affected doc_ids: " + ", ".join(result["affected_ids"][:args.show]))
        if errors:
            print("  errors: " + ", ".join(result["error_types"]))
        debug_not_passing(name, details.get(name, {}))
    status = ("PASS", "FAIL", "ERROR")[worst]
    print("result: " + status)
    _write_report(args, args.rules, digest, n, limit, status, results)
    return worst
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run one cleaning rule on a gold set and report the harm.")
    ap.add_argument("--pack", required=True, type=Path)
    ap.add_argument("--gold", required=True, type=Path)
    ap.add_argument("--rules", type=Path, help="rules .py or medicine.md (not needed for a rule ID)")
    ap.add_argument("--rule", required=True,
                    help="rule ID such as DA-B3-001, a disease code such as B3, or all (every rule in --rules)")
    ap.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    ap.add_argument("--limit", default="1",
                    help="failing limit in percent of the gold set (default 1); at least 1 document is always allowed")
    ap.add_argument("--show", type=int, default=0, metavar="N", help="list the first N affected doc_ids")
    ap.add_argument("--timeout", type=float, default=5.0, help="seconds per document (default 5)")
    ap.add_argument("--report", type=Path, help="write reproducible JSON, IDs only")
    ap.add_argument("--debug", action="store_true",
                    help="on stderr: what the test does, and a line for each gold document that was changed, dropped "
                         "or raised an error (ids, lengths and offsets, never text)")
    args = ap.parse_args(argv)
    global DEBUG
    DEBUG = args.debug

    try:
        try:
            limit = Fraction(args.limit)
        except (ValueError, ZeroDivisionError) as e:
            raise Fatal(f"--limit must be a number from 0 through 100: {args.limit!r}") from e
        if not Fraction(0) <= limit <= Fraction(100):
            raise Fatal("--limit must be from 0 through 100")
        if not math.isfinite(args.timeout) or args.timeout <= 0:
            raise Fatal("--timeout must be finite and greater than zero")
        if args.show < 0:
            raise Fatal("--show must be nonnegative")
        if not HAS_ALARM:  # Windows has no SIGALRM: the faulthandler watchdog enforces --timeout instead
            print(f"note: no SIGALRM on this platform; a rule call over --timeout {args.timeout:g}s ends the whole "
                  "test (exit code 1) instead of being reported for one document. Add --debug to see which "
                  "document it was.", file=sys.stderr)
        code = args.rule.strip().upper()
        if args.rules is None and (code == "ALL" or not RULE_ID.fullmatch(code)):
            raise Fatal("--rule all or disease-code tests require --rules")
        if code == "ALL":
            return main_all(args, limit)

        ids = read_gold(args.gold)
        debug(f"gold: {len(ids)} doc_ids from {args.gold.name}")
        texts = read_pack_docs(args.pack, ids)
        debug(f"pack: {len(texts)} documents read from {args.pack.name}")
        if args.rules and not RULE_ID.fullmatch(code):
            check_one_section(args.rules, code)
        rules_path = args.rules or registry_file_for(code, args.registry)
        ns, digest = load_rules(rules_path)
        debug(f"rules file: {rules_path} (code sha256 {digest[:12]})")
        rule = find_rule(ns, code)
        debug(f"rule {code}: function {getattr(rule, '__name__', '<inline>')}, "
              f"gets the {'document dict' if takes_doc(rule) else 'text'}")
        detail = {} if DEBUG else None
        changed, dropped, errors = run_rule(rule, texts, args.timeout, detail)
        n = len(texts)
        result = _result(code, (changed, dropped, errors), n, limit)
        print(f"pack: {args.pack.name}   gold: {args.gold.name} ({n} documents)   rule: {code}")
        print(f"rules file: {rules_path.name}   code sha256: {digest}")
        print(f"documents changed: {len(changed)}")
        print(f"documents dropped: {len(dropped)}")
        print(f"percent: {float(result['percent']):.2f}% of the gold set")
        print(limit_line(n, limit, args.limit))
        if args.show:
            print("  affected doc_ids: " + ", ".join(result["affected_ids"][:args.show]))
        if errors:
            print("  errors: " + ", ".join(result["error_types"]))
        print("result: " + result["result"])
        debug_not_passing(code, detail or {})
        _write_report(args, rules_path, digest, n, limit, result["result"], [result])
        return {"PASS": 0, "FAIL": 1, "ERROR": 2}[result["result"]]
    except NotTestable as e:
        print(f"result: NOT TESTABLE ({e})")
        return 2
    except (Fatal, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
if __name__ == "__main__":
    sys.exit(main())
