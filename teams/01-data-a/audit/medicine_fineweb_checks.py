"""Checks for the four medicines of the en-fineweb-edu audit (en-fineweb-edu-audit-Deepjyoti_Saha.md, "Medicines").

The medicines are read from the report's python blocks, one module per block, as harmtest/harm_test.py reads them.

  speed    time each medicine against the rule it replaces (spec_v1/clean.py) on whole parquet files, count what each
           does, and check that a second run of the medicine changes nothing (idempotence). DA-C1-015 is new, so it is
           timed against _c1_loop below, a line-by-line version with the same output, and the outputs are compared.
  treated  what each medicine does to the 35 audit documents (raw text), next to the rule it replaces, and all four
           in run order.

Pack text is never printed or written: only doc_ids, counts and times.

  python -I teams/01-data-a/audit/medicine_fineweb_checks.py speed --files <pre>/en-fineweb-edu/Part_1/part-00000.parquet ...
  python -I teams/01-data-a/audit/medicine_fineweb_checks.py treated --pre-root <pre> --labels teams/01-data-a/audit
"""
import argparse, csv, json, os, platform, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent
REPORT = HERE / "en-fineweb-edu-audit-Deepjyoti_Saha.md"
ORDER = ("DA-C1-015", "DA-C3-009", "DA-C5-008", "DA-F1-010")      # run order: C1 first in stage 30, F1 last
REPLACES = {"DA-C3-009": "_english_c3_003", "DA-C5-008": "_english_c5_002", "DA-F1-010": "_english_f1_005"}


def load_medicines(path=REPORT):
    """Each ```python block under a '## Medicine' heading, run in its own namespace: RULE_ID -> medicine(doc)."""
    text = path.read_text(encoding="utf-8")
    rules = {}
    for sec in re.split(r"(?m)^## Medicine \d", text)[1:]:
        for block in re.findall(r"```python\n(.*?)```", sec, re.S):
            ns = {"__name__": "medicine_card"}
            exec(compile(block, f"{path.name}:{ns.get('RULE_ID', '?')}", "exec"), ns)
            rules[ns["RULE_ID"]] = ns["medicine"]
    missing = [r for r in ORDER if r not in rules]
    if missing:
        sys.exit(f"no python block for {missing} in {path}")
    return rules


def old_rules():
    sys.path.insert(0, str(TEAM / "spec_v1"))
    import clean
    return {rid: getattr(clean, name)() for rid, name in REPLACES.items()}


def _c1_loop(rules):
    """DA-C1-015 written as a plain loop over lines, without its key-phrase gate: the reference for 'same output'."""
    ns = {}
    exec(compile(_block_source("DA-C1-015"), "c1", "exec"), ns)
    line_re = re.compile(ns["C1_LINE"].pattern.replace("(?mi)^", "(?i)^").replace("$\\n?", "$"))

    def medicine(doc):
        if doc.get("pack_id") not in ns["SPEC_PACKS"] or doc.get("lang_label") not in ns["LANGS"]:
            return ("keep", doc["text"])
        text = ns["C1_AD"].sub("", doc["text"])
        text = ns["C1_RELATED"].sub("", text)
        lines = text.split("\n")
        kept = []
        for line in lines:
            if not line_re.fullmatch(line):
                kept.append(line)
        new = "\n".join(kept)
        # the regex removes a line with the line break after it; a removed last line has none, so the break before
        # it stays
        if kept and len(kept) < len(lines) and line_re.fullmatch(lines[-1]):
            new += "\n"
        return ns["_std"](new, doc["text"])
    return medicine


def _block_source(rid, path=REPORT):
    text = path.read_text(encoding="utf-8")
    for block in re.findall(r"```python\n(.*?)```", text, re.S):
        if f'RULE_ID = "{rid}"' in block:
            return block
    raise KeyError(rid)


def cpu_name():
    """The CPU's marketing name where the OS gives it, else platform.processor()."""
    try:
        if sys.platform == "win32":
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as k:
                return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
        if sys.platform == "darwin":
            import subprocess
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def read_docs(path):
    import pyarrow.parquet as pq
    t = pq.read_table(path)
    cols = [c for c in ("doc_id", "text", "url", "lang_label", "pack_id") if c in t.column_names]
    docs = t.select(cols).to_pylist()
    for d in docs:
        d.setdefault("pack_id", "en-fineweb-edu")
        d.setdefault("lang_label", "en")
    return docs


def run(rule, docs):
    t0 = time.perf_counter()
    out = [rule(d) for d in docs]
    return time.perf_counter() - t0, out


def speed(args):
    t_start = time.perf_counter()
    new = load_medicines(Path(args.report))
    old = old_rules()
    old["DA-C1-015"] = _c1_loop(new)
    lines, summary = [], {"machine": f"{cpu_name()} (CPU), Python {platform.python_version()}, "
                                     f"{platform.system()} {platform.release()}, one process", "files": []}
    lines.append(f"rules loaded: {time.perf_counter() - t_start:.2f} s after start")
    lines.append(summary["machine"])
    tot_docs = tot_chars = 0
    idem_bad = same_bad = 0
    for f in args.files:
        t0 = time.perf_counter()
        docs = read_docs(f)
        chars = sum(len(d["text"] or "") for d in docs)
        tot_docs += len(docs); tot_chars += chars
        lines.append("")
        lines.append(f"{Path(f).parent.name}/{Path(f).name}: {len(docs):,} docs, {chars / 1e6:.1f} M chars, read {time.perf_counter() - t0:.2f} s")
        lines.append(f"  {'rule':<12}{'replaces':<12}{'old s':>8}{'new s':>8}{'ratio':>7}{'docs/s new':>12}"
                     f"{'old fix':>9}{'old drop':>9}{'new fix':>9}{'new drop':>9}{'differ':>8}{'not idem':>9}")
        frec = {"file": f"{Path(f).parent.name}/{Path(f).name}", "docs": len(docs), "chars": chars, "rules": {}}
        for rid in ORDER:
            t_old, o_out = run(old[rid], docs)
            t_new, n_out = run(new[rid], docs)
            differ = sum(1 for a, b in zip(o_out, n_out) if a != b)
            # idempotence: run the medicine again on every text it fixed
            fixed = [dict(d, text=o[1]) for d, o in zip(docs, n_out) if o[0] == "fix"]
            not_idem = sum(1 for d in fixed if new[rid](d)[0] != "keep")
            idem_bad += not_idem
            if rid == "DA-C1-015":
                same_bad += differ
            cnt = lambda out, a: sum(1 for o in out if o[0] == a)
            rep = REPLACES.get(rid, "loop ref").replace("_english_", "DA-").replace("_", "-").upper() if rid in REPLACES else "loop ref"
            lines.append(f"  {rid:<12}{rep:<12}{t_old:>8.2f}{t_new:>8.2f}{t_old / t_new if t_new else 0:>6.1f}x"
                         f"{len(docs) / t_new if t_new else 0:>12,.0f}{cnt(o_out, 'fix'):>9}{cnt(o_out, 'drop'):>9}"
                         f"{cnt(n_out, 'fix'):>9}{cnt(n_out, 'drop'):>9}{differ:>8}{not_idem:>9}")
            frec["rules"][rid] = dict(replaces=rep, old_s=round(t_old, 3), new_s=round(t_new, 3),
                                      old_fix=cnt(o_out, "fix"), old_drop=cnt(o_out, "drop"),
                                      new_fix=cnt(n_out, "fix"), new_drop=cnt(n_out, "drop"),
                                      differ=differ, not_idempotent=not_idem)
        # all four in run order: a drop ends the document, a fix passes its text on
        t0 = time.perf_counter()
        n_fix = n_drop = 0
        for d in docs:
            cur, changed = dict(d), False
            for rid in ORDER:
                a, t = new[rid](cur)
                if a == "drop":
                    n_drop += 1; break
                if a == "fix":
                    cur["text"], changed = t, True
            else:
                n_fix += changed
        t_all = time.perf_counter() - t0
        lines.append(f"  {'all four, in order':<24}{t_all:>16.2f}{len(docs) / t_all:>19,.0f}{'':>18}{n_fix:>9}{n_drop:>9}")
        frec["in_order"] = dict(s=round(t_all, 3), fix=n_fix, drop=n_drop)
        summary["files"].append(frec)
    wall = time.perf_counter() - t_start
    lines.append("")
    lines.append(f"ALL: {tot_docs:,} docs, {tot_chars / 1e6:.1f} M chars; wall time {wall:.1f} s from start; "
                 f"DA-C1-015 docs that differ from its loop version: {same_bad}; docs a second run changes: {idem_bad}")
    summary.update(docs=tot_docs, chars=tot_chars, wall_s=round(wall, 1), c1_differ=same_bad, not_idempotent=idem_bad)
    out = "\n".join(lines)
    print(out)
    if args.log:
        Path(args.log).write_text(out + "\n", encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return 0 if same_bad == 0 and idem_bad == 0 else 1


def treated(args):
    new = load_medicines(Path(args.report))
    old = old_rules()
    labels = {}
    for p in ("Part_1", "Part_2"):
        for r in csv.DictReader(open(Path(args.labels) / f"en-fineweb-edu-{p}-Deepjyoti_Saha.csv", encoding="utf-8")):
            labels[r["doc_id"]] = (p, r)
    docs = {}
    for p, fn in (("Part_1", "part-00000.parquet"), ("Part_2", "part-00001.parquet")):
        for d in read_docs(Path(args.pre_root) / "en-fineweb-edu" / p / fn):
            if d["doc_id"] in labels:
                docs[d["doc_id"]] = d
    lines = [f"audit docs: {len(labels)}; found in the raw files: {len(docs)}", ""]
    res = {}
    for rid in ORDER:
        rows = []
        for did, d in docs.items():
            a_new = new[rid](d)[0]
            a_old = old[rid](d)[0] if rid in old else "-"
            if a_new != "keep" or a_old not in ("keep", "-"):
                rows.append((did, a_old, a_new))
        res[rid] = rows
        lines.append(f"{rid} (replaces {REPLACES[rid].replace('_english_', 'DA-').replace('_', '-').upper()})" if rid in REPLACES else f"{rid} (new)")
        for did, a_old, a_new in rows:
            lines.append(f"  {labels[did][0]} {did}  old {a_old:<5} new {a_new}")
        lines.append("")
    # in run order
    lines.append("all four in run order (raw text):")
    for did, d in docs.items():
        cur, acts = dict(d), []
        for rid in ORDER:
            a, t = new[rid](cur)
            if a != "keep":
                acts.append(f"{rid} {a}")
            if a == "drop":
                break
            if a == "fix":
                cur["text"] = t
        if acts:
            lines.append(f"  {labels[did][0]} {did}  " + ", ".join(acts))
    out = "\n".join(lines)
    print(out)
    if args.log:
        Path(args.log).write_text(out + "\n", encoding="utf-8")
    if args.json:
        Path(args.json).write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--report", default=str(REPORT))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("speed"); s.add_argument("--files", nargs="+", required=True)
    s.add_argument("--log"); s.add_argument("--json")
    t = sub.add_parser("treated"); t.add_argument("--pre-root", required=True); t.add_argument("--labels", default=str(HERE))
    t.add_argument("--log"); t.add_argument("--json")
    args = ap.parse_args(argv)
    return speed(args) if args.cmd == "speed" else treated(args)


if __name__ == "__main__":
    sys.exit(main())
