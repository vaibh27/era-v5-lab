#!/usr/bin/env python3
"""Show the cleaned text behind each Spec v1.1 candidate, for the hand check and for writing its medicine.

Reads the collector's output (results/reading/diseases_by_domain.csv and findings_all.csv) and the cleaned
sample parts the readers read ("tokenizer sample v0-clean", Train parts). For every candidate in the chosen
domains (spec_v1_1 = yes) it writes each reader's row and the cleaned text at their offsets, with context.

The output holds pack text, so it must be written outside the repo (the lab repo is public): the script
refuses an --out inside it. F1 and E3 rows are never shown. Only counts are printed to the terminal.

From spec_v1_1/scripts/:
    python3 -I show_examples.py --domain math \\
        --parts <folder>/math-finemath*.parquet <folder>/math-megamath*.parquet \\
        --out ~/ERA5/Capstone/Week2-G1-work/spec_v1_1/examples/examples_math.md

Needs pyarrow (only to read the parts).
"""
import argparse
import csv
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent  # teams/01-data-a/spec_v1_1/scripts
REPO = HERE.parents[3]                  # era-v5-lab
READING = HERE.parent / "results" / "reading"
DOMAINS = ("english", "code", "math", "science")
PRIVATE_CODES = ("F1", "E3")            # type and position only, never the value
OPEN, CLOSE = "⟦", "⟧"        # marks around the reader's span


def visible(text: str) -> str:
    """Write the characters that do not show (controls, format characters, odd spaces) as <U+XXXX>."""
    out = []
    for ch in text:
        if ch in "\n\t " or unicodedata.category(ch)[0] not in "CZ":
            out.append(ch)
        else:
            out.append(f"<U+{ord(ch):04X}>")
    return "".join(out)


def parse_offsets(value: str):
    """'1204-1388' -> (1204, 1388). None if it is not a start-end pair."""
    start, sep, end = (value or "").strip().partition("-")
    start, end = start.strip(), end.strip()
    if not sep or not start.isdigit() or not end.isdigit():
        return None
    a, b = int(start), int(end)
    return (min(a, b), max(a, b))


def window(text: str, span, context: int) -> str:
    """The text of span with `context` characters on each side; the span itself is between the marks."""
    a, b = (max(0, min(x, len(text))) for x in span)
    lo, hi = max(0, a - context), min(len(text), b + context)
    head = "..." if lo > 0 else ""
    tail = "..." if hi < len(text) else ""
    return head + visible(text[lo:a]) + OPEN + visible(text[a:b]) + CLOSE + visible(text[b:hi]) + tail


def fence_for(body: str) -> str:
    """A code fence longer than any run of backticks in the body."""
    longest = run = 0
    for ch in body:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    return "`" * max(3, longest + 1)


def inside_repo(path: Path) -> bool:
    p = path.expanduser().resolve()
    return p == REPO or REPO in p.parents


def load_candidates(path: Path, domains):
    """The Spec v1.1 candidates of these domains, most docs first. F1 and E3 are left out."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f)
                if r["spec_v1_1"] == "yes" and r["domain"] in domains and r["code"] not in PRIVATE_CODES]
    return sorted(rows, key=lambda r: (DOMAINS.index(r["domain"]), -int(r["docs"]), r["code"]))


def load_findings(path: Path, keys):
    """The reading rows of each (domain, kind, code) in keys, in file order."""
    found = defaultdict(list)
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            key = (r["domain"], r["kind"], r["code"])
            if key in keys:
                found[key].append(r)
    return found


def load_texts(parts, doc_ids):
    """doc_id -> cleaned text, for the docs wanted. The first part that holds a doc wins."""
    import pyarrow.parquet as pq
    texts, wanted = {}, sorted(doc_ids)
    for part in parts:
        table = pq.read_table(part, columns=["doc_id", "text"], filters=[("doc_id", "in", wanted)])
        for doc_id, text in zip(table.column("doc_id").to_pylist(), table.column("text").to_pylist()):
            texts.setdefault(doc_id, text)
    return texts


def render(candidates, findings, texts, parts, context: int):
    """The Markdown file, and the doc_ids that were not in the parts."""
    lines = [
        "# Spec v1.1 candidates: the cleaned text behind each reading row",
        "",
        "Local working file with pack text: do not commit or share it. Offsets are the readers' own, in the cleaned text.",
        f"The reader's span is between {OPEN} and {CLOSE}. Characters that do not show are written <U+XXXX>.",
        "",
        "Parts: " + ", ".join(Path(p).name for p in parts),
        "",
    ]
    missing = set()
    for c in candidates:
        rows = findings.get((c["domain"], c["kind"], c["code"]), [])
        lines += [f"## {c['domain']} · {c['code']} · {c['docs']} docs, {c['percent']}% of {c['docs_read']} read · "
                  f"{c['readers']}", ""]
        for r in rows:
            lines.append(f"### `{r['doc_id']}` · {r['pack']} · {r['reader']} · offsets {r['offsets'] or '-'} · "
                         f"source {r['source'] or '-'}")
            lines.append("")
            if r["why"]:
                lines.append(f"Why: {r['why']}")
            if r["quote"]:
                lines.append(f"Quote: {visible(r['quote'])}")
            text = texts.get(r["doc_id"])
            if text is None:
                missing.add(r["doc_id"])
                lines += ["", "(This doc is not in the parts given.)", ""]
                continue
            span = parse_offsets(r["offsets"])
            if span is None or span[0] >= len(text):
                lines.append(f"(The offsets do not fit this {len(text)}-character text; its start is shown.)")
                span = (0, 0)
            body = window(text, span, context)
            fence = fence_for(body)
            lines += ["", fence + "text", body, fence, ""]
    if missing:
        lines += ["## Docs not found in the parts", ""] + [f"- `{d}`" for d in sorted(missing)] + [""]
    return "\n".join(lines), missing


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parts", nargs="+", required=True, type=Path, help="cleaned sample Parquet parts (Train)")
    ap.add_argument("--out", required=True, type=Path, help="Markdown file to write, outside the repo")
    ap.add_argument("--domain", action="append", choices=DOMAINS, help="repeat for more than one; default all")
    ap.add_argument("--reading", type=Path, default=READING, help="the collector's output folder")
    ap.add_argument("--context", type=int, default=300, help="characters shown on each side of the span (default 300)")
    args = ap.parse_args(argv)
    if inside_repo(args.out):
        ap.error(f"--out {args.out} is inside the repo; pack text must stay out of it")
    domains = set(args.domain or DOMAINS)
    candidates = load_candidates(args.reading / "diseases_by_domain.csv", domains)
    keys = {(c["domain"], c["kind"], c["code"]) for c in candidates}
    findings = load_findings(args.reading / "findings_all.csv", keys)
    doc_ids = {r["doc_id"] for rows in findings.values() for r in rows}
    texts = load_texts(args.parts, doc_ids)
    text, missing = render(candidates, findings, texts, args.parts, args.context)
    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    n_rows = sum(len(v) for v in findings.values())
    print(f"{len(candidates)} candidates, {n_rows} rows, {len(doc_ids)} docs; "
          f"{len(doc_ids) - len(missing)} found in {len(args.parts)} parts -> {out}")
    if missing:
        print(f"  not found: {len(missing)} docs (listed at the end of the file)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
