#!/usr/bin/env python3
"""Build reading/reading_check_v1.md: the Week 2 reading audits of the four domains in one file.

Reads the collector's output (spec_v1_1/results/reading/summary.md, diseases_by_domain.csv and findings_all.csv)
and spec_v1_1/sheets/NEW_DISEASES.md. Writes, for each domain (english, code, math, science):
  - the docs read and who read them;
  - the diseases Spec v1 missed;
  - the new diseases;
  - the harm rows;
and then the new diseases at or above the 1% bar. It runs no rule and reads no pack, so run collect_reading_prs.py
first to bring the data up to date.

The rate of a disease is the number of distinct docs it was found in, divided by the docs read in its domain, x 100.

From spec_v1_1/scripts/:
    python3 -I build_reading_check.py
"""
import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent  # teams/01-data-a/spec_v1_1/scripts
READING = HERE.parent / "results" / "reading"
REGISTER = HERE.parent / "sheets" / "NEW_DISEASES.md"
DEFAULT_OUT = HERE.parents[1] / "reading" / "reading_check_v1.md"
DOMAINS = ("english", "code", "math", "science")
FAMILY_TO_DOMAIN = {"english": "english", "code": "code", "math": "math", "sci": "science"}
PACK_PREFIX = (("en-", "english"), ("code-", "code"), ("math-", "math"), ("sci-", "science"))
DISEASES_MD_CODE = re.compile(r"[A-H]\d")  # a code from diagnosis/DISEASES.md (A1-H3) or a new one in the same form
BAR = 1.0  # percent of the docs read in the domain


def cells(line):
    """The cells of one Markdown table row; an escaped pipe stays inside its cell."""
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", line.strip().strip("|"))]


def md_table(text, heading):
    """The rows of the first table after the line `heading`, as dicts keyed by the header cells."""
    lines = text.split("\n")
    start = next((k for k, line in enumerate(lines) if line.strip() == heading), None)
    if start is None:
        return []
    j = start + 1
    while j < len(lines) and not lines[j].startswith("|"):
        if lines[j].startswith("## "):
            return []  # the next section began before any table
        j += 1
    block = []
    while j < len(lines) and lines[j].startswith("|"):
        block.append(lines[j])
        j += 1
    if len(block) < 2:
        return []
    head = cells(block[0])
    return [dict(zip(head, cells(line))) for line in block[2:]]


def plain(text):
    return re.sub(r"[*`]", "", text or "").strip()


def domain_of_pack(pack):
    for prefix, domain in PACK_PREFIX:
        if pack.startswith(prefix):
            return domain
    return None


def read_register(path):
    """From NEW_DISEASES.md: {(domain, reading code): (code, rule ID)} for new diseases, and
    {(domain, reading code): DISEASES.md code} for the rows under "Not new". The Indic family is left out."""
    if not path.exists():
        return {}, {}
    text = path.read_text(encoding="utf-8")
    new, not_new = {}, {}
    for r in md_table(text, "## Summary: new diseases"):
        domain = FAMILY_TO_DOMAIN.get(plain(r.get("family")))
        if domain:
            new[(domain, plain(r.get("reading code")))] = (plain(r.get("code")).split()[0], plain(r.get("rule ID")))
    for r in md_table(text, "## Not new: the disease is already in DISEASES.md"):
        domain = FAMILY_TO_DOMAIN.get(plain(r.get("family")))
        if domain:
            not_new[(domain, plain(r.get("reading code")))] = plain(r.get("DISEASES.md code"))
    return new, not_new


def read_diseases(path):
    """The collector's rows, with the rate as a number."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["docs"], r["docs_read"] = int(r["docs"]), int(r["docs_read"] or 0)
        r["rate"] = 100.0 * r["docs"] / r["docs_read"] if r["docs_read"] else 0.0
    return rows


def classify(domain, code, new, not_new):
    """Where a row the reader marked new stands against DISEASES.md and NEW_DISEASES.md."""
    if (domain, code) in not_new:
        return "not new", f"in DISEASES.md as {not_new[(domain, code)]}"
    if DISEASES_MD_CODE.fullmatch(code):
        return "not new", f"in DISEASES.md ({code})"
    if (domain, code) in new:
        v11_code, rule = new[(domain, code)]
        return "new", f"{v11_code}, `{rule}`" if rule and not rule.startswith("(") else f"{v11_code}, no ID yet {rule}".strip()
    return "new", "not yet in NEW_DISEASES.md"


def pct(rate):
    return f"{rate:.1f}%"


def build(reading=READING, register=REGISTER):
    summary = (reading / "summary.md").read_text(encoding="utf-8")
    collected = re.search(r"collected (.+)$", summary.split("\n", 1)[0])
    collected = collected.group(1) if collected else "unknown time"
    docs_read = {r["domain"]: r for r in md_table(summary, "## Docs read per domain")}
    missing = {d: [] for d in DOMAINS}
    for r in md_table(summary, "## README rows"):
        if "none yet" in r.get("reading files", ""):
            domains = {domain_of_pack(p.strip()) for p in r.get("packs", "").split(",")}
            for d in domains - {None}:
                missing[d].append(f"{r['reader']} ({r['packs']})")
    prs = md_table(summary, "## Open reading PRs")
    warnings = [line[2:] for line in summary.split("## Warnings", 1)[-1].split("\n") if line.startswith("- ")] \
        if "## Warnings" in summary else []
    rows = read_diseases(reading / "diseases_by_domain.csv")
    new, not_new = read_register(register)
    sources = Counter()
    findings = reading / "findings_all.csv"
    if findings.exists():
        with open(findings, newline="", encoding="utf-8") as f:
            sources = Counter((r.get("source") or "blank").strip() or "blank" for r in csv.DictReader(f))

    out = [
        "# reading_check_v1: the Week 2 reading audits of the cleaned sample, all four domains",
        "",
        f"**Built** by `spec_v1_1/scripts/build_reading_check.py` from the collector run of {collected} "
        "(`spec_v1_1/results/reading/`) and from `spec_v1_1/sheets/NEW_DISEASES.md`. Don't edit this file by hand: "
        "run the collector, then this script.",
        "",
        "**Kinds of row** (from the reading README):",
        "- **missed:** Spec v1 has a rule for the disease, and the cleaned doc still has it.",
        "- **new:** a disease that Spec v1 has no rule for.",
        "- **harm:** a Spec v1 rule damaged real content.",
        "",
        "**Occurrence rate:** the distinct docs with the disease, divided by the docs read in its domain, × 100.",
        "",
        f"**Flag:** a new disease at {BAR:g}% or more, which needs a medicine and a harm test for Spec v1.1. "
        "\"New\" means not in `diagnosis/DISEASES.md`. A row the reader marked new that `DISEASES.md` already "
        "describes is shown, but not flagged.",
        "",
        "## Audits per domain",
        "",
        "| domain | docs read | readers with a reading file | readers with no file yet |",
        "|---|---|---|---|",
    ]
    for d in DOMAINS:
        r = docs_read.get(d)
        out.append(f"| {d} | {r['docs read'] if r else 0} | {r['readers'] if r else 'none'} | "
                   f"{'; '.join(missing[d]) or 'none'} |")
    out += ["", "Reading PRs used: " + (", ".join(plain(p.get("PR")).split("](")[0].lstrip("[") for p in prs) or "none") + ".", ""]

    def section(title, kind, columns, render_row, note=""):
        out.extend([f"## {title}", ""] + ([note, ""] if note else []))
        for d in DOMAINS:
            out.append(f"### {d}")
            out.append("")
            if d not in docs_read:
                out.extend(["No audit has been submitted for this domain.", ""])
                continue
            these = sorted((r for r in rows if r["domain"] == d and r["kind"] == kind),
                           key=lambda r: (-r["docs"], r["code"]))
            if not these:
                out.extend(["None found.", ""])
                continue
            out.extend(["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)])
            out.extend("| " + " | ".join(render_row(r)) + " |" for r in these)
            out.append("")

    section("1. Diseases missed by Spec v1", "missed",
            ["code", "docs", "rate", "readers", "Spec v1 rules for this family"],
            lambda r: [r["code"], str(r["docs"]), pct(r["rate"]), r["readers"], r["spec_v1_rules"] or "none"],
            "A disease Spec v1 has a rule for, still in the cleaned doc. A code with no Spec v1 rule for the "
            "family (\"none\") is a gap in Spec v1's pack scope.")
    section("2. New diseases", "new",
            ["code", "docs", "rate", f"≥ {BAR:g}%", "status", "Spec v1.1 code and rule"],
            lambda r: [r["code"], str(r["docs"]), pct(r["rate"]), "yes" if r["rate"] >= BAR else "no",
                       *classify(r["domain"], r["code"], new, not_new)])
    section("3. Harm: Spec v1 rules that damaged real content", "harm",
            ["rule", "docs", "rate", "readers", "Spec v1 rule"],
            lambda r: [r["code"], str(r["docs"]), pct(r["rate"]), r["readers"], r["spec_v1_rules"] or "none"])

    flagged = [r for r in rows if r["kind"] == "new" and r["rate"] >= BAR
               and classify(r["domain"], r["code"], new, not_new)[0] == "new"]
    shown = [r for r in rows if r["kind"] == "new" and r["rate"] >= BAR
             and classify(r["domain"], r["code"], new, not_new)[0] == "not new"]
    out += [f"## 4. Flagged: new diseases at {BAR:g}% or more", ""]
    if flagged:
        out += ["| domain | code | docs | docs read | rate | Spec v1.1 code and rule |", "|---|---|---|---|---|---|"]
        for r in sorted(flagged, key=lambda r: (DOMAINS.index(r["domain"]), -r["docs"], r["code"])):
            out.append(f"| {r['domain']} | {r['code']} | {r['docs']} | {r['docs_read']} | {pct(r['rate'])} | "
                       f"{classify(r['domain'], r['code'], new, not_new)[1]} |")
    else:
        out.append("None.")
    out.append("")
    per_domain = Counter(r["domain"] for r in flagged)
    out.append("**Flagged per domain:** " + ", ".join(f"{d} {per_domain.get(d, 0)}" for d in DOMAINS) + ".")
    if shown:
        out += ["", f"**Marked new by the reader at {BAR:g}% or more, but already in `DISEASES.md`:** "
                + ", ".join(f"{r['domain']} {r['code']} ({pct(r['rate'])})" for r in shown) + "."]
    out += ["", "## Notes", ""]
    out.append("- **Rows by source:** " + (", ".join(f"{k} {v}" for k, v in sorted(sources.items())) or "none")
               + ". `ai` means an AI tool suggested the row and the reader checked it. Each row is a first pass "
                 "until a second reader has checked it.")
    if any(d not in docs_read for d in DOMAINS):
        out.append("- **Domains with no audit:** "
                   + ", ".join(d for d in DOMAINS if d not in docs_read) + ". They have no rates yet.")
    out += [f"- **Collector warning:** {w}" for w in warnings]
    return "\n".join(out) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reading", type=Path, default=READING, help="the collector's output folder")
    ap.add_argument("--register", type=Path, default=REGISTER, help="NEW_DISEASES.md")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="the file to write (default: reading/reading_check_v1.md)")
    args = ap.parse_args(argv)
    text = build(args.reading, args.register)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"{args.out}: written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
