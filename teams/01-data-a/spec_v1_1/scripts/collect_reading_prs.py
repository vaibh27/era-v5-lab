#!/usr/bin/env python3
"""Collect the Week 2 reading audits from GitHub and count each disease per domain, for Spec v1.1.

The readers put reading_<pack>_<name>.csv and .md in teams/01-data-a/reading/ and open a PR to g1-reading in
mittalutkarsh/era-v5-lab (see that folder's README). Some PRs land in another repo by mistake, so this looks at the
open PRs of all three era-v5-lab repos whose branch starts with g1-reading- or whose title starts with "reading",
and at the files already merged into g1-reading.

From spec_v1_1/scripts/:

    python3 -I collect_reading_prs.py                       collect once
    python3 -I collect_reading_prs.py --watch 30            collect again every 30 minutes (Ctrl-C stops)
    python3 -I collect_reading_prs.py --docs-read pyru=199  a reader who read more or fewer than 100 docs
    python3 -I collect_reading_prs.py --alias NEW-missing-formula=NEW-missing-formulas
                                                            count two readers' names for a disease as one

What it writes (default: spec_v1_1/results/reading/; the spec v1 rules come from spec_v1/sheets/specv1.md):
    findings_all.csv         every row of every reading CSV, with domain, pack, reader and where it came from
    diseases_by_domain.csv   per domain, kind and code: distinct docs, % of the docs read, readers, example
                             doc_ids, and the spec v1 rules for that code in that family (from specv1.md)
    summary.md               the PRs found, the README rows with no reading yet, and the Spec v1.1 list: new
                             diseases in 1% or more of the docs read in a domain (--min-percent; the README's
                             example is 2+ docs out of 200, so 2 of 200 counts and 2 of 299 does not)
    raw/                     the downloaded files, named by git blob SHA, so an unchanged file is not fetched again
    state.json               the PR heads of the last run, to print what changed since then

Which copy of a file counts: the reader's own open PR (its branch ends in their name), else the merged copy on
g1-reading, else any other open PR that carries it. Downloaded files are only read as CSV, never run.
The quote column is left empty for F1 and E3 rows: type and offset only, never the value.

Token: GH_TOKEN or GITHUB_TOKEN, else `gh auth token`, else none (60 API requests an hour; a run uses about 10).
Exit code: 0 = done, 2 = GitHub could not be read.
"""
import argparse
import csv
import io
import json
import math
import os
import re
import ssl
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPOS = ("mittalutkarsh/era-v5-lab", "theschoolofai/era-v5-lab", "patilvijay23/era-v5-lab")
READING_REPO, READING_BRANCH = "mittalutkarsh/era-v5-lab", "g1-reading"
READING_DIR = "teams/01-data-a/reading"
READING_FILE = re.compile(r"^teams/01-data-a/reading/(?P<name>reading_(?P<pack>.+)_(?P<reader>[^_/]+)\.(?P<ext>csv|md))$")
COLUMNS = ("doc_id", "kind", "code", "offsets", "quote", "why", "source")
KINDS = ("new", "missed", "harm")
PREFIXES = (("en-", "english"), ("code-", "code"), ("math-", "math"), ("sci-", "science"))
DOMAIN_WORDS = {"english": "english", "en": "english", "code": "code", "math": "math", "maths": "math",
                "science": "science", "sci": "science"}
SPEC_FAMILY = {"english": "english", "code": "code", "math": "math", "science": "sci"}
DOCS_PER_READER = 100  # README: "100 docs per person"
PRIVATE_CODES = ("F1", "E3")
HERE = Path(__file__).resolve().parent  # teams/01-data-a/spec_v1_1/scripts
DEFAULT_OUT = HERE.parent / "results" / "reading"
DEFAULT_SPEC = HERE.parents[1] / "spec_v1" / "sheets" / "specv1.md"
API = "https://api.github.com"
SPEC_HEADING = re.compile(r"^## Medicine \d+ — (?P<family>[\w-]+) / (?P<code>[A-Z]\d) / `(?P<rule>[^`]+)`", re.M)
STATUS_LINE = re.compile(r'^STATUS = "([A-Z]+)"', re.M)
PACK_ID = re.compile(r"\b(?:en|code|math|sci|ind)-[a-z0-9]+(?:-[a-z0-9]+)*")


# ---------- reading the files (no network) ----------

def domain_of(pack: str) -> str:
    """en-dclm -> english, code-go -> code, math-megamath -> math, sci-finepdfs -> science; 'english' -> english."""
    p = pack.strip().lower()
    for prefix, domain in PREFIXES:
        if p.startswith(prefix):
            return domain
    return DOMAIN_WORDS.get(p, "unknown")


def normalize_code(code: str) -> str:
    """NEW-Missing Formulas -> NEW-missing-formulas; c1 -> C1; da-c2-001 -> DA-C2-001."""
    c = code.strip()
    m = re.match(r"(?i)^new(?![a-z0-9])[-_ :]*(.*)$", c)
    if m:
        slug = re.sub(r"[^a-z0-9]+", "-", m.group(1).lower()).strip("-")
        return "NEW-" + (slug or "unnamed")
    return c.upper()


def normalize_doc_id(doc_id: str) -> str:
    """'[1/500] doc_id=0ca18aa03db2ef77' -> '0ca18aa03db2ef77'."""
    d = doc_id.strip()
    m = re.search(r"doc_id\s*=\s*(\S+)", d)
    return m.group(1) if m else d


def is_private(code: str) -> bool:
    """F1, E3 and their rule IDs (DA-F1-005): never keep a quote for these."""
    return any(code == p or f"-{p}-" in code for p in PRIVATE_CODES)


def decode(data: bytes):
    """Return (text, warning or None). The CSVs are meant to be UTF-8; a byte-order mark is allowed."""
    try:
        return data.decode("utf-8-sig"), None
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace"), "not UTF-8, read as Windows-1252"


def parse_reading_csv(text: str, pack: str, reader: str):
    """Return (rows, warnings). A `pack` column, if there is one, beats the pack in the file name."""
    table = csv.DictReader(io.StringIO(text))
    fields = {(f or "").strip().lower(): f for f in (table.fieldnames or [])}
    missing = [c for c in COLUMNS if c not in fields]
    if missing:
        return [], [f"missing columns: {', '.join(missing)}"]
    rows, warnings = [], []
    for raw in table:
        def get(column):
            return (raw.get(fields[column]) or "").strip() if column in fields else ""
        doc_id = normalize_doc_id(get("doc_id"))
        if not doc_id:
            continue
        kind, code, row_pack = get("kind").lower(), normalize_code(get("code")), get("pack") or pack
        if kind not in KINDS:
            warnings.append(f"line {table.line_num}: kind {kind!r} is not new, missed or harm")
        rows.append({"domain": domain_of(row_pack), "pack": row_pack, "reader": reader, "doc_id": doc_id,
                     "kind": kind, "code": code, "offsets": get("offsets"),
                     "quote": "" if is_private(code) else get("quote"), "why": get("why"), "source": get("source")})
    return rows, warnings


def choose(candidates):
    """One copy per file name: {name: (chosen, others)}. The reader's own open PR, then g1-reading, then the rest."""
    def rank(c):
        own = c["pr"] is not None and c["head_ref"].lower().endswith("-" + c["reader"].lower())
        return (0 if own else 1 if c["pr"] is None else 2, -c["updated_ts"])
    by_name = defaultdict(list)
    for c in candidates:
        by_name[c["name"]].append(c)
    out = {}
    for name, copies in by_name.items():
        ordered = sorted(copies, key=rank)
        out[name] = (ordered[0], ordered[1:])
    return out


def parse_spec(text: str):
    """specv1.md -> ({(family, code): ['DA-C1-004 CANDIDATE', ...]}, {rule_id: 'math CANDIDATE'})."""
    by_code, by_id = defaultdict(list), {}
    heads = list(SPEC_HEADING.finditer(text))
    for i, h in enumerate(heads):
        block = text[h.end(): heads[i + 1].start() if i + 1 < len(heads) else len(text)]
        s = STATUS_LINE.search(block)
        status = s.group(1) if s else "?"
        by_code[(h["family"], h["code"])].append(f"{h['rule']} {status}")
        by_id[h["rule"]] = f"{h['family']} {status}"
    return dict(by_code), by_id


def parse_readme_rows(text: str):
    """The README's 'Who reads what' table: [(row, reader, [packs])]."""
    out = []
    for line in text.splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and cells[0].isdigit():
            packs = PACK_ID.findall(cells[2])
            if packs:
                out.append((int(cells[0]), cells[1], packs))
    return out


def match_rows(readme_rows, files):
    """For each README row, the reading files that are its reader's: by name, or by a pack only that row reads.
    files: [(name, reader, {packs})]. Returns [(row, reader, packs, [names])]."""
    pack_rows = defaultdict(int)
    for _, _, packs in readme_rows:
        for p in packs:
            pack_rows[p] += 1
    out = []
    for row, reader, packs in readme_rows:
        tokens = set(re.findall(r"[a-z]+", reader.lower()))
        names = [name for name, file_reader, file_packs in files
                 if file_reader.lower() in tokens
                 or any(p in packs and pack_rows[p] == 1 for p in file_packs)]
        out.append((row, reader, packs, sorted(names)))
    return out


def summarize(rows, docs_read, min_percent, spec_by_code, spec_by_id):
    """One line per (domain, kind, code), most docs first. Flagged at min_percent or more of the docs read in the
    domain; a domain with no docs-read count is never flagged."""
    docs, readers = defaultdict(set), defaultdict(set)
    for r in rows:
        key = (r["domain"], r["kind"], r["code"])
        docs[key].add((r["pack"], r["doc_id"]))
        readers[key].add(r["reader"])
    out = []
    for (domain, kind, code), ids in docs.items():
        family = SPEC_FAMILY.get(domain)
        if kind == "harm":
            rules = [f"{code} {spec_by_id[code]}"] if code in spec_by_id else []
        elif code.startswith("NEW-"):
            rules = []
        else:
            rules = spec_by_code.get((family, code), [])
        n_read = docs_read.get(domain, 0)
        flag = ""
        if n_read and 100 * len(ids) >= min_percent * n_read:
            if kind == "new":
                flag = "yes"
            elif kind == "missed" and not rules:
                flag = f"check: no {family} rule"
        out.append({"domain": domain, "kind": kind, "code": code, "docs": len(ids), "docs_read": n_read,
                    "percent": f"{100 * len(ids) / n_read:.1f}" if n_read else "",
                    "readers": " ".join(sorted(readers[(domain, kind, code)])),
                    "spec_v1_rules": "; ".join(rules) or "none", "spec_v1_1": flag,
                    "example_doc_ids": " ".join(sorted(d for _, d in ids)[:3])})
    order = {k: i for i, k in enumerate(KINDS)}
    return sorted(out, key=lambda x: (x["domain"], order.get(x["kind"], 9), -x["docs"], x["code"]))


# ---------- GitHub ----------

def ssl_context():
    try:
        import certifi  # python.org Python on macOS has no certificate bundle of its own
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def find_token():
    for name in ("GH_TOKEN", "GITHUB_TOKEN"):
        if os.environ.get(name, "").strip():
            return os.environ[name].strip()
    try:
        out = subprocess.run(["gh", "auth", "token"], capture_output=True, text=True, timeout=10)
        return (out.stdout.strip() or None) if out.returncode == 0 else None
    except (OSError, subprocess.TimeoutExpired):
        return None


def next_link(link_header: str):
    for part in link_header.split(","):
        m = re.match(r'\s*<([^>]+)>;\s*rel="next"', part)
        if m:
            return m.group(1)
    return None


class GitHub:
    def __init__(self, token):
        self.token, self.ctx, self.remaining = token, ssl_context(), None

    def get(self, url):
        req = urllib.request.Request(url, headers={"Accept": "application/vnd.github+json",
                                                   "User-Agent": "collect-reading-prs"})
        if self.token and url.startswith(API):
            req.add_header("Authorization", f"Bearer {self.token}")
        with urllib.request.urlopen(req, context=self.ctx, timeout=60) as resp:
            if url.startswith(API):
                self.remaining = resp.headers.get("X-RateLimit-Remaining")
            return resp.read(), resp.headers

    def pages(self, path, params=None):
        url = f"{API}{path}?{urllib.parse.urlencode({**(params or {}), 'per_page': 100})}"
        items = []
        while url:
            body, headers = self.get(url)
            data = json.loads(body)
            if not isinstance(data, list):
                raise ValueError(f"{path}: expected a list from GitHub")
            items.extend(data)
            url = next_link(headers.get("Link", ""))
        return items


def is_reading_pr(pr) -> bool:
    head = ((pr.get("head") or {}).get("ref") or "").lower()
    title = (pr.get("title") or "").strip().lower()
    return head.startswith("g1-reading-") or title.startswith("reading") or "g1 reading" in title


def iso_ts(value: str) -> float:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() if value else 0.0


def fetch(gh, warnings):
    """Return (prs, candidates, readme_text)."""
    prs, candidates = [], []
    for repo in REPOS:
        for pr in gh.pages(f"/repos/{repo}/pulls", {"state": "open"}):
            if not is_reading_pr(pr):
                continue
            base = f"{repo}:{pr['base']['ref']}"
            info = {"from": f"{repo}#{pr['number']}", "url": pr["html_url"], "title": pr["title"],
                    "author": pr["user"]["login"], "head_ref": pr["head"]["ref"], "head_sha": pr["head"]["sha"],
                    "base": base, "files": []}
            if base != f"{READING_REPO}:{READING_BRANCH}":
                warnings.append(f"{info['from']}: base is {base}, not {READING_REPO}:{READING_BRANCH}")
            for f in gh.pages(f"/repos/{repo}/pulls/{pr['number']}/files"):
                m = READING_FILE.match(f["filename"])
                if not m or f["status"] == "removed" or m["name"].startswith("reading_check"):
                    continue
                info["files"].append(m["name"])
                candidates.append({"name": m["name"], "pack": m["pack"], "reader": m["reader"], "ext": m["ext"],
                                   "blob": f["sha"], "url": f["raw_url"], "from": info["from"], "pr": pr["number"],
                                   "head_ref": pr["head"]["ref"], "updated_ts": iso_ts(pr["updated_at"])})
            prs.append(info)
    readme = ""
    for item in gh.pages(f"/repos/{READING_REPO}/contents/{READING_DIR}", {"ref": READING_BRANCH}):
        if item.get("type") != "file":
            continue
        if item["name"] == "README.md":
            readme = gh.get(item["download_url"])[0].decode("utf-8", errors="replace")
            continue
        m = READING_FILE.match(f"{READING_DIR}/{item['name']}")
        if m and not m["name"].startswith("reading_check"):
            candidates.append({"name": m["name"], "pack": m["pack"], "reader": m["reader"], "ext": m["ext"],
                               "blob": item["sha"], "url": item["download_url"],
                               "from": f"{READING_REPO}@{READING_BRANCH}", "pr": None, "head_ref": "",
                               "updated_ts": 0.0})
    return prs, candidates, readme


def download(gh, c, raw_dir: Path) -> bytes:
    path = raw_dir / f"{c['blob'][:12]}_{c['name']}"
    if path.exists():
        return path.read_bytes()
    data = gh.get(c["url"])[0]
    path.write_bytes(data)
    return data


# ---------- writing ----------

def write_csv(path: Path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def md_table(header, rows):
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(x).replace("|", "\\|") for x in r) + " |" for r in rows]
    return "\n".join(lines)


def write_summary(path, when, prs, used, readme_matches, docs_read, readers_by_domain, overrides, table,
                  min_percent, warnings):
    out = [f"# Week 2 reading audits, collected {when}", ""]
    out += ["## Open reading PRs", ""]
    out += [md_table(["PR", "title", "author", "branch -> base", "reading files"],
                     [(f"[{p['from']}]({p['url']})", p["title"], p["author"], f"{p['head_ref']} -> {p['base']}",
                       ", ".join(p["files"]) or "none") for p in prs]) if prs else "None.", ""]
    out += ["## Files used", "", "One copy per file: the reader's own open PR, else g1-reading, else another PR.", ""]
    out += [md_table(["file", "from", "rows", "also in"],
                     [(n, src, rows if rows is not None else "-", ", ".join(o) or "") for n, src, rows, o in used]),
            ""]
    if readme_matches:
        out += ["## README rows", ""]
        out += [md_table(["row", "reader", "packs", "reading files"],
                         [(r, rd, ", ".join(p), ", ".join(n) or "**none yet**") for r, rd, p, n in readme_matches]),
                ""]
    out += ["## Docs read per domain", "",
            f"{DOCS_PER_READER} per reader unless `--docs-read` says otherwise. Used for the % column and the "
            "Spec v1.1 list.", ""]
    out += [md_table(["domain", "docs read", "readers"],
                     [(d, docs_read[d], ", ".join(f"{r} {overrides.get(r.lower(), DOCS_PER_READER)}"
                                                 for r in sorted(readers_by_domain[d])))
                      for d in sorted(docs_read)]), ""]
    flagged = [t for t in table if t["spec_v1_1"]]
    out += [f"## Spec v1.1 list: {min_percent:g}% or more of the docs read in a domain", "",
            "`yes`: a new disease. `check`: a missed disease with no rule for it in this family's spec v1.", ""]
    out += [md_table(["domain", "docs read", "docs needed"],
                     [(d, docs_read[d], math.ceil(min_percent * docs_read[d] / 100)) for d in sorted(docs_read)]),
            ""]
    for domain in sorted({t["domain"] for t in flagged}):
        out += [f"### {domain}", ""]
        out += [md_table(["code", "kind", "docs", "% of docs read", "readers", "spec v1 rules", "flag",
                          "example doc_ids"],
                         [(t["code"], t["kind"], t["docs"], t["percent"], t["readers"], t["spec_v1_rules"],
                           t["spec_v1_1"], t["example_doc_ids"]) for t in flagged if t["domain"] == domain]), ""]
    if not flagged:
        out += ["Nothing yet.", ""]
    for kind, title in (("missed", "Missed: spec v1 has a rule, the cleaned doc still has the disease"),
                        ("harm", "Harm: a spec v1 rule damaged real content")):
        part = [t for t in table if t["kind"] == kind]
        out += [f"## {title}", ""]
        out += [md_table(["domain", "code", "docs", "% of docs read", "readers", "spec v1 rules"],
                         [(t["domain"], t["code"], t["docs"], t["percent"], t["readers"], t["spec_v1_rules"])
                          for t in part]) if part else "None.", ""]
    out += ["## Warnings", ""] + ([f"- {w}" for w in warnings] or ["None."]) + [""]
    path.write_text("\n".join(out), encoding="utf-8")


def run_once(gh, args, overrides, aliases):
    warnings = []
    prs, candidates, readme = fetch(gh, warnings)
    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    rows, used, notes, files, readers_by_domain = [], [], [], [], defaultdict(set)
    for name, (c, others) in sorted(choose(candidates).items()):
        data = download(gh, c, raw_dir)
        if c["ext"] != "csv":
            notes.append((name, c["from"], None, sorted({o["from"] for o in others})))
            continue
        text, problem = decode(data)
        if problem:
            warnings.append(f"{name} ({c['from']}): {problem}")
        file_rows, problems = parse_reading_csv(text, c["pack"], c["reader"])
        warnings += [f"{name} ({c['from']}): {p}" for p in problems]
        for r in file_rows:
            r["code"] = aliases.get(r["code"], r["code"])
            r["file"], r["from"] = name, c["from"]
        rows += file_rows
        used.append((name, c["from"], len(file_rows), sorted({o["from"] for o in others})))
        files.append((name, c["reader"], {r["pack"] for r in file_rows} | {c["pack"]}))
        readers_by_domain[domain_of(c["pack"])].add(c["reader"])
    used += notes

    spec_by_code, spec_by_id = {}, {}
    if args.spec and args.spec.exists():
        spec_by_code, spec_by_id = parse_spec(args.spec.read_text(encoding="utf-8"))
    else:
        warnings.append(f"spec not found ({args.spec}): the spec v1 rules column is empty")

    docs_read = {d: sum(overrides.get(r.lower(), DOCS_PER_READER) for r in rs) for d, rs in readers_by_domain.items()}
    table = summarize(rows, docs_read, args.min_percent, spec_by_code, spec_by_id)
    when = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    write_csv(args.out / "findings_all.csv", rows,
              ["domain", "pack", "reader", "file", "from"] + list(COLUMNS))
    write_csv(args.out / "diseases_by_domain.csv", table,
              ["domain", "kind", "code", "docs", "docs_read", "percent", "readers", "spec_v1_rules", "spec_v1_1",
               "example_doc_ids"])
    readme_matches = match_rows(parse_readme_rows(readme), files) if readme else []
    write_summary(args.out / "summary.md", when, prs, used, readme_matches, docs_read, readers_by_domain,
                  overrides, table, args.min_percent, warnings)

    heads = {p["from"]: p["head_sha"] for p in prs}
    report_changes(args.out / "state.json", heads)
    flagged = sum(1 for t in table if t["spec_v1_1"])
    print(f"{when}: {len(prs)} open reading PRs, {len(used)} files, {len(rows)} rows, "
          f"{flagged} Spec v1.1 candidates -> {args.out / 'summary.md'}")
    for w in warnings:
        print(f"  warning: {w}")
    if gh.remaining is not None and gh.remaining.isdigit() and int(gh.remaining) < 15:
        print(f"  note: {gh.remaining} GitHub API requests left this hour")


def report_changes(state_path: Path, heads: dict):
    old = {}
    if state_path.exists():
        try:
            old = json.loads(state_path.read_text(encoding="utf-8")).get("heads", {})
        except (ValueError, AttributeError):
            old = {}
    for pr in sorted(set(heads) - set(old)):
        print(f"  new PR: {pr}")
    for pr in sorted(p for p in set(heads) & set(old) if heads[p] != old[p]):
        print(f"  new commits: {pr}")
    for pr in sorted(set(old) - set(heads)):
        print(f"  no longer open (merged or closed): {pr}")
    state_path.write_text(json.dumps({"heads": heads}, indent=1), encoding="utf-8")


def parse_pairs(values, what, convert):
    out = {}
    for v in values:
        key, sep, val = v.partition("=")
        if not sep or not key.strip() or not val.strip():
            raise ValueError(f"{what} must look like A=B: {v!r}")
        out[key.strip()] = convert(val.strip())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Collect the Week 2 reading audits and count diseases per domain.")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder (default spec_v1_1/results/reading)")
    ap.add_argument("--spec", type=Path, default=DEFAULT_SPEC, help="specv1.md, for the spec v1 rules column")
    ap.add_argument("--min-percent", type=float, default=1.0,
                    help="share of a domain's docs read for the Spec v1.1 list (default 1, i.e. 1%% or more)")
    ap.add_argument("--docs-read", action="append", default=[], metavar="READER=N",
                    help="docs a reader read, if not 100 (reader as in the file name)")
    ap.add_argument("--alias", action="append", default=[], metavar="CODE=CODE",
                    help="count the first code as the second, e.g. NEW-missing-formula=NEW-missing-formulas")
    ap.add_argument("--watch", type=float, metavar="MINUTES", help="collect again every MINUTES until Ctrl-C")
    args = ap.parse_args(argv)
    try:
        overrides = {k.lower(): v for k, v in parse_pairs(args.docs_read, "--docs-read", int).items()}
        aliases = {normalize_code(k): normalize_code(v) for k, v in parse_pairs(args.alias, "--alias", str).items()}
    except ValueError as e:
        ap.error(str(e))
    if not 0 < args.min_percent <= 100:
        ap.error("--min-percent must be more than 0 and at most 100")
    if any(n < 1 for n in overrides.values()):
        ap.error("--docs-read must be 1 or more")
    if args.watch is not None and args.watch < 5:
        ap.error("--watch must be 5 minutes or more (GitHub allows 60 requests an hour without a token)")
    args.out.mkdir(parents=True, exist_ok=True)

    gh = GitHub(find_token())
    try:
        while True:
            try:
                run_once(gh, args, overrides, aliases)
            except (urllib.error.URLError, OSError, ValueError, KeyError) as e:
                print(f"ERROR: GitHub could not be read: {e}", file=sys.stderr)
                if args.watch is None:
                    return 2
            if args.watch is None:
                return 0
            time.sleep(args.watch * 60)
    except KeyboardInterrupt:
        print("stopped")
        return 0


if __name__ == "__main__":
    sys.exit(main())
