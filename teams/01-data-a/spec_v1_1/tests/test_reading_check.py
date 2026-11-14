"""Tests for scripts/build_reading_check.py. Every row here is made up.

  python3 -m unittest discover -s spec_v1_1/tests      (from teams/01-data-a)
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC / "scripts"))
import build_reading_check as b  # noqa: E402

SUMMARY = """# Week 2 reading audits, collected 2026-11-12 19:09 UTC

## Open reading PRs

| PR | title | author | branch -> base | reading files |
|---|---|---|---|---|
| [someone/era-v5-lab#2](https://github.com/someone/era-v5-lab/pull/2) | reading: math | pyru | a -> b | x.csv |

## README rows

| row | reader | packs | reading files |
|---|---|---|---|
| 7 | Reader E | en-dclm, en-fineweb-edu | **none yet** |
| 9 | Ramesh | math-finemath | reading_math-finemath_pyru.csv |
| 10 | Reader S | sci-finepdfs | **none yet** |

## Docs read per domain

| domain | docs read | readers |
|---|---|---|
| math | 300 | pyru 200, sat 100 |
| science | 100 | adi 100 |

## Warnings

- someone#9: base is main
"""

REGISTER = """# NEW_DISEASES

## Summary: new diseases

| code | disease | reading code | family | packs (docs) | ≥1% of docs read | rule ID | medicine |
|---|---|---|---|---|---|---|---|
| **C7** | Lost formulas | NEW-lost | math | math-finemath (6) | yes | `DA-C7-001` | written |
| **A9** | Split accent | NEW-accent | sci | sci-finepdfs (1) | 1 doc | (`DA-A9-001`) | - |
| **C6** | Punctuation | C6 (labels) | **Indic** | ind-te (161) | yes | `DB-C6-001` | Data B |

## Not new: the disease is already in DISEASES.md

| reading code | family | packs (docs) | DISEASES.md code | spec v1 rule in this family | note |
|---|---|---|---|---|---|
| NEW-ligature | sci | sci-finepdfs (5) | A5 wrong normal form | none | |
"""

FIELDS = ("domain", "kind", "code", "docs", "docs_read", "percent", "readers", "spec_v1_rules", "spec_v1_1",
          "example_doc_ids")
ROWS = [
    {"domain": "math", "kind": "new", "code": "NEW-lost", "docs": "6", "docs_read": "300", "readers": "pyru"},
    {"domain": "math", "kind": "new", "code": "NEW-rare", "docs": "2", "docs_read": "300", "readers": "pyru"},
    {"domain": "math", "kind": "new", "code": "B1", "docs": "3", "docs_read": "300", "readers": "pyru"},
    {"domain": "math", "kind": "missed", "code": "C1", "docs": "30", "docs_read": "300", "readers": "pyru sat",
     "spec_v1_rules": "DA-C1-001 CANDIDATE"},
    {"domain": "math", "kind": "harm", "code": "DA-C2-001", "docs": "6", "docs_read": "300", "readers": "pyru",
     "spec_v1_rules": "DA-C2-001 math CANDIDATE"},
    {"domain": "science", "kind": "new", "code": "NEW-ligature", "docs": "5", "docs_read": "100", "readers": "adi"},
    {"domain": "science", "kind": "new", "code": "NEW-accent", "docs": "1", "docs_read": "100", "readers": "adi"},
    {"domain": "science", "kind": "new", "code": "NEW-unlisted", "docs": "2", "docs_read": "100", "readers": "adi"},
]


class Tables(unittest.TestCase):
    def test_a_table_after_its_heading(self):
        rows = b.md_table(SUMMARY, "## Docs read per domain")
        self.assertEqual([r["domain"] for r in rows], ["math", "science"])
        self.assertEqual(rows[0]["readers"], "pyru 200, sat 100")

    def test_no_table_or_no_heading(self):
        self.assertEqual(b.md_table("## A\n\ntext\n\n## B\n| x |\n|---|\n| 1 |", "## A"), [])
        self.assertEqual(b.md_table(SUMMARY, "## Missing"), [])

    def test_escaped_pipes_stay_in_their_cell(self):
        self.assertEqual(b.cells(r"| a | `\| 1 \|` | c |"), ["a", "`| 1 |`", "c"])


class Register(unittest.TestCase):
    def test_new_and_not_new_rows_and_the_indic_family_left_out(self):
        path = Path(tempfile.mkdtemp()) / "NEW_DISEASES.md"
        path.write_text(REGISTER, encoding="utf-8")
        new, not_new = b.read_register(path)
        self.assertEqual(new, {("math", "NEW-lost"): ("C7", "DA-C7-001"), ("science", "NEW-accent"): ("A9", "(DA-A9-001)")})
        self.assertEqual(not_new, {("science", "NEW-ligature"): "A5 wrong normal form"})

    def test_classify(self):
        new = {("math", "NEW-lost"): ("C7", "DA-C7-001")}
        not_new = {("science", "NEW-ligature"): "A5"}
        self.assertEqual(b.classify("math", "NEW-lost", new, not_new), ("new", "C7, `DA-C7-001`"))
        self.assertEqual(b.classify("math", "B1", new, not_new), ("not new", "in DISEASES.md (B1)"))
        self.assertEqual(b.classify("science", "NEW-ligature", new, not_new), ("not new", "in DISEASES.md as A5"))
        self.assertEqual(b.classify("math", "NEW-x", new, not_new), ("new", "not yet in NEW_DISEASES.md"))


class Build(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        (self.dir / "summary.md").write_text(SUMMARY, encoding="utf-8")
        with open(self.dir / "diseases_by_domain.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            for r in ROWS:
                w.writerow({k: r.get(k, "") for k in FIELDS})
        with open(self.dir / "findings_all.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=("doc_id", "source"))
            w.writeheader()
            w.writerows([{"doc_id": "a", "source": "ai"}, {"doc_id": "b", "source": "ai"}, {"doc_id": "c", "source": "manual"}])
        (self.dir / "NEW_DISEASES.md").write_text(REGISTER, encoding="utf-8")
        self.text = b.build(self.dir, self.dir / "NEW_DISEASES.md")

    def flagged(self):
        return self.text.split("## 4. Flagged")[1].split("## Notes")[0]

    def test_rates_are_docs_over_docs_read(self):
        self.assertIn("| C1 | 30 | 10.0% | pyru sat | DA-C1-001 CANDIDATE |", self.text)
        self.assertIn("| NEW-rare | 2 | 0.7% | no | new | not yet in NEW_DISEASES.md |", self.text)

    def test_only_new_diseases_at_the_bar_are_flagged(self):
        flagged = self.flagged()
        self.assertIn("| math | NEW-lost | 6 | 300 | 2.0% | C7, `DA-C7-001` |", flagged)
        self.assertIn("| science | NEW-accent | 1 | 100 | 1.0% |", flagged)
        self.assertIn("| science | NEW-unlisted | 2 | 100 | 2.0% | not yet in NEW_DISEASES.md |", flagged)
        for code in ("NEW-rare", "| B1 ", "NEW-ligature |"):
            self.assertNotIn(code, flagged.split("**Marked new")[0])
        self.assertIn("math B1 (1.0%)", flagged)
        self.assertIn("science NEW-ligature (5.0%)", flagged)
        self.assertIn("**Flagged per domain:** english 0, code 0, math 1, science 2.", flagged)

    def test_domains_with_no_audit_and_missing_readers(self):
        self.assertIn("| english | 0 | none | Reader E (en-dclm, en-fineweb-edu) |", self.text)
        self.assertIn("| science | 100 | adi 100 | Reader S (sci-finepdfs) |", self.text)
        self.assertIn("### english\n\nNo audit has been submitted for this domain.", self.text)
        self.assertIn("- **Domains with no audit:** english, code.", self.text)

    def test_sections_harm_sources_and_warnings(self):
        self.assertIn("| DA-C2-001 | 6 | 2.0% | pyru | DA-C2-001 math CANDIDATE |", self.text)
        self.assertIn("**Rows by source:** ai 2, manual 1.", self.text)
        self.assertIn("- **Collector warning:** someone#9: base is main", self.text)
        self.assertIn("collector run of 2026-11-12 19:09 UTC", self.text)
        self.assertIn("Reading PRs used: someone/era-v5-lab#2.", self.text)

    def test_main_writes_the_file(self):
        out = self.dir / "reading" / "reading_check_v1.md"
        b.main(["--reading", str(self.dir), "--register", str(self.dir / "NEW_DISEASES.md"), "--out", str(out)])
        self.assertEqual(out.read_text(encoding="utf-8"), self.text)


if __name__ == "__main__":
    unittest.main()
