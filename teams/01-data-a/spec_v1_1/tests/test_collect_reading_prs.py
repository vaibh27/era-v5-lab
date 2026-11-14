"""Tests for scripts/collect_reading_prs.py: the parts that need no network. Every row here is made up.

  python3 -m unittest discover -s spec_v1_1/tests      (from teams/01-data-a; python -m pytest spec_v1_1/tests also works)
"""
import sys
import unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC / "scripts"))
import collect_reading_prs as c  # noqa: E402

HEADER = "doc_id,kind,code,offsets,quote,why,source\n"


class DomainAndCodes(unittest.TestCase):
    def test_domain_from_pack_prefix(self):
        self.assertEqual(c.domain_of("en-dclm"), "english")
        self.assertEqual(c.domain_of("code-js-ts"), "code")
        self.assertEqual(c.domain_of("math-megamath"), "math")
        self.assertEqual(c.domain_of("sci-finepdfs"), "science")

    def test_domain_from_a_domain_named_file(self):
        self.assertEqual(c.domain_of("english"), "english")
        self.assertEqual(c.domain_of("Science"), "science")
        self.assertEqual(c.domain_of("ind-bn-sangraha-v"), "unknown")

    def test_new_codes_are_one_spelling(self):
        self.assertEqual(c.normalize_code("NEW-missing-formulas"), "NEW-missing-formulas")
        self.assertEqual(c.normalize_code("new_Missing Formulas"), "NEW-missing-formulas")
        self.assertEqual(c.normalize_code("NEW: SEO filler"), "NEW-seo-filler")
        self.assertEqual(c.normalize_code("NEW-"), "NEW-unnamed")

    def test_existing_codes_and_rule_ids_upper_case(self):
        self.assertEqual(c.normalize_code(" c1 "), "C1")
        self.assertEqual(c.normalize_code("da-c2-001"), "DA-C2-001")

    def test_doc_id_with_a_row_prefix(self):
        self.assertEqual(c.normalize_doc_id("[1/500] doc_id=0ca18aa03db2ef77"), "0ca18aa03db2ef77")
        self.assertEqual(c.normalize_doc_id(" 0ca18aa03db2ef77 "), "0ca18aa03db2ef77")


class ReadingCsv(unittest.TestCase):
    def test_rows_get_domain_pack_and_reader(self):
        rows, warnings = c.parse_reading_csv(HEADER + "d1,new,NEW-seo-filler,1-5,,filler,ai\n", "math-megamath", "satya")
        self.assertEqual(warnings, [])
        self.assertEqual(rows[0]["domain"], "math")
        self.assertEqual(rows[0]["pack"], "math-megamath")
        self.assertEqual(rows[0]["reader"], "satya")
        self.assertEqual(rows[0]["code"], "NEW-seo-filler")

    def test_byte_order_mark_and_header_case(self):
        text, problem = c.decode(("﻿" + HEADER.upper() + "d1,missed,C1,1-5,,footer,manual\n").encode("utf-8"))
        self.assertIsNone(problem)
        rows, warnings = c.parse_reading_csv(text, "en-dclm", "aswani")
        self.assertEqual((len(rows), warnings), (1, []))

    def test_windows_1252_is_read_with_a_warning(self):
        text, problem = c.decode((HEADER + "d1,new,NEW-x,1-5,caf\xe9,why,ai\n").encode("cp1252"))
        self.assertIn("Windows-1252", problem)
        self.assertIn("café", text)

    def test_pack_column_beats_the_file_name(self):
        text = "doc_id,kind,code,offsets,quote,why,source,pack\nd1,new,NEW-x,1-5,,why,ai,en-fineweb-edu\n"
        rows, _ = c.parse_reading_csv(text, "english", "aswani")
        self.assertEqual((rows[0]["pack"], rows[0]["domain"]), ("en-fineweb-edu", "english"))

    def test_no_quote_kept_for_f1_and_e3(self):
        text = HEADER + "d1,missed,F1,1-5,someone@example.com,email,ai\nd2,harm,DA-E3-002,1-5,key,why,ai\n"
        rows, _ = c.parse_reading_csv(text, "code-go", "mahesh")
        self.assertEqual([r["quote"] for r in rows], ["", ""])

    def test_missing_columns(self):
        rows, warnings = c.parse_reading_csv("doc_id,kind\nd1,new\n", "en-dclm", "x")
        self.assertEqual(rows, [])
        self.assertIn("missing columns", warnings[0])

    def test_unknown_kind_is_kept_and_warned(self):
        rows, warnings = c.parse_reading_csv(HEADER + "d1,unsure,C1,1-5,,why,ai\n", "en-dclm", "x")
        self.assertEqual(len(rows), 1)
        self.assertIn("unsure", warnings[0])


def candidate(name, reader, pr=None, head_ref="", updated=0.0, src="x"):
    return {"name": name, "reader": reader, "pr": pr, "head_ref": head_ref, "updated_ts": updated, "from": src}


class ChooseCopy(unittest.TestCase):
    def test_readers_own_pr_beats_g1_reading_and_carried_copies(self):
        name = "reading_math-finemath_pyru.csv"
        merged = candidate(name, "pyru", src="g1-reading")
        carried = candidate(name, "pyru", pr=350, head_ref="g1-reading-math-megamath-Satyanarayana", updated=9.0)
        own = candidate(name, "pyru", pr=2, head_ref="g1-reading-math-finemath-pyru", updated=1.0)
        chosen, others = c.choose([carried, merged, own])[name]
        self.assertIs(chosen, own)
        self.assertEqual([o["from"] for o in others], ["g1-reading", "x"])

    def test_g1_reading_beats_a_carried_copy(self):
        name = "reading_math-finemath_pyru.csv"
        merged = candidate(name, "pyru", src="g1-reading")
        carried = candidate(name, "pyru", pr=350, head_ref="g1-reading-math-megamath-Satyanarayana", updated=9.0)
        self.assertIs(c.choose([carried, merged])[name][0], merged)


class Summary(unittest.TestCase):
    def rows(self, *items):
        return [{"domain": d, "pack": p, "reader": r, "doc_id": i, "kind": k, "code": code}
                for d, p, r, i, k, code in items]

    def test_distinct_docs_and_the_spec_v1_1_flag(self):
        rows = self.rows(("math", "math-finemath", "pyru", "a", "new", "NEW-seo-filler"),
                         ("math", "math-finemath", "pyru", "a", "new", "NEW-seo-filler"),
                         ("math", "math-megamath", "satya", "b", "new", "NEW-seo-filler"),
                         ("math", "math-megamath", "satya", "c", "new", "NEW-template-pages"))
        table = {t["code"]: t for t in c.summarize(rows, {"math": 200}, 1.0, {}, {})}
        self.assertEqual(table["NEW-seo-filler"]["docs"], 2)
        self.assertEqual(table["NEW-seo-filler"]["percent"], "1.0")
        self.assertEqual(table["NEW-seo-filler"]["readers"], "pyru satya")
        self.assertEqual(table["NEW-seo-filler"]["spec_v1_1"], "yes")
        self.assertEqual(table["NEW-template-pages"]["spec_v1_1"], "")

    def test_spec_rules_for_the_family_and_a_missed_code_with_none(self):
        rows = self.rows(("math", "math-finemath", "pyru", "a", "missed", "C1"),
                         ("math", "math-finemath", "pyru", "b", "missed", "C1"),
                         ("math", "math-finemath", "pyru", "a", "missed", "B5"),
                         ("math", "math-finemath", "pyru", "b", "missed", "B5"),
                         ("math", "math-finemath", "pyru", "a", "harm", "DA-C2-001"))
        by_code = {("math", "C1"): ["DA-C1-010 CANDIDATE"], ("code", "B5"): ["DA-B5-001 CANDIDATE"]}
        table = {t["code"]: t for t in c.summarize(rows, {"math": 100}, 1.0, by_code, {"DA-C2-001": "math CANDIDATE"})}
        self.assertEqual((table["C1"]["spec_v1_rules"], table["C1"]["spec_v1_1"]), ("DA-C1-010 CANDIDATE", ""))
        self.assertEqual((table["B5"]["spec_v1_rules"], table["B5"]["spec_v1_1"]), ("none", "check: no math rule"))
        self.assertEqual(table["DA-C2-001"]["spec_v1_rules"], "DA-C2-001 math CANDIDATE")

    def test_same_doc_id_in_two_packs_counts_twice(self):
        rows = self.rows(("code", "code-go", "m", "a", "new", "NEW-x"), ("code", "code-cpp", "m", "a", "new", "NEW-x"))
        self.assertEqual(c.summarize(rows, {}, 1.0, {}, {})[0]["docs"], 2)

    def test_flag_at_one_percent_of_docs_read(self):
        two = self.rows(*[("math", "math-finemath", "pyru", i, "new", "NEW-x") for i in "ab"])
        three = self.rows(*[("math", "math-finemath", "pyru", i, "new", "NEW-x") for i in "abc"])
        flag = lambda rows, n: c.summarize(rows, {"math": n}, 1.0, {}, {})[0]["spec_v1_1"]
        self.assertEqual(flag(two, 200), "yes")   # README example: 2 of 200 is 1%
        self.assertEqual(flag(two, 299), "")      # 0.7%
        self.assertEqual(flag(three, 299), "yes")  # 1.003%
        self.assertEqual(flag(three, 200), "yes")

    def test_no_flag_without_a_docs_read_count(self):
        rows = self.rows(*[("unknown", "x", "m", i, "new", "NEW-x") for i in "abcde"])
        self.assertEqual(c.summarize(rows, {}, 1.0, {}, {})[0]["spec_v1_1"], "")


class SpecAndReadme(unittest.TestCase):
    def test_parse_spec_headings_and_status(self):
        text = ('## Medicine 1 — math / C1 / `DA-C1-010`\n```python\nSTATUS = "CANDIDATE"\n```\n'
                '## Medicine 2 — indic / F1 / `DB-F1-009`\n```python\nSTATUS = "DEFERRED"\n```\n')
        by_code, by_id = c.parse_spec(text)
        self.assertEqual(by_code[("math", "C1")], ["DA-C1-010 CANDIDATE"])
        self.assertEqual(by_id["DB-F1-009"], "indic DEFERRED")

    def test_readme_rows_and_matching(self):
        readme = ("| Row | Reader | Pack(s) |\n|---|---|---|\n"
                  "| 9 | Ramesh | math-finemath (100) |\n"
                  "| 9 | Satyanarayana Murthy Toleti | math-megamath (100) |\n"
                  "| 10 | Utkarsh Mittal | sci-finepdfs, docs 1-100 of the shuffled list |\n"
                  "| 10 | Aditeya | sci-finepdfs, docs 101-200 of the shuffled list |\n"
                  "| Thu 12 Nov | PR raised |\n")
        rows = c.parse_readme_rows(readme)
        self.assertEqual(rows[0], (9, "Ramesh", ["math-finemath"]))
        self.assertEqual(len(rows), 4)
        files = [("reading_math-finemath_pyru.csv", "pyru", {"math-finemath"}),
                 ("reading_math-megamath_satyanarayana.csv", "satyanarayana", {"math-megamath"}),
                 ("reading_sci-finepdfs_utkarsh.csv", "utkarsh", {"sci-finepdfs"})]
        matched = {reader: names for _, reader, _, names in c.match_rows(rows, files)}
        self.assertEqual(matched["Ramesh"], ["reading_math-finemath_pyru.csv"])  # only row with that pack
        self.assertEqual(matched["Satyanarayana Murthy Toleti"], ["reading_math-megamath_satyanarayana.csv"])
        self.assertEqual(matched["Utkarsh Mittal"], ["reading_sci-finepdfs_utkarsh.csv"])
        self.assertEqual(matched["Aditeya"], [])  # shares the pack, so only a name match counts


class Misc(unittest.TestCase):
    def test_reading_pr_by_branch_or_title(self):
        self.assertTrue(c.is_reading_pr({"head": {"ref": "g1-reading-en-dclm-aswani"}, "title": "x"}))
        self.assertTrue(c.is_reading_pr({"head": {"ref": "x"}, "title": "G1 reading math megamath satyanarayana"}))
        self.assertTrue(c.is_reading_pr({"head": {"ref": "x"}, "title": "reading: en-dclm audit"}))
        self.assertFalse(c.is_reading_pr({"head": {"ref": "g1-reading"}, "title": "Speed up DA-F1-005"}))

    def test_next_link(self):
        link = '<https://api.github.com/x?page=2>; rel="next", <https://api.github.com/x?page=5>; rel="last"'
        self.assertEqual(c.next_link(link), "https://api.github.com/x?page=2")
        self.assertIsNone(c.next_link(""))

    def test_parse_pairs(self):
        self.assertEqual(c.parse_pairs(["pyru=199"], "--docs-read", int), {"pyru": 199})
        with self.assertRaises(ValueError):
            c.parse_pairs(["pyru"], "--docs-read", int)


if __name__ == "__main__":
    unittest.main()
