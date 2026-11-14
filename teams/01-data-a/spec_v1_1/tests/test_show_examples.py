"""Tests for scripts/show_examples.py: the parts that need no pack. Every row here is made up.

  python3 -m unittest discover -s spec_v1_1/tests      (from teams/01-data-a)
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC / "scripts"))
import show_examples as s  # noqa: E402

CANDIDATE_FIELDS = ("domain", "kind", "code", "docs", "docs_read", "percent", "readers", "spec_v1_rules",
                    "spec_v1_1", "example_doc_ids")
FINDING_FIELDS = ("domain", "pack", "reader", "file", "from", "doc_id", "kind", "code", "offsets", "quote", "why",
                  "source")


def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


class Text(unittest.TestCase):
    def test_hidden_characters_are_written_out(self):
        self.assertEqual(s.visible("a​b c\x0cd"), "a<U+200B>b<U+00A0>c<U+000C>d")

    def test_ordinary_text_is_unchanged(self):
        self.assertEqual(s.visible("x² = 1\n\tﬁ — ok"), "x² = 1\n\tﬁ — ok")

    def test_offsets(self):
        self.assertEqual(s.parse_offsets("1204-1388"), (1204, 1388))
        self.assertEqual(s.parse_offsets(" 30 - 10 "), (10, 30))
        self.assertIsNone(s.parse_offsets(""))
        self.assertIsNone(s.parse_offsets("12"))
        self.assertIsNone(s.parse_offsets("a-b"))

    def test_window_marks_the_span_with_context(self):
        self.assertEqual(s.window("0123456789", (4, 6), 2), "...23" + s.OPEN + "45" + s.CLOSE + "67...")
        self.assertEqual(s.window("0123", (0, 4), 5), s.OPEN + "0123" + s.CLOSE)

    def test_window_clips_offsets_past_the_end(self):
        self.assertEqual(s.window("abc", (2, 99), 1), "...b" + s.OPEN + "c" + s.CLOSE)

    def test_fence_is_longer_than_any_backtick_run(self):
        self.assertEqual(s.fence_for("plain"), "```")
        self.assertEqual(s.fence_for("a ```` b"), "`````")


class Safety(unittest.TestCase):
    def test_out_inside_the_repo_is_refused(self):
        self.assertTrue(s.inside_repo(s.REPO / "teams" / "x.md"))
        self.assertTrue(s.inside_repo(s.REPO))
        self.assertFalse(s.inside_repo(s.REPO.parent / "examples" / "x.md"))

    def test_main_refuses_out_inside_the_repo(self):
        with self.assertRaises(SystemExit):
            s.main(["--parts", "p.parquet", "--out", str(s.REPO / "examples.md")])


class Inputs(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        write_csv(self.tmp / "diseases_by_domain.csv", CANDIDATE_FIELDS, [
            {"domain": "math", "kind": "new", "code": "NEW-a", "docs": "3", "spec_v1_1": "yes"},
            {"domain": "math", "kind": "new", "code": "NEW-b", "docs": "9", "spec_v1_1": "yes"},
            {"domain": "math", "kind": "new", "code": "NEW-c", "docs": "1", "spec_v1_1": ""},
            {"domain": "math", "kind": "new", "code": "F1", "docs": "5", "spec_v1_1": "yes"},
            {"domain": "science", "kind": "new", "code": "NEW-d", "docs": "4", "spec_v1_1": "yes"},
            {"domain": "math", "kind": "missed", "code": "C1", "docs": "50", "spec_v1_1": "check"},
        ])
        write_csv(self.tmp / "findings_all.csv", FINDING_FIELDS, [
            {"domain": "math", "pack": "math-finemath", "doc_id": "d1", "kind": "new", "code": "NEW-b", "offsets": "2-4"},
            {"domain": "math", "pack": "math-finemath", "doc_id": "d2", "kind": "new", "code": "NEW-a", "offsets": "x"},
            {"domain": "math", "pack": "math-finemath", "doc_id": "d3", "kind": "new", "code": "NEW-c", "offsets": "1-2"},
            {"domain": "math", "pack": "math-finemath", "doc_id": "d4", "kind": "new", "code": "NEW-b", "offsets": "0-1"},
        ])

    def test_only_flagged_candidates_of_the_domain_most_docs_first(self):
        got = s.load_candidates(self.tmp / "diseases_by_domain.csv", {"math"})
        self.assertEqual([r["code"] for r in got], ["NEW-b", "NEW-a"])

    def test_findings_of_the_candidates_only(self):
        keys = {("math", "new", "NEW-a"), ("math", "new", "NEW-b")}
        got = s.load_findings(self.tmp / "findings_all.csv", keys)
        self.assertEqual([r["doc_id"] for r in got[("math", "new", "NEW-b")]], ["d1", "d4"])
        self.assertNotIn(("math", "new", "NEW-c"), got)

    def test_render_shows_the_span_and_lists_missing_docs(self):
        candidates = s.load_candidates(self.tmp / "diseases_by_domain.csv", {"math"})
        keys = {(c["domain"], c["kind"], c["code"]) for c in candidates}
        findings = s.load_findings(self.tmp / "findings_all.csv", keys)
        text, missing = s.render(candidates, findings, {"d1": "abcdef", "d2": "hello"}, ["p.parquet"], 1)
        self.assertIn("...b" + s.OPEN + "cd" + s.CLOSE + "e...", text)
        self.assertIn("The offsets do not fit", text)  # d2: "x" is not an offset pair
        self.assertEqual(missing, {"d4"})
        self.assertIn("`d4`", text.split("## Docs not found in the parts")[1])


if __name__ == "__main__":
    unittest.main()
