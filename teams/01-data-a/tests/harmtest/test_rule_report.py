"""Tests for rule_report.py. Run: python -I teams/01-data-a/tests/harmtest/test_rule_report.py"""
import contextlib
import io
import sys
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harmtest"))  # python -I: not on the path by itself
import medicine_extract  # noqa: E402
import rule_report  # noqa: E402


def rule(rid, hits=(), harm="PASS", errors=0, pct=None):
    return {"id": rid, "hits": set(hits), "harm": harm, "errors": errors, "pct": pct}


class RecommendCase(unittest.TestCase):
    def test_the_rule_that_does_nothing_is_not_rewarded(self):
        a = rule("DA-B3-001", hits=[])                   # passes the harm test because it changes nothing
        b = rule("DA-B3-002", hits=["d1", "d2", "d3"])
        adv = rule_report.recommend([a, b], n_sick=3)
        self.assertEqual(adv["DA-B3-002"][0], "KEEP")
        self.assertEqual(adv["DA-B3-001"][0], "DROP")
        self.assertIn("does nothing", adv["DA-B3-001"][1])

    def test_rules_that_fix_different_cases_are_both_kept(self):
        adv = rule_report.recommend([rule("DA-B3-001", ["d1", "d2"]), rule("DA-B3-002", ["d3"])], n_sick=3)
        self.assertEqual({k: v[0] for k, v in adv.items()}, {"DA-B3-001": "KEEP", "DA-B3-002": "KEEP"})
        self.assertIn("fixes 1 sick rows that DA-B3-001 does not", adv["DA-B3-002"][1])

    def test_a_rule_whose_rows_are_all_covered_is_dropped(self):
        adv = rule_report.recommend([rule("DA-B3-001", ["d1", "d2", "d3"]), rule("DA-B3-002", ["d2"])], n_sick=3)
        self.assertEqual(adv["DA-B3-001"][0], "KEEP")
        self.assertEqual(adv["DA-B3-002"][0], "DROP")
        self.assertIn("all fixed by DA-B3-001", adv["DA-B3-002"][1])

    def test_a_rule_that_fails_the_harm_test_never_wins(self):
        failing = rule("DA-B3-001", ["d1", "d2", "d3"], harm="FAIL", pct=Fraction(5))
        adv = rule_report.recommend([failing, rule("DA-B3-002", ["d1"])], n_sick=3)
        self.assertEqual(adv["DA-B3-001"][0], "DROP")
        self.assertIn("fails the harm test (5.00%)", adv["DA-B3-001"][1])
        self.assertEqual(adv["DA-B3-002"][0], "KEEP")

    def test_a_rule_with_errors_is_dropped(self):
        adv = rule_report.recommend([rule("DA-B3-001", ["d1"], errors=2), rule("DA-B3-002", ["d1"])], n_sick=1)
        self.assertEqual(adv["DA-B3-001"][0], "DROP")

    def test_nothing_to_compare_without_labelled_rows(self):
        self.assertEqual(rule_report.recommend([rule("a"), rule("b")], n_sick=0), {})

    def test_without_a_gold_set_the_advice_says_so(self):
        adv = rule_report.recommend([rule("DA-B3-001", ["d1"], harm="n/a")], n_sick=1)
        self.assertIn("harm test not run", adv["DA-B3-001"][1])


class EndToEndCase(unittest.TestCase):
    """Two annotators, one pack of 20 documents. d00-d03 have <m>, d04-d05 have <n>, d10-d19 are the gold set."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.texts = ["has <m> marker"] * 4 + ["has <n> marker"] * 2 + ["plain text"] * 14
        self.ids = [f"d{i:02d}" for i in range(20)]
        pq.write_table(pa.table({"doc_id": self.ids, "text": self.texts}), self.tmp / "pack.parquet")
        self.gold = self.tmp / "gold.csv"
        self.gold.write_text("doc_id,reader_1,reader_2\n" + "".join(f"{i},a,b\n" for i in self.ids[10:]))
        self.registry = self.tmp / "rules" / "registry.json"

    def submit(self, handle, body, labelled_b3=range(6)):
        d = self.tmp / "submissions" / "code-rust" / handle
        d.mkdir(parents=True)
        (d / "medicine.md").write_text(
            f"**Disease:** B3, markers\n3. **Action:** fix the text.\n```python\ndef cure_b3(t):\n{body}\n```\n")
        rows = "".join(f"{self.ids[i]},full,fix,B3,x,,\n" for i in labelled_b3)
        rows += "".join(f"{self.ids[i]},full,clean,,,,\n" for i in range(10, 14))
        (d / "labels.csv").write_text("doc_id,read,verdict,diseases,evidence,hex,note\n" + rows)
        out = self.tmp / "rules" / "code-rust" / f"{handle}.py"
        with contextlib.redirect_stdout(io.StringIO()):
            medicine_extract.main([str(d / "medicine.md"), "--out", str(out), "--registry", str(self.registry)])

    def report(self, gold=True):
        argv = ["--pack-id", "code-rust", "--pack-file", str(self.tmp / "pack.parquet"),
                "--registry", str(self.registry), "--submissions", str(self.tmp / "submissions")]
        if gold:
            argv += ["--gold", str(self.gold)]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = rule_report.main(argv)
        text = (self.tmp / "rules" / "code-rust" / "REPORT.md").read_text()
        return rc, out.getvalue(), text

    def test_the_wider_rule_wins_and_the_subset_rule_is_dropped(self):
        self.submit("alice", "    return t.replace('<m>', '')")
        self.submit("bob", "    return t.replace('<m>', '').replace('<n>', '')")
        rc, out, text = self.report()
        self.assertEqual(rc, 0)
        self.assertIn("| DA-B3-001 | B3 | alice | fix / fix | 0 | 0 | 0.00% | PASS | 6 | 4 | 0 | 0 | 2 | 0 |", text)
        self.assertIn("| DA-B3-002 | B3 | bob | fix / fix | 0 | 0 | 0.00% | PASS | 6 | 6 | 0 | 0 | 0 | 0 |", text)
        self.assertIn("| DA-B3-002 | KEEP | passes and fixes the most sick rows (6 of 6) |", text)
        self.assertIn("| DA-B3-001 | DROP | its 4 sick rows are all fixed by DA-B3-002 |", text)

    def test_different_cases_keeps_both(self):
        self.submit("alice", "    return t.replace('<m>', '')")
        self.submit("bob", "    return t.replace('<n>', '')")
        _, _, text = self.report()
        self.assertIn("| DA-B3-001 | KEEP |", text)
        self.assertIn("| DA-B3-002 | KEEP | fixes 2 sick rows that DA-B3-001 does not |", text)

    def test_a_rule_that_touches_the_gold_set_fails_and_is_not_kept(self):
        self.submit("alice", "    return t.replace('<m>', '')")
        self.submit("bob", "    return t.upper()")            # changes every row, gold rows included
        _, _, text = self.report()
        self.assertIn("| DA-B3-002 | B3 | bob | fix / fix | 10 | 0 | 100.00% | FAIL |", text)
        self.assertIn("| DA-B3-002 | DROP | fails the harm test (100.00%) |", text)
        self.assertIn("| DA-B3-001 | KEEP |", text)

    def test_harm_numbers_match_harm_test(self):
        import harm_test
        self.submit("alice", "    return t.upper()")
        rc, _, text = self.report()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            harm_rc = harm_test.main(["--pack", str(self.tmp / "pack.parquet"), "--gold", str(self.gold),
                                      "--registry", str(self.registry), "--rule", "DA-B3-001"])
        self.assertEqual(harm_rc, 1)
        self.assertIn("documents changed: 10", out.getvalue())
        self.assertIn("| 10 | 0 | 100.00% | FAIL |", text)

    def test_without_a_gold_set(self):
        self.submit("alice", "    return t.replace('<m>', '')")
        rc, _, text = self.report(gold=False)
        self.assertEqual(rc, 0)
        self.assertIn("none: the harm test was not run", text)
        self.assertIn("| fix / fix | - | - | - | n/a |", text)

    def test_flags(self):
        self.submit("alice", "    return t.replace('<m>', '')")
        self.submit("bob", "    return ('drop', '') if '<n>' in t else t")   # the medicine.md says fix
        _, _, text = self.report()
        self.assertIn("DA-B3-002: the medicine.md says fix, but the rule does drop", text)

    def test_unlabelled_and_other_rows_are_counted_apart(self):
        self.submit("alice", "    return t.replace('<m>', '')", labelled_b3=[0, 1])   # d02, d03 are not labelled
        _, _, text = self.report()
        self.assertIn("| PASS | 2 | 2 | 0 | 2 | 0 | 0 |", text)   # sick 2, hit 2, extra 0, unlabelled 2, missed 0

    def test_a_code_with_many_labels_and_no_rule_is_flagged(self):
        self.submit("alice", "    return t")
        d = self.tmp / "submissions" / "code-rust" / "alice"
        (d / "labels.csv").write_text("doc_id,read,verdict,diseases,evidence,hex,note\n"
                                      + "".join(f"d{i:02d},full,fix,C2,x,,\n" for i in range(6)))
        _, _, text = self.report()
        self.assertIn("C2: 6 sick rows in the labels and no rule for it", text)

    def test_a_rules_file_that_cannot_load_is_an_error_not_a_crash(self):
        self.submit("alice", "    return t.replace('<m>', '')")
        self.submit("bob", "    return t")
        (self.tmp / "rules" / "code-rust" / "bob.py").write_text("raise ValueError('broken')\n")
        rc, _, text = self.report()
        self.assertEqual(rc, 0)
        self.assertIn("| DA-B3-002 | B3 | bob |", text)
        self.assertIn("ERROR", text)
        self.assertIn("DA-B3-002: bob.py: ValueError: broken", text)
        self.assertIn("| DA-B3-002 | DROP | the rule raised errors on some documents |", text)

    def test_no_rules_for_the_pack(self):
        self.submit("alice", "    return t")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rc = rule_report.main(["--pack-id", "code-go", "--pack-file", str(self.tmp / "pack.parquet"),
                                   "--registry", str(self.registry)])
        self.assertEqual(rc, 2)
        self.assertIn("no active rules for code-go", err.getvalue())


if __name__ == "__main__":
    unittest.main()
