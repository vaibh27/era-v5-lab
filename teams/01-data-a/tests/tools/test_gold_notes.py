"""Tests for gold_notes.py. Run: python -I teams/01-data-a/tests/tools/test_gold_notes.py"""
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))  # python -I: not on the path by itself
import gold_notes  # noqa: E402

HEADER = "date,pack_id,doc_id,decision,code,raised_by,second_reader,rule_id,reason\n"


class GoldNotesCase(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.gold = self.dir / "en-x_gold.csv"
        self.write_gold(["a", "b", "c"])

    def write_gold(self, ids):
        self.gold.write_text("doc_id,reader_1,reader_2\n" + "".join(f"{i},r1,r2\n" for i in ids))

    def write_notes(self, *rows):
        (self.dir / "gold_notes.csv").write_text(HEADER + "".join(r + "\n" for r in rows))

    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = gold_notes.main(["--gold-dir", str(self.dir), *argv])
        return rc, out.getvalue()

    def test_record_then_check(self):
        rc, out = self.run_main("--record")
        self.assertEqual(rc, 0)
        self.assertIn("3 documents added", out)
        self.assertEqual(gold_notes.read_manifest(self.dir), {"en-x": ["a", "b", "c"]})
        self.assertEqual(self.run_main("--record")[1].splitlines()[0], "gold_manifest.csv: 0 documents added")

    def test_without_a_manifest_only_a_warning(self):
        problems, warnings = gold_notes.check(self.gold, self.dir)
        self.assertEqual(problems, [])
        self.assertIn("cannot be checked", warnings[0])

    def test_a_removal_without_a_note_is_a_problem(self):
        self.run_main("--record")
        self.write_gold(["a", "b"])  # someone took c out, for example to make a rule pass
        rc, out = self.run_main()
        self.assertEqual(rc, 1)
        self.assertIn("c left the gold set without a note", out)

    def test_a_removal_with_a_second_readers_note_is_fine(self):
        self.run_main("--record")
        self.write_gold(["a", "b"])
        self.write_notes("2026-10-28,en-x,c,removed,C1,ann,bob,DA-C1-002,a share line at the end")
        self.assertEqual(self.run_main()[0], 0)

    def test_the_note_must_be_complete(self):
        self.run_main("--record")
        self.write_gold(["a"])
        self.write_notes("2026-10-28,en-x,b,removed,C1,ann,,,no second reader",
                         "2026-10-28,en-x,c,removed,C1,ann,bob,,")
        problems, _ = gold_notes.check(self.gold, self.dir)
        self.assertEqual(problems, ["b: the note has no second_reader", "c: the note has no reason"])

    def test_kept_does_not_explain_a_removal(self):
        self.run_main("--record")
        self.write_gold(["a", "b"])
        self.write_notes("2026-10-28,en-x,c,kept,C1,ann,bob,,the second reader found it clean")
        problems, _ = gold_notes.check(self.gold, self.dir)
        self.assertIn("c left the gold set without a note in gold_notes.csv", problems)

    def test_new_documents_are_fine_and_reported(self):
        self.run_main("--record")
        self.write_gold(["a", "b", "c", "d"])
        problems, warnings = gold_notes.check(self.gold, self.dir)
        self.assertEqual(problems, [])
        self.assertIn("1 documents of en-x_gold.csv are not in gold_manifest.csv", warnings[0])

    def test_a_bad_decision_is_a_problem(self):
        self.run_main("--record")
        self.write_notes("2026-10-28,en-x,a,deleted,C1,ann,bob,,x")
        self.assertIn("decision must be removed or kept", gold_notes.check(self.gold, self.dir)[0][0])

    def test_a_copy_elsewhere_is_checked_against_the_team_manifest(self):
        self.assertEqual(gold_notes.notes_dir(self.dir / "nowhere" / "en-x_gold.csv"), gold_notes.DEFAULT_GOLD_DIR)
        self.run_main("--record")
        self.assertEqual(gold_notes.notes_dir(self.gold), self.dir)


if __name__ == "__main__":
    unittest.main()
