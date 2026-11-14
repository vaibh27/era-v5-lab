"""Tests for medicine_conflicts.py. Run: python -I teams/01-data-a/tests/harmtest/test_medicine_conflicts.py"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harmtest"))  # python -I: not on the path by itself
import medicine_conflicts as mc  # noqa: E402
import medicine_extract  # noqa: E402

FENCE = "```"
TEMPLATE = ("# Medicine sheet: <pack_id>\n\n## Medicine 1\n\n1. **Disease:** code, count in your pack, three examples "
            "(`doc_id` and the exact characters).\n3. **Action:** fix the text, or drop the document.\n")


def sheet(*sections):
    """sections: (code, count, action, medicine words, code or None)."""
    out = []
    for code, count, action, words, body in sections:
        out += [f"1. **Disease:** {code}, {count} documents.", f"2. **Medicine:** {words}"]
        if body:
            out.append(f"{FENCE}python\n{body}\n{FENCE}")
        out += [f"3. **Action:** {action}.", ""]
    return "\n".join(out)


class ConflictCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.subs, self.gold, self.registry = self.tmp / "subs", self.tmp / "gold", self.tmp / "rules" / "registry.json"
        self.gold.mkdir()

    def medicine(self, pack, handle, text):
        d = self.subs / pack / handle
        d.mkdir(parents=True, exist_ok=True)
        (d / "medicine.md").write_text(text)

    def write_gold(self, pack, n):
        (self.gold / f"{pack}_gold.csv").write_text("doc_id,reader_1,reader_2\n" + "".join(f"d{i},a,b\n" for i in range(n)))

    def run_check(self, *extra):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            if self.subs.exists():
                medicine_extract.main(["--dir", str(self.subs), "--registry", str(self.registry)])
            if not self.registry.exists():
                self.registry.parent.mkdir(parents=True, exist_ok=True)
                self.registry.write_text(json.dumps({"rules": []}))
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                rc = mc.main(["--medicines", str(self.subs), "--gold-dir", str(self.gold), "--registry", str(self.registry),
                              "--out", str(self.tmp / "CHECK.md"), *extra])
        return rc, out.getvalue(), (self.tmp / "CHECK.md").read_text()

    def test_a_gold_file_with_only_a_blank_template(self):
        self.write_gold("en-a", 34)
        self.medicine("en-a", "harini", TEMPLATE)
        rc, out, report = self.run_check()
        self.assertEqual(rc, 1)
        self.assertIn("NO MEDICINE: no sheet has a medicine: harini (blank template)", out)
        self.assertIn("SMALL GOLD SET: 34 documents", out)

    def test_a_gold_file_without_any_sheet(self):
        self.write_gold("en-a", 200)
        self.subs.mkdir()
        rc, out, _ = self.run_check()
        self.assertIn("NO MEDICINE: no medicine sheet", out)

    def test_a_rule_whose_code_lacks_a_name_cannot_run(self):
        self.write_gold("en-a", 200)
        self.medicine("en-a", "mahesh", sheet(("C1", 76, "fix", "delete boilerplate lines",
                                                "def strip(t):\n    return BOILERPLATE.sub('', t), False")))
        rc, out, _ = self.run_check()
        self.assertEqual(rc, 1)
        self.assertIn("NOT ENOUGH", out)
        self.assertIn("CANNOT RUN: DB-C1-001 (C1, mahesh) needs BOILERPLATE", out)  # en-a is not a Data A pack: DB
        self.assertIn("NO RULE: C1: found more than 5 times (76)", out)

    def test_names_bound_together_and_imports_count_as_defined(self):
        self.write_gold("en-a", 200)
        self.medicine("en-a", "deep", sheet(("B4", 8, "fix", "remove template junk",
                                              "A, B = 2, 6\ndef cure_b4(t):\n    return re.sub('x' * A * B, '', t)")))
        _, out, _ = self.run_check()
        self.assertIn("en-a: 1 rules can run", out)
        self.assertNotIn("CANNOT RUN", out)

    def test_two_annotators_clash(self):
        self.write_gold("en-a", 200)
        self.medicine("en-a", "shekar", sheet(("C3", 98, "drop", "drop short stubs", "def m_c3(t):\n    return t, False")))
        self.medicine("en-a", "deep", sheet(("C3", 13, "fix", "cut the last line", "def cut(t):\n    return t")))
        self.medicine("en-a", "siva", "# Draft\n\n## Row 16: manual C4 medicine\n\n- Rows 75, 76: joined; C3 provisional\n")
        _, out, report = self.run_check()
        self.assertIn("#### C3: ACTIONS DIFFER, RULE VS HAND, CHOOSE", report)
        self.assertIn("| siva | hand edit (line 5) | fix | - |", report)
        self.assertNotIn("ONE ANNOTATOR", out)

    def test_two_rules_of_one_annotator_are_to_choose_from(self):
        self.write_gold("en-a", 200)
        self.medicine("en-a", "mahesh", sheet(("B4", 8, "fix", "meta comments", "def meta(t):\n    return t"),
                                              ("B4", 18, "fix", "quote spacing", "def quotes(t):\n    return t")))
        _, out, report = self.run_check()
        self.assertIn("#### B4: CHOOSE", report)
        self.assertIn("ONE ANNOTATOR: only mahesh", out)

    def test_across_the_packs_of_a_language(self):
        self.write_gold("en-a", 200)
        self.write_gold("en-b", 200)
        self.medicine("en-a", "shekar", sheet(("F1", 8, "fix", "replace with a placeholder", "def m_f1(t):\n    return t")))
        self.medicine("en-b", "deep", sheet(("F1", 8, "drop", "remove them", "def f1(t):\n    return t")))
        _, _, report = self.run_check()
        self.assertIn("### en-* F1: ACTIONS DIFFER", report)

    def test_document_quotes_are_left_out_of_the_report(self):
        self.write_gold("en-a", 200)
        self.medicine("en-a", "x", sheet(("C1", 9, "fix", "remove lines like `call 555 0100 now`", "def c(t):\n    return t")))
        self.medicine("en-a", "y", sheet(("C1", 9, "fix", "other", "def d(t):\n    return t")))
        _, _, report = self.run_check()
        self.assertNotIn("555", report)

    def test_rules_from_a_standard_sheet(self):
        # en-medicines/medicine_en.md: one block per rule, each run on its own by harm_test.py
        self.write_gold("en-a", 200)
        self.medicine("en-a", "deep", sheet(("C1", 9, "fix", "remove lines", None), ("F1", 8, "fix", "remove emails", None)))
        block = ('RULE_ID = "{id}"\nCODE = "{code}"\nACTION = "fix"\nORIGIN = "en-a"\nAUTHOR = "deep"\nimport re\n'
                 "{body}\ndef medicine(doc):\n    return ('fix', run(doc['text']))\n")
        rules = self.tmp / "medicine_en.md"
        rules.write_text("# Sheet\n\n" + "\n".join(f"{FENCE}python\n{block.format(**b)}{FENCE}\n" for b in [
            {"id": "DA-C1-009", "code": "C1", "body": "def run(t):\n    return re.sub('x', '', t)\n"},
            {"id": "DA-F1-009", "code": "F1", "body": "def run(t):\n    return EMAIL.sub('', t)\n"}]))
        _, out, report = self.run_check("--sheet", str(rules))
        self.assertIn("en-a: 1 rules can run", out)
        self.assertIn("CANNOT RUN: DA-F1-009 (F1, deep) needs EMAIL", out)
        self.assertIn("Rules: `medicine_en.md`", report)

    def test_a_rule_for_several_packs_counts_for_each(self):
        self.write_gold("en-a", 200)
        self.write_gold("en-b", 200)
        self.medicine("en-b", "harini", TEMPLATE)
        rules = self.tmp / "medicine_en.md"
        rules.write_text(f'{FENCE}python\nRULE_ID = "DA-C1-009"\nCODE = "C1"\nACTION = "fix"\nORIGIN = "en-a, en-b"\n'
                         f'AUTHOR = "aneeshads"\ndef medicine(doc):\n    return ("keep", doc["text"])\n{FENCE}\n')
        rc, out, report = self.run_check("--sheet", str(rules))
        self.assertEqual(rc, 0)
        self.assertNotIn("NO MEDICINE", out)
        self.assertIn("BLANK SHEET: harini", out)
        self.assertIn("en-b: 1 rules can run", out)
        self.assertIn("also run here (1): `DA-C1-009` (C1)", report)

    def test_the_local_layout_and_names(self):
        # <pack>-medicine/<handle>-medicine.md, saved under another name than the team's copy.
        team = self.tmp / "team"
        (team / "en-a" / "deepjyotisaha").mkdir(parents=True)
        text = sheet(("C1", 9, "fix", "remove lines", "def c(t):\n    return t"))
        (team / "en-a" / "deepjyotisaha" / "medicine.md").write_text(text)
        local = self.tmp / "local" / "en-a-medicine"
        local.mkdir(parents=True)
        (local / "dsaha-medicine.md").write_text(text)
        found = mc.find_sheets(self.tmp / "local")
        self.assertEqual(list(found), ["en-a"])
        self.assertEqual(mc.registry_handles(found, team), {("en-a", "dsaha"): "deepjyotisaha"})


if __name__ == "__main__":
    unittest.main()
