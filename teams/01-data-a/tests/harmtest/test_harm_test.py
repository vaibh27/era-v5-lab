"""Tests for harm_test.py. Run: python -I teams/01-data-a/tests/harmtest/test_harm_test.py."""
import contextlib
import io
import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harmtest"))  # python -I: not on the path by itself
import harm_test  # noqa: E402

MARKER_RULE = "def cure_b3(t):\n    return t.replace('<issue_start>', '[issue_start]')\n"  # changes d000 and d001


class HarmTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.texts = {f"d{i:03d}": f"clean document {i}" for i in range(200)}
        self.texts["d000"] = "<issue_start> marked document"
        self.texts["d001"] = "<issue_start> another one"
        self.texts["d002"] = "DROPME"
        pq.write_table(pa.table({"doc_id": list(self.texts), "text": list(self.texts.values())}), self.tmp / "pack.parquet")
        self.write_gold(list(self.texts))

    def write_gold(self, ids):
        (self.tmp / "gold.csv").write_text("doc_id,reader_1,reader_2\n" + "".join(f"{i},a,b\n" for i in ids))

    def run_ht(self, rules_src, code, name="rules.py", *extra):
        (self.tmp / name).write_text(rules_src)
        argv = ["--pack", str(self.tmp / "pack.parquet"), "--gold", str(self.tmp / "gold.csv"),
                "--rules", str(self.tmp / name), "--rule", code, *extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = harm_test.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def test_pass_at_exactly_one_percent(self):
        # 2 of 200 = 1.00%: PASS. The rule changes d000 and d001.
        rc, out, _ = self.run_ht("def cure_b3(t):\n    return t.replace('<issue_start>', '[issue_start]')\n", "B3")
        self.assertEqual(rc, 0)
        self.assertIn("documents changed: 2", out)
        self.assertIn("percent: 1.00%", out)
        self.assertIn("result: PASS", out)

    def test_fail_above_one_percent(self):
        # 3 of 200 = 1.50%: FAIL. Changed and dropped add up.
        src = ("RULES = {'X': lambda t: ('drop', '') if t == 'DROPME' else "
               "('fix', t.replace('<issue_start>', ''))}\n")
        rc, out, _ = self.run_ht(src, "x")
        self.assertEqual(rc, 1)
        self.assertIn("documents changed: 2", out)
        self.assertIn("documents dropped: 1", out)
        self.assertIn("result: FAIL", out)

    def test_small_gold_set_allows_one_document(self):
        # TSAI decision: the limit is max(1, floor(1% of n)) documents, so 1 of 34 (2.94%) passes
        self.write_gold(list(self.texts)[1:35])               # d001..d034: the rule changes d001 only
        rc, out, _ = self.run_ht(MARKER_RULE, "B3")
        self.assertEqual(rc, 0)
        self.assertIn("documents changed: 1", out)
        self.assertIn("limit: 1 document changed or dropped (max(1, 1% of 34))", out)
        self.assertIn("result: PASS", out)

    def test_small_gold_set_fails_at_two_documents(self):
        self.write_gold(list(self.texts)[:34])                # d000..d033: the rule changes d000 and d001
        rc, out, _ = self.run_ht(MARKER_RULE, "B3")
        self.assertEqual(rc, 1)
        self.assertIn("documents changed: 2", out)
        self.assertIn("result: FAIL", out)

    def test_the_limit_is_one_document_up_to_199_rows(self):
        self.write_gold(list(self.texts)[1:200])              # 199 rows, d001 changed: floor(1.99) = 1 allowed
        rc, out, _ = self.run_ht(MARKER_RULE, "B3")
        self.assertEqual(rc, 0)
        self.assertIn("limit: 1 document changed or dropped (max(1, 1% of 199))", out)
        self.write_gold(list(self.texts)[:199])               # 199 rows, d000 and d001 changed
        rc, out, _ = self.run_ht(MARKER_RULE, "B3")
        self.assertEqual(rc, 1)
        self.assertIn("result: FAIL", out)

    def test_an_empty_gold_set_is_not_testable(self):
        self.write_gold([])
        for src, code in ((MARKER_RULE, "B3"), ("RULES = {'B3': lambda t: t}\n", "all")):
            rc, out, _ = self.run_ht(src, code)
            self.assertEqual(rc, 2)
            self.assertIn("result: NOT TESTABLE (gold.csv: the gold set is empty)", out)
            self.assertNotIn("documents changed", out)

    def test_the_report_records_the_limit_in_documents(self):
        self.write_gold(list(self.texts)[1:35])
        rc, _, _ = self.run_ht(MARKER_RULE, "B3", "rules.py", "--report", str(self.tmp / "report.json"))
        report = json.loads((self.tmp / "report.json").read_text())
        self.assertEqual((rc, report["status"], report["gold_count"], report["limit_documents"]), (0, "PASS", 34, 1))

    def test_debug_lists_the_documents_that_did_not_pass_without_their_text(self):
        src = ("RULES = {'X': lambda t: ('drop', '') if t == 'DROPME' else "
               "('fix', t.replace('<issue_start> ', ''))}\n")
        rc, out, err = self.run_ht(src, "x", "rules.py", "--debug")
        self.assertEqual(rc, 1)
        self.assertIn("debug: gold: 200 doc_ids from gold.csv", err)
        self.assertIn("debug: X: 3 gold documents did not pass", err)
        self.assertIn("  d000  changed  29 -> 15 characters, first difference at offset 0", err)
        self.assertIn("  d002  dropped  (6 characters)", err)
        self.assertNotIn("marked document", err + out)            # ids and offsets only, never text
        rc, out, err = self.run_ht(src, "x")                       # nothing extra without --debug
        self.assertNotIn("debug:", err)

    def test_debug_names_the_error_of_a_crashing_rule_and_works_with_rule_all(self):
        src = "def medicine_b3(t):\n    return 1 / 0 if t == 'DROPME' else t\n"
        md = "**Disease:** B3, x\n```python\n" + src + "```\n"
        rc, _, err = self.run_ht(md, "all", "medicine.md", "--debug")
        self.assertEqual(rc, 2)
        self.assertIn("debug: section B3: rule medicine_b3 (from function medicine_b3)", err)
        self.assertIn("  d002  error ZeroDivisionError", err)

    def test_limit_option(self):
        rc, out, _ = self.run_ht("cure_a1 = lambda t: t.upper()\n", "A1", "rules.py", "--limit", "100")
        self.assertEqual(rc, 0)
        self.assertIn("documents changed: 199", out)  # "DROPME" is already upper case

    def test_unchanged_fix_is_not_counted(self):
        rc, out, _ = self.run_ht("RULES = {'A1': lambda t: ('fix', t)}\n", "A1")
        self.assertEqual((rc, "documents changed: 0" in out), (0, True))

    def test_keep_with_different_text_counts_as_changed(self):
        rc, out, _ = self.run_ht("RULES = {'A1': lambda t: ('keep', t + ' ')}\n", "A1")
        self.assertEqual(rc, 1)
        self.assertIn("documents changed: 200", out)

    def test_show_lists_ids_only(self):
        _, out, _ = self.run_ht("cure_b3 = lambda t: t.replace('<issue_start>', '')\n", "B3", "rules.py", "--show", "5")
        self.assertIn("affected doc_ids: d000, d001", out)
        self.assertNotIn("marked document", out)

    def test_rule_exception_is_an_error_not_a_pass(self):
        rc, out, _ = self.run_ht("def cure_a1(t):\n    raise ValueError(t)\n", "A1")
        self.assertEqual(rc, 2)
        self.assertIn("result: ERROR", out)
        self.assertNotIn("clean document", out)  # the document text must not leak into the output

    def test_bad_return_value_is_an_error(self):
        rc, out, _ = self.run_ht("cure_a1 = lambda t: None\n", "A1")
        self.assertEqual(rc, 2)

    @unittest.skipUnless(harm_test.HAS_ALARM, "needs SIGALRM")
    def test_timeout(self):
        src = "import time\ndef cure_a1(t):\n    time.sleep(5)\n    return t\n"
        rc, out, _ = self.run_ht(src, "A1", "rules.py", "--timeout", "0.01")
        self.assertEqual(rc, 2)
        self.assertIn("RuleTimeout", out)

    def test_runs_without_sigalrm_with_a_warning(self):
        # Windows has no SIGALRM: the test still runs, without the per-document limit, and says so
        saved = harm_test.HAS_ALARM
        harm_test.HAS_ALARM = False
        try:
            rc, out, err = self.run_ht("def cure_b3(t):\n    return t.replace('<issue_start>', '')\n", "B3")
        finally:
            harm_test.HAS_ALARM = saved
        self.assertEqual(rc, 0)
        self.assertIn("documents changed: 2", out)
        self.assertIn("no SIGALRM on this platform", err)

    def test_watchdog_stops_a_hanging_rule_without_sigalrm(self):
        # Without SIGALRM the faulthandler watchdog ends the run at --timeout. It exits the process, so run it in one.
        (self.tmp / "rules.py").write_text("import time\ndef cure_a1(t):\n    time.sleep(30)\n    return t\n")
        script = (f"import sys; sys.path.insert(0, {str(Path(harm_test.__file__).parent)!r})\n"
                  "import harm_test; harm_test.HAS_ALARM = False\n"
                  f"sys.exit(harm_test.main(['--pack', {str(self.tmp / 'pack.parquet')!r}, '--gold', "
                  f"{str(self.tmp / 'gold.csv')!r}, '--rules', {str(self.tmp / 'rules.py')!r}, '--rule', 'A1', "
                  "'--timeout', '0.5']))\n")
        start = time.monotonic()
        proc = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=60)
        self.assertLess(time.monotonic() - start, 20)  # stopped at the limit, not after the 30 s sleep
        self.assertEqual(proc.returncode, 1)  # never a PASS
        self.assertIn("Timeout", proc.stderr)
        self.assertNotIn("clean document", proc.stderr + proc.stdout)  # a stack trace only, no document text

    def test_debug_names_the_document_a_timeout_ended_the_run_on(self):
        # The watchdog's stack trace does not say which document hung, so --debug prints it before each call.
        (self.tmp / "rules.py").write_text(
            "import time\ndef cure_a1(t):\n    if t == 'DROPME':\n        time.sleep(30)\n    return t\n")
        script = (f"import sys; sys.path.insert(0, {str(Path(harm_test.__file__).parent)!r})\n"
                  "import harm_test; harm_test.HAS_ALARM = False\n"
                  f"sys.exit(harm_test.main(['--pack', {str(self.tmp / 'pack.parquet')!r}, '--gold', "
                  f"{str(self.tmp / 'gold.csv')!r}, '--rules', {str(self.tmp / 'rules.py')!r}, '--rule', 'A1', "
                  "'--timeout', '0.5', '--debug']))\n")
        proc = subprocess.run([sys.executable, "-I", "-c", script], capture_output=True, text=True, timeout=60)
        self.assertEqual(proc.returncode, 1)
        lines = [ln for ln in proc.stderr.splitlines() if ln.startswith("debug: calling the rule on")]
        self.assertEqual(lines[-1], "debug: calling the rule on d002")   # the document whose text is DROPME
        self.assertLess(proc.stderr.index("debug: calling the rule on d002"), proc.stderr.index("Timeout"))
        self.assertNotIn("DROPME", proc.stderr + proc.stdout)

    def test_a_rules_file_on_another_drive_has_no_relative_path(self):
        # Windows: os.path.relpath raises ValueError when the two paths are on different drives
        saved = harm_test.os.path.relpath
        def other_drive(*_):
            raise ValueError("path is on mount 'D:', start on mount 'C:'")
        harm_test.os.path.relpath = other_drive
        try:
            path = harm_test.repo_relative(self.tmp / "rules.py")
        finally:
            harm_test.os.path.relpath = saved
        self.assertEqual(path, (self.tmp / "rules.py").resolve().as_posix())

    def test_the_report_is_written_with_lf_line_endings(self):
        report = self.tmp / "report.json"
        self.run_ht("cure_b3 = lambda t: t\n", "B3", "rules.py", "--report", str(report))
        self.assertNotIn(b"\r", report.read_bytes())

    def test_missing_rule(self):
        rc, _, err = self.run_ht("def cure_b3(t):\n    return t\n", "Z9")
        self.assertEqual(rc, 2)
        self.assertIn("no rule for Z9", err)

    def test_gold_id_missing_from_pack(self):
        self.write_gold(["d000", "nope"])
        rc, _, err = self.run_ht("cure_b3 = lambda t: t\n", "B3")
        self.assertEqual(rc, 2)
        self.assertIn("not in the pack", err)

    def test_duplicate_gold_ids_rejected(self):
        self.write_gold(["d005", "d005", "d006"])
        rc, out, err = self.run_ht("cure_b3 = lambda t: t\n", "B3")
        self.assertEqual(rc, 2)
        self.assertIn("duplicate", err)

    FENCE = "```"

    def plain_md(self, *sections):
        """A plain medicine.md: (code, python) for each section."""
        return "".join(f"**Disease:** {code}, x\n{self.FENCE}python\n{src}\n{self.FENCE}\n" for code, src in sections)

    def test_markdown_rejects_nonrunnable_python_fragments(self):
        md = ("# Medicine\n\n- fragment:\n  ```python\n  text = MARKER.sub('', text)\n  ```\n\n"
              "```python\nimport re\n\ndef cure_b3(text):\n    return re.sub('<issue_start>', '', text)\n```\n")
        rc, out, err = self.run_ht(md, "B3", "medicine.md")
        self.assertEqual(rc, 2)
        self.assertIn("medicine.md block 1", err)

    def test_markdown_rejects_a_block_with_a_syntax_error(self):
        md = self.plain_md(("B3", "def cure_b3(text) return text"))
        rc, _, err = self.run_ht(md, "B3", "medicine.md")
        self.assertEqual(rc, 2)
        self.assertIn("medicine.md block 1 has a syntax error", err)

    def test_markdown_function_names_other_than_cure_are_found(self):
        md = self.plain_md(("B3", "def _strip(text):\n    return text.replace('<issue_start> ', '')\n\n"
                                  "def medicine_b3(text):\n    return _strip(text)"))
        rc, out, _ = self.run_ht(md, "B3", "medicine.md")
        self.assertEqual((rc, "documents changed: 2" in out), (0, True))

    def test_markdown_without_a_disease_line_is_an_error(self):
        rc, _, err = self.run_ht("```python\ndef cure_b3(text):\n    return text\n```\n", "B3", "medicine.md")
        self.assertEqual(rc, 2)
        self.assertIn("no '**Disease:** <code>' line found", err)

    def test_markdown_same_name_with_different_code_is_an_error(self):
        md = self.plain_md(("B3", "def _s(t):\n    return 'a'\ndef medicine_b3(t):\n    return _s(t)"),
                           ("B4", "def _s(t):\n    return 'b'\ndef medicine_b4(t):\n    return _s(t)"))
        rc, _, err = self.run_ht(md, "B3", "medicine.md")
        self.assertEqual(rc, 2)
        self.assertIn("_s is defined with different code", err)

    def test_rule_all_on_a_plain_markdown(self):
        md = self.plain_md(("B3", "def medicine_b3(t):\n    return t.replace('<issue_start> ', '')"),
                           ("E1", "def medicine_e1(t):\n    return t == 'DROPME'"))
        rc, out, _ = self.run_ht(md, "all", "medicine.md", "--limit", "5")      # 3 of 200 documents: over 1%, under 5%
        self.assertEqual(rc, 0)
        self.assertIn("B3", out)
        self.assertIn("E1", out)
        self.assertIn("ALL IN ORDER", out)

    def test_the_harm_test_runs_the_code_the_extractor_writes(self):
        md = self.plain_md(("B3", "def medicine_b3(t):\n    return t"))
        (self.tmp / "medicine.md").write_text(md)
        _, digest = harm_test.load_rules(self.tmp / "medicine.md")
        code = harm_test.medicine_extract.rules_from_markdown(md, "medicine.md")[1]
        self.assertEqual(digest, harm_test.medicine_extract.fingerprint(code))

    def test_the_code_sha256_ignores_comments_and_spacing_but_not_code(self):
        def digest(src):
            (self.tmp / "medicine.md").write_text(self.plain_md(("B3", src)))
            return harm_test.load_rules(self.tmp / "medicine.md")[1]
        base = digest("def medicine_b3(t):\n    return t.strip()")
        self.assertEqual(base, digest("# strips\ndef medicine_b3( t ):\n\n    return t.strip()   # done\n"))
        self.assertNotEqual(base, digest("def medicine_b3(t):\n    return t.lstrip()"))

    def test_two_sections_for_one_code_stop_the_test(self):
        md = ("**Disease:** B3, markers\n```python\ndef cure_b3(text):\n    return text.replace('<issue_start> ', '')\n```\n"
              "**Disease:** B3, more markers\n```python\ndef other(text):\n    return text\n```\n"
              "**Disease:** B4, tags\n```python\ndef cure_b4(text):\n    return text\n```\n")
        rc, out, err = self.run_ht(md, "B3", "medicine.md")
        self.assertEqual(rc, 2)
        self.assertIn("2 sections for B3 in medicine.md, so I can't tell which to test", err)
        self.assertNotIn("documents changed", out)
        rc, out, _ = self.run_ht(md, "B4", "medicine.md")      # one section for B4: still tested
        self.assertEqual(rc, 0)
        self.assertIn("documents changed: 0", out)



if __name__ == "__main__":
    unittest.main()
