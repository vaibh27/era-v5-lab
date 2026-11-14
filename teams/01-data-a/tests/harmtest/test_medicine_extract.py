"""Tests for medicine_extract.py. Run: python -I teams/01-data-a/tests/harmtest/test_medicine_extract.py"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harmtest"))  # python -I: not on the path by itself
import harm_test  # noqa: E402
import medicine_extract  # noqa: E402

FENCE = "```"


def md(*parts):
    return "\n".join(parts)


def py(code):
    return f"{FENCE}python\n{code}\n{FENCE}"


class ExtractCase(unittest.TestCase):
    def extract(self, text, load=True):
        tmp = Path(tempfile.mkdtemp())
        src, out = tmp / "medicine.md", tmp / "rules.py"
        src.write_text(text)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            rc = medicine_extract.main([str(src), "--out", str(out)])
        ns = harm_test.load_rules(out)[0] if load and out.exists() else None  # loading RUNS the generated code
        return rc, stdout.getvalue(), ns

    def test_cure_functions_found_by_name(self):
        rc, out, ns = self.extract(md(
            "1. **Disease:** B3, ghost markers", py("def cure_b3(t):\n    return t.replace('<x>', '')"),
            "## Medicine 2", "1. **Disease:** B4, tags", py("import re\ndef cure_b4(t):\n    return re.sub('<y>', '', t)")))
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(ns["RULES"]), ["B3", "B4"])
        self.assertEqual(ns["RULES"]["B3"]("a<x>b"), "ab")

    def test_re_is_always_imported(self):
        rc, _, ns = self.extract(md("**Disease:** B3, markers", py("def cure_b3(t):\n    return re.sub('<x>', '', t)")))
        self.assertEqual(rc, 0)
        self.assertEqual(ns["RULES"]["B3"]("a<x>b"), "ab")      # the medicine.md never imports re

    def test_medicine_code_name_wins_over_helpers(self):
        rc, out, ns = self.extract(md("**Disease:** B3, markers", py(
            "def strip_header(t):\n    return t.replace('<h>', '')\n\n"
            "def _fix_regex(t):\n    return t.replace('<x>', '')\n\n"
            "def medicine_b3(t):\n    return _fix_regex(strip_header(t))")))
        self.assertEqual(rc, 0)
        self.assertIn("function medicine_b3", out)
        self.assertEqual(ns["RULES"]["B3"]("a<x>b<h>"), "ab")

    def test_any_function_name_when_it_is_the_only_one_in_the_section(self):
        rc, out, ns = self.extract(md("**Disease:** A1, junk", py("def strip_junk(t):\n    return t.strip()")))
        self.assertEqual(rc, 0)
        self.assertIn("the only function in the section", out)
        self.assertEqual(ns["RULES"]["A1"]("  x "), "x")

    def test_several_functions_are_reported_not_guessed(self):
        rc, out, ns = self.extract(md("**Disease:** A1, junk", py("def one(t):\n    return t\ndef two(t):\n    return t")))
        self.assertEqual(rc, 1)
        self.assertIn("several functions in the section (one, two)", out)
        self.assertIsNone(ns)  # nothing written

    def test_fragments_are_left_out_and_an_annotators_own_rules_dict_is_ignored(self):
        rc, out, ns = self.extract(md(
            "## Medicine 1", "1. **Disease:** B3, markers", py("MARKER = 1\ntext = MARKER + text"),
            py("def b3(t):\n    return 'function'"),
            "## Medicine 2", "1. **Disease:** E1, generated", py("if x drop"),
            "## Appendix", py("RULES = {'B3': lambda t: 'dict', 'E1': lambda t: ('drop', '')}")))
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(ns["RULES"]), ["B3"])                   # E1 has no function in its section
        self.assertEqual(ns["RULES"]["B3"]("x"), "function")            # the annotator's dict is not read
        self.assertIn("left out: block 1, fragment", out)
        self.assertIn("left out: block 3, syntax error", out)
        self.assertIn("E1: no function found for this section", out)

    def test_support_block_that_fails_does_not_break_the_file(self):
        rc, out, ns = self.extract(md("**Disease:** A1, junk", py("PATTERN = re.compile('x')"),  # `re` never imported
                                      py("def cure_a1(t):\n    return t")))
        self.assertEqual(rc, 0)
        self.assertIn("1 support", out)
        self.assertEqual(ns["RULES"]["A1"]("q"), "q")

    def test_second_medicine_without_ids_is_reported(self):
        rc, out, ns = self.extract(md(
            "**Disease:** B3, first", py("def cure_b3(t):\n    return 'first'"),
            "**Disease:** B3, second", py("def other(t):\n    return 'second'")))
        self.assertEqual(rc, 0)
        self.assertEqual(ns["RULES"]["B3"](""), "first")              # RULES holds one function for each code
        self.assertIn("only the first rule of each code", out)

    def test_no_disease_line(self):
        rc, _, ns = self.extract(md("# Medicine sheet", py("def cure_b3(t):\n    return t")))
        self.assertEqual(rc, 1)
        self.assertIsNone(ns)

    def test_extractor_does_not_run_the_code(self):
        marker = Path(tempfile.mkdtemp()) / "ran"
        rc, _, _ = self.extract(md("**Disease:** A1, junk", py(
            f"open({str(marker)!r}, 'w').write('x')\ndef cure_a1(t):\n    return t")), load=False)
        self.assertEqual(rc, 0)
        self.assertFalse(marker.exists())

    def test_default_out_only_inside_submissions(self):
        self.assertIsNone(medicine_extract.default_out(Path("/tmp/x/medicine.md")))
        p = Path("/r/diagnosis/submissions/code-rust/amukka/medicine.md")
        self.assertEqual(medicine_extract.default_out(p).parts[-3:], ("rules", "code-rust", "amukka.py"))


class SharedReaderCase(unittest.TestCase):
    """rules_from_markdown is the one reader of a medicine.md: the extractor's file and harm_test's code are the same."""

    TEXT = md("**Disease:** B3, markers", py("def medicine_b3(t):\n    return t.replace('<m>', '')"),
              "**Disease:** E1, generated", py("def helper(t):\n    return t\n\ndef cure_e1(t):\n    return ('drop', '')"),
              "**Disease:** A1, nothing")

    def test_returns_the_rules_the_notes_and_the_code(self):
        found, code = medicine_extract.rules_from_markdown(self.TEXT)
        self.assertEqual([(r["code"], r["expr"]) for r in found.chosen], [("B3", "medicine_b3"), ("E1", "cure_e1")])
        self.assertEqual(found.notes, ["A1: no function found for this section"])
        self.assertEqual(len(found.sections), 3)
        self.assertIn("import re", code)
        self.assertIn('"B3": medicine_b3', code)
        self.assertNotIn("RULES_BY_ID", code)                      # no IDs here

    def test_the_code_is_never_run(self):
        marker = Path(tempfile.mkdtemp()) / "ran"
        text = md("**Disease:** A1, x", py(f"open({str(marker)!r}, 'w').write('x')\ndef medicine_a1(t):\n    return t"))
        _, code = medicine_extract.rules_from_markdown(text)
        self.assertFalse(marker.exists())
        self.assertIn("def medicine_a1", code)

    def test_the_extractor_writes_the_same_code(self):
        tmp = Path(tempfile.mkdtemp())
        src, out = tmp / "medicine.md", tmp / "rules.py"
        src.write_text(self.TEXT)
        with contextlib.redirect_stdout(io.StringIO()):
            medicine_extract.main([str(src), "--out", str(out)])            # a path with no pack: no IDs
        self.assertEqual(out.read_text(), medicine_extract.rules_from_markdown(self.TEXT, "medicine.md")[1])


class FingerprintCase(unittest.TestCase):
    def test_comments_spacing_and_the_header_do_not_count(self):
        a = medicine_extract.rules_from_markdown(md("**Disease:** B3, x", py("def medicine_b3(t):\n    return t")), "a.md")[1]
        b = medicine_extract.rules_from_markdown(md("**Disease:** B3, y", py("# note\ndef medicine_b3( t ):\n    return t  # ok")), "b.md")[1]
        c = medicine_extract.rules_from_markdown(md("**Disease:** B3, x", py("def medicine_b3(t):\n    return t + ' '")), "a.md")[1]
        self.assertEqual(medicine_extract.fingerprint(a), medicine_extract.fingerprint(b))
        self.assertNotEqual(medicine_extract.fingerprint(a), medicine_extract.fingerprint(c))

    def test_the_registry_keeps_the_fingerprint_that_harm_test_prints(self):
        tmp = Path(tempfile.mkdtemp())
        src = tmp / "submissions" / "code-rust" / "alice" / "medicine.md"
        src.parent.mkdir(parents=True)
        src.write_text(md("**Disease:** B3, x", py("def medicine_b3(t):\n    return t")))
        reg = tmp / "rules" / "registry.json"
        with contextlib.redirect_stdout(io.StringIO()):
            medicine_extract.main([str(src), "--registry", str(reg)])
        entry = json.loads(reg.read_text())["rules"][0]
        self.assertEqual(entry["code_sha256"], harm_test.load_rules(src)[1])


class ClashCase(unittest.TestCase):
    """A name defined with different code in two blocks: the last one would win for every rule."""

    def clashes(self, *sections):
        text = md(*[x for code, py_code in sections for x in (f"**Disease:** {code}, x", py(py_code))])
        return medicine_extract.read_rules(text).clashes

    def test_the_same_helper_copied_into_two_sections_is_fine(self):
        helper = "def _strip(t):\n    return t.strip()\n"
        self.assertEqual(self.clashes(("B3", helper + "def medicine_b3(t):\n    return _strip(t)"),
                                      ("E1", helper + "def medicine_e1(t):\n    return _strip(t)")), [])

    def test_formatting_and_comments_do_not_count_as_different(self):
        self.assertEqual(self.clashes(("B3", "def _h(t):\n    return t.strip()  # one"),
                                      ("E1", "def _h( t ):\n    # two\n    return t.strip()")), [])

    def test_same_name_with_different_code_is_reported(self):
        got = self.clashes(("B3", "def _strip(t):\n    return t.replace('<m>', '')"),
                           ("E1", "def _strip(t):\n    return t.replace('<h>', '')"))
        self.assertEqual(len(got), 1)
        self.assertIn("_strip is defined with different code in block 1 (B3), block 2 (E1)", got[0])
        self.assertIn("the last one is used by every rule", got[0])

    def test_constants_and_imports_count_too(self):
        got = self.clashes(("B3", "import re\nMARKER = re.compile('<m>')"),
                           ("E1", "import regex as re\nMARKER = re.compile('<h>')"))
        self.assertEqual(sorted(g.split(" ")[0] for g in got), ["MARKER", "re"])

    def test_a_name_changed_inside_one_block_is_not_a_clash(self):
        self.assertEqual(self.clashes(("B3", "X = 1\nX = 2\ndef medicine_b3(t):\n    return t")), [])

    def test_the_extractor_warns_and_still_writes(self):
        text = md("**Disease:** B3, x", py("def _s(t):\n    return 'a'\ndef medicine_b3(t):\n    return _s(t)"),
                  "**Disease:** E1, y", py("def _s(t):\n    return 'b'\ndef medicine_e1(t):\n    return _s(t)"))
        tmp = Path(tempfile.mkdtemp())
        src, out = tmp / "medicine.md", tmp / "rules.py"
        src.write_text(text)
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = medicine_extract.main([str(src), "--out", str(out)])
        self.assertEqual(rc, 0)
        self.assertIn("warning: _s is defined with different code", stdout.getvalue())
        self.assertTrue(out.exists())


class IdCase(unittest.TestCase):
    """Rule IDs and the registry. Medicine files live in <tmp>/submissions/<pack>/<handle>/medicine.md."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.registry = self.tmp / "rules" / "registry.json"

    def write_medicine(self, pack, handle, text):
        d = self.tmp / "submissions" / pack / handle
        d.mkdir(parents=True, exist_ok=True)
        (d / "medicine.md").write_text(text)
        return d / "medicine.md"

    def run_extract(self, path):
        out = self.tmp / "rules" / path.parent.parent.name / f"{path.parent.name}.py"
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(io.StringIO()):
            rc = medicine_extract.main([str(path), "--out", str(out), "--registry", str(self.registry)])
        return rc, stdout.getvalue(), out

    def entries(self):
        return {e["id"]: e for e in json.loads(self.registry.read_text())["rules"]}

    @staticmethod
    def b3(ret="t"):
        return md("**Disease:** B3, markers", py(f"def cure_b3(t):\n    return {ret}"))

    def test_team_prefix_and_counter(self):
        self.run_extract(self.write_medicine("code-rust", "alice", self.b3()))
        self.run_extract(self.write_medicine("code-rust", "bob", self.b3()))
        self.run_extract(self.write_medicine("cadj-stackexchange", "carol", self.b3()))
        self.assertEqual(sorted(self.entries()), ["DA-B3-001", "DA-B3-002", "DB-B3-001"])
        self.assertEqual(self.entries()["DA-B3-002"]["handle"], "bob")

    def test_rerun_keeps_ids_and_marks_withdrawn(self):
        two = md("**Disease:** B3, a", py("def cure_b3(t):\n    return t"), "**Disease:** B4, b", py("def cure_b4(t):\n    return t"))
        path = self.write_medicine("code-rust", "alice", two)
        self.run_extract(path)
        self.run_extract(self.write_medicine("code-rust", "bob", self.b3()))      # DA-B3-002
        before = self.entries()
        self.run_extract(path)
        self.assertEqual(self.entries(), before)                                   # rerun changes nothing
        self.run_extract(self.write_medicine("code-rust", "alice", self.b3()))   # B4 removed
        after = self.entries()
        self.assertTrue(after["DA-B4-001"]["withdrawn"])
        self.assertFalse(after["DA-B3-001"]["withdrawn"])
        self.assertEqual(sorted(after), ["DA-B3-001", "DA-B3-002", "DA-B4-001"])  # the ID is kept, not reused
        self.run_extract(self.write_medicine("code-rust", "alice", two))         # B4 comes back
        self.assertFalse(self.entries()["DA-B4-001"]["withdrawn"])

    def test_generated_file_has_rules_by_id(self):
        _, out, path = self.run_extract(self.write_medicine("code-rust", "alice", self.b3("'x'")))
        self.assertIn("-> DA-B3-001", out)
        ns = harm_test.load_rules(path.parent / "alice.py")[0]
        self.assertEqual(ns["RULES_BY_ID"]["DA-B3-001"]("anything"), "x")

    def test_harm_test_finds_the_rule_by_id(self):
        self.run_extract(self.write_medicine("code-rust", "alice", self.b3("t.replace('<m>', '')")))
        pack, gold = self.tmp / "pack.parquet", self.tmp / "gold.csv"
        ids = [f"d{i}" for i in range(100)]
        pq.write_table(pa.table({"doc_id": ids, "text": ["plain"] * 99 + ["has <m> marker"]}), pack)
        gold.write_text("doc_id,reader_1,reader_2\n" + "".join(f"{i},a,b\n" for i in ids))
        argv = ["--pack", str(pack), "--gold", str(gold), "--registry", str(self.registry)]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = harm_test.main(argv + ["--rule", "DA-B3-001"])
        self.assertEqual(rc, 0)                                   # 1 of 100 = 1.00%
        self.assertIn("documents changed: 1", out.getvalue())
        self.assertIn("rule: DA-B3-001", out.getvalue())
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(harm_test.main(argv + ["--rule", "DA-B3-009"]), 2)   # not in the registry
            self.assertEqual(harm_test.main(argv + ["--rule", "B3"]), 2)           # a code needs --rules
        self.assertIn("not in the registry", err.getvalue())
        self.assertIn("require --rules", err.getvalue())

    def test_path_outside_submissions_gets_no_ids(self):
        src = self.tmp / "medicine.md"
        src.write_text(self.b3())
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            rc = medicine_extract.main([str(src), "--out", str(self.tmp / "r.py"), "--registry", str(self.registry)])
        self.assertEqual(rc, 0)
        self.assertIn("no IDs", stdout.getvalue())
        self.assertFalse(self.registry.exists())

    def two_b3(self):
        return md("**Disease:** B3, first", "3. **Action:** fix the text. No document is dropped.",
                  py("def cure_b3(t):\n    return 'first'"),
                  "**Disease:** B3, second", "3. **Action:** drop.", py("def strict_b3(t):\n    return ('drop', '')"))

    def test_two_medicines_for_one_code_get_two_ids(self):
        _, out, path = self.run_extract(self.write_medicine("code-rust", "alice", self.two_b3()))
        self.assertIn("-> DA-B3-001", out)
        self.assertIn("-> DA-B3-002", out)
        entries = self.entries()
        self.assertEqual([(e["n"], e["action"]) for e in entries.values()], [(1, "fix"), (2, "drop")])
        ns = harm_test.load_rules(path)[0]
        self.assertEqual(ns["RULES_BY_ID"]["DA-B3-001"]("x"), "first")
        self.assertEqual(ns["RULES_BY_ID"]["DA-B3-002"]("x"), ("drop", ""))
        self.assertEqual(ns["RULES"]["B3"]("x"), "first")

    def test_second_medicine_cannot_reuse_the_first_function(self):
        text = md("**Disease:** B3, first", py("def cure_b3(t):\n    return t"),
                  "**Disease:** B3, second", "(no code)")
        _, out, _ = self.run_extract(self.write_medicine("code-rust", "alice", text))
        self.assertIn("B3 (medicine 2): no function found", out)
        self.assertEqual(sorted(self.entries()), ["DA-B3-001"])

    def test_old_registry_entries_without_n_count_as_the_first_rule(self):
        path = self.write_medicine("code-rust", "alice", self.two_b3())
        self.registry.parent.mkdir(parents=True)
        self.registry.write_text(json.dumps({"rules": [{"id": "DA-B3-001", "pack": "code-rust", "handle": "alice",
                                                        "code": "B3", "file": "x.py", "withdrawn": False}]}))
        self.run_extract(path)
        self.assertEqual({k: (v["n"], v["file"]) for k, v in self.entries().items()},
                         {"DA-B3-001": (1, "code-rust/alice.py"), "DA-B3-002": (2, "code-rust/alice.py")})

    def test_new_ids_skip_ids_used_in_medicine_sheets(self):
        # a team sheet (<team>/*-medicines/*.md) already numbers its own B3 rule DA-B3-001
        sheet = self.tmp / "code-medicines" / "medicine_code.md"
        sheet.parent.mkdir(parents=True)
        sheet.write_text(py('RULE_ID = "DA-B3-001"\nCODE = "B3"\ndef medicine(doc):\n    return ("keep", doc["text"])'))
        _, out, _ = self.run_extract(self.write_medicine("code-rust", "alice", self.two_b3()))
        self.assertEqual(sorted(self.entries()), ["DA-B3-002", "DA-B3-003"])
        self.assertNotIn("-> DA-B3-001", out)

    def test_a_rules_file_on_another_drive_is_registered_with_its_full_path(self):
        # Windows: os.path.relpath raises ValueError when the two paths are on different drives
        saved = medicine_extract.os.path.relpath
        def other_drive(*_):
            raise ValueError("path is on mount 'D:', start on mount 'C:'")
        medicine_extract.os.path.relpath = other_drive
        try:
            rc, _, out = self.run_extract(self.write_medicine("code-rust", "alice", self.b3()))
        finally:
            medicine_extract.os.path.relpath = saved
        self.assertEqual(rc, 0)
        self.assertEqual(self.entries()["DA-B3-001"]["file"], out.resolve().as_posix())

    def test_the_registry_and_the_rules_file_are_written_with_lf_line_endings(self):
        _, _, out = self.run_extract(self.write_medicine("code-rust", "alice", self.b3()))
        for path in (self.registry, out):
            self.assertNotIn(b"\r", path.read_bytes(), path.name)

    def test_registry_paths_use_forward_slashes(self):
        self.registry.parent.mkdir(parents=True)
        self.registry.write_text(json.dumps({"rules": [{"id": "DA-B3-001", "pack": "code-rust", "handle": "alice",
                                                        "code": "B3", "n": 1, "file": "code-rust\\alice.py",
                                                        "withdrawn": False}]}))
        self.run_extract(self.write_medicine("code-rust", "alice", self.two_b3()))
        self.assertTrue(all("\\" not in e["file"] for e in self.entries().values()))
        self.assertEqual(self.entries()["DA-B3-001"]["file"], "code-rust/alice.py")


    def test_a_sheet_that_no_longer_gives_any_rule_withdraws_its_ids(self):
        path = self.write_medicine("code-rust", "alice", self.b3())
        self.run_extract(path)                                                     # DA-B3-001, rules file written
        self.run_extract(self.write_medicine("code-rust", "bob", self.b3()))      # DA-B3-002
        broken = md("**Disease:** B3, markers", py("keep = [l for l in t.split() if not B3.match(l)]"))  # fragment only
        rc, out, rules_file = self.run_extract(self.write_medicine("code-rust", "alice", broken))
        self.assertEqual(rc, 1)
        self.assertTrue(self.entries()["DA-B3-001"]["withdrawn"])
        self.assertFalse(self.entries()["DA-B3-002"]["withdrawn"])               # other handles untouched
        self.assertFalse(rules_file.exists())                                      # the stale rules file is gone
        self.assertIn("DA-B3-001", out)
        rc, _, rules_file = self.run_extract(self.write_medicine("code-rust", "alice", self.b3()))  # fixed again
        self.assertEqual(rc, 0)
        self.assertFalse(self.entries()["DA-B3-001"]["withdrawn"])                # same ID comes back
        self.assertTrue(rules_file.exists())

    def test_a_sheet_with_no_disease_line_withdraws_its_ids(self):
        path = self.write_medicine("code-rust", "alice", self.b3())
        self.run_extract(path)
        rc, _, _ = self.run_extract(self.write_medicine("code-rust", "alice", "doc_id,read,verdict\n"))  # labels pasted
        self.assertEqual(rc, 1)
        self.assertTrue(self.entries()["DA-B3-001"]["withdrawn"])

    def test_a_file_the_tool_did_not_write_is_kept(self):
        path = self.write_medicine("code-rust", "alice", md("**Disease:** B3, x", py("keep = B3.sub('', t)")))
        own = self.tmp / "rules" / "code-rust" / "alice.py"
        own.parent.mkdir(parents=True, exist_ok=True)
        own.write_text("# hand-written, not generated\n")
        rc, _, _ = self.run_extract(path)
        self.assertEqual(rc, 1)
        self.assertTrue(own.exists())


class DirCase(unittest.TestCase):
    """--dir: many medicine.md files, one run, one registry."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.src = self.tmp / "downloads"          # any folder name works: the pack and handle come from the folders
        self.registry = self.tmp / "rules" / "registry.json"

    def medicine(self, pack, handle, text):
        d = self.src / pack / handle
        d.mkdir(parents=True, exist_ok=True)
        (d / "medicine.md").write_text(text)

    def run_dir(self, *extra):
        argv = ["--dir", str(self.src), "--registry", str(self.registry), *extra]
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = medicine_extract.main(argv)
        return rc, out.getvalue(), err.getvalue()

    def entries(self):
        return {e["id"]: e for e in json.loads(self.registry.read_text())["rules"]}

    @staticmethod
    def b3(ret="t"):
        return md("**Disease:** B3, markers", py(f"def cure_b3(t):\n    return {ret}"))

    def test_all_packs_and_handles_in_one_run(self):
        self.medicine("code-rust", "alice", self.b3())
        self.medicine("code-rust", "bob", self.b3())
        self.medicine("en-dclm", "carol", self.b3())
        self.medicine("cadj-stackexchange", "dave", self.b3())
        rc, out, _ = self.run_dir()
        self.assertEqual(rc, 0)
        self.assertEqual(sorted(self.entries()), ["DA-B3-001", "DA-B3-002", "DA-B3-003", "DB-B3-001"])
        self.assertEqual({e["handle"] for e in self.entries().values()}, {"alice", "bob", "carol", "dave"})
        for pack, handle in [("code-rust", "alice"), ("code-rust", "bob"), ("en-dclm", "carol"), ("cadj-stackexchange", "dave")]:
            self.assertTrue((self.tmp / "rules" / pack / f"{handle}.py").exists())
        self.assertIn("4 of 4 medicine.md files done", out)

    def test_same_registry_as_running_the_files_one_by_one(self):
        files = [("code-rust", "alice"), ("code-rust", "bob"), ("en-dclm", "carol")]
        for pack, handle in files:
            self.medicine(pack, handle, self.b3())
        self.run_dir()
        by_dir = self.registry.read_text()
        self.registry.unlink()
        sub = self.tmp / "submissions"                      # one by one needs the real layout: .../submissions/<pack>/<handle>/
        for pack, handle in files:                          # in the same (sorted) order that --dir uses
            d = sub / pack / handle
            d.mkdir(parents=True)
            (d / "medicine.md").write_text(self.b3())
            with contextlib.redirect_stdout(io.StringIO()):
                medicine_extract.main([str(d / "medicine.md"), "--registry", str(self.registry)])
        self.assertEqual(self.registry.read_text(), by_dir)

    def test_rerun_changes_nothing(self):
        self.medicine("code-rust", "alice", self.b3())
        self.medicine("en-dclm", "carol", self.b3())
        self.run_dir()
        before = self.registry.read_text()
        self.run_dir()
        self.assertEqual(self.registry.read_text(), before)

    def test_a_medicine_without_rules_is_listed_and_the_others_still_done(self):
        self.medicine("code-rust", "alice", self.b3())
        self.medicine("code-rust", "bob", "# Medicine sheet\nnothing here yet\n")
        rc, out, err = self.run_dir()
        self.assertEqual(rc, 1)
        self.assertEqual(sorted(self.entries()), ["DA-B3-001"])
        self.assertIn("1 of 2 medicine.md files done", out)
        self.assertIn("no rules: code-rust/bob", out)
        self.assertIn("no '**Disease:**", err)

    def test_a_deleted_medicine_withdraws_its_rules(self):
        self.medicine("code-rust", "alice", self.b3())
        self.medicine("code-rust", "bob", self.b3())
        self.run_dir()                                                              # DA-B3-001 alice, -002 bob
        (self.src / "code-rust" / "bob" / "medicine.md").unlink()                  # bob's PR closed
        rc, out, _ = self.run_dir()
        self.assertEqual(rc, 0)
        self.assertTrue(self.entries()["DA-B3-002"]["withdrawn"])
        self.assertFalse(self.entries()["DA-B3-001"]["withdrawn"])
        self.assertFalse((self.tmp / "rules" / "code-rust" / "bob.py").exists())
        self.assertIn("medicine.md gone, rules withdrawn: code-rust/bob", out)
        self.medicine("code-rust", "bob", self.b3())                               # bob comes back
        self.run_dir()
        self.assertFalse(self.entries()["DA-B3-002"]["withdrawn"])                 # with the same ID

    def test_a_folder_with_one_pack_leaves_the_other_packs_alone(self):
        self.medicine("code-rust", "alice", self.b3())
        self.medicine("en-dclm", "carol", self.b3())
        self.run_dir()
        (self.src / "en-dclm" / "carol" / "medicine.md").unlink()                  # a --pack-id code-rust fetch:
        (self.src / "en-dclm" / "carol").rmdir()                                    # only code-rust is in the folder
        (self.src / "en-dclm").rmdir()
        rc, out, _ = self.run_dir()
        self.assertEqual(rc, 0)
        self.assertFalse(any(e["withdrawn"] for e in self.entries().values()))
        self.assertTrue((self.tmp / "rules" / "en-dclm" / "carol.py").exists())

    def test_the_run_summary_lists_new_withdrawn_reinstated_and_changed_rules(self):
        self.medicine("code-rust", "alice", self.b3())
        self.medicine("code-rust", "bob", self.b3())
        self.medicine("en-dclm", "carol", self.b3())
        _, out, _ = self.run_dir()
        self.assertIn("REGISTRY CHANGES: 3 new", out)
        self.assertIn("new (3)\n  code-rust/alice  DA-B3-001\n  code-rust/bob    DA-B3-002\n  en-dclm/carol    DA-B3-003\n", out)
        _, out, _ = self.run_dir()
        self.assertIn("REGISTRY CHANGES: none", out)
        (self.src / "code-rust" / "bob" / "medicine.md").unlink()                  # bob is gone
        self.medicine("code-rust", "alice", self.b3("t.strip()"))                  # alice's code changes
        _, out, _ = self.run_dir()
        self.assertIn("REGISTRY CHANGES: 1 withdrawn, 1 changed", out)
        self.assertIn("withdrawn (1)\n  code-rust/bob  DA-B3-002", out)
        self.assertIn("changed (1)\n  code-rust/alice  DA-B3-001", out)
        self.medicine("code-rust", "bob", self.b3())                               # bob comes back
        _, out, _ = self.run_dir()
        self.assertIn("REGISTRY CHANGES: 1 reinstated", out)
        self.assertIn("reinstated (1)\n  code-rust/bob  DA-B3-002", out)

    def test_a_registry_from_before_code_sha256_is_filled_in_without_a_change_report(self):
        self.medicine("code-rust", "alice", self.b3())
        self.run_dir()
        data = json.loads(self.registry.read_text())
        for e in data["rules"]:
            del e["code_sha256"]
        self.registry.write_text(json.dumps(data))
        _, out, _ = self.run_dir()
        self.assertIn("REGISTRY CHANGES: none", out)
        self.assertIn("code_sha256", self.registry.read_text())

    def test_the_last_medicine_of_a_pack_withdraws_its_rules(self):
        self.medicine("code-rust", "alice", self.b3())
        self.medicine("en-dclm", "carol", self.b3())
        self.run_dir()
        (self.src / "code-rust" / "alice" / "medicine.md").unlink()                # alice was the only one in code-rust
        (self.src / "code-rust" / "alice").rmdir()                                  # the pack folder stays, empty
        rc, out, _ = self.run_dir()
        self.assertEqual(rc, 0)
        self.assertTrue(self.entries()["DA-B3-001"]["withdrawn"])                  # code-rust / alice
        self.assertFalse((self.tmp / "rules" / "code-rust" / "alice.py").exists())
        self.assertTrue((self.tmp / "rules" / "en-dclm" / "carol.py").exists())
        self.assertIn("medicine.md gone, rules withdrawn: code-rust/alice", out)

    def test_only_empty_pack_folders_and_nothing_to_withdraw_is_still_an_error(self):
        (self.src / "code-rust").mkdir(parents=True)
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(medicine_extract.main(["--dir", str(self.src), "--registry", str(self.registry)]), 1)
        self.assertFalse(self.registry.exists())
        self.assertIn("no <pack_id>/<handle>/medicine.md under", err.getvalue())

    def test_arguments(self):
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(medicine_extract.main([]), 2)                                        # nothing given
            self.assertEqual(medicine_extract.main(["x.md", "--dir", str(self.src)]), 2)          # both given
            self.assertEqual(medicine_extract.main(["--dir", str(self.src), "--out", "r.py"]), 2)  # --out with --dir
            self.assertEqual(medicine_extract.main(["--dir", str(self.tmp / "missing")]), 2)      # not a folder
        self.src.mkdir()
        with contextlib.redirect_stderr(err):
            self.assertEqual(medicine_extract.main(["--dir", str(self.src), "--registry", str(self.registry)]), 1)  # empty folder
        self.assertIn("give one medicine.md, or --dir", err.getvalue())
        self.assertIn("--out is for one medicine.md", err.getvalue())
        self.assertIn("is not a folder", err.getvalue())
        self.assertIn("no <pack_id>/<handle>/medicine.md under", err.getvalue())

    def test_two_runs_of_dir_with_a_new_annotator_keep_old_ids(self):
        self.medicine("code-rust", "bob", self.b3())
        self.run_dir()
        self.assertEqual(self.entries()["DA-B3-001"]["handle"], "bob")
        self.medicine("code-rust", "alice", self.b3())     # sorts before bob, but must not take bob's ID
        self.run_dir()
        self.assertEqual({i: e["handle"] for i, e in self.entries().items()}, {"DA-B3-001": "bob", "DA-B3-002": "alice"})


if __name__ == "__main__":
    unittest.main()
