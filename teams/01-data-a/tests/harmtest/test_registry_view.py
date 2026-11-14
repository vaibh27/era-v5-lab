"""Tests for registry_view.py. Run: python -I teams/01-data-a/tests/harmtest/test_registry_view.py"""
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harmtest"))  # python -I: not on the path by itself
import medicine_extract  # noqa: E402
import registry_view  # noqa: E402

FENCE = "```"

ALICE = f"""**Disease:** B3, markers
3. **Action:** fix
{FENCE}python
x = 1
{FENCE}
**Disease:** E1, generated
3. **Action:** drop
{FENCE}python
cure_e1 = lambda t: ("drop", "") if "gen" in t else ("keep", t)
{FENCE}
## Appendix
{FENCE}python
import re
import unicodedata
MARKER = re.compile("<m>")
UNUSED = 5

def _strip(text):
    return MARKER.sub("", text)

def cure_b3(text):
    return _strip(text)
{FENCE}
"""

BOB = f"""**Disease:** B3, markers
3. **Action:** fix
{FENCE}python
def cure_b3(text):
    return text.replace("<m>", "")
{FENCE}
"""


class ViewCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.registry = self.tmp / "rules" / "registry.json"
        for pack, handle, text in [("code-rust", "alice", ALICE), ("code-rust", "bob", BOB)]:
            self.add(pack, handle, text)

    def add(self, pack, handle, text):
        d = self.tmp / "submissions" / pack / handle
        d.mkdir(parents=True, exist_ok=True)
        (d / "medicine.md").write_text(text)
        with contextlib.redirect_stdout(io.StringIO()):
            medicine_extract.main([str(d / "medicine.md"), "--registry", str(self.registry)])

    def rule(self, rule_id):
        entry = next(e for e in json.loads(self.registry.read_text())["rules"] if e["id"] == rule_id)
        return registry_view.read_rule(self.registry.parent / entry["file"], rule_id)

    def run_main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = registry_view.main(["--registry", str(self.registry), *argv])
        return rc, out.getvalue(), err.getvalue()

    def test_a_plain_function(self):
        r = self.rule("DA-B3-002")                      # bob's, written as cure_b3
        self.assertEqual(r["function"], "cure_b3")
        self.assertIn('text.replace("<m>", "")', r["source"])
        self.assertEqual(r["helpers"], [])

    def test_helpers_used_by_the_rule_are_shown_and_unused_ones_are_not(self):
        r = self.rule("DA-B3-001")                      # alice's cure_b3 uses _strip, which uses MARKER, which uses re
        self.assertEqual(r["function"], "cure_b3")
        self.assertTrue(r["source"].startswith("def cure_b3(text):"))
        self.assertEqual(r["helpers"], ["import re", 'MARKER = re.compile("<m>")', "def _strip(text):\n    return MARKER.sub(\"\", text)"])
        self.assertNotIn("UNUSED", "".join(r["helpers"]))
        self.assertNotIn("unicodedata", "".join(r["helpers"]))

    def test_a_lambda_assigned_to_a_name(self):
        r = self.rule("DA-E1-001")
        self.assertEqual(r["function"], "cure_e1")
        self.assertTrue(r["source"].startswith("cure_e1 = lambda t:"))

    def test_the_code_is_never_run(self):
        marker = self.tmp / "ran"
        (self.tmp / "rules" / "x.py").write_text(
            f"open({str(marker)!r}, 'w').write('x')\ndef f(t):\n    return t\nRULES_BY_ID = {{'DA-A1-001': f}}\n")
        r = registry_view.read_rule(self.tmp / "rules" / "x.py", "DA-A1-001")
        self.assertEqual(r["function"], "f")
        self.assertFalse(marker.exists())

    def test_problems_are_reported_not_raised(self):
        r = registry_view.read_rule(self.tmp / "nope.py", "DA-B3-001")
        self.assertIn("cannot read nope.py", r["error"])
        (self.tmp / "bad.py").write_text("def (:\n")
        self.assertIn("SyntaxError", registry_view.read_rule(self.tmp / "bad.py", "DA-B3-001")["error"])
        self.assertIn("is not in RULES_BY_ID", self.rule_missing())

    def rule_missing(self):
        entry = json.loads(self.registry.read_text())["rules"][0]
        return registry_view.read_rule(self.registry.parent / entry["file"], "DA-Z9-999")["error"]

    def test_page_is_written_with_every_rule_including_withdrawn(self):
        data = json.loads(self.registry.read_text())
        data["rules"][0]["withdrawn"] = True
        self.registry.write_text(json.dumps(data))
        rc, out, _ = self.run_main()
        self.assertEqual(rc, 0)
        self.assertIn("3 rules (1 withdrawn) from 1 packs", out)
        page = (self.registry.parent / "registry.html").read_text()
        rows = json.loads(page.split('type="application/json">')[1].split("</script>")[0])
        self.assertEqual([r["id"] for r in rows], ["DA-B3-001", "DA-B3-002", "DA-E1-001"])
        self.assertEqual([r["withdrawn"] for r in rows], [True, False, False])
        self.assertEqual({r["team"] for r in rows}, {"DA"})

    def test_code_cannot_break_out_of_the_page(self):
        evil = f"""**Disease:** A1, junk
3. **Action:** fix
{FENCE}python
def cure_a1(text):
    \"\"\"</script><img src=x onerror=alert(1)> &amp; <!-- \"\"\"
    return text
{FENCE}
"""
        self.add("code-rust", "mallory", evil)
        self.run_main()
        page = (self.registry.parent / "registry.html").read_text()
        self.assertNotIn("<img", page)
        self.assertEqual(page.count("</script>"), 2)           # the data block and the page script, nothing more
        rows = json.loads(page.split('type="application/json">')[1].split("</script>")[0])
        mallory = next(r for r in rows if r["handle"] == "mallory")
        self.assertIn("</script><img", mallory["source"])       # the text itself is kept exactly, as data

    def test_an_unreadable_rules_file_still_gives_a_page(self):
        (self.registry.parent / "code-rust" / "bob.py").unlink()
        rc, out, err = self.run_main()
        self.assertEqual(rc, 0)
        self.assertIn("source not readable for: DA-B3-002", err)
        self.assertTrue((self.registry.parent / "registry.html").exists())

    def test_show(self):
        rc, out, _ = self.run_main("--show", "da-b3-001")
        self.assertEqual(rc, 0)
        self.assertIn("DA-B3-001  code B3  code-rust/alice  action fix", out)
        self.assertIn("def cure_b3(text):", out)
        self.assertIn("# used by the rule", out)
        rc, _, err = self.run_main("--show", "DA-B3-999")
        self.assertEqual(rc, 2)
        self.assertIn("is not in", err)

    def test_missing_registry(self):
        with self.assertRaises(SystemExit) as cm:
            registry_view.load(self.tmp / "missing.json")
        self.assertIn("cannot read the registry", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
