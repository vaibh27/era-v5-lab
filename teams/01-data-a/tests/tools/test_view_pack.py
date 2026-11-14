"""Regression tests for the Indic/Unicode pack viewer (standard-library unittest).

Run from the repository root:
    python -I teams/01-data-a/tests/tools/test_view_pack.py

No Parquet fixture or pyarrow installation is required: CLI tests mock load_pack.
"""

import contextlib
import io
import sys
import unittest
import unicodedata
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))
import view_pack as vp


def sample_row(doc_id: str, text: str) -> dict:
    return {
        "doc_id": doc_id,
        "source": "sample",
        "lang_label": "hi",
        "text": text,
    }


class UnicodeViewerTests(unittest.TestCase):
    def test_raw_text_does_not_replace_zwj_zwnj_or_bom(self):
        text = "क्\u200dष क्\u200cष \ufeff"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            vp.print_document(sample_row("raw", text), 0, 1)
        self.assertIn(text, output.getvalue())
        self.assertNotIn("⟦ZWJ⟧", output.getvalue())
        self.assertNotIn("⟦ZWNJ⟧", output.getvalue())
        self.assertNotIn("⟦BOM⟧", output.getvalue())
        self.assertEqual(text, sample_row("raw", text)["text"])
        self.assertEqual(vp.hidden_characters(text)[:2], [(2, "\u200d"), (7, "\u200c")])

    def test_unicode_range_shows_every_character_and_exact_zero_based_offsets(self):
        text = "कs् १0७"
        report = vp.unicode_range(text, 0, len(text))
        self.assertEqual(len(report.splitlines()), len(text) + 1)
        for i, char in enumerate(text):
            self.assertIn(f"{i:6d}", report)
            self.assertIn(f"U+{ord(char):04X}", report)
        self.assertIn("LATIN SMALL LETTER S", report)
        self.assertIn("DIGIT ZERO", report)
        self.assertIn("DEVANAGARI SIGN VIRAMA", report)

    def test_unicode_range_clamps_bounds_and_reports_empty_range(self):
        self.assertIn("Unicode range 0:2", vp.unicode_range("ab", -5, 999))
        self.assertIn("empty range", vp.unicode_range("ab", 5, 7))

    def test_carriage_return_is_named_not_silently_rendered(self):
        report = vp.unicode_range("a\rb", 0, 3)
        self.assertIn("1  '\\r'", report)
        self.assertIn("U+000D", report)
        self.assertIn("CARRIAGE RETURN", report)
        self.assertIn("1 U+000D CARRIAGE RETURN", vp.audit_report("a\rb"))

    def test_audit_has_no_two_thousand_character_cutoff(self):
        text = "a" * 2500 + "\u200b"
        report = vp.audit_report(text)
        self.assertIn("2500 U+200B ZERO WIDTH SPACE", report)

    def test_mixed_ascii_letter_inside_devanagari_sequence_is_visible(self):
        text = "कs्"
        self.assertEqual(vp.g1_candidates(text), [(2, "्", "s")])
        report = vp.unicode_range(text, 0, len(text))
        self.assertRegex(report, r"U\+0073\s+LATIN SMALL LETTER S")
        self.assertIn("G1 candidates", vp.audit_report(text))

    def test_mixed_ascii_and_devanagari_digits_are_reported(self):
        self.assertEqual(vp.mixed_digit_candidates("पाठ १0७ आणि 123"), [(4, "१0७")])
        report = vp.audit_report("१0७")
        self.assertIn("Mixed Devanagari/ASCII digit candidates", report)
        self.assertIn("U+0967 + U+0030 + U+096D", report)
        self.assertEqual(vp.mixed_digit_candidates("१२३ 456"), [])

    def test_nfc_precomposed_nukta_letters_are_all_candidates(self):
        for cp in range(0x0958, 0x0960):
            with self.subTest(cp=f"U+{cp:04X}"):
                char = chr(cp)
                normalized = unicodedata.normalize("NFC", char)
                self.assertEqual(unicodedata.combining(char), 0)
                self.assertNotEqual(char, normalized)
                self.assertEqual(vp.nfc_candidates(char), [(0, char, normalized)])
                report = vp.audit_report(char)
                self.assertIn("A5/NFC candidates", report)
                self.assertIn(f"U+{cp:04X}", report)
                self.assertNotIn("(no audit candidates found)", report)

    def test_nfc_reports_correct_offset_after_prefix(self):
        self.assertEqual(vp.nfc_candidates("prefix \u0958"), [(7, "\u0958", "\u0915\u093c")])

    def test_nfc_combining_cluster_is_reported_once(self):
        self.assertEqual(vp.nfc_candidates("x e\u0301!"), [(2, "e\u0301", "é")])
        self.assertEqual(vp.nfc_candidates("कखग"), [])

    def test_nfc_two_part_vowel_signs(self):
        pairs = (
            ("\u09c7\u09be", "\u09cb"),  # Bengali
            ("\u0b47\u0b3e", "\u0b4b"),  # Odia
            ("\u0bc6\u0bbe", "\u0bca"),  # Tamil
            ("\u0d46\u0d3e", "\u0d4a"),  # Malayalam
        )
        for source, normalized in pairs:
            with self.subTest(source=source):
                self.assertEqual([unicodedata.combining(ch) for ch in source], [0, 0])
                self.assertEqual(vp.nfc_candidates(source), [(0, source, normalized)])
                self.assertEqual(vp.nfc_candidates("x" + source), [(1, source, normalized)])
                self.assertEqual(vp.nfc_candidates(normalized), [])
        report = vp.audit_report("x \u09c7\u09be")
        self.assertIn("A5/NFC candidates:", report)
        self.assertIn("2 U+09C7", report)
        self.assertIn("U+09BE", report)
        self.assertIn("U+09CB", report)

    def test_g1_does_not_treat_avagraha_as_dependent_mark(self):
        self.assertNotIn(0x093D, vp.DEVANAGARI_DEPENDENT_MARKS)
        self.assertEqual(vp.g1_candidates("a\u093d"), [])
        self.assertNotIn("G1 candidates", vp.audit_report("a\u093d"))

    def test_g1_includes_all_four_devanagari_initial_marks(self):
        for cp in range(0x0900, 0x0904):
            with self.subTest(cp=f"U+{cp:04X}"):
                self.assertIn(cp, vp.DEVANAGARI_DEPENDENT_MARKS)
                self.assertEqual(vp.g1_candidates("x" + chr(cp)), [(1, chr(cp), "x")])

    def test_g1_flags_detached_matra_but_not_attached_matra(self):
        self.assertEqual(vp.g1_candidates("क ि"), [(2, "ि", " ")])
        self.assertEqual(vp.g1_candidates("कि"), [])
        self.assertEqual(vp.g1_candidates("\u093f"), [(0, "ि", None)])

    def test_joiners_do_not_create_false_g1_candidate(self):
        self.assertEqual(vp.g1_candidates("क्\u200dि क्\u200cि"), [])

    def test_no_candidates_message_for_clean_text(self):
        self.assertEqual(vp.audit_report("नमस्ते world 123"), "(no audit candidates found)")


class CliTests(unittest.TestCase):
    def run_cli(self, argv, rows, inputs):
        output = io.StringIO()
        with mock.patch.object(vp, "load_pack", return_value=rows), \
             mock.patch("builtins.input", side_effect=inputs), \
             contextlib.redirect_stdout(output):
            vp.main(argv)
        return output.getvalue()

    def test_row_option_is_one_based_and_start_is_zero_based(self):
        rows = [sample_row("first", "one"), sample_row("second", "two")]
        self.assertIn("[2/2] doc_id=second", self.run_cli(["pack.parquet", "--row", "2"], rows, ["q"]))
        self.assertIn("[2/2] doc_id=second", self.run_cli(["pack.parquet", "--start", "1"], rows, ["q"]))

    def test_legacy_hex_and_new_audit_flag_both_work(self):
        rows = [sample_row("hidden", "x\u200d")]
        for flag in ("--hex", "--audit"):
            with self.subTest(flag=flag):
                output = self.run_cli(["pack.parquet", flag], rows, ["q"])
                self.assertIn("Hidden/control characters:", output)

    def test_back_on_first_row_does_not_exit(self):
        rows = [sample_row("first", "one")]
        output = self.run_cli(["pack.parquet"], rows, ["b", "q"])
        self.assertEqual(output.count("[1/1] doc_id=first"), 2)

    def test_audit_toggle_and_unicode_command_preserve_current_row(self):
        rows = [sample_row("first", "कs्")]
        output = self.run_cli(["pack.parquet"], rows, ["h", "u 0 3", "q"])
        self.assertEqual(output.count("[1/1] doc_id=first"), 3)
        self.assertIn("G1 candidates", output)
        self.assertRegex(output, r"U\+0073\s+LATIN SMALL LETTER S")

    def test_unicode_cli_option_shows_ascii_at_requested_offsets(self):
        rows = [sample_row("first", "कs्")]
        output = self.run_cli(["pack.parquet", "--unicode", "1", "3"], rows, ["q"])
        self.assertIn("Unicode range 1:3", output)
        self.assertRegex(output, r"U\+0073\s+LATIN SMALL LETTER S")
        self.assertRegex(output, r"U\+094D\s+DEVANAGARI SIGN VIRAMA")

    def test_eof_on_piped_stdin_exits_cleanly(self):
        rows = [sample_row("first", "one")]
        output = self.run_cli(["pack.parquet"], rows, [EOFError()])
        self.assertIn("[1/1] doc_id=first", output)

    def test_rejects_invalid_row_and_start(self):
        rows = [sample_row("first", "one")]
        for argv, expected in ((["pack.parquet", "--row", "0"], "--row"),
                               (["pack.parquet", "--start", "2"], "--start")):
            with self.subTest(argv=argv):
                with mock.patch.object(vp, "load_pack", return_value=rows):
                    with self.assertRaisesRegex(SystemExit, expected):
                        vp.main(argv)


if __name__ == "__main__":
    unittest.main()
