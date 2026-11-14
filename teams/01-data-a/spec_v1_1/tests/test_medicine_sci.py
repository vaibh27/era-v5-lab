"""Tests for the rules in sheets/medicine_sci.md, loaded the way tools/harm_test.py loads them. All text is made up.

  python3 -m unittest discover -s spec_v1_1/tests      (from teams/01-data-a)
"""
import sys
import unittest
from pathlib import Path

SPEC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SPEC.parent / "tools"))
import harm_test  # noqa: E402  (only its loader is used; the file is not changed)

SHEET = SPEC / "sheets" / "medicine_sci.md"
RULES, _ = harm_test.load_rules(SHEET)
RULES = RULES["RULES_BY_ID"]


def doc(text, pack="sci-finepdfs"):
    return {"doc_id": "0" * 16, "text": text, "url": "", "lang_label": "en", "pack_id": pack}


def run(rule_id, text, **kw):
    return RULES[rule_id](doc(text, **kw))


class Sheet(unittest.TestCase):
    def test_one_rule_for_each_new_science_disease(self):
        self.assertEqual(sorted(RULES), ["DA-A7-001", "DA-B6-001", "DA-C7-003", "DA-C8-003", "DA-C8-004", "DA-C9-002"])

    def test_every_rule_is_off_on_other_packs(self):
        text = "company- owned ____________\n| 332 50.4 |  |\nIf you ally dependence such a referred book"
        for rid in RULES:
            self.assertEqual(run(rid, text, pack="math-finemath"), ("keep", text), rid)
            self.assertEqual(run(rid, text, pack=None), ("keep", text), rid)


class A7(unittest.TestCase):
    R = "DA-A7-001"

    def test_split_words_are_joined_with_the_hyphen_kept(self):
        self.assertEqual(run(self.R, "McDonald's company- owned and performance- based stores"),
                         ("fix", "McDonald's company-owned and performance-based stores"))
        self.assertEqual(run(self.R, "the stereo- scopic glasses"), ("fix", "the stereo-scopic glasses"))

    def test_day_to_day(self):
        self.assertEqual(run(self.R, "the day- to-day work"), ("fix", "the day-to-day work"))

    def test_suspended_hyphens_stay(self):
        for text in ("pre- and post-test scores", "a two- to three-fold rise", "first- or second-order", "low- versus high-dose"):
            self.assertEqual(run(self.R, text), ("keep", text), text)

    def test_other_dashes_stay(self):
        for text in ("x- axis", "- a list item", "a - b", "Section A- Part", "COVID- 19", "pages 10- 12"):
            self.assertEqual(run(self.R, text), ("keep", text), text)

    def test_idempotent(self):
        once = run(self.R, "company- owned, day- to-day")[1]
        self.assertEqual(run(self.R, once), ("keep", once))


class C9(unittest.TestCase):
    R = "DA-C9-002"

    def test_spinner_pages_are_dropped(self):
        pages = "\n".join(f"Page {n}/33" for n in range(1, 6))
        for text in ("If you ally dependence such a referred physics book that will provide you worth",
                     "Right here, we have countless books probability theory and examples",
                     "Access Free Ebook Physics Notes\nSome words.",
                     "Physics Ebook\n" + pages):
            self.assertEqual(run(self.R, text)[0], "drop", text)

    def test_a_report_with_page_markers_is_kept(self):
        text = "Page 1/3 of the report.\nDownload the PDF here.\nPage 2/3\nPage 3/3"
        self.assertEqual(run(self.R, text), ("keep", text))


class B6(unittest.TestCase):
    R = "DA-B6-001"

    def test_a_blank_form_is_dropped(self):
        text = "Name: ____________________ Phone: ____________\nAddress: ________________________"
        self.assertEqual(run(self.R, text)[0], "drop")

    def test_a_few_blanks_in_a_long_text_are_kept(self):
        text = "Fill in: x = _____ .\n" + "The rest of this page explains the method in words. " * 10
        self.assertEqual(run(self.R, text), ("keep", text))

    def test_a_page_under_30_percent_blanks_is_kept(self):
        text = "Q: " + "_" * 25 + " " + "word " * 15  # 25 of 87 non-space characters
        self.assertEqual(run(self.R, text), ("keep", text))


class C7Tables(unittest.TestCase):
    R = "DA-C7-003"

    def test_a_dangling_list_intro_is_cut(self):
        text = ("The Anti Ragging Committee of Samata College is constituted with the following members.\n"
                "The committee will look into the matters regarding complaints received from the students.")
        self.assertEqual(run(self.R, text), ("fix", text.replace(" with the following members.", ".")))

    def test_an_intro_followed_by_its_list_stays(self):
        for text in ("The committee is constituted with the following members:\n1. Dr. A, Chair\n2. Mr. B",
                     "The committee has the following members:\nDr. A. Kumar (Chair)",
                     "Consider the following: a long sentence that goes on to explain the whole idea in detail here."):
            self.assertEqual(run(self.R, text), ("keep", text), text)


class C8Order(unittest.TestCase):
    R = "DA-C8-004"

    def test_steps_out_of_order_are_put_back(self):
        text = "1. Remove the mount.\n2. Position washer.\n3. Attach tube.\n5. Tighten nut.\n4. Attach mount.\nObserve specs."
        want = "1. Remove the mount.\n2. Position washer.\n3. Attach tube.\n4. Attach mount.\n5. Tighten nut.\nObserve specs."
        self.assertEqual(run(self.R, text), ("fix", want))

    def test_other_lists_stay(self):
        for text in ("1. a\n2. b\n3. c", "3. c\n2. b\n1. a", "1. a\n3. b\n2. c\n7. d", "2. b\n1. a\nend"):
            self.assertEqual(run(self.R, text), ("keep", text), text)


class C8Cells(unittest.TestCase):
    R = "DA-C8-003"

    def test_two_numbers_in_one_cell_are_split_into_the_empty_cell(self):
        text = "| Female | 332 50.4 |  | 48.5 |\n| Male | 327 49.6 |  | 51.5 |"
        want = "| Female | 332 | 50.4 | 48.5 |\n| Male | 327 | 49.6 | 51.5 |"
        self.assertEqual(run(self.R, text), ("fix", want))

    def test_full_cells_and_prose_stay(self):
        for text in ("| A | 10 20 | 30 |", "332 50.4 |  |", "| Total | 1,323 |  |"):
            self.assertEqual(run(self.R, text), ("keep", text), text)

    def test_idempotent(self):
        once = run(self.R, "| Female | 332 50.4 |  | 48.5 |")[1]
        self.assertEqual(run(self.R, once), ("keep", once))


if __name__ == "__main__":
    unittest.main()
