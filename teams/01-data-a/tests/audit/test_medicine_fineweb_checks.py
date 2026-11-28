"""Tests for the four medicines of the en-fineweb-edu audit, on made-up texts only.

python -m pytest teams/01-data-a/tests/audit/test_medicine_fineweb_checks.py -q
"""
import importlib.util
from pathlib import Path

import pytest

AUDIT = Path(__file__).resolve().parents[2] / "audit"
_spec = importlib.util.spec_from_file_location("medicine_fineweb_checks", AUDIT / "medicine_fineweb_checks.py")
checks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(checks)
M = checks.load_medicines()
OLD = checks.old_rules()

PROSE = " ".join(f"Sentence number {i} is a plain sentence about water." for i in range(60))


def doc(text, pack="en-fineweb-edu", lang="en"):
    return {"doc_id": "t", "text": text, "url": "", "lang_label": lang, "pack_id": pack}


# ---- DA-C1-015 -------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("junk", ["Share this page on", "Share this page", "  share this page via:  ",
                                  "Get the latest version of the Adobe Acrobat reader or Acrobat Reader for Windows",
                                  "- freely available", "|This section needs additional citations for verification. (May 2014)"])
def test_c1_removes_junk_lines(junk):
    a, t = M["DA-C1-015"](doc(f"Intro line.\n{junk}\nBody line."))
    assert (a, t) == ("fix", "Intro line.\nBody line.")


def test_c1_removes_junk_last_line():
    assert M["DA-C1-015"](doc("Body line.\nShare this page on")) == ("fix", "Body line.\n")


def test_c1_keeps_phrase_inside_a_sentence():
    text = "You can share this page on any site you like.\nIt is freely available to all readers."
    assert M["DA-C1-015"](doc(text)) == ("keep", text)


def test_c1_essay_ad_only_as_three_lines():
    ad = "Before.\nGet your grade\nor your money back\nusing our Essay Writing Service!\nAfter."
    assert M["DA-C1-015"](doc(ad)) == ("fix", "Before.\nAfter.")
    one = "Before.\nGet your grade\nAfter."
    assert M["DA-C1-015"](doc(one)) == ("keep", one)


def test_c1_related_titles_before_trailing_heading():
    text = "Real text ends here.\nFirst Title Second Title Third M\nYou May Also Like"
    assert M["DA-C1-015"](doc(text)) == ("fix", "Real text ends here.")
    keep_sentence = "Real text ends here.\nYou May Also Like"
    assert M["DA-C1-015"](doc(keep_sentence)) == ("fix", "Real text ends here.")


def test_c1_scope():
    text = "Body.\nShare this page on"
    assert M["DA-C1-015"](doc(text, pack="en-dclm")) == ("keep", text)
    assert M["DA-C1-015"](doc(text, lang="hi")) == ("keep", text)


def test_c1_idempotent():
    a, t = M["DA-C1-015"](doc("A.\nShare this page\n- freely available\nB.\nTitles M\nYou May Also Like"))
    assert M["DA-C1-015"](doc(t))[0] == "keep"


# ---- DA-C3-009 -------------------------------------------------------------------------------------------------
def test_c3_curly_quote_ends_a_text():
    text = "He said: “They are some of the scariest things I’ve worked on.”"
    assert OLD["DA-C3-009"](doc(text))[0] == "drop"
    assert M["DA-C3-009"](doc(text)) == ("keep", text)


def test_c3_trims_long_prose_tail():
    text = PROSE + "\nThe last line ends well. A"
    a, t = M["DA-C3-009"](doc(text))
    assert a == "fix" and t == PROSE + "\nThe last line ends well."


def test_c3_removes_last_line_without_sentence_end():
    text = PROSE + "\nShare this page on"
    a, t = M["DA-C3-009"](doc(text))
    assert a == "fix" and t == PROSE


def test_c3_short_fragment_still_dropped():
    assert M["DA-C3-009"](doc("A short note that stops at the"))[0] == "drop"


def test_c3_marker_still_dropped():
    assert M["DA-C3-009"](doc(PROSE + "\nA teaser that goes on […]"))[0] == "drop"


def test_c3_long_list_still_dropped():
    lst = "\n".join(f"- Link title number {i}" for i in range(200)) + "\n- Things people search for"
    assert len(lst) > 2000
    assert M["DA-C3-009"](doc(lst))[0] == "drop"


def test_c3_same_as_old_on_complete_text():
    text = PROSE
    assert M["DA-C3-009"](doc(text)) == OLD["DA-C3-009"](doc(text)) == ("keep", text)


def test_c3_idempotent():
    a, t = M["DA-C3-009"](doc(PROSE + "\nEnds well. And then the"))
    assert a == "fix" and M["DA-C3-009"](doc(t))[0] == "keep"


# ---- DA-C5-008 -------------------------------------------------------------------------------------------------
def test_c5_numeric_table_line_kept():
    text = "Query output:\n11|22.1|2013-10-08 12|2.1|2013-10-09 13|-22.1|2013-10-10\nDone."
    assert OLD["DA-C5-008"](doc(text))[0] == "drop"
    assert M["DA-C5-008"](doc(text)) == ("keep", text)


def test_c5_keyword_line_still_dropped():
    text = "Intro.\nshoes|cheap shoes|red shoes|buy shoes|shoe sale|best shoes|shoes online\nEnd."
    assert M["DA-C5-008"](doc(text))[0] == OLD["DA-C5-008"](doc(text))[0] == "drop"


# ---- DA-F1-010 -------------------------------------------------------------------------------------------------
def test_f1_doi_not_masked():
    text = "Physiol. Plant. 102: 1-10. doi:10.1034/j.1399-3054.1998.1020101.x"
    assert "<PHONE>" in OLD["DA-F1-010"](doc(text))[1]
    assert M["DA-F1-010"](doc(text)) == ("keep", text)


def test_f1_years_not_masked():
    text = "Seasons played: 1933 1934 1935"
    assert "<PHONE>" in OLD["DA-F1-010"](doc(text))[1]
    assert M["DA-F1-010"](doc(text)) == ("keep", text)


def test_f1_real_phone_still_masked():
    text = "Call me on 555 123 4567 after six."
    assert M["DA-F1-010"](doc(text)) == OLD["DA-F1-010"](doc(text))
    assert "<PHONE>" in M["DA-F1-010"](doc(text))[1]


def test_f1_scope():
    text = "Seasons played: 1933 1934 1935"
    assert M["DA-F1-010"](doc(text, pack="code-python")) == ("keep", text)
