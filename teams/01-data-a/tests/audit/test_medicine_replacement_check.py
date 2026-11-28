"""Tests for audit/medicine_replacement_check.py: finding the cards in a report, finding the replaced rule in the sheet, and swapping it.

    python3 -m pytest teams/01-data-a/tests/audit/test_medicine_replacement_check.py -q      (from the repository root)

The cards and sheets in the first tests are made up; the last two use the real sheet and the code-go report.
"""
import difflib
import importlib.util
import re
from pathlib import Path

import pytest

TEAM = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("medicine_replacement_check", TEAM / "audit" / "medicine_replacement_check.py")
mrc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mrc)

FENCE = "```"


def block(*lines):
    return FENCE + "python\n" + "\n".join(lines) + "\n" + FENCE + "\n"


OLD_A = block('RULE_ID = "DA-X-001"', 'STATUS = "CANDIDATE"', 'ORIGIN = "pack-a, pack-b"', "def medicine(doc):", '    return ("keep", doc["text"])')
OLD_B = block('RULE_ID = "DA-X-002"', 'STATUS = "ACTIVE"', 'SPEC_PACKS = {"pack-a"}', "def medicine(doc):", '    return ("keep", doc["text"])')
SHEET = f"# sheet\n\n## Medicine 1\n\ntext\n\n{OLD_A}\n## Medicine 2\n\ntext\n\n{OLD_B}\n## Medicine 3\n\nend\n"

NEW_A = block('REPLACES = "DA-X-001"', 'RULE_ID = "DA-X-101"', 'STATUS = "PROPOSAL"', 'ORIGIN = "pack-a, pack-b"', 'LANGS = {"go"}',
              "def medicine(doc):", '    return ("fix", doc["text"].strip())')
NEW_B = block('SCOPE = {"DA-X-002": frozenset({"pack-a"})}', 'REPLACES = "DA-X-002"', 'RULE_ID = "DA-X-102"', 'STATUS = "PROPOSAL"',
              'SPEC_PACKS = SCOPE["DA-X-002"]', "def medicine(doc):", '    return ("fix", doc["text"].strip())')
OTHER = block("x = 1")  # a python block that is not a card
REPORT = f"# report\n\n{OTHER}\n### Medicine 1\n\n{NEW_A}\n### Medicine 2\n\n{NEW_B}\n"


@pytest.fixture
def cards(tmp_path):
    p = tmp_path / "report.md"
    p.write_text(REPORT, encoding="utf-8")
    return mrc.report_cards(p)


def test_the_cards_are_found_and_the_other_blocks_are_not(cards):
    assert [(c.new_id, c.old_id) for c in cards] == [("DA-X-101", "DA-X-001"), ("DA-X-102", "DA-X-002")]


def test_the_scope_comes_from_origin_or_scope(cards):
    assert cards[0].packs == ["pack-a", "pack-b"]
    assert cards[1].packs == ["pack-a"]
    assert cards[0].lang == "go" and cards[1].lang == "en"


def test_a_report_with_no_card_stops(tmp_path):
    p = tmp_path / "r.md"
    p.write_text(f"# r\n\n{OTHER}", encoding="utf-8")
    with pytest.raises(SystemExit):
        mrc.report_cards(p)


def test_the_replaced_rule_is_found_once():
    assert mrc.sheet_block(SHEET, "DA-X-001").group(1).startswith('RULE_ID = "DA-X-001"')
    with pytest.raises(SystemExit):
        mrc.sheet_block(SHEET, "DA-X-999")
    with pytest.raises(SystemExit):
        mrc.sheet_block(SHEET + OLD_A, "DA-X-001")  # two blocks with the same RULE_ID


def test_in_place_keeps_the_old_id_and_status_and_drops_replaces(cards):
    old_block = mrc.sheet_block(SHEET, "DA-X-001").group(1)
    code = mrc.in_place(cards[0], old_block)
    assert 'RULE_ID = "DA-X-001"' in code and "DA-X-101" not in code and "REPLACES" not in code
    assert 'STATUS = "CANDIDATE"' in code and "PROPOSAL" not in code


def test_a_computed_spec_packs_is_written_as_a_literal(cards):
    code = mrc.in_place(cards[1], mrc.sheet_block(SHEET, "DA-X-002").group(1))
    assert "SPEC_PACKS = {'pack-a'}" in code and 'SPEC_PACKS = SCOPE[' not in code
    assert 'STATUS = "ACTIVE"' in code


def test_the_swap_changes_only_the_two_blocks(cards):
    after = mrc.swapped_sheet(SHEET, cards)
    diff = list(difflib.unified_diff(SHEET.splitlines(), after.splitlines(), lineterm="", n=0))
    changed = [l for l in diff if l[:1] in "+-" and l[:3] not in ("+++", "---")]
    assert changed and all(not l[1:].startswith("#") and "Medicine" not in l for l in changed)
    headings = lambda text: [l for l in text.splitlines() if l.startswith("#")]  # noqa: E731
    assert headings(after) == headings(SHEET)  # only the code inside the cards changed
    assert after.startswith("# sheet") and after.rstrip().endswith("end")
    assert mrc.swapped_sheet(after, cards) == after  # swapping again changes nothing


def test_the_real_report_and_the_real_sheet():
    report = TEAM / "audit" / "code-go_audit_smitasasindran.md"
    if not report.exists():
        pytest.skip("the code-go report is not in this tree")
    cards = mrc.report_cards(report)
    assert [(c.new_id, c.old_id) for c in cards] == [("DA-A6-004", "DA-A6-001"), ("DA-E4-011", "DA-E4-005")]
    assert cards[1].packs == ["code-go"] and "code-python" in cards[0].packs
    sheet = (TEAM / "spec_v1_1" / "sheets" / "specv1_1.md").read_text(encoding="utf-8")
    after = mrc.swapped_sheet(sheet, cards)
    diff = "".join(difflib.unified_diff(sheet.splitlines(True), after.splitlines(True), n=3))
    assert len(re.findall(r"^@@", diff, re.M)) == 3  # all inside the two cards
    assert after.count('RULE_ID = "DA-A6-001"') == sheet.count('RULE_ID = "DA-A6-001"')  # the ID is kept, not duplicated
