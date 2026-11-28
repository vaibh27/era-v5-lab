"""collated_medicine.md: the sheet is free of clashes, and the two merged cards do what both sources did. All text is made up."""
import importlib.util
import json
from collections import Counter
from pathlib import Path

import pytest

DATA_A = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("collated_medicine_check", DATA_A / "audit" / "collated_medicine_check.py")
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)
check.load()
CARDS = {card["RULE_ID"]: card for card in check.cards}
SYNTH = "en-nemotroncc-synth"


def out(rid, text, pack=SYNTH):
    return CARDS[rid]["medicine"]({"doc_id": "x", "text": text, "lang_label": "en", "pack_id": pack})


def test_ids_are_unique_and_not_in_the_registry():
    assert len(CARDS) == len(check.cards) == 15
    registry = json.loads((DATA_A / "rules" / "registry.json").read_text(encoding="utf-8"))["rules"]
    assert not CARDS.keys() & ({r["id"] for r in registry} | set(check.replay.c.SCOPE))


def test_no_two_cards_replace_the_same_rule_on_a_pack():
    pairs = Counter((pack, card["REPLACES"]) for card in check.cards if card.get("REPLACES") for pack in check.packs_of(card))
    assert max(pairs.values()) == 1


@pytest.mark.parametrize("pack", check.PACKS)
def test_every_card_gets_its_place(pack):
    rules, _ = check.with_cards(pack)
    ids = [rid for rid, _ in rules]
    mine = [card for card in check.cards if pack in check.packs_of(card)]
    assert len(set(ids)) == len(ids) and {card["RULE_ID"] for card in mine} <= set(ids)
    assert not {card.get("REPLACES") for card in mine} & set(ids)
    assert "f1_rule.py" not in ids or ids[-1] == "f1_rule.py"


FAR = "\n".join(["The weather was mild and the path was dry."] * 14)  # more than the 6-line window
ORG = "The museum"
PHONE = "555-010-0199"


def test_f1_masked_once_is_masked_on_every_line():  # from #75
    action, text = out("DA-F1-009", f"Ask for Sam on {PHONE}.\n{FAR}\n{ORG} lists {PHONE} too.")
    assert action == "fix" and text.count("<PHONE>") == 2 and PHONE not in text


def test_f1_privacy_first_over_public_evidence_on_another_line():
    text = f"Ask for Sam on {PHONE}.\n{FAR}\nInquiries at the youth bureau: {PHONE}."
    assert out("DA-F1-009", text)[1].count("<PHONE>") == 2


def test_f1_public_evidence_on_its_only_line_keeps_it():  # from #76
    text = f"Inquiries at the youth bureau: {PHONE}."
    assert out("DA-F1-009", text) == ("keep", text)


def test_f1_organisation_number_is_kept():
    text = f"{ORG} lists {PHONE}."
    assert out("DA-F1-009", text) == ("keep", text)


@pytest.mark.parametrize("text", [
    "The binary value is 0101 0101 0101 for this row.",
    "Use the subnet mask 255.255.255.0 here.",
    "Sand it with 80-120-180 grit paper, sanding 80-120-180 in order.",
    "Freephone 0800 555 0199 for help.",  # from #75
])
def test_f1_technical_values_and_freephone_are_kept(text):
    assert out("DA-F1-009", text) == ("keep", text)


def test_c2_keeps_code_lines_and_record_fields_and_cuts_furniture():
    text = "\n".join(["cd /etc/example", "Cost: 50 points", "Read more stories", "x"] * 3)  # #75's, #76's, neither's
    action, new = out("DA-C2-010", text)
    assert action == "fix"
    assert (new.count("cd /etc/example"), new.count("Cost: 50 points"), new.count("Read more stories")) == (3, 3, 1)


def test_c2_and_f1_cards_stay_on_their_packs():
    text = "\n".join(["Read more stories", "x"] * 3)
    for rid, pack in (("DA-C2-010", "en-dclm"), ("DA-F1-009", "en-dclm"), ("DA-C2-009", SYNTH), ("DA-F1-007", SYNTH)):
        assert out(rid, text + f"\nAsk for Sam on {PHONE}.", pack)[0] == "keep"


def test_c1_licence_line_only_at_the_start():
    line = "Distributed under a Creative Commons Attribution 4.0 International License"
    assert out("DA-C1-014", line + "\nBody.", "sci-finepdfs") == ("fix", "Body.")
    assert out("DA-C1-014", "Body.\n" + line + "\nMore.", "sci-finepdfs")[0] == "keep"
