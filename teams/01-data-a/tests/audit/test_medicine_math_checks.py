"""Tests for the two medicines of audit/math-audit-Aneesha_Das.md (DA-C1-012, DA-C2-007) and for the frozen copies in
audit/medicine_math_checks.py. Every text here is made up; none comes from a pack.

    python -m pytest teams/01-data-a/tests/audit -q
"""
import importlib.util
import random
from pathlib import Path

import pytest

TEAM = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("medicine_math_checks", TEAM / "audit" / "medicine_math_checks.py")
mc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mc)

PASSAGE = ("The solution starts from the definition of the limit and bounds the difference by a geometric series, "
           "which shows that the sequence is Cauchy and therefore converges in the reals.")
RULES = {"DA-C1-012": mc.C1["medicine"], "DA-C2-007": mc.C2["medicine"]}


def run(rule_id, text, pack="math-megamath"):
    return RULES[rule_id]({"doc_id": "0" * 16, "text": text, "url": "", "lang_label": "en", "pack_id": pack})


# ---------------------------------------------------------------- scope

@pytest.mark.parametrize("rule_id", sorted(RULES))
@pytest.mark.parametrize("pack", ["en-dclm", "code-go", None])
def test_every_rule_is_off_outside_the_math_packs(rule_id, pack):
    text = "Gold Member\nx\nGold Member\n" + "\n".join(["Please log in to see more"] * 5)
    assert run(rule_id, text, pack) == ("keep", text)


@pytest.mark.parametrize("pack", ["math-finemath", "math-megamath"])
def test_both_math_packs(pack):
    assert run("DA-C1-012", "Gold Member\nx", pack) == ("fix", "x")


# ---------------------------------------------------------------- DA-C1-012

def test_c1_every_copy_of_a_chrome_line_goes():
    assert run("DA-C1-012", "Q1\nHomework Helper\nanswer\n  Homework Helper \nend") == ("fix", "Q1\nanswer\nend")


def test_c1_chrome_inside_a_real_line_stays():
    text = "He was a Homework Helper for years."
    assert run("DA-C1-012", text) == ("keep", text)


def test_c1_inline_labels():
    assert run("DA-C1-012", "## Proof [edit]\nLet x {[ promptMessage ]}be") == ("fix", "## Proof\nLet x be")


def test_c1_a_label_then_its_line():  # "Published [edit]" is chrome once the label is cut
    assert run("DA-C1-012", "Published [edit]\nx") == ("fix", "x")


def test_c1_a_journal_label_with_its_date_stays():
    assert run("DA-C1-012", "Received: 12 March 2019") == ("keep", "Received: 12 March 2019")


# ---------------------------------------------------------------- DA-C2-007

def test_c2_formula_line_keeps_every_copy():
    text = "\n".join(["P(E) = n(E)/ n(S)", "step"] * 5)
    assert run("DA-C2-007", text) == ("keep", text)


def test_c2_prose_line_keeps_its_first_copy():
    text = "\n".join(["Please log in to see more", "x"] * 5)
    assert run("DA-C2-007", text) == ("fix", "Please log in to see more\n" + "\n".join(["x"] * 5))


def test_c2_speaker_turn_keeps_every_copy():
    text = "\n".join(["MALE SPEAKER: [INAUDIBLE]", "PROFESSOR: Yes, that's right."] * 22)
    assert run("DA-C2-007", text) == ("keep", text)


def test_c2_caption_seen_twice():
    text = "Figure 4.4 Small-scale eddies\nbody\nFigure 4.4 Small-scale eddies\nend"
    assert run("DA-C2-007", text) == ("fix", "Figure 4.4 Small-scale eddies\nbody\nend")


def test_c2_repeated_passage():
    assert run("DA-C2-007", f"{PASSAGE}\n\nshort\n\n{PASSAGE}\n\nend") == ("fix", f"{PASSAGE}\n\nshort\n\nend")


def test_c2_repeated_option_list_stays():
    opts = "Which is right?\nA. one two three four five six seven eight nine ten\nB. " + "x" * 60 + "\nC. " + "y" * 60
    text = f"{opts}\n\n{opts}"
    assert run("DA-C2-007", text) == ("keep", text)


def test_c2_code_line_stays():
    text = "\n".join(["    total = total + value;", "x"] * 6)
    assert run("DA-C2-007", text) == ("keep", text)


# ---------------------------------------------------------------- idempotence (spec v1.1 defers rules that are not)

@pytest.mark.parametrize("rule_id", sorted(RULES))
@pytest.mark.parametrize("text", [
    "Published [edit]\nx",
    "## Proof [edit] [edit]\nx",
    "Received{[ promptMessage ]}\nx",
    "Gold Member\nGold Member\nx\nGold Member",
    f"{PASSAGE}\n\n{PASSAGE}\n\n{PASSAGE}",
    "\n".join(["Please log in to see more", "Figure 1 A", "x"] * 6),
])
def test_a_second_run_changes_nothing(rule_id, text):
    _, once = run(rule_id, text)
    assert run(rule_id, once) == ("keep", once)


# ---------------------------------------------------------------- the cards against the frozen copies

EDGES = [
    "",
    "one line only",
    "\n".join(["Please log in to see more"] * 6),                      # first line and last line are copies
    "a\r\nPlease log in to see more\r\n" * 6,                          # CRLF: \r is stripped, not split
    " Please log in to see more \nx\n" * 5,                   # no-break spaces round the line
    "x\nPlease log in to see more" * 5,                                 # the last copy has no line break
    f"{PASSAGE}\n\n{PASSAGE}\n \n{PASSAGE}",                           # adjacent copies, odd separator
    f"{PASSAGE}\n\n- {PASSAGE}\n\n- {PASSAGE}",                        # bullet copies stay
    "Table 2 Results\nTable 2 Results\nTable 2 Results",
    "\n".join(["Homework Helper", "Gold Member", "MALE SPEAKER: hi"] * 7),
    "## A [edit]\nno label here\n## B [edit] [edit]",                  # the [edit] gate, on and off
]
PIECES = [
    "Homework Helper", "Gold Member", "2022 Award", "Published", "Received: 12 March 2019", "## Proof [edit]",
    "Published [edit]", "Let x {[ promptMessage ]}be", "Please log in to see more", "P(E) = n(E)/ n(S)",
    "MALE SPEAKER: [INAUDIBLE]", "Figure 4.4 Small-scale eddies", "Table 2 Results", "    total = total + value;",
    "A. one two three four five six seven", "- a bullet item that is long enough", "", " ", " ", "x", PASSAGE,
]
N_RANDOM = 4000


def random_text(rng):
    sep = rng.choice(["\n", "\n", "\n\n", "\r\n", "\n \n"])
    return sep.join(rng.choice(PIECES) for _ in range(rng.randint(0, 40))) + rng.choice(["", "\n", " "])


def same_as_frozen(text):
    assert run("DA-C1-012", text)[1] == mc.old_c1(text), repr(text[:60])
    assert run("DA-C2-007", text)[1] == mc.old_c2(text), repr(text[:60])


@pytest.mark.parametrize("text", EDGES)
def test_same_as_frozen_on_edge_cases(text):
    same_as_frozen(text)


def test_same_as_frozen_on_random_texts():
    rng = random.Random(0)
    for _ in range(N_RANDOM):
        same_as_frozen(random_text(rng))


def test_pack_of():
    assert mc.pack_of(Path("Cap1-packs/math-finemath/Part_1/part-00000.parquet")) == "math-finemath"
    assert mc.pack_of(Path("packs/math-megamath.parquet")) == "math-megamath"
