"""Equivalence tests for the speed-ups of clean.py (results/spec_v1/speed_v1.md). Each rule that was made faster is checked
against a frozen copy of the rule as it was before the change: the same text goes through both and the results must be
equal, on hand-written cases at the edges of the fast path and on random documents. Every text here is made up; none comes
from a pack.

  python -m pytest tests/spec_v1/test_speed_equivalence.py

  DA-F1-005  PII: the organisation word is searched line by line instead of in a joined 13-line window
  DA-D3-003  foreign-script lines: ASCII lines skipped, each letter's script looked up once
  DA-C5-002  keyword spam: the nine patterns are searched as one union first
  DA-B4-010  template and code-like text: the line scan stops once the answer is certain
  DA-B4-008  detokenised quotes: only the apostrophes are examined
  DA-B2-002  HTML tags: the cheap tests (count of "<", tag search) run before the slow "is it about HTML" search
  DA-C3-003  cut-off pages: the last word is read backwards from the end instead of searched for from the start

The frozen copies (the f1_joined_window, d3_reference, c5_result, b4_reference, b4_008_reference, b2_reference, c3_reference functions) record what each
rule did before. If a rule's behaviour is ever changed on purpose, update or delete its copy here. The rest of the rule's
own tests are in test_clean.py. One test file per speed-up round keeps test_clean.py as it was.
"""
import random
import re
import sys
import textwrap
import unicodedata
from functools import cache
from pathlib import Path

import pytest

SPEC = Path(__file__).resolve().parents[2] / "spec_v1"
sys.path.insert(0, str(SPEC))
import clean as c

AT = "@"  # the email scans read added lines, so made-up values are joined at run time
SHEET = (SPEC / "sheets" / "specv1.md").read_text(encoding="utf-8")
EN_TEXT = "The river rises in the hills and runs south to the sea.\nFarmers along its banks grow rice and wheat.\n"
PARA = ("The river rises in the hills and runs south to the sea. "
        "Farmers along its banks grow rice and wheat in the dry months.")
OTHER_LANGS = ("hi", "mr", "fr", None)  # not English and not a code language: every rule keeps


def doc(pack, text, lang): return {"doc_id": "x", "text": text, "url": "http://a/b", "lang_label": lang, "pack_id": pack}


def en(text, lang="en"):
    """A document of an English pack: en-dclm is tagged en, en-nemotroncc is tagged eng."""
    return doc("en-nemotroncc" if lang == "eng" else "en-dclm", text, lang)


# DA-F1-005 asks whether an organisation word sits in the 13 lines around a contact line (6 each way), and whether a
# public role (reporter, editor ...) sits in the line or the 2 lines above an email. These lines hold no digit, no
# at-sign and none of those words, and each differs from the others, so no other rule sees a repeat.
FILL = ["The river rises in the hills.", "It runs south to the sea.", "Growers plant rice and wheat.", "The dry months are long.",
        "Rain returns in June.", "Boats cross the bend at dawn.", "Fish are sold at noon.", "Evenings are quiet and cool.",
        "Children swim near the sandbar."]
ORG_LINE, ORG_LINE_2 = "The library is on the corner.", "The museum opens at nine."
PHONE_LINE, PHONE_MASKED = "Call 555-867-5309 now.", "Call <PHONE> now."
EMAIL_LINE, EMAIL_MASKED = "Write to ravi.kumar" + AT + "brightwave.net today.", "Write to <EMAIL> today."
ROLE_LINE, REACH_LINE = "The reporter wrote this story.", "Reach jane.doe" + AT + "newsdesk.net today."
ORG_VOICE_LINE = "Please call us any time."  # one phrase like this makes the whole page an organisation's


def gap(n):
    """n different lines with nothing in them, as a block of text."""
    return "\n".join(FILL[:n])


# DA-F1-005, the 13-line window, as (family, function, document, action, the text after a fix: None checks only that it changed)

F1_WINDOW_ROWS = [
    # DA-F1-005, the 13-line window: an organisation word 6 lines away keeps the contact, one 7 lines away does not
    ("english", "F1", en(ORG_LINE + "\n" + gap(5) + "\n" + PHONE_LINE), "keep", None),
    ("english", "F1", en(ORG_LINE + "\n" + gap(6) + "\n" + PHONE_LINE), "fix",
     ORG_LINE + "\n" + gap(6) + "\n" + PHONE_MASKED),
    ("english", "F1", en(EMAIL_LINE + "\n" + gap(5) + "\n" + ORG_LINE), "keep", None),  # the same below the contact line
    ("english", "F1", en(EMAIL_LINE + "\n" + gap(6) + "\n" + ORG_LINE), "fix",
     EMAIL_MASKED + "\n" + gap(6) + "\n" + ORG_LINE),
    ("english", "F1", en(FILL[0] + "\n" + PHONE_LINE + "\n" + ORG_LINE_2), "keep", None),  # an organisation on the next line
    ("english", "F1", en(ORG_VOICE_LINE + "\n" + gap(9) + "\n" + PHONE_LINE), "keep", None),  # an organisation's page, far away
    ("english", "F1", en("Email me at ravi dot kumar at brightwave dot net.\nPhone: five five five, eight six seven.\n" + ORG_LINE),
     "keep", None),  # no digit and no at-sign on any line
    ("english", "F1", en(FILL[0] + "\n\n" + PHONE_LINE + "\n\n" + FILL[1] + "\n"), "fix",
     FILL[0] + "\n\n" + PHONE_MASKED + "\n\n" + FILL[1] + "\n"),  # only the contact line changes: blank lines and the last newline stay
    # a public role in the email line or the 2 lines above it keeps an email at an organisation's domain; 3 lines above does not
    ("english", "F1", en(ROLE_LINE + "\n" + FILL[1] + "\n" + REACH_LINE), "keep", None),
    ("english", "F1", en(ROLE_LINE + "\n" + FILL[1] + "\n" + FILL[2] + "\n" + REACH_LINE), "fix",
     ROLE_LINE + "\n" + FILL[1] + "\n" + FILL[2] + "\nReach <EMAIL> today."),
]


@pytest.mark.parametrize("family, fn, row, action, after", F1_WINDOW_ROWS, ids=[f"f1-window-{i}" for i in range(len(F1_WINDOW_ROWS))])
def test_f1_treats_a_window_document(family, fn, row, action, after):
    a, new = c.RULES[fn](row)
    assert a == action
    if action in ("drop", "keep"):
        assert new == row["text"]
    elif after is not None:
        assert new == after
    else:
        assert new != row["text"]


@pytest.mark.parametrize("text", [r[2]["text"] for r in F1_WINDOW_ROWS], ids=[f"f1-window-{i}" for i in range(len(F1_WINDOW_ROWS))])
def test_a_f1_window_row_of_another_language_passes_unchanged(text):
    for lang in OTHER_LANGS:
        row = en(text, lang)
        assert all(fn(row) == ("keep", text) for fn in c.RULES.values())
        assert c.clean(row) == ("keep", text)


@pytest.mark.parametrize("text", [r[2]["text"] for r in F1_WINDOW_ROWS if r[3] == "fix"] + [PARA],
                         ids=[f"f1-window-{i}" for i, r in enumerate(F1_WINDOW_ROWS) if r[3] == "fix"] + ["plain"])
def test_clean_twice_is_clean_once_on_f1_window_rows(text):
    a, once = c.clean(en(text))
    if a != "drop":
        assert c.clean(en(once)) == ("keep", once)


@cache
def f1_namespace():
    """The DA-F1-005 block of specv1.md run on its own, to reach its regexes and its medicine_f1."""
    body = re.search(r"^## Medicine \d+ — english / F1 / `DA-F1-005`\n(.*?)(?=^## Medicine |\Z)", SHEET, re.S | re.M).group(1)
    ns = {"__name__": "rules"}
    exec(textwrap.dedent(re.search(r"^```python\n(.*?)^```", body, re.S | re.M).group(1)), ns)
    return ns


def f1_joined_window(text):
    """DA-F1-005 as it was before the window was searched line by line: for every line, ORGANISATION runs on the joined
    13-line window. The reference that medicine_f1 must equal."""
    ns = f1_namespace()
    lines = text.split("\n")
    org_page = bool(ns["ORG_VOICE"].search(text))
    out = []
    for i, line in enumerate(lines):
        org = org_page or bool(ns["ORGANISATION"].search("\n".join(lines[max(0, i - 6):i + 7])))
        new = ns["EMAIL"].sub(lambda m: ns["_email"](m, org or bool(ns["PUBLIC_ROLE"].search(
            "\n".join(lines[max(0, i - 2):i + 1])))), line)
        if ns["PHONE"].search(new) or ns["ADDRESS"].search(new):
            if not org:
                new = ns["PHONE"].sub(lambda m: m.group() if ns["TOLL_FREE"].match(m.group()) else "<PHONE>", new)
                if len(line.strip()) <= ns["CONTACT_LINE"]:
                    new = ns["ADDRESS"].sub("<ADDRESS>", new)
        out.append(new)
    return "\n".join(out)


def test_f1_window_fixtures_hold_what_the_cases_assume():
    ns = f1_namespace()
    plain = FILL + [PHONE_LINE, EMAIL_LINE, REACH_LINE]
    assert not any(ns[name].search(x) for name in ("ORGANISATION", "ORG_VOICE", "PUBLIC_ROLE") for x in plain)
    assert all(ns["ORGANISATION"].search(x) for x in (ORG_LINE, ORG_LINE_2))
    assert ns["ORG_VOICE"].search(ORG_VOICE_LINE) and not ns["ORGANISATION"].search(ORG_VOICE_LINE)
    assert ns["PUBLIC_ROLE"].search(ROLE_LINE) and not ns["ORGANISATION"].search(ROLE_LINE)


def test_f1_searching_the_window_line_by_line_equals_searching_it_joined():
    """The optimisation must change nothing: random documents of contact lines, organisation words, roles and plain lines,
    most of them close enough together for the 13-line windows to overlap."""
    pieces = (FILL + [ORG_LINE, ORG_LINE_2, ORG_VOICE_LINE, ROLE_LINE, PHONE_LINE, EMAIL_LINE, REACH_LINE, "", " ", "123",
              "Open Mon-Fri, fax 020 7946 0958", "Toll free 1-800-555-0199", "Ring +1 (312) 555-0142 or 312 555 0142",
              "Home: 42 Elm Street, Springfield, IL 62704", "12 N. Oak Avenue", "He lived at No. 12 High Street.",
              "sam99" + AT + "uni.edu", "info" + AT + "brightwave.net", "ravi_k" + AT + "gmail.com", "a" + AT + "b", "ten at the real\nestate office"])
    rng = random.Random(5)
    docs = ["\n".join(" ".join(rng.choices(pieces, k=rng.randint(0, 2))) for _ in range(rng.randint(1, 40))) for _ in range(2000)]
    expected = [f1_joined_window(d) for d in docs]
    assert [f1_namespace()["medicine_f1"](d) for d in docs] == expected
    assert 200 < sum(a != d for a, d in zip(expected, docs)) < len(docs) - 200  # both outcomes are exercised, so the check is not vacuous
    for d, e in zip(docs[:200], expected):  # clean.py's own F1, through the shared form
        want = ("drop", d) if not e.strip() else ("fix" if e != d else "keep", e)
        assert c.RULES["F1"](en(d)) == want


# Two pages laid out like real ones in week1-packs: en-dclm 91baef5b95248038 (short contact lines beside organisation lines)
# and en-nemotroncc fab230d1a3b03a77 (one last line holding a webmail address and a phone). Only the layout is copied:
# which lines are blank, where the organisation words and the contacts sit, how long the page is. Every word is made up.
SHAPE_PROSE = ["Rivers carry silt to the delta each spring.", "Her grandfather kept bees on the hill.",
               "The trail follows a ridge above the valley.", "Snow arrived early and stayed until April.",
               "Children learned to swim in the lake.", "A narrow bridge crosses the stream near the mill.",
               "Lanterns were lit along the harbour wall.", "The orchard produces apples, pears and plums.",
               "Winter evenings were spent reading by the fire.", "Few visitors reach the northern cliffs.",
               "Wild herbs grow between the old stones.", "The ferry leaves at dawn and returns at dusk.",
               "Fog settled over the fields before sunrise.", "Songs drifted from the open windows.",
               "Maps of the coast were drawn by hand.", "A quiet road leads past the old windmill.",
               "Gulls circled the boats all afternoon.", "Rain fell softly on the slate roofs.",
               "The baker's cart rolled down the lane.", "Distant bells rang across the water."]
SHAPE_A_BLANK = {0, 1, 3, 5, 7, 9, 12, 14, 16, 18, 20, 22, 24, 28, 30}  # 33 lines
SHAPE_A_CONTACT = {4: "Tel: 555-867-5309", 6: "Our address: 1425 Willow Street", 26: "880 Rose Road"}
SHAPE_A_MASKED = {4: "Tel: <PHONE>", 6: "Our address: <ADDRESS>", 26: "<ADDRESS>"}
SHAPE_A_ORG = {2: "Free consultation for new clients.",
               17: "The clinic keeps long opening hours and offers health checks on most days of the week.",
               21: "A small auto workshop stands beside the river, and the owner knows every engine in town.",
               25: "Family law and wills."}
SHAPE_A_PLAIN = {2: "Welcome to the village page.", 17: "Snow lay thick on the fields all week.",
                 21: "Rain fell on the roofs of the old town.", 25: "Candles burned late in the hall."}


def shape_a(org):
    """The first page: contacts on lines 4, 6 and 26, organisation words only on the lines named in org."""
    prose = iter(SHAPE_PROSE)
    lines = []
    for i in range(33):
        if i in SHAPE_A_CONTACT:
            lines.append(SHAPE_A_CONTACT[i])
        elif i in SHAPE_A_ORG:
            lines.append(SHAPE_A_ORG[i] if i in org else SHAPE_A_PLAIN[i])
        elif i in SHAPE_A_BLANK:
            lines.append("")
        else:
            lines.append(next(prose))
    return lines


@pytest.mark.parametrize("org, masked", [
    ((2, 17, 21, 25), ()),            # as in the real page: every contact has an organisation within 6 lines, nothing changes
    ((17, 21, 25), (4, 6)),           # line 2 is the only organisation word near lines 4 and 6, and 24 lines from line 26
    ((2, 17), (26,)),                 # lines 21 and 25 are the organisation words near line 26; line 17 is 9 lines away
    ((), (4, 6, 26)),                 # no organisation word: the phone and both addresses go
], ids=["organisations-everywhere", "top-organisation-removed", "bottom-organisations-removed", "no-organisation"])
def test_f1_on_a_page_laid_out_like_a_real_one(org, masked):
    lines = shape_a(org)
    assert len(lines) == 33
    text = "\n".join(lines)
    want = "\n".join(SHAPE_A_MASKED[i] if i in masked else line for i, line in enumerate(lines))
    assert c.RULES["F1"](en(text)) == (("fix", want) if masked else ("keep", text))
    assert f1_joined_window(text) == want  # and the joined-window reference agrees


def test_f1_masks_the_webmail_address_and_keeps_the_phone_on_one_organisation_line():
    """The second page: 41 lines, text on the even ones, and a last line with both, beside an organisation word."""
    lines = [SHAPE_PROSE[i // 2] if i % 2 == 0 else "" for i in range(38)] + ["The university runs the survey.", "",
             "Further information: front desk (555) 867\u20105309 or sam.lee" + AT + "gmail.com"]
    assert len(lines) == 41
    text = "\n".join(lines)
    want = "\n".join(lines[:-1] + ["Further information: front desk (555) 867\u20105309 or <EMAIL>"])
    assert c.RULES["F1"](en(text)) == ("fix", want)  # the phone stays: the line belongs to an organisation; a webmail address is a person's


def test_f1_shaped_pages_hold_what_their_cases_assume():
    ns = f1_namespace()
    plain = SHAPE_PROSE + list(SHAPE_A_PLAIN.values()) + list(SHAPE_A_CONTACT.values())
    assert not any(ns[name].search(x) for name in ("ORGANISATION", "ORG_VOICE", "PUBLIC_ROLE") for x in plain)
    assert all(ns["ORGANISATION"].search(x) for x in SHAPE_A_ORG.values())
    assert not any(ns["ORG_VOICE"].search(x) or ns["PUBLIC_ROLE"].search(x) for x in SHAPE_A_ORG.values())


# DA-D3-003: a line goes when it has 3 or more letters and more than half of them are not Latin. The rule skips ASCII lines
# (an ASCII letter is always Latin) and looks each letter's script up once, so these cases pin what it still must do.

def d3_reference(text):
    """DA-D3-003 as it was before the ASCII skip and the cache: unicodedata.name on every letter of every line."""
    def foreign(line):
        letters = [ch for ch in line if ch.isalpha()]
        return len(letters) >= 3 and sum(unicodedata.name(ch, "").split(" ")[0] != "LATIN" for ch in letters) / len(letters) > 0.5
    return "\n".join(line for line in text.split("\n") if not foreign(line))


def d3_result(text):
    """What d3 gives for text, in the shared form, from the reference."""
    out = d3_reference(text)
    return ("drop", text) if not out.strip() else ("fix" if out != text else "keep", out)


# (the line, whether D3 deletes it). Each line sits between two English paragraphs.
D3_LINES = [
    ("Hello world, and more. It's 5 o'clock (almost)!", False),   # ASCII only
    ("12345 67890 -- ??", False),                                 # no letters at all
    ("Café naïve façade déjà vu Ångström", False),                # Latin letters outside ASCII
    ("Это строка на русском языке.", True),                       # Cyrillic
    ("日本語のテキストです", True),                                 # Han and Hiragana
    ("한국어 문장입니다", True),                                     # Hangul
    ("यह हिंदी की एक पंक्ति है", True),                           # Devanagari
    ("Ελληνικό κείμενο εδώ", True),                               # Greek
    ("Hello Привет мир", True),                                   # ASCII word and Cyrillic: 9 of 14 letters are not Latin
    ("The word привет means hello in Russian.", False),           # a foreign word inside an English sentence: 6 of 32
    ("12345 -- привет", True),                                    # digits and punctuation do not count as letters
    ("ab аб", False),                                             # exactly half non-Latin is not more than half
    ("ab абв", True),                                             # 3 of 5
    ("abc аб", False),                                            # 2 of 5
    ("аб", False),                                                # fewer than 3 letters: kept however foreign
    ("я", False),
    ("абв", True),                                                # exactly 3 letters
]


@pytest.mark.parametrize("line, goes", D3_LINES, ids=[f"d3-line-{i}" for i in range(len(D3_LINES))])
def test_d3_deletes_a_line_only_when_more_than_half_its_letters_are_not_latin(line, goes):
    text = PARA + "\n" + line + "\n" + EN_TEXT
    a, new = c.d3(en(text))
    assert (a, new) == (("fix", PARA + "\n" + EN_TEXT) if goes else ("keep", text))
    assert (a, new) == d3_result(text)  # and the reference agrees


def test_d3_cached_script_lookup_equals_unicodedata_name_for_every_letter():
    """A line of one repeated letter goes exactly when the reference says so, for every alphabetic code point. The same
    letters come up again and again, so this also reads the cache after it is filled."""
    for _ in range(2):
        for cp in range(0x110000):
            if chr(cp).isalpha():
                line = chr(cp) * 3
                text = PARA + "\n" + line
                a, new = c.d3(en(text))
                assert (a == "fix") == (unicodedata.name(chr(cp), "").split(" ")[0] != "LATIN"), hex(cp)
                assert new == d3_reference(text), hex(cp)


def test_every_ascii_letter_is_latin():
    """The reason D3 may skip an ASCII line."""
    assert all(unicodedata.name(chr(cp)).split(" ")[0] == "LATIN" for cp in range(128) if chr(cp).isalpha())


def test_d3_equals_the_unicodedata_reference_on_random_documents():
    """Random lines from scripts, digits, punctuation and spaces, so that ASCII, Latin, foreign and mixed lines all come up."""
    pools = ["abcdefghijklmnopqrstuvwxyz", "ABCDEFGHIJKLMNOPQRSTUVWXYZ", "0123456789 .,;:!?'-()", "    ", "éèàüöñçøåßÿ", "абвгдежзийклмн", "ΑΒΓαβγδεζ",
             "日本語中文字漢", "한국어문장", "कखगघचछजझ्ािी", "ابتثجحخد", "😀🙂★☆"]
    rng = random.Random(11)
    docs = []
    for _ in range(3000):
        lines = []
        for _ in range(rng.randint(1, 8)):
            mix = rng.sample(pools, rng.randint(1, 3))
            lines.append("".join(rng.choice(rng.choice(mix)) for _ in range(rng.randint(0, 40))))
        docs.append("\n".join(lines))
    cut = [d for d in docs if d3_reference(d) != d]
    assert 300 < len(cut) < len(docs) - 300  # lines go in some documents and stay in others: the check is not vacuous
    for d in docs:
        assert c.d3(en(d)) == d3_result(d)


# DA-C5-002: drop a document when two different spam patterns match, or when one line is 7 or more short "|"-separated phrases.
# The rule searches the union of the nine patterns first and checks for 6 "|" before the line pattern; these cases pin that
# it still drops exactly what the nine separate searches and the line pattern alone drop.

C5_SPAM_PATTERNS = (  # the nine patterns as the author wrote them, searched one by one
    r"\bcheap\s+(?:nfl|nba|mlb|nhl)\b",
    r"\bcheap\s+(?:michael\s+kors|oakley|ray\s*bans?|jerseys)\b",
    r"\bray\s*bans?\s+(?:sunglasses|outlet)\b",
    r"\b(?:auto|car)\s+insur\w*\s+(?:quotes?|qiotes?|rates?)\b",
    r"\bfree\s+auto\s+insurance\s+quotes\b",
    r"\bsex\s+dating\s+in\b",
    r"\bwrite\s+my\s+essay\b",
    r"\b(?:카지노|바카라|룰렛)\b",
    r"\bpigmentación\s+capilar\b",
)
C5_SPAM = tuple(re.compile(p, re.I) for p in C5_SPAM_PATTERNS)
C5_KEYWORD_LINE = re.compile(r"^(?![ \t]*[•·*\-])(?:[^|\n]{2,40}\|){6,}[^|\n]{2,40}[ \t]*$", re.M)


def c5_result(text):
    """DA-C5-002 as it was before the union search: nine searches, then the line pattern on every document."""
    dropped = sum(1 for p in C5_SPAM if p.search(text)) >= 2 or C5_KEYWORD_LINE.search(text)
    return ("drop", text) if dropped or not text.strip() else ("keep", text)


def c5_002(text):
    return c._run(en(text), c._english_c5_002)


C5_CASES = [
    ("Visit our shop for cheap NFL jerseys today.", False),                     # one pattern is not enough
    ("Cheap NFL gear. Cheap NBA gear. Cheap MLB gear.", False),                 # the same pattern three times counts once
    ("cheap nfl jerseys and free auto insurance quotes", True),                 # two different patterns
    ("CHEAP MICHAEL KORS and WRITE MY ESSAY", True),                            # case does not matter
    ("Cheap\n\nRay Bans Sunglasses and Sex Dating In Texas", True),             # \s spans a line break, \b needs a word edge
    ("cheapnfl and sex dating in", False),                                      # no \b between "cheap" and "nfl": one pattern only
    ("카지노 and 바카라 sites", False),                                          # one Korean pattern is one pattern
    ("카지노 and write my essay", True),
    ("Pigmentación Capilar y write my essay", True),
    ("ſex dating in town and write my essay", True),                            # re.I folds U+017F to s: the union must see it too
    ("auto insurance qiotes and car insur rates", False),                       # both words match the same single pattern
    ("A | b | c | d | e | f | g", True),                                        # 6 pipes on a line: 7 phrases
    ("A | b | c | d | e | f", False),                                           # 5 pipes
    ("ab|cd|ef|gh|ij|kl|mn", True),                                             # no spaces
    ("a|b|c|d|e|f|g", False),                                                   # phrases need 2 or more characters
    ("• ab|cd|ef|gh|ij|kl|mn", False),                                          # a bullet line is not a keyword line
    ("one|two\nthree|four\nfive|six\nseven|eight", False),                      # 7 pipes in all, but no line has 6
    ("An intro line.\nab|cd|ef|gh|ij|kl|mn\nA closing line.", True),            # the line sits inside a document
    ("Nothing to see here, just an English line.", False),
]


@pytest.mark.parametrize("text, goes", C5_CASES, ids=[f"c5-case-{i}" for i in range(len(C5_CASES))])
def test_c5_002_drops_what_the_nine_searches_and_the_line_pattern_drop(text, goes):
    full = PARA + "\n" + text + "\n" + EN_TEXT
    assert c5_002(full) == c5_result(full)
    assert (c5_002(full)[0] == "drop") == goes


def test_c5_002_equals_the_reference_on_random_documents():
    """Documents built from spam phrases, pipe runs, bullets and filler, so that every pattern, every pair and the line
    pattern come up alone, together and not at all."""
    phrases = ["cheap nfl", "Cheap Oakley", "ray bans outlet", "auto insurance quotes", "car insur rates", "free auto insurance quotes",
               "sex dating in", "write my essay", "카지노", "룰렛", "pigmentación capilar", "ſex dating in", "CHEAP JERSEYS", "ray-bans sunglasses"]
    filler = ["the river rises", "in the hills", "and runs south", "to the sea.", "cheap", "dating", "insurance", "essay", "|", "- item", "• item"]
    rng = random.Random(5)
    docs = []
    for _ in range(4000):
        lines = []
        for _ in range(rng.randint(1, 6)):
            kind = rng.random()
            if kind < 0.3:
                lines.append(" ".join(rng.choice(filler + phrases) for _ in range(rng.randint(1, 8))))
            elif kind < 0.6:
                lines.append(("• " if rng.random() < 0.2 else "") + " | ".join(rng.choice(filler[:4] + ["ab", "x", "cd ef"]) for _ in range(rng.randint(2, 10))))
            else:
                lines.append(rng.choice(phrases) + " " + rng.choice(filler))
        docs.append("\n".join(lines))
    dropped = [d for d in docs if c5_result(d)[0] == "drop"]
    assert 400 < len(dropped) < len(docs) - 400  # documents go in some cases and stay in others: the check is not vacuous
    assert sum(1 for d in docs if sum(1 for p in C5_SPAM if p.search(d)) == 1) > 200  # one pattern alone is common
    for d in docs:
        assert c5_002(d) == c5_result(d)


# DA-B4-010: drop a document when more than half of it is code-looking lines, else remove template expressions ({{ x }},
# ${x}, NaN ...). The rule stops reading lines once the answer is certain and skips the template search for a text with no
# template hint; these cases pin that it still gives what the full line scan and the full template search give.

B4_TEMPLATE_REF = re.compile(
    r"\{\{[^{}\n]{0,80}\}\}"
    r"|\{%[^\n]{0,80}?%\}"
    r"|\$\{[^{}\n]{0,80}\}"
    r"|`[\w.]+(?:\s*(?:\|\||&&)\s*[\w.]+)+`"
    r"|(?<=[:=$])\s*(?:undefined|NaN)\b"
    r"|^[ \t]*(?:undefined|NaN|null)[ \t]*$", re.M)
B4_CODE_TOKENS_REF = re.compile(
    r"::|->|=>|==|!=|<=|>=|&&|\|\||\+\+|\+=|-=|\);|\)\s*\{|[{};]\s*$|\}"
    r"|^\s*(?:#include|import\s|from\s+\S+\s+import|def\s|class\s|public\s|private\s|static\s|void\s|return\b"
    r"|require\s|var\s|let\s|const\s|function\b|Sub\s|End Sub|Dim\s|\$\s|>>>\s|>\s)"
    r"|\b(?:End Sub|ByVal|As Integer|System\.out|printf|println)\b", re.M)
B4_TABLE_ROW_REF = re.compile(r"^\s*\|.*\|\|")


def b4_reference(text):
    """DA-B4-010 as it was before the early stop and the hint check: every line scanned, the template regex on every text."""
    code = 0
    for line in text.split("\n"):
        tokens = len(B4_CODE_TOKENS_REF.findall(line))
        if tokens >= 2 and tokens * 6 >= len(line.split()) and not B4_TABLE_ROW_REF.match(line):
            code += len(line)
    if text.strip() and code > 0.5 * len(text.strip()):
        return ("drop", text)
    new = B4_TEMPLATE_REF.sub("", text)
    return ("fix" if new != text else "keep", new)


def b4_010(text):
    return c._run(en(text), c._english_b4_010)


B4_CODE_LINE = "x = a == b && c != d;"            # 4 code tokens in 7 words
B4_PROSE_LINE = "The river rises in the hills and runs south to the sea."
B4_CASES = [
    (B4_CODE_LINE + "\n" + B4_CODE_LINE, "drop"),                                      # all code
    (B4_PROSE_LINE + "\n" + B4_PROSE_LINE, "keep"),                                    # all prose
    (B4_PROSE_LINE + "\n" + B4_CODE_LINE * 6, "drop"),                                 # code only in the second half: no early stop to "keep"
    ("\n".join([B4_CODE_LINE] * 6 + [B4_PROSE_LINE]), "drop"),                         # code first
    ("\n".join([B4_PROSE_LINE] * 6 + [B4_CODE_LINE]), "keep"),                         # a little code at the end
    ("if (a == b) { return c; }", "drop"),
    ("| a | b ||\n| c | d ||\n| e == f && g ||", "keep"),                              # table rows are never code lines
    ("Hello {{ name }}, welcome back.", "drop"),                                       # two "}" in 5 words: a code line, so the whole text is code
    ("The river rises in the hills and runs south to the sea, so hello {{ name }} and welcome back.", "fix"),   # more than 12 words: not a code line
    ("The river rises in the hills and runs south to the sea, so {% if x %}Hi{% endif %} and welcome back.", "fix"),
    ("The total is ${price} today.", "fix"),
    ("Price: NaN\nName=undefined", "fix"),
    ("Set `page.data.title || page.data.name` first.", "fix"),
    ("A line\nundefined\nanother line\nnull\nNaN", "fix"),
    ("A nullable value is not null-safe, and the NaNs and undefineds stay.", "keep"),   # hints present, no match
    ("Nothing special here at all.", "keep"),
    ("", "keep"),
    ("   \n\n  ", "keep"),
    ("{{", "keep"),
    ("{{ unterminated\n}}", "keep"),
]


@pytest.mark.parametrize("text, action", B4_CASES, ids=[f"b4-case-{i}" for i in range(len(B4_CASES))])
def test_b4_010_gives_what_the_full_scan_gives(text, action):
    got = b4_010(text)
    assert got == b4_reference(text)
    assert got[0] == action


def test_b4_010_threshold_is_exact_around_half_the_text():
    """A code line and a prose line of every length around the point where code is exactly half of the text."""
    for n in range(len(B4_CODE_LINE) - 6, len(B4_CODE_LINE) + 7):
        for tail in ("", "\n", " "):
            for text in (B4_CODE_LINE + "\n" + "p" * n + tail, "p" * n + "\n" + B4_CODE_LINE + tail):
                assert b4_010(text) == b4_reference(text), (n, tail)
    half = len(B4_CODE_LINE) - 1                    # the code line plus "\n" plus this many characters: code is exactly half
    assert b4_010(B4_CODE_LINE + "\n" + "p" * half)[0] == "keep"        # exactly half is not more than half
    assert b4_010(B4_CODE_LINE + "\n" + "p" * (half - 1))[0] == "drop"  # one character less prose: just over half
    assert b4_010(B4_CODE_LINE + "\n" + "p" * (half + 1))[0] == "keep"


def test_b4_010_equals_the_reference_on_random_documents():
    """Documents of code lines, prose, template pieces, table rows, blank lines and hint words, so that every outcome and
    both early stops come up."""
    pieces = [B4_CODE_LINE, "if (a == b) { return c; }", "import os", "#include <stdio.h>", "foo->bar(baz);", "}", B4_PROSE_LINE,
              "Some words in a plain sentence.", "Hello {{ user }}!", "Cost ${total}", "{% for x in y %}", "Price: NaN", "undefined", "null",
              "`a.b || c.d`", "| a | b ||", "| a == b && c ||", "", "   ", "nullable and NaNs", "a ==", "x = y;"]
    rng = random.Random(9)
    docs = ["\n".join(rng.choice(pieces) for _ in range(rng.randint(1, 12))) for _ in range(4000)]
    results = [b4_reference(d) for d in docs]
    assert sum(r[0] == "drop" for r in results) > 300
    assert sum(r[0] == "fix" for r in results) > 300
    assert sum(r[0] == "keep" for r in results) > 300
    for d, expected in zip(docs, results):
        assert b4_010(d) == expected


# DA-B4-008: a detokenised opening quote ("said'sure it is' and") gets its space back: said 'sure it is' and. The rule looks at
# the apostrophes with str.find and asks the pattern about each one in place, instead of scanning every position;
# these cases pin that it still changes exactly what DETOK.sub changes.

B4_008_DETOK = re.compile("(?<=[A-Za-z])(?<![^A-Za-z][frbuFRBU])(?<![^A-Za-z][rR][bB])(?<![^A-Za-z][bB][rR])'(?!s\\b|t\\b|re\\b|ll\\b|ve\\b|d\\b|m\\b|em\\b|n'|er\\b|clock\\b|all\\b|am\\b)(?=[srmSRM][A-Za-z][^'\\n]{0,40}'(?:[\\s,.;:)!?]|$))", re.M)


def b4_008_reference(text):
    """DA-B4-008 as it was: one DETOK.sub over the whole text."""
    new = B4_008_DETOK.sub(" '", text)
    return ("fix" if new != text else "keep", new)


def b4_008(text):
    return c._run(en(text), c._english_b4_008)


B4_008_CASES = [
    ("He said'sure that is fine' and left.", "fix"),
    ("He said'sure it is'", "fix"),                                     # the closing quote ends the text
    ("One said'sure it is' and two said'maybe so' later.", "fix"),      # two matches
    ("It's fine, don't worry, we'll go, they've left, I'm here.", "keep"),   # contractions
    ("No apostrophe in this text at all.", "keep"),
    ("", "keep"),
    ("'sure thing' he said", "keep"),                                   # a quote at the very start has no letter before it
    ("a'sure it is' ok", "fix"),                                        # the f/r/b/u guard needs two characters behind the quote: at the start of the text there are none
    ("the f'sure it is' ok", "keep"),                                   # f, r, b, u after a non-letter are prefixes (f'..., r'..., b'...)
    ("the rb'sure it is' ok", "keep"),
    ("the br'sure it is' ok", "keep"),
    ("He said'sure it\nis' and left.", "keep"),                         # the span may not cross a line
    ("He said'sure it is'x and left.", "keep"),                         # the closing quote must end a word
    ("He said'sure it is'. And left.", "fix"),                          # or be followed by punctuation
    ("café'sure it is' ok", "keep"),                                    # é is not A-Z or a-z
    ("He said’sure it is’ and left.", "keep"),                # curly quotes are not touched
    ("SAID'SURE IT IS' OK", "fix"),
    ("He said's it is' and left.", "keep"),                             # 's is a contraction
    ("He said'rm it is' and 'quoted' words 'here' too.", "fix"),
]


@pytest.mark.parametrize("text, action", B4_008_CASES, ids=[f"b4-008-case-{i}" for i in range(len(B4_008_CASES))])
def test_b4_008_gives_what_detok_sub_gives(text, action):
    got = b4_008(text)
    assert got == b4_008_reference(text)
    assert got[0] == action


def test_b4_008_span_length_boundary():
    """The quoted span may hold 0 to 40 characters after its first two letters: 40 is a match, 41 is not."""
    for n in range(0, 46):
        text = "He said'sa" + "b" * n + "' and left."
        assert b4_008(text) == b4_008_reference(text), n
    assert b4_008("He said'sa" + "b" * 40 + "' x")[0] == "fix"
    assert b4_008("He said'sa" + "b" * 41 + "' x")[0] == "keep"


def test_b4_008_equals_the_reference_on_random_documents():
    """Random mixes of words, quotes, contractions, newlines and punctuation, so that matches, near misses and quote-free
    texts all come up."""
    pieces = ["He said", "'", "sure", "it is", "'s", "n't", "'re", "don", "rb", "f", "u", "br", "'sure that is fine'", "'me too'", "x'",
              "\n", " ", ".", ",", "!", "word", "café", "’", "'ll", "o'clock", "rock'n'roll", "''", "a", "R"]
    rng = random.Random(8)
    docs = ["".join(rng.choice(pieces) for _ in range(rng.randint(1, 30))) for _ in range(6000)]
    results = [b4_008_reference(d) for d in docs]
    assert sum(r[0] == "fix" for r in results) > 300
    assert sum("'" not in d for d in docs) > 100
    assert sum(r[0] == "keep" and "'" in d for r, d in zip(results, docs)) > 1000
    for d, expected in zip(docs, results):
        assert b4_008(d) == expected


# DA-B2-002: with 2 or more HTML tags in a text that is not about HTML, the tags become spaces and runs of spaces shrink to one.
# The rule counts "<" first (a tag holds exactly one), then the tags, and only then asks whether the text is about HTML (the slow
# search); these cases pin that it still gives what the original order gives.

B2_TAG_REF = re.compile(r"</?(?:div|span|p|br|li|ul|ol|img|table|tr|td|th|tbody|thead|strong|em|font|iframe|script|style|center|hr)"
                        r"(?:\s[^<>]{0,200})?\s*/?>", re.I)
B2_ABOUT_HTML_REF = re.compile(r"```|\bclass\s*=|<!DOCTYPE|\bhtml\b|\btag\b|\belement\b", re.I)


def b2_reference(text):
    """DA-B2-002 as it was: the about-HTML search first, then the tag count."""
    if B2_ABOUT_HTML_REF.search(text) or len(B2_TAG_REF.findall(text)) < 2:
        return ("keep", text)
    new = re.sub(r"[ \t]{2,}", " ", B2_TAG_REF.sub(" ", text))
    return ("fix" if new != text else "keep", new)


def b2_002(text):
    return c._run(en(text), c._english_b2_002)


B2_CASES = [
    ("Hello <div>world</div> and more.", "fix"),
    ("Hello <DIV>world</DIV>  and   more.", "fix"),                    # tag names are case-insensitive; spaces collapse
    ("Hello <b>bold</b> and <i>it</i>.", "keep"),                      # b and i are not in the list
    ("Hello <div>world and more.", "keep"),                            # one tag is not enough
    ("No angle bracket in this text at all.", "keep"),
    ("", "keep"),
    ("if a < b and b < c then a < c", "keep"),                         # three "<" but no tag
    ("Use the html element: <div>a</div> <span>b</span>", "keep"),     # about HTML ("html", "element")
    ("Each tag is shown here: <div>a</div> <p>b</p>", "keep"),         # "tag"
    ("Code: ```<div>a</div> <p>b</p>```", "keep"),                     # a code fence
    ("<div class=\"x\">a</div> <p>b</p>", "keep"),                     # class=
    ("<!DOCTYPE x> <div>a</div> <p>b</p>", "keep"),
    ("<p>One</p>\t<p>Two</p>   <br/>  Three", "fix"),
    ("<div <div>a</div> <p>b</p>", "fix"),                             # the first "<div" is no tag; the other two are
    ("<img src=a.png><hr>", "fix"),
    ("<table><tr><td>cell</td></tr></table>", "fix"),
]


@pytest.mark.parametrize("text, action", B2_CASES, ids=[f"b2-case-{i}" for i in range(len(B2_CASES))])
def test_b2_002_gives_what_the_original_order_gives(text, action):
    got = b2_002(text)
    assert got == b2_reference(text)
    assert got[0] == action


def test_b2_002_attribute_length_boundary():
    """A tag may hold up to 200 characters of attributes: 200 is a tag, 201 is not."""
    for n in range(195, 206):
        text = "a <p " + "x" * n + "> b <hr> c"
        assert b2_002(text) == b2_reference(text), n
    assert b2_002("a <p " + "x" * 199 + "> b <hr> c")[0] == "fix"       # " " plus 199 characters
    assert b2_002("a <p " + "x" * 205 + "> b <hr> c")[0] == "keep"      # only <hr> is a tag


def test_b2_002_equals_the_reference_on_random_documents():
    """Texts built from tags, bare angle brackets, HTML-about words, spaces, tabs and newlines."""
    pieces = ["<div>", "</div>", "<p>", "</p>", "<br/>", "<br />", "<span style=\"a\">", "<TD>", "<b>", "<", ">", "a < b", "<!DOCTYPE html>",
              "html", "tag", "element", "class=", "class =", "```", "word", "more words here", " ", "  ", "\t", "\n", "<img src=x>", "<hr>",
              "<p " + "y" * 200 + ">", "<p " + "y" * 201 + ">", "<scripts>", "</ p>"]
    rng = random.Random(2)
    docs = ["".join(rng.choice(pieces) for _ in range(rng.randint(1, 14))) for _ in range(6000)]
    results = [b2_reference(d) for d in docs]
    assert sum(r[0] == "fix" for r in results) > 300
    assert sum(r[0] == "keep" and d.count("<") >= 2 for r, d in zip(results, docs)) > 300   # two "<" and still kept
    for d, expected in zip(docs, results):
        assert b2_002(d) == expected


# DA-C3-003: drop a text that ends in a "[...]" marker, or that is cut off after a dangling word (the, and, of ...); a text ending
# in sentence punctuation is kept. The rule reads the last word backwards from the end instead of searching the whole text for
# it with a regex; these cases pin that it still drops exactly what the regex version dropped.

C3_MARKER_REF = re.compile(r"\[\s*(?:…|\.\.\.)\s*\]$")
C3_DANGLING_REF = frozenset("a an the and or but of to in on at for with from by as is are was were "
                            "that this these those which who because however".split())
C3_LAST_TOKEN_REF = re.compile(r"([A-Za-z0-9']+)[^A-Za-z0-9']*$")


def c3_reference(text):
    """DA-C3-003 as it was: a regex searched the whole text for the last token. Returned in the shared form."""
    tail = text.rstrip()
    if not tail:
        out = text
    elif C3_MARKER_REF.search(tail):
        out = ("drop", "")
    elif tail.endswith((".", "!", "?", '"', "'", ")", ":", "…")):
        out = text
    else:
        token = C3_LAST_TOKEN_REF.search(tail)
        out = ("drop", "") if token and token.group(1).lower() in C3_DANGLING_REF else text
    if isinstance(out, tuple) or not out.strip():
        return ("drop", text)
    return ("keep", text)


def c3_003(text):
    return c._run(en(text), c._english_c3_003)


C3_CASES = [
    ("He went to the", "drop"),
    ("He went to the market.", "keep"),
    ("Read more [...]", "drop"),
    ("Read more [ … ]", "drop"),
    ("Read more [\u2026]\n\n", "drop"),                         # trailing whitespace is ignored
    ("[...] and then more text here", "keep"),                   # a marker that does not end the text
    ("We stopped because", "drop"),
    ("WE STOPPED BECAUSE", "drop"),                              # case does not matter
    ("He went to the,", "drop"),                                 # punctuation after the word is skipped
    ("He went to the \n\n", "drop"),
    ("He went to the \u00e9", "drop"),                           # a non-ASCII character after it is not part of any token
    ("He went to the...", "keep"),                               # ends in a full stop
    ("There were 12", "keep"),                                   # a number is not a dangling word
    ("The word it's", "keep"),
    ("a", "drop"),                                               # the whole text is one dangling word
    ("", "drop"),                                                # a blank text is dropped by the shared form
    ("  \n ", "drop"),
    ("!!!", "keep"),
    ("Il est all\u00e9 \u00e0 la", "keep"),                      # "la" is not in the list
]


@pytest.mark.parametrize("text, action", C3_CASES, ids=[f"c3-case-{i}" for i in range(len(C3_CASES))])
def test_c3_003_gives_what_the_regex_version_gives(text, action):
    got = c3_003(text)
    assert got == c3_reference(text)
    assert got[0] == action


def test_c3_003_last_word_length_boundary():
    """The longest dangling word has 7 letters: a last token of 7 letters can be one, a longer one cannot."""
    for word in ("because", "however", "bbecause", "xhowever", "because'", "'because", "be", "becaus", "x" * 40, "because" * 5):
        for tail in ("", " ", "  ,", ";;"):
            text = "He said " + word + tail
            assert c3_003(text) == c3_reference(text), (word, tail)
    assert c3_003("He said because")[0] == "drop"
    assert c3_003("He said bbecause")[0] == "keep"


def test_c3_003_equals_the_reference_on_random_documents():
    """Texts built from dangling words, other words, markers, punctuation, spaces, newlines, digits and non-ASCII characters."""
    pieces = ["the", "and", "of", "because", "however", "market", "He went", "12", "it's", "who", "a", "x" * 9, "[...]", "[ \u2026 ]", "[",
              "]", ".", "!", "?", ",", ";", ":", '"', "'", ")", "\u2026", " ", "  ", "\n", "\t", "\u00e9", "\u4e2d", "-", "...", "THE", "To"]
    rng = random.Random(3)
    docs = ["".join(rng.choice(pieces) + (" " if rng.random() < 0.6 else "") for _ in range(rng.randint(0, 10))) for _ in range(8000)]
    results = [c3_reference(d) for d in docs]
    assert sum(r[0] == "drop" for r in results) > 500
    assert sum(r[0] == "keep" for r in results) > 500
    for d, expected in zip(docs, results):
        assert c3_003(d) == expected
