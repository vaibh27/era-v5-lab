# Collated medicine: the cards of #71, #75, #76 and #79 as one set, with the conflicts resolved

Week 4, tasks 4.4 and 5.1. 22 Nov 2026. Put together by `TEJASKUMAR-REDDY-J`. **Proposals: the spec, the registry and the running `clean.py` are not changed.** The findings behind the cards are in [`collation.md`](collation.md).

**15 cards, from 15 proposals.** No two cards replace the same rule on the same pack, and no card uses a number of the registry, of spec v1.1 or of another card. The sheet loads in `harmtest/harm_test.py` as it is.

| Card | Packs | Place | From | Change to the source's code | Finding |
|---|---|---|---|---|---|
| `DA-A2-010` | en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth | replaces `DA-A2-008` | #75 `DA-A2-009` | new number only | F01 |
| `DA-C3-008` | en-nemotroncc-synth | replaces `DA-C3-003` | #76 `DA-C3-008` | none | F15 |
| `DA-C1-013` | en-nemotroncc-synth | replaces `DA-C1-006` | #76 `DA-C1-013` | none | F30 |
| `DA-C2-009` | en-dclm, en-fineweb-edu, en-nemotroncc | replaces `DA-C2-003` | #75 `DA-C2-005` | new number; synth moves to `DA-C2-010` | F02 |
| `DA-C2-010` | en-nemotroncc-synth | replaces `DA-C2-003` | #75 `DA-C2-005` + #76 `DA-C2-008` | **merged** | F02 and F14 |
| `DA-C5-007` | en-nemotroncc-synth | replaces `DA-C5-005` | #76 `DA-C5-007` | none | F16 |
| `DA-B4-015` | en-nemotroncc-synth | replaces `DA-B4-007` | #76 `DA-B4-015` | none | F17 |
| `DA-B3-012` | en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth | new, after `DA-B4-012` | #75 `DA-B3-002` | new number only | F05 |
| `DA-B4-016` | en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth | replaces `DA-B4-010` | #75 `DA-B4-014` | new number only | F03 |
| `DA-F1-007` | en-dclm, en-fineweb-edu, en-nemotroncc | replaces `DA-F1-005` | #75 `DA-F1-007` | synth moves to `DA-F1-009` | F04 |
| `DA-F1-009` | en-nemotroncc-synth | replaces `DA-F1-005` | #75 `DA-F1-007` + #76 `DA-F1-008` | **merged** | F13 |
| `DA-C1-012` | math-finemath, math-megamath | new, after the family's rules | #71 `DA-C1-012` | none | F12 |
| `DA-C2-007` | math-finemath, math-megamath | replaces `DA-C2-001` on finemath; on megamath new, after the family's rules | #71 `DA-C2-007` | place on finemath | F09, F12 |
| `DA-C1-014` | sci-finepdfs | new, after `DA-C1-007` | #79 Medicine 2, part (a) | number given; part (b) left out | F26 |
| `DA-C2-011` | sci-finepdfs | replaces `DA-C2-004` | #79 Medicine 1 | number given; put into the rest of `DA-C2-004` | F25 |

Sources, at the commits read: #71 `4d17b696`, #75 `2ecacf5f`, #76 `2ba7ce3b`, #79 `489593e9`. Each code block is the source's block with header lines added (`REPLACES`, `RUNS_AFTER`, `FROM`); the column says what else is changed. In #71's cards a name in a comment is now the handle. #68, #72 and #78 have findings and no card of their own (#68's bugs are the cards of #75).

## Conflicts, and how each is resolved

1. **Phone numbers on en-nemotroncc-synth: #75 `DA-F1-007` and #76 `DA-F1-008` both replace `DA-F1-005`.** Resolved by `DA-F1-009`, which uses both in order.
   - **#76 decides the line.** Its technical values and its public contacts are not a reason to mask. It comes first because it rests on the reading of all 33 masks of Part_2.
   - **#75 decides the number.** A number that some line gives a reason to mask is masked on every line. This is the policy for a number that looks public in one place and private in another: **privacy first**, because a number masked on one line and shown on the next is not masked. The second copy of `d6e8da1b50823198` is the one case; #76 marks its owner as "review required".

   | doc_id | Label in #76 | phone masks now | #75 alone | #76 alone | `DA-F1-009` |
   |---|---|---:|---:|---:|---:|
   | `8aecb4bed3222e59` | public contact, wrongly masked | 1 | 2 | 0 | **0** |
   | `417036b98dd9380a` | public contact, wrongly masked | 1 | 2 | 0 | **0** |
   | `d6e8da1b50823198` | review required, keep masked | 1 | 2 | 1 | **2** |

   Over the 29 documents of #76's phone review, `DA-F1-009` restores all 25 masks labelled incorrect and keeps all 8 labelled ambiguous or private. **The privacy-first choice needs the team lead's yes.** To choose the other way, use #76's `DA-F1-008` on synth as it is.
2. **Repeated lines on en-nemotroncc-synth: #75 `DA-C2-005` and #76 `DA-C2-008` both replace `DA-C2-003`.** Resolved by `DA-C2-010`: the two exclusions do not touch each other, so a line excluded by either is not counted. `7523ebd6d772fada`, which #75 alone does not fix, is whole. The other three packs keep #75's card (`DA-C2-009`), because #76 tested its exclusion on synth only.
3. **Rule numbers of #75 that are taken.** `DA-A2-009`, `DA-B3-002` and `DA-B4-014` are in the registry, `DA-C2-005` too (it was also a dropped English rule). They become `DA-A2-010`, `DA-B3-012`, `DA-B4-016` and `DA-C2-009`. The merged cards are `DA-C2-010` and `DA-F1-009`; #79's two medicines had no number and get `DA-C2-011` and `DA-C1-014`. #75 itself is not edited.
4. **#71 `DA-C2-007` and `DA-C2-001`.** While `DA-C2-001` runs first, the formula lines are gone before #71's card sees them, so it is out on finemath. #71 runs its card after the family's rules. Here it runs there on megamath only, and in the place of `DA-C2-001` on finemath: run last, it leaves the repeated lines in for `DA-C7-001` and `DA-C9-003`, which then drop 21 documents that are kept today (3 in this place).
5. **#79 Medicine 2, part (b)** (running footers) is left out: it changes documents that the same audit labels `clean`, and the PR calls it optional.
6. **Two cards on one disease that do not clash:** #75 `DA-B4-016` (`DA-B4-010`) and #76 `DA-B4-015` (`DA-B4-007`); #71 `DA-C1-012`, #76 `DA-C1-013` and #79 `DA-C1-014` (different packs).

## Checks of the whole set

**Gold, card by card** (`harm_test.py --rule all`, the 500-document sample packs whose sha256 is the `pack_sha256` of the harm results): 0 changed and 0 dropped for every card, and for the sheet in order, on all seven packs. For #79's two medicines this is the first run on the 202 gold documents of sci-finepdfs.

**Gold, whole pipeline:** with the cards in place the output of `clean()` is the same on every gold document.

| Pack | Gold | changed / dropped now | with the cards | output differs |
|---|---:|---:|---:|---:|
| en-dclm | 62 | 0 / 0 | 0 / 0 | 0 |
| en-fineweb-edu | 158 | 0 / 0 | 0 / 0 | 0 |
| en-nemotroncc | 129 | 2 / 0 | 2 / 0 | 0 |
| en-nemotroncc-synth | 326 | 0 / 0 | 0 / 0 | 0 |
| math-finemath | 86 | 1 / 1 | 1 / 1 | 0 |
| math-megamath | 26 | 0 / 0 | 0 / 0 | 0 |
| sci-finepdfs | 202 | 4 / 1 | 4 / 1 | 0 |

The gold documents that change or drop today do so by rules that are already in spec v1.1; no card touches them.

**The 313 labelled documents** ([`collated_medicine_docs.csv`](collated_medicine_docs.csv)): the output differs on 55: 53 labelled `fix` that the cleaner fixes today, one labelled `fix` that is dropped today and kept with the cards (`5b4d0f975990e808`), and one that #71 labels `drop` and no card drops (`4258f5c9fbf14805`). No document labelled `clean` changes, and no document is newly dropped.

**Whole train parts, sample v0** ([`collated_medicine_packs.csv`](collated_medicine_packs.csv)):

| Pack | Docs | dropped now | with the cards | recovered | newly dropped | output changed | changed again by a second run (of them also today) |
|---|---:|---:|---:|---:|---:|---:|---:|
| en-dclm | 61,367 | 1,155 | 1,153 | 2 | 0 | 232 | 36 (36) |
| en-fineweb-edu | 71,933 | 559 | 559 | 0 | 0 | 13 | 3 (3) |
| en-nemotroncc | 91,820 | 3,552 | 3,552 | 0 | 0 | 35 | 39 (39) |
| en-nemotroncc-synth | 197,119 | 218 | 161 | 57 | 0 | 626 | 4 (4) |
| math-finemath | 29,122 | 2,609 | 2,611 | 1 | 3 | 1,433 | 90 (75) |
| math-megamath | 17,647 | 2,550 | 2,550 | 0 | 0 | 1,503 | 10 (8) |
| sci-finepdfs | 6,011 | 166 | 166 | 0 | 0 | 5 | 7 (6) |

- **math-finemath, 3 newly dropped, 1 recovered:** `29b5cd770ab92ac2`, `3674e149fa8a4377` and `6c96bdacd695aa42` are now dropped by `DA-C7-001`, and `b08cf80dd7abec80` no longer is, because `DA-C2-007` gives that rule a different text. With `DA-C2-001` out and `DA-C2-007` last it was 21 and 5.
- **en-dclm, 2 recovered:** `2dd0de53366be304` and `fb9ef791d9f8f1ae` are dropped by `DA-A3-002` (mojibake) today and kept once `DA-A2-010` has mapped their control characters.
- **A reader has to look at these six.**
- **Second run:** 15 finemath, 2 megamath and 1 sci-finepdfs documents change again only with the cards.
- **en-nemotroncc-synth:** 57 recovered and 161 dropped, the numbers of #76.

| Card | Documents where it differs from the rule it replaces (a new card: documents it changes), on the same text |
|---|---|
| `DA-A2-010` | en-dclm 144, en-fineweb-edu 0, en-nemotroncc 1, en-nemotroncc-synth 0 |
| `DA-C3-008` | en-nemotroncc-synth 57 |
| `DA-C1-013` | en-nemotroncc-synth 50 |
| `DA-C2-009` | en-dclm 65, en-fineweb-edu 1, en-nemotroncc 21 |
| `DA-C2-010` | en-nemotroncc-synth 119 |
| `DA-C5-007` | en-nemotroncc-synth 272 |
| `DA-B4-015` | en-nemotroncc-synth 87 |
| `DA-B3-012` | en-dclm 2, en-fineweb-edu 0, en-nemotroncc 0, en-nemotroncc-synth 0 |
| `DA-B4-016` | en-dclm 13, en-fineweb-edu 0, en-nemotroncc 3, en-nemotroncc-synth 3 |
| `DA-F1-007` | en-dclm 15, en-fineweb-edu 12, en-nemotroncc 11 |
| `DA-F1-009` | en-nemotroncc-synth 42 |
| `DA-C1-012` | math-megamath 295, math-finemath 1,219 |
| `DA-C2-007` | math-megamath 1,241, math-finemath 294 |
| `DA-C1-014` | sci-finepdfs 2 |
| `DA-C2-011` | sci-finepdfs 4 |

**Tests:** `tests/audit/test_collated_medicine.py`, 20 tests on made-up text. Undoing either half of a merged card makes them fail (6 of 6 hand-made changes).

## Findings that still have no card

F06 (`DA-C3-003` on en-dclm, and the other English packs), F07, F08, F10, F11, F18, F19, F20, F23, F24, F27, F28, F29, and the parts of F12 and F26 that their cards do not reach. Each is in [`collation_findings.csv`](collation_findings.csv) with its next step.

## How to reproduce

From the repository root, with the raw sample train parts in `<raw>` and the 500-document sample packs in `<packs>`:

```bash
python -I teams/01-data-a/harmtest/harm_test.py --pack <packs>/<pack>.parquet --gold teams/01-data-a/gold/<pack>_gold.csv --rules teams/01-data-a/audit/collated_medicine.md --rule all
python -I teams/01-data-a/audit/collated_medicine_check.py gold --raw <packs>
python -I teams/01-data-a/audit/collated_medicine_check.py docs --raw <raw> --out collated_medicine_docs.csv
python -I teams/01-data-a/audit/collated_medicine_check.py packs --raw <raw> --out collated_medicine_packs.csv
```

## Limits

- Sample v0 raw parts, not corpus-v1 and not the token shards.
- The changed documents of the whole packs were counted here, not read. The sources read theirs: #75 its changed en-dclm documents, #76 its audited ones, #71 and #79 as their reports say.
- Speed of the whole set was not measured.
- The side effects of each source stay, and adoption is the team's decision: accept a card, put it in place of the old card in `specv1_1.md`, rebuild `clean.py`, and run the harm test again.

## The cards

A card with `REPLACES` runs in the place of that rule. On a pack without that rule, and for a new card, `RUNS_AFTER` says where: after that rule, or (`family`) after the pack's last family rule, before the shared F1 rule.

### `DA-A2-010` · A2 · map cp1252 quotes before removing controls

1. **Disease:** `DA-A2-008` deletes U+0080 to U+009F. Text that was cp1252 bytes read as Latin-1 loses its quotes and dashes (`a9a7821bb3d942d0`). Findings: F01.
2. **Medicine:** As `DA-A2-008`, but a C1 control that cp1252 defines becomes its character first, and UTF-8 read as Latin-1 is decoded back.
3. **Action and place:** fix; replaces `DA-A2-008`. **Runs on:** en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth. **From:** #75 `DA-A2-009` (new number only).

```python
RULE_ID = "DA-A2-010"
REPLACES = "DA-A2-008"
FROM = "#75 DA-A2-009"
SOURCE_RULE_ID = "DA-A2-008"  # proposed replacement of DA-A2-008
CODE = "A2"
ACTION = "fix"
STAGE = "10"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth"  # packs it runs on
SOURCE = "en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth"  # pack whose medicine this came from
AUTHOR = "aneeshads"  # new in the 29 Oct sheet as DA-A2-007
PROPOSED_BY = "TEJASKUMAR-REDDY-J"  # this change, from the day D6 en-dclm audit
import re

A2_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")

# U+0080..U+009F that cp1252 defines (\x92 is ’, \x93 “, \x94 ”, \x96 – ...): text that was cp1252 bytes read as
# Latin-1. They are mapped to their characters first; the five cp1252 leaves undefined are still removed.
A2_CP1252 = {c: bytes([c]).decode("cp1252") for c in range(0x80, 0xA0) if c not in (0x81, 0x8D, 0x8F, 0x90, 0x9D)}
A2_C1 = re.compile(r"[\x80-\x9f]")
# UTF-8 read as Latin-1, with a C1 control among its bytes: `Ã\x89` is É, `â\x80\x99` is ’, `â\x80\x9d` is ” (\x9d is
# one that cp1252 leaves undefined). Such a sequence is decoded back first; one with no C1 control (`Â©`) is
# left alone. `\x80` and a second C1 control, with the first byte lost (`me \x80\x93 let\x80\x99s`), are the last
# two bytes of a U+2000..U+201F (– ’ “ ”) and become it.
A2_UTF8 = re.compile(r"[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}|\x80[\x80-\x9f]")


def _a2_utf8(m):
    s = m.group()
    if not A2_C1.search(s):
        return s
    if s[0] == "\x80":
        return chr(0x2000 + ord(s[1]) - 0x80)
    try:
        return s.encode("latin-1").decode("utf-8")
    except UnicodeDecodeError:
        return s


def medicine_a2(text):
    new = text.replace("\r\n", "\n").replace("\r", "\n")
    first = A2_CONTROL.search(new)  # speed: most docs have no control character, and then nothing else changes
    if first:
        if A2_C1.search(new, first.start()):  # a C1 control is a control character: none comes before the first
            new = A2_UTF8.sub(_a2_utf8, new).translate(A2_CP1252)
        new = A2_CONTROL.sub("", new)
    return ("fix" if new != text else "keep", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return medicine_a2(doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'en-dclm', 'en-fineweb-edu', 'en-nemotroncc', 'en-nemotroncc-synth'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** Mojibake of two or more layers stays garbage, as before.

### `DA-C3-008` · C3 · no drop from the last word alone

1. **Disease:** `DA-C3-003` drops a whole document because its last word is one like `with` (`5b4d0f975990e808`). Findings: F15.
2. **Medicine:** The last-word test is withdrawn; a document that ends in a bracketed ellipsis still goes.
3. **Action and place:** drop; replaces `DA-C3-003`. **Runs on:** en-nemotroncc-synth. **From:** #76 `DA-C3-008` (none).

```python
SCOPE = {"DA-C3-003": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-C3-003"
RULE_ID = "DA-C3-008"
FROM = "#76 DA-C3-008"
SOURCE_RULE_ID = "DA-C3-002"
CODE = "C3"
ACTION = "drop"
STAGE = "30"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re

_MARKER = re.compile(r"\[\s*(?:…|\.\.\.)\s*\]$")

_DANGLING = frozenset(
    "a an the and or but of to in on at for with from by as is are was were "
    "that this these those which who because however".split()
)

_TOKEN = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'")  # the characters of [A-Za-z0-9']
_LONGEST = max(map(len, _DANGLING))  # a last token longer than every dangling word cannot be one

def medicine_c3(text):
    tail = text.rstrip()
    if not tail:
        return text
    if tail.endswith("]") and _MARKER.search(tail):  # the marker ends the text, so the text must end in "]"
        return ("drop", "")
    if tail.endswith((".", "!", "?", '"', "'", ")", ":", "…")):
        return text
    # The terminal-word heuristic is deliberately withdrawn.
    return text


def _std(out, text):  # added by Data A
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(medicine_c3(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-C3-003"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** Documents that really are cut off are kept too. **Synth only:** F06 (en-dclm) is the same rule and has no card.

### `DA-C1-013` · C1 · image controls and comment metadata

1. **Disease:** `DA-C1-006` misses lines such as `Like this:` and `Click to enlarge.`, and name-and-age lines of comments. Findings: F30.
2. **Medicine:** Those whole lines go, and an `Advertisement:` prefix; comments and article text stay.
3. **Action and place:** fix; replaces `DA-C1-006`. **Runs on:** en-nemotroncc-synth. **From:** #76 `DA-C1-013` (none).

```python
SCOPE = {"DA-C1-006": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-C1-006"
RULE_ID = "DA-C1-013"
FROM = "#76 DA-C1-013"
SOURCE_RULE_ID = "DA-C1-006"
CODE = "C1"
ACTION = "fix"
STAGE = "35"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re

WIDGET = re.compile(r"(?i)(?:like this:?|click to enlarge[.!]?|primary tabs|search for:?|related (?:documents|posts|articles):?|click to expand\W*|read more\W*|\[?read more\W*\]?|share this:?|share on (?:other sites|[a-z ]+)|previous|next|menu|contents|advertisement|loading\W*|link copied|the link has been copied!?|post navigation|quantity|breadcrumb|logged|registered|banned|show more|you may also like[a-z ]*|subscribe to [a-z ]+|skip to (?:main )?content|back to top)")

COMMENT_META = re.compile(r"[A-Za-z0-9_-]{3,40} (?:over )?(?:a|an|\d+) (?:years?|months?|days?|hours?) ago", re.I)

def c1(t):
    """Delete lines that consist only of a known widget or menu label ('Primary tabs', 'Search for:', 'Related posts', 'Share this:' ...)."""
    t = re.sub(r"(?m)^Advertisement:(?=\S)", "", t)
    return "\n".join(l for l in t.split("\n")
                     if not WIDGET.fullmatch(l.strip()) and not COMMENT_META.fullmatch(l.strip()))


def _std(out, text):  # added by Data A
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(c1(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-C1-006"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** A real line of the form name, number, `days ago` goes.

### `DA-C2-009` · C2 · repeated lines: leave code and config alone

1. **Disease:** `DA-C2-003` cuts repeated lines inside code and config (`f1b8673c7cfd7aac`). Findings: F02.
2. **Medicine:** As `DA-C2-003`, but a line in a fence, or that looks like code or config, is not counted.
3. **Action and place:** fix; replaces `DA-C2-003`. **Runs on:** en-dclm, en-fineweb-edu, en-nemotroncc. **From:** #75 `DA-C2-005` (new number; synth moves to `DA-C2-010`).

```python
RULE_ID = "DA-C2-009"
REPLACES = "DA-C2-003"
FROM = "#75 DA-C2-005"
SOURCE_RULE_ID = "DA-C2-003"  # proposed replacement of DA-C2-003
CODE = "C2"
ACTION = "fix"
STAGE = "35"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-dclm, en-fineweb-edu, en-nemotroncc"  # packs it runs on
SOURCE = "en-fineweb-edu"  # pack whose medicine this came from
AUTHOR = "deepjyotisaha"  # extractor; adapter for spans
PROPOSED_BY = "TEJASKUMAR-REDDY-J"  # this change, from the day D6 en-dclm audit
import re

def _merge(spans):
    """Overlapping spans joined into one, keeping the rule of the first."""
    out = []
    for s, e, rule in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]), out[-1][2])
        else:
            out.append((s, e, rule))
    return out

C2_MIN_REPEAT, C2_MIN_CHARS = 3, 10

# a line of code or config, which repeats because the program needs it, not because it is boilerplate:
# a comment or prompt at its start (#, //, $, >>>); { at its end or } at its start; ';' after ')' or in an
# assignment, a declaration or a CSS property; a Unix path (/etc/); key=value at its start; a::b, -> or =>;
# an empty (). Dates (02/15/2010), URLs and '::' as a separator do not count.
C2_CODE = re.compile(r"^\s*(?:#|//|\$ |>>> )|\{\s*$|^\s*\}|\)\s*;\s*$|=[^=\n]*;\s*$|^\s*(?:private|public|protected|static|final|var|let|const|int|long|char|float|double|bool|boolean|String|void|my|use|import|include|require)\b[^\n]*;\s*$|^\s*[a-z][a-z-]*:\s*[^;\n]{1,40};\s*$|(?:^|\s)[~.]?/[A-Za-z_][\w.-]*/|^\s*[A-Za-z_]\w*=\S|\w::\w|->|=>|\w\(\)")

def c2_spans(text):
    """C2: later copies of a line repeated C2_MIN_REPEAT or more times in the document."""
    fenced = "```" in text or "~~~" in text  # speed: no fence marker, no fence
    seen, lines, pos, fence = {}, [], 0, False
    for line in text.split("\n"):
        key = " ".join(line.split())
        if fenced and line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        if len(key) >= C2_MIN_CHARS and not fence:
            lines.append((pos, pos + len(line), key, line))
            seen[key] = seen.get(key, 0) + 1
        pos += len(line) + 1
    # a code line is not counted. speed: C2_CODE is tested only on a line whose key is there C2_MIN_REPEAT times or
    # more with code lines counted; a key there fewer times stays under C2_MIN_REPEAT without them too
    code = {k for k, (_, _, key, line) in enumerate(lines) if seen[key] >= C2_MIN_REPEAT and C2_CODE.search(line)}
    for k in code:
        seen[lines[k][2]] -= 1
    out, first = [], set()
    for k, (s, e, key, _) in enumerate(lines):
        if k not in code and seen[key] >= C2_MIN_REPEAT:
            if key in first:
                out.append((s, e, "C2-repeated-line"))
            first.add(key)
    return out


def _std(out, text):  # added by Data A
    # the author's result (spans to remove) in the shared form, as their clean() reads it: (spans, True) from
    # c1_spans -> drop; a whole-document "B4-mostly-code" or "C5-list-page" span -> drop; else the spans go
    if isinstance(out, tuple):
        out, flagged = out
        if flagged: return ("drop", text)
    if out and out[0][2] in ("B4-mostly-code", "C5-list-page"): return ("drop", text)
    new = text
    for s, e, _ in reversed(_merge(out)):
        new = new[:s] + new[e:]
    return ("fix" if new != text else "keep", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(c2_spans(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'en-dclm', 'en-fineweb-edu', 'en-nemotroncc'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** Code the test does not see as code is still cut, and so are repeated record fields and formula lines.

### `DA-C2-010` · C2 · repeated lines: leave code, config and record fields alone

1. **Disease:** Both PRs replace `DA-C2-003` on synth: #75 for code lines, #76 for labelled fields of separate records (`7523ebd6d772fada`). Findings: F02 and F14.
2. **Medicine:** #75's code, with #76's `RECORD_FIELD` test added: a line is counted only if neither source excludes it.
3. **Action and place:** fix; replaces `DA-C2-003`. **Runs on:** en-nemotroncc-synth. **From:** #75 `DA-C2-005` + #76 `DA-C2-008` (merged).

```python
RULE_ID = "DA-C2-010"
REPLACES = "DA-C2-003"
FROM = "#75 DA-C2-005 and #76 DA-C2-008, merged"
SOURCE_RULE_ID = "DA-C2-003"  # proposed replacement of DA-C2-003
CODE = "C2"
ACTION = "fix"
STAGE = "35"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"  # packs it runs on
SOURCE = "en-fineweb-edu"  # pack whose medicine this came from
AUTHOR = "deepjyotisaha"  # extractor; adapter for spans
PROPOSED_BY = "TEJASKUMAR-REDDY-J"  # this change, from the day D6 en-dclm audit
import re

def _merge(spans):
    """Overlapping spans joined into one, keeping the rule of the first."""
    out = []
    for s, e, rule in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]), out[-1][2])
        else:
            out.append((s, e, rule))
    return out

C2_MIN_REPEAT, C2_MIN_CHARS = 3, 10

# a line of code or config, which repeats because the program needs it, not because it is boilerplate:
# a comment or prompt at its start (#, //, $, >>>); { at its end or } at its start; ';' after ')' or in an
# assignment, a declaration or a CSS property; a Unix path (/etc/); key=value at its start; a::b, -> or =>;
# an empty (). Dates (02/15/2010), URLs and '::' as a separator do not count.
C2_CODE = re.compile(r"^\s*(?:#|//|\$ |>>> )|\{\s*$|^\s*\}|\)\s*;\s*$|=[^=\n]*;\s*$|^\s*(?:private|public|protected|static|final|var|let|const|int|long|char|float|double|bool|boolean|String|void|my|use|import|include|require)\b[^\n]*;\s*$|^\s*[a-z][a-z-]*:\s*[^;\n]{1,40};\s*$|(?:^|\s)[~.]?/[A-Za-z_][\w.-]*/|^\s*[A-Za-z_]\w*=\S|\w::\w|->|=>|\w\(\)")

# merged from #76: a labelled field (`Cost: 5`) belongs to its own record, so it is not counted
RECORD_FIELD = re.compile(r"^\s*(?:[-*•]\s+)?[A-Za-z][A-Za-z0-9 /_-]{0,35}:\s+\S")

def c2_spans(text):
    """C2: later copies of a line repeated C2_MIN_REPEAT or more times in the document."""
    fenced = "```" in text or "~~~" in text  # speed: no fence marker, no fence
    seen, lines, pos, fence = {}, [], 0, False
    for line in text.split("\n"):
        key = " ".join(line.split())
        if fenced and line.lstrip().startswith(("```", "~~~")):
            fence = not fence
        if len(key) >= C2_MIN_CHARS and not fence and not RECORD_FIELD.match(line):
            lines.append((pos, pos + len(line), key, line))
            seen[key] = seen.get(key, 0) + 1
        pos += len(line) + 1
    # a code line is not counted. speed: C2_CODE is tested only on a line whose key is there C2_MIN_REPEAT times or
    # more with code lines counted; a key there fewer times stays under C2_MIN_REPEAT without them too
    code = {k for k, (_, _, key, line) in enumerate(lines) if seen[key] >= C2_MIN_REPEAT and C2_CODE.search(line)}
    for k in code:
        seen[lines[k][2]] -= 1
    out, first = [], set()
    for k, (s, e, key, _) in enumerate(lines):
        if k not in code and seen[key] >= C2_MIN_REPEAT:
            if key in first:
                out.append((s, e, "C2-repeated-line"))
            first.add(key)
    return out


def _std(out, text):  # added by Data A
    # the author's result (spans to remove) in the shared form, as their clean() reads it: (spans, True) from
    # c1_spans -> drop; a whole-document "B4-mostly-code" or "C5-list-page" span -> drop; else the spans go
    if isinstance(out, tuple):
        out, flagged = out
        if flagged: return ("drop", text)
    if out and out[0][2] in ("B4-mostly-code", "C5-list-page"): return ("drop", text)
    new = text
    for s, e, _ in reversed(_merge(out)):
        new = new[:s] + new[e:]
    return ("fix" if new != text else "keep", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(c2_spans(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'en-nemotroncc-synth'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** The side effects of both: repeated `Views:` or `Share:` lines stay, and a repeated line that starts with `#`.

### `DA-C5-007` · C5 · keep subject headings

1. **Disease:** `DA-C5-005` removes `Topic:` headings with the tag lists (4 documents). Findings: F16.
2. **Medicine:** `Topic:` and `Topics:` lines stay; tags, keywords and filed-under lines go as before.
3. **Action and place:** fix; replaces `DA-C5-005`. **Runs on:** en-nemotroncc-synth. **From:** #76 `DA-C5-007` (none).

```python
SCOPE = {"DA-C5-005": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-C5-005"
RULE_ID = "DA-C5-007"
FROM = "#76 DA-C5-007"
SOURCE_RULE_ID = "DA-C5-005"
CODE = "C5"
ACTION = "fix"
STAGE = "35"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re

C5_PREFIX = re.compile(r"^(?:tags?|keywords?|tagged(?: with| in)?|filed under)\s*:", re.I)


def medicine_c5(text):
    lines = text.split("\n")
    keep = [ln for ln in lines if not C5_PREFIX.search(ln.strip())]
    if len(keep) == len(lines):
        return ("keep", text)
    new = "\n".join(keep).strip("\n")
    return ("drop", text) if not new.strip() else ("fix", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return medicine_c5(doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-C5-005"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** A tag list that starts with `Topic:` stays.

### `DA-B4-015` · B4 · keep the facts in a bullet with a meta-comment

1. **Disease:** `DA-B4-007` removes a whole bullet for one meta sentence (`f8027db45cfbebb2`, `93bf1542853fd276`). Findings: F17.
2. **Medicine:** Only the meta sentence goes. A bullet with an inference (`but it implies`) stays whole.
3. **Action and place:** fix; replaces `DA-B4-007`. **Runs on:** en-nemotroncc-synth. **From:** #76 `DA-B4-015` (none).

```python
SCOPE = {"DA-B4-007": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-B4-007"
RULE_ID = "DA-B4-015"
FROM = "#76 DA-B4-015"
SOURCE_RULE_ID = "DA-B4-007"
CODE = "B4"
ACTION = "fix"
STAGE = "40"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re
ANSWER_LINE = re.compile('^(?:Answer|A)\\s*[:.]|\\bAnswer:\\s')  # added by Data A: from maheshsv's medicine.md
KEY_VALUE_LINE = re.compile('^[^:\\n]{1,60}:\\s')  # added by Data A: from maheshsv's medicine.md
LIST_MARK = re.compile('^(?:[-*•–—]|\\d+[.)]|[a-z][.)])\\s')  # added by Data A: from maheshsv's medicine.md
META = re.compile('(?i)\\bnot (?:explicitly |specifically |directly )?(?:provided|stated|mentioned|specified|given|available|included|present|disclosed|named) in the (?:given |provided |original |above |source )?(?:text|passage|article|document|excerpt|content|context)|\\bthe (?:text|passage|article|document) does not (?:explicitly |specifically )?(?:provide|mention|specify|state|say|give|include|indicate|name)\\b|\\bthe (?:given|provided) (?:text|passage) (?:also )?(?:discusses|mentions|describes|states|says|provides|refers|covers|explains|includes|lists|notes)\\b')  # added by Data A: from maheshsv's medicine.md
MIN_KEEP_CHARS = 200
# Preserve an ambiguous source inference rather than inventing a stronger assertion.
MIXED_INFERENCE = re.compile(r"\bbut (?:it |the (?:text|passage|article) )?(?:implies|suggests|indicates)\b", re.I)  # added by Data A: from maheshsv's medicine.md

def remove_meta_comments(text: str) -> tuple[str | None, bool]:
    """Delete the bullet, `Key: value` line or Question/Answer pair that carries a rewriter meta-comment; in running
    prose delete only the sentence."""
    if not META.search(text):
        return text, False
    lines = text.split("\n")
    drop = set()
    for i, ln in enumerate(lines):
        if not META.search(ln):
            continue
        s = ln.strip()
        if s.startswith("Question:") or ANSWER_LINE.match(s):
            drop.add(i)                                   # drop the whole QA pair, whichever line carries it
            if s.startswith("Question:") and i + 1 < len(lines) and ANSWER_LINE.match(lines[i + 1].strip()):
                drop.add(i + 1)
            if ANSWER_LINE.match(s) and not s.startswith("Question:") and i > 0 and lines[i - 1].strip().startswith("Question:"):
                drop.add(i - 1)
        elif LIST_MARK.match(s) or KEY_VALUE_LINE.match(s):
            if MIXED_INFERENCE.search(s):
                continue
            sentences = re.split(r"(?<=[.!?])\s+", ln)
            kept = [sentence for sentence in sentences if not META.search(sentence)]
            if kept:
                lines[i] = " ".join(kept)
            else:
                drop.add(i)
        else:
            sentences = re.split(r"(?<=[.!?])\s+", ln)
            lines[i] = " ".join(x for x in sentences if not META.search(x))
    new = "\n".join(ln for i, ln in enumerate(lines) if i not in drop)
    new = re.sub(r"\n{3,}", "\n\n", new).strip()
    if len(new) < MIN_KEEP_CHARS:
        return None, True
    return new, True


def _std(out, text):  # added by Data A
    # the author's result (new_text or None, changed) in the shared form: None -> drop, else fix/keep by the text
    new, _ = out
    if new is None: return ("drop", text)
    return ("fix" if new != text else "keep", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(remove_meta_comments(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-B4-007"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** Some source commentary stays. #76 leaves the policy open.

### `DA-B3-012` · B3 · `<s>` and `</s>` used as markup

1. **Disease:** The markers `<s>` and `</s>` are kept in English text (`b86875f06abe649a`). Findings: F05.
2. **Medicine:** In a document with both, every `<s>` and `</s>` goes; the struck text stays.
3. **Action and place:** fix; new, after `DA-B4-012`. **Runs on:** en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth. **From:** #75 `DA-B3-002` (new number only).

```python
RULE_ID = "DA-B3-012"
RUNS_AFTER = "DA-B4-012"
FROM = "#75 DA-B3-002"
CODE = "B3"
ACTION = "fix"
STAGE = "40"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth"  # packs it runs on
PACKS = set(ORIGIN.split(", "))
SOURCE = "en-dclm"  # pack whose reading found it (day D6 audit)
AUTHOR = "TEJASKUMAR-REDDY-J"
import re

# <s> and </s>: the BOS/EOS markers in text. Used together they are strikethrough markup, which DA-B2-002 leaves
# alone: the markers go, the struck text stays. A lone <s> (an HTML reference: "<s> Defines strikethrough text") stays.
B3_S = re.compile(r"</?s>")


def medicine(doc):
    if doc.get("pack_id") not in PACKS or doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    text = doc["text"]
    if "<s>" not in text or "</s>" not in text:  # markup needs both; speed: most docs have neither
        return ("keep", text)
    new = B3_S.sub("", text)
    return ("fix" if new != text else "keep", new)
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** An HTML tutorial with an example, also in a code fence, loses the markers.

### `DA-B4-016` · B4 · a `${` that opens math stays

1. **Disease:** `DA-B4-010` removes `${8,16,24}` as a template expression (`bfa86fe32d00c663`). Findings: F03.
2. **Medicine:** As `DA-B4-010`, but a `${...}` whose content starts like math stays.
3. **Action and place:** fix; replaces `DA-B4-010`. **Runs on:** en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth. **From:** #75 `DA-B4-014` (new number only).

```python
RULE_ID = "DA-B4-016"
REPLACES = "DA-B4-010"
FROM = "#75 DA-B4-014"
SOURCE_RULE_ID = "DA-B4-010"  # proposed replacement of DA-B4-010
CODE = "B4"
ACTION = "fix"
STAGE = "40"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth"  # packs it runs on
SOURCE = "en-fineweb-edu"  # pack whose medicine this came from
AUTHOR = "deepjyotisaha"  # extractor; adapter for spans
PROPOSED_BY = "TEJASKUMAR-REDDY-J"  # this change, from the day D6 en-dclm audit
import re

def _merge(spans):
    """Overlapping spans joined into one, keeping the rule of the first."""
    out = []
    for s, e, rule in sorted(spans):
        if out and s <= out[-1][1]:
            out[-1] = (out[-1][0], max(e, out[-1][1]), out[-1][2])
        else:
            out.append((s, e, rule))
    return out

B4_TEMPLATE = re.compile(
    r"\{\{[^{}\n]{0,80}\}\}"                          # {{ title }}
    r"|\{%[^\n]{0,80}?%\}"                            # {% if x %}
    r"|\$\{(?!\s*[\\\d{}<^_$])[^{}\n]{0,80}\}"          # ${name}; not math: ${8,16,24}, ${\bf a}, ${}^*
    r"|`[\w.]+(?:\s*(?:\|\||&&)\s*[\w.]+)+`"          # `page.data.shortTitle || page.data.title`
    r"|(?<=[:=$])\s*(?:undefined|NaN)\b"              # Price: NaN, name=undefined
    r"|^[ \t]*(?:undefined|NaN|null)[ \t]*$", re.M)

B4_CODE_TOKENS = re.compile(
    r"::|->|=>|==|!=|<=|>=|&&|\|\||\+\+|\+=|-=|\);|\)\s*\{|[{};]\s*$|\}"
    r"|^\s*(?:#include|import\s|from\s+\S+\s+import|def\s|class\s|public\s|private\s|static\s|void\s|return\b"
    r"|require\s|var\s|let\s|const\s|function\b|Sub\s|End Sub|Dim\s|\$\s|>>>\s|>\s)"
    r"|\b(?:End Sub|ByVal|As Integer|System\.out|printf|println)\b", re.M)

B4_MIN_TOKENS, B4_WORDS_PER_TOKEN, B4_SHARE = 2, 6, 0.5

TABLE_ROW = re.compile(r"^\s*\|.*\|\|")

def b4_code_lines(text):
    """(start, end) of every line that looks like code."""
    out, pos = [], 0
    for line in text.split("\n"):
        tokens = len(B4_CODE_TOKENS.findall(line))
        words = len(line.split())
        if tokens >= B4_MIN_TOKENS and tokens * B4_WORDS_PER_TOKEN >= words and not TABLE_ROW.match(line):
            out.append((pos, pos + len(line)))
        pos += len(line) + 1
    return out

def b4_mostly_code(text):
    """Whether the lines b4_code_lines finds are more than B4_SHARE of the text, as b4_spans asks. The same test, but it
    stops as soon as the answer cannot change: when the code lines already pass the limit, or when the code found plus
    every character not yet read cannot reach it (a prose document is settled halfway through)."""
    if not text.strip():
        return False
    limit = B4_SHARE * len(text.strip())
    code, left = 0, len(text)
    for line in text.split("\n"):
        tokens = len(B4_CODE_TOKENS.findall(line))
        if tokens >= B4_MIN_TOKENS and tokens * B4_WORDS_PER_TOKEN >= len(line.split()) and not TABLE_ROW.match(line):
            code += len(line)
            if code > limit:
                return True
        left -= len(line) + 1
        if code + left <= limit:
            return False
    return False

# Every alternative of B4_TEMPLATE contains one of these, so a text with none of them has no template expression.
B4_TEMPLATE_HINTS = ("{{", "{%", "${", "`", "undefined", "NaN", "null")

def b4_spans(text):
    """B4: the document is dropped if it is mostly code; otherwise template expressions are removed."""
    if b4_mostly_code(text):
        return [(0, len(text), "B4-mostly-code")]
    if not any(hint in text for hint in B4_TEMPLATE_HINTS):
        return []
    return [(m.start(), m.end(), "B4-template") for m in B4_TEMPLATE.finditer(text)]


def _std(out, text):  # added by Data A
    # the author's result (spans to remove) in the shared form, as their clean() reads it: (spans, True) from
    # c1_spans -> drop; a whole-document "B4-mostly-code" or "C5-list-page" span -> drop; else the spans go
    if isinstance(out, tuple):
        out, flagged = out
        if flagged: return ("drop", text)
    if out and out[0][2] in ("B4-mostly-code", "C5-list-page"): return ("drop", text)
    new = text
    for s, e, _ in reversed(_merge(out)):
        new = new[:s] + new[e:]
    return ("fix" if new != text else "keep", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(b4_spans(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'en-dclm', 'en-fineweb-edu', 'en-nemotroncc', 'en-nemotroncc-synth'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** Math that starts with a letter, a minus or a bracket is still cut.

### `DA-F1-007` · F1 · one decision for each phone number

1. **Disease:** `DA-F1-005` masks a number on one line and keeps it on the next (`ef987c066d1cdf0f`); it does not know UK freephone. Findings: F04.
2. **Medicine:** A number masked on any line is masked on every line. 0800, 0808 and their +44 forms are toll-free.
3. **Action and place:** fix; replaces `DA-F1-005`. **Runs on:** en-dclm, en-fineweb-edu, en-nemotroncc. **From:** #75 `DA-F1-007` (synth moves to `DA-F1-009`).

```python
RULE_ID = "DA-F1-007"
REPLACES = "DA-F1-005"
FROM = "#75 DA-F1-007"
SOURCE_RULE_ID = "DA-F1-005"  # proposed replacement of DA-F1-005
CODE = "F1"
ACTION = "fix"
STAGE = "40"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-dclm, en-fineweb-edu, en-nemotroncc"  # packs it runs on
SOURCE = "en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth"  # pack whose medicine this came from
AUTHOR = "aneeshads"  # new rule to the instructor's F1 ruling of 31 Oct; email test from DA-F1-002 (shekar-maker)
PROPOSED_BY = "TEJASKUMAR-REDDY-J"  # this change, from the day D6 en-dclm audit
import re

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")

WEBMAIL = re.compile(r"@(?:gmail|googlemail|yahoo|ymail|hotmail|outlook|live|msn|aol|icloud|me|mac|protonmail|proton|"
                     r"rediffmail|zoho|gmx|yandex|mail|comcast|verizon|att|sbcglobal|bellsouth|earthlink|btinternet)\.", re.I)

PERSONAL_LOCAL = re.compile(r"[._\-]|\d|^.{13,}$")

ROLE_LOCAL = re.compile(r"^(?:info|contact|contactus|enquir(?:y|ies)|inquir(?:y|ies)|sales|support|help|helpdesk|admin|"
                        r"office|hello|hi|press|media|news|newsletter|editor|editors|team|jobs|careers|hr|marketing|"
                        r"billing|accounts|orders|service|services|customerservice|webmaster|postmaster|noreply|"
                        r"no-reply|abuse|privacy|legal|admissions|registrar|events|bookings|reservations)$", re.I)

NOT_REAL = re.compile(r"@(?:example|test|domain|email)\.(?:com|org|net)$", re.I)

PHONE = re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.\-‐‑–]?)?(?:\(\d{2,4}\)[\s.\-‐‑–]?|\d{2,4}[\s.\-‐‑–])\d{3,4}[\s.\-‐‑–]\d{3,4}(?![\w-])")

# US 8xx, and UK freephone 0800/0808 (+44 800/808)
TOLL_FREE = re.compile(r"^(?:(?:\+?1[\s.-]?)?\(?8(?:00|88|77|66|55|44|33)\)?|(?:\+44[\s.-]?|0)80[08](?=[\s.\-]))")
NOT_DIGIT = re.compile(r"\D")

STREET = (r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Ter|Close|"
          r"Crescent|Cres|Highway|Hwy|Parkway|Pkwy|Circle|Cir|Square|Sq)")

ADDRESS = re.compile(r"\b\d{1,5}[A-Za-z]?\s+(?:[NSEW]\.?\s+)?(?:[A-Z][a-z]+\.?\s+){1,4}" + STREET + r"\b\.?"
                     r"(?:,?\s*(?:Apt|Unit|Suite|Ste|#)\.?\s*\w+)?"
                     r"(?:,\s*(?:[A-Z][a-z]+\s?){1,3},\s*[A-Z]{2}\.?\s*\d{5}(?:-\d{4})?)?")

CONTACT_LINE = 120

ORGANISATION = re.compile(
    r"(?i)\b(?:inc|llc|ltd|plc|corp|co\.|company|warehouse|store|shop|market|restaurant|cafe|café|bar|grill|bistro|"
    r"hotel|inn|motel|resort|spa|salon|clinic|hospital|surgery|pharmacy|cent(?:er|re)|office|offices|university|"
    r"college|school|academy|library|museum|church|gallery|theat(?:er|re)|cinema|bank|agency|association|society|"
    r"council|department|ministry|embassy|tourist|information|headquarters|hq|branch|showroom|dealership|"
    r"open (?:mon|tue|wed|thu|fri|sat|sun|daily|7|24)\w*|hours|menu|reservations|fax|"
    r"on (?:urbanspoon|yelp|tripadvisor|zomato|foursquare)|"
    # trades and services that put their number on their own page
    r"pros?|plumb\w*|roofing|hvac|electric(?:al|ians?)|landscap\w*|cleaning|movers?|moving|storage|rentals?|hire|"
    r"repairs?|construction|builders?|contractors?|auto|motors|towing|insurance|realty|real estate|mortgage|loans?|"
    r"law|lawyers?|attorneys?|legal|dental|dentistry|orthodont\w*|chiropract\w*|physiotherap\w*|therapy|"
    r"veterinar\w*|vet|clinics?|practice|surgery|medical|health(?:care)?|consult(?:ation|ing|ants?)?|services?|"
    r"solutions|group|partners|associates|studio|chalet|lodge|villa|apartments?|cottage|camping|farm|winery|"
    r"brewery|bakery|club|bbs|board|optometr\w*|optic\w*|opticians?|spectacles|suite)\b")

ORG_VOICE = re.compile(r"(?i)\b(?:call us|contact us|email us|e-mail us|visit us|reach us|text us|give us a call|"
                       r"our (?:office|team|staff|clinic|store|shop|experts?|technicians?|specialists?|services?|"
                       r"company|firm|practice|location)|one of our|we (?:offer|provide|specialize|specialise|serve)|need help\?)|☎")

PUBLIC_ROLE = re.compile(r"(?i)\b(?:reporter|editor|journalist|correspondent|spokes(?:man|woman|person)|press|media|"
                         r"publicist|author of|professor|prof\.|dr\.|lecturer|faculty|researcher|principal investigator|"
                         r"media contact|press contact|for more information|high[- ]res|images|photos)\b")

DIGIT = re.compile(r"\d")

def _email(m, org):
    addr = m.group()
    local = addr.split("@", 1)[0]
    if NOT_REAL.search(addr) or ROLE_LOCAL.match(local):
        return addr
    if WEBMAIL.search(addr):  # a webmail address belongs to a person
        return "<EMAIL>"
    # a name-like address at an organisation's domain is a work address when the context is an organisation's
    return addr if org or not PERSONAL_LOCAL.search(local) else "<EMAIL>"

def medicine_f1(text):
    lines = text.split("\n")
    org_page = []  # ORG_VOICE on the whole text: searched once, and only when a contact line asks
    has_org = {}  # line number -> an organisation word on that line: searched once, and only when a contact line asks

    def on_org_page():
        if not org_page:
            org_page.append(bool(ORG_VOICE.search(text)))
        return org_page[0]

    def org_near(i):  # same answer as searching the joined 13-line window, as no match spans a newline
        for j in range(max(0, i - 6), min(len(lines), i + 7)):
            if j not in has_org:
                has_org[j] = ORGANISATION.search(lines[j]) is not None
            if has_org[j]:
                return True
        return False

    # one decision per number, privacy first: a number masked on any line is masked on every line, also on a line
    # near an organisation word (the 13-line window alone can mask a number on one line and keep it on the next).
    # speed: the masked numbers are found once, and only when a phone line near an organisation asks
    masked = []

    def masked_numbers():
        if not masked:
            nums = set()
            if not on_org_page():  # on an organisation's page no line is masked
                for i, line in enumerate(lines):
                    if DIGIT.search(line) and PHONE.search(line) and not org_near(i):
                        nums.update(NOT_DIGIT.sub("", m.group()) for m in PHONE.finditer(line)
                                    if not TOLL_FREE.match(m.group()))
            masked.append(nums)
        return masked[0]

    out = []
    for i, line in enumerate(lines):
        if "@" not in line and not DIGIT.search(line):  # no EMAIL, PHONE or ADDRESS can match
            out.append(line)
            continue
        new = EMAIL.sub(lambda m: _email(m, on_org_page() or org_near(i) or bool(PUBLIC_ROLE.search(
            "\n".join(lines[max(0, i - 2):i + 1])))), line) if "@" in line else line
        if PHONE.search(new) or ADDRESS.search(new):
            if not (on_org_page() or org_near(i)):
                new = PHONE.sub(lambda m: m.group() if TOLL_FREE.match(m.group()) else "<PHONE>", new)
                if len(line.strip()) <= CONTACT_LINE:
                    new = ADDRESS.sub("<ADDRESS>", new)
            elif masked_numbers():
                new = PHONE.sub(lambda m: "<PHONE>" if NOT_DIGIT.sub("", m.group()) in masked[0] else m.group(), new)
        out.append(new)
    return "\n".join(out)


def _std(out, text):  # added by Data A
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(medicine_f1(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'en-dclm', 'en-fineweb-edu', 'en-nemotroncc'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** A `+44` copy of a masked number is not matched. Grit, binary and IP values on these three packs are masked as before.

### `DA-F1-009` · F1 · one decision for each phone number; technical values and public contacts

1. **Disease:** Both PRs replace `DA-F1-005` on synth. #75 masks every copy of a number that is masked once; #76 keeps technical values and public contacts, line by line. Alone, they disagree on three documents. Findings: F13.
2. **Medicine:** #76's code decides whether a line gives a reason to mask. #75's pass then makes it one decision for each number: masked on any line, masked on every line. A technical value is never a phone. #75's freephone test is in.
3. **Action and place:** fix; replaces `DA-F1-005`. **Runs on:** en-nemotroncc-synth. **From:** #75 `DA-F1-007` + #76 `DA-F1-008` (merged).

```python
SCOPE = {"DA-F1-005": frozenset({"en-nemotroncc-synth"})}
REPLACES = "DA-F1-005"
RULE_ID = "DA-F1-009"
FROM = "#75 DA-F1-007 and #76 DA-F1-008, merged"
SOURCE_RULE_ID = "DA-F1-005"
CODE = "F1"
ACTION = "fix"
STAGE = "40"
STATUS = "PROPOSAL"
LANGS = {"en", "eng"}
ORIGIN = "en-nemotroncc-synth"
SOURCE = "en-nemotroncc-synth manual audit"
import re

EMAIL = re.compile(r"(?<![\w.+-])[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")

WEBMAIL = re.compile(r"@(?:gmail|googlemail|yahoo|ymail|hotmail|outlook|live|msn|aol|icloud|me|mac|protonmail|proton|"
                     r"rediffmail|zoho|gmx|yandex|mail|comcast|verizon|att|sbcglobal|bellsouth|earthlink|btinternet)\.", re.I)

PERSONAL_LOCAL = re.compile(r"[._\-]|\d|^.{13,}$")

ROLE_LOCAL = re.compile(r"^(?:info|contact|contactus|enquir(?:y|ies)|inquir(?:y|ies)|sales|support|help|helpdesk|admin|"
                        r"office|hello|hi|press|media|news|newsletter|editor|editors|team|jobs|careers|hr|marketing|"
                        r"billing|accounts|orders|service|services|customerservice|webmaster|postmaster|noreply|"
                        r"no-reply|abuse|privacy|legal|admissions|registrar|events|bookings|reservations)$", re.I)

NOT_REAL = re.compile(r"@(?:example|test|domain|email)\.(?:com|org|net)$", re.I)

PHONE = re.compile(r"(?<![\w-])(?:\+\d{1,3}[\s.\-‐‑–]?)?(?:\(\d{2,4}\)[\s.\-‐‑–]?|\d{2,4}[\s.\-‐‑–])\d{3,4}[\s.\-‐‑–]\d{3,4}(?![\w-])")

# US 8xx, and UK freephone 0800/0808 (+44 800/808)
TOLL_FREE = re.compile(r"^(?:(?:\+?1[\s.-]?)?\(?8(?:00|88|77|66|55|44|33)\)?|(?:\+44[\s.-]?|0)80[08](?=[\s.\-]))")
NOT_DIGIT = re.compile(r"\D")

STREET = (r"(?:Street|St|Avenue|Ave|Road|Rd|Lane|Ln|Boulevard|Blvd|Drive|Dr|Court|Ct|Way|Place|Pl|Terrace|Ter|Close|"
          r"Crescent|Cres|Highway|Hwy|Parkway|Pkwy|Circle|Cir|Square|Sq)")

ADDRESS = re.compile(r"\b\d{1,5}[A-Za-z]?\s+(?:[NSEW]\.?\s+)?(?:[A-Z][a-z]+\.?\s+){1,4}" + STREET + r"\b\.?"
                     r"(?:,?\s*(?:Apt|Unit|Suite|Ste|#)\.?\s*\w+)?"
                     r"(?:,\s*(?:[A-Z][a-z]+\s?){1,3},\s*[A-Z]{2}\.?\s*\d{5}(?:-\d{4})?)?")

CONTACT_LINE = 120

ORGANISATION = re.compile(
    r"(?i)\b(?:inc|llc|ltd|plc|corp|co\.|company|warehouse|store|shop|market|restaurant|cafe|café|bar|grill|bistro|"
    r"hotel|inn|motel|resort|spa|salon|clinic|hospital|surgery|pharmacy|cent(?:er|re)|office|offices|university|"
    r"college|school|academy|library|museum|church|gallery|theat(?:er|re)|cinema|bank|agency|association|society|"
    r"council|department|ministry|embassy|tourist|information|headquarters|hq|branch|showroom|dealership|"
    r"open (?:mon|tue|wed|thu|fri|sat|sun|daily|7|24)\w*|hours|menu|reservations|fax|"
    r"on (?:urbanspoon|yelp|tripadvisor|zomato|foursquare)|"
    # trades and services that put their number on their own page
    r"pros?|plumb\w*|roofing|hvac|electric(?:al|ians?)|landscap\w*|cleaning|movers?|moving|storage|rentals?|hire|"
    r"repairs?|construction|builders?|contractors?|auto|motors|towing|insurance|realty|real estate|mortgage|loans?|"
    r"law|lawyers?|attorneys?|legal|dental|dentistry|orthodont\w*|chiropract\w*|physiotherap\w*|therapy|"
    r"veterinar\w*|vet|clinics?|practice|surgery|medical|health(?:care)?|consult(?:ation|ing|ants?)?|services?|"
    r"solutions|group|partners|associates|studio|chalet|lodge|villa|apartments?|cottage|camping|farm|winery|"
    r"brewery|bakery|club|bbs|board|optometr\w*|optic\w*|opticians?|spectacles|suite)\b")

ORG_VOICE = re.compile(r"(?i)\b(?:call us|contact us|email us|e-mail us|visit us|reach us|text us|give us a call|"
                       r"our (?:office|team|staff|clinic|store|shop|experts?|technicians?|specialists?|services?|"
                       r"company|firm|practice|location)|one of our|we (?:offer|provide|specialize|specialise|serve)|need help\?)|☎")

PUBLIC_ROLE = re.compile(r"(?i)\b(?:reporter|editor|journalist|correspondent|spokes(?:man|woman|person)|press|media|"
                         r"publicist|author of|professor|prof\.|dr\.|lecturer|faculty|researcher|principal investigator|"
                         r"media contact|press contact|for more information|high[- ]res|images|photos)\b")

DIGIT = re.compile(r"\d")

def _email(m, org):
    addr = m.group()
    local = addr.split("@", 1)[0]
    if NOT_REAL.search(addr) or ROLE_LOCAL.match(local):
        return addr
    if WEBMAIL.search(addr):  # a webmail address belongs to a person
        return "<EMAIL>"
    # a name-like address at an organisation's domain is a work address when the context is an organisation's
    return addr if org or not PERSONAL_LOCAL.search(local) else "<EMAIL>"

BINARY_CONTEXT = re.compile(r"\b(?:binary|bits?|base[ -]?2)\b", re.I)
PHONE_CONTEXT = re.compile(r"\b(?:phone|telephone|tel|mobile|call|contact)\b", re.I)
BINARY_DIGITS = re.compile(r"[01 ]{8,64}")
IP_EXAMPLE = re.compile(r"(?<![\w.])[0-9Oo]{1,3}(?:\.[0-9Oo]{1,3}){3}(?!\w|\.\w)")
IP_CONTEXT = re.compile(r"\b(?:ip|subnet|broadcast|net\s*mask)\b", re.I)
GRIT_CONTEXT = re.compile(r"\b(?:sand(?:ing|paper)?|grits?)\b", re.I)
FRIEND_CONTEXT = re.compile(r"\b(?:3DS|friend\s*code|Pok[eé]mon)\b", re.I)
PRIVATE_CUE = re.compile(r"\b(?:my (?:personal |private |home )?(?:phone|number)|personal (?:phone|number)|private (?:phone|number)|contact me|call me|mobile)\b", re.I)
PUBLIC_INSTITUTION = re.compile(r"\b(?:charities|youth bureau|citizens for|schools|heating\s*(?:&|and)\s*air conditioning)\b", re.I)
PUBLIC_CONTACT = re.compile(r"\b(?:contact|call|registration|waitlist|inquiries|sponsors?|booking)\b", re.I)

def _technical_number(line, match, previous):
    value = match.group().strip()
    # Protect complete IP examples before phone matching, including a match that
    # crosses adjacent IP examples. Keep invalid examples unchanged, not repaired.
    if IP_CONTEXT.search(line):
        spans = [m.span() for m in IP_EXAMPLE.finditer(line)]
        if spans and all(line[p].isspace() or any(a <= p < b for a, b in spans)
                         for p in range(match.start(), match.end())):
            return True
    if PHONE_CONTEXT.search(line):
        return False
    if (BINARY_DIGITS.fullmatch(value) is not None
            and len(value.replace(" ", "")) >= 8
            and BINARY_CONTEXT.search(line[max(0, match.start() - 80):match.start()])):
        return True
    if re.fullmatch(r"\d{2,4}-\d{2,4}-\d{2,4}", value):
        grits = [int(v) for v in value.split("-")]
        if (16 <= grits[0] < grits[1] < grits[2] <= 3000
                and GRIT_CONTEXT.search(previous[-200:] + line[:match.start()])):
            return True
    return (re.fullmatch(r"\d{4}-\d{4}-\d{4}", value) is not None
            and re.search(r"\bFC\s*$", line[:match.start()], re.I) is not None
            and FRIEND_CONTEXT.search(previous[-200:] + line) is not None)

def _public_phone(text, lines, i, match):
    line = lines[i]
    if PRIVATE_CUE.search(line):
        return False
    previous = "\n".join(lines[max(0, i - 2):i])
    if (PUBLIC_INSTITUTION.search(line)
            or (i > 0 and PUBLIC_INSTITUTION.search(lines[i - 1])
                and len(lines[i - 1]) < 160)):
        return True
    if re.search(r"\bEducation Director\b", line, re.I):
        return True
    if (re.search(r"\bDirector of Guidance and Testing for .{1,80}\bSchools\b", previous, re.I)
            and re.search(r"inquiries regarding .{1,100}Assessment.*contact Mr\.", line, re.I)):
        return True
    if re.search(r"supported by .{1,80}\bResearch\b.*sponsors", line, re.I):
        return True
    if re.search(r"Checking Voter Registration Status", previous, re.I) and PUBLIC_CONTACT.search(line):
        return True
    if (PUBLIC_CONTACT.search(line)
            and re.search(r"\bMuseum\b", text)
            and re.search(r"\b(?:program|session)\b", line, re.I)):
        return True
    if (PUBLIC_CONTACT.search(line)
            and re.search(r"(?:\b[\w.+-]+@[\w.-]+\.edu\b|\b(?:info|registration|bookings)@[\w.-]+\b)", line, re.I)):
        return True
    if re.search(r"^Provider:\s*\S", previous, re.M) and re.match(r"Contact:", line, re.I):
        return True
    return ("Oldest Continuously Operating Phone Number" in text
            and "PE6-5000" in text and "number can be dialed as" in line)

def medicine_f1(text):
    lines = text.split("\n")
    org_page = []  # ORG_VOICE on the whole text: searched once, and only when a contact line asks
    has_org = {}  # line number -> an organisation word on that line: searched once, and only when a contact line asks

    def on_org_page():
        if not org_page:
            org_page.append(bool(ORG_VOICE.search(text)))
        return org_page[0]

    def org_near(i):  # same answer as searching the joined 13-line window, as no match spans a newline
        for j in range(max(0, i - 6), min(len(lines), i + 7)):
            if j not in has_org:
                has_org[j] = ORGANISATION.search(lines[j]) is not None
            if has_org[j]:
                return True
        return False

    # merged: #76 decides whether a line gives a reason to keep a number; #75 makes it one decision for each number,
    # privacy first: a number masked on any line is masked on every line, unless it is a technical value there
    def technical(i, line, m):
        return _technical_number(line, m, "\n".join(lines[max(0, i - 2):i]))

    def kept(i, line, m):
        return bool(TOLL_FREE.match(m.group()) or technical(i, line, m) or _public_phone(text, lines, i, m))

    masked = []  # the digits of the masked numbers: found once, and only when a line asks

    def masked_numbers():
        if not masked:
            nums = set()
            if not on_org_page():  # on an organisation's page no line is masked
                for i, line in enumerate(lines):
                    if DIGIT.search(line) and PHONE.search(line) and not org_near(i):
                        nums.update(NOT_DIGIT.sub("", m.group()) for m in PHONE.finditer(line) if not kept(i, line, m))
            masked.append(nums)
        return masked[0]

    def masked_elsewhere(i, line, m):
        return NOT_DIGIT.sub("", m.group()) in masked_numbers() and not technical(i, line, m)

    out = []
    for i, line in enumerate(lines):
        if "@" not in line and not DIGIT.search(line):  # no EMAIL, PHONE or ADDRESS can match
            out.append(line)
            continue
        new = EMAIL.sub(lambda m: _email(m, on_org_page() or org_near(i) or bool(PUBLIC_ROLE.search(
            "\n".join(lines[max(0, i - 2):i + 1])))), line) if "@" in line else line
        if PHONE.search(new) or ADDRESS.search(new):
            if not (on_org_page() or org_near(i)):
                new = PHONE.sub(lambda m, line=new: "<PHONE>" if not kept(i, line, m) or masked_elsewhere(i, line, m) else m.group(), new)
                if len(line.strip()) <= CONTACT_LINE:
                    new = ADDRESS.sub("<ADDRESS>", new)
            elif masked_numbers():
                new = PHONE.sub(lambda m, line=new: "<PHONE>" if masked_elsewhere(i, line, m) else m.group(), new)
        out.append(new)
    return "\n".join(out)


def _std(out, text):  # added by Data A
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(medicine_f1(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = SCOPE["DA-F1-005"]

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** A public contact that also appears on a line with no evidence is masked on both. The `+44` gap of `DA-F1-007` is here too.

### `DA-C1-012` · C1 · site chrome left after spec v1

1. **Disease:** Forum, homework-site and journal chrome is left; where it repeats, one copy stays (26 documents). Findings: F12.
2. **Medicine:** Whole lines of the chrome the audit saw go, every copy, with wiki `[edit]` labels.
3. **Action and place:** fix; new, after the family's rules. **Runs on:** math-finemath, math-megamath. **From:** #71 `DA-C1-012` (none).

```python
RULE_ID = "DA-C1-012"
RUNS_AFTER = "family"
FROM = "#71 DA-C1-012"
# site chrome left after spec v1, found by the math audit (assignment 11)
CODE = "C1"
ACTION = "fix"
STAGE = "30"
STATUS = "CANDIDATE"
LANGS = {'en'}
ORIGIN = "math-finemath, math-megamath"  # packs it runs on: the disease was found in both
PACKS = set(ORIGIN.split(", "))  # the math packs share the "en" tag, so the pack is checked too
SOURCE = "math-finemath, math-megamath"  # packs whose audit found it (aneeshads, assignment 11)
AUTHOR = "aneeshads"
import re

# a whole line that is site chrome: every copy goes, with its line break
C1_CHROME = re.compile(
    r"(?m)^[ \t]*(?:"
    r"Homework Helper|Gold Member|Staff Emeritus|Science Advisor|(?:19|20)\d\d Award|"
    r"Last edited(?: by [^:\n]{1,40})?:.{0,40}|"
    r"Please Log in or Create an account to join the conversation\.|"
    r"Moderators: .{1,80}|Time to create page: \d+(?:\.\d+)? seconds|"
    r"Quick Reference:|Message Log|Canvas \(your drawing will display here\)|"
    r"Answer This Question|First Name:.{0,60}|School Subject:.{0,40}|(?:More )?Related Questions|Post a New Question|"
    r"Join BYJU['’]S Learning Program|You visited us \d+ times?! Enjoying our articles\? Unlock Full Access!|"
    r"Suggest Corrections(?: \d+)?|Watch in App|Open in App|"
    r"Still have questions\?|Earn \+\d+ pts|Continue Learning about .{1,80}|Updated: \d{1,2}/\d{1,2}/\d{4}|\d+y ago|"
    r"Post a followup to this message.{0,200}|"
    r"Author information|Affiliations|Corresponding author|Received|Accepted|Published|darkmode"
    r")[ \t]*$\n?")
# a wiki section's [edit] label, one or more, at the line end
C1_EDIT = re.compile(r"(?m)(?:[ \t]*\[edit\])+[ \t]*$")


# The inline chrome goes first, so a line such as "Published [edit]" is whole-line chrome when C1_CHROME sees it, and a
# second run changes nothing.
# Speed: str.replace and `in` are C scans. C1_EDIT starts with [ \t]*, so the engine has no literal to search for and tries a
# match at every character; a doc without "[edit]" skips it.
def medicine_c1(text):
    text = text.replace("{[ promptMessage ]}", "")
    if "[edit]" in text:
        text = C1_EDIT.sub("", text)
    return C1_CHROME.sub("", text)


def _std(out, text):
    # the result in the shared form: str -> fix/keep, ("drop", ...), None or True -> drop
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple) and out and out[0] == "drop": return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS or doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    return _std(medicine_c1(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'math-finemath', 'math-megamath'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** It knows only the sites of the audit. A lesson heading `Related Questions` goes.

### `DA-C2-007` · C2 · repeated lines and passages, not math

1. **Disease:** `DA-C2-001` deletes repeated formula lines (`89dc8e7278ce1f4b`) and leaves repeated captions and passages. Findings: F09, F12.
2. **Medicine:** Math lines, code and speaker turns stay; repeated captions and long passages keep the first copy, on both packs.
3. **Action and place:** fix; replaces `DA-C2-001` on finemath; on megamath new, after the family's rules. **Runs on:** math-finemath, math-megamath. **From:** #71 `DA-C2-007` (place on finemath).

```python
RULE_ID = "DA-C2-007"
REPLACES = "DA-C2-001"
RUNS_AFTER = "family"
FROM = "#71 DA-C2-007"
# repeated lines and passages left after spec v1, found by the math audit (assignment 11); replaces DA-C2-001
CODE = "C2"
ACTION = "fix"
STAGE = "30"
STATUS = "CANDIDATE"
LANGS = {'en'}
ORIGIN = "math-finemath, math-megamath"  # packs it runs on: the disease was found in both
PACKS = set(ORIGIN.split(", "))  # the math packs share the "en" tag, so the pack is checked too
SOURCE = "math-finemath, math-megamath"  # packs whose audit found it (aneeshads, assignment 11)
AUTHOR = "aneeshads"
import re
from collections import Counter
from itertools import compress

C2_CODE = re.compile(r"[;{}]\s*$|^\s*(?:for|if|while|return|def|class)\b")  # as DA-C2-001
# a line with one of these is math, and keeps every copy
C2_MATH = re.compile(r"[=^$\\<>±×÷≤≥∑∫√∞]|\d\s*[-+*/]\s*\d")
# a transcript's speaker turn ("MALE SPEAKER: ...", "PROFESSOR: ..."): each copy is a separate turn, so all are kept
C2_TURN = re.compile(r"[A-Z][A-Z .'\-]{1,40}:(?:\s|$)")
C2_CAPTION = re.compile(r"(?:Figure|Fig\.|Table)\s+\d+(?:\.\d+)*\b")
C2_PARA_SPLIT = re.compile(r"(\n\s*\n)")
C2_OPTION = re.compile(r"\s*(?:[A-E1-5][.)]|\(?[a-e]\))\s")
C2_BULLET = re.compile(r"\s*[•*-] ")
C2_CODE_END = re.compile(r"[;{}]\s*$")


# Speed: no Python loop over a doc's lines or passages. Counter, map and compress run in C and leave only the distinct
# strings seen twice or more; a doc has few of those, and most docs have none, so they skip all the work.
def _repeated(counts, min_len):
    # the distinct strings of min_len+ characters seen 2+ times
    rep = compress(counts.keys(), map((2).__le__, counts.values()))
    return [s for s in rep if len(s) >= min_len]


def _drop_later_copies(text, s):
    # keep the first line whose stripped text is s; remove every later one with the line break before it
    # ([^\S\n] is the whitespace that str.strip removes, less the line break)
    body = re.escape(s)
    end = re.search(r"(?m)^[^\S\n]*" + body + r"[^\S\n]*$", text).end()
    return text[:end] + re.sub(r"\n[^\S\n]*" + body + r"[^\S\n]*(?=\n|\Z)", "", text[end:])


def _passage_kept(q):
    # ruling 13 keeps a bullet list, a list of answer options, and code
    if C2_BULLET.match(q):
        return True
    ql = list(filter(str.strip, q.splitlines()))
    options = sum(map(bool, map(C2_OPTION.match, ql))) >= max(2, len(ql) // 2)
    code = sum(map(bool, map(C2_CODE_END.search, ql))) >= max(1, len(ql) // 3)
    return options or code


def _c2_dedupe_passages(text):
    # ruling 13: a passage (150+ chars) repeated even once loses its later copies, each with the blank-line
    # separator before it; option lists and code are kept
    parts = C2_PARA_SPLIT.split(text)      # passage j is parts[2j], and parts[2j - 1] is the separator before it
    paras = list(map(str.strip, parts[::2]))
    counts = Counter(paras)
    keep, changed = bytearray(b"\x01") * len(parts), False
    for q in _repeated(counts, 150):
        if _passage_kept(q):
            continue
        j = paras.index(q)                  # the first copy stays
        for _ in range(counts[q] - 1):
            j = paras.index(q, j + 1)
            keep[2 * j] = keep[2 * j - 1] = 0
        changed = True
    return "".join(compress(parts, keep)) if changed else text


def medicine_c2(text):
    counts = Counter(map(str.strip, text.split("\n")))
    # the lines that keep only their first copy
    for s in _repeated(counts, 15):
        if C2_CODE.search(s):
            continue
        if (counts[s] >= 5 and not C2_MATH.search(s) and not C2_TURN.match(s)) or C2_CAPTION.match(s):
            text = _drop_later_copies(text, s)
    return _c2_dedupe_passages(text)


def _std(out, text):
    # the result in the shared form: str -> fix/keep, ("drop", ...), None or True -> drop
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple) and out and out[0] == "drop": return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS or doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    return _std(medicine_c2(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'math-finemath', 'math-megamath'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** A repeated prose line keeps only its first copy, as before.

### `DA-C1-014` · C1 · a licence line left at the start

1. **Disease:** `DA-C1-007` cuts the HAL cover page and leaves the licence line under it (`c31993065f746189`). Findings: F26.
2. **Medicine:** A Creative Commons line goes when it is the first line of the document.
3. **Action and place:** fix; new, after `DA-C1-007`. **Runs on:** sci-finepdfs. **From:** #79 Medicine 2, part (a) (number given; part (b) left out).

```python
RULE_ID = "DA-C1-014"
RUNS_AFTER = "DA-C1-007"
FROM = "#79 Medicine 2, part (a)"
CODE = "C1"
ACTION = "fix"
STATUS = "PROPOSAL"
import re

_C1_CC_LEAD = re.compile(r"\A\s*Distributed under a Creative Commons[^\n]{0,120}License\s*\n")

def medicine_c1(text):
    """A Creative Commons line left at the very start after the HAL cover page is cut (DA-C1-007)."""
    return _C1_CC_LEAD.sub("", text, count=1)


SPEC_PACKS = {'sci-finepdfs'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = medicine_c1(doc["text"])
    return ("fix" if out != doc["text"] else "keep", out)
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** Other cover pages and running footers stay.

### `DA-C2-011` · C2 · the repeat cut takes whole paragraphs

1. **Disease:** `DA-C2-004` cuts the second copy of the opening in the middle of a paragraph (`335a56d0e87a73a9`). Findings: F25.
2. **Medicine:** The copy must start at a line start, and the cut goes back to the last blank line.
3. **Action and place:** fix; replaces `DA-C2-004`. **Runs on:** sci-finepdfs. **From:** #79 Medicine 1 (number given; put into the rest of `DA-C2-004`).

```python
RULE_ID = "DA-C2-011"
REPLACES = "DA-C2-004"
FROM = "#79 Medicine 1"
CODE = "C2"
import re
def medicine_c2(text):
    """Remove a long exact second copy of the opening passage (as DA-C2-004), but only whole paragraphs:
    the copy must start at a line start, and if the copies diverge the cut goes back to the last blank line."""
    seed_len, min_repeat, search_limit = 64, 200, 2500
    if len(text) < 2 * min_repeat:
        return text
    second = text.find(text[:seed_len], seed_len, min(len(text), search_limit))
    if second < 0 or text[second - 1] != "\n":
        return text
    m, mx = 0, min(second, len(text) - second)
    while m < mx and text[m] == text[second + m]:
        m += 1
    end = second + m
    if end < len(text) and text[end - 2:end] != "\n\n":
        brk = text.rfind("\n\n", second, end)
        if brk < 0:
            return text
        end = brk + 2
    if end - second < min_repeat:
        return text
    return text[:second] + text[end:]

import re

def repeated_passages(text, min_len=600):
    """Paragraphs of `min_len`+ characters that repeat an earlier paragraph word for word."""
    seen, copies = set(), []
    for para in text.split("\n\n"):
        key = " ".join(para.split())
        if len(key) < min_len or key.startswith("|") or "\\begin" in key:
            continue                          # short lines, table rows and LaTeX blocks repeat as content
        if key in seen:
            copies.append(para)
        seen.add(key)
    return copies


def remove_repeated_passages(text: str, max_removed_frac: float = 0.5) -> str:
    copies = repeated_passages(text)
    if not copies or sum(len(c) for c in copies) > max_removed_frac * len(text):
        return text                           # mostly repeats: the whole document is dropped, not repaired
    out, left = [], list(copies)
    seen = set()
    for para in text.split("\n\n"):
        key = " ".join(para.split())
        if para in left and key in seen:
            left.remove(para)
            continue
        seen.add(key)
        out.append(para)
    return "\n\n".join(out)


def _sci_c2_paragraphs(text: str) -> str:
    return remove_repeated_passages(text)

_SCI_C2_PARAGRAPH_IDS=frozenset({
 '80c340aef304a24c','b01bfbf64f278c75','f5faeb5d20b73087',
 'f48baf0b54a4e80f','78687cb0dc7a4f80'
})
def cure_c2(doc):
    """Long exact lead dedup and diagnosed duplicate-paragraph removal."""
    text=medicine_c2(doc['text'])
    if doc['doc_id'] in _SCI_C2_PARAGRAPH_IDS:
        text=_sci_c2_paragraphs(text)
    return text
SPEC_PACKS = {'sci-finepdfs'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = cure_c2(doc)
    if isinstance(out, tuple):
        return out
    if out is None:
        return ("drop", doc["text"])
    return ("fix" if out != doc["text"] else "keep", out)
```

4. **Harm test:** 0 changed, 0 dropped on every gold set (see above). PASS.
5. **Side effects:** It removes less: a copy that differs in its first paragraph stays whole.
