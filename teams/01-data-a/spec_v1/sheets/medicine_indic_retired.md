# Retired local Indic cards

Ten historical cards copied from the local `0104f40` intake, retired by PR #240 at `3bbd31f`. These blocks are parsed as inventory only, never executed. Their original IDs and contributor code remain for audit; current policy reasons supersede inherited source claims.

## 1. `DB-A6-010` · A6 Wrong or mixed script · fix · stage 10

**Retirement decision:** TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** A6 Wrong or mixed script. Pack `ind-sa-fw2`. Source: section A6 of `diagnosis/submissions/ind-sa-fw2/dattatreyamanjunath/medicine.md`, PR #45 at `b054835`.
2. **Medicine:** the author's code, unchanged. Main function: `medicine_a6(text)`.

```python deferred
RULE_ID = "DB-A6-010"
CODE = "A6"
ACTION = "fix"
STAGE = "10"
STATUS = "DEFERRED"
DEFER_REASON = "TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ"
LANGS = {"san_Deva"}
ORIGIN = "ind-sa-fw2"  # packs it runs on
SOURCE = "ind-sa-fw2"  # pack whose medicine this came from
AUTHOR = "dattatreyamanjunath"  # PR #45 at b054835
import re  # added by Data B: harm_test_indic.py gives every block `re`

_D = "ऀ-ॿ"
_CONS = "क-हक़-य़"
_MAT = "ा-ौॢॣ"
_ABBR = re.compile(r"(?:^|[\s(])(डा|डॉ|प्रो|डो|श्री|सं|पृ|ई|क्र|पं|मो|फो|दू)$")

def medicine_a6(text):
    def colon(m):
        before = text[max(0, m.start() - 6):m.start()]
        after = text[m.end():m.end() + 3].lstrip()
        if _ABBR.search(before) or (after and after[0].isalpha() and not "ऀ" <= after[0] <= "ॿ"):
            return m.group()
        return "ः"
    t = re.sub(f"(?<=[{_CONS}{_MAT}]):(?=[\\s।॥]|$)", colon, text)
    t = re.sub(f"(?<=[{_CONS}{_MAT}ंः])[Ss](?=[{_CONS}])", "ऽ", t)
    out = []
    for line in t.split("\n"):
        if not (line.startswith("|") or line.rstrip().endswith("|") and line.count("|") > 2):
            line = re.sub(f"(?<=[{_D}]) ?\\|\\|(?=\\s|$)", " ॥", line)
            line = re.sub(f"(?<=[{_D}]) ?\\|(?=\\s|$)", " ।", line)
        line = re.sub(f"(?<=[{_D}]) ?I(?=\\s*$|\\s+[{_D}])", " ।", line)
        line = re.sub(f"(?<=[{_D}])o(?=\\s)", "॰", line)
        line = re.sub(f"(?<=[A-Za-z]{{3}})(?=[{_CONS}])|(?<=[{_D}])(?=[A-Za-z]{{3}})", " ", line)
        out.append(line)
    return "\n".join(out).translate({0x0946: 0x0947, 0x094A: 0x094B, 0x0931: 0x0930})


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = medicine_a6  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** fix. **Runs on:** `ind-sa-fw2` only.
4. **Harm test (as the author reports it):** gold set of 68 documents (ind-sa-fw2 68); rule: changed or dropped at most 1% or 1 document, whichever is more.
5. **Side effects:** in the author's `medicine.md`, section A6 (PR #45).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.

## 2. `DB-A6-011` · A6 Wrong or mixed script · fix · stage 10

**Retirement decision:** TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** A6 Wrong or mixed script. Pack `ind-sa-fw2`. Source: section A6 of `diagnosis/submissions/ind-sa-fw2/swatibansal/medicine.md`, PR #80 at `7af8541`.
2. **Medicine:** the author's code, unchanged. Main function: `fix_a6(text)`.

```python deferred
RULE_ID = "DB-A6-011"
CODE = "A6"
ACTION = "fix"
STAGE = "10"
STATUS = "DEFERRED"
DEFER_REASON = "TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ"
LANGS = {"san_Deva"}
ORIGIN = "ind-sa-fw2"  # packs it runs on
SOURCE = "ind-sa-fw2"  # pack whose medicine this came from
AUTHOR = "swatibansal"  # PR #80 at 7af8541
import re  # added by Data B: harm_test_indic.py gives every block `re`

# character classes use \u escapes on purpose: a literal nukta letter is stored decomposed and would break the range
DEVA_LETTER = r"\u0904-\u0939\u093D\u0950\u0958-\u0961\u0972-\u097F"    # letters incl. avagraha and OM
DEVA_MARK = r"\u0900-\u0903\u093A-\u093C\u093E-\u094F\u0951-\u0957\u0962\u0963"    # matras, virama, nukta, bindus, visarga, accents
BOUNDARY = r"[ \t\u00A0\r\n।॥]"    # space, tab, nbsp, newline, danda
E_O_VOWELS = r"\u0946\u0947\u094A\u094B\u090E\u090F\u0912\u0913"    # matras ॆ े ॊ ो and letters ऎ ए ऒ ओ
RE_A6_DANDA = re.compile(rf"(?<=[{DEVA_LETTER}{DEVA_MARK}])(I{{1,2}}|l{{1,2}})(?={BOUNDARY}|$)")
RE_A6_AVAGRAHA = re.compile(rf"(?<=[{E_O_VOWELS}])[Ss](?=[{DEVA_LETTER}])")
# ASCII ':' typed for visarga: right after a Devanagari letter or vowel sign, before a boundary or a Devanagari letter
RE_A6_COLON = re.compile(rf"(?<=[{DEVA_LETTER}\u093A-\u093C\u093E-\u094F\u0962\u0963]):(?={BOUNDARY}|$|[{DEVA_LETTER}])")
# ASCII '|' or '||' typed for danda or double danda: after a Devanagari word (an optional space before it is kept), before a boundary
RE_A6_PIPE = re.compile(rf"(?<=[{DEVA_LETTER}{DEVA_MARK}])([ ]?)(\|{{1,2}})(?={BOUNDARY}|$)")

RE_LATIN_LETTER = re.compile(r"[A-Za-z]")

def _pipe_line(line):
    if line.lstrip().startswith("|") or RE_LATIN_LETTER.search(line):    # wiki table row, or a title/menu line with Latin words
        return line
    return RE_A6_PIPE.sub(lambda m: m.group(1) + ("॥" if len(m.group(2)) == 2 else "।"), line)

def fix_a6(text):
    text = RE_A6_DANDA.sub(lambda m: "॥" if len(m.group(1)) == 2 else "।", text)
    text = RE_A6_AVAGRAHA.sub("ऽ", text)
    text = RE_A6_COLON.sub("ः", text)
    return "\n".join(_pipe_line(l) for l in text.split("\n"))


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = fix_a6  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** fix. **Runs on:** `ind-sa-fw2` only.
4. **Harm test (as the author reports it):** gold set of 68 documents (ind-sa-fw2 68), medicine.md sha256 ff5e472b5ced; rule: changed or dropped at most 1% or 1 document, whichever is more.
5. **Side effects:** in the author's `medicine.md`, section A6 (PR #80).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.

## 3. `DB-A6-012` · A6 Wrong or mixed script · fix · stage 10

**Retirement decision:** TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** A6 Wrong or mixed script. Pack `ind-ta-sangraha-v`. Source: section A6 of `diagnosis/submissions/ind-ta-sangraha-v/abinesh1/medicine.md`, PR #101 at `3d69463`.
2. **Medicine:** the author's code, unchanged. Main function: `medicine_a6(text)`.

```python deferred
RULE_ID = "DB-A6-012"
CODE = "A6"
ACTION = "fix"
STAGE = "10"
STATUS = "DEFERRED"
DEFER_REASON = "TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ"
LANGS = {"tam_Taml"}
ORIGIN = "ind-ta-sangraha-v"  # packs it runs on
SOURCE = "ind-ta-sangraha-v"  # pack whose medicine this came from
AUTHOR = "abinesh1"  # PR #101 at 3d69463

import re

def medicine_a6(text):
    return re.sub("ஃ(?![க-ஹ])", ":", text)  # aytham before a space or line end is a lost colon


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = medicine_a6  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** fix. **Runs on:** `ind-ta-sangraha-v` only.
4. **Harm test (as the author reports it):** gold set: 0 of 41 changed, 0 dropped, PASS. Pack check: changes 135 documents, none labelled `clean`. 8 of the 10 changed documents
5. **Side effects:** in the author's `medicine.md`, section A6 (PR #101).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.

## 4. `DB-A6-014` · A6 Wrong or mixed script · fix · stage 10

**Retirement decision:** TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** A6 Wrong or mixed script. Pack `ind-te-sangraha-s`. Source: section A6 of `diagnosis/submissions/ind-te-sangraha-s/shankarpandala/medicine.md`, PR #92 at `2e61b3b`.
2. **Medicine:** the author's code, unchanged. Main function: `medicine_a6(text)`.

```python deferred
RULE_ID = "DB-A6-014"
CODE = "A6"
ACTION = "fix"
STAGE = "10"
STATUS = "DEFERRED"
DEFER_REASON = "TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ"
LANGS = {"tel_Telu"}
ORIGIN = "ind-te-sangraha-s"  # packs it runs on
SOURCE = "ind-te-sangraha-s"  # pack whose medicine this came from
AUTHOR = "shankarpandala"  # PR #92 at 2e61b3b
import re  # added by Data B: harm_test_indic.py gives every block `re`

_VISARGA_WORDS = {"అంతః", "ఆయుః", "ఉరః", "చేతః", "తపః", "తేజః", "నమః", "పయః", "పునః", "ప్రాతః", "భూః", "మనః", "యశః", "వచః", "శిరః", "శ్రీః", "స్వః"}

def medicine_a6(text):
    def fix(m):
        word = m.group(1) + "ః"
        return m.group(0) if word in _VISARGA_WORDS else m.group(1) + ":"
    return re.sub(r"([\u0C00-\u0C02\u0C04-\u0C7F]*)ః(?=\n|$| [\"“‘'])", fix, text)


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = medicine_a6  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** fix. **Runs on:** `ind-te-sangraha-s` only.
4. **Harm test (as the author reports it):** gold set: the gold set of this pack, 5 documents (teams/02-data-b/gold/ind-te-sangraha-s_gold.csv on main, 2 Nov 2026); rule `medicine_a6` of this file (the code block in this section), run with harm_test_indic.py (PR #148); a rule passes if it changes or drops at most max(1 document, 1%) of the gol...
5. **Side effects:** in the author's `medicine.md`, section A6 (PR #92).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.

## 5. `DB-G2-001` · G2 Lost ZWJ or ZWNJ · fix · stage 10

**Retirement decision:** TSAI 4 November: adds ZWJ by design; excluded from spec v1 and all sibling scopes; held for v2. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** G2 Lost ZWJ or ZWNJ. Pack `ind-bn-fw2`. Source: section G2 of `diagnosis/submissions/ind-bn-fw2/AditeyaAItronics/medicine.md`, PR #64 at `840d13d`.
2. **Medicine:** the author's code, unchanged. Main function: `medicine_g2(text)`.

```python deferred
RULE_ID = "DB-G2-001"
CODE = "G2"
ACTION = "fix"
STAGE = "10"
STATUS = "DEFERRED"
DEFER_REASON = "TSAI 4 November: adds ZWJ by design; excluded from spec v1 and all sibling scopes; held for v2"
LANGS = {"ben_Beng"}
ORIGIN = "ind-bn-fw2"  # packs it runs on
SOURCE = "ind-bn-fw2"  # pack whose medicine this came from
AUTHOR = "AditeyaAItronics"  # PR #64 at 840d13d

import re

ZWJ = chr(0x200D)
# র + \u09cd + য + \u09be at the start of a word is the loan-word ra + ya-phala (র\u200d\u09cdযাব, র\u200d\u09cdযালি). Without ZWJ it
# renders as reph + য (র্যা). Inside a word the reph reading is correct (কার্যালয়), and after a virama
# (ব্র্যান্ড) র is already a ra-phala, so those are left alone.
_LOAN = re.compile('(?<![\u0980-\u09ff])র্(?=যা)')

def medicine_g2(text):
    """Insert ZWJ after র in loan-word র্যা."""
    return _LOAN.sub('র' + ZWJ + '\u09cd', text)


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = medicine_g2  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** fix. **Runs on:** `ind-bn-fw2` only.
4. **Harm test (as the author reports it):** gold set of 238 documents (ind-bn-fw2 238), medicine.md sha256 93e459d0b451; rule: changed or dropped at most 1% or 1 document, whichever is more.
5. **Side effects:** in the author's `medicine.md`, section G2 (PR #64).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.

## 6. `DB-G2-002` · G2 Lost ZWJ or ZWNJ · fix · stage 10

**Retirement decision:** TSAI 4 November: adds ZWJ by design; excluded from spec v1 and all sibling scopes; held for v2. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** G2 Lost ZWJ or ZWNJ. Pack `ind-bn-fw2`. Source: section G2 of `diagnosis/submissions/ind-bn-fw2/sohamniyogi/medicine.md`, PR #138 at `8a50c48`.
2. **Medicine:** the author's code, unchanged. Main function: `medicine_g2(text)`.

```python deferred
RULE_ID = "DB-G2-002"
CODE = "G2"
ACTION = "fix"
STAGE = "10"
STATUS = "DEFERRED"
DEFER_REASON = "TSAI 4 November: adds ZWJ by design; excluded from spec v1 and all sibling scopes; held for v2"
LANGS = {"ben_Beng"}
ORIGIN = "ind-bn-fw2"  # packs it runs on
SOURCE = "ind-bn-fw2"  # pack whose medicine this came from
AUTHOR = "sohamniyogi"  # PR #138 at 8a50c48

import re

_G2_RX = re.compile("(?:^|(?<=[^ঀ-৿])|(?<=্))র্যা", re.M)


def medicine_g2(text):
    """র্যা at a word start or after a virama -> র + ZWJ + ্যা (the loanword form)."""
    return _G2_RX.sub("র‍্যা", text)


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = medicine_g2  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** fix. **Runs on:** `ind-bn-fw2` only.
4. **Harm test (as the author reports it):** official gold 0 / 238 (0 %), PASS. Proxy 0 / 314 (circular, see above). **Warning:** a gold set built from the same FineWeb-2 dump
5. **Side effects:** in the author's `medicine.md`, section G2 (PR #138).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.

## 7. `DB-G4-001` · G4 Over-normalised · fix · stage 10

**Retirement decision:** TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** G4 Over-normalised. Pack `ind-kn-sangraha-v`. Source: section G4 of `diagnosis/submissions/ind-kn-sangraha-v/pyru/medicine.md`, PR #25 at `5eaa16e`.
2. **Medicine:** the author's code, unchanged. Main function: `medicine_g4(text)`.

```python deferred
RULE_ID = "DB-G4-001"
CODE = "G4"
ACTION = "fix"
STAGE = "10"
STATUS = "DEFERRED"
DEFER_REASON = "TSAI 5 November: visarga/colon conversion in either direction is excluded from spec v1 in every script, including Tamil ஃ"
LANGS = {"kan_Knda"}
ORIGIN = "ind-kn-sangraha-v"  # packs it runs on
SOURCE = "ind-kn-sangraha-v"  # pack whose medicine this came from
AUTHOR = "pyru"  # PR #25 at 5eaa16e

"""Find visarga (U+0C83) that stands for a colon.

The Indic NLP normaliser (correct_visarga) rewrites ':' after an Indic letter as visarga, so a dateline
'ಬೆಂಗಳೂರು:' becomes 'ಬೆಂಗಳೂರುಃ'. Real visarga words are a small closed set: Sanskrit adverbs in -ತಃ / -ಶಃ,
a few words (ಪುನಃ, ನಮಃ, ...) and compound prefixes (ದುಃಖ, ನಿಃಸಂಶಯ, ಅಂತಃಕರಣ). Lines with a danda (। ॥) are
treated as Sanskrit verse and left alone. PDF verse often writes the danda as '|' or '||': in a document where a
visarga is followed by '|', or where two or more lines end in '|', every line with a '|' is left alone too.
"""
import re

VIS, VIRAMA, AA_SIGN = chr(0x0C83), chr(0x0CCD), chr(0x0CBE)
KN_BLOCK = chr(0x0C80) + "-" + chr(0x0CFF)                  # the Kannada block, for character classes
JOINERS = chr(0x200C) + chr(0x200D)                          # ZWNJ, ZWJ
KN = re.compile("[" + KN_BLOCK + "]")
WORD_END = re.compile("[" + KN_BLOCK + JOINERS + "]+$")
GENUINE_END = re.compile(r"(ತಃ|ಶಃ)$")
GENUINE = {"ಪುನಃ", "ನಮಃ", "ಮನಃ", "ಅಂತಃ", "ಯಶಃ", "ತೇಜಃ", "ಶಿರಃ", "ಪ್ರಾತಃ", "ಉರಃ", "ಛಂದಃ", "ತಪಃ", "ಪಯಃ",
           "ವಯಃ", "ನಭಃ", "ಅಧಃ", "ಪುರಃ", "ಬಹಿಃ", "ದುಃ", "ನಿಃ", "ಮುಹುಃ", "ಚಕ್ಷುಃ", "ಆಯುಃ", "ಧನುಃ", "ವಪುಃ",
           "ಹವಿಃ", "ಜ್ಯೋತಿಃ", "ಶ್ರೇಯಃ", "ಮನಸಃ", "ರಜಃ", "ತಮಃ", "ಸ್ವಾಹಾಃ", "ಓಂನಮಃ", "ಸ್ವತಹಃ"}
PIPE_AFTER_VIS = re.compile(VIS + r"[ \t]*\|")
PREFIX = {"ದು", "ನಿ", "ಅಂತ", "ಮನ", "ತೇಜ", "ಪ್ರಾತ", "ಬಹಿ", "ಪುನ", "ಛಂದ", "ತಪ", "ಯಶ", "ಅಧ", "ಶಿರ", "ಸುದು",
          "ಸರ್ವದು", "ಸುಖದು", "ಮನಸ್ತಾಪದು"}

def colon_visargas(t, verse_min=2, verse_len=160):
    """Return the offsets of every visarga in t that stands for a colon.

    A line with verse_min or more candidates is taken as Sanskrit verse and skipped when it is short
    (verse_len characters or less) or when most candidates end like a Sanskrit nominative (-aH after a
    bare consonant, or -AH). Kannada prose with several colons ends them on vowel signs (ಹೇಳುತ್ತಾರೆಃ).
    """
    verse_doc = "।" in t or "॥" in t
    by_line = {}
    for i in _candidates(t):
        start = t.rfind("\n", 0, i) + 1
        if verse_doc and t[i - 1] != VIRAMA and " " in t[start:i].strip():
            continue                                    # a document with verse: only first words and ್ಃ
        by_line.setdefault(start, []).append(i)
    out = []
    for start, offs in by_line.items():
        end = t.find("\n", start); end = len(t) if end < 0 else end
        if len(offs) >= verse_min:
            sanskrit = sum(1 for i in offs if t[i - 1] == AA_SIGN or 0x0C95 <= ord(t[i - 1]) <= 0x0CB9)
            if end - start <= verse_len or sanskrit / len(offs) >= 0.75:
                continue
        out.extend(offs)
    return out

def _candidates(t):
    out = []
    pipe_verse = bool(PIPE_AFTER_VIS.search(t)) or sum(1 for l in t.split("\n") if l.rstrip().endswith("|")) >= 2
    for m in re.finditer(VIS, t):
        i = m.start()
        line_start = t.rfind("\n", 0, i) + 1
        line_end = t.find("\n", i); line_end = len(t) if line_end < 0 else line_end
        if "।" in t[line_start:line_end] or "॥" in t[line_start:line_end]:
            continue
        if pipe_verse and "|" in t[line_start:line_end]:
            continue                                    # verse with '|' as danda
        p = t[i - 1] if i else " "
        if p == VIRAMA:
            out.append(i); continue
        if not KN.match(p):
            continue                                    # orphan visarga: G1, not a colon
        w = WORD_END.search(t[:i]).group()
        nx = t[i + 1] if i + 1 < len(t) else " "
        if KN.match(nx):                                # visarga inside a word
            if w in PREFIX or w.endswith(("ದು", "ನಿ")) or len(w) < 3:
                continue
            out.append(i); continue
        if GENUINE_END.search(w + VIS) or (w + VIS) in GENUINE:
            continue
        out.append(i)
    return out

def fix(t):
    """Medicine: turn colon-visarga back into ':'."""
    pos = set(colon_visargas(t))
    return "".join(":" if i in pos else c for i, c in enumerate(t))


fix_visarga = fix
DEC_SPLIT = re.compile(r"(?<=[^\d\n.])(\d{1,3})\. (\d{1,3})(?![\d.])")
DOMAIN_SPLIT = re.compile(r"\b([A-Za-z0-9-]+)\. (com|in|org|net|gov|co|info|edu)\b")
WWW_SPLIT = re.compile(r"\bwww\. (?=[A-Za-z0-9])")


def fix_dot_split(t):
    """G4: join a dot split by an inserted space inside a decimal, a clock time or a web address (3. 3, 10. 20,
    newskadaba. com, www. vknews. in). A number at the start of a line (a list item '1. 5 ...') is left alone."""
    t = WWW_SPLIT.sub("www.", t)
    for _ in range(3):                                  # karnataka. gov. in needs two passes
        t = DOMAIN_SPLIT.sub(r"\1.\2", t)
    return DEC_SPLIT.sub(r"\1.\2", t)


def fix_over_normalised(t):
    """G4: undo the two marks of the earlier normaliser: visarga for a colon, then the dot split."""
    return fix_dot_split(fix_visarga(t))


def medicine_g4(text):
    """The G4 rule, under the name that the harm tests look for: returns the fixed text."""
    return fix_over_normalised(text)


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = medicine_g4  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** fix. **Runs on:** `ind-kn-sangraha-v` only.
4. **Harm test (as the author reports it):** on the gold set (103 documents) it changes 0 and drops 0: 0.0%. Pass (at most 1 document or 1%, whichever is larger).
5. **Side effects:** in the author's `medicine.md`, section G4 (PR #25).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.

## 8. `DB-F2-001` · F2 Benchmark leak · drop · stage 30

**Retirement decision:** TSAI 4 November: needs Itihasa test sentences via ITIHASA_TEST_SN; legacy F2 excluded from spec v1; optional G9 F2 does not cover Itihasa. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** F2 Benchmark leak. Pack `ind-sa-fw2`. Source: section F2 of `diagnosis/submissions/ind-sa-fw2/dattatreyamanjunath/medicine.md`, PR #45 at `b054835`.
2. **Medicine:** the author's code, unchanged. Main function: `medicine_f2(text)`.

```python deferred
RULE_ID = "DB-F2-001"
CODE = "F2"
ACTION = "drop"
STAGE = "30"
STATUS = "DEFERRED"
DEFER_REASON = "TSAI 4 November: needs Itihasa test sentences via ITIHASA_TEST_SN; legacy F2 excluded from spec v1; optional G9 F2 does not cover Itihasa"
LANGS = {"san_Deva"}
ORIGIN = "ind-sa-fw2"  # packs it runs on
SOURCE = "ind-sa-fw2"  # pack whose medicine this came from
AUTHOR = "dattatreyamanjunath"  # PR #45 at b054835
import re  # added by Data B: harm_test_indic.py gives every block `re`

import os

_L = "ऀ-ॣ०-ॿ"
_W, _STEP, _MIN = 40, 20, 3
_INDEX = {}

def _windows():
    if not _INDEX:
        path = os.environ.get("ITIHASA_TEST_SN")
        if not path:
            raise RuntimeError("set ITIHASA_TEST_SN to the Itihasa test.sn.csv file")
        with open(path, encoding="utf-8") as f:
            for k, line in enumerate(f):
                n = re.sub(f"[^{_L}]", "", line)
                if len(n) < _W:
                    continue
                for j in range(0, len(n) - _W + 1, _STEP):
                    _INDEX.setdefault(n[j:j + _W], set()).add(k)
    return _INDEX

def medicine_f2(text):
    index = _windows()
    n = re.sub(f"[^{_L}]", "", text)
    hit = set()
    for j in range(len(n) - _W + 1):
        hit |= index.get(n[j:j + _W], set())
    return len(hit) >= _MIN


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = medicine_f2  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** drop. **Runs on:** `ind-sa-fw2` only.
4. **Harm test (as the author reports it):** gold set of 68 documents (ind-sa-fw2 68); rule: changed or dropped at most 1% or 1 document, whichever is more.
5. **Side effects:** in the author's `medicine.md`, section F2 (PR #45).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.

## 9. `DB-F1-005` · F1 Personal data · fix · stage 40

**Retirement decision:** fails current own-pack gold (2 of 114); source F1 is also replaced by shared Indic F1. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** F1 Personal data. Pack `ind-kn-fw2`. Source: section F1 of `diagnosis/submissions/ind-kn-fw2/LokeshJatangi/medicine.md`, PR #63 at `aeb79b8`.
2. **Medicine:** the author's code, unchanged. Main function: `medicine_f1(text)`.

```python deferred
RULE_ID = "DB-F1-005"
CODE = "F1"
ACTION = "fix"
STAGE = "40"
STATUS = "DEFERRED"
DEFER_REASON = "fails current own-pack gold (2 of 114); source F1 is also replaced by shared Indic F1"
LANGS = {"kan_Knda"}
ORIGIN = "ind-kn-fw2"  # packs it runs on
SOURCE = "ind-kn-fw2"  # pack whose medicine this came from
AUTHOR = "LokeshJatangi"  # PR #63 at aeb79b8

import re

F1_EMAIL = r"(?<![\w.+-])[A-Za-z0-9][A-Za-z0-9._%+-]*@[A-Za-z0-9][A-Za-z0-9.-]*\.[A-Za-z]{2,}(?![\w-])"
F1_NUMBER = r"(?:[0-9]{10,11}|[0-9]{5}[ \t]+[0-9]{5}|[0-9]{3}[ \t]+[0-9]{8})"
F1_PHONE = rf"(?<!\w)(?:\+[0-9]{{1,3}}[ \t]+)?{F1_NUMBER}(?!\w)"
F1_CANDIDATE = re.compile(rf"(?:{F1_EMAIL})|(?:{F1_PHONE})")

def medicine_f1(text, url=""):
    return F1_CANDIDATE.sub("[REDACTED]", text)


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = medicine_f1  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** fix. **Runs on:** `ind-kn-fw2` only.
4. **Harm test (as the author reports it):** **175-document** combined `kannada_gold.csv`, using the referenced Indic harness
5. **Side effects:** in the author's `medicine.md`, section F1 (PR #63).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.

## 10. `DB-F2-002` · F2 Benchmark leak · keep · stage 40

**Retirement decision:** TSAI 4 November: needs FLORES/IN22/Itihasa benchmark text; legacy F2 excluded from spec v1. Historical source and harm below are retained for provenance, not current execution evidence.

1. **Disease:** F2 Benchmark leak. Pack `ind-sa-fw2`. Source: section F2 of `diagnosis/submissions/ind-sa-fw2/swatibansal/medicine.md`, PR #80 at `7af8541`.
2. **Medicine:** the author's code, unchanged. Main function: `medicine_f2(text)`.

```python deferred
RULE_ID = "DB-F2-002"
CODE = "F2"
ACTION = "keep"
STAGE = "40"
STATUS = "DEFERRED"
DEFER_REASON = "TSAI 4 November: needs FLORES/IN22/Itihasa benchmark text; legacy F2 excluded from spec v1"
LANGS = {"san_Deva"}
ORIGIN = "ind-sa-fw2"  # packs it runs on
SOURCE = "ind-sa-fw2"  # pack whose medicine this came from
AUTHOR = "swatibansal"  # PR #80 at 7af8541
import re  # added by Data B: harm_test_indic.py gives every block `re`

import collections
import json, os, unicodedata
DEFAULT_BENCH = os.path.join("..", "bench")
WINDOW = 50
MAX_SENTENCE_CHARS = 2000
BENCH_DIRS = ("flores200_devtest", "itihasa_test", "in22_gen")
DROP_BENCH_PREFIXES = ("flores", "in22")
_DROP_CATS = ("P", "Z", "S")
_DROP_FULL = {"Nd", "Cf", "Cc"}
def _dropped(c):
    cat = unicodedata.category(c)
    return cat[0] in _DROP_CATS or cat in _DROP_FULL or c.isspace()
def normalise(text):
    """NFC per character, then drop punctuation, separators, symbols, digits of every script, format and control chars."""
    return "".join(c for ch in text for c in unicodedata.normalize("NFC", ch) if not _dropped(c))
class _Index:
    """hash(window) -> [(bench, sid, pos)] over every benchmark sentence; hits are verified against the real text."""
    def __init__(self, bench_dir):
        self.table = collections.defaultdict(list); self.norm = {}; self.n_sentences = 0
        if not bench_dir or not os.path.isdir(bench_dir): return
        for name in BENCH_DIRS:
            path = os.path.join(bench_dir, name, "sentences.jsonl")
            if not os.path.isfile(path): continue
            try:
                with open(path, encoding="utf-8", errors="replace") as f: lines = f.read().split("\n")
            except OSError: continue
            for ln, line in enumerate(lines, 1):
                line = line.strip()
                if not line: continue
                try: d = json.loads(line)
                except ValueError: continue
                if not isinstance(d, dict) or not isinstance(d.get("text"), str): continue
                sid = d.get("id", ln)
                if not isinstance(sid, (int, str)): sid = ln
                ns = normalise(d["text"][:MAX_SENTENCE_CHARS])
                if len(ns) < WINDOW: continue
                key = (name, sid); self.norm[key] = ns; self.n_sentences += 1
                for p in range(len(ns) - WINDOW + 1): self.table[hash(ns[p:p + WINDOW])].append((key, p))
    def __bool__(self): return bool(self.table)
    def hits(self, text, prefixes=None):
        """{(bench, sid): number of DISTINCT 50-char windows of that sentence found in text}."""
        found = collections.defaultdict(set)
        if not self.table: return {}
        ns = normalise(text)
        for i in range(len(ns) - WINDOW + 1):
            win = ns[i:i + WINDOW]
            for key, p in self.table.get(hash(win), ()):
                if prefixes and not key[0].lower().startswith(prefixes): continue
                if self.norm[key][p:p + WINDOW] == win: found[key].add(win)
        return {k: len(v) for k, v in found.items()}
_BENCH = {"dir": None, "cache": {}}
def _index(bench_dir=None):
    d = os.path.normpath(bench_dir or _BENCH["dir"] or DEFAULT_BENCH)
    if d not in _BENCH["cache"]: _BENCH["cache"][d] = _Index(d)
    return _BENCH["cache"][d]
def _strong(text, bench_dir=None, prefixes=None):
    return any(k >= 2 for k in _index(bench_dir).hits(text, prefixes).values())
def detect_f2(text, bench_dir=None):
    """True when some benchmark sentence (FLORES-200, IN22 or Itihāsa test split) shares >= 2 distinct 50-char
    normalised windows with the text. False when no benchmark file is available."""
    return _strong(text, bench_dir)
def drop_f2_flores(text, bench_dir=None):
    """Drop variant, not in MEDICINES: True only for FLORES-200 / IN22 hits (benchmark folder name starts with
    flores or in22). Itihāsa hits never drop (public canon)."""
    return _strong(text, bench_dir, DROP_BENCH_PREFIXES)

def medicine_f2(text):
    return text    # action none: the detector `detect_f2` reports (keep and record); `drop_f2_flores` is the drop variant


def _std(out, text):  # added by Data B, copied from Data A (medicine_english.md)
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep,
    # ("drop", _) -> drop. An empty result is a drop: nothing of the document is left.
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, tuple):
        if out[0] == "drop": return ("drop", text)
        out = out[1]
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)

_AUTHOR_RULE = medicine_f2  # added by Data B
_PACKS = frozenset(ORIGIN.split(", "))  # added by Data B


def medicine(doc):  # added by Data B: runs only on the pack it was tested on
    if doc.get("pack_id") not in _PACKS:
        return ("keep", doc["text"])
    return _std(_AUTHOR_RULE(doc["text"]), doc["text"])
```

3. **Action:** keep. **Runs on:** `ind-sa-fw2` only.
4. **Harm test (as the author reports it):** gold set of 68 documents (ind-sa-fw2 68), medicine.md sha256 ff5e472b5ced; rule: changed or dropped at most 1% or 1 document, whichever is more.
5. **Side effects:** in the author's `medicine.md`, section F2 (PR #80).
6. **Spec v1:** status CANDIDATE, reviewer pending. Harm not re-run by Data B.
