# en-fineweb-edu audit: issues and proposals

Week 3 (22 to 23 Nov 2026). Reviewer: Deepjyoti Saha (rotation, Team 1). Pack: `en-fineweb-edu`, 35 documents: 25 from Part_1 (`part-00000`) and 10 from Part_2 (`part-00001`). Taken over from the unsubmitted assignment, as agreed in "01 Data A" on 22 Nov.

Each document was compared raw (`Cap1-packs/`) against cleaned (`tokenizer_sample_v0_clean/train/`), matched by `doc_id`. Disease codes follow [DISEASES.md](../../../diagnosis/DISEASES.md). The labels are in `en-fineweb-edu-Part_1-Deepjyoti_Saha.csv` and `en-fineweb-edu-Part_2-Deepjyoti_Saha.csv` (the viewer's seven columns; `evidence` and `hex` left empty). A dropped document has no verdict; its note says whether the drop looks right or wrong, as in the en-dclm audit.

## Inputs and method

| Input | File | sha256 |
|---|---|---|
| Raw Part_1 | `Cap1-packs/en-fineweb-edu/Part_1/part-00000.parquet` | `53cbbf7a77b84e46…` |
| Raw Part_2 | `Cap1-packs/en-fineweb-edu/Part_2/part-00001.parquet` | `bdd5d5b4f7aa8ec0…` |
| Cleaned Part_1 | `tokenizer_sample_v0_clean/train/en-fineweb-edu/part-00000.parquet` | `2d0ac7331258ba29…` |
| Cleaned Part_2 | `tokenizer_sample_v0_clean/train/en-fineweb-edu/part-00001.parquet` | `f2f89744215cf72c…` |

- **Selection:** only changed documents, as with the viewer's "Only changed" default (`diagnosis/tools/view_packs_hub.py`, branch `g1-web-viewer-pre-post` at `be1571f8`). A document counts as changed if its text differs after cleaning or it is missing after cleaning. Part_1 has 1,428 changed documents of 54,004 (406 dropped); Part_2 has 514 of 17,929 (153 dropped). The 35 are a random sample of the changed ones (`random.Random(20261122).sample`, Part_1 first).
- **Which rule did it:** every sampled document was run again through `teams/01-data-a/spec_v1/clean.py` (English pipeline, then the shared F1 rule), one rule at a time. The rerun gives exactly the published cleaned text for all 35, so each change in the CSV notes names the rule that made it.
- **Labels:** first drafted from that rerun, then checked by the reviewer reading each raw and cleaned pair in full in the viewer (`read` = full 35); the reviewer's changes are in the CSVs.
- **Pack-wide counts:** the same rerun over every dropped document in both parts (559 drops), to size each finding beyond the sample.
- Not done: decoding the sampled documents from the corpus shards.

## Summary

| Verdict | Docs |
|---|---:|
| clean | 14 |
| fix | 9 |
| drop | 0 |
| Dropped by cleaning: drop looks right or acceptable | 6 |
| Dropped by cleaning: drop looks wrong | 6 |
| **Total** | **35** |

Mojibake, stray tags, foreign-language title lines, "Skip to main content" lines and real email addresses are handled well. The problems are in the drops (6 of 12 sampled drops look wrong), in the phone mask (numbers masked that are not phones, and phones left unmasked), and in boilerplate left behind (4 kept documents).

## 1. Good documents dropped as "truncated" (DA-C3-003) (4 in the sample)

**What it is.** DA-C3-003 drops the whole document when its last word is a "dangling" word (a, the, on, of, that ...) and it does not end in `. ! ? " ' ) :`. In this pack that hits complete articles for three reasons:
- a footer line at the end ("Share this page on"): the article is complete; only the last line is boilerplate;
- one cut-off last sentence after pages of good text ("... prior to seed germination. A");
- a curly closing quote: `worked on.”` does not count as an ending, because the rule lists only the straight quotes `"` and `'`.

**How big it is.** DA-C3-003 makes 392 of the 559 drops in the two parts. 268 of those documents are 2,000 characters or longer; 28 end in a curly closing quote (’ or ”); 32 end in the "[…]" marker, which is the case the rule is right about.

**What to do.**
1. Add `”` and `’` to the end punctuation (a clear false positive, small enough for the freeze).
2. For v1.1: when the document is long, cut the last line or partial sentence instead of dropping the document; keep the drop for short fragments and for the "[…]" marker.

| Part | doc_id |
|---|---|
| Part_1 | `40fb0256c0b2c8fe` |
| Part_1 | `eeadc959d809b46e` |
| Part_1 | `0309ba9879105216` |
| Part_2 | `5716019c1d54b86c` |

## 2. A long article dropped for one ad line (DA-D4-001) (1)

**What it is.** DA-D4-001 drops any document that contains "essay writing service" (or spintax). On ukessays.com a 21,000-character anatomy lab write-up has one inline ad ("... using our Essay Writing Service!"), so the whole write-up was dropped.

**How big it is.** DA-D4-001 makes 60 drops in the two parts; 30 of them are 5,000 characters or longer, which is unusual for an essay-mill page.

**What to do.** For v1.1: when the phrase sits in a short line inside a long document, remove that line (C1) instead of dropping; keep the drop for short pages.

| Part | doc_id |
|---|---|
| Part_1 | `bdacc614dc1007a6` |

## 3. Table or code output read as a keyword line (DA-C5-002) (1)

**What it is.** DA-C5-002 drops a document with any line of 7 or more `|`-separated fields. A Raspberry Pi tutorial prints SQLite query output (`1|22.1|2013-10-08 2|...`), which matched.

**How big it is.** DA-C5-002 makes 56 drops in the two parts.

**What to do.** Do not count lines with digits in most fields, or lines inside code, as keyword lines.

| Part | doc_id |
|---|---|
| Part_1 | `1781746e90858653` |

## 4. The phone mask: wrong numbers masked, real phones missed (DA-F1-005) (4)

**What it is.** Two opposite errors.
- **Masked, but not a phone (2).** The pattern (2 to 4 digits, a separator, 3 to 4 digits, a separator, 3 to 4 digits) also matches parts of journal DOIs (`10.1034/j.1399-3054.1998...`) and runs of years (`1933 1934 1935`). Both became `<PHONE>`, which corrupts citations and tables.
- **A phone, but not masked (2).** International numbers with more groups than the pattern allows were masked only in part, leaving the last digits, or not at all. In a contact block, two staff phone numbers written without separators or in 4-2-4 groups were left as they are, next to the staff names (F1). The values are not copied here.

**How big it is.** 139 kept documents in the two parts gained a `<PHONE>` mask; in 25 of them the mask sits inside a DOI. The missed numbers were found by reading; they are not counted pack-wide.

**What to do.** Skip matches inside a DOI (after `10.NNNN/`) and runs of 4-digit years (now). For v1.1, also match whole international numbers (`00` or `+` and the country code, then several groups) and 10 to 11 digit numbers next to "Tel" or "Phone".

| Part | doc_id |
|---|---|
| Part_1 | `c28d662ea8889277` |
| Part_2 | `5fbc8b949d34abb0` |
| Part_1 | `5f7d6fc8766a9c26` |
| Part_1 | `78ad531b7e9e702f` |

## 4a. Boilerplate left after cleaning (C1) (4)

**What it is.** The cleaner removes some page furniture ("Skip to main content", "You May Also Like") but leaves the rest:
- a block of site links in the first lines of a page;
- a "Get the latest version of the Adobe Acrobat reader" footer;
- the related-article titles that followed a removed "You May Also Like" heading;
- a stray "- freely available" label at the top of a paper.

**What to do.** For v1.1: when a C1 rule removes a heading such as "You May Also Like" or "Related", also remove the short title lines that follow it. Add the Acrobat-reader line to the footer patterns.

| Part | doc_id |
|---|---|
| Part_1 | `7bdfe8559c7f7634` |
| Part_1 | `d8ee293d6c80f4cd` |
| Part_1 | `32118f1f0794bd21` |
| Part_1 | `c28d662ea8889277` |

## 5. Repeated section headings removed (DA-C2-003) (2)

**What it is.** Repeated-line removal also removes headings that legitimately repeat, once per section: "Geology and landforms" in a proposal with several wilderness units, "Demand of the question:" before each answer on a UPSC page. Only the first section keeps its heading.

**What to do.** Keep short heading-like lines (no end punctuation, under about 60 characters) when the lines between the repeats differ.

| Part | doc_id |
|---|---|
| Part_2 | `ffe9a2072a43aa0e` |
| Part_2 | `97cc11bc7799273a` |

## Smaller notes

- DA-B4-007 deleted an author's own sentence, "The text does not say.", from a Bible commentary: its pattern for rewriter meta-comments also matches ordinary prose. Small loss.
- DA-C5-005 removes "Keywords:" lines from papers and abstracts. Harmless, though they are metadata rather than SEO.

## Proposals, in order

| # | Rule | Change | When | Medicine (below) |
|---|---|---|---|---|
| 1 | DA-C3-003 | Count `”` and `’` as end punctuation | now (clear false positive) | `DA-C3-009` |
| 2 | DA-F1-005 | Do not mask inside DOIs or runs of years | now (clear false positive) | `DA-F1-010` |
| 3 | DA-C5-002 | Exempt numeric table or code-output lines | now (clear false positive) | `DA-C5-008` |
| 4 | DA-C3-003 | Trim the last line of a long document instead of dropping it | v1.1 | `DA-C3-009` |
| 5 | DA-D4-001 | Remove a short ad line inside a long document instead of dropping it | v1.1 | `DA-C1-015` removes the UKEssays ad before `DA-D4-001` runs; a general rule is still to write |
| 6 | DA-C2-003 | Keep repeated section headings | v1.1 | not written (2 docs) |
| 7 | DA-F1-005 | Mask whole international numbers and 10 to 11 digit numbers after "Tel" | v1.1 | not written (2 docs) |
| 8 | C1 rules | Remove the title lines after a removed "Related" heading; add the Acrobat-reader footer | v1.1 | `DA-C1-015` |

## Medicines

C1 and C3 are the two diseases this audit found in more than five docs (6 each), so each has a medicine (DISEASES.md section 4). Two more medicines fix the clear false positives of section 3 and 4, so that the fix for the freeze can be tested as code. The cards follow the form of `spec_v1_1/sheets/specv1_1.md`; three of them replace a spec v1 rule and keep its code, with one change each.

- **Rule form.** `doc = {doc_id, text, url, lang_label, pack_id}`, and `medicine(doc)` returns `(action, text)`. Each block checks `lang_label` and `pack_id`.
- **Run order.** `DA-C1-015` runs first in stage 30, before `DA-C3-001`, so a page-junk last line ("Share this page on") is gone before the truncation check. The three replacements take the place, in the English order, of the rule they replace.
- **Limit.** Gold docs changed plus dropped: at most `max(1, 1%)` = 1 doc (en-fineweb-edu gold, 158 docs).
- **IDs.** `DA-C1-015`, `DA-C3-009`, `DA-C5-008`, `DA-F1-010` are the next numbers unused by any Data A rule on TSAI main, on Vijay's branches and in the open audit PRs (checked 22 Nov).

| # | ID | disease | action | replaces | en-fineweb-edu gold |
|---|---|---|---|---|---|
| 1 | `DA-C1-015` | C1 page junk | fix | new | 0 changed, 0 dropped of 158 gold docs (0.00%): PASS |
| 2 | `DA-C3-009` | C3 truncated | drop, or fix (trim) | `DA-C3-003` | 0 changed, 0 dropped of 158 gold docs (0.00%): PASS |
| 3 | `DA-C5-008` | C5 keyword lines | drop | `DA-C5-002` | 0 changed, 0 dropped of 158 gold docs (0.00%): PASS |
| 4 | `DA-F1-010` | F1 personal data | fix | `DA-F1-005` | 0 changed, 0 dropped of 158 gold docs (0.00%): PASS |

All four in order: 0 changed, 0 dropped of 158 (0.00%): PASS.

**Speed.** Timed on every doc of both audited parts and the diagnosis pack (72,433 docs, 342.3 M characters), AMD Ryzen AI 7 PRO 350 w/ Radeon 860M (CPU), Python 3.12.10, Windows 11, one process, from a cold start (210.3 s for the whole run). Each replacement is timed against the rule it replaces, from `spec_v1/clean.py`; `DA-C1-015` is new, so it is timed against a line-by-line loop version of itself, and no docs differ. A second run on every text a medicine fixed changes 0 docs, so all four are idempotent on these files. On the two parts, `DA-C1-015` is 5.8x to 6.4x faster than its loop version, because its key-phrase gate skips most docs. `DA-C5-008` (0.9x to 1.1x) and `DA-F1-010` (1.0x) are about as fast as the rules they replace (a ratio under 1 is slower). `DA-C3-009` is 0.7x to 0.9x of `DA-C3-003`: it trims instead of dropping, which costs +0.02 s over both parts. No medicine loops in Python over a doc's tokens or characters; the loops that remain are over a few regex matches (`DA-C5-008`'s candidate keyword lines) or are the replaced rule's own (`DA-F1-010` keeps `DA-F1-005`'s loop over lines that hold a digit or `@`). There is no GPU code, so there is no host-device sync.

| file (docs) | `DA-C1-015` vs loop version | `DA-C3-009` vs DA-C3-003 | `DA-C5-008` vs DA-C5-002 | `DA-F1-010` vs DA-F1-005 | all four, in order |
|---|---|---|---|---|---|
| Part_1/part-00000.parquet (54,004) | 13.64 → 2.37 s (5.8x) | 0.07 → 0.09 s (0.9x) | 14.34 → 15.47 s (0.9x) | 29.14 → 30.44 s (1.0x) | 49.53 s |
| Part_2/part-00001.parquet (17,929) | 5.33 → 0.84 s (6.4x) | 0.02 → 0.02 s (0.7x) | 5.50 → 4.97 s (1.1x) | 9.46 → 9.45 s (1.0x) | 15.41 s |
| pack/en-fineweb-edu.parquet (500) | 0.14 → 0.03 s (5.2x) | 0.00 → 0.00 s (1.0x) | 0.16 → 0.13 s (1.2x) | 0.26 → 0.25 s (1.1x) | 0.42 s |

What each one does on the same files, next to the rule it replaces (counts of docs):

| file | rule | old rule: fix / drop | medicine: fix / drop | docs that differ |
|---|---|---|---|---|
| Part_1/part-00000.parquet | `DA-C1-015` | — | 102 / 0 | 0 (vs its loop version) |
| Part_1/part-00000.parquet | `DA-C3-009` | 0 / 296 | 150 / 132 | 164 |
| Part_1/part-00000.parquet | `DA-C5-008` | 0 / 39 | 0 / 35 | 4 |
| Part_1/part-00000.parquet | `DA-F1-010` | 141 / 0 | 121 / 0 | 20 |
| Part_2/part-00001.parquet | `DA-C1-015` | — | 39 / 0 | 0 (vs its loop version) |
| Part_2/part-00001.parquet | `DA-C3-009` | 0 / 107 | 42 / 51 | 56 |
| Part_2/part-00001.parquet | `DA-C5-008` | 0 / 17 | 0 / 14 | 3 |
| Part_2/part-00001.parquet | `DA-F1-010` | 42 / 0 | 33 / 0 | 9 |
| pack/en-fineweb-edu.parquet | `DA-C1-015` | — | 2 / 0 | 0 (vs its loop version) |
| pack/en-fineweb-edu.parquet | `DA-C3-009` | 0 / 1 | 1 / 0 | 1 |
| pack/en-fineweb-edu.parquet | `DA-C5-008` | 0 / 0 | 0 / 0 | 0 |
| pack/en-fineweb-edu.parquet | `DA-F1-010` | 0 / 0 | 0 / 0 | 0 |

**Checks and logs.**
- Script: [`medicine_fineweb_checks.py`](medicine_fineweb_checks.py), next to this report. It reads the four python blocks below, as `harm_test.py` does, and has two checks:
  - `speed`: timing against the replaced rules, counts, and idempotence on whole files. Log: [`2026-11-22-fineweb-audit-medicines-speed-local.txt`](../jobs/logs/2026-11-22-fineweb-audit-medicines-speed-local.txt), numbers in [`2026-11-22-fineweb-audit-medicines-summary-local.json`](../jobs/logs/2026-11-22-fineweb-audit-medicines-summary-local.json).
  - `treated`: what each medicine does to the 35 audit docs. Log: [`2026-11-22-fineweb-audit-medicines-treated-local.txt`](../jobs/logs/2026-11-22-fineweb-audit-medicines-treated-local.txt).
- Tests: [`tests/audit/test_medicine_fineweb_checks.py`](../tests/audit/test_medicine_fineweb_checks.py), made-up texts only.
- Harm test: `harmtest/harm_test.py --pack en-fineweb-edu.parquet --gold gold/en-fineweb-edu_gold.csv --rules teams/01-data-a/audit/en-fineweb-edu-audit-Deepjyoti_Saha.md --rule all`. Log: [`2026-11-22-fineweb-audit-medicines-harm-local.txt`](../jobs/logs/2026-11-22-fineweb-audit-medicines-harm-local.txt).

## Medicine 1 — english / C1 / `DA-C1-015`

**Consolidation:** Source: this audit (35 docs); identity `DA-C1-015`, new. Counts are this audit's label counts, one reader.

1. **Disease:** C1, 6 docs. Examples: `eeadc959d809b46e` (a complete article whose last line is "Share this page on"), `d8ee293d6c80f4cd` ("Get the latest version of the Adobe Acrobat reader..." footer), `bdacc614dc1007a6` (a three-line essay-service ad inside a 21,000-character write-up), `32118f1f0794bd21` (a trailing "You May Also Like" with a run-on line of article titles before it), `c28d662ea8889277` ("- freely available" at the top), `7bdfe8559c7f7634` (Wikipedia's "This section needs additional citations" banner).

<!-- CARD EVIDENCE -->
**Current gold harm:** en-fineweb-edu: 0 changed, 0 dropped of 158 gold docs (0.00%): PASS.

**Examples:** the doc_ids and short quotes in item 1. Exact source characters are not published here, because pack text stays out of the public repo.
<!-- END CARD EVIDENCE -->

2. **Medicine:**
   - **What it fixes.** It deletes whole lines of page junk that this audit found: "Share this page (on)", the Acrobat-reader footer, "- freely available", and Wikipedia's citation banners. It deletes UKEssays' three-line ad ("Get your grade / or your money back / using our Essay Writing Service!") only as the three lines together. It deletes a trailing "You May Also Like" heading with the line just before it when that line has no end punctuation (the run-on titles).
   - **Why it runs first.** Two of these lines decided a drop: "Share this page on" ends in "on", so `DA-C3-003` dropped the article, and the essay-service ad made `DA-D4-001` drop a real write-up. With the junk gone first, both docs are kept.
   - **What it leaves alone.** Any line with other text on it; the infobox rows at the top of a Wikipedia page (they hold facts).
   - **What it can't fix.** Junk of sites this audit did not see.

```python
RULE_ID = "DA-C1-015"
# page junk left after spec v1, found by the en-fineweb-edu audit; runs first in stage 30 (before DA-C3-001)
SOURCE_RULE_ID = "DA-C1-015"  # new rule
CODE = "C1"
ACTION = "fix"
STAGE = "30"
STATUS = "CANDIDATE"
LANGS = {"en", "eng"}
ORIGIN = "en-fineweb-edu"  # pack whose audit found it (Deepjyoti Saha, 22 Nov); it runs on SPEC_PACKS
SOURCE = "en-fineweb-edu"
AUTHOR = "deepjyotisaha"
import re

# whole lines of page junk: every copy goes, with its line break
C1_LINE = re.compile(
    r"(?mi)^[ \t]*(?:"
    r"share this page(?: on| via| with)?:?|"
    r"get the latest version of the adobe acrobat reader[^\n]*|"
    r"-?[ \t]*freely available|"
    r"\|?this (?:section|article) (?:needs additional citations|does not cite any sources)[^\n]*"
    r")[ \t]*$\n?")
# UKEssays' inline ad: the three lines together
C1_AD = re.compile(r"(?mi)^get your grade[ \t]*\nor your money back[ \t]*\nusing our essay writing service!?[ \t]*$\n?")
# a trailing "You May Also Like" heading, with the run-on line of article titles just before it
# (that line has no end punctuation; a sentence before the heading is kept)
C1_RELATED = re.compile(r"(?i)(?:\n[^\n]*[^\s.!?\"'”’):])?\n[ \t]*you may also like[ \t]*\s*\Z")
# Speed: one lower() and a few `in` scans (C); the regexes run only on the few docs that hold a key phrase.
_KEYS = ("share this page", "acrobat reader", "freely available", "needs additional citations",
         "does not cite any sources", "essay writing service", "you may also like")


def medicine_c1(text):
    low = text.lower()
    if not any(k in low for k in _KEYS):
        return text
    if "essay writing service" in low:
        text = C1_AD.sub("", text)
    if "you may also like" in low:
        text = C1_RELATED.sub("", text)
    return C1_LINE.sub("", text)


def _std(out, text):
    # the result in the shared form: str -> fix/keep; an empty result is a drop
    if not out.strip(): return ("drop", text)
    return ("fix" if out != text else "keep", out)


def medicine(doc):
    if doc.get("lang_label") not in LANGS:
        return ("keep", doc["text"])
    return _std(medicine_c1(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'en-fineweb-edu'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    return _spec_source_medicine(doc)
```

3. **Action:** fix. **Runs on:** en-fineweb-edu. **Audit docs it changes:** 6 (`7bdfe8559c7f7634`, `d8ee293d6c80f4cd`, `eeadc959d809b46e`, `bdacc614dc1007a6`, `32118f1f0794bd21`, `c28d662ea8889277`); 6 of them are labelled C1. It leaves `32118f1f0794bd21`'s glued "Advertisement" prefix, which is inside a sentence line.
4. **Harm test:** 0 changed, 0 dropped of 158 gold docs (0.00%): PASS.
5. **Side effects:**
   - A page whose own text is a line "Share this page" (a how-to about sharing) loses that line.
   - The "You May Also Like" part removes one line before the heading; if that line is a real last sentence without end punctuation, it goes too.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. New rule, not in spec v1.1.

---

## Medicine 2 — english / C3 / `DA-C3-009`

**Consolidation:** Source: this audit; identity `DA-C3-009`, replaces `DA-C3-003` (en-dclm's medicine 32) and keeps its code, with two changes. Counts are this audit's label counts, one reader.

1. **Disease:** C3, 6 docs; and `DA-C3-003`'s drops: 392 of the 559 drops in the two parts, 4 of the 9 sampled ones wrong. Examples: `40fb0256c0b2c8fe` (a complete news article ending `worked on.”`), `0309ba9879105216` (a complete guide whose last line ends `germination. A`), `5716019c1d54b86c` (a manual cut off at `heel. That`).

<!-- CARD EVIDENCE -->
**Current gold harm:** en-fineweb-edu: 0 changed, 0 dropped of 158 gold docs (0.00%): PASS.

**Examples:** the doc_ids and short quotes in item 1. Exact source characters are not published here, because pack text stays out of the public repo.
<!-- END CARD EVIDENCE -->

2. **Medicine:**
   - **What it fixes.** (a) A text ending in a curly closing quote (`”`, `’`) or `»` is not cut off: `DA-C3-003` lists only the straight quotes. (b) A text of 2,000+ characters whose tail is cut off is trimmed, not dropped, when the rest is prose (10+ sentence ends): the last line is cut back to its last sentence end, or, with none, removed.
   - **What it keeps from `DA-C3-003`.** The dangling-word test, and the drop of a text that ends in "[…]" or is short.
   - **What it leaves alone.** A long list whose last item is cut off: with no sentence end in the last line and a list mark at its start, it stays a drop (a list of links or titles).
   - **What it can't fix.** A text cut in the middle of a long last line with no sentence end before the cut keeps losing that whole line.

```python
RULE_ID = "DA-C3-009"
# replaces DA-C3-003: curly quotes end a text; a long prose text with a cut-off tail is trimmed, not dropped
SOURCE_RULE_ID = "DA-C3-003"
CODE = "C3"
ACTION = "drop"  # or fix: a long prose text is trimmed
STAGE = "30"
STATUS = "CANDIDATE"
LANGS = {"en", "eng"}
ORIGIN = "en-fineweb-edu"  # pack whose audit found it (Deepjyoti Saha, 22 Nov); it runs on SPEC_PACKS
SOURCE = "en-fineweb-edu"
AUTHOR = "deepjyotisaha"
import re

_MARKER = re.compile(r"\[\s*(?:…|\.\.\.)\s*\]$")

_DANGLING = frozenset(
    "a an the and or but of to in on at for with from by as is are was were "
    "that this these those which who because however".split()
)

_TOKEN = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789'")  # the characters of [A-Za-z0-9']
_LONGEST = max(map(len, _DANGLING))  # a last token longer than every dangling word cannot be one

# end punctuation: DA-C3-003's list, plus the curly closing quotes and the guillemet that web text uses
_ENDS = (".", "!", "?", '"', "'", ")", ":", "…", "”", "’", "»")
LONG = 2000      # a document this long is trimmed, not dropped, when only its tail is cut off
PROSE = 10       # ... and when the rest is prose: at least this many sentence ends (". " or ".\n")
# the last sentence end in a line, with any closing quotes or brackets after it
_SENT_END = re.compile(r"[.!?…][\"'”’»)\]]*(?=\s|$)")


def _dangles(tail):
    """DA-C3-003's test: the text ends in a dangling word and not in end punctuation."""
    if tail.endswith(_ENDS):
        return False
    end = len(tail)
    while end and tail[end - 1] not in _TOKEN:
        end -= 1
    start = end
    while start and tail[start - 1] in _TOKEN and end - start <= _LONGEST:
        start -= 1
    return end - start <= _LONGEST and tail[start:end].lower() in _DANGLING


def _trim(tail):
    """Cut the last line back to its last sentence end; with none, remove the last line, unless it is a list item
    (then the text is a list of links or titles, and stays a drop)."""
    head, _, last = tail.rpartition("\n")
    ends = list(_SENT_END.finditer(last))
    if ends:
        return (head + "\n" if head else "") + last[:ends[-1].end()]
    if last.lstrip().startswith(("-", "*", "•", "·")):
        return None
    return head.rstrip()


def medicine_c3(text):
    tail = text.rstrip()
    if not tail:
        return text
    if tail.endswith("]") and _MARKER.search(tail):  # "[…]" ends the text: a teaser, dropped as before
        return ("drop", "")
    if not _dangles(tail):
        return text
    if len(tail) >= LONG:
        cut = _trim(tail)
        if cut and not _dangles(cut) and cut.count(". ") + cut.count(".\n") >= PROSE:
            return cut
    return ("drop", "")


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

SPEC_PACKS = {'en-dclm', 'en-fineweb-edu', 'en-nemotroncc', 'en-nemotroncc-synth'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    return _spec_source_medicine(doc)
```

3. **Action:** drop, as before, or fix when it trims. **Runs on:** the four English packs, in `DA-C3-003`'s place. **Audit docs:** of the 9 docs `DA-C3-003` dropped, it keeps or trims 3 and still drops 6. It keeps 3 of the 4 whose drop looked wrong, and still drops 5 of the 5 whose drop looked right. The one wrong drop it still makes alone is `eeadc959d809b46e` (1,246 characters, under the 2,000 limit); in run order `DA-C1-015` removes its "Share this page on" line first, and it is kept.
4. **Harm test:** 0 changed, 0 dropped of 158 gold docs (0.00%): PASS.
   **On the two whole parts:** `DA-C3-003` drops 403 docs on its own; `DA-C3-009` still drops 183, trims 192 and keeps 28. Only the 9 sampled ones were read, so the trims need a spot-check before the rule is adopted.
5. **Side effects:**
   - A long prose text that really is a fragment is kept, trimmed, instead of dropped.
   - Trimming the last line can lose the last real sentence when it has no end punctuation.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Replaces `DA-C3-003`; for the team to decide.

---

## Medicine 3 — english / C5 / `DA-C5-008`

**Consolidation:** Source: this audit; identity `DA-C5-008`, replaces `DA-C5-002` (en-dclm's medicine 33) and keeps its code, with one change.

1. **Disease:** C5 keyword lines; the false positive is `1781746e90858653`, an SQLite tutorial whose query output (`1|22.1|2013-10-08 2|2.1|...`) is one line of 7+ `|`-separated fields. `DA-C5-002` makes 56 of the 559 drops in the two parts.

<!-- CARD EVIDENCE -->
**Current gold harm:** en-fineweb-edu: 0 changed, 0 dropped of 158 gold docs (0.00%): PASS.

**Examples:** the doc_ids and short quotes in item 1. Exact source characters are not published here, because pack text stays out of the public repo.
<!-- END CARD EVIDENCE -->

2. **Medicine:**
   - **What it fixes.** A line of 7+ `|`-separated fields counts as a keyword line only when fewer than half of its fields hold a digit. Table rows and query output are mostly numbers; tag clouds are words.
   - **What it keeps from `DA-C5-002`.** The nine spam patterns (two must match) and the rest of the keyword-line test.
   - **What it can't fix.** A keyword line made of numbers (lists of years or prices) is kept.

```python
RULE_ID = "DA-C5-008"
# replaces DA-C5-002: a line of mostly numeric |-fields is a table, not a keyword line
SOURCE_RULE_ID = "DA-C5-002"
CODE = "C5"
ACTION = "drop"
STAGE = "30"
STATUS = "CANDIDATE"
LANGS = {"en", "eng"}
ORIGIN = "en-fineweb-edu"  # pack whose audit found it (Deepjyoti Saha, 22 Nov); it runs on SPEC_PACKS
SOURCE = "en-fineweb-edu"
AUTHOR = "deepjyotisaha"
import re

_SPAM_PATTERNS = (
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
_SPAM = tuple(re.compile(p, re.I) for p in _SPAM_PATTERNS)
_SPAM_ANY = re.compile("|".join(_SPAM_PATTERNS), re.I)  # matches exactly when one of the nine does

_KEYWORD_LINE = re.compile(r"^(?![ \t]*[•·*\-])(?:[^|\n]{2,40}\|){6,}[^|\n]{2,40}[ \t]*$", re.M)

_DIGIT = re.compile(r"\d")


def _keyword_line(text):
    """A line of 7+ "|"-separated fields is a keyword line, unless most of its fields hold a digit:
    then it is a table or query output ("1|22.1|2013-10-08 2|2.1|..."), not keywords."""
    for m in _KEYWORD_LINE.finditer(text):
        fields = m.group().split("|")
        if 2 * sum(1 for f in fields if _DIGIT.search(f)) < len(fields):
            return True
    return False


def medicine_c5(text):
    # Two different patterns must match to drop. Almost no document matches even one, so one search of their union
    # comes first and the nine run one by one only when it matches. A keyword line needs 6 or more "|" characters.
    hits = sum(1 for p in _SPAM if p.search(text)) if _SPAM_ANY.search(text) else 0
    if hits >= 2 or (text.count("|") >= 6 and _keyword_line(text)):
        return ("drop", "")
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
    return _std(medicine_c5(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'en-dclm', 'en-fineweb-edu', 'en-nemotroncc', 'en-nemotroncc-synth'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    return _spec_source_medicine(doc)
```

3. **Action:** drop. **Runs on:** the four English packs, in `DA-C5-002`'s place. **Audit docs:** `1781746e90858653` (old → new: drop → keep).
   **On the two whole parts:** `DA-C5-002` drops 56 docs; `DA-C5-008` drops 49 and keeps the other 7. Only the sampled one was read.
4. **Harm test:** 0 changed, 0 dropped of 158 gold docs (0.00%): PASS.
5. **Side effects:** a spam page whose keyword line mixes numbers in (model numbers, years) is no longer dropped by this test.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Replaces `DA-C5-002`; a clear false positive, small enough for the freeze.

---

## Medicine 4 — english / F1 / `DA-F1-010`

**Consolidation:** Source: this audit; identity `DA-F1-010`, replaces `DA-F1-005` (the instructor's English F1 rule) and keeps its code, with one change.

1. **Disease:** F1 masking, in reverse: digits that are not a phone number are masked as `<PHONE>`. Examples: `c28d662ea8889277` (parts of journal DOIs, `10.1034/j.1399-3054.1998...`), `5fbc8b949d34abb0` (runs of years, `1933 1934 1935`). In the two parts, 25 kept docs have a `<PHONE>` inside a DOI.

<!-- CARD EVIDENCE -->
**Current gold harm:** en-fineweb-edu: 0 changed, 0 dropped of 158 gold docs (0.00%): PASS.

**Examples:** the doc_ids and short quotes in item 1. Exact source characters are not published here, because pack text stays out of the public repo.
<!-- END CARD EVIDENCE -->

2. **Medicine:**
   - **What it fixes.** A phone-pattern match is left as it is when it is a run of years (two or more 4-digit years, 1500 to 2099) or when it sits inside a DOI (a `10.NNNN/` token before it, within 80 characters, with no space).
   - **What it keeps from `DA-F1-005`.** Everything else: emails, addresses, the organisation and public-role contexts, toll-free numbers.
   - **What it can't fix.** The phones `DA-F1-005` misses (international numbers with more groups, numbers without separators, `78ad531b7e9e702f`, `5f7d6fc8766a9c26`). That needs a wider phone pattern, which can mask more non-phones; it is left for v1.1.

```python
RULE_ID = "DA-F1-010"
# replaces DA-F1-005: digits in a DOI and runs of years are not masked as <PHONE>
SOURCE_RULE_ID = "DA-F1-005"
CODE = "F1"
ACTION = "fix"
STAGE = "40"
STATUS = "CANDIDATE"
LANGS = {"en", "eng"}
ORIGIN = "en-fineweb-edu"  # pack whose audit found it (Deepjyoti Saha, 22 Nov); it runs on SPEC_PACKS
SOURCE = "en-fineweb-edu"
AUTHOR = "deepjyotisaha"
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

TOLL_FREE = re.compile(r"^(?:\+?1[\s.-]?)?\(?8(?:00|88|77|66|55|44|33)\)?")

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

# not a phone number: a run of years ("1933 1934 1935"), or digits inside a DOI ("10.1034/j.1399-3054.1998...")
YEARS = re.compile(r"(?:1[5-9]|20)\d\d(?:[\s.\-‐‑–]+(?:1[5-9]|20)\d\d)+")
DOI_BEFORE = re.compile(r"\b10\.\d{4,9}/\S*$")


def _not_phone(m):
    if YEARS.fullmatch(m.group()):
        return True
    return DOI_BEFORE.search(m.string, max(0, m.start() - 80), m.start()) is not None

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
    org_page = bool(ORG_VOICE.search(text))
    has_org = {}  # line number -> an organisation word on that line: searched once, and only when a contact line asks

    def org_near(i):  # same answer as searching the joined 13-line window, as no match spans a newline
        for j in range(max(0, i - 6), min(len(lines), i + 7)):
            if j not in has_org:
                has_org[j] = ORGANISATION.search(lines[j]) is not None
            if has_org[j]:
                return True
        return False

    out = []
    for i, line in enumerate(lines):
        if "@" not in line and not DIGIT.search(line):  # no EMAIL, PHONE or ADDRESS can match
            out.append(line)
            continue
        new = EMAIL.sub(lambda m: _email(m, org_page or org_near(i) or bool(PUBLIC_ROLE.search(
            "\n".join(lines[max(0, i - 2):i + 1])))), line) if "@" in line else line
        if PHONE.search(new) or ADDRESS.search(new):
            if not (org_page or org_near(i)):
                new = PHONE.sub(lambda m: m.group() if TOLL_FREE.match(m.group()) or _not_phone(m) else "<PHONE>", new)
                if len(line.strip()) <= CONTACT_LINE:
                    new = ADDRESS.sub("<ADDRESS>", new)
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

SPEC_PACKS = {'en-dclm', 'en-fineweb-edu', 'en-nemotroncc', 'en-nemotroncc-synth'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    return _spec_source_medicine(doc)
```

3. **Action:** fix. **Runs on:** the four English packs, in `DA-F1-005`'s place (last in the English order). **Audit docs:** `5f7d6fc8766a9c26` (fix → fix), `bca462c24c2c6cda` (fix → fix), `c28d662ea8889277` (fix → keep), `5fbc8b949d34abb0` (fix → keep).
   **On the two whole parts:** `DA-F1-005` changes 183 docs; `DA-F1-010` changes 154, and its output differs in 29 (the masks it no longer makes). Only the sampled ones were read.
4. **Harm test:** 0 changed, 0 dropped of 158 gold docs (0.00%): PASS.
5. **Side effects:** a real phone number written like a run of years, or inside a URL that looks like a DOI, is not masked.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Replaces `DA-F1-005`; a clear false positive, small enough for the freeze.
