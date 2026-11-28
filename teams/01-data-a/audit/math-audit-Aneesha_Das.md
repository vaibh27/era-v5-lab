# Math packs audit: math-finemath and math-megamath

Reviewer: Aneesha Das (`aneeshads`). Date: 20 Nov 2026. Assignment 11: 22 docs from math-finemath Part_1 and 13 docs from math-megamath Part_1.

Each doc was read side by side in the pack viewer: the raw text (`Cap1-packs`) and the cleaned text (`tokenizer_sample_v0_clean/train`). Each doc is labelled `clean`, `fix` or `drop`, with disease codes from [DISEASES.md](../../../diagnosis/DISEASES.md) and a note. The labels are in this folder:

- [`math-finemath-Part_1-Aneesha_Das.csv`](math-finemath-Part_1-Aneesha_Das.csv) (22 rows)
- [`math-megamath-Part_1-Aneesha_Das.csv`](math-megamath-Part_1-Aneesha_Das.csv) (13 rows)

The cleaner did many things right (email masking, `[math]` to `$`, ligatures, mojibake, HTML line breaks). But only 4 of the 35 docs are clean after cleaning. The main problems, most serious first:

1. **Cleaning removed real content.** In megamath, reference entries with six or more authors are deleted, leaving empty numbered slots: 421 lines in 70 docs across the pack. In finemath, the rule that removes repeated lines also removes repeated formula lines from worked solutions.
2. **Personal data left in.** Emails written as `name (at) gmail (dot) com` or `name at domain dot com` are not masked: 840 in 37 megamath docs. Phone numbers and an office address are also left.
3. **Boilerplate left in (C1), in 26 of 35 docs.** When a junk line repeats, the cleaner often removes every copy but the first. Junk that appears once (footers, signatures, site buttons) is not touched.
4. **Spam and non-math pages kept.** 7 docs the cleaner kept should be dropped.

## Inputs and method

| Input | File | SHA-256 |
|---|---|---|
| Raw finemath | `Cap1-packs/math-finemath/Part_1/part-00000.parquet` | `a5dbb4da6d0c2e890901aecb10b5c2bd71378d8e66e5975adc0d7d98592f51b2` |
| Cleaned finemath | `tokenizer_sample_v0_clean/train/math-finemath/part-00000.parquet` | `7b3366baf9b8b756eb8b8579940918cbc4cd5eac7b54394e639a722ce117d339` |
| Raw megamath | `Cap1-packs/math-megamath/Part_1/part-00000.parquet` | `e8c53baf0e1c842d303199b93cc36768199edbc64a142925df890915c4cb7d91` |
| Cleaned megamath | `tokenizer_sample_v0_clean/train/math-megamath/part-00000.parquet` | `152c811a29bdd254f4c8778553222b6dcc5df04081267bba69d82345c3c87715` |

- **Viewer:** `diagnosis/tools/view_packs_hub.py` on branch `g1-web-viewer-pre-post` of `patilvijay23/era-v5-lab`, commit `be1571f`, with "Only changed" and "Random next" on.
- **Sample:** only docs the cleaner changed or dropped. One finemath doc was picked in the viewer. The other 34 were drawn at random, with a fixed seed, from the changed or dropped docs of each pack.

  | Pack | Docs | Changed | Dropped by cleaner | Reviewed |
  |---|---:|---:|---:|---:|
  | math-finemath | 29,122 | 8,361 | 600 | 22 |
  | math-megamath | 17,647 | 1,739 | 13 | 13 |

- **Reading:** 34 docs read in full. One doc (`01cbee543c7751fd`, 31,000 characters of SEO spam) was read at the start, middle and end, and is marked `partial`.
- **Labels:** `read`, `verdict`, `diseases` and `note` are filled in. `evidence` and `hex` are left empty, as the audit instructions ask. Offsets in the notes are character positions in the cleaned text ("post offset").
- **How the labels were made:** drafted with an AI assistant (Claude) from the full raw and cleaned text, then checked by the reviewer.
- **Whole-pack counts:** the "How big it is" numbers below come from a short script that compares the raw and cleaned text of every doc in the pack, line by line. They are pattern counts, not hand-checked labels.
- **Personal data:** no email address, phone number or address is copied into this report or the label files. Only the position is given.

This audits the `tokenizer_sample_v0_clean` copy named in the audit instructions. The [D3 report](https://github.com/theschoolofai/era-v5-lab/blob/main/teams/01-data-a/audit/day_D3.md) notes that this older copy can differ from the frozen p1536 shard output, so these findings have not been checked against the shards. 35 docs is a small sample, not a quality estimate for the whole pack.

## Summary

| Pack | Reviewed | Dropped by cleaner | clean | fix | drop |
|---|---:|---:|---:|---:|---:|
| math-finemath | 22 | 3 | 4 | 12 | 6 |
| math-megamath | 13 | 0 | 0 | 9 | 4 |
| **Total** | **35** | **3** | **4** | **21** | **10** |

`drop` includes the 3 docs the cleaner already dropped (the reviewer agrees). The other 7 `drop` docs were kept by the cleaner and should be dropped.

| Disease | finemath | megamath | Docs |
|---|---:|---:|---:|
| C1 boilerplate | 16 | 10 | 26 |
| C2 repeated lines | 1 | 5 | 6 |
| C5 keyword lists, SEO spam | 3 | 2 | 5 |
| C3 truncated (all three dropped Course Hero previews) | 3 | 0 | 3 |
| C4 several documents joined | 1 | 2 | 3 |
| A3 mojibake | 1 | 2 | 3 |
| F1 personal data | 0 | 3 | 3 |
| A4 replacement character U+FFFD | 1 | 1 | 2 |
| B2 markup tags | 1 | 1 | 2 |
| B5 broken LaTeX | 0 | 2 | 2 |
| E2 data blob (raw PDF bytes) | 0 | 1 | 1 |
| Content removed by cleaning (no code fits; in the note) | 1 | 3 | 4 |

A doc can have several diseases, so the counts overlap. C1 and C2 are each found more than five times, so each has a medicine, in [Medicines](#medicines) (DISEASES.md section 4).

## Content removed by cleaning (4 docs)

**What it is.** Cleaning deleted real content. No disease code fits damage done by cleaning to English text (the G codes are for Indic), so these rows describe it in the note.

**Reference entries with six or more authors are deleted.** The entry is removed but its number stays, so the reference list has empty slots. In `f079a0d75820b184`, 12 entries are gone (slots 3–8, 10–11, 13–15 and 19), including reference 4, which the Methods section points to for the growth details. In `0cb915d7762c2e45`, the one deleted entry is the paper's own method, cited 4 times in the text. In `27c5838098758c5b`, two article entries lost their author lines.

**How big it is.** Across math-megamath, 421 reference lines with six or more authors were removed, in 70 docs. In math-finemath, 1 line.

**Repeated formula lines are deleted.** In `89dc8e7278ce1f4b`, the formula line `P(E) = n(E)/ n(S)` starts each worked probability solution. The rule that removes repeated lines kept the first copy and deleted the other 4, so solutions Q12 to Q15 now go from `n(E)` straight to the answer.

**How big it is.** Across math-finemath, 755 docs had a repeated line containing `=`, `^`, `$` or a LaTeX command deleted while one copy was kept (5,412 lines). Not all of these are formulas, so this is an upper bound that needs a spot-check. In math-megamath, the repeated-line rule changed only 3 docs.

**Email masking also removes nearby words.** In `0ef68457f8488afa`, the mask swallowed the last word of many comments together with the commenter name (`way better <EMAIL>`). The doc is a drop anyway, but the same rule runs on every doc.

**What to do.** Run the current `clean.py` on these docs to find the rule that deleted each line. Then:
- Exempt reference lists from the rule that removes author lists.
- Limit the repeated-line rule to boilerplate-like lines (short, no math symbols), or to lines repeated next to each other. Medicine 2 (`DA-C2-007`) keeps every copy of a line with a math sign.
- Make the email pattern stop at word boundaries.

## Personal data left in (F1) (3 docs)

**What it is.** Plain emails were masked to `<EMAIL>` correctly in every doc where they appeared. But emails written in a disguised form were not, and phone numbers were not masked at all.

- `4258f5c9fbf14805`: 73 emails written as `(at) … (dot) com` or `at … dot com`, first at post offset 3070, with commenters' full names and social-media links.
- `31c47797e0ec552c`: a 1996 mailing-list signature with a name, an office address and a phone number (post offset 1669).
- `27c5838098758c5b`: a research group's office phone number (post offset 390). This is a judgement call: it is a work number on a public page.

**How big it is.** Across math-megamath, 840 disguised emails are left in 37 docs. Across math-finemath, 7 in 6 docs.

**What to do.** Extend the F1 rule to the `(at)`/`(dot)` and `at … dot` forms and to phone numbers. Check it against the gold set for harm before adopting it.

## Boilerplate left in (C1) (26 docs)

**What it is.** Site chrome is left in the cleaned text: forum badges, sign-in prompts, Q&A site buttons, archive headers and footers, empty page labels from journal sites, and signatures.

**Pattern.** In 5 docs, the cleaner removed every copy of a repeated junk line except the first:

| doc_id | Junk line | Removed |
|---|---|---|
| `03adce6b428657cc` | `Quick Reference:` | 5 of 6 |
| `b1efb7907a0c8cbb` | `Please Log in or Create an account to join the conversation.` | 9 of 10 |
| `caec68d5a4dc4373` | `Homework Helper` | 10 of 11 |
| `e45925da5672827e` | `Standard: MATH 1` | 5 of 6 |
| `73670428f7373dee` | `by Subhotosh Khan` | 4 of 5 |

Junk that appears only once is not touched: `Join BYJU'S Learning Program`, `Post a followup to this message`, `Moderators: …`, wiki `[edit]` labels, and journal labels such as `Received / Accepted / Published`.

**What to do.** When a line is judged boilerplate, remove every copy, not all but one. Add patterns for the site chrome above. Medicine 1 (`DA-C1-012`) does both.

## Spam and non-math pages kept (7 docs)

**What it is.** The cleaner kept these docs and made only small fixes. They should be dropped.

| Pack | doc_id | Why drop |
|---|---|---|
| math-finemath | `73670428f7373dee` | Forum search results: thread titles, views and last-post dates, no math (C5) |
| math-finemath | `e45925da5672827e` | Worksheet catalogue, almost no math; 2 U+FFFD (C5, A4) |
| math-finemath | `01cbee543c7751fd` | SEO spam stitched from many sites; ends with a clothing ad (C5, C4) |
| math-megamath | `41f03ba56962c1e9` | SEO sentence soup, 51,000 characters in 2 lines (C5, C4, C2) |
| math-megamath | `9cb311703c53c1e0` | SEO spam mixed with raw PDF bytes; 179 U+FFFD (C5, C4, E2, A4) |
| math-megamath | `4258f5c9fbf14805` | Kids' film giveaway, not math; 73 unmasked emails (F1) |
| math-megamath | `0ef68457f8488afa` | Raw Blogger feed of a parent discussion on school placement |

**What to do.** These need a doc-level rule (keyword density, share of PDF syntax or U+FFFD, topic). Check any new drop rule against the gold set.

## Characters and markup left in (8 docs)

- **A3 mojibake:**
  - `94c0014436db12b5`: one mojibake was fixed, but the broken emoji `đź™‚` is left at post offset 9633.
  - `9cb311703c53c1e0`: `FØlix Muæoz-García`.
  - `41f03ba56962c1e9`: `ááá`, `¥` and `آ` from PDF fonts.
- **A4 U+FFFD:**
  - `e45925da5672827e`: 2.
  - `9cb311703c53c1e0`: 179.
- **B2 markup:**
  - `caec68d5a4dc4373`: a BBCode `[ATTACH]` tag.
  - `32d96679ed59c9d5`: a MediaWiki diff header with raw `<math>` tags, `'''bold'''` markup and 44 lines of `− | |||`.
- **B5 broken LaTeX:**
  - `0cb915d7762c2e45`: `\left{c}_{\ell}\right}` lost its `|`.
  - `31c47797e0ec552c`: a broken TeX accent `Riec\u can`.
- **E2:** `9cb311703c53c1e0` contains raw PDF object syntax and bytes.

## Dropped by cleaning (3 docs, finemath)

All three are Course Hero paywall previews. The drop is right. They are labelled `drop` with C1 and C3, and their notes start "No matching doc_id".

| doc_id | Verdict | Note |
|---|---|---|
| `b5b886b63aeb9e2e` | drop | Signals homework preview; formulas garbled in extraction; cut off after page 5 |
| `67f407a0ccd09d74` | drop | Finance lecture problems; `{[ promptMessage ]}` template junk and paywall text. Borderline: the worked solutions were usable |
| `6df989a2bcd44389` | drop | Fluids textbook preview; equations lost their symbols; testimonials |

## What the cleaner did right

- Plain emails masked to `<EMAIL>` (megamath), including in C code headers.
- `[math]…[/math]` converted to `$…$` (179 pairs in `32d96679ed59c9d5`).
- Ligatures `ﬁ`, `ﬀ` and `ﬂ` replaced with `fi`, `ff` and `fl`.
- Mojibake `â€™` fixed to `’`.
- `<br />` turned into line breaks and `<i>` removed.
- Zero-width spaces removed.
- Jiskha `posted by .` lines removed.
- A paragraph repeated inside a forum quote removed, which does no harm (`11c56a701b6c6f82`, `c87c76a202bf9370`).

## Per-doc findings

### math-finemath (22)

| doc_id | Cleaner | Verdict | Diseases | Finding |
|---|---|---|---|---|
| `caec68d5a4dc4373` | changed | fix | C1;B2 | Physics forum. Badges left (`Gold Member` x20, `2022 Award` x7), BBCode tag |
| `6df989a2bcd44389` | dropped | drop | C1;C3 | Course Hero preview. Drop right |
| `4376102f99b2fdfd` | changed | fix | C1 | Jiskha. `Answer This Question` footer left |
| `1fbf0a290e52e534` | changed | clean | — | Mojibake fixed, breadcrumb removed. Source flattens exponents (`x2 = 4`) |
| `1e700ddf3f3e4e0d` | changed | clean | — | `posted by on .` removed correctly |
| `73670428f7373dee` | changed | drop | C5;C1 | Forum search results, no math |
| `c87c76a202bf9370` | changed | fix | C1 | Forum signature left |
| `94c0014436db12b5` | changed | fix | A3 | Broken emoji left at post offset 9633 |
| `4dd13476a6af3f3f` | changed | clean | — | `posted by .` removed correctly |
| `b5b886b63aeb9e2e` | dropped | drop | C1;C3 | Course Hero preview. Drop right |
| `411aeec06d260b92` | changed | fix | C1 | Game buttons and Twitter testimonials left |
| `fd90a7692cbe0dda` | changed | fix | C1 | BYJU'S UI left |
| `67f407a0ccd09d74` | dropped | drop | C1;C3 | Course Hero preview. Borderline |
| `a878064814cc3108` | changed | fix | C1 | `Similar Questions` link list left |
| `e45925da5672827e` | changed | drop | C1;C5;A4 | Worksheet catalogue; 2 U+FFFD |
| `21bfa34841bcc936` | changed | fix | C2 | Three figure lines still doubled; OCR errors in source |
| `01cbee543c7751fd` | changed | drop | C5;C4;C1 | SEO spam. Read `partial` |
| `89dc8e7278ce1f4b` | changed | fix | C1 | Formula line deleted from 4 of 5 solutions; `darkmode` left |
| `03adce6b428657cc` | changed | fix | C1 | `Quick Reference:` removed 5 of 6 times; page UI left |
| `b1efb7907a0c8cbb` | changed | fix | C1 | Sign-in prompt removed 9 of 10 times; forum footer left |
| `11c56a701b6c6f82` | changed | clean | — | Repeated paragraph in a quote removed; harmless |
| `76718c4a528f2cee` | changed | fix | C1 | answers.com UI left; low value |

### math-megamath (13)

| doc_id | Cleaner | Verdict | Diseases | Finding |
|---|---|---|---|---|
| `32ce96905cc39b34` | changed | fix | C1 | Good GLPK page with C code; emails masked; 9 `[edit]` labels left |
| `0ef68457f8488afa` | changed | drop | C1;C2 | Raw Blogger feed; masking ate nearby words |
| `4258f5c9fbf14805` | changed | drop | F1;C2;C1 | Giveaway, not math; 73 unmasked emails |
| `c2014b24148deb41` | changed | fix | C2;C1 | Good thread, but every post appears twice; one list repeated 8 times |
| `9cb311703c53c1e0` | changed | drop | A4;A3;E2;C4;C5 | SEO spam with raw PDF bytes |
| `31c47797e0ec552c` | changed | fix | F1;C1;B5 | Email masked, but phone and office address left in signature |
| `0cb915d7762c2e45` | changed | fix | C1;B5 | Key reference deleted; broken LaTeX |
| `32d96679ed59c9d5` | changed | fix | C1;C2;B2 | Good Polymath math; wiki diff header left |
| `f079a0d75820b184` | changed | fix | C1 | 12 reference entries deleted |
| `27c5838098758c5b` | changed | fix | F1 | Office phone left; 2 author lines deleted |
| `41f03ba56962c1e9` | changed | drop | C5;C4;C2;A3 | SEO sentence soup |
| `edfafeaa35e38bc9` | changed | fix | C1 | Good Usenet post; archive header and footer left |
| `ca8e91121e85235c` | changed | fix | C1 | Good Usenet post; archive header and footer left |

The full note for each doc is in the label CSVs.

## Medicines

C1 and C2 are the two diseases this audit found in more than five docs, so each has a medicine (DISEASES.md section 4). The cards follow the form of `spec_v1_1/sheets/specv1_1.md`.

- **Rule form.** `doc = {doc_id, text, url, lang_label, pack_id}`, and `medicine(doc)` returns `(action, text)`. Both math packs carry the `en` tag, so each rule also checks `pack_id`.
- **Run order.** Both rules run after the math family's spec v1.1 rules, because this audit read text that spec v1 had cleaned. Both are stage 30, C1 first.
- **Limit.** Gold docs changed plus dropped: at most `max(1, 1%)` = 1 doc on each pack (finemath 86 gold docs, megamath 26).
- **IDs.** `DA-C1-012` and `DA-C2-007` are the next numbers unused by any Data A rule on main, g1-specv1, g1-spec-v1-1 and the sheets.

| # | ID | disease | action | stage | finemath | megamath | audit docs treated |
|---|---|---|---|---|---|---|---|
| 1 | `DA-C1-012` | C1 boilerplate | fix | 30 | 0.00% (0/86) PASS | 0.00% (0/26) PASS | 12 / 23 |
| 2 | `DA-C2-007` | C2 repeated lines | fix | 30 | 0.00% (0/86) PASS | 0.00% (0/26) PASS | 3 / 6 |

Both rules in order: finemath 0.00% (0/86) PASS, megamath 0.00% (0/26) PASS. Code sha256 `d4a121e924393611dacbd82c0858eb5b0846194216ab3e0bac9b7542ea5e8ecb`. "Audit docs treated" counts the docs labelled with the code that the rule changes, out of those still in the cleaned files (the cleaner dropped 3 of the C1 docs). Neither rule changes an audit doc that is not labelled with its code.

**Speed.** Neither rule loops in Python over a doc's lines or passages. They were timed on every doc of math-finemath and math-megamath (Part_1, Heldout and the diagnosis packs: 48,250 docs, 303.4 M characters), on an Apple M4 CPU with Python 3.12.10, one process, from a cold start (42.9 s for the whole run). Each output was compared with a version of the same rule with no gates and with Python loops over lines and passages: 0 docs differ. A second run on each output changes 0 docs, so both rules are idempotent on these packs.

| file (docs) | `DA-C1-012` without gates → this version | `DA-C2-007` loop version → this version | both, in order |
|---|---|---|---|
| finemath Part_1 (29,122) | 6.13 → 1.67 s (3.7x) | 2.79 → 2.58 s (1.1x) | 4.60 s |
| megamath Part_1 (17,647) | 6.01 → 1.60 s (3.8x) | 2.47 → 2.48 s (1.0x) | 4.07 s |
| finemath Heldout (312) | 0.06 → 0.02 s | 0.03 → 0.02 s | 0.07 s |
| megamath Heldout (169) | 0.05 → 0.01 s | 0.02 → 0.01 s | 0.06 s |

`DA-C1-012` gains from running its `[edit]` regex only on docs with that literal. `DA-C2-007` is about as fast as the loop version: the loops only did work on the few lines that repeat, and its time is in splitting and counting lines and passages, which runs in C (1.0x to 1.1x on the full packs). There is no GPU code, so there is no sync check.

**Checks and logs.**
- Script: [`medicine_math_checks.py`](medicine_math_checks.py), next to this report. It holds the frozen copies of both rules from before the speed-up, and two checks:
  - `speed`: timing, same output and idempotence on full packs. Log: [`2026-11-21-math-audit-medicines-local.txt`](../jobs/logs/2026-11-21-math-audit-medicines-local.txt), numbers in [`…-summary-local.json`](../jobs/logs/2026-11-21-math-audit-medicines-summary-local.json).
  - `treated`: audit docs treated, on the cleaned packs. Log: [`…-treated-local.txt`](../jobs/logs/2026-11-21-math-audit-medicines-treated-local.txt).
- Tests: [`tests/audit/test_medicine_math_checks.py`](../tests/audit/test_medicine_math_checks.py), made-up texts only: what each rule does, its pack scope, idempotence, and each card against its frozen copy on edge cases and 4,000 random texts. `python -m pytest teams/01-data-a/tests/audit -q`: 67 passed, 45 of them in this file.
- Harm test: `harmtest/harm_test.py --rules teams/01-data-a/audit/math-audit-Aneesha_Das.md --rule all`, on each pack's gold set. Log: [`…-harm-local.txt`](../jobs/logs/2026-11-21-math-audit-medicines-harm-local.txt).

## Medicine 1 — math / C1 / `DA-C1-012`

**Consolidation:** Source: this audit (assignment 11, 35 docs); identity `DA-C1-012`, new. Counts are this audit's label counts: one reader, with the second-reader check still to do.

1. **Disease:** C1, 26 docs: finemath 16, megamath 10. Examples: `caec68d5a4dc4373` (`Homework Helper`, 1 of 11 copies left; `Gold Member` x20; `2022 Award` x7), `b1efb7907a0c8cbb` (`Please Log in or Create an account to join the conversation.`, 1 of 10 copies left), `32ce96905cc39b34` (9 wiki `[edit]` labels). Site chrome is left in the cleaned text. Where a chrome line repeats, one copy is left. By its code, that is `DA-C2-001`: a line of 15+ characters seen 5+ times keeps its first copy. Chrome seen once is not touched.

<!-- CARD EVIDENCE -->
**Current gold harm:** math-finemath: 0 changed / 0 dropped of 86 (PASS); math-megamath: 0 changed / 0 dropped of 26 (PASS).

**Examples:** the doc_ids and short quotes in item 1. Exact source characters are not published here, because pack text stays out of the public repo.
<!-- END CARD EVIDENCE -->

2. **Medicine:**
   - **What it fixes.** It deletes every copy of a whole line that is site chrome this audit found: forum badges and prompts (`Homework Helper`, `Gold Member`, `2022 Award`, `Last edited: …`, the sign-in prompt, `Moderators: …`, `Time to create page: …`), homework-site and Q&A buttons (`Answer This Question`, `Related Questions`, `Post a New Question`), BYJU'S and answers.com UI, the comp.compilers footer, bare journal page labels (`Received`, `Accepted`, `Published`, `Affiliations`, `Corresponding author`) and `darkmode`. It also cuts a wiki `[edit]` label at the end of a line, and Course Hero's `{[ promptMessage ]}` slot.
   - **Why only whole lines.** A chrome phrase inside a sentence can be real text (`He was a Homework Helper for years.`), so a line goes only when the chrome is all of it, with spaces at either end allowed.
   - **What it leaves alone.** A journal label with its date on the same line, and any line with other text on it.
   - **What it can't fix.** Chrome of sites this audit did not see: other forums' badges, signatures and game UI (`411aeec06d260b92`).

```python
RULE_ID = "DA-C1-012"
# site chrome left after spec v1, found by the math audit (assignment 11)
CODE = "C1"
ACTION = "fix"
STAGE = "30"
STATUS = "CANDIDATE"
LANGS = {'en'}
ORIGIN = "math-finemath, math-megamath"  # packs it runs on: the disease was found in both
PACKS = set(ORIGIN.split(", "))  # the math packs share the "en" tag, so the pack is checked too
SOURCE = "math-finemath, math-megamath"  # packs whose audit found it (Aneesha Das, assignment 11)
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

3. **Action:** fix. **Runs on:** finemath, megamath. **Audit docs it treats:** 12 of the 23 C1 docs still in the cleaned files (the cleaner dropped the other 3). Of the 11 it leaves, 5 are docs this audit drops. The other 6 are fix docs whose chrome is not on its list: forum signatures (`c87c76a202bf9370`, `31c47797e0ec552c`), game UI (`411aeec06d260b92`), a related-questions list (`a878064814cc3108`), forum pagination (`c2014b24148deb41`) and a wiki diff header (`32d96679ed59c9d5`).
4. **Harm test:** finemath 0.00% (0/86) PASS; megamath 0.00% (0/26) PASS.
5. **Side effects:**
   - A real page with one of these words alone on a line loses that line: a lesson heading `Related Questions`, or a journal page whose `Received` and `Published` labels sit alone above their dates (the dates stay, without their labels).
   - It only knows the sites this audit saw.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Source: this audit. New rule, not in spec v1.1.

---


## Medicine 2 — math / C2 / `DA-C2-007`

**Consolidation:** Source: this audit (assignment 11, 35 docs); identity `DA-C2-007`, new. It takes both parts of `DA-C2-001` (SomaKorada07, theschoolofai/era-v5-lab#61): its passage rule as part (c), rewritten without loops but with the same output, and its repeated-line rule as part (a), with a math and speaker-turn guard. Counts are this audit's label counts: one reader, with the second-reader check still to do.

1. **Disease:** C2, 6 docs: finemath 1, megamath 5. Examples: `21bfa34841bcc936` (`Figure 4.4 Small-scale eddies...` and two more figure lines still appear twice), `c2014b24148deb41` (every post appears twice, with 26 `Quote from`; nested quotes repeat the list `An input relation R with 3 attributes...` 8 times), `32d96679ed59c9d5` (a second copy of the Corollary proof in the diff header). `DA-C2-001` is the math C2 rule, and it runs on finemath only. It leaves these repeats, and it also deletes real math. In `89dc8e7278ce1f4b`, `P(E) = n(E)/ n(S)` (17 characters, 5 copies) keeps only its first copy, so solutions Q12 to Q15 jump from `n(E)` to the answer.

<!-- CARD EVIDENCE -->
**Current gold harm:** math-finemath: 0 changed / 0 dropped of 86 (PASS); math-megamath: 0 changed / 0 dropped of 26 (PASS).

**Examples:** the doc_ids and short quotes in item 1. Exact source characters are not published here, because pack text stays out of the public repo.
<!-- END CARD EVIDENCE -->

2. **Medicine:**
   - **What it fixes.** (a) A line of 15+ characters seen 5+ times keeps its first copy. (b) A figure or table caption line (`Figure 4.4 …`, `Fig. 2 …`, `Table 3 …`) seen twice or more keeps its first copy. (c) A passage (blank-line block) of 150+ characters seen more than once keeps its first copy (ruling 13, as in `DA-C2-001`), now on megamath too.
   - **Why not every repeated line.** A repeated formula is math, not junk: a worked solution restates it each time. A transcript's speaker turn (`MALE SPEAKER: [INAUDIBLE]`) is a separate turn each time.
   - **What it leaves alone.** Code lines, lines with a math sign (`= ^ $ \ < > ± × ÷ ≤ ≥ ∑ ∫ √ ∞`, or an operator between two digits), speaker turns (an upper-case name and a colon at the line start), bullet lists, and lists of answer options.
   - **What it can't fix.** Repeats that are not whole lines or whole passages: a Blogger comment that starts with a cut-off copy of itself (`0ef68457f8488afa`), or sentences repeated inside one long line (`41f03ba56962c1e9`). Both docs are drops anyway. A copy in another markup, such as a MediaWiki diff header that repeats the page in wiki markup (`32d96679ed59c9d5`), is also left. That copy goes only with the whole header, which is a C1 or B2 medicine, not a C2 one.
   - **It replaces `DA-C2-001`.** While `DA-C2-001` runs first, the formula lines are already gone. Retiring `DA-C2-001` is a change to spec v1.1, for the team to decide.

```python
RULE_ID = "DA-C2-007"
# repeated lines and passages left after spec v1, found by the math audit (assignment 11); replaces DA-C2-001
CODE = "C2"
ACTION = "fix"
STAGE = "30"
STATUS = "CANDIDATE"
LANGS = {'en'}
ORIGIN = "math-finemath, math-megamath"  # packs it runs on: the disease was found in both
PACKS = set(ORIGIN.split(", "))  # the math packs share the "en" tag, so the pack is checked too
SOURCE = "math-finemath, math-megamath"  # packs whose audit found it (Aneesha Das, assignment 11)
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

3. **Action:** fix. **Runs on:** finemath, megamath. **Audit docs it treats:** 3 of 6: `21bfa34841bcc936`, `c2014b24148deb41`, and `4258f5c9fbf14805` (a doc this audit drops). It leaves `0ef68457f8488afa` and `41f03ba56962c1e9`, as item 2 says. It also leaves `32d96679ed59c9d5`, one of the examples in item 1. Its copy of the proof sits in the MediaWiki diff header (the 938 characters before `## Revision as of`). There it is in wiki markup (`<math>` tags, `'''Proof'''`) and split across 44 diff-table rows (`− | |||`), while the body has `$…$` and `**Proof**`. With the markup mapped to the body's form, 189 characters match the body exactly. But no line or blank-line passage is the same in both places, so none of parts (a) to (c) sees it.
4. **Harm test:** finemath 0.00% (0/86) PASS; megamath 0.00% (0/26) PASS. Before the speaker-turn guard, megamath was 1 of 26 changed (3.85%, PASS at the 1-doc limit), and that change was harm. `ad62378f3ebf63f1` is a lecture transcript. Part (a) kept the first of its 22 `MALE SPEAKER: [INAUDIBLE]` lines and deleted the other 21 (546 characters), so the lecturer's replies lost the questions they answer.
5. **Side effects:**
   - A prose line repeated 5+ times on purpose (the same sentence closing each worked solution) keeps only its first copy, as with `DA-C2-001`.
   - A repeated math line that really is junk (a formula banner on every page) is kept, because the math guard wins.
   - The speaker-turn guard was added after one megamath gold doc, so it is fitted to that doc. A repeated line that starts with an upper-case word and a colon is always kept, even when it is junk (`NOTE: …` on every page).
   - A figure shown twice with its caption loses the second caption.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Source: this audit, and `DA-C2-001` for parts (a) and (c). New rule, not in spec v1.1.

## Follow-up

1. Run the current `clean.py` on the 4 over-cleaned docs and name the rule behind each deletion. By its code, the repeated-line deletions match `DA-C2-001`. The author-list removal and the email masking are still to name.
2. Spot-check the 755 finemath docs where a repeated formula-like line was deleted, to see how many lost real math.
3. Check the F1, C1 and over-cleaning findings against the p1536 shard output before proposing changes to the frozen spec.
4. Decide whether `DA-C2-007` replaces `DA-C2-001` in spec v1.1 (Medicine 2).
5. Arrange the second-reader check of 10% of these docs (4 of 35).
