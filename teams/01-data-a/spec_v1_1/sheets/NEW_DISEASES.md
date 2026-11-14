# NEW_DISEASES: the new diseases found for Spec v1.1 (Data A reading audits and the Indic family)

**The task (pipeline owner, due Fri 13 Nov):** for each new disease above 1%, write a medicine and a harm test for Spec v1.1.

- **New** means a disease that is not in `diagnosis/DISEASES.md`, the list of diseases spec v1 made medicines for.
- **Above 1%** means it is in 1% or more of the docs read in its domain.

This file records each new disease: its name, a description, its code, its rule ID and the packs where it was found. Rows that readers marked `new` but that `DISEASES.md` already describes are listed under "Not new" at the end. Update this file after every collector run (`scripts/collect_reading_prs.py`, output in `results/reading/summary.md`).

**Two teams are in this file:**
- **Data A families** (english, code, math, sci): their rules are `DA-…`.
- **The Indic family** (Data B): its rules are `DB-<code>-<nnn>`, so they can't clash with the `DA-…` ones. Its new diseases come from theschoolofai/era-v5-lab#216 and are marked **Indic** in the family column.

**C6 is the Indic family's code.** The Indic family proposed C6 first (31 Oct), for inconsistent punctuation and formatting, and TSAI ruled on 5 Nov that C6 goes into the spec. Data A's template pages first had C6 too, so on 12 Nov they moved to C9 as `DA-C9-003`, beside SEO filler. Now every code means one disease for both teams.

**Last updated:** 12 Nov 2026, from the collector run of 16:52 UTC, which matches the 14:53 UTC run: 4 reading PRs, 490 rows. Math is complete: pyru read 199 finemath docs and Satyanarayana 100 megamath docs. Science has Aditeya's 100 docs, with Utkarsh's still to come. Code (Revant) has no new diseases so far, and English has no readings yet.

**Status of the codes:** every new code and ID below is **proposed** and needs the team's agreement. A new code also needs an entry in `diagnosis/DISEASES.md` and in `CODE_ORDER` in `spec_v1/tools/build_clean.py`, or `clean.py` can't be built with it.

## Naming rules

1. **New or not.** A disease is new only if no disease in `diagnosis/DISEASES.md` describes it. If one does, the disease is not new, even in a family that has no spec v1 rule for it. It goes under "Not new", and Spec v1.1 doesn't cover it.
2. **New code.** A new disease gets the next free code in the nearest group of `DISEASES.md`:
   - A: Characters
   - B: Markup and ghost tokens
   - C: Document structure
   - D: Language
   - E: Code
   - F: Safety, privacy and contamination
   - G: Damage from earlier cleaning
   - H: Duplicates

   A code is one letter A–H plus one digit 1–9, the only form that `harm_test.py` and `build_clean.py` accept. Related diseases share a code when a group runs out of digits.
3. **Rule ID.** `DA-<code>-<nnn>` for the Data A families, and `DB-<code>-<nnn>` for the Indic family (Data B) ("Stable `DA-<code>-<nnn>`. Never reused", `spec_v1/README.md`). `nnn` is the next number that team has not used for that code on main, on `g1-specv1` or in the sheets. A new code starts at 001.
   - **One code, one meaning.** A disease code means the same disease for both teams. Only the prefix differs.
4. **One rule per family.** A disease found in two families gets one rule in each, with its own ID. `build_clean.py` puts each rule in one family and needs every ID in the collated sheet to be unique.
5. **When an ID is assigned.** Only when the disease qualifies. A disease below the bar has no ID yet, and its next free ID is shown in brackets.

## Summary: new diseases

| code | disease | reading code | family | packs (docs) | ≥1% of docs read | rule ID | medicine |
|---|---|---|---|---|---|---|---|
| **A7** | Word split at a line-end hyphen | NEW-hyphen-split-words | sci | sci-finepdfs (3) | yes | `DA-A7-001` | written, tested |
| **A8** | Words run together | NEW-missing-spaces | math | math-finemath (5), math-megamath (8) | yes | `DA-A8-001` | written, tested |
| **A8** | Words run together | NEW-missing-spaces | sci | sci-finepdfs (1) | 1 doc: waits for Utkarsh | (`DA-A8-002`) | - |
| **A9** | Accent split from its letter | NEW-detached-diacritics | sci | sci-finepdfs (1) | 1 doc: waits for Utkarsh | (`DA-A9-001`) | - |
| **B6** | Blank form fields | NEW-underscore-blanks | sci | sci-finepdfs (2) | yes | `DA-B6-001` | written, tested |
| **C6** | Inconsistent punctuation and formatting (runs of dots) | C6 (diagnosis labels, #177) | **Indic** | ind-te-sangraha-v (161 of 500 labelled docs, one reader) | yes (32%) | `DB-C6-001` | written by Data B (#177, Medicine 1), harm test passed on the 18-doc gold; to re-run on the 80-doc gold |
| **C7** | Content lost in extraction: formulas | NEW-missing-formulas | math | math-finemath (26), math-megamath (3) | yes | `DA-C7-001` | written, tested |
| **C7** | Content lost in extraction: figures | NEW-missing-figures | math | math-megamath (5) | yes | `DA-C7-002` | written, tested |
| **C7** | Content lost in extraction: table body | NEW-missing-table-data | sci | sci-finepdfs (3) | yes | `DA-C7-003` | written, tested |
| **C7** | Content lost in extraction: a doc that was all figures | NEW-figure-only-doc | sci | sci-finepdfs (1) | 1 doc: waits for Utkarsh | (`DA-C7-004`) | - |
| **C7** | Content lost in extraction: figure labels left alone | NEW-orphan-figure-labels | sci | sci-finepdfs (1) | 1 doc: waits for Utkarsh | (`DA-C7-004`) | - |
| **C8** | Layout lost in extraction: flattened formulas | NEW-flattened-formulas | math | math-finemath (20), math-megamath (2) | yes | `DA-C8-001` | written, tested |
| **C8** | Layout lost in extraction: merged table cells | NEW-merged-table-cells | math | math-finemath (10), math-megamath (1) | yes | `DA-C8-002` | written, tested |
| **C8** | Layout lost in extraction: merged table cells | NEW-merged-table-cells | sci | sci-finepdfs (3) | yes | `DA-C8-003` | written, tested |
| **C8** | Layout lost in extraction: reading order | NEW-reading-order | sci | sci-finepdfs (2) | yes | `DA-C8-004` | written, tested |
| **C8** | Layout lost in extraction: split superscripts | NEW-split-superscripts | sci | sci-finepdfs (1) | 1 doc: waits for Utkarsh | (`DA-C8-005`) | - |
| **C9** | SEO filler | NEW-seo-filler | math | math-finemath (10), math-megamath (5) | yes | `DA-C9-001` | written, tested |
| **C9** | SEO filler | NEW-seo-filler | sci | sci-finepdfs (2) | yes | `DA-C9-002` | written, tested |
| **C9** | Pages made for search: a page made from a template for one value | NEW-template-pages | math | math-finemath (8), math-megamath (1) | yes | `DA-C9-003` | written, tested |
| **C9** | SEO filler: spun text | NEW-spun-text | sci | sci-finepdfs (1) | 1 doc: waits for Utkarsh | (with `DA-C9-002`) | - |
| **D6** | Machine-written text | NEW-ai-generated-text | math | math-finemath (8) | yes | `DA-D6-001` | written, tested |
| **D6** | Machine-written text: picture captions | NEW-generated-image-captions | sci | sci-finepdfs (1) | 1 doc: waits for Utkarsh | (`DA-D6-002`) | - |

**Totals:** 22 rows. 15 qualify (8 math, 6 science, 1 Indic), and 7 science rows wait for Utkarsh's reading. When two waiting rows share a next free ID, the first to qualify takes it, and the other moves to the next number.

**The "≥1% of docs read" column** shows whether a disease passes the Spec v1.1 bar: it is in 1% or more of the docs read in its domain. Only those diseases need a medicine and a harm test.
- **Math:** 299 docs read, so a disease needs 3. Every math row passes.
- **Science:** 100 docs read today, so a disease needs 1. That becomes 2 of 200 once Utkarsh's reading is in.
  - **yes:** 2 or more docs, so the disease stays above the bar after Utkarsh.
  - **1 doc: waits for Utkarsh:** it passes only if his reading finds the disease at least once more.
- **Indic:** the count comes from Data B's diagnosis labels of the 500-doc pack, not from the Week 2 reading: 161 of 500 docs (32%).

## A. Characters

### A7 Word split at a line-end hyphen

- **Disease:** a word broken across a PDF line keeps its hyphen, and the line break becomes a space: `stereo- scopic`, `company- owned`, `day- to-day`.
- **Not in DISEASES.md:** G1 covers Indic words cut at a vowel sign only.
- **Found:** sci-finepdfs, 3 docs (Aditeya): `5a201c19575522f9`, `75768b1b6b165557`, `93412124dc441459`.
- **Rule:** `DA-A7-001`, fix, in `medicine_sci.md`. The space goes and the hyphen stays.

### A8 Words run together

- **Disease:** a space or line break between two pieces of text is lost, so they run together. Examples: a title glued to the sentence after it, a list of glossary entries on one line, a variable stuck to the next word.
- **Found:**
  - math-finemath, 5 docs (pyru), and math-megamath, 8 docs (Satyanarayana): `03aeffcf72c72418`, `167ef4a340862a0e`, `1ab888d9366a43fd`.
  - sci-finepdfs, 1 doc (Aditeya): `f163172e98bdaa35`.
- **Rule:** `DA-A8-001`, fix, math, in `medicine_math.md`. Science waits for Utkarsh; it would get `DA-A8-002`.

### A9 Accent split from its letter

- **Disease:** an accent comes out as a separate spacing character next to its letter: `Mo´nica` for `Mónica`.
- **Found:** sci-finepdfs, 1 doc (Aditeya): `21f33206f24a2f0e`.
- **Rule:** waits for Utkarsh. It would get `DA-A9-001`.

## B. Markup and ghost tokens

### B6 Blank form fields

- **Disease:** a blank fill-in form or questionnaire. Long runs of underscores mark the blanks and make up much of the text.
- **Not in DISEASES.md:** B4 covers template and code placeholders (`{{ variable }}`, `${...}`), not unfilled form lines.
- **Found:** sci-finepdfs, 2 docs (Aditeya): `3b2c0420a33ccaab`, `d8e3f724095e4aaa`.
- **Rule:** `DA-B6-001`, drop, in `medicine_sci.md`. It drops a doc whose underscore runs make up 30% or more of its text.

## C. Document structure

### C7 Content lost in extraction

- **Disease:** when the page was turned into text, formulas, pictures or table bodies that were images or PDF objects were dropped, and what is left can't be followed. Nothing can put the content back. Typical signs:
  - a sentence ending in `=` or `:` with nothing after it
  - empty answer options `(A)`–`(D)`
  - "the figure below" with no figure
  - table headings with no rows
  - diagram labels with no diagram
- **Found:**
  - **NEW-missing-formulas:** math-finemath, 26 docs (pyru), and math-megamath, 3 docs (Satyanarayana): `0614bc6ecf034495`, `06f2d75b8c5610f1`, `0e89fda818263262`. Rule `DA-C7-001`.
  - **NEW-missing-figures:** math-megamath, 5 docs (Satyanarayana): `019fda08843696eb`, `03aeffcf72c72418`, `0ab64ed894610ef7`. Rule `DA-C7-002`, or the same rule as `DA-C7-001` if the team folds the two together.
  - **NEW-missing-table-data:** sci-finepdfs, 3 docs (Aditeya): `0863ac75c9ec153e`, `4a33733776358fac`, `fac0d63aba9fab0c`. Rule `DA-C7-003`.
  - **NEW-figure-only-doc:** sci-finepdfs, 1 doc (Aditeya): `35d51dcf919068f8`. Only the title, authors and licence are left. Waits for Utkarsh.
  - **NEW-orphan-figure-labels:** sci-finepdfs, 1 doc (Aditeya): `75183caf56d8e54b`. Diagram callouts are left as lines of single letters and numbers. Waits for Utkarsh.

### C8 Layout lost in extraction

- **Disease:** the text is all there, but its layout isn't. Typical signs:
  - superscripts and subscripts on the base line (`x2` for `x²`), or one token per line
  - table cells run together or shifted a column
  - steps or sentences out of order
  - an ordinal suffix moved onto its own line
- **Found:**
  - **NEW-flattened-formulas:** math-finemath, 20 docs (pyru), and math-megamath, 2 docs (Satyanarayana): `033ab42a96110ff9`, `0f2f052fd95da6d4`, `1ab888d9366a43fd`. Rule `DA-C8-001`.
  - **NEW-merged-table-cells, math:** math-finemath, 10 docs (pyru), and math-megamath, 1 doc (Satyanarayana): `06f2d75b8c5610f1`, `1b1c42eddd14c8e7`, `1e7fad525310d310`. Rule `DA-C8-002`.
  - **NEW-merged-table-cells, science:** sci-finepdfs, 3 docs (Aditeya): `680400a6c14bde86`, `6cdcf2807cbf5156`, `6deaff0b02c57734`. Rule `DA-C8-003`.
  - **NEW-reading-order:** sci-finepdfs, 2 docs (Aditeya): `e4d1400f9236b8c9`, `ef812364b6cca7a0`. Rule `DA-C8-004`.
  - **NEW-split-superscripts:** sci-finepdfs, 1 doc (Aditeya): `db955ac64864c36c`. Waits for Utkarsh.
- **Note:** merged table cells and reading order may have no safe repair. If so, the medicine is a drop rule, and the harm test sets its status.

### C9 Pages made for search traffic: SEO filler and template pages

- **Disease:** a page written to rank in search, not to teach. There are two kinds.
  - **SEO filler**, with these typical signs:
    - generic filler paragraphs
    - tutoring or homework-app sales copy
    - "download this ebook" pages
    - the title repeated many times
    - text run through a synonym spinner
  - **Template pages:** a calculator, unit-converter or number-fact page that a site generates for every value, such as "Prime factorization of 36871", "68.71 kg in pounds" or "15 percent off $41". It is the same template with one number swapped in, and it often states wrong results.
- **Not in DISEASES.md:** C5 is lists of keywords or links. These pages are prose and generated pages.
- **Why template pages are here:** they were C6 at first. C6 is the Indic family's code (see "Indic family" below), so on 12 Nov they moved here, beside SEO filler.
- **Found:**
  - **NEW-template-pages:** math-finemath, 8 docs (pyru), and math-megamath, 1 doc (Satyanarayana): `28f56e454b2ade39`, `32b1f031d8c464e4`, `647d9713533c5664`. Rule `DA-C9-003` (was `DA-C6-001`), drop, both math packs, in `medicine_math.md`.
  - **NEW-seo-filler, math:** math-finemath, 10 docs (pyru), and math-megamath, 5 docs (Satyanarayana): `053c7e6dfbdb4e46`, `085f9f7987b7cf56`, `0e2b9cc0beb5fb11`. Rule `DA-C9-001`.
  - **NEW-seo-filler, science:** sci-finepdfs, 2 docs (Aditeya): `365b8539483e5a3f`, `d8a1ed9b1607a1dd`. Rule `DA-C9-002`.
  - **NEW-spun-text:** sci-finepdfs, 1 doc (Aditeya): `f82d23a50e08a31f`. Waits for Utkarsh. If the team agrees, it can be counted with NEW-seo-filler (`--alias NEW-spun-text=NEW-seo-filler`).

## D. Language

### D6 Machine-written text

- **Disease:** text generated by a model and added to the page. Typical signs:
  - machine-written SEO articles, often with false statements
  - AI summaries and FAQ blocks that a site adds under a forum thread
  - machine-written picture descriptions inserted into the text
- **Not in DISEASES.md:** D4 and D5 are about translation.
- **Found:**
  - **NEW-ai-generated-text:** math-finemath, 8 docs (pyru), 3 of them marked "unsure": `10d547035dd610e2`, `36c3e8b4b4457083`, `8f1f8a9c194e88f2`. Rule `DA-D6-001`.
  - **NEW-generated-image-captions:** sci-finepdfs, 1 doc (Aditeya): `79596acc4ee8648a`. Waits for Utkarsh.

## Indic family (Data B)

The Indic family's rules are `DB-<code>-<nnn>`.

- **Source:** Data B's list of proposed new codes for Spec v1.1 (shankarpandala, theschoolofai/era-v5-lab#216, 12 Nov). Data B checked every `ind-*` and `cadj-*` sheet on main. Only one code in use isn't in `DISEASES.md`, and the checking reads and rotation audits used existing codes only.

### C6 Inconsistent punctuation and formatting · Indic

- **Disease:** punctuation that has been broken up or repeated: runs of two or more dots, with or without spaces between them (`. .`, `..`, `. . .`), where one full stop or a standard `...` was meant. A separate C6 case is a decimal broken by a space (`3. 67` for `3.67`).
- **Not in DISEASES.md:** proposed as a new code by oddjoker91 on 31 Oct (G2). TSAI ruled on 5 Nov that C6 goes into spec v1 once #177 is approved. #177 is merged, but theschoolofai/era-v5-lab#244 doesn't turn C6 on, so Data B proposes it for v1.1.
- **Found:** ind-te-sangraha-v, one reader (oddjoker91, #177). 161 of 500 docs (32%): 138 fix and 23 drop, 108 of them with C6 alone. Examples: `2a0cfe90fd36e4ed` (`. .`), `11960d16258cd4b5` (`..`), `05fa567dfc849e6b` (`. . .`).
- **Rule:** `DB-C6-001`, fix, from Medicine 1 in `diagnosis/submissions/ind-te-sangraha-v/oddjoker91/medicine.md`. It turns a run of two or more ASCII dots, including spaces between them, into exactly `...`, never across a line break. A single dot, an existing `...` and `…` stay. It never drops a doc.
- **Harm test (Data B, 9 Nov):** PASS on the 18-doc gold: 1 of 18 changed (`85644a4840025b6f`), within the 1-doc allowance, and 0 dropped. The joiner (ZWJ/ZWNJ) and normal-form audits passed. Run in order C6 → C1 → C2, it also passed at 1 of 18.
- **Still to do:** the ind-te-sangraha-v gold has had 80 docs since #362, so the C6 → C1 → C2 run has to be repeated on it.
- **Deferred:** the decimal repair (`3. 67` → `3.67`). Together with the dot rule it changes 2 of 18 gold docs and FAILS, so it isn't active.
- **ID check:** no `DB-C6-…` ID is used on main or on `g1-specv1`, so `DB-C6-001` is free.

## Not new: the disease is already in DISEASES.md

Readers flagged these as new, either as `NEW-…` codes or as a known code with kind `new`. In each case a disease in `DISEASES.md` already describes them, so Spec v1.1 doesn't cover them and they get no v1.1 ID. Where the family has no spec v1 rule for the disease, that is a gap in spec v1's pack scope, not a new disease.

| reading code | family | packs (docs) | DISEASES.md code | spec v1 rule in this family | note |
|---|---|---|---|---|---|
| NEW-pdf-ligatures | sci | sci-finepdfs (5) | A5 wrong normal form (ligatures, as in math's `DA-A5-001`) | none | A rule drafted as `DA-A5-101` was taken out of `medicine_sci.md` on 12 Nov. |
| NEW-pdf-quote-glyphs | sci | sci-finepdfs (1) | A6 wrong or mixed script (wrong glyphs, as in math's `DA-A6-001`) | `DA-A6-002` | |
| B1 | math | math-finemath (3) | B1 HTML entities | none (`DA-B1-002` runs on English only) | A rule drafted as `DA-B1-101` was taken out of `medicine_math.md` on 12 Nov. |
| NEW-toc-dot-leaders | sci | sci-finepdfs (2): `59aef569bc52cdff`, `b1857f12c0d28f07` | C1 boilerplate | `DA-C1-007` | A rule drafted as `DA-C1-101` was taken out of `medicine_sci.md` on 12 Nov. |
| NEW-aggregated-questions | math | math-megamath (4) | C4 two documents joined | none | |
| C5 | sci | sci-finepdfs (1) | C5 lists of keywords or links | none | |

## Open

- **The team's agreement** is still needed for every proposed code, both Data A's and Data B's.
- **Gold contradictions.** Three gold docs have a new disease that a v1.1 rule targets:
  - finemath `2d328b828fa97706` (A8)
  - finemath `0f119a596e8b7f63` (C9 template pages)
  - sci `920855df51f37db8` (A7)

  The proposal (aneeshads, 13 Nov) is to record them as gold contradictions: out of the count for those rules only, and re-read at the next gold refresh. That would let the finemath combined run pass. **Waiting for Vijay's ruling.**

## How to update this file

After each collector run:
1. Add a row for every new reading code of kind `new`. If a disease in `DISEASES.md` describes it, list it under "Not new". Otherwise give it the next free code in the nearest group.
2. Update the doc counts per pack and the bar per domain (1% of docs read).
3. When a row reaches the bar, give it the next free ID for its code. Never reuse an ID.
4. Note the date and the collector run under "Last updated".
5. For the Indic family, read Data B's latest list of proposed codes (theschoolofai/era-v5-lab#216) and add its diseases with `DB-` IDs. Check that no code means one thing for Data A and another for Data B.

## Changes

- **12 Nov 2026:** template pages moved from C6 to C9 as `DA-C9-003` (was `DA-C6-001`), beside SEO filler, so that C6 means only the Indic family's punctuation disease.
- **12 Nov 2026:** the Indic family added, from theschoolofai/era-v5-lab#216: C6, inconsistent punctuation and formatting (ind-te-sangraha-v, 161 of 500 docs), rule `DB-C6-001`. Indic rules use the `DB-` prefix. C6 now clashes with Data A's template pages, which need a new code. 22 rows, 15 qualifying.
- **12 Nov 2026:** first version, from the collector run of 14:53 UTC. 27 rows, 18 qualifying. Codes A7, A8, A9, C6, C7, C8, C9 and D6 are proposed new codes.
- **12 Nov 2026:** NEW-toc-dot-leaders moved to "Not new". It is C1 boilerplate, and sci has a C1 rule (`DA-C1-007`), so it is a missed C1 case. 26 rows, 17 qualifying.
- **12 Nov 2026, 16:52 UTC:** collector run, no change: same 4 PRs, 490 rows and 27 candidates. Aswani, Harini, Mahesh and Utkarsh still have no reading files.
- **12 Nov 2026:** every one of the 14 qualifying new diseases has a written medicine, a unit test and a harm test.
  - **Harm, each rule alone:** every rule passes.
  - **Harm, in order:** megamath and science pass. Finemath is at 2/86, and both gold docs touched have the disease (gold contradictions).
  - **Reading docs treated:** see `medicine_math.md` and `medicine_sci.md`.
  - **IDs:** the two drafts now carry their IDs from this file (`DA-C6-001`, later `DA-C9-003`, and `DA-A7-001`).
- **12 Nov 2026:** the task restated: new means not in `DISEASES.md`. A5 ligatures, A6 quote glyphs, B1, C4 aggregated questions and C5 moved to "Not new". The blank-form disease, which was B4, is now new code B6, because B4 covers template placeholders. 21 rows, 14 qualifying.
