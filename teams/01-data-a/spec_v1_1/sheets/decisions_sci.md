# decisions_sci: Spec v1.1 science candidates and what to do with each

**Draft, 12 Nov.** A first pass from the reader's notes and the cleaned text at their offsets. Nothing here is decided until the hand check of the examples is done.

- **Task.** Write a medicine and a harm test for each **new** disease above 1%. New means not in `diagnosis/DISEASES.md`, the spec v1 list. Codes and IDs are in `NEW_DISEASES.md`.
- **Inputs.** Aditeya's sci-finepdfs audit: 100 docs (rows 101–200 of the shuffle), mittalutkarsh/era-v5-lab#3. Every row is `source = ai`.
- **Still to come.** Utkarsh's 100 docs (rows 1–100). With them, docs read become 200 and a disease needs **2 docs**.
- **Pack and limit.** sci-finepdfs has 202 gold docs, so the limit is `max(1, floor(1% of 202))` = 2 docs.

## New diseases with 2 or more docs

These stay above the bar whatever Utkarsh finds.

| reading code | docs | code (rule ID) | action | medicine idea | why / risk |
|---|---|---|---|---|---|
| NEW-hyphen-split-words | 3 | A7 (`DA-A7-001`) | fix | Join a word broken at a line end: `stereo- scopic` becomes `stereo-scopic`. The hyphen is kept and the space goes. Suspended hyphens (`pre- and post-test`) are left alone. | Written and tested: 1/202 on gold, and it catches 3 of 3 reading docs. Its one gold hit has the disease. |
| NEW-merged-table-cells | 3 | C8 (`DA-C8-003`); math has its own rule | fix | Split a table cell that holds two numbers into the empty cell next to it (`\| 332 50.4 \|  \|` → `\| 332 \| 50.4 \|`). | Shifted rows and tables flattened into lines can't be put back. It treats 1 of 3. |
| NEW-missing-table-data | 3 | C7 (`DA-C7-003`) | fix | Cut a dangling "… with the following members." when no list follows. | The content is gone; cutting the empty frame is the most that can be done. Headings with no data are left. It treats 1 of 3. |
| NEW-reading-order | 2 | C8 (`DA-C8-004`) | fix | Put numbered steps that are out of order (3, 5, 4) back in order. | Scrambled sentences can't be put back. It treats 1 of 2. |
| NEW-seo-filler | 2 | C9 (`DA-C9-002`); math has its own rule | drop | Start from the math rule, plus the "download / access free ebook" pages with `Page n/N` markers. | Folding NEW-spun-text in would make it 3. |
| NEW-underscore-blanks | 2 | B6 (`DA-B6-001`) | drop | Drop a doc whose underscore runs make up 30% or more of its text (a blank form). | B4 covers template placeholders, not form lines, so this is new. The reading docs are at 31% and 63%; the highest gold doc is at 26%. |

## New diseases with 1 doc: waiting for Utkarsh

Each needs a second doc from Utkarsh's reading to qualify:
- NEW-missing-spaces (A8; it qualifies in math)
- NEW-detached-diacritics (A9, `Mo´nica`)
- NEW-figure-only-doc (C7)
- NEW-orphan-figure-labels (C7)
- NEW-split-superscripts (C8)
- NEW-spun-text (C9)
- NEW-generated-image-captions (D6)

## Not new: already in DISEASES.md

| reading code | docs | DISEASES.md code | what happened |
|---|---|---|---|
| NEW-pdf-ligatures | 5 | A5 wrong normal form (ligatures, as in math's `DA-A5-001`) | Outside v1.1. A rule drafted as `DA-A5-101` was taken out of `medicine_sci.md` on 12 Nov. |
| NEW-toc-dot-leaders | 2 | C1 boilerplate | Outside v1.1. A rule drafted as `DA-C1-101` was taken out of `medicine_sci.md` on 12 Nov. |
| NEW-pdf-quote-glyphs | 1 | A6 wrong glyphs (as in math's `DA-A6-001`) | Outside v1.1. |
| C5 | 1 | C5 lists of keywords or links | Outside v1.1. |

## Open

- **New codes.** See `NEW_DISEASES.md`.
- **Fold NEW-spun-text into NEW-seo-filler?** If so, run the collector with `--alias NEW-spun-text=NEW-seo-filler`.
