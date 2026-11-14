# Spec v1.1 harm tests: summary (Data A)

**Committed:** `14d45499a5213b7a981debe1c41abdd7ff825774`, 13 Nov 2026, 00:36:54 IST (+0530), by Aneesha Das. The harm reports and the results in this file belong to that commit. This line was filled in by a follow-up commit.
**Written:** 13 Nov 2026.
**Branch:** `g1-spec-v1-1` on patilvijay23/era-v5-lab, for theschoolofai/era-v5-lab#365.

The rules are the Spec v1.1 medicines for the new diseases in `../../sheets/NEW_DISEASES.md`. These are diseases not in `diagnosis/DISEASES.md` that appear in 1% or more of the docs read in a domain: 8 math and 6 science. The Indic family's C6 (`DB-C6-001`) is Data B's, and is not tested here.

## How they were tested

- **Tool:** `teams/01-data-a/tools/harm_test.py --rule all`, unchanged. It runs each rule alone, then all of a sheet's rules in sheet order (ALL IN ORDER), with a drop ending the doc and a fix passing its text on.
- **Sheets:** `sheets/medicine_math.md` and `sheets/medicine_sci.md`. Only the v1.1 rules were run, not v1 + v1.1.
- **Data:** the Week 1 pack files, with `teams/01-data-a/gold/<pack>_gold.csv` (docs that two readers agreed are clean).
- **Limit:** gold docs changed plus dropped must be at most `max(1, floor(1% of gold))`, per pack, never pooled. That is 1 doc on finemath (86) and megamath (26), and 2 on sci-finepdfs (202). A gold doc counts as touched when any character changes or it is dropped.
- **Reading docs treated:** the readers' docs with the rule's code that it changes or drops, on the cleaned sample (`scripts/reading_recall.py`).

## Results

### math-finemath: 86 gold docs, limit 1

| rule | disease | changed | dropped | result | gold docs touched | reading docs treated |
|---|---|---|---|---|---|---|
| `DA-A8-001` | words run together | 1 | 0 | PASS | `2d328b828fa97706` | 5 / 13 |
| `DA-C9-003` | template pages | 0 | 1 | PASS | `0f119a596e8b7f63` | 8 / 9 |
| `DA-C9-001` | SEO filler | 0 | 0 | PASS | none | 8 / 15 |
| `DA-D6-001` | machine-written text | 0 | 0 | PASS | none | 3 / 8 |
| `DA-C7-001` | lost formulas | 0 | 0 | PASS | none | 4 / 29 |
| `DA-C7-002` | lost figures | - | - | off (megamath only) | - | 4 / 5 |
| `DA-C8-001` | flattened formulas | 0 | 0 | PASS | none | 3 / 22 |
| `DA-C8-002` | merged table cells | 0 | 0 | PASS | none | 3 / 11 |
| **ALL IN ORDER** | | 1 | 1 | **FAIL (2/86)** | both above | |

The reading counts are for math as a whole, across both packs.

### math-megamath: 26 gold docs, limit 1

All 8 rules touch 0 gold docs, alone and in order: **PASS**. `DA-D6-001` is off on this pack.

### sci-finepdfs: 202 gold docs, limit 2

| rule | disease | changed | dropped | result | gold docs touched | reading docs treated |
|---|---|---|---|---|---|---|
| `DA-A7-001` | hyphen-split words | 1 | 0 | PASS | `920855df51f37db8` | 3 / 3 |
| `DA-C9-002` | SEO filler (ebook spinner) | 0 | 0 | PASS | none | 2 / 2 |
| `DA-B6-001` | blank forms | 0 | 0 | PASS | none | 2 / 2 |
| `DA-C7-003` | lost list | 0 | 0 | PASS | none | 1 / 3 |
| `DA-C8-004` | steps out of order | 0 | 0 | PASS | none | 1 / 2 |
| `DA-C8-003` | two numbers in one table cell | 0 | 0 | PASS | none | 1 / 3 |
| **ALL IN ORDER** | | 1 | 0 | **PASS (1/202)** | `920855df51f37db8` | |

## Gold contradictions

Every gold doc touched has the very disease that the rule targets. The gold set was built in Week 1, before these diseases were named.

| gold doc | pack | rule | what's in it |
|---|---|---|---|
| `2d328b828fa97706` | finemath | `DA-A8-001`, words run together (fix) | `fieldtrip.Write an equation`: a missing space after a full stop. The rule makes it `fieldtrip. Write`. |
| `0f119a596e8b7f63` | finemath | `DA-C9-003`, template pages (drop) | A grams-to-kilograms converter page, with the same template headings as reading doc `d0693c18056bae02` ("Alternative conversion", "Approximate result", "Conversion table", "… chart"). |
| `920855df51f37db8` | sci-finepdfs | `DA-A7-001`, hyphen-split words (fix) | A word split at a line-end hyphen in a Finnish abstract. This is within the limit. |

**Proposed handling** (aneeshads, 13 Nov; **waiting for Vijay's ruling**): record them as gold contradictions, as spec v1 did for `eee25e81417bc1b7` in sci.
- They stay out of the count for these rules only.
- They are flagged for a reader to check again at the next gold refresh.
- The gold CSVs and `harm_test.py` are not changed.

With the two finemath docs left out, finemath in order touches 0 of the other 84 gold docs and passes.

If they count instead, spec v1's hold rule holds `DA-A8-001`, which treats fewer reading docs. That leaves "words run together" without a working medicine.

## Rules narrowed after a false positive

Each fix is recorded in that rule's side effects in the sheet.

- **`DA-C7-001`:** it counted option labels and `=` with the answer on the next line as gaps. It also counted line-ending words before options, bullets, equations and wrapped lines. These no longer count, and the cut is now 4 gaps.
- **`DA-C8-001`:** answer-option labels such as `(a)` now break a run.
- **`DA-C8-002`:** it needed 2 long numeric rows and so caught none of the 11 reading docs. One row now suffices; no gold doc has such a row.
- **`DA-B6-001`:** at 10% it dropped 4 gold docs. The threshold is now 30%. The reading docs are at 31% and 63%; the highest gold doc is at 26%.
- **Taken out as not new:** `DA-B1-101`, `DA-A5-101` and `DA-C1-101`. B1, A5 and C1 are in `DISEASES.md`.

## Reports

`math-finemath_harm.json`, `math-megamath_harm.json` and `sci-finepdfs_harm.json` are in this folder. They hold doc_ids and counts only.

## Open

1. **Vijay's ruling** on the gold contradictions.
2. **Team agreement on the new codes** (A7–A9, B6, C7–C9, D6), and their entries in `DISEASES.md` and `CODE_ORDER`.
3. **The final collector run** after the readers' deadline, for code, English and the 1-doc science rows.
4. **The human check** of the reading examples.
5. **Collation:** `medicine_conflicts.py` and `specv1_1.md`.
6. **Whether the counting run** should be v1 + v1.1 rules together.
