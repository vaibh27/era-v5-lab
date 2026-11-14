# reading_check_v1: the Week 2 reading audits of the cleaned sample, all four domains

**Built** by `spec_v1_1/scripts/build_reading_check.py` from the collector run of 2026-11-12 19:09 UTC (`spec_v1_1/results/reading/`) and from `spec_v1_1/sheets/NEW_DISEASES.md`. Don't edit this file by hand: run the collector, then this script.

**Kinds of row** (from the reading README):
- **missed:** Spec v1 has a rule for the disease, and the cleaned doc still has it.
- **new:** a disease that Spec v1 has no rule for.
- **harm:** a Spec v1 rule damaged real content.

**Occurrence rate:** the distinct docs with the disease, divided by the docs read in its domain, × 100.

**Flag:** a new disease at 1% or more, which needs a medicine and a harm test for Spec v1.1. "New" means not in `diagnosis/DISEASES.md`. A row the reader marked new that `DISEASES.md` already describes is shown, but not flagged.

## Audits per domain

| domain | docs read | readers with a reading file | readers with no file yet |
|---|---|---|---|
| english | 0 | none | Aswani Keerthi (en-dclm, en-fineweb-edu); Harini Jasti (en-nemotroncc, en-nemotroncc-synth) |
| code | 100 | revant 100 | Mahesh Sv (code-cpp, code-go, code-java) |
| math | 299 | pyru 199, satyanarayana 100 | none |
| science | 100 | aditeya 100 | Utkarsh Mittal (sci-finepdfs) |

Reading PRs used: mittalutkarsh/era-v5-lab#4, mittalutkarsh/era-v5-lab#3, mittalutkarsh/era-v5-lab#2, theschoolofai/era-v5-lab#350.

## 1. Diseases missed by Spec v1

A disease Spec v1 has a rule for, still in the cleaned doc. A code with no Spec v1 rule for the family ("none") is a gap in Spec v1's pack scope.

### english

No audit has been submitted for this domain.

### code

| code | docs | rate | readers | Spec v1 rules for this family |
|---|---|---|---|---|
| E4 | 3 | 3.0% | revant | DA-E4-005 CANDIDATE; DA-E4-010 CANDIDATE; DA-E4-009 CANDIDATE; DA-E4-007 CANDIDATE |
| C3 | 2 | 2.0% | revant | DA-C3-002 CANDIDATE |
| E1 | 2 | 2.0% | revant | DA-E1-001 CANDIDATE; DA-E1-004 CANDIDATE; DA-E1-008 CANDIDATE; DA-E1-006 CANDIDATE; DA-E1-005 CANDIDATE; DA-E1-007 CANDIDATE |

### math

| code | docs | rate | readers | Spec v1 rules for this family |
|---|---|---|---|---|
| C1 | 167 | 55.9% | pyru satyanarayana | DA-C1-001 CANDIDATE |
| C5 | 22 | 7.4% | pyru satyanarayana | DA-C5-001 CANDIDATE; DA-C5-006 CANDIDATE |
| C2 | 20 | 6.7% | pyru satyanarayana | DA-C2-001 CANDIDATE |
| B5 | 18 | 6.0% | pyru | DA-B5-001 CANDIDATE |
| A3 | 7 | 2.3% | pyru satyanarayana | DA-A3-001 CANDIDATE |
| B4 | 5 | 1.7% | pyru satyanarayana | DA-B4-001 CANDIDATE |
| C3 | 4 | 1.3% | pyru satyanarayana | DA-C3-004 CANDIDATE; DA-C3-005 CANDIDATE |
| B2 | 3 | 1.0% | pyru | DA-B2-001 CANDIDATE |
| D3 | 1 | 0.3% | pyru | DA-D3-005 CANDIDATE |
| F2 | 1 | 0.3% | pyru | none |

### science

| code | docs | rate | readers | Spec v1 rules for this family |
|---|---|---|---|---|
| C1 | 18 | 18.0% | aditeya | DA-C1-007 ? |
| C2 | 4 | 4.0% | aditeya | DA-C2-004 ? |
| F1 | 4 | 4.0% | aditeya | DA-F1-006 ? |
| C3 | 3 | 3.0% | aditeya | DA-C3-006 ? |
| A1 | 1 | 1.0% | aditeya | DA-A1-006 ? |
| D1 | 1 | 1.0% | aditeya | DA-D1-001 ? |

## 2. New diseases

### english

No audit has been submitted for this domain.

### code

None found.

### math

| code | docs | rate | ≥ 1% | status | Spec v1.1 code and rule |
|---|---|---|---|---|---|
| NEW-missing-formulas | 29 | 9.7% | yes | new | C7, `DA-C7-001` |
| NEW-flattened-formulas | 22 | 7.4% | yes | new | C8, `DA-C8-001` |
| NEW-seo-filler | 15 | 5.0% | yes | new | C9, `DA-C9-001` |
| NEW-missing-spaces | 13 | 4.3% | yes | new | A8, `DA-A8-001` |
| NEW-merged-table-cells | 11 | 3.7% | yes | new | C8, `DA-C8-002` |
| NEW-template-pages | 9 | 3.0% | yes | new | C9, `DA-C9-003` |
| NEW-ai-generated-text | 8 | 2.7% | yes | new | D6, `DA-D6-001` |
| NEW-missing-figures | 5 | 1.7% | yes | new | C7, `DA-C7-002` |
| NEW-aggregated-questions | 4 | 1.3% | yes | not new | in DISEASES.md as C4 two documents joined |
| B1 | 3 | 1.0% | yes | not new | in DISEASES.md as B1 HTML entities |
| D4 | 2 | 0.7% | no | not new | in DISEASES.md (D4) |
| E2 | 2 | 0.7% | no | not new | in DISEASES.md (E2) |
| NEW-orphan-footnote-markers | 2 | 0.7% | no | new | not yet in NEW_DISEASES.md |
| A4 | 1 | 0.3% | no | not new | in DISEASES.md (A4) |
| C4 | 1 | 0.3% | no | not new | in DISEASES.md (C4) |
| D2 | 1 | 0.3% | no | not new | in DISEASES.md (D2) |
| E1 | 1 | 0.3% | no | not new | in DISEASES.md (E1) |
| NEW-dropped-ligatures | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |
| NEW-hyphen-split-words | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |
| NEW-interleaved-columns | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |
| NEW-merged-documents | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |
| NEW-missing-code-lines | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |
| NEW-missing-interactive-content | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |
| NEW-offtopic-poll | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |
| NEW-orphan-figure-labels | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |
| NEW-truncated-previews | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |
| NEW-unattributed-copy | 1 | 0.3% | no | new | not yet in NEW_DISEASES.md |

### science

| code | docs | rate | ≥ 1% | status | Spec v1.1 code and rule |
|---|---|---|---|---|---|
| NEW-pdf-ligatures | 5 | 5.0% | yes | not new | in DISEASES.md as A5 wrong normal form (ligatures, as in math's DA-A5-001) |
| NEW-hyphen-split-words | 3 | 3.0% | yes | new | A7, `DA-A7-001` |
| NEW-merged-table-cells | 3 | 3.0% | yes | new | C8, `DA-C8-003` |
| NEW-missing-table-data | 3 | 3.0% | yes | new | C7, `DA-C7-003` |
| NEW-reading-order | 2 | 2.0% | yes | new | C8, `DA-C8-004` |
| NEW-seo-filler | 2 | 2.0% | yes | new | C9, `DA-C9-002` |
| NEW-toc-dot-leaders | 2 | 2.0% | yes | not new | in DISEASES.md as C1 boilerplate |
| NEW-underscore-blanks | 2 | 2.0% | yes | new | B6, `DA-B6-001` |
| C5 | 1 | 1.0% | yes | not new | in DISEASES.md as C5 lists of keywords or links |
| NEW-detached-diacritics | 1 | 1.0% | yes | new | A9, no ID yet (DA-A9-001) |
| NEW-figure-only-doc | 1 | 1.0% | yes | new | C7, no ID yet (DA-C7-004) |
| NEW-generated-image-captions | 1 | 1.0% | yes | new | D6, no ID yet (DA-D6-002) |
| NEW-missing-spaces | 1 | 1.0% | yes | new | A8, no ID yet (DA-A8-002) |
| NEW-orphan-figure-labels | 1 | 1.0% | yes | new | C7, no ID yet (DA-C7-004) |
| NEW-pdf-quote-glyphs | 1 | 1.0% | yes | not new | in DISEASES.md as A6 wrong or mixed script (wrong glyphs, as in math's DA-A6-001) |
| NEW-split-superscripts | 1 | 1.0% | yes | new | C8, no ID yet (DA-C8-005) |
| NEW-spun-text | 1 | 1.0% | yes | new | C9, no ID yet (with DA-C9-002) |

## 3. Harm: Spec v1 rules that damaged real content

### english

No audit has been submitted for this domain.

### code

None found.

### math

| rule | docs | rate | readers | Spec v1 rule |
|---|---|---|---|---|
| DA-C2-001 | 6 | 2.0% | pyru | DA-C2-001 math CANDIDATE |
| DA-D3-005 | 1 | 0.3% | pyru | DA-D3-005 math CANDIDATE |

### science

None found.

## 4. Flagged: new diseases at 1% or more

| domain | code | docs | docs read | rate | Spec v1.1 code and rule |
|---|---|---|---|---|---|
| math | NEW-missing-formulas | 29 | 299 | 9.7% | C7, `DA-C7-001` |
| math | NEW-flattened-formulas | 22 | 299 | 7.4% | C8, `DA-C8-001` |
| math | NEW-seo-filler | 15 | 299 | 5.0% | C9, `DA-C9-001` |
| math | NEW-missing-spaces | 13 | 299 | 4.3% | A8, `DA-A8-001` |
| math | NEW-merged-table-cells | 11 | 299 | 3.7% | C8, `DA-C8-002` |
| math | NEW-template-pages | 9 | 299 | 3.0% | C9, `DA-C9-003` |
| math | NEW-ai-generated-text | 8 | 299 | 2.7% | D6, `DA-D6-001` |
| math | NEW-missing-figures | 5 | 299 | 1.7% | C7, `DA-C7-002` |
| science | NEW-hyphen-split-words | 3 | 100 | 3.0% | A7, `DA-A7-001` |
| science | NEW-merged-table-cells | 3 | 100 | 3.0% | C8, `DA-C8-003` |
| science | NEW-missing-table-data | 3 | 100 | 3.0% | C7, `DA-C7-003` |
| science | NEW-reading-order | 2 | 100 | 2.0% | C8, `DA-C8-004` |
| science | NEW-seo-filler | 2 | 100 | 2.0% | C9, `DA-C9-002` |
| science | NEW-underscore-blanks | 2 | 100 | 2.0% | B6, `DA-B6-001` |
| science | NEW-detached-diacritics | 1 | 100 | 1.0% | A9, no ID yet (DA-A9-001) |
| science | NEW-figure-only-doc | 1 | 100 | 1.0% | C7, no ID yet (DA-C7-004) |
| science | NEW-generated-image-captions | 1 | 100 | 1.0% | D6, no ID yet (DA-D6-002) |
| science | NEW-missing-spaces | 1 | 100 | 1.0% | A8, no ID yet (DA-A8-002) |
| science | NEW-orphan-figure-labels | 1 | 100 | 1.0% | C7, no ID yet (DA-C7-004) |
| science | NEW-split-superscripts | 1 | 100 | 1.0% | C8, no ID yet (DA-C8-005) |
| science | NEW-spun-text | 1 | 100 | 1.0% | C9, no ID yet (with DA-C9-002) |

**Flagged per domain:** english 0, code 0, math 8, science 13.

**Marked new by the reader at 1% or more, but already in `DISEASES.md`:** math NEW-aggregated-questions (1.3%), math B1 (1.0%), science NEW-pdf-ligatures (5.0%), science NEW-toc-dot-leaders (2.0%), science C5 (1.0%), science NEW-pdf-quote-glyphs (1.0%).

## Notes

- **Rows by source:** ai 489, manual 1. `ai` means an AI tool suggested the row and the reader checked it. Each row is a first pass until a second reader has checked it.
- **Domains with no audit:** english. They have no rates yet.
- **Collector warning:** theschoolofai/era-v5-lab#350: base is theschoolofai/era-v5-lab:main, not mittalutkarsh/era-v5-lab:g1-reading
