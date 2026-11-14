# Week 2 reading audits, collected 2026-11-12 19:09 UTC

## Open reading PRs

| PR | title | author | branch -> base | reading files |
|---|---|---|---|---|
| [mittalutkarsh/era-v5-lab#4](https://github.com/mittalutkarsh/era-v5-lab/pull/4) | reading: code audit, 100 cleaned docs (task 8, Revant Gupta) | revantg | g1-reading-code-revant -> mittalutkarsh/era-v5-lab:g1-reading | reading_code_revant.csv, reading_code_revant.md |
| [mittalutkarsh/era-v5-lab#3](https://github.com/mittalutkarsh/era-v5-lab/pull/3) | reading: sci-finepdfs audit, 100 cleaned docs (task 10, aditeya) | AditeyaAItronics | g1-reading-sci-finepdfs-aditeya -> mittalutkarsh/era-v5-lab:g1-reading | reading_sci-finepdfs_aditeya.csv, reading_sci-finepdfs_aditeya.md |
| [mittalutkarsh/era-v5-lab#2](https://github.com/mittalutkarsh/era-v5-lab/pull/2) | reading: math-finemath audit, 99 extra cleaned docs (task 9, pyru) | pyru | g1-reading-math-finemath-pyru -> mittalutkarsh/era-v5-lab:g1-reading | reading_math-finemath_pyru.csv, reading_math-finemath_pyru.md |
| [theschoolofai/era-v5-lab#350](https://github.com/theschoolofai/era-v5-lab/pull/350) | G1 reading math megamath satyanarayana | murthyap63-a11y | g1-reading-math-megamath-Satyanarayana -> theschoolofai/era-v5-lab:main | reading_math-finemath_pyru.csv, reading_math-finemath_pyru.md, reading_math-megamath_satyanarayana.csv, reading_math-megamath_satyanarayana.md |

## Files used

One copy per file: the reader's own open PR, else g1-reading, else another PR.

| file | from | rows | also in |
|---|---|---|---|
| reading_code_revant.csv | mittalutkarsh/era-v5-lab#4 | 7 |  |
| reading_math-finemath_pyru.csv | mittalutkarsh/era-v5-lab#2 | 332 | mittalutkarsh/era-v5-lab@g1-reading, theschoolofai/era-v5-lab#350 |
| reading_math-megamath_satyanarayana.csv | theschoolofai/era-v5-lab#350 | 86 |  |
| reading_sci-finepdfs_aditeya.csv | mittalutkarsh/era-v5-lab#3 | 65 |  |
| reading_code_revant.md | mittalutkarsh/era-v5-lab#4 | - |  |
| reading_math-finemath_pyru.md | mittalutkarsh/era-v5-lab#2 | - | mittalutkarsh/era-v5-lab@g1-reading, theschoolofai/era-v5-lab#350 |
| reading_math-megamath_satyanarayana.md | theschoolofai/era-v5-lab#350 | - |  |
| reading_sci-finepdfs_aditeya.md | mittalutkarsh/era-v5-lab#3 | - |  |

## README rows

| row | reader | packs | reading files |
|---|---|---|---|
| 7 | Aswani Keerthi | en-dclm, en-fineweb-edu | **none yet** |
| 7 | Harini Jasti | en-nemotroncc, en-nemotroncc-synth | **none yet** |
| 8 | Mahesh Sv | code-cpp, code-go, code-java | **none yet** |
| 8 | Revant Gupta | code-js-ts, code-python, code-rust | reading_code_revant.csv |
| 9 | Ramesh | math-finemath | reading_math-finemath_pyru.csv |
| 9 | Satyanarayana Murthy Toleti | math-megamath | reading_math-megamath_satyanarayana.csv |
| 10 | Utkarsh Mittal | sci-finepdfs | **none yet** |
| 10 | Aditeya | sci-finepdfs | reading_sci-finepdfs_aditeya.csv |

## Docs read per domain

100 per reader unless `--docs-read` says otherwise. Used for the % column and the Spec v1.1 list.

| domain | docs read | readers |
|---|---|---|
| code | 100 | revant 100 |
| math | 299 | pyru 199, satyanarayana 100 |
| science | 100 | aditeya 100 |

## Spec v1.1 list: 1% or more of the docs read in a domain

`yes`: a new disease. `check`: a missed disease with no rule for it in this family's spec v1.

| domain | docs read | docs needed |
|---|---|---|
| code | 100 | 1 |
| math | 299 | 3 |
| science | 100 | 1 |

### math

| code | kind | docs | % of docs read | readers | spec v1 rules | flag | example doc_ids |
|---|---|---|---|---|---|---|---|
| NEW-missing-formulas | new | 29 | 9.7 | pyru satyanarayana | none | yes | 0614bc6ecf034495 06f2d75b8c5610f1 0e89fda818263262 |
| NEW-flattened-formulas | new | 22 | 7.4 | pyru satyanarayana | none | yes | 033ab42a96110ff9 0f2f052fd95da6d4 1ab888d9366a43fd |
| NEW-seo-filler | new | 15 | 5.0 | pyru satyanarayana | none | yes | 053c7e6dfbdb4e46 085f9f7987b7cf56 0e2b9cc0beb5fb11 |
| NEW-missing-spaces | new | 13 | 4.3 | pyru satyanarayana | none | yes | 03aeffcf72c72418 167ef4a340862a0e 1ab888d9366a43fd |
| NEW-merged-table-cells | new | 11 | 3.7 | pyru satyanarayana | none | yes | 06f2d75b8c5610f1 1b1c42eddd14c8e7 1e7fad525310d310 |
| NEW-template-pages | new | 9 | 3.0 | pyru satyanarayana | none | yes | 28f56e454b2ade39 32b1f031d8c464e4 647d9713533c5664 |
| NEW-ai-generated-text | new | 8 | 2.7 | pyru | none | yes | 10d547035dd610e2 36c3e8b4b4457083 8f1f8a9c194e88f2 |
| NEW-missing-figures | new | 5 | 1.7 | satyanarayana | none | yes | 019fda08843696eb 03aeffcf72c72418 0ab64ed894610ef7 |
| NEW-aggregated-questions | new | 4 | 1.3 | satyanarayana | none | yes | 2a062df14b5b83a9 2c58af66b7d3bfe8 99d9b817f8c74cdf |
| B1 | new | 3 | 1.0 | pyru | none | yes | 61515ee94cba4c0c 8758fc16164dd1b6 d805befbd984308c |

### science

| code | kind | docs | % of docs read | readers | spec v1 rules | flag | example doc_ids |
|---|---|---|---|---|---|---|---|
| NEW-pdf-ligatures | new | 5 | 5.0 | aditeya | none | yes | 21f33206f24a2f0e 255f57592e1e996a 507e3afaa15e2375 |
| NEW-hyphen-split-words | new | 3 | 3.0 | aditeya | none | yes | 5a201c19575522f9 75768b1b6b165557 93412124dc441459 |
| NEW-merged-table-cells | new | 3 | 3.0 | aditeya | none | yes | 680400a6c14bde86 6cdcf2807cbf5156 6deaff0b02c57734 |
| NEW-missing-table-data | new | 3 | 3.0 | aditeya | none | yes | 0863ac75c9ec153e 4a33733776358fac fac0d63aba9fab0c |
| NEW-reading-order | new | 2 | 2.0 | aditeya | none | yes | e4d1400f9236b8c9 ef812364b6cca7a0 |
| NEW-seo-filler | new | 2 | 2.0 | aditeya | none | yes | 365b8539483e5a3f d8a1ed9b1607a1dd |
| NEW-toc-dot-leaders | new | 2 | 2.0 | aditeya | none | yes | 59aef569bc52cdff b1857f12c0d28f07 |
| NEW-underscore-blanks | new | 2 | 2.0 | aditeya | none | yes | 3b2c0420a33ccaab d8e3f724095e4aaa |
| C5 | new | 1 | 1.0 | aditeya | none | yes | 39bb03a0df7d4af5 |
| NEW-detached-diacritics | new | 1 | 1.0 | aditeya | none | yes | 21f33206f24a2f0e |
| NEW-figure-only-doc | new | 1 | 1.0 | aditeya | none | yes | 35d51dcf919068f8 |
| NEW-generated-image-captions | new | 1 | 1.0 | aditeya | none | yes | 79596acc4ee8648a |
| NEW-missing-spaces | new | 1 | 1.0 | aditeya | none | yes | f163172e98bdaa35 |
| NEW-orphan-figure-labels | new | 1 | 1.0 | aditeya | none | yes | 75183caf56d8e54b |
| NEW-pdf-quote-glyphs | new | 1 | 1.0 | aditeya | none | yes | a286f7d691161a05 |
| NEW-split-superscripts | new | 1 | 1.0 | aditeya | none | yes | db955ac64864c36c |
| NEW-spun-text | new | 1 | 1.0 | aditeya | none | yes | f82d23a50e08a31f |

## Missed: spec v1 has a rule, the cleaned doc still has the disease

| domain | code | docs | % of docs read | readers | spec v1 rules |
|---|---|---|---|---|---|
| code | E4 | 3 | 3.0 | revant | DA-E4-005 CANDIDATE; DA-E4-010 CANDIDATE; DA-E4-009 CANDIDATE; DA-E4-007 CANDIDATE |
| code | C3 | 2 | 2.0 | revant | DA-C3-002 CANDIDATE |
| code | E1 | 2 | 2.0 | revant | DA-E1-001 CANDIDATE; DA-E1-004 CANDIDATE; DA-E1-008 CANDIDATE; DA-E1-006 CANDIDATE; DA-E1-005 CANDIDATE; DA-E1-007 CANDIDATE |
| math | C1 | 167 | 55.9 | pyru satyanarayana | DA-C1-001 CANDIDATE |
| math | C5 | 22 | 7.4 | pyru satyanarayana | DA-C5-001 CANDIDATE; DA-C5-006 CANDIDATE |
| math | C2 | 20 | 6.7 | pyru satyanarayana | DA-C2-001 CANDIDATE |
| math | B5 | 18 | 6.0 | pyru | DA-B5-001 CANDIDATE |
| math | A3 | 7 | 2.3 | pyru satyanarayana | DA-A3-001 CANDIDATE |
| math | B4 | 5 | 1.7 | pyru satyanarayana | DA-B4-001 CANDIDATE |
| math | C3 | 4 | 1.3 | pyru satyanarayana | DA-C3-004 CANDIDATE; DA-C3-005 CANDIDATE |
| math | B2 | 3 | 1.0 | pyru | DA-B2-001 CANDIDATE |
| math | D3 | 1 | 0.3 | pyru | DA-D3-005 CANDIDATE |
| math | F2 | 1 | 0.3 | pyru | none |
| science | C1 | 18 | 18.0 | aditeya | DA-C1-007 ? |
| science | C2 | 4 | 4.0 | aditeya | DA-C2-004 ? |
| science | F1 | 4 | 4.0 | aditeya | DA-F1-006 ? |
| science | C3 | 3 | 3.0 | aditeya | DA-C3-006 ? |
| science | A1 | 1 | 1.0 | aditeya | DA-A1-006 ? |
| science | D1 | 1 | 1.0 | aditeya | DA-D1-001 ? |

## Harm: a spec v1 rule damaged real content

| domain | code | docs | % of docs read | readers | spec v1 rules |
|---|---|---|---|---|---|
| math | DA-C2-001 | 6 | 2.0 | pyru | DA-C2-001 math CANDIDATE |
| math | DA-D3-005 | 1 | 0.3 | pyru | DA-D3-005 math CANDIDATE |

## Warnings

- theschoolofai/era-v5-lab#350: base is theschoolofai/era-v5-lab:main, not mittalutkarsh/era-v5-lab:g1-reading
