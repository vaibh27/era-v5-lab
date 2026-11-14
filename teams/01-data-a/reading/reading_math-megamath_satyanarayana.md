# Math cleaned-pack reading audit — math-megamath 

**Assigned reader:** Satyanarayana Murthy Toleti (Task: math-megamath).
**Review status:** AI-assisted, evidence-located draft for the assigned reader to verify before submission; **not a signed human full-text review**.
**Findings CSV:** `reading_math-megamath_satyanarayana.csv`.

## How it was read

- **Data:** `train/math-megamath/part-00000.parquet` (uploaded as `part-00000 (1).parquet`); tokenizer sample **v0-clean**, Train Part 1; 17,634 rows in this shard.
- **Pack file SHA-256:** `152c811a29bdd254f4c8778553222b6dcc5df04081267bba69d82345c3c87715`.
- **Selection:** `random.Random(20261101).shuffle(rows)` in file order; the first 100 IDs exactly match the supplied `part-1-100.txt` in the same order. No other IDs were assessed.
- **Coverage:** All 100 selected texts (714,044 Unicode code points total) were loaded and scanned for multiple possible defects; evidence spans and selected contexts were checked algorithmically, with AI-assisted context review. Long documents were not certified as exhaustively read by a human.
- **Cleaner provenance:** The supplied `README (8).md` identifies spec v1 `clean.py`, PR #244, as the producer of v0-clean. 
- **AI use:** Yes. Suggestions and evidence selection were made using AI assistance and deterministic script checks. All CSV rows therefore use `source=ai`.
- **Offsets:** Zero-based Python Unicode character positions in the supplied **cleaned** `text` field, end-exclusive. Each row points to a valid substring; multiple rows for a document represent different defects.
- **Quote policy:** Only short, non-sensitive exact snippets are included. No email addresses, phone numbers, credentials or other personal identifiers were deliberately copied. Any later F1/E3 findings must use empty `quote`.
- Out of 100 assigned documents, 23 document IDs have multiple findings.
- **Documents with 3 findings**

| Document ID | Findings |  
| :--- | :--- |
| 6234eb526d1b557f | NEW-missing-formulas (twice), C1| 
| 03aeffcf72c72418 | NEW-missing-figures, A3, NEW-missing-spaces|

- **Documents with 2 findings**

| Document ID | Findings |
| :--- | :--- |
| b109a58675c8efe4 | NEW-missing-spaces, C1 |
| cdfd72f2378bce40 | NEW-missing-spaces, NEW-flattened-formulas |
| e8df6ab0114ce01a | C1, C2 |
| 7b60768f817aab9d | NEW-template-pages, C1 |
| e915676c7a499a9e | C1 (twice) |
| 9bb621d8a0109e88 | NEW-seo-filler, C1 |
| 0e2b9cc0beb5fb11 | C5, NEW-seo-filler |
| 1e6794a405e417bc | C1, NEW-flattened-formulas |
| 8489105dd8ce9c39 | NEW-missing-spaces, C1 |
| 5c01c771f7f6c5a5 | C1, NEW-missing-spaces |
| c4ebacf9726dabf0 | C5, NEW-seo-filler |
| eb63a97816f5fbf5 | B4, C2 |
| 0ab64ed894610ef7 | C1, NEW-missing-figures |
| 99d9b817f8c74cdf | NEW-aggregated-questions, NEW-seo-filler |
| ebe7396e6712181d | C1, NEW-missing-formulas |
| 2c3cb9502e41cc15 | C1, NEW-missing-interactive-content |
| e11bb3d07abca5e6 | C1, NEW-merged-table-cells |
| 019fda08843696eb | NEW-missing-figures, C1 |
| 3d0c0e3284e57c48 | NEW-offtopic-poll, C1 |
| 3abc0b94ff60aced | C1, C5 |
| 0e0f204b1378b7ab | C1, NEW-truncated-previews |

- **Documents with no findings** not included in csv file

f6b0c5ab5e253ab2, 7cec07cc6fe8d58a, 6b9fe9c584ffb8c8, 96be7391f0fc2313, 3c443a374c59ec54, 1ddda4a3eb40f607, 081938780e1b462b, 62ec3a18aaaddbf0, 0e3eed22b9d5bb04, f29c7abbb1f25658, d6b8c2f6fb88adce, 3701aaa0e3f96c34, c68f66a9875a9390, 9ac49cc72ae88eea, cbf3c90fe391a88d, 2b8e7bc3f5834f25, 378680bc1686774d, 03ad8cc5282497ef, e237bf76a8e927d9, e9a1d57739c27140, 88a5f5488ae09e9b, a4a62b0b7250a5f9, ec50676544d50699, a6e8af81d71bd6cc, 40dc827f7dd9f308, e879baa97732a35d, 6f8233a84f4e4256, 3060ab2c476d7103, 72f9102d7df29da1, 1aacc525eede7dfe, 64c98bf48bb3916b, 42a5fcce28c6d951, 90e3e5f7ac3fade6, a7446ad23d76e908, 4a838ea16f534024, aa7a9be243350b93, 973c761beb0a1f75, 14c622c95752bf43, d8acdbab0ddf9655

## Findings summary

- **86 candidate findings** across **61 of 100 assigned documents**; 39 have no logged finding.
- Kinds: `missed` 50, `new` 36, `harm` 0.
- **Classification caution:** `missed` means a recognizable previously specified disease such as C1/C2/B4/C3/C5/A3 appears in cleaned text; the exact rule firing/coverage needs confirmation against the official `specv1.md`/`medicine_math.md`/`decisions_math.md`, not provided here.

| Code | Kind | Docs affected | % of 100 docs | Findings |
|---|---|---:|---:|---:|
| `C1` | missed | 31 | 31% | 32 |
| `C5` | missed | 12 | 12% | 12 |
| `NEW-missing-spaces` | new | 8 | 8% | 8 |
| `NEW-missing-figures` | new | 5 | 5% | 5 |
| `NEW-seo-filler` | new | 5 | 5% | 5 |
| `NEW-aggregated-questions` | new | 4 | 4% | 4 |
| `NEW-missing-formulas` | new | 3 | 3% | 4 |
| `A3` | missed | 2 | 2% | 2 |
| `C2` | missed | 2 | 2% | 2 |
| `NEW-flattened-formulas` | new | 2 | 2% | 2 |
| `NEW-orphan-footnote-markers` | new | 2 | 2% | 2 |
| `B4` | missed | 1 | 1% | 1 |
| `C3` | missed | 1 | 1% | 1 |
| `NEW-merged-documents` | new | 1 | 1% | 1 |
| `NEW-merged-table-cells` | new | 1 | 1% | 1 |
| `NEW-missing-interactive-content` | new | 1 | 1% | 1 |
| `NEW-offtopic-poll` | new | 1 | 1% | 1 |
| `NEW-template-pages` | new | 1 | 1% | 1 |
| `NEW-truncated-previews` | new | 1 | 1% | 1 |

## New diseases

| Name | Description | Documents |
|---|---|---:|
| `NEW-missing-spaces` | Word/section boundaries disappeared during text extraction. | 8 |
| `NEW-seo-filler` | Marketing and search-engine-oriented filler dominates subject matter. | 5 |
| `NEW-missing-figures` | Question refers to an illustration, diagram or graph absent from text. | 5 |
| `NEW-aggregated-questions` | Independent questions or comment threads merged into one record. | 4 |
| `NEW-missing-formulas` | Referenced expression or matrix is missing from the cleaned text. | 3 |
| `NEW-flattened-formulas` | Mathematical operators, grouping or superscripts were flattened or lost. | 2 |
| `NEW-orphan-footnote-markers` | Citation/superscript markers remain without a referent. | 2 |
| `NEW-template-pages` | Automatically templated low-information conversion pages. | 1 |
| `NEW-merged-documents` | Unrelated source topics interleaved in one document. | 1 |
| `NEW-missing-interactive-content` | Required interactive demonstration is not included in text extraction. | 1 |
| `NEW-merged-table-cells` | Table values have lost their row/column structure. | 1 |
| `NEW-offtopic-poll` | Non-mathematics interactive poll or voting content in math pack. | 1 |
| `NEW-truncated-previews` | Commercial preview truncation removes portions of the actual content. | 1 |

**Escalation:** The instructions note a domain-level >1% threshold (2+ documents out of 200). Confirm totals with the other math reader (`math-finemath`); this MegaMath-only sample cannot establish a combined-domain count on its own.

## Harm audit and limitations

**No `harm` findings are claimed. This does not establish zero harm.** The current upload provides only post-clean texts and a separate peer report for another pack. Without the MegaMath pre-cleaned version or a matching rule-by-rule cleaner trace, it is not possible to attribute a missing passage to a specific `DA-*` rule or locate a real deletion position.

## Examples for the lead

- `6234eb526d1b557f`: A Binet formula proof contains blank expressions, leaving key mathematical symbols absent.
- `c653ba40a72ae6a4`: ILS localizer/polarization procedures are interleaved with salary and taxation explanations in one cleaned record.
- `7c9fc7cb4cf0abc2`: Repeated `â€ś` sequences show surviving mojibake.
- `03aeffcf72c72418`: Rocket problem depends on an absent thrust-time chart and contains an anomalous `ยท` token.
- `eb63a97816f5fbf5`: `{moe}` template placeholder and a duplicated survey advice sentence remain.


## Coverage — the 100 document IDs in reviewed order

```text
f6b0c5ab5e253ab2 b109a58675c8efe4 7cec07cc6fe8d58a cdfd72f2378bce40 92af8a34e2277b29
6b9fe9c584ffb8c8 96be7391f0fc2313 e8df6ab0114ce01a 3c443a374c59ec54 b5d94d9f3b5ce7e4
7b60768f817aab9d e915676c7a499a9e 1ddda4a3eb40f607 9bb621d8a0109e88 081938780e1b462b
0e2b9cc0beb5fb11 c653ba40a72ae6a4 1e6794a405e417bc 6234eb526d1b557f 5d8a4e048b58b77b
3bba0803064abf48 2fd88179ccda81c5 8489105dd8ce9c39 5c01c771f7f6c5a5 e3308d611d2f995a
62ec3a18aaaddbf0 0e3eed22b9d5bb04 55298032cdb687cc 7c9fc7cb4cf0abc2 a4ba0e437a4f83b7
1797bf0bd4dcaddb 1364327b4a0afb15 f29c7abbb1f25658 ea5127ac0114999b d6b8c2f6fb88adce
3701aaa0e3f96c34 c68f66a9875a9390 9ac49cc72ae88eea a1f9ff1e77420780 2a77c711219ae876
bd99ef08cf1a9e40 cbf3c90fe391a88d 2b8e7bc3f5834f25 c4ebacf9726dabf0 378680bc1686774d
03ad8cc5282497ef e237bf76a8e927d9 6abbd5acb98679aa eb63a97816f5fbf5 efd3fb8487371128
aa3d833ed62e1d80 03aeffcf72c72418 362faf2a9b7861f9 c12731538fd49dc1 92332b39392821f5
e9a1d57739c27140 0ab64ed894610ef7 af77b2b619f87b0e 4cefaf2b18e1d8f5 2a062df14b5b83a9
380db5207dc7f47c 8141c9a74a964872 51f2d859f79bc925 88a5f5488ae09e9b 0246631d18a6ee21
a4a62b0b7250a5f9 ec50676544d50699 a6e8af81d71bd6cc bf7bebb6c74a660f 827108cc9357c8c5
ea37d46cbd5f339c 905f8b2a2c459bac 99d9b817f8c74cdf 40dc827f7dd9f308 e879baa97732a35d
ebe7396e6712181d 6f8233a84f4e4256 3060ab2c476d7103 72f9102d7df29da1 1aacc525eede7dfe
64c98bf48bb3916b 2c3cb9502e41cc15 7dd5c0e1365d04dc 42a5fcce28c6d951 167ef4a340862a0e
90e3e5f7ac3fade6 a7446ad23d76e908 2c58af66b7d3bfe8 2581a275cbb83d16 4a838ea16f534024
aa7a9be243350b93 e11bb3d07abca5e6 efe267fe695e9834 973c761beb0a1f75 019fda08843696eb
14c622c95752bf43 3d0c0e3284e57c48 3abc0b94ff60aced d8acdbab0ddf9655 0e0f204b1378b7ab
```

## Checks and submission notes

- CSV header is **exactly** `doc_id,kind,code,offsets,quote,why,source`.
- Every logged `doc_id` belongs to the supplied 100-ID list; the original Parquet was **not edited**.
- `source=ai` for every row; no manual or human sign-off is claimed.
- As per the assignment, **only the final `.csv` and `.md`** are submitted in in `teams/01-data-a/reading/` . Parquet files, local audit scripts or notes are not comitted.
