# code-python audit: findings and follow-ups

Reviewer: `patilvijay23`. Recorded 22 Nov 2026 (Week 4 D1), completing the Week 3 Python audit carryover. **45 documents read in full: Part_1 30, Part_2 15**, exceeding the 25 + 10 target.

## Inputs and method

Compared raw with supplied cleaned tokenizer-sample-v0 Parquets, matched by `doc_id`. This is not a corpus-v1 or decoded-shard audit. Raw layout: `my-audit/pre/code-python/Part_N/part-NNNNN.parquet`; cleaned layout: `my-audit/post/code-python/part-NNNNN.parquet`. Files were copied from `tokernizer_sample/train/code-python/` and `tokenizer_sample_v0_clean/train/code-python/`. Selection seed and viewer revision were not recorded; this is not an unbiased whole-pack quality estimate.

Replayed the approved `teams/01-data-a/spec_v1_1/clean.py`, SHA-256 `6a57276c0e83312b09facfba656cca28db0ef59ca851bcfbfa3ef63f16fdd1f2`, on all reviewed raw documents. **45/45 match the supplied post output exactly.** Replay confirms reproducibility, not the desirability of each transformation. No source data or cleaner was changed.

| Part | Raw docs | Cleaned docs | Full reads | clean | fix | drop recommendation |
|---|---:|---:|---:|---:|---:|---:|
| Part_1 | 55,143 | 53,419 | 30 | 29 | 1 | 0 |
| Part_2 | 19,800 | 19,185 | 15 | 14 | 0 | 1 |
| **Total** | **74,943** | **72,604** | **45** | **43** | **1** | **1** |

The 2,339 documents absent after cleaning are whole-pack counts, not manually reviewed drops. The CSV's one `drop` is a recommendation for a retained document. `clean` records the reviewer's judgment, not a proof of compilation or semantic equivalence.

Submitted labels are preserved unchanged, including the supplied evidence field and B5 code. [Part_1 CSV](day_D1_code-python_Part_1.csv) and [Part_2 CSV](day_D1_code-python_Part_2.csv).

## Findings and medicine decisions

### Stray terminal Markdown fence — `8e6878eef322a84a` (Part_1)

Label: `full / fix / B5`; evidence supplied: triple backticks at offset 2937. The raw file has 2,941 characters and the cleaned file 2,929. Both contain one triple-backtick marker at the end: the marker was **not introduced by cleaning**. The submitted offset refers to the raw tail; it must not be treated as a verified post offset. Source path ends in `program/recog.py`.

The issue is a retained/unmatched Markdown wrapper in a Python file. B5 normally describes broken mathematical notation; this label needs second-reader adjudication, potentially as a markup/wrapper issue rather than B5. No destructive cleaner rule is attributed to this residual finding.

**Suggested medicine follow-up:** investigate a code-python-only removal of an unmatched terminal fence outside Python strings/comments and genuine Markdown examples. Do not globally delete backticks or balanced examples. A replacement/card requires positive and negative cases, current code-python gold harm results and idempotence evidence before approval. No executable medicine is claimed validated here.

### Unresolved Git merge conflict — `b45240bf444a60d8` (Part_2)

Label: `full / drop`; reviewer note: “should be in code-commits”. Raw and cleaned text retain conflict markers such as `<<<<<<< HEAD` and `=======` around duplicated Python implementations. Source path ends in `CodeChef/Python/chef_ and_special_numbers.py`; raw length is 2,608 characters and post length is 2,463.

This is a Python source file containing an unresolved merge conflict, not established to be a commit-message/diff document. The proposed move to a commits pack is therefore a **routing/quality decision**, not a confirmed correction. Preserve the drop recommendation pending review; do not silently move it into `cadj-commits` or change the approved corpus universe.

**Suggested medicine follow-up:** decide whether a source file with a complete, unresolved conflict block should be excluded from the code pack. If approved, use a scoped whole-line conflict-marker detector that requires a structurally complete block and protects string literals, comments and conflict-handling test fixtures. Do not automatically choose either side of the conflict. Record current code-python gold harm, retained-code counterexamples and proposed action before adoption. No new drop rule or harm-pass claim is introduced by this audit.

## All reviewed documents

| Part | doc_id | Read | Verdict | Disease | Finding |
|---|---|---|---|---|---|
| 1 | `1875128630a1bba2` | full | clean | — | No issue reported in full read. |
| 1 | `73823a512f6e2e3d` | full | clean | — | No issue reported in full read. |
| 1 | `f83e99cda95bb746` | full | clean | — | No issue reported in full read. |
| 1 | `27fbd914ef0648d1` | full | clean | — | No issue reported in full read. |
| 1 | `b614ee926735d4d8` | full | clean | — | No issue reported in full read. |
| 1 | `9dc85c280fe76a44` | full | clean | — | No issue reported in full read. |
| 1 | `c78c89448a5ad6f5` | full | clean | — | No issue reported in full read. |
| 1 | `0ce9250251c58b8c` | full | clean | — | No issue reported in full read. |
| 1 | `efcec3d43a383c6d` | full | clean | — | No issue reported in full read. |
| 1 | `51aa5e9fb7139141` | full | clean | — | No issue reported in full read. |
| 1 | `31b29bb9b9350b48` | full | clean | — | No issue reported in full read. |
| 1 | `f9a0e0adb6b005fb` | full | clean | — | No issue reported in full read. |
| 1 | `8e6878eef322a84a` | full | fix | B5 | Residual terminal fence already present in raw; B5 classification/offset needs review. |
| 1 | `f6698b8fa4221c7b` | full | clean | — | No issue reported in full read. |
| 1 | `92ad8826618e62f1` | full | clean | — | No issue reported in full read. |
| 1 | `f8e5b5b597c778c8` | full | clean | — | No issue reported in full read. |
| 1 | `3c4ef32a5765ccd7` | full | clean | — | No issue reported in full read. |
| 1 | `974d7e4648101d38` | full | clean | — | No issue reported in full read. |
| 1 | `51a498becce55b8c` | full | clean | — | No issue reported in full read. |
| 1 | `c4fcc81ef2677310` | full | clean | — | No issue reported in full read. |
| 1 | `a062d06c53d7ab2e` | full | clean | — | No issue reported in full read. |
| 1 | `e072b6e022783686` | full | clean | — | No issue reported in full read. |
| 1 | `c6de0e378bf4f0b0` | full | clean | — | No issue reported in full read. |
| 1 | `57b789826e6986e4` | full | clean | — | No issue reported in full read. |
| 1 | `03464151a2ea51fb` | full | clean | — | No issue reported in full read. |
| 1 | `aff07204ca30765b` | full | clean | — | No issue reported in full read. |
| 1 | `3a240a04571db9ae` | full | clean | — | No issue reported in full read. |
| 1 | `a8d8b064ce8f86de` | full | clean | — | No issue reported in full read. |
| 1 | `c821aef889b9ebf9` | full | clean | — | No issue reported in full read. |
| 1 | `c419c3f8804d155a` | full | clean | — | No issue reported in full read. |
| 2 | `b45240bf444a60d8` | full | drop | — | Retained unresolved merge conflict; drop/routing recommendation pending adjudication. |
| 2 | `51565dd215dbae05` | full | clean | — | No issue reported in full read. |
| 2 | `d9fdbb9e9adaa0c2` | full | clean | — | No issue reported in full read. |
| 2 | `5126cb4cb1aec655` | full | clean | — | No issue reported in full read. |
| 2 | `e4a14609cf11b949` | full | clean | — | No issue reported in full read. |
| 2 | `32ae866658fa5d2a` | full | clean | — | No issue reported in full read. |
| 2 | `16c8824d2dc82152` | full | clean | — | No issue reported in full read. |
| 2 | `1b64ec95cd8ba250` | full | clean | — | No issue reported in full read. |
| 2 | `243e3260c3b229ec` | full | clean | — | No issue reported in full read. |
| 2 | `742b641ca4a41956` | full | clean | — | No issue reported in full read. |
| 2 | `c89b4df61e9c8032` | full | clean | — | No issue reported in full read. |
| 2 | `f747a576c3c320b6` | full | clean | — | No issue reported in full read. |
| 2 | `0cefd682421b72a4` | full | clean | — | No issue reported in full read. |
| 2 | `8bf313d17189287b` | full | clean | — | No issue reported in full read. |
| 2 | `0c2d0fd0c5fe32c7` | full | clean | — | No issue reported in full read. |

## Input identities

| Part | Input | Rows | SHA-256 |
|---|---|---:|---|
| 1 | raw | 55,143 | `8d9575b43cfb9cf3647e63e91d026eb8ed40f3981b698da33b83d09e6326641d` |
| 1 | cleaned | 53,419 | `1ea06089a9cf4bd58e5ffb21f6a6e92d7cb1a986de7f5af7addd078374909068` |
| 2 | raw | 19,800 | `985831590bc02ebddc0276223e1f022fde0e7c2e809fb7d97e74492bb6f5aa16` |
| 2 | cleaned | 19,185 | `ee73f83bb2daffe4fec15e54fca77e06d0cd4a1b19b1873bc9db392e3529f726` |

