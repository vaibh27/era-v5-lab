# Audit collation: every finding, the rule behind it and its proposal

Week 4, tasks 4.4 and 5.1. 22 Nov 2026. Collected by `TEJASKUMAR-REDDY-J` from `day_D3.md` and the audit PRs #68, #71, #72, #76, #78 and #79 of patilvijay23/era-v5-lab. No cleaner rule is changed here.

**The medicines of these audits are put together in [`collated_medicine.md`](collated_medicine.md), with the conflicts between them resolved. What was used is listed under [Medicines used](#medicines-used).**

The audits read different cleaned outputs. So every labelled document was run again through the current `spec_v1_1/clean.py`, rule by rule, to see which findings are still there and which rule causes each.

| File | What it holds |
|---|---|
| [`collation_findings.csv`](collation_findings.csv) | 30 findings: audit, documents, disease, current rule, whether it is still there, proposal, its card in the collated medicine, next step |
| [`collation_docs.csv`](collation_docs.csv) | 313 labelled documents: label, what the current cleaner does, the rules that acted in order |
| [`collation_scope.csv`](collation_scope.csv) | Documents each rule fixes and drops on the English and math train parts |
| [`collation_replay.py`](collation_replay.py) | The script that writes `collation_docs.csv` and `collation_scope.csv` |
| [`collated_medicine.md`](collated_medicine.md) | 15 cards from #71, #75, #76 and #79 as one set that loads in `harm_test.py` |
| [`collated_medicine_docs.csv`](collated_medicine_docs.csv), [`collated_medicine_packs.csv`](collated_medicine_packs.csv) | What the set does on the 313 documents and on seven whole packs |
| [`collated_medicine_check.py`](collated_medicine_check.py) | The script that writes these two and checks the gold sets |

The files hold ids, rule ids and counts only: no document text and no personal data.

## Coverage

| Audit | Pack | Labels | Current cleaner: fix / keep / drop |
|---|---|---:|---|
| `day_D3.md` | 29 label files | 71 | 10 / 17 / 44 |
| #68 | en-dclm | 35 | 28 / 0 / 7 |
| #71 | math-finemath, math-megamath | 35 | 28 / 0 / 7 |
| #72 | code-go | 45 | 37 / 0 / 8 |
| #76 | en-nemotroncc-synth | 76 | 75 / 0 / 1 |
| #78 | code-js-ts | 45 | 41 / 0 / 4 |
| #79 | sci-finepdfs | 6 of 35 | 6 / 0 / 0 |
| **Total** | | **313** | **225 / 17 / 71** |

- **#79:** its label CSV lists only the 6 `fix` and `drop` documents, so the 29 `clean` ones are not here.
- **Not here:** the Gujarati and Marathi supplement of `day_D3.md`, and the packs with no audit PR on 22 Nov: code-cpp, code-java, code-python, code-rust, en-fineweb-edu and en-nemotroncc.
- **#68 tally:** the report and the CSVs agree: 25 `fix`, 2 `clean`, 1 `drop`, and 7 dropped by the cleaner (2 right, 3 acceptable, 2 wrong).

## Main findings

1. **`DA-C3-003` drops good documents, and only one pack has a proposal.** It drops a document whose last word is a word like `this`, `at` or `by`. On the four English packs it drops 3,019 documents, 1,959 of them over 2,000 characters. It is 707 of the 1,155 drops on en-dclm, and 14 of the 22 English drops in `day_D3.md`. Both wrong drops of #68 and the one of #76 come from it.
   - The endings of 40 of the 707 en-dclm drops were read (one reader, endings only). About 32 end in site chrome or in a link or contact that was already removed (`Share This`, `find us at`), about 6 are cut off, one is a finished sentence that ends in a curly quote, and one is a raw feed.
   - #76's `DA-C3-008` withdraws the rule on en-nemotroncc-synth, which has 63 of the 3,019. Withdrawing it on en-dclm also brings back #68's documents whose drop was right or acceptable, so it needs a narrower test.
2. **`DA-C7-002` drops 1,498 of 17,647 megamath documents (8.5%)** because the text mentions figures. Two of them are labelled `fix` in #71, one as good content.
3. **#71 read an older output.** In it the cleaner had dropped 13 megamath documents; the current cleaner drops 2,550. On finemath it is 600 and 2,609. Its labels hold for the documents still kept. **#79 too:** its report says the cleaner drops none of the 6,011 sci-finepdfs documents; the current cleaner drops 166.
4. **The rules #71 could not name:** the deleted reference and author lines are `DA-C5-006`, and the deleted formula lines are `DA-C2-001`. `DA-C5-006` removes 690 comma-list lines in 199 megamath documents; about 607 of them look like reference entries (a rough pattern, not read).
5. **`day_D3.md` is partly out of date.** Of its 56 dropped documents, 12 (all Indic) are kept now and 44 are still dropped; `collation_docs.csv` names the rule for each. All 6 visarga documents come through unchanged. The 9 licence-header documents still have licence text at the top.
6. **Cleaner damage with no proposal:** `DA-A6-001` changes the Cyrillic letters of two Go test fixtures (#72).

## Medicines used

Every executable proposal of the audits is used: 15 proposals, 15 cards.

| Source | Used | As | Not used |
|---|---|---|---|
| #75 (`2ecacf5f`) | all 5 | `DA-A2-010`, `DA-B3-012`, `DA-B4-016`, `DA-C2-009`, `DA-F1-007`; also one half of `DA-C2-010` and `DA-F1-009` | |
| #76 (`2ba7ce3b`) | all 6 | `DA-C1-013`, `DA-C3-008`, `DA-C5-007`, `DA-B4-015` as they are; the other half of `DA-C2-010` (its `DA-C2-008`) and of `DA-F1-009` (its `DA-F1-008`) | M7: a policy question, no code |
| #71 (`4d17b696`) | both | `DA-C1-012`, `DA-C2-007`; on finemath `DA-C2-007` runs in the place of `DA-C2-001` | |
| #79 (`489593e9`) | both | `DA-C2-011` (Medicine 1), `DA-C1-014` (Medicine 2, part a) | Medicine 2, part (b): it changes documents labelled `clean` |
| #68, #72, #78, `day_D3.md` | findings only | | no executable medicine in them (#68's five bugs are #75's cards) |

**Results of the whole set:** gold 0 changed and 0 dropped by every card on all seven packs, and the output of `clean()` is the same on all 989 gold documents with the cards in place. Of the 313 labelled documents 55 get another output, none labelled `clean`, none newly dropped. On the whole packs 60 documents are recovered and 3 are newly dropped (finemath, `DA-C7-001`); these 3, and the 3 recovered outside synth, are named for a reader. All five bugs of #68 are fixed, and all 25 wrong phone masks of #76's review are restored while its 8 ambiguous or private ones stay. Details and the whole packs are in [`collated_medicine.md`](collated_medicine.md).

## Conflicts and how they are resolved

- **#75 and #76 replace the same rules** (`DA-C2-003` and `DA-F1-005`) on en-nemotroncc-synth. Both are merged into one card for that pack; the other three packs keep #75's.
  - `DA-C2-010`: a repeated line is left alone if either source leaves it alone.
  - `DA-F1-009`: #76 decides whether a line gives a reason to mask, #75 makes it one decision for each number. A number that looks public in one place and private in another is masked everywhere (privacy first).

  | doc_id | phone masks now | with #75 | with #76 | with `DA-F1-009` |
  |---|---:|---:|---:|---:|
  | `8aecb4bed3222e59` | 1 | 2 | 0 | 0 |
  | `417036b98dd9380a` | 1 | 2 | 0 | 0 |
  | `d6e8da1b50823198` | 1 | 2 | 1 | 2 |

  #76 judges the first two as public contacts and marks the third for review, so the merged card agrees with its labels on all three. **For the team lead to confirm:** privacy first, and masking the second copy in the third document.
- **Rule IDs of #75:** `DA-A2-009`, `DA-B3-002`, `DA-B4-014` and `DA-C2-005` are already in `rules/registry.json`, and `DA-C2-005` was also an English rule dropped in `spec_v1/sheets/decisions_english.md`. In the collated medicine they are `DA-A2-010`, `DA-B3-012`, `DA-B4-016` and `DA-C2-009`; `DA-F1-007` is free and stays. #75 itself is not edited, so its sheet still has the old numbers.
- **#71 `DA-C2-007` takes the place of `DA-C2-001`** on finemath: run last, it would let `DA-C7-001` and `DA-C9-003` drop 21 documents that are kept today. Retiring `DA-C2-001` is still a team decision.

## #68: each finding and its medicine

| Finding in #68 | Current rule | Medicine |
|---|---|---|
| cp1252 quotes deleted | `DA-A2-008` | `DA-A2-010` (#75 `DA-A2-009`) |
| Lines cut inside code | `DA-C2-003` | `DA-C2-009` (#75 `DA-C2-005`); record fields (not on synth) and formula lines are still cut |
| Math removed inside dollar signs | `DA-B4-010` | `DA-B4-016` (#75 `DA-B4-014`); sets that start with a letter, minus or bracket are still cut (2 en-dclm documents, up to 3 synth) |
| Phone masked once | `DA-F1-005` | `DA-F1-007` (#75); a `+44` copy of the number is not matched |
| `<s>` and `</s>` kept | none | `DA-B3-012` (#75 `DA-B3-002`); also strips literal examples and code fences |
| Two wrong drops | `DA-C3-003` | **None for this pack** |
| Wrong language, gutted | `DA-D3-003` | **None** |
| Boilerplate (C1), 25 documents | `DA-C1-006`, `DA-C2-003` | **None for this pack** (#76 `DA-C1-013` is synth only, #71 `DA-C1-012` math only) |
| Joined pages (C4), PDF bullets (A3), SEO line (C5) | none | **None** |

The five cards of #75 were checked again on its branch (`2ecacf5f`): gold harm 0 changed and 0 dropped on the four English packs, and `clean()` at 0.97x of spec v1.1's speed on 2,000 documents (the earlier run was 1.02x; runs vary by about 5%).

## How to reproduce

From the repository root, with the raw sample train parts in `<raw>`:

```bash
python -I teams/01-data-a/audit/collation_replay.py docs --raw <raw> --labels teams/01-data-a/audit/collation_docs.csv --out collation_docs.csv
python -I teams/01-data-a/audit/collation_replay.py scope --raw <raw> --out collation_scope.csv
```

`collated_medicine.md` has the commands for its own checks. `docs` takes its labels from the committed CSV and writes the same file again. Each replay is compared with `clean()` and the script stops on a difference. `clean.py` sha256 `6a57276c0e83312b…`, the same as in #76.

## Limits

- The replay uses the raw tokenizer sample v0 train parts, not corpus-v1 and not the token shards.
- A finding marked "still there" means the same characters or the same drop are in the current output. It does not say how common the problem is. Audited documents were mostly picked from the ones the cleaner changed.
- The 40 endings were read by one person, and the wrong and right drops of `DA-C3-003` are not labelled.
- No new medicine is written here: the cards are the audits' own, merged where they clash. The findings marked **None** still need a card and a harm test.
