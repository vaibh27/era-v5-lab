# Marathi reader audit, day D4 — ind-mr-fw2

Refs #795. Reader: `patilvijay23`. Reviewed 25 decoded documents from
`ind-mr-fw2-train-000000`, using the issue's day-4 draw:
`random.Random(4).sample(range(11250), 25)`, in draw order.
This report covers the fw2 shard only; it does not cover `ind-mr-sangraha-v`.

Decoded with frozen p1536; tokenizer SHA-256:
`e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b`.
Source shard payload SHA-256:
`c6aa0cf3adb8e3f01741ad41cab255d033ac47a802040bd6ce60fe5a61681be4`.

Codes follow `diagnosis/DISEASES.md`. Offsets are zero-based Unicode character
positions in the decoded shard text. Multiple findings have separate rows;
summary counts use unique documents. D3 annotations on `ok` rows describe natural
language mixing and do not call for removal.

## Reader findings

| doc_id | shard | verdict | code | offset | note |
|---|---|---|---|---|---|
| `19dc4537a81c3357` | `ind-mr-fw2-train-000000` | finding | G1 | 171 | duplicated vowel sign in source; unchanged by cleaning |
| `6e3adb2e9dd2db54` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `a6a30e47fe1ae415` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `8aba1558f9777ca0` | `ind-mr-fw2-train-000000` | ok | D3 | 14 | Natural language mixing: romanised place name in the opening line |
| `fedd3a53aec6a062` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `f9c3f45f756a0473` | `ind-mr-fw2-train-000000` | ok | D3 | 303 | Natural language mixing: Latin-script rendering of a person name |
| `4b6a5bb8591220cf` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `38b78b36215beef6` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `e18a114a936cc021` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `6b1c0cc656315927` | `ind-mr-fw2-train-000000` | finding | C1 | 4106 | Related-news footer |
| `6b1c0cc656315927` | `ind-mr-fw2-train-000000` | finding | C2 | 196 | Repeated English horoscope phrases: one occurs 3x and another 2x |
| `13b9567325470c42` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `7b81e07d7d9098e5` | `ind-mr-fw2-train-000000` | finding | C1 | 1050 | English boilerplate starts here until end of doc |
| `4a6f4022afc284fd` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `9e14cfb48c502dc7` | `ind-mr-fw2-train-000000` | ok |  |  | Cleaner replaced ASCII vertical bar with Devanagari danda |
| `a17353b196212210` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `d1eaa48f0c647291` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `00b8a065d445abcb` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `d4f50f7bd5a46bec` | `ind-mr-fw2-train-000000` | finding | C1 | 1084 | Trailing headline and page-number fragment; probable navigation link |
| `d872d6172ae9c7f2` | `ind-mr-fw2-train-000000` | finding | C1 | 929 | News article boilerplate from 929 to EOD |
| `efc6e13a34102a1b` | `ind-mr-fw2-train-000000` | finding | C1 | 0 | App-install promotional boilerplate at start. Source observation: Nonstandard adjacent vowel-sign sequences exist 42x (first offset 185), all identical in raw source |
| `c2e67c57aba8c085` | `ind-mr-fw2-train-000000` | finding | C1 | 184 | Residual table-of-contents navigation marker |
| `c2e67c57aba8c085` | `ind-mr-fw2-train-000000` | finding | C1 | 8492 | Dangling further-information and source headings |
| `a7d9c397fee12e11` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `f6e598a21b1e9128` | `ind-mr-fw2-train-000000` | ok |  |  |  |
| `48c41e2628c095bd` | `ind-mr-fw2-train-000000` | ok |  |  | Article-related image caption with news-agency attribution; natural language mixing |
| `3350cc5269689c9e` | `ind-mr-fw2-train-000000` | finding | C1 | 1438 | Publisher promotion for news coverage and mobile app |

## Source comparison

All 25 document IDs were matched to the raw sample-v0 Marathi fw2 part, whose SHA-256 matches `audit/raw_parts_SHA256SUMS`. For 24 documents, decoded shard text is identical to raw text after the frozen tokenizer's normalization. The remaining document, `9e14cfb48c502dc7`, differs only by an intentional vertical-bar-to-danda punctuation repair at offset 6 (`DB-G4-002`).

The G1-coded duplicated vowel sign in `19dc4537a81c3357` is already in the raw source, unchanged by cleaning; it is not evidence of damage introduced by the cleaner. Likewise, the 42 adjacent vowel-sign sequences noted for `efc6e13a34102a1b` are source observations, not 42 additional findings or confirmed word splits. No cleaning-induced G1–G4 damage or joiner loss was identified in this sample.

The local cleaned parquet differs from the decoded shard for `7b81e07d7d9098e5`; the decoded shard matches the raw source for that document. This audit uses the decoded shard text, including the publisher promotion at offset 1050, rather than that local cleaned copy.

## Summary

- Documents read: 25
- ok: 17
- With a finding: 8
- Finding rows: 10
- Codes found: C1 (8), C2 (1), G1 (1)
- Descriptive annotations: D3 in 2 otherwise okay documents.
