# Native Indic share: Data B's proposal for Gate 2 (v1, draft of 12 Nov 2026)

Deliverable of #217 ("Our proposal for the native Indic share, by Thu 12 Nov") and the first item of #216. TSAI decides
the share at Gate 2 (14 Nov) and said that a pack with few clean documents is important evidence for it (G2, 31 Oct).
This file brings that evidence and a proposal. **What is measured and what is a proposed value is kept apart below.**
Only counts and pack IDs are used; no document text.

## 1. What "native Indic share" means here

The Indic text in the mix is of three kinds, and Data B has measured each:

| Kind | Packs | How it is measured |
|---|---|---|
| **Native**: a document written in the language's own script by people | `fw2`, `hplt`, `indiccorp`, `sangraha-v` | gold sets: documents that two readers both called clean |
| **Translated**: machine-translated text | the four `sangraha-s` packs (#212) | translation rating, 1 to 5, by native readers (due Thu 12 and Fri 13 Nov, #216) |
| **Romanised**: an Indian language in Latin letters (D2) | any pack | the document rule and the cap of #213 |

**Native share of a language = tokens of native documents ÷ all Indic tokens of that language in the mix**, after
cleaning and after the romanised cap. Counted in tokens, as TSAI's rule for the romanised cap (5 Nov) counts them,
because the data mix is set in tokens. (`sangraha-v` and `sangraha-s` are Sangraha's verified and synthetic splits; only
`sangraha-s` = translated is stated in #212. The other packs are read as native.)

## 2. Evidence (all measured, 22 sample packs of 500 documents each)

### 2.1 Clean documents in each pack (gold sets, `gold/gold_coverage.csv`)

"Clean" = both readers marked the document clean, and nobody disputed it. This is a **lower bound** of the clean share:
disputed documents are left out (ind-hi-hplt 71, ind-mr-fw2 57, ind-te-sangraha-v 37), and a pack with one reader has no gold
set, so it cannot be measured.

| Pack | Readers | Labelled | Gold (clean) | Clean | Disputed |
|---|---:|---:|---:|---:|---:|
| ind-hi-indiccorp | 2 | 500 | 425 | 85.0% | 0 |
| ind-bn-fw2 | 2 | 500 | 244 | 48.8% | 0 |
| ind-hi-fw2 | 2 | 500 | 234 | 46.8% | 0 |
| ind-bn-sangraha-v | 2 | 500 | 142 | 28.4% | 0 |
| ind-mr-fw2 | 3 | 500 | 142 | 28.4% | 57 |
| ind-ta-sangraha-v | 2 | 500 | 141 | 28.2% | 0 |
| ind-kn-fw2 | 2 | 500 | 114 | 22.8% | 0 |
| ind-kn-sangraha-v | 2 | 500 | 103 | 20.6% | 0 |
| ind-mr-sangraha-v | 2 | 500 | 88 | 17.6% | 0 |
| ind-gu-fw2 | 2 | 500 | 80 | 16.0% | 0 |
| ind-sa-fw2 | 2 | 500 | 71 | 14.2% | 0 |
| ind-hi-hplt | 4 | 500 | 62 | 12.4% | 71 |
| ind-kn-sangraha-s | 2 | 500 | 27 | 5.4% | 0 |
| ind-hi-sangraha-s | 2 | 500 | 24 | 4.8% | 0 |
| ind-te-sangraha-s | 2 | 500 | 24 | 4.8% | 0 |
| ind-te-fw2 | 3 | 500 | 18 | 3.6% | 0 |
| ind-te-sangraha-v | 3 | 500 | 18 | 3.6% | 37 |
| ind-ta-fw2, ind-ta-sangraha-s, ind-hi-sangraha-v, ind-ml-fw2, ind-or-fw2 | 1 | 500 (ml 422) | 0 | not measurable | 0 |

### 2.2 By source and by language (the 17 packs with two or more readers: 8,500 documents, 1,957 clean, 23.0%)

| Source | Packs | Clean |
|---|---:|---:|
| indiccorp | 1 | 85.0% |
| fw2 | 7 | 25.8% |
| sangraha-v | 5 | 19.7% |
| hplt | 1 | 12.4% |
| **sangraha-s (translated)** | **3** | **5.0%** (75 of 1,500) |

| Language | Packs | Labelled | Clean | Note |
|---|---:|---:|---:|---|
| bn | 2 | 1,000 | 38.6% | |
| hi | 4 | 2,000 | 37.2% | one pack alone is 85%; without indiccorp 21.3% (320 of 1,500) |
| ta | 1 | 500 | 28.2% | ta-fw2 and ta-sangraha-s not measurable |
| mr | 2 | 1,000 | 23.0% | |
| kn | 3 | 1,500 | 16.3% | |
| gu | 1 | 500 | 16.0% | |
| sa | 1 | 500 | 14.2% | |
| **te** | **3** | **1,500** | **4.0%** | **60 clean documents in three packs (18, 24, 18)** |
| ml, or | 0 | | not measurable | one reader each |

### 2.3 Romanised (`romanised/romanised_policy_v1.md`, #213)
The word-list detector finds 0 mostly-romanised documents in the 11,000 sample documents, so the cap share is 0% for
every language (12.1 million tokens). Romanised segments inside native documents are 0.025% of tokens. Proposed cap there:
1% per language (option B).

### 2.4 Translated
Ratings are not in yet. The review pack (#212) has 50 documents per language for Hindi, Kannada, Tamil and Telugu, 20
shared by two readers. `translation/quality_v1.md` and `agreement_v1.md` (due Fri 13 Nov) will give the mean score and
the agreement of each language.

### 2.5 What the cleaning code does to the clean documents
The filters of #214 (D1 script share, A1 invisible characters, G1 broken vowel signs) change or drop 0 of the gold
documents in each of the five packs tested so far (ind-bn-fw2, ind-bn-sangraha-v, ind-kn-fw2, ind-kn-sangraha-s,
ind-te-sangraha-s). So the clean counts above are not reduced by these filters; the other 12 packs are not run yet.

## 3. What the evidence says
1. **Translated text is the least clean source.** 5.0% clean in `sangraha-s`, against 25.8% in `fw2` and 19.7% in
   `sangraha-v`. This is the "pack with few clean documents" evidence TSAI asked for. It is a quality signal for the
   readers' notion of clean, not yet the translation rating.
2. **Telugu is short of clean text in every source**: 4.0% over three packs (fw2 3.6%, sangraha-s 4.8%, sangraha-v 3.6%).
   No share fixes this, and filtering alone cannot: at the sample rate 500 documents leave about 20 clean ones.
3. **The clean share depends on the source** (3.6% to 85.0%), so one global share would be wrong for some languages.
4. **Five packs cannot be judged** (one reader), among them Malayalam, Odia, and two Tamil packs.

## 4. Proposal

*Measured:* sections 2 and 3. *Proposed (my values, not derived from the data: the data gives the direction and the
per-language conditions, TSAI sets the numbers):*

| | Option A: native only | **Option B (proposed)** | Option C: permissive |
|---|---|---|---|
| Native share of each language's Indic tokens | 100% | **at least 90%** | at least 80% |
| Romanised (D2), within the above | 0% | **at most 1%** (#213 option B) | at most 2% (#213 option C) |
| Translated (`sangraha-s`) | 0% | **at most 9%, and only when the rating condition holds** | at most 18% |
| What it costs in the sample | drops the 3 measured `sangraha-s` packs: 75 of 1,957 clean documents (3.8%) | the same, but allows part of it back where rated well | keeps all of it |

Rating condition for translated text in Option B (scale 1 to 5, #212; both are proposed thresholds):
- a language's translated share may be used only if its mean rating is **3.5 or higher** and the two readers agree
  within 1 point on at least 80% of the 20 shared documents (the agreement rule of #216);
- a mean of **4.0 or higher** allows the full 9%; from 3.5 up to 4.0, **at most 4.5%** (half);
- a language with no rating, one reader, or a mean under 3.5 gets **0%** translated: nothing then shows it is safe.

Per-language rules:
- Languages short of clean text (Telugu now; ml, or and ta-fw2 once they have gold) are **not topped up with translated
  text**. Their clean supply is reported to #215 (new sources and licences) as the gap to fill.
- The share is checked per language on the training corpus, in tokens, with the project tokenizer (BrahmicTokenizer-131K
  is the stand-in, as in `token_share.py`).

**Why B and not A:** A throws away the one source that may be good (if the Hindi, Kannada, Tamil or Telugu ratings come
back high), at a cost of 3.8% of the clean gold documents. **Why not C:** at 5.0% clean, translated text is the weakest
evidence we have, and TSAI asked for a few-clean-documents pack to weigh most.

## 5. What is not known yet (so this draft will change)
- The translation ratings and the agreement check (Thu 12 and Fri 13 Nov): they decide the rating condition above.
- Tokens per source for the full corpus. The sample has 500 documents per pack, so the share of translated text in the
  sample is an artefact of how it was drawn and is not used here.
- Gold for `ind-ta-fw2`, `ind-ta-sangraha-s`, `ind-hi-sangraha-v`, `ind-ml-fw2`, `ind-or-fw2` (one reader each).
- The final tokenizer (the token share will be recomputed with it).
- Gold is rebuilt after #229 (2,081 documents in 18 packs, per the romanised policy); this table uses the 1,957 documents
  on main today.

## 6. Where the numbers come from
`teams/02-data-b/gold/gold_coverage.csv` (sections 2.1 and 2.2); `romanised/romanised_policy_v1.md` and
`romanised/romanised_token_share_v1.csv` (2.3); #212 (2.4, the scale and the review pack); `filters/README.md` and
`filters/harm_<pack>.md` (2.5); #216 and #217 (what is asked, and when).
