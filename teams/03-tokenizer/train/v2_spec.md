# v2 spec: o200k pruning by usage, slot splits chosen on held-out text (#163)

## Goal

Widen the English margin with **o200k pruning by usage**, **without compromising any Indic or Dravidian
language**, and choose the slot split on held-out FineWeb-2 text. **FLORES is not used**: not for choices, and not
for reporting v2.

v1-c1 on held-out check text: English +1.74%, code +1.41% vs BrahmicTokenizer-131K. The base keeps the first
114,332 admissible o200k ranks. o200k's rank order follows OpenAI's training mix, so the base holds tens of
thousands of Cyrillic, CJK, Arabic, Hebrew, Thai and Greek tokens that our text never uses, while English and code
tokens just past the cut are missing.

## Fixed

| What | Value |
|---|---|
| Vocab size | 131,072 |
| Special ids | tokenizer v0's 356 entries at their exact ids (#107) |
| Script blocks | ids 114,333 and up, 16,384 merges in total, as v1-c1 |
| Base ids of every token that is kept | as v1-c1 |
| Normalizer, pre-tokenizer, byte map | as v1-c1 |

Only base ids whose token is **never used** on our text change: their tokens are pruned and the ids are reused.

## Data (held-out, `fetch_heldout.py`)

| Set | Source | Used for |
|---|---|---|
| Train (Indic) | FineWeb-2 `train`, the v1 corpus (60 MB Ta/Te/Kn/Ml, 20 MB others) | script merges; base usage for pruning |
| Usage (En / code) | FineWeb `sample-10BT` 30 MB, codeparrot-clean-valid 20 MB | base usage for pruning; ranking new tokens |
| Dev | FineWeb-2 `test` 2 MB per Indic language; next 5 MB En / 3 MB code | all choices |
| Check | the next 2 MB / 5 MB / 3 MB | report only, never used for a choice |

- A held-out document is dropped if half or more of its long lines occur in the training corpus.
- A code file is dropped if it shares a 13-gram with HumanEval or MBPP.
- Hashes: `heldout_manifest.json`.
- FLORES, IN22, HumanEval, MBPP and GSM8K are not read by any v2 script.

## Method

### 1. Prune list (`prune_o200k.py`)
- Encode the usage text and the full Indic training corpus with the v1 base alone, and count how often each base
  token is used.
- A base token can be pruned when it is used **0 times**, has more than one byte, contains **no Brahmic
  character**, and **no kept base token is merged from it**. The base must stay closed under its merges.
- Order: non-Latin scripts first, then Common, then Latin. Within each group the rarest o200k rank goes first.
  Every prefix of the list is a valid prune set.

### 2. New tokens (`prune_o200k.py`)
- Candidates: o200k tokens past the v1 cut (rank 114,483), at most 32 bytes, at most one script.
- Ranked **separately** for English and for code by `count × (base tokens saved − 1)`. A token counts at most 4
  times per 20 KB chunk and must appear in at least 3 chunks, so one generated file cannot dominate.
  Each token brings its o200k merge parents with it.

### 3. Swap size (`slot_search.py`)
- Find the smallest code prefix and the smallest English prefix that reach **parity with BrahmicTokenizer-131K
  on dev** (`--target 0`), then check the combination.
- `train_bpe.py` removes as many prune-list tokens as there are new tokens, skipping base ancestors of the new
  tokens. It puts the new tokens at the freed ids, in o200k rank order, with their merges after the kept base
  merges.

### 4. Slot split (`slot_search.py`)
- Train each script once with up to 1,536 extra merges. The first `n` merges of that run are exactly a run with
  `n` slots, so one run gives a dev fertility curve per language.
- Floor: no language worse on dev than with its v1 slot count. The script budget stays 16,384, and every curve
  only goes down as slots are added, so the only split that meets the floor is v1-c1's split.
- The mean-only split is reported for comparison.

### 5. Gates
- The 3 verify scripts PASS, and the round-trip sample passes (`train_bpe.py` exits 1 otherwise).
- `check_ids.py --manifest`: specials identical, every id from 114,333 up identical, and every base id identical
  except the swapped ids listed in the manifest. Kept base merges are identical and in the same order.
- Without `o200k_swap`, v1-c1 still rebuilds byte-identically.

## Results (held-out; full tables in COMPARE.md, section v2)

- Swap: 13,850 ids (3,840 code + 10,240 English tokens). The pruned tokens are Cyrillic 5,546, CJK 3,264,
  Arabic 2,751, Hebrew 1,029, Thai 679, Greek 517 and Sinhala 64. No Latin, Common or Brahmic token was pruned.
  22,737 more prunable tokens remain unused.
- Check split vs BrahmicTokenizer-131K: English **−0.10%** (v1-c1 +1.74%), code **−0.13%** (v1-c1 +1.41%),
  mean Indic −17.45% (v1-c1 −17.41%).
- Check split vs v1-c1: all 11 Indic languages are 0.02–0.08% better. On dev, Bn is +0.01% (2.1736 vs 2.1734),
  which is noise; every other language is better.
- Ids: all 16,739 ids from 114,333 up and all specials are identical to v1-c1. The script merges come out
  byte-identical, because the pruned tokens never occur in Indic text. 100,483 base ids are unchanged.
- Slot split: v1-c1's split. The mean-only split would make Hi +6.0%, Mr +8.0%, Ml +2.0%, Ta +1.1% and Te +1.1%
  worse on dev.

## Open question for the lead

The README says "never reuse an id a model was trained with". v2 reuses 13,850 ids, all of which held tokens
that are never used on our English, code and Indic text. If G4/G5 have trained on v1-c1, please confirm this is
acceptable. If it isn't, the alternative is to take ids from the script blocks, which costs Indic quality (tried
and rejected: dev mean Indic 5.6% worse for about 1 point of English margin).

## Next improvements

Ordered by expected value. Each becomes its own `[G3]` issue when it starts.

1. **More Indic merges from the remaining prunable ids.** 22,737 unused base tokens are still prunable. Giving
   their ids to script merges would improve Indic, the Dravidian scripts most: on dev, +1,536 merges lowers
   Ta/Te/Kn/Ml by roughly 8–10%. This breaks "one contiguous id block per script", so it needs a team decision.
   The verify scripts don't require contiguous blocks.
2. **More and cleaner training text.** Use Data A/B's cleaned output, and drop boilerplate (C1) and machine-
   translated spam (D4) with the diagnosis medicines: in the Bengali FineWeb-2 diagnosis pack, 95 of 500 docs had
   boilerplate and 36 were MT spam. Then go from 20–60 MB to 100 MB or more per language.
3. **ZWJ / ZWNJ.** The Bengali FineWeb-2 pack has 0 ZWJ/ZWNJ in 1.34M characters (stripped upstream), so no merge
   involving them is learned (for example র‍্যা). Measure their rate in held-out text from other sources and decide
   between the normalizer and the training text.
4. **Bigger usage sets.** Prune decisions rest on 30 MB of English, 20 MB of Python and the Indic corpus. Add
   other code languages (JS, Java, C++) and other English sources before pruning past the non-Latin tokens.
5. **Per-language weights.** The objective is the plain mean of 11 languages. Weighting by expected share of
   training tokens (Data A/B, G4) would match what the model sees. This needs a team note.
6. **Test with a model (G4/G5).** Compare v1-c1 and v2 on bits per byte with a small model.
