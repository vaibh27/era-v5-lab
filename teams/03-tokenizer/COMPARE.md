# COMPARE: tokenizer comparison (Week 1)

Produced by `harness/run_harness.py --tokenizers all --suites flores,code --check-reference` on 27 Oct 2026.
HF tokenizers are pinned to the commit in `REGISTRY` (`harness/run_harness.py`), all load with `trust_remote_code=False` and no HF login.

Tokenizer v0 (`v0/`, #107) only renames reserved special slots, so it encodes plain text to the same ids and has the same numbers as the BrahmicTokenizer-131K rows.

**Pass bar:** beat BrahmicTokenizer-131K on mean Indic fertility; at most 2% worse on English (FLORES En fertility), HumanEval and MBPP (tokens / char).

## FLORES-200 fertility (tokens / word, lower is better; Mean = 11 Indic languages)

| Tokenizer | HF repo | En | Hi | Bn | Ta | Te | Kn | Ml | Mr | Gu | Pa | Or | As | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **BrahmicTokenizer-131K** | theschoolofai/BrahmicTokenizer-131K | 1.2352 | 1.6429 | 2.3295 | 3.1166 | 3.0292 | 3.2734 | 3.4738 | 2.5723 | 2.2748 | 2.5955 | 4.1287 | 2.7535 | **2.8355** |
| o200k_base | tiktoken | 1.2229 | 1.6432 | 2.3330 | 3.1496 | 3.0486 | 3.3039 | 3.4955 | 2.5741 | 2.2901 | 2.7187 | 6.7948 | 2.7624 | 3.1013 |
| Gemma-4 | google/gemma-4-E4B | 1.2366 | 1.3857 | 1.6764 | 2.3649 | 2.8299 | 3.2606 | 3.3266 | 1.9986 | 2.4006 | 2.8237 | 4.8339 | 2.8158 | 2.7015 |
| Qwen3.8 | Qwen/Qwen3.8-27B | 1.2530 | 2.1566 | 3.0695 | 4.2182 | 4.6199 | 5.3352 | 5.5795 | 3.3347 | 4.0362 | 4.0662 | 5.4188 | 4.2060 | 4.1855 |
| Qwen3-8B | Qwen/Qwen3-8B | 1.2572 | 4.7500 | 7.0953 | 9.9963 | 11.3522 | 11.8438 | 13.2906 | 6.6980 | 8.8635 | 7.7576 | 13.5997 | 7.7003 | 9.3588 |
| DeepSeek-V4 | deepseek-ai/DeepSeek-V4-Flash-Base | 1.2306 | 2.9477 | 2.7854 | 4.9161 | 5.9688 | 6.2065 | 7.9431 | 4.2720 | 4.9805 | 4.6761 | 7.6055 | 3.7633 | 5.0968 |
| Sarvam-30B | sarvamai/sarvam-30b | 1.2368 | 1.3857 | 1.6765 | 2.3649 | 2.3186 | 2.5618 | 2.9350 | 1.9986 | 1.9340 | 1.6313 | 1.9659 | 2.8159 | 2.1444 |
| Sarvam-m | sarvamai/sarvam-m | 1.2672 | 1.9405 | 2.9155 | 3.5911 | 3.7030 | 3.8345 | 4.7443 | 3.1445 | 3.6214 | 3.1976 | 18.1782 | 4.7235 | 4.8722 |
| Sarvam-1 | sarvamai/sarvam-1 | 1.4313 | 1.4039 | 2.0467 | 2.1554 | 2.1321 | 2.4001 | 2.8345 | 1.7765 | 1.8024 | 1.6826 | 2.3251 | 4.7451 | 2.3004 |
| c0 (`train/train_bpe.py`) | not published | 1.2596 | 1.6310 | 2.2673 | 2.6597 | 2.5954 | 2.7836 | 3.0869 | 2.1067 | 2.1193 | 2.0053 | 2.6162 | 2.3402 | 2.3829 |
| c0-v0ids (`--special-layout v0`) | not published | 1.2596 | 1.6311 | 2.2675 | 2.6597 | 2.5955 | 2.7836 | 3.0869 | 2.1067 | 2.1194 | 2.0053 | 2.6162 | 2.3405 | 2.3829 |

## Code and math (tokens / char, lower is better)

| Tokenizer | HumanEval | MBPP | GSM8K |
|---|---|---|---|
| **BrahmicTokenizer-131K** | 0.2951 | 0.3195 | 0.3010 |
| o200k_base | 0.2934 | 0.3169 | 0.2995 |
| Gemma-4 | 0.3340 | 0.3876 | 0.3505 |
| Qwen3.8 | 0.3170 | 0.3318 | 0.3525 |
| Qwen3-8B | 0.2980 | 0.3169 | 0.3469 |
| DeepSeek-V4 | 0.2976 | 0.3265 | 0.3015 |
| Sarvam-30B | 0.3340 | 0.3876 | 0.3505 |
| Sarvam-m | 0.3073 | 0.3378 | 0.3508 |
| Sarvam-1 | 0.3780 | 0.4240 | 0.3805 |
| c0 (`train/train_bpe.py`) | 0.2979 | 0.3226 | 0.3038 |
| c0-v0ids (`--special-layout v0`) | 0.2979 | 0.3226 | 0.3038 |

## Cross-checks against the paper's reference outputs

Rows that also appear in `BrahmicTokenizer-131K/reference_outputs/evaluate_fertility_flores.json` match to 4 decimals:
Brahmic, Qwen3-8B, Sarvam-30B, Sarvam-1. Also identical to 4 decimals:

- DeepSeek-V4 = the reference DeepSeek-R1 row: V4 kept the R1 tokenizer.
- Gemma-4 = the reference Gemma-3-1B row: same tokenizer.
- o200k_base Indic columns = the reference GPT-OSS-120B row (GPT-OSS uses o200k).

Sarvam-30B and Gemma-4 tokenize Hindi, Bengali and code identically but differ on Te/Kn/Ml/Gu/Pa/Or, where Sarvam-30B is much better.

## Sarvam-m vs Tekken: structural identity (#130)

`verify_sarvam_m_structural_identity.py` was run on Sarvam-m against Tekken (`mistralai/Mistral-Nemo-Base-2407`), not just on the baseline against itself. Full commands, pinned revisions, hashes and output: `harness/structural_identity_sarvam_m_vs_tekken.md`.

| Check | Result |
|---|---|
| BPE merges (269,443 each) | identical, same rank order |
| Normal vocab strings | 131,066 of 131,072 identical |
| Special-token strings | 994 of 1,000 identical; 6 renamed (`[IMG]`, `[IMG_BREAK]`, `[IMG_END]`, `[SYSTEM_PROMPT]`, `[/SYSTEM_PROMPT]`, `[TOOL_CONTENT]` vs `<SPECIAL_14>` to `<SPECIAL_19>`) |
| Token ids (`harness/structural_identity_ids.py`) | 131,062 same string and same id; ids 10 to 19 differ, so `<pad>`, `[PREFIX]`, `[MIDDLE]`, `[SUFFIX]` moved by 1 to 3 |
| Encode check, 8 sentences (En, Hi, Bn, Ta, Te, Or, code, mixed) | 8 of 8 identical ids |

So Sarvam-m is Tekken's merge table with different special tokens. The difference is more than a rename, because ids 10 to 19 shifted. This is why Sarvam-m's fertility matches Tekken's in the paper. The Sarvam-m row above stands in for Tekken, and the repo has no separate Tekken row yet.

## Reading the table

- Two comparators beat BrahmicTokenizer-131K on Indic already: Sarvam-30B (2.14) and Sarvam-1 (2.30). Both are **> 13% worse on code**, so neither passes the bar.
- **c0 passes the bar, with almost no English margin:**
  - Indic: −15.96%
  - HumanEval: +0.95%
  - MBPP: +0.97%
  - English: **+1.98% (limit 2.00%)**

  The English loss comes from keeping only the first 114,432 o200k ranks. Pruning the base by English and code token frequency instead of by rank is the main Week 2 change.
- c0 is the Week 1 pipeline proof, not a release candidate. It is trained on 20 MB of FineWeb-2 per language, and its slot sizes in `train/slots.json` are not tuned.

## Reproduce c0 (about 2 minutes after the download)

```bash
python train/fetch_corpus.py --out $WORK/corpus --mb 20        # FineWeb-2, never FLORES
python train/train_bpe.py --corpus $WORK/corpus --out $WORK/c0 --brahmic-dir $BRAHMIC
#   3 verify scripts PASS + structural diff vs BrahmicTokenizer-131K reported; sha256 a635bc3d601925a80f7792ba73f7aba6867a688a61bb9031481eeed5578173c1
python harness/run_harness.py --tokenizers BrahmicTokenizer-131K --candidate c0=$WORK/c0/tokenizer.json \
    --suites flores,code --check-reference --brahmic-dir $BRAHMIC --flores-dir $WORK/flores200_dataset
```

The build is deterministic: two runs gave the same sha256. Layout, corpus hashes and verify results are in `train/c0_manifest.json`.
The tokenizer.json (5,674,537 bytes) is not committed because pr-guard rejects files over 5 MiB (5,242,880 bytes).

## c0-v0ids: c0 with tokenizer v0's special ids

`train_bpe.py --special-layout v0 --v0-tokenizer <v0 tokenizer.json>` keeps all 356 v0 special tokens (#107) at their exact ids and strings. This includes the 9 at scattered ids (36, 128512, 128601, 129279, 129411, 129994, 130173, 130174, 130226) and the g10/lang/ctrl slot blocks.

- Normal tokens fill the other 130,716 ids, so the o200k base has 100 fewer tokens than c0. The pass-bar numbers do not change, and no column moves by more than 0.0003.
- The build checks the v0 file against `v0/SHA256SUMS`.
- Its added tokens match v0 entry for entry, and a chat-formatted sample gives the same special-token ids as v0.
- Three verify scripts PASS. The structural diff against v0 reports `added identical: True`.
- sha256 `7e9bb94cb43e2395b9b250d1c64500a6f66da020aba136b9147f0590bfd5839d`. The layout is in `train/c0_v0ids_manifest.json`; 4 v0 special ids fall inside the Gurmukhi block.

## v0 weak-spot check: per-language results (FLORES-200)

Fertility = tokens per word on FLORES-200 dev+devtest (2,009 sentences per language), lower is better. v0 encodes text exactly like BrahmicTokenizer-131K (it only renames reserved slots, see `v0/special_slots.md`), so the v0 column is the BrahmicTokenizer-131K row above. The IndicBERTv2-SS values come from `evaluate_fertility_flores.py --tokenizers all --languages all` in the BrahmicTokenizer-131K repo; it is not in the harness registry.

- **vs Sarvam-m (Tekken)**, the same 131K vocab class: v0 is better on **all 12 languages**.
- **vs Indic-specialised tokenizers** (best of Sarvam-30B, Sarvam-1, IndicBERTv2-SS): v0 is behind on 10 of 11 Indic languages, level on English and ahead on Assamese.
- **v0 + normalizer** (#134, no id changes): small gains on clean text, mean −0.13%.
- **v1-c1** (section below) is shown for progress: it is better than v0 on all 11 Indic languages (mean −17.0%), but still behind the best specialist on most of them.

| Lang | v0 | v0 + normalizer | v1-c1 | Sarvam-m (Tekken) | v0 vs Sarvam-m | Best Indic specialist | v0 gap to specialist |
|---|---|---|---|---|---|---|---|
| En | 1.2352 | 1.2352 | 1.2596 | 1.2672 | −2.5% | 1.2368 (Sarvam-30B) | −0.1% |
| Hi | 1.6429 | 1.6418 | 1.6284 | 1.9405 | −15.3% | 1.2460 (IndicBERTv2-SS) | +31.9% |
| Bn | 2.3295 | 2.3061 | 2.2312 | 2.9155 | −20.1% | 1.6765 (Sarvam-30B) | +39.0% |
| Ta | 3.1166 | 3.1165 | 2.5932 | 3.5911 | −13.2% | 2.1554 (Sarvam-1) | +44.6% |
| Te | 3.0292 | 3.0288 | 2.5562 | 3.7030 | −18.2% | 2.1321 (Sarvam-1) | +42.1% |
| Kn | 3.2734 | 3.2724 | 2.7472 | 3.8345 | −14.6% | 2.4001 (Sarvam-1) | +36.4% |
| Ml | 3.4738 | 3.4672 | 3.0331 | 4.7443 | −26.8% | 2.8345 (Sarvam-1) | +22.6% |
| Mr | 2.5723 | 2.5723 | 2.1067 | 3.1445 | −18.2% | 1.4678 (IndicBERTv2-SS) | +75.2% |
| Gu | 2.2748 | 2.2748 | 2.2239 | 3.6214 | −37.2% | 1.8024 (Sarvam-1) | +26.2% |
| Pa | 2.5955 | 2.5926 | 2.0137 | 3.1976 | −18.8% | 1.6313 (Sarvam-30B) | +59.1% |
| Or | 4.1287 | 4.1245 | 2.4844 | 18.1782 | −77.3% | 1.9659 (Sarvam-30B) | +110.0% |
| As | 2.7535 | 2.7512 | 2.2791 | 4.7235 | −41.7% | 2.8159 (Sarvam-30B) | −2.2% |
| **Mean Brahmic (11)** | **2.8355** | **2.8317** | **2.3543** | **4.8722** | **−41.8%** | 2.1444 (Sarvam-30B) | +32.2% |

**Largest v0 gaps to a specialist:** Odia (+110%), Marathi (+75%), Punjabi (+59%), Tamil (+45%), Telugu (+42%), Bengali (+39%), Kannada (+36%). These are the targets for the v1 ideas: Odia/Punjabi leading-space tokens (idea 1), the normalizer (idea 2, #134) and more Dravidian training text (idea 3, v1-c1).

Notes:

- The specialists reach these numbers at a cost elsewhere. Sarvam-30B and Sarvam-1 are more than 13% worse on code (code table above), and IndicBERTv2-SS is strong only on Devanagari (Hi, Mr) and 2–3× worse on other scripts. None of them meets our pass bar.
- Gemma-4 (not counted as a specialist) also beats v0 on mean Indic (2.7015 vs 2.8355), but it is 13–21% worse on code, so it does not meet the bar either.
- Mean row: "v0 vs Sarvam-m" compares the two means; "gap to specialist" compares v0's mean with the best specialist mean (Sarvam-30B).

## v1-c1: Dravidian rebalance with the v1 normalizer (#128)

| Tokenizer | En | Hi | Bn | Ta | Te | Kn | Ml | Mr | Gu | Pa | Or | As | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BrahmicTokenizer-131K | 1.2352 | 1.6429 | 2.3295 | 3.1166 | 3.0292 | 3.2734 | 3.4738 | 2.5723 | 2.2748 | 2.5955 | 4.1287 | 2.7535 | 2.8355 |
| c0 | 1.2596 | 1.6310 | 2.2673 | 2.6597 | 2.5954 | 2.7836 | 3.0869 | 2.1067 | 2.1193 | 2.0053 | 2.6162 | 2.3402 | 2.3829 |
| **v1-c1** | 1.2596 | 1.6284 | 2.2312 | 2.5932 | 2.5562 | 2.7472 | 3.0331 | 2.1067 | 2.2239 | 2.0137 | 2.4844 | 2.2791 | **2.3543** |

Code is the same as c0: HumanEval 0.2979, MBPP 0.3226, GSM8K 0.3038 tokens / char.

The pass bar against BrahmicTokenizer-131K still holds:

| Metric | v1-c1 vs BrahmicTokenizer-131K |
|---|---|
| Mean Indic | **−16.97%** (beats it) |
| English | +1.98% |
| HumanEval | +0.95% |
| MBPP | +0.97% |

Every Indic language is better than BrahmicTokenizer-131K.

- **Build:**
  - `train_bpe.py --slots train/slots_v1.json --special-layout v0 --normalizer v1`
  - sha256 `7d2cff92053170fdb17b301448964f1770d88df247b2e794256254227f86af44`
  - Manifest: `train/v1_c1_manifest.json`. The three gating verify scripts PASS, and the structural diff against v0 reports `added identical: True`.
- **Changes vs c0:**
  1. The #134 normalizer (`nfc+chillu+nukta-BnPa`) is applied before counting words and written into tokenizer.json. Measured alone, it lowers the mean from 2.3829 to 2.3646. Pa −3.0%, As −2.6% and Bn −1.6% gain the most, because the merges are now learned on normalized text.
  2. Ta, Te, Kn and Ml use 60 MB of FineWeb-2 each, up from 20 MB.
  3. Slots: Ta, Te, Kn and Ml get +512 merges in total, plus Odia +256. Gujarati gives up 512 and Gurmukhi 256. Devanagari and the o200k base are unchanged, so English and code do not move.
- **vs c0:** Ta −2.5%, Te −1.5%, Kn −1.3%, Ml −1.7%, Or −5.0%. The cost is Gu +4.9% (still −2.2% vs BrahmicTokenizer-131K) and Pa +0.4%.
- **Training text:** no FLORES sentence (40 or more characters) appears verbatim in the training corpus. All 11 languages were checked. No IN22, HumanEval, MBPP or GSM8K text is used.
- **Other slot splits tried** (same data and normalizer; FLORES mean Indic):

| Split | Mean Indic | Note |
|---|---|---|
| c0 slots | 2.3590 | |
| v1-c1 | 2.3543 | |
| Greedy per-merge-gain allocation, Devanagari kept at 3,072 | 2.3488 | Dravidian worse than c0, Gu only −0.3% vs v0 |
| Unconstrained greedy allocation | 2.3391 | **rejected**: Hindi +6.3% vs BrahmicTokenizer-131K |

  These five splits were chosen on FLORES, so the gaps between them are small and somewhat tuned to it. The real limit is the fixed 16,384 merge budget: freeing base slots (o200k pruning by usage) is the next lever.
- **Reproduce:**
  1. `fetch_corpus.py --mb 20` (all languages), then `fetch_corpus.py --mb 60 --langs Ta,Te,Kn,Ml` into the same directory.
  2. Run the build above. Corpus hashes are in the manifest.

## v2: o200k pruning by usage, slots chosen on held-out text (#163)

v2 removes o200k tokens that our text never uses from the base, and gives their ids to o200k tokens that English and code use. **FLORES is not used anywhere in v2**: not for choices, not for this report. Every choice was made on held-out **dev** text, and the numbers below include a held-out **check** split that no choice used. The text comes from FineWeb-2 `test` for the 11 Indic languages, FineWeb for English and codeparrot Python for code (`train/fetch_heldout.py`). Fertility is tokens per whitespace word; code is tokens per character. Spec: `train/v2_spec.md`.

**dev**

| Tokenizer | En | Code | Hi | Bn | Ta | Te | Kn | Ml | Mr | Gu | Pa | Or | As | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BrahmicTokenizer-131K | 1.3103 | 0.2362 | 1.5744 | 2.3152 | 3.0593 | 3.0722 | 3.0813 | 3.3974 | 2.4801 | 2.1896 | 2.4937 | 3.9185 | 2.6459 | 2.7480 |
| v1-c1 | 1.3343 | 0.2398 | 1.5602 | 2.1734 | 2.5376 | 2.5961 | 2.6140 | 2.8650 | 1.9963 | 2.1248 | 1.9060 | 2.3350 | 2.1892 | 2.2634 |
| v2 | 1.3087 | 0.2358 | 1.5589 | 2.1736 | 2.5370 | 2.5948 | 2.6131 | 2.8638 | 1.9957 | 2.1240 | 1.9053 | 2.3340 | 2.1883 | 2.2626 |
| v1-c1 vs baseline | +1.83% | +1.50% | -0.90% | -6.12% | -17.05% | -15.50% | -15.17% | -15.67% | -19.51% | -2.96% | -23.57% | -40.41% | -17.26% | -17.63% |
| v2 vs baseline | -0.12% | -0.19% | -0.98% | -6.11% | -17.07% | -15.54% | -15.19% | -15.71% | -19.53% | -3.00% | -23.59% | -40.44% | -17.29% | -17.66% |
| v2 vs v1-c1 | -1.92% | -1.67% | -0.08% | +0.01% | -0.03% | -0.05% | -0.03% | -0.04% | -0.03% | -0.04% | -0.04% | -0.04% | -0.04% | -0.04% |

**check**

| Tokenizer | En | Code | Hi | Bn | Ta | Te | Kn | Ml | Mr | Gu | Pa | Or | As | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BrahmicTokenizer-131K | 1.2907 | 0.2456 | 1.5834 | 2.2785 | 3.0526 | 2.9790 | 3.0159 | 3.3596 | 2.5032 | 2.1481 | 2.4996 | 3.8723 | 2.6430 | 2.7214 |
| v1-c1 | 1.3132 | 0.2491 | 1.5717 | 2.1697 | 2.5266 | 2.5249 | 2.5592 | 2.8572 | 2.0323 | 2.0846 | 1.9057 | 2.2957 | 2.1960 | 2.2476 |
| v2 | 1.2895 | 0.2453 | 1.5704 | 2.1681 | 2.5256 | 2.5237 | 2.5581 | 2.8565 | 2.0313 | 2.0837 | 1.9047 | 2.2953 | 2.1950 | 2.2466 |
| v1-c1 vs baseline | +1.74% | +1.41% | -0.74% | -4.78% | -17.23% | -15.24% | -15.14% | -14.95% | -18.81% | -2.96% | -23.76% | -40.71% | -16.91% | -17.41% |
| v2 vs baseline | -0.10% | -0.13% | -0.82% | -4.85% | -17.27% | -15.28% | -15.18% | -14.97% | -18.85% | -3.00% | -23.80% | -40.73% | -16.95% | -17.45% |
| v2 vs v1-c1 | -1.81% | -1.51% | -0.08% | -0.07% | -0.04% | -0.05% | -0.04% | -0.03% | -0.05% | -0.04% | -0.06% | -0.02% | -0.04% | -0.04% |

- **English:** −1.8% vs v1-c1 on the check split. v2 is now level with or slightly better than BrahmicTokenizer-131K on English (−0.10%) and on code (−0.13%). v1-c1 was +1.74% and +1.41%.
- **Indic and Dravidian not compromised:** every one of the 11 languages is equal or better than v1-c1. On the check split, all 11 are 0.02–0.08% better, because English words inside Indic pages now take fewer tokens. On dev, Bn is +0.01% (2.1736 vs 2.1734); every other language is better. The script blocks are byte-identical to v1-c1 (below).
- **Ids:**
  - All 356 special ids and all 16,739 ids from 114,333 up (every script block) hold exactly the same token as in v1-c1.
  - 100,483 of the 114,333 base ids are unchanged.
  - The 13,850 changed ids held o200k tokens that are never produced on 30 MB of held-out English, 20 MB of held-out code or the whole 380 MB Indic training corpus: Cyrillic 5,546, CJK 3,264, Arabic 2,751, Hebrew 1,029, Thai 679, Greek 517 and Sinhala 64. No Latin, Common or Brahmic token was pruned. `train/check_ids.py` checks all of this and PASSES.
  - **Team decision needed:** the README says "never reuse an id a model was trained with". These 13,850 ids are reused. A model trained on v1-c1 text would almost never have seen their old tokens, but if G4/G5 have trained on v1-c1, the lead should confirm this is acceptable.
- **How the tokens were chosen** (`train/prune_o200k.py`):
  - **Prune list:** v1 base tokens used 0 times when the held-out English and code and the full Indic training corpus are encoded with the base. Multi-byte only, never a token with a Brahmic character, and only tokens that no kept base token is merged from (the base stays closed under its merges). The order is non-Latin scripts first, then Common, then Latin; within each group the rarest o200k rank goes first. 36,587 tokens qualify. v2 uses the first 13,850, all in non-Latin scripts.
  - **New tokens:** o200k tokens past the v1 cut (rank 114,483), ranked separately for English and code by count × (base tokens saved − 1). Counts are capped at 4 per 20 KB chunk, with a minimum of 3 chunks, so that one generated file cannot dominate. The base ancestors of the new tokens are never pruned.
  - **Swap size** (`train/slot_search.py`): the smallest code prefix and the smallest English prefix that reach parity with BrahmicTokenizer-131K on dev were 3,840 code and 10,240 English tokens (13,850 after duplicates).

| New English tokens | Dev English vs baseline |
|---|---|
| 0 (v1-c1) | +1.83% |
| 2,048 | +1.33% |
| 4,096 | +0.90% |
| 6,144 | +0.55% |
| 8,192 | +0.23% |
| **10,240 (v2)** | **−0.07%** |

- **Slot split** (`train/slot_search.py`):
  - Each script was trained once with up to 1,536 extra merges, which gives a dev fertility curve for every slot count.
  - Rule: no language may be worse than with its v1 slot count. Every curve only goes down as slots are added, so with the same 16,384 script ids the only split that meets this rule is v1-c1's split. The held-out search therefore keeps it.
  - The split that optimises only the dev mean (Devanagari 1,536, Bengali 2,944, Oriya 2,432, ...) would make Hi +6.0%, Mr +8.0%, Ml +2.0%, Ta +1.1% and Te +1.1% worse, so it was rejected.
- **Build:**
  - `train_bpe.py --slots train/slots_v2.json --special-layout v0 --v0-tokenizer <v0> --normalizer v1`
  - sha256 `0862f6832229f16bf35acb211307d473b2eea8aa812eb2f3d2c1419191e5bc47`
  - Manifest: `train/v2_manifest.json`. Its `o200k_swap.swapped_ids` lists every changed id.
  - The three gating verify scripts PASS, and `check_ids.py --manifest` against v1-c1 PASSES.
  - Without `o200k_swap`, `train_bpe.py` still rebuilds v1-c1 byte-identically (sha256 `7d2cff92…`).
- **Reproduce:**
  1. Fetch the v1 corpus (see the v1-c1 section).
  2. `fetch_heldout.py --out <heldout> --train-corpus <corpus>`. Hashes: `train/heldout_manifest.json`.
  3. `prune_o200k.py --heldout <heldout> --corpus <corpus> --out train/o200k_v2.json`
  4. `slot_search.py --corpus <corpus> --heldout <heldout> --brahmic <BrahmicTokenizer-131K/tokenizer.json> --v1 <v1-c1 tokenizer.json> --work <dir>` (about 7 minutes). This writes `slots_v2.json` and the chosen `tokens`.
  5. Run the build above.

## Week 2 candidates C1, C2, C3 (v0.1 specials)

Each candidate adds one change to the one before it. All are trained on the v1 corpus (`corpus60`; hashes in each manifest) and use the v0.1 special tokens (#173): G10's ChatML markers at ids 130822-130826, every id unchanged. Numbers are from the held-out **check** split (FineWeb-2 `test`, FineWeb, codeparrot; see the v2 section). FLORES is not used.

| Candidate | Change | Build | sha256 | Manifest |
|---|---|---|---|---|
| C0 | reference: c0 slots, v0 specials | `--slots slots.json` | `53c1047f…` | not committed (reference only) |
| **C1** | + o200k swap only | `--slots slots_c1.json` | `5d386e2e…` | `train/c1_manifest.json` |
| **C2** | + slots_v1 split | `--slots slots_v2.json` | `84204d78…` | `train/c2_manifest.json` |
| **C3** | + normalizer v1 (= v2) | `--slots slots_v2.json --normalizer v1` | `54f20825…` | `train/c3_manifest.json` |

All builds use `train_bpe.py --special-layout v0 --v0-tokenizer <v0.1 tokenizer.json>` (C0: the v0 file).

| Tokenizer | En | Code | Hi | Bn | Ta | Te | Kn | Ml | Mr | Gu | Pa | Or | As | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BrahmicTokenizer-131K | 1.2907 | 0.2456 | 1.5834 | 2.2785 | 3.0526 | 2.9790 | 3.0159 | 3.3596 | 2.5032 | 2.1481 | 2.4996 | 3.8723 | 2.6430 | 2.7214 |
| C0 (c0, v0 ids) | 1.3132 | 0.2491 | 1.5717 | 2.1760 | 2.5898 | 2.5521 | 2.5878 | 2.8572 | 2.0323 | 1.9893 | 1.8324 | 2.4021 | 2.1996 | 2.2536 |
| C1 (+o200k swap) | 1.2895 | 0.2453 | 1.5704 | 2.1744 | 2.5888 | 2.5508 | 2.5867 | 2.8565 | 2.0313 | 1.9884 | 1.8313 | 2.4017 | 2.1987 | 2.2526 |
| C2 (+slots_v1) | 1.2895 | 0.2453 | 1.5704 | 2.1744 | 2.5256 | 2.5237 | 2.5581 | 2.8565 | 2.0313 | 2.0837 | 1.9036 | 2.2953 | 2.1987 | 2.2474 |
| C3 (+normalizer v1 = v2) | 1.2895 | 0.2453 | 1.5704 | 2.1681 | 2.5256 | 2.5237 | 2.5581 | 2.8565 | 2.0313 | 2.0837 | 1.9047 | 2.2953 | 2.1950 | 2.2466 |
| C0 vs baseline | +1.74% | +1.41% | -0.74% | -4.50% | -15.16% | -14.33% | -14.20% | -14.95% | -18.81% | -7.39% | -26.69% | -37.97% | -16.77% | -17.19% |
| C1 vs baseline | -0.10% | -0.13% | -0.82% | -4.57% | -15.20% | -14.37% | -14.23% | -14.97% | -18.85% | -7.43% | -26.74% | -37.98% | -16.81% | -17.22% |
| C2 vs baseline | -0.10% | -0.13% | -0.82% | -4.57% | -17.27% | -15.28% | -15.18% | -14.97% | -18.85% | -3.00% | -23.84% | -40.73% | -16.81% | -17.42% |
| C3 vs baseline | -0.10% | -0.13% | -0.82% | -4.85% | -17.27% | -15.28% | -15.18% | -14.97% | -18.85% | -3.00% | -23.80% | -40.73% | -16.95% | -17.45% |
| C1 vs C0 | -1.81% | -1.51% | -0.08% | -0.07% | -0.04% | -0.05% | -0.04% | -0.03% | -0.05% | -0.04% | -0.06% | -0.02% | -0.04% | -0.04% |
| C2 vs C1 | +0.00% | +0.00% | +0.00% | +0.00% | -2.44% | -1.06% | -1.11% | +0.00% | +0.00% | +4.79% | +3.95% | -4.43% | +0.00% | -0.23% |
| C3 vs C2 | +0.00% | +0.00% | +0.00% | -0.29% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.00% | +0.06% | +0.00% | -0.17% | -0.04% |

- **C1, the swap,** gives the whole English and code gain: −1.81% English and −1.51% code vs C0. That takes both from about +1.6% above the baseline to level with it. Every Indic language also gets 0.02–0.08% better.
- **C2, the v1 slot split,** moves merges between scripts: Ta −2.44%, Te −1.06%, Kn −1.11% and Or −4.43%, paid for by Gu +4.79% and Pa +3.95%. Mean Indic is −0.23%. English and code don't change.
- **C3, the normalizer,** gives Bn −0.29% and As −0.17% (nukta re-join). Pa is +0.06%. C3 is v2 with the v0.1 specials: its merges, vocab ids and normalizer are identical to v2, and only the 5 renamed special strings differ.
- **Ids:** `check_ids.py` passes for C1 vs C0, C2 vs v1-c1 and C3 vs v2. It now accepts a reserved slot that is filled by renaming it with its id and flags kept (README rule) and fails any other special change. The three verify scripts pass for each candidate.
- **C0 sha:** C0's sha differs from `c0_v0ids_manifest.json` (`7e9bb94c…`). That build used the older 20 MB Ta/Te/Kn/Ml corpus; the manifest hashes show the difference. The same slots on `corpus60` give `53c1047f…`, with both the current and the #137 `train_bpe.py`.
