# FLORES-200, Task 5: findings (for Team 9's Indic evaluation)

Numbers: `flores_task5.md` / `.json`, made by `harness/flores_task5.py` (FLORES-200 devtest, 1,012 sentences per language).
Word rule for every language: maximal runs of non-whitespace (`str.split()`). FLORES is used only to report on the frozen
p1536 (`e6b93a93…`), never to choose it.

## Headline

| | Indic mean tokens/word (11 languages) |
|---|---:|
| **p1536 (frozen)** | **2.139** |
| Sarvam-1 (68,096 ids) | 2.308 |
| Published Indic tokenizers (brief) | about 2.2 to 2.35 |
| BrahmicTokenizer-131K (our base) | 2.844 |
| o200k_base (200,019 ids) | 3.111 |
| Sarvam-M / Krutrim-2 (Mistral Tekken, 131,072 ids) | 4.885 |
| Qwen2.5 (151,665 ids) | 9.393 |

- **p1536 beats the published 2.2–2.35** and the best lossless published Indic tokenizer here, Sarvam-1, on the 11-language mean (−7.3%).
- **All 20 languages are lossless** with p1536 (decode(encode(x)) == normalizer(x) on every sentence).
- **WordPiece encoders are lower but not comparable.** IndicBERTv2 (1.584) and MuRIL (1.674) are encoder vocabularies that lose text: only 0–75% of sentences round trip, because they split on spaces and punctuation and use [UNK].

## Where p1536 loses, and why

| Languages | Loses to | By | Why |
|---|---|---:|---|
| Hi, Mr, Ta, Te, Gu | Sarvam-1 | 8–10% fewer tokens | 73% of Sarvam-1's ids (49,707 of 68,096) are Indic; p1536 has 27,136 Indic ids (20.7%), because 103,580 ids keep o200k for English and code. Sarvam-1 pays for that outside Indic: English 1.435 vs 1.237 tokens/word (+16%), French +68%, Russian 13.2 vs 3.1. |
| Kn, Pa, Sa | Sarvam-1 | 2–5% fewer | same cause, a smaller gap |
| En | o200k_base | 0.8% fewer | the o200k ids p1536 pruned or gave to Indic merges (#414) |
| Fr, De, Es | o200k_base, Tekken | 9–12% fewer | the pruning kept Latin tokens by **English** use (`usage/En.txt`), so French, German and Spanish word pieces went first (8,501 Latin ids in the #414 swap) |
| Ru | o200k_base | 38% fewer | Cyrillic ids pruned (470 in the swap) and the base cut at o200k rank 114,483. p1536 is still 4× better than the base BrahmicTokenizer (12.48 tokens/word). |
| Zh, Ja, Ko | Qwen2.5, o200k_base, Tekken | 18–43% fewer | CJK ids pruned (309 in the swap), and the base drops cross-script merges |

Bn, As, Ml and Or: no lossless tokenizer here needs fewer tokens than p1536.

## Not done here

- **"Your own unit list".** The brief asks for these numbers with G3's own unit list as well. That list is the SuperBPE-style unit list of the design tasks (1–4) and is not built yet. The same script takes any `tokenizer.json`, so it can be added when the list exists.
- **Sanskrit** (san_Deva) is in the tables as a reference. It costs 3.06 tokens per word on FLORES (Sanskrit is not in p1536's training text; see `stage1_coverage.md`).
