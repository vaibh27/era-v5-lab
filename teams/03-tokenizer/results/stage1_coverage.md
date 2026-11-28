# Does p1536's training sample represent the Stage 1 mix? (G3, 28 Nov, Task C)

**No.** The p1536 training sample covers the 11 Indic languages well, but not the rest of the Stage 1 mix. On the
non-Indic sources p1536 is o200k_base with fewer ids, so there it is slightly worse than o200k. Numbers:
`fertility_domains.md` (heldout part of every Stage 1 pack, tokens per word and bytes per token vs o200k_base).

## What p1536 was trained on, and what Stage 1 trains on

| | p1536 training text (`train/p1536_s0_manifest.json`, `train/o200k_v2.json`) | Stage 1 (38 sample-v0 packs, TSAI 22 Nov; Data B `mix/indic_share_v1.md`) |
|---|---|---|
| Indic | 11 languages: corpus60 (FineWeb-2), plus corpus_s0 (sample v0) for Hi, Mr, Gu, Or. Merges are learned only here. | Native Indic is a fixed **8%** of every batch, across 10 languages plus Sanskrit (`ind-sa-fw2`). |
| Sanskrit | **none** | about 0.95% of every batch (Data B v1) |
| English | o200k base tokens, kept or pruned by use on 30 MB of FineWeb sample-10BT (`usage/En.txt`). No merges learned. | en-dclm, en-fineweb-edu, en-nemotroncc, en-nemotroncc-synth |
| Code | the same, on 20 MB of Python (`usage/Code.txt`) | Python, C++, Go, Java, JS/TS, Rust, notebooks, commits, Stack Exchange |
| Math, science PDFs | **none** | math-finemath, math-megamath, sci-finepdfs |
| Vocabulary | 103,580 ids o200k base, 27,136 Indic (20.7%), 356 special | |

## What it costs

From `fertility_domains.md` (o200k_base is the outside tokenizer with a bigger vocabulary, 200,019 ids):

| Source group | p1536 tokens vs o200k | p1536 tokens per word |
|---|---:|---:|
| Indic, 10 languages of the sample (packs) | −5% (Hindi) to −70% (Odia) | 1.48 (Hindi) to 2.66 (Kannada, translated) |
| Sanskrit (`ind-sa-fw2`) | −7% | **3.45 to 3.82**, the worst group |
| English (4 packs) | +0.9% to +1.0% | 1.27 to 1.31 |
| Code (9 packs) | +1.8% to +4.5% | 1.87 (Stack Exchange) to 3.25 (C++) |
| Math, science PDFs | +1.4% to +4.8% | 1.47 to 1.74 |

**Findings:**

1. **Non-Indic text costs 1–5% more tokens than o200k.** These sources are about 92% of every Stage 1 batch. The loss comes from the 13,824 o200k ids that p1536 gave to Indic merges (#414, chosen on English and Python use) and the o200k tokens it never kept. Math and C++ lose most (+4.8%, +4.2%): the pruning was judged only on FineWeb English and Python.
2. **Sanskrit is unseen by the tokenizer.** It costs 3.8 tokens per word, against 1.5 for Hindi in the same script. Two causes, from the most-split words in `fertility_domains.md`:
   - **Vedic accent marks never merge** with their letters. U+0951 (udatta) and U+0952 (anudatta) are each a token of their own and break every accented word apart. The Vedic Extensions marks (U+1CD5, U+1CD6, …) fall back to 3 byte tokens each.
   - **Tirhuta script** (U+11480 block) in the Sanskrit pack is outside every script block, so every character falls back to 4 byte tokens.
3. **Indic share of the ids: 20.7%, against 8% of tokens.** That is by design (the Indic goal). The cost to English and code is the 1–5% in finding 1.
4. **Every document is lossless** under both tokenizers: decode(encode(x)) == normalizer(x) on all 38 packs.

## What I would change if a re-freeze were allowed (TSAI's call; p1536 stays frozen)

- **Sanskrit:** learn merges on Sanskrit text in the Devanagari block, and add the Vedic accents (U+0951–U+0954, U+1CD0–U+1CFF) as single tokens.
- **Code and math:** judge the o200k pruning on all 9 code packs and the 2 math packs, not on Python alone.

Reproduce: `python teams/03-tokenizer/harness/fertility_domains.py --tokenizer release/tokenizer.json --sample <sample v0> --heldout <heldout> --out teams/03-tokenizer/results/fertility_domains`
