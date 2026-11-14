# Attainability survey, code arm at a 2048-token cap: stand-in diagnostics, not the capstone model

12 Nov 2026 (IST). One question: if the generation cap goes from 1024 to 2048 tokens, does the code arm finish more often, and do the finish rate and correctness go up? Math was not re-run.

The stand-in is `Qwen/Qwen3.6-35B-A3B-FP8`. It is not the capstone model, and none of these numbers is a claim about the capstone model, Gate 4 or the RL-v1 pool as a whole. They describe these 32 prompts and this stand-in under this protocol.

## What changed and what did not

Exactly one parameter differs between the two runs: `max_tokens` 1024 → 2048. The comparison script checks this from the run records before it prints anything (`params that differ: ['max_tokens']`).

| | 1024 run (attainability survey) | 2048 run (this file) |
|---|---|---|
| Prompts | the 32 code prompts of the survey: aizu 8, atcoder 8, codechef 8, hackerearth 8 (v1 `cc.jsonl`, split seed 20261111) | same 32 ids, same prompt token counts, same rendered prompt |
| Model | `Qwen/Qwen3.6-35B-A3B-FP8` @ c653bba46dfa06e9658dcc0b0dc87fe4685f14a2, official FP8 | same |
| Engine | vLLM 0.32.0, `llm.chat` with the model's own template, `enable_thinking=False`, max_model_len 8192, `VLLM_USE_DEEP_GEMM=0`, 1× H100 80GB HBM3 (Modal, `H100!`) | same |
| Sampling | n = 8, temperature 1.0, top_p 1.0, seed 1234, **max_tokens 1024** | same, **max_tokens 2048** |
| System prompt | "You are an expert competitive programmer. Respond with exactly one ```python code block and nothing else. The program must read from stdin and write to stdout." | same |
| Scoring | `stdio_check.py` (unchanged from the `rl-data` tree of PR saitej123/era-v5-lab#54 @ 957534c) in Docker 29.9.2: a fresh `python:3.12-slim` container per program, `--network=none`; a sample passes only if every test passes | same scorer, same Docker |

Because the seed is the same, the first 1024 tokens mostly repeat: in 254 of 256 samples the 1024 text is an exact prefix of the 2048 text. So for almost every sample the 2048 run is the same sample continued, and the comparison below is close to sample-by-sample. The two samples that diverged are counted in the totals and named under "Truncation".

pass@8 = the prompt has at least one passing sample out of 8 (a direct count with n = 8). A parser failure is a sample with no usable ```python block.

## Per judge stratum

| Stratum | pass@8 1024 | pass@8 2048 | samples passed 1024 | 2048 | parser failures 1024 | 2048 | hit the cap 1024 | 2048 | pass rate of finished samples 1024 | 2048 |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| aizu | 4/8 | 5/8 | 26/64 | 30/64 | 36 | 27 | 36 | 27 | 26/28 | 30/37 |
| atcoder | 2/8 | 2/8 | 15/64 | 15/64 | 38 | 37 | 38 | 37 | 15/26 | 15/27 |
| codechef | 5/8 | 6/8 | 31/64 | 33/64 | 17 | 10 | 17 | 10 | 31/47 | 33/54 |
| hackerearth | 6/8 | 7/8 | 32/64 | 38/64 | 20 | 13 | 20 | 13 | 32/44 | 38/51 |
| **All 32** | **17/32** | **20/32** | **104/256** | **116/256** | **111** | **87** | **111** | **87** | **104/145** | **116/169** |

Bands (0/8 all-fail, 1..7/8 mixed, 8/8 all-pass), 1024 → 2048: all-fail 15 → 12, mixed 11 → 14, all-pass 6 → 6. Three prompts moved from all-fail to mixed (`cc-aizu-p01055_Bomb_Removal` 0 → 1, `cc-codechef-chefseg` 0 → 1, `cc-hackerearth-e_musical_sequences` 0 → 5). No prompt lost a passing sample.

## Truncation and parser failures

- **Truncated at 1024: 111 of 256 code samples.** At 2048, **24 of those 111 finish** (23 paired samples, plus `cc-codechef-gerald04` sample 5, which diverged and finished at 1608 tokens). **12 of the 24 pass**; the other 12 give a wrong answer, a runtime error or a timeout on a test.
- **87 samples still hit the cap at 2048.** All 87 are parser failures; there is no parser failure on a sample that finished, at either cap. 73 of the 87 had opened a ```python block and were still inside it. 14 began with prose despite the instruction; these are the same 14 prose-first samples that were cut at 1024, and none of them finished at 2048. A simple repeated-line check on the last 60 lines found no repetition loops among the 87.
- **Parser failures: 111 → 87** (43.4% → 34.0% of code samples).
- The 24 samples that now finish used 1037 to 2021 tokens. Mean completion length rose from 635 to 1035 tokens; the median stayed at 584, because samples that finished under 1024 are unchanged.
- The pass rate of finished samples fell slightly, 104/145 (72%) → 116/169 (69%), because the newly finished long samples pass at 12/24 (50%).
- The second diverged sample, `cc-hackerearth-mozart_numbers` sample 1, hit the cap in both runs.

## Per prompt

k = samples passed out of 8; pf = parser failures; cap = samples that hit the cap; the last column counts samples cut at 1024 whose paired 2048 continuation finished.

| Stratum | Prompt id | pass@8 1024 | pass@8 2048 | k 1024 | k 2048 | pf 1024 | pf 2048 | cap 1024 | cap 2048 | cut at 1024, finish at 2048 |
|---|---|:-:|:-:|--:|--:|--:|--:|--:|--:|--:|
| aizu | `cc-aizu-p00916_Count_the_Regions` | 0 | 0 | 0 | 0 | 8 | 8 | 8 | 8 | 0 of 8 |
| aizu | `cc-aizu-p01055_Bomb_Removal` | 0 | 1 | 0 | 1 | 8 | 7 | 8 | 7 | 1 of 8 |
| aizu | `cc-aizu-p02343_Disjoint_Set__Union_Find_Tree` | 1 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 of 0 |
| aizu | `cc-aizu-p01563_Connect` | 0 | 0 | 0 | 0 | 8 | 5 | 8 | 5 | 3 of 8 |
| aizu | `cc-aizu-p00633_Crop_Circle` | 0 | 0 | 0 | 0 | 8 | 7 | 8 | 7 | 1 of 8 |
| aizu | `cc-aizu-p00663_SAT_EN_3` | 1 | 1 | 3 | 6 | 4 | 0 | 4 | 0 | 4 of 4 |
| aizu | `cc-aizu-p00789_Square_Coins` | 1 | 1 | 7 | 7 | 0 | 0 | 0 | 0 | 0 of 0 |
| aizu | `cc-aizu-p02248_String_Search` | 1 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 of 0 |
| atcoder | `cc-atcoder-p03752_square869120Contest__4___Buildings_are_Colorful_` | 0 | 0 | 0 | 0 | 6 | 5 | 6 | 5 | 1 of 6 |
| atcoder | `cc-atcoder-p03267_AtCoder_Beginner_Contest_108___All_Your_Paths_are_Dif` | 0 | 0 | 0 | 0 | 8 | 8 | 8 | 8 | 0 of 8 |
| atcoder | `cc-atcoder-p03238_AtCoder_Beginner_Contest_112___Programming_Education` | 1 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 of 0 |
| atcoder | `cc-atcoder-p03655_AtCoder_Grand_Contest_018___Sightseeing_Plan` | 0 | 0 | 0 | 0 | 8 | 8 | 8 | 8 | 0 of 8 |
| atcoder | `cc-atcoder-p03586_Tenka1_Programmer_Contest___ModularPowerEquation__` | 0 | 0 | 0 | 0 | 8 | 8 | 8 | 8 | 0 of 8 |
| atcoder | `cc-atcoder-p02699_AtCoder_Beginner_Contest_164___Sheep_and_Wolves` | 1 | 1 | 7 | 7 | 0 | 0 | 0 | 0 | 0 of 0 |
| atcoder | `cc-atcoder-p03669_AtCoder_Grand_Contest_017___Jigsaw` | 0 | 0 | 0 | 0 | 8 | 8 | 8 | 8 | 0 of 8 |
| atcoder | `cc-atcoder-p03608_AtCoder_Beginner_Contest_073___joisino_s_travel` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 of 0 |
| codechef | `cc-codechef-cielnum2` | 1 | 1 | 6 | 6 | 0 | 0 | 0 | 0 | 0 of 0 |
| codechef | `cc-codechef-chefseg` | 0 | 1 | 0 | 1 | 8 | 5 | 8 | 5 | 3 of 8 |
| codechef | `cc-codechef-gerald04` | 0 | 0 | 0 | 0 | 8 | 5 | 8 | 5 | 2 of 8 (+1 diverged) |
| codechef | `cc-codechef-fact25` | 1 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 of 0 |
| codechef | `cc-codechef-chfmax` | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 of 0 |
| codechef | `cc-codechef-ism05` | 1 | 1 | 4 | 4 | 0 | 0 | 0 | 0 | 0 of 0 |
| codechef | `cc-codechef-kingship` | 1 | 1 | 6 | 7 | 1 | 0 | 1 | 0 | 1 of 1 |
| codechef | `cc-codechef-skyscr` | 1 | 1 | 7 | 7 | 0 | 0 | 0 | 0 | 0 of 0 |
| hackerearth | `cc-hackerearth-hanuman_language` | 1 | 1 | 5 | 5 | 0 | 0 | 0 | 0 | 0 of 0 |
| hackerearth | `cc-hackerearth-skits_patties_3` | 1 | 1 | 4 | 5 | 4 | 3 | 4 | 3 | 1 of 4 |
| hackerearth | `cc-hackerearth-mozart_numbers` | 0 | 0 | 0 | 0 | 8 | 8 | 8 | 8 | 0 of 8 |
| hackerearth | `cc-hackerearth-atul_and_multiplication_2` | 1 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 of 0 |
| hackerearth | `cc-hackerearth-e_musical_sequences` | 0 | 1 | 0 | 5 | 8 | 2 | 8 | 2 | 6 of 8 |
| hackerearth | `cc-hackerearth-minimizing_path_cost` | 1 | 1 | 8 | 8 | 0 | 0 | 0 | 0 | 0 of 0 |
| hackerearth | `cc-hackerearth-thanks_giving` | 1 | 1 | 2 | 2 | 0 | 0 | 0 | 0 | 0 of 0 |
| hackerearth | `cc-hackerearth-shil_and_palindrome` | 1 | 1 | 5 | 5 | 0 | 0 | 0 | 0 | 0 of 0 |

## Answer

Doubling the cap helps the code arm, but not by much. 24 of the 111 cut-off samples finish and half of those pass, so pass@8 goes from 17/32 to 20/32 and sample pass from 104/256 to 116/256. Every stratum except atcoder gains one prompt at pass@8. Most of the truncation stays: 87 samples still run past 2048 tokens. 6 prompts (1 aizu, 4 atcoder, 1 hackerearth) have all 8 samples at the cap at both 1024 and 2048. For those prompts, the stand-in in non-thinking mode writes programs, or prose, longer than 2048 tokens.

## Caveat

2048 is not 1024. The numbers above describe this stand-in at a 2048-token cap. The team's evaluation corpora run at a different cap on this exact axis: the effort suite's stages A and B use a common cap of 1024 (`eval/effort_suite.py`, `STAGE_CAP`, per G10_TRAINING.md section 4), and the RL loop's generation cap is set by the caller. So a pass rate or band measured here at 2048 does not carry over to a rollout or evaluation capped at 1024, and pass-rate filtering done at one cap does not select the same tasks as filtering at the other. Eight prompts per stratum also means that one prompt moves a stratum by 12.5 points.

## Records

Generation and scoring ran from the private survey folder (raw completions are not committed). sha256 of the files this readout is computed from:

| File | sha256 |
|---|---|
| `gen_code_qwen.json` (1024 completions) | 3261eaf4e6a6e2f11bafacc91e5e8956befb34cab7d17f3eaec0144c29fe291d |
| `gen_code2048_qwen.json` (2048 completions) | af5ff97efcf31eec71c7dad437737f79f44943559c0d4b6791d4be11ec372837 |
| `scored_code_qwen.json` | 5c952920a81af9dd635d6d1239784352013516a972588f4b4e4a1c043b8f95a6 |
| `scored_code2048_qwen.json` | 7e7fdd577cd08005b9d7cfd8d5b0c3c7b7790ad1e03ac15392f827d789f7a3c8 |
| `survey_prompts.jsonl` | 1c6735dc73c9efca1870ba97c38dcab78ddc0001e21f8352bf84205e3bacced4 |
| `gen_modal.py` (unchanged since the 1024 run) | 5fafc084c234a790357aa1e81c72f4bf8b95f08e2185adbc12852840ed56a3dc |
| `score.py` | dae4662ada1f9033d6b2cc9092b21c99dffbb7dcc9dd2a5139d13e532ae552e1 |
| `compare.py` | 3c4402b815f22f1571f11c61accb78d76198e67bce5a0614d2edca0de12686d5 |

Commands (from the survey folder):

```
modal run gen_modal.py --phase code2048 --model qwen --kind stdio --n 8 --max-tokens 2048
python -I score.py logs/gen_code2048_qwen.json --out logs/scored_code2048_qwen.json
python3 -I compare.py . logs/compare_code2048.json
```

Cost (Modal billing): the successful 2048 run cost $0.3315 (load 207.9 s, generate 47.0 s on the H100). Two earlier launches waited for an H100 that never came up, never got a GPU container, and were stopped ($0.0043 and $0.0038, CPU only). The ceiling for this run was $3.
