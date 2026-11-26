# Deletion Tuning on a 7.65B MoE: Phi-mini-MoE-instruct, SFT on GSM8K (22 Nov 2026, updated 25 Nov)

**Evidence level: independent study on a public model, one seed, small evaluation.** This is the "try Deletion Tuning on a
large model" item from the class discussion. It does not use the capstone model, the lab data, or any lab checkpoint, and
it is not Gate 4 evidence for the capstone. The Gate 4 comparison on the dry run 2 checkpoint stays with
`capstone/dryrun2_dt_compare.py` and `results/dt_vs_fft.md`. LoRA and full fine-tuning on the same data and steps were
added on 25 Nov (section "LoRA and full fine-tuning on the same data"); one seed each, learning rates not tuned.

## What was done

| Item | Value |
|---|---|
| Model | Phi-mini-MoE-instruct (MIT licence), 7.65B parameters, 32 layers x 16 experts, bf16 (15.3 GB), split in two blocks of layers over two GPUs |
| Method | the **official** Deletion Tuning code (commit `94f0340`, vendored unmodified, hashes re-checked at every run), paper values: 400 steps, batch 4, lr 5e-4, sharpness 100 -> 1000, weight_reg 1e-5 decaying to 0, init sparsity 1/2048, Wanda prior, seed 0 |
| Targets | the 512 expert gate projections (`w1`), 2,013,265,920 weights, one mask logit each |
| Data | 2,048 GSM8K training questions in the official `{question}\n{answer}` SFT format, loss on the answer only |
| Hardware | 2 x NVIDIA RTX 6000 Ada (47 and 48 GiB) |
| Code | `RaviNaik/ERA`, folder `v5/capstone/dt_phi_moe/`, commit `8bef04a`. Our files only wrap the official ones for two GPUs and for this model's remote code; the README lists every difference from the official scripts |

One server fault matters for reproduction: a direct GPU-to-GPU copy of a large tensor returns garbage on this machine
(`p2p1.txt`, `p2p2.txt`: 200 of 200 whole-tensor copies wrong at 1 Mi elements and above). Gates and labels moved between the two cards
arrived as zeros or random numbers. Every cross-GPU copy therefore goes through the CPU (`devcopy.py`). The training was
run twice, before and after this fix, and deleted the same number of weights (12,296,351). The evaluation below is of the
second run (`..._v4`), made after the fix.

## Training result

| | |
|---|---|
| Deleted at step 0 (Wanda prior) | 0.1837% (3,698,655 weights) |
| Deleted at step 400 | **0.6108%** (12,296,351 of 2,013,265,920) |
| Saved mask | 23.2 MB (official packbits + zstd) |
| Time | 7,035 s training (12.8 s per step), 7,135 s with loading and calibration |
| Peak GPU memory | 35.2 GiB (card 0), 31.0 GiB (card 1) |

## Evaluation result

200 GSM8K test questions (greedy, 256 new tokens), the six prior-task benchmarks with lm-evaluation-harness (zero-shot,
chat template, bf16, **50 items per task**, 50 per subject for MMLU), forward KL on 50 of the base model's own replies.
"Training format" is the question followed by a newline (what the SFT optimises); "chat template" is how the instruct
model is used, with the instruction to end with `#### <number>`.

| | base | Deletion Tuning | change |
|---|---:|---:|---:|
| GSM8K, chat template | 58.5% (117/200) | 71.5% (143/200) | **+13.0** |
| GSM8K, training format | 33.5% (67/200) | 78.5% (157/200) | +45.0 |
| replies that contain `####`, training format | 23.5% | 99.5% | +76.0 |
| hellaswag | 70.00 | 70.00 | 0.00 |
| truthfulqa_mc2 | 56.96 | 56.26 | -0.70 |
| mmlu | 69.54 | 69.75 | +0.21 |
| ifeval | 67.11 | 63.16 | -3.95 |
| winogrande | 78.00 | 80.00 | +2.00 |
| humaneval_instruct | 14.00 | 12.00 | -2.00 |
| **mean of the six prior tasks** | **59.27** | **58.53** | **-0.74** |
| Team 10 `indic_v1` exact (44 rows) | 52.27 | 52.27 | 0.00 |
| Team 10 `safety_v1` exact (41 rows) | 56.10 | 56.10 | 0.00 |
| forward KL per token (GSM8K replies) | 0 | 0.1412 | |

**HumanEval caveat (added 25 Nov): the `humaneval_instruct` numbers in this file are not meaningful, for any arm.** The
tokenizer drops the first space when the generated text is decoded on its own, so every model's code body starts with 3
spaces instead of 4, and any multi-line body then fails with an indentation error (checked on the real tokenizer: a
continuation that should start with 4 spaces decodes with 3). On the first 20 problems all 20 replies of the base model
and all 20 of the full fine-tuned model start with 3 spaces. Scored as the evaluation scores them, base passes 3 and full
fine-tuning 0; with only the first line's indentation set to 4 spaces, base passes 17 and full fine-tuning 14 (20
problems, not a significant difference). The other five prior tasks do not depend on leading whitespace. A corrected run
on all 164 problems for the four arms, with a decode that keeps the first space (`decode_keep_leading_space` in
`dtphi/eval_prior.py`), is pending; until then read the mean of the five other tasks.

## How to read it

- **The chat-template gain is the one to quote.** With 200 questions the standard error of one accuracy is about 3
  points, so +13 is about 2.8 standard errors of the difference (unpaired, so conservative).
- **The training-format +45 is mostly format.** The base model puts `####` in 23.5% of its replies, the tuned model in
  99.5%. The scorer falls back to the last number in a reply that has no `####`, so a reply in another format is scored
  on whatever number comes last. This column mixes learning the format with solving the problem.
- **No forgetting was detected, which is not the same as none.** At 50 items one item is 2 points: the humaneval -2 is
  one problem (that row is unreliable, see the HumanEval caveat above), the winogrande +2 one item, and the ifeval -3.95 about three prompts. All six changes are inside that
  noise, and so is the mean of -0.74.
- **The Team 10 sets did not move in total** (52.27 and 56.10 both before and after). Inside the safety set the
  refusal rate went from 57.1% to 52.4%, on a handful of rows. Non-English safety rows are at 0.0 for both models, so
  that is a property of the base model, not of Deletion Tuning.
- **0.61% of the gate weights is a small deletion** that was enough to change the output format and add 13 points. It is
  not shown that the deletion, rather than any 400-step update of these weights, is what helps. That needs the LoRA and
  full fine-tuning arms on the same data.

## LoRA and full fine-tuning on the same data (25 Nov)

The same data (2,048 GSM8K questions), 400 steps, batch 4, seed 0 and format as the Deletion Tuning run. **The learning
rates are common defaults and were not tuned** (LoRA 2e-4, full fine-tuning 1e-5); Deletion Tuning used the paper values.

| | Deletion Tuning | LoRA | Full fine-tuning |
|---|---|---|---|
| What changes | 512 gate-projection masks, 12,296,351 weights deleted (0.61%) | adapters (r 16, alpha 32) on the same 512 `w1` gate projections: 41,418,752 parameters (0.54%) | all 7,647,632,704 weights (bf16 with stochastic rounding) |
| Learning rate | 5e-4 (paper) | 2e-4 | 1e-5 |
| Training time | 7,035 s (12.8 s per step) | 4,466 s (11.16 s per step) | 7,045 s (17.61 s per step) |
| Peak GPU memory | 35.2 / 31.0 GiB (two cards) | 16.5 GiB (one card) | 32.7 / 32.8 GiB (two cards) |

**Larger evaluation (the one to quote).** All 1,319 GSM8K test questions in the chat template (greedy, 256 new tokens),
four log-likelihood prior tasks at 200 items each, the same questions for every arm.

| | base | Deletion Tuning | LoRA | Full fine-tuning |
|---|---:|---:|---:|---:|
| GSM8K, chat template (%) | 55.95 (738/1319) | 67.10 (885) | **74.98** (989) | 54.36 (717) |
| change against base | | +11.14 | +19.03 | -1.59 |
| hellaswag | 63.00 | 62.50 | 62.00 | 63.00 |
| mmlu | 70.48 | 70.21 | 69.99 | 68.74 |
| truthfulqa_mc2 | 52.21 | 52.63 | 52.26 | 50.46 |
| winogrande | 71.50 | 72.50 | 74.00 | 72.50 |
| **mean of the four** | **64.30** | **64.46** | **64.56** | **63.68** |

Paired comparison with the base model on the same 1,319 questions (bootstrap 10,000 resamples, seed 0; exact McNemar):

| variant | change (points) | 95% interval | base-only / variant-only | McNemar p |
|---|---:|---|---:|---:|
| Deletion Tuning | +11.1 | [+8.8, +13.5] | 62 / 209 | 8.5e-20 |
| LoRA | +19.0 | [+16.5, +21.6] | 49 / 300 | 3.9e-45 |
| Full fine-tuning | -1.6 | [-4.6, +1.5] | 226 / 205 | 0.34 |

Paired comparison **between the variants** (same 1,319 questions, same test; second minus first):

| pair | change (points) | 95% interval | first-only / second-only | McNemar p |
|---|---:|---|---:|---:|
| LoRA against Deletion Tuning | +7.9 | [+5.5, +10.2] | 76 / 180 | 6.7e-11 |
| Full fine-tuning against Deletion Tuning | -12.7 | [-15.6, -9.9] | 291 / 123 | 8.5e-17 |
| Full fine-tuning against LoRA | -20.6 | [-23.5, -17.8] | 349 / 77 | 2e-42 |

**Full fine-tuning in the first-evaluation setting** (first 200 questions, 50 items per prior task; LoRA was not run
in this setting). Evaluated on both GPUs with the sharded code (`EVAL_SHARDS=2`), 400 new tokens on GSM8K:

| | base | Deletion Tuning | Full fine-tuning |
|---|---:|---:|---:|
| GSM8K, training format, first 200 (%) | 33.5 | 78.5 | 50.0 |
| GSM8K, chat template, first 200 (%) | 58.5 | 71.5 | 54.0 |
| GSM8K, training format, all 1,319 (%) | not run | not run | 52.99 (699/1319); 95.5% of replies contain `####` |
| hellaswag / truthfulqa_mc2 / mmlu | 70.00 / 56.96 / 69.54 | 70.00 / 56.26 / 69.75 | 70.00 / 58.52 / 68.49 |
| ifeval / winogrande | 67.11 / 78.00 | 63.16 / 80.00 | 59.21 / 70.00 |
| humaneval_instruct (harness decode, **unreliable**) | 14.00 | 12.00 | 0.00 |
| mean of the five tasks without HumanEval | 68.32 | 67.83 | 65.24 |
| mean of the six tasks (includes the unreliable HumanEval) | 59.27 | 58.53 | 54.37 |
| Team 10 `indic_v1` exact (44 rows) | 52.27 | 52.27 | 40.91 |
| Team 10 `safety_v1` exact (41 rows) | 56.10 | 56.10 | 60.98 |

**Forward KL** (base to tuned, per token, on the base model's own replies to the same 50 GSM8K questions; run 25 Nov):

| | Deletion Tuning | LoRA | Full fine-tuning |
|---|---:|---:|---:|
| forward KL per token (nats) | **0.1412** | 0.2691 | 0.4825 |

An earlier full fine-tuning value of 0.5124 was over 167 prompts and is superseded by the 50-prompt value above.

### How to read the new rows

- **Deletion Tuning and LoRA both raise GSM8K significantly and change none of the four prior tasks by more than the
  noise.** LoRA is higher on accuracy (+19.0 against +11.1) and trained faster. This is one seed with an untuned LoRA
  learning rate and the paper's Deletion Tuning values, so it does not show that LoRA is better than Deletion Tuning in
  general; on these 1,319 questions LoRA is ahead of Deletion Tuning by 7.9 points (interval +5.5 to +10.2, paired table
  above).
- **Full fine-tuning did not improve the chat-template score** (-1.6, p = 0.34, not distinguishable from base), although
  95% of its replies end in `####`. It learned the format, not the accuracy. The training loss stayed at about 0.45
  (0.55 at step 10, 0.61 at step 400, a single batch of 4 each time). At 1e-5 and batch 4 on all weights the run may
  simply be too noisy; a lower learning rate was not tried. Do not read this as "full fine-tuning does not work".
- **The first-evaluation setting shows somewhat more forgetting for full fine-tuning than for Deletion Tuning**: the
  mean of the five tasks that are not HumanEval is 65.24 against 67.83 (Deletion Tuning) and 68.32 (base), with IFEval
  (59.2 against 67.1 for base) and Winogrande (70.0 against 78.0) the largest drops, and Indic exact 40.9 against 52.3. At
  50 items per task one item is 2 points, so these are on the edge of the noise; the Indic drop is 5 of 44 rows, which
  is p = 0.06 by the exact McNemar test even if all five flips go the same way, so it is not a significant difference.
  The HumanEval zero is a decoding artifact, not evidence of lost coding ability (see the HumanEval caveat above).
- **The full fine-tuning weights did move; the update was not lost to bf16 rounding.** Comparing the saved weights with the
  base model (every 4th tensor, 490 tensors): about 90% of the attention, expert and router weights differ, by 2.3e-4 on
  average (about 1.7% of a typical weight, 25 to 33 bf16 steps). An optimizer probe with the training's own stochastic
  bf16 AdamW on synthetic weights (N(0, 0.02), lr 1e-5, 400 steps, constant gradient) keeps 1.00 of the intended movement,
  plain bf16 AdamW only 0.11. The measured size of the change is about what random-sign updates give (0.05 to 0.08 of the
  4e-3 that 400 steps in one direction would give), so the run is consistent with noise-dominated updates at batch 4,
  not with lost updates. That is consistent with, not proof of, a noise explanation: one learning rate and one seed, and
  gradient noise was not measured. It also means the arms are not at matched fit (see below).
- **Forward KL orders the arms Deletion Tuning < LoRA < full fine-tuning** (0.14, 0.27, 0.48), so Deletion Tuning moved
  the model's GSM8K replies least even though LoRA gained more accuracy. The KL is measured on GSM8K replies, the training
  domain, so part of it is the intended format change; it is 50 prompts and one seed, and it is not a measure of forgetting
  on other tasks. The other-task evidence is the prior-task and Team 10 rows above. **The arms are at different fit**
  (+11.1, +19.0 and -1.6 points of GSM8K), so this order is not a matched-fit comparison: for full fine-tuning, which
  did not learn, a KL with no accuracy gain is not comparable to the other two, and a matched-fit control on a smaller
  model (`results/dt_vs_fft.md`, section 7) found the raw order of such numbers can flip once fit is equal.
- **The larger evaluation has a smaller prior-task spread** (63.68 to 64.56 for the four tasks) because it covers only
  four log-likelihood tasks and omits the two generation tasks where full fine-tuning lost most.

## Not shown

- One seed, one training run per setting. No variance across seeds.
- 200 GSM8K questions and 50 items per prior task. ifeval and humaneval generate for hours, so they were kept small. The humaneval numbers are not meaningful (see the HumanEval caveat).
- LoRA and full fine-tuning: one seed each, learning rates 2e-4 and 1e-5 not tuned, 400 steps at batch 4. No statement
  about the methods in general. LoRA has no six-task or Team 10 result in the first-evaluation setting. Full
  fine-tuning did not learn (-1.6, p = 0.34), so the forgetting and KL rows compare arms at different fit; a matched-fit
  comparison (other learning rates, a larger effective batch, or checkpoints stopped at DT's gain) was not run.
- Not the capstone architecture (no KE2, no untied KAS head, no Muon) and a public model, so no claim about dry run 2.
- The lm-evaluation-harness benchmarks here are public benchmarks on a public model. No lab model or lab checkpoint was
  scored, and nothing from this run goes to Team 9's watcher or to `forgetting.py`.

## Exact commands and output (TSAI run rule)

Training (`train_v4.out`, 22 Nov, 7,135 s):

```
RUN_NAME=dt_phi_mini_sft_gsm8k_seed0_v4 nohup bash run.sh train > train_v4.out 2>&1 &
...
[gsm8k] iter 400/400 | loss=0.6039
done: 12,296,351 of 2,013,265,920 gate weights deleted (0.6108%), mask 23.2 MB, 7035 s
[train] finished in 7135 s
```

Evaluation (`eval_v4.out`, base 18,956 s, Deletion Tuning 21,007 s, report 0 s):

```
RUN_NAME=dt_phi_mini_sft_gsm8k_seed0_v4 GSM8K_LIMIT=200 PRIOR_LIMIT=50 KL_LIMIT=50 MAX_NEW_TOKENS=256 \
  nohup bash run.sh base_eval dt_eval report > eval_v4.out 2>&1 &
...
prior tasks: {'hellaswag': 0.7, 'truthfulqa_mc2': 0.5626, 'mmlu': 0.6975, 'ifeval': 0.6316, 'winogrande': 0.8, 'humaneval_instruct': 0.12} mean 0.5853
[dt_eval] finished in 21007 s
```

Both logs, with the server paths replaced, are in `RaviNaik/ERA` at `v5/capstone/dt_phi_moe/` (`train_v4.out`,
`eval_v4.out`, `README.md`).

## Next

1. LoRA in the first-evaluation setting (six prior tasks at 50 items, Team 10 sets), so the three arms are compared on
   IFEval, HumanEval and the Indic and safety sets as well.
2. HumanEval on all 164 problems with the decode fix for base, Deletion Tuning, LoRA and full fine-tuning, and a lower
   full fine-tuning learning rate if time allows.
3. This file is updated in place with those rows.

## Commands and output for the 25 Nov additions (TSAI run rule)

All runs used the default `CUDA_LAUNCH_BLOCKING=1` of `run.sh` (synchronous CUDA launches; asynchronous mode crashed on
two GPUs), so the times below are **not speed-optimized** and are an upper bound. Code and logs: `RaviNaik/ERA`,
`v5/capstone/dt_phi_moe/`, result files at commit `d9b406b` (branch `dt-phi-moe-results-nov25`), the sharded evaluation
code at `46c0b13`.

Training, 24 Nov (`train_lora.out`, `train_ft.out`):

```
RUN_NAME=dt_phi_mini_sft_gsm8k_seed0_v4 bash run.sh train_lora      # lora_lr 2e-4
done: lora, 41,418,752 trainable parameters, 4466 s
[train_lora] finished in 4562 s
RUN_NAME=dt_phi_mini_sft_gsm8k_seed0_v4 bash run.sh train_ft        # ft_lr 1e-5, scope all
done: ft, 7,647,632,704 trainable parameters, 7045 s
[train_ft] finished in 7244 s
```

Larger evaluation (`ext_*.out`, one GPU per stage, 256 new tokens, 200 items per prior task):

```
PROFILE=extended EXTENDED_STAGES="lora_eval" GPUS=1 N_GPUS=1 bash run.sh     # likewise base_eval, dt_eval, ft_eval
[base_eval] finished in 11531 s   [dt_eval] 11498 s   [ft_eval] 11144 s   [lora_eval] 11975 s   [paired] 127 s
prior tasks: {'hellaswag': 0.62, 'truthfulqa_mc2': 0.5226, 'mmlu': 0.6999, 'winogrande': 0.74} mean 0.6456   (LoRA)
```

Full fine-tuning, first-evaluation setting, two GPUs (`ft_eval_fast.out`):

```
EVAL_SHARDS=2 GPUS=0,1 GEN_BATCH=96 PRIOR_LIMIT=50 EVAL_PARTS_FT=gsm8k_chat,kl,team10,prior EVAL_SUFFIX=_fast \
  RUN_NAME=dt_phi_mini_sft_gsm8k_seed0_v4 bash run.sh ft_eval
forward KL: {'forward_kl_per_token_mean': 0.5124341230727956, 'n_prompts': 167}
prior tasks: {'hellaswag': 0.7, 'truthfulqa_mc2': 0.5852, 'mmlu': 0.6849, 'ifeval': 0.5921, 'winogrande': 0.7, 'humaneval_instruct': 0.0} mean 0.5437
[ft_eval] finished in 13070 s
```

Forward KL on the same 50 prompts, one GPU each (`kl_lora.out`, `kl_ft.out`, 25 Nov):

```
GPUS=0 N_GPUS=1 EVAL_PARTS_LORA=kl KL_LIMIT=50 EVAL_SUFFIX=_kl50 FORCE=1 RUN_NAME=dt_phi_mini_sft_gsm8k_seed0_v4 bash run.sh lora_eval
forward KL: {'forward_kl_per_token_mean': 0.2690888488292694, 'n_prompts': 50}
[lora_eval] finished in 133 s
GPUS=1 N_GPUS=1 EVAL_PARTS_FT=kl KL_LIMIT=50 EVAL_SUFFIX=_kl50 FORCE=1 RUN_NAME=dt_phi_mini_sft_gsm8k_seed0_v4 bash run.sh ft_eval
forward KL: {'forward_kl_per_token_mean': 0.4825420159101486, 'n_prompts': 50}
[ft_eval] finished in 144 s
```

HumanEval check on the first 20 problems, base and full fine-tuning, one GPU each (`tools/humaneval_check.py` then
`tools/humaneval_rescore.py`; the replies are on the server, not in git). The first two lines are the evaluation's own decode, the
last two add the rescoring with the first line set to 4 spaces:

```
{"tag": "ft", "n": 20, "pass@1": 0.0, "with_code_block": 0, "with_def": 1, "with_####": 0}
{"tag": "base", "n": 20, "pass@1": 0.15, "with_code_block": 16, "with_def": 2, "with_####": 0}
{"file": "he_ft.json", "n": 20, "as_is": 0.0, "first_line_fixed": 0.7, ..., "first_line_indents": [3]}
{"file": "he_base.json", "n": 20, "as_is": 0.15, "first_line_fixed": 0.85, ..., "first_line_indents": [3]}
```

Full fine-tuning weights against base, optimizer probe, paired test between variants (25 Nov, CPU, `tools/weight_change.py`,
`tools/paired_variants.py`):

```
constant_gradient / stochastic_bf16 (training)   changed 100.0%  mean movement 1.000 x intended  spread 0.143
constant_gradient / plain_adamw_bf16             changed  15.6%  mean movement 0.112 x intended  spread 0.287
490 tensors compared (0 not found in the base model)
experts w1  90.58% changed, mean|change| 2.285e-04, mean|w| 1.334e-02, 32.7 bf16 steps   (attention 89.87%, router 90.32%)
| LoRA against Deletion Tuning, chat template | 1319 | 67.1 | 75.0 | +7.9 | [+5.5, +10.2] | 76 / 180 | 6.7e-11 |
```

The 52.99% (all 1,319 questions, training format, 400 new tokens) came from an earlier two-GPU run of the same stage that was
stopped before its other parts; its replies are not in the repo.
