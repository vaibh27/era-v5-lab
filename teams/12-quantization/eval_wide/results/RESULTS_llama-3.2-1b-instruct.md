# Wide benchmark pass: Llama-3.2-1B-Instruct (G12, week 4)

**Q8_0 loses nothing measurable. Q4_K_M loses 0.9 points on MMLU, a small drop that the full 14,042-item split calls real (p 0.001).**
The 570-item MMLU slice of eval_mini saw −0.2 (p 1.0) for the same model (#295). Under `docs/bitwidth_proposal.md`, Q4_K_M
fails MMLU on the p < 0.01 rule, although its point drop is far inside the −3 limit. Q8_0 passes everything. This is the
third model of the wide pass, after sarvam-1 and Granite 3.0 3B-A800M ([`RESULTS.md`](RESULTS.md)), and it points the same way.

Two of the five Belebele languages could not be scored for this model: the 5-shot Kannada and Telugu prompts do not fit in
4,096 tokens (section 4). Only Hindi is reported.

## 1. Scores

`unsloth/Llama-3.2-1B-Instruct` @ `5a8abab4a5d6` (ungated mirror of `meta-llama/Llama-3.2-1B-Instruct`, the same revision as
#295), llama.cpp b11669 GGUF files (output layer Q8_0 in the quantized files), llama-cpp-python 0.3.36 in-process on one
Tesla T4, `n_ctx` 4,096, seed 42, no request truncated in any scored run. Accuracy in %; ARC-C is `acc_norm`, the others `acc`.
Chance is 25% on every task.

| Task | Items | Shots | bf16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|---:|---:|
| MMLU | 14,042 | 0 | 48.24 | 48.35 | 47.35 |
| ARC-C | 1,172 | 0 | 37.88 | 38.05 | 38.14 |
| Belebele Hindi | 900 | 5 | 34.00 | 34.22 | 32.67 |

| | bf16 | Q8_0 | Q4_K_M |
|---|---:|---:|---:|
| GGUF size | 2.48 GB | 1.32 GB | 0.87 GB |
| Guard test (`test_llamacpp_lm.py`), prefix-cache diff | PASS, 0.0000 | PASS, 0.0002 | PASS, 0.0013 |
| Scoring time, 3 tasks | 62 min | 26 min | 25 min |

## 2. Each format against bf16 on the same items

Difference in points (format minus bf16), 95% CI from the paired variance, b = right in bf16 and wrong in the format, c = the
reverse, McNemar's exact test (same statistics as `../eval_mini/paired.py`). Full table:
[`paired_llama-3.2-1b-instruct.md`](paired_llama-3.2-1b-instruct.md). **Bold**: p < 0.01.

| Task | Q8_0 | p | Q4_K_M | p |
|---|---|---:|---|---:|
| MMLU | +0.11 (−0.1 to +0.3) | 0.386 | **−0.9** (−1.4 to −0.4) | 0.001 |
| ARC-C | +0.2 (−0.5 to +0.9) | 0.815 | +0.3 (−1.3 to +1.8) | 0.828 |
| Belebele Hindi | +0.2 (−0.8 to +1.2) | 0.824 | −1.3 (−3.4 to +0.7) | 0.241 |

## 3. What it shows

- **Q8_0 is indistinguishable from bf16 on every task.** On MMLU the interval is about ±0.2 points, so a 0.5-point loss is
  ruled out. 261 of 14,042 items (1.9%) change answer, in both directions about equally.
- **Q4_K_M fails MMLU under `docs/bitwidth_proposal.md`, narrowly.** The point drop, −0.9, is inside the −3 limit, but the
  proposal also fails a task at McNemar p < 0.01, and MMLU's p is 0.001: 1,459 of 14,042 items (10.4%) change answer, 792 right
  to wrong against 667 wrong to right. OLMoE's Q4_K_M flipped 7.5%. ARC-C (+0.3) is inside its limit.
- **The 570-item slice could not see this.** The proposal lists Llama-3.2-1B-Instruct as inside every Q4_K_M limit, from the
  1,870-item eval_mini (MMLU −0.2, p 1.0). With 570 items its interval was wide enough to hold the −0.9 found here.
  This is the point of the wide pass.
- **Belebele Hindi is not informative for this model.** bf16 scores 34.0%, 9 points above chance. Q4_K_M's −1.3 has an
  interval of −3.4 to +0.7 (p 0.241), so a drop of up to 3 points is not ruled out. The proposal sets no Indic limit.
- **Speed.** Over the three tasks, Q8_0 and Q4_K_M score 2.4 and 2.5 times faster than bf16 on a T4 (MMLU: 0.08 s/item against 0.22).

A recommendation for this model, from these results alone: serve **Q8_0**. If Q4_K_M is wanted for size (0.87 GB against 1.32 GB),
the cost is under a point on MMLU and it passes the point limit. Whether that is acceptable is the lead's decision.
`bitwidth_proposal.md` still lists Q4_K_M as passing for this model. This PR does not edit that file.

## 4. What did not work: Belebele Kannada and Telugu

The first run of this pass (Colab T4, 23 Nov, bf16 only, before the session ended) included all five Belebele languages. Kannada
and Telugu scored 28.78% and 25.89% (chance is 25%) and each reported **3,600 left-truncated requests, which is 900 items
times 4 choices, so every request**. Hindi, MMLU and ARC-C reported 0. The summary of that run is
[`raw/llama-3.2-1b-instruct/colab_first_run_bf16_summary.json`](raw/llama-3.2-1b-instruct/colab_first_run_bf16_summary.json).
The likely cause is that Llama's tokenizer needs far more tokens for Kannada and Telugu text than sarvam-1's, so five shots plus the
passage do not fit in `n_ctx` 4,096. That cause is not checked: nobody counted the tokens. The two scores are not model scores
and are not in the tables above.

Tamil and Bengali never ran (the session ended). A second run was restricted to MMLU, ARC-C and Belebele Hindi, with the same
Belebele settings. Rerunning the four languages at `n_ctx` 8,192 is possible. It would differ from the setup of #737 and is
slower, so it was not done.

## 5. How it was run

Notebook: [`raw/llama-3.2-1b-instruct/eval_wide_llama32_1b.ipynb`](raw/llama-3.2-1b-instruct/eval_wide_llama32_1b.ipynb), the
`eval_wide.ipynb` of #714 with only its settings changed: model and revision, formats bf16 / Q8_0 / Q4_K_M, tasks MMLU, ARC-C
and Belebele Hindi, full splits, `RUN_BY = "vpw"`. The code is unchanged (`run_wide_gguf.py` sha256 prefix `66f9de0b6b168f49`).

- **Hardware.** The scored results are from one Tesla T4 (15 GB, driver 580.178) on a GCP `n1-standard-4`, run headless from the
  notebook on 24 Nov. Colab gave no GPU and the EC2 T4 had no capacity.
- **bf16 reproduces across machines.** The first Colab run (T4, 23 Nov) scored bf16 identically to the last digit on all three
  tasks: MMLU 48.24, ARC-C 37.88, Belebele Hindi 34.00. Only the 24 Nov run has Q8_0 and Q4_K_M.
- **One lost stretch.** An unattended driver upgrade on the GCP box (NVIDIA 580 libraries over the loaded 570 kernel module) made
  the first Q8_0 attempt fall back to CPU. It was stopped and discarded, the box rebooted, and Q8_0 and Q4_K_M were rerun from the
  start. bf16 had finished before the upgrade and was kept. Every number above is from a run with GPU offload (all layers).
- **No item text** is stored: only task, doc_id, filter and the metric per item (Team 9's firewall).
- Files: [`raw/llama-3.2-1b-instruct/`](raw/llama-3.2-1b-instruct/) has per-item samples and `summary.json` per format, plus the guard-test
  and quantize logs.
