# Wide benchmark pass: Llama-3.2-1B-Instruct (G12, week 4)

**Q8_0 loses nothing measurable. Q4_K_M loses 0.9 points on MMLU, a small drop that the full 14,042-item split calls real (p 0.001).**
The 570-item MMLU slice of eval_mini saw −0.2 (p 1.0) for the same model (#295). Under `docs/bitwidth_proposal.md`, Q4_K_M
fails MMLU on the p < 0.01 rule, although its point drop is far inside the −3 limit. Q8_0 passes everything. This is the
third model of the wide pass, after sarvam-1 and Granite 3.0 3B-A800M ([`RESULTS.md`](RESULTS.md)), and it points the same way.

**Q6_K passes as well (26 Nov, section 6): MMLU +0.12, p 0.50. Q5_K_M was not scored: it failed check 4 of the guard test.**

Belebele Kannada, Telugu, Tamil and Bengali needed a longer context than #737's 4,096 tokens for this model (their 5-shot
prompts are 2 to 3.4 times as long as Hindi's). Rerun at `n_ctx` 12,288 with no truncation, bf16 scores 25 to 29% on them,
chance level, so they cannot show quantization drift for Llama-3.2-1B-Instruct (section 4). Hindi is the only
Indic language above chance by a useful margin.

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

## 4. Belebele Kannada, Telugu, Tamil, Bengali: truncation, then chance level

**The first run truncated them.** The first run of this pass (Colab T4, 23 Nov, bf16 only, before the session ended) included all
five Belebele languages. Kannada and Telugu scored 28.78% and 25.89% and each reported **3,600 left-truncated requests, which is
900 items times 4 choices, so every request**. Hindi, MMLU and ARC-C reported 0. The summary of that run is
[`raw/llama-3.2-1b-instruct/colab_first_run_bf16_summary.json`](raw/llama-3.2-1b-instruct/colab_first_run_bf16_summary.json).
Tamil and Bengali never ran (the session ended). The second run, section 1, was restricted to MMLU, ARC-C and Belebele Hindi.

**The cause is the tokenizer.** On 25 Nov the 5-shot requests were counted with the Llama-3.2-1B tokenizer (same revision), built by
lm_eval with the settings of the run
([`token_counts.py`](raw/llama-3.2-1b-instruct/indic_n12288/token_counts.py), output in `token_counts.txt` beside it):

| Belebele | Mean tokens | Longest | Requests over 4,096 | Over 8,192 |
|---|---:|---:|---:|---:|
| Hindi | 2,319 | 2,853 | 0 | 0 |
| Bengali | 4,704 | 5,856 | 3,600 | 0 |
| Tamil | 6,562 | 8,111 | 3,600 | 0 |
| Telugu | 7,057 | 8,793 | 3,600 | 16 |
| Kannada | 7,867 | 9,727 | 3,600 | 540 |

All 3,600 requests of each of the four languages exceed 4,096 tokens, so the Kannada and Telugu scores of the first run were
truncated, and Tamil and Bengali would have been too. At `n_ctx` 8,192 about 15% of Kannada would still be truncated, so the
rerun used 12,288.

**Rerun at `n_ctx` 12,288.** Same harness, seed, formats and GPU as section 1 (one T4, 26 Nov); only `n_ctx` and the tasks differ,
which is also a difference from #737's setup. No request was truncated in any of the 12 runs. The Q4_K_M file was quantized again
from the same bf16 GGUF with the same command (871 MB, as before). The guard test was not rerun, since the code and formats
passed it for section 1. Accuracy in %, chance is 25%; difference from bf16 in points with McNemar p
([`paired_indic_llama-3.2-1b-instruct_n12288.md`](paired_indic_llama-3.2-1b-instruct_n12288.md)):

| Belebele | bf16 | Q8_0 | Diff (p) | Q4_K_M | Diff (p) |
|---|---:|---:|---|---:|---|
| Kannada | 26.67 | 26.89 | +0.2 (0.82) | 29.00 | +2.3 (0.10) |
| Telugu | 24.78 | 25.00 | +0.2 (0.82) | 27.11 | +2.3 (0.07) |
| Tamil | 25.78 | 25.33 | -0.4 (0.52) | 26.11 | +0.3 (0.86) |
| Bengali | 29.44 | 29.22 | -0.2 (0.84) | 31.33 | +1.9 (0.20) |

**These four tasks carry no quantization signal for this model.** bf16 is at chance on Kannada, Telugu and Tamil (24.8 to 25.8%) and
only a little above it on Bengali (29.4%; a 95% interval of about 26 to 33% on 900 items). A model that cannot answer the
questions cannot lose accuracy on them. The Q4_K_M differences are all positive and none is significant (p 0.07 to 0.86); they are
noise around chance, not a gain from quantizing. Read nothing into their sign. For Llama-3.2-1B-Instruct the Indic drift
question rests on Hindi alone, and Hindi is weakly informative (section 3). A model that reads these languages, such as
sarvam-1 in #737, is needed to measure Indic drift on them.

## 5. How it was run

Notebook: [`raw/llama-3.2-1b-instruct/eval_wide_llama32_1b.ipynb`](raw/llama-3.2-1b-instruct/eval_wide_llama32_1b.ipynb), the
`eval_wide.ipynb` of #714 with only its settings changed: model and revision, formats bf16 / Q8_0 / Q4_K_M, tasks MMLU, ARC-C
and Belebele Hindi, full splits, `RUN_BY = "vpw"`. The code is unchanged (`run_wide_gguf.py` sha256 prefix `66f9de0b6b168f49`).

- **Hardware.** The scored results are from one Tesla T4 (15 GB, driver 580.178) on a GCP `n1-standard-4`, run headless from the
  notebook on 24 Nov (section 1) and from `run_wide_gguf.py` directly on 26 Nov (section 4). Colab gave no GPU and the EC2 T4
  had no capacity.
- **bf16 reproduces across machines.** The first Colab run (T4, 23 Nov) scored bf16 identically to the last digit on all three
  tasks: MMLU 48.24, ARC-C 37.88, Belebele Hindi 34.00. Only the 24 Nov run has Q8_0 and Q4_K_M.
- **One lost stretch.** An unattended driver upgrade on the GCP box (NVIDIA 580 libraries over the loaded 570 kernel module) made
  the first Q8_0 attempt fall back to CPU. It was stopped and discarded, the box rebooted, and Q8_0 and Q4_K_M were rerun from the
  start. bf16 had finished before the upgrade and was kept. Every number above is from a run with GPU offload (all layers).
- **No item text** is stored: only task, doc_id, filter and the metric per item (Team 9's firewall).
- Files: [`raw/llama-3.2-1b-instruct/`](raw/llama-3.2-1b-instruct/) has per-item samples and `summary.json` per format, plus the guard-test
  and quantize logs. [`raw/llama-3.2-1b-instruct/indic_n12288/`](raw/llama-3.2-1b-instruct/indic_n12288/) has the same for the
  four-language rerun, with its run logs, `token_counts.py` and `token_counts.txt`.

## 6. Q6_K and Q5_K_M (26 Nov)

Assigned by the Team 12 lead on 26 Nov. Same model and revision, runner (`run_wide_gguf.py` `66f9de0b…`), `n_ctx` 4,096,
seed 42, full splits and tasks as sections 1 and 2, output layer (the tied token embedding) Q8_0 as in the files above, no
request truncated. Run with `../eval_wide.ipynb` on a member's own Kaggle quota, on **2 × Tesla T4** (see the last point).

| Task | Items | Shots | bf16 | Q8_0 | Q6_K | Q4_K_M |
|---|---:|---:|---:|---:|---:|---:|
| MMLU | 14,042 | 0 | 48.24 | 48.35 | 48.36 | 47.35 |
| ARC-C | 1,172 | 0 | 37.88 | 38.05 | 37.88 | 38.14 |
| Belebele Hindi | 900 | 5 | 34.00 | 34.22 | 32.67 | 32.67 |

| | Q6_K | Q5_K_M |
|---|---:|---:|
| GGUF size | 1.09 GB | not scored |
| Guard test (`test_llamacpp_lm.py`) | PASS, prefix-cache diff 0.0002 | **FAIL, check 4** (checks 1 to 3 pass, prefix-cache diff 0.0002) |

Q6_K against bf16 on the same items (statistics as in section 2; full table in
[`paired_llama-3.2-1b-instruct.md`](paired_llama-3.2-1b-instruct.md)):

| Task | Q6_K | p | b / c |
|---|---|---:|---:|
| MMLU | +0.12 (−0.21 to +0.45) | 0.502 | 275 / 292 |
| ARC-C | +0.00 (−0.85 to +0.85) | 1.000 | 13 / 13 |
| Belebele Hindi | −1.33 (−2.64 to −0.03) | 0.065 | 24 / 12 |

- **Q6_K passes `docs/bitwidth_proposal.md`:** no task at p < 0.01, and MMLU and ARC-C are inside their limits. MMLU moves
  as little as at Q8_0 (+0.11). Belebele Hindi loses 1.33, the same as Q4_K_M; as section 3 says, this task is 9 points
  above chance and not informative for this model.
- **Q5_K_M was not scored.** Check 4 of the guard asks an easy multiple-choice item and requires the right answer to score
  highest. The Q5_K_M file gives the right answer −0.96 and a wrong one −0.92, so it fails by 0.04 nats. Checks 1 to 3, which
  test the scoring path itself, pass. The guard is not loosened for one format, as in [`RESULTS.md`](RESULTS.md) section 4.
- **Hardware differs from the rows above.** bf16, Q8_0 and Q4_K_M ran on one T4; Q6_K ran on two, where llama.cpp splits the
  layers across both GPUs. Section 5 shows bf16 reproducing exactly across two one-T4 machines, but a two-GPU bf16 run was not
  made, so how much of Q6_K's per-item difference comes from the split is not measured. The scoring time (29 min for the 3
  tasks) is not comparable with section 1's.
- Files: `raw/llama-3.2-1b-instruct/Q6_K/` (per-item rows and `summary.json`, no item text) and, in
  `raw/llama-3.2-1b-instruct/logs/`, the guard-test and quantize logs of both formats.
