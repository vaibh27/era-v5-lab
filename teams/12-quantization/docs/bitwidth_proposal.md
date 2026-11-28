# Bit-width proposal (G12), version 2

**Proposal v2, 27 Nov 2026. Serve Q8_0 and keep bf16 as the reference. Q6_K is the candidate when the file must be smaller. Do not ship Q5_K_M or Q4_K_M. Do not keep Q3_K_M or Q2_K.** TSAI decides the serving bit width at Gate 4.

This replaces version 1 (13 Nov), which was written from eval_mini's 200 to 570 items per task. The wide pass ([`eval_wide/results/RESULTS.md`](../eval_wide/results/RESULTS.md)) scores full MMLU (14,042 items), ARC-C (1,172) and Belebele (900 per language) on four public models, and shows what eval_mini could not: **Q4_K_M fails MMLU by the p < 0.01 rule on all four models**, although its point drops (−0.9 to −2.9) look small. Version 1's "ship Q4_K_M when it passes" no longer selects any model. Every quantized file keeps `output.weight` at Q8_0 and the router (`ffn_gate_inp`) at F32.

## What changed from version 1

| | Version 1 (13 Nov) | Version 2 (27 Nov) |
|---|---|---|
| Evidence | eval_mini, 1,870 items in five tasks; a full format ladder only on OLMoE | wide pass, full splits, four models, three to five formats each |
| Default format | bf16, and Q4_K_M when it passes | **Q8_0** (bf16 as the reference) |
| Q4_K_M | passed on 6 of 10 models | fails MMLU at p ≤ 0.001 on 4 of 4 wide-pass models |
| Indic tasks | no limit (OLMoE sat at chance) | Belebele has a proposed limit (below); MILU stays a control |
| Open question | next format to measure | how to treat formats the guard test stops (below) |

## Formats

| Decision | Formats | Why |
|---|---|---|
| Ship | bf16 (reference), **Q8_0** (default) | Q8_0 passes on 4 of 4 models: no p below 0.07 on any task, MMLU interval about ±0.35 points on sarvam-1. It is 53% of bf16's size. |
| Candidate, when size matters | Q6_K | Passes on the 3 models where it was scored (sarvam-1, gemma-2-2b, Llama-3.2-1B); not scored on Granite (guard test). 42% to 44% of bf16, about 80% of Q8_0. On sarvam-1 its Indic reading leans slightly negative (pooled −0.69, p 0.028), so confirm on the lab's own model before shipping it. |
| Do not ship | Q5_K_M | Fails sarvam-1 on MMLU (−0.70, p 0.002); passes gemma-2-2b; not scored on Granite and Llama (guard test). It saves only 4 to 5 percentage points of bf16's size over Q6_K, which does not buy the risk. |
| Do not ship | Q4_K_M | Fails MMLU on all four models (−0.9 to −2.9, every p ≤ 0.001), and loses 4.0 to 7.6 points on every Belebele language of the Indic model. |
| Do not keep | Q3_K_M, Q2_K | Q3_K_M misses OLMoE's MMLU (−4.2, p 0.004) and HumanEval+ (−3.7, p 0.031), and sarvam-1's Q3_K_M did not pass the guard test. Q2_K misses ARC-C, HellaSwag, MMLU and GSM8K on OLMoE. |

A model is not given a quantized format until it has been scored against its own bf16. The limits are not raised to make a format pass.

## Evidence: four models, paired against each model's bf16

Difference in points, format minus bf16, McNemar p in brackets, on the same items. Sizes are the GGUF files, with the share of the model's bf16 file. **Bold**: p < 0.01. Source: [`RESULTS.md`](../eval_wide/results/RESULTS.md) (sarvam-1, Granite 3.0 3B-A800M, gemma-2-2b) and [`RESULTS_llama-3.2-1b-instruct.md`](../eval_wide/results/RESULTS_llama-3.2-1b-instruct.md).

| Model | Format | Size (share of bf16) | MMLU | Largest drop in the other tasks | Under the proposal |
|---|---|---|---|---|---|
| sarvam-1 (dense, Indic) | Q8_0 | 2.69 GB (53%) | −0.24 (0.19) | Belebele Bengali −1.22 (0.071) | pass |
| | Q6_K | 2.11 GB (42%) | −0.01 (1.00) | Belebele Bengali −1.78 (0.029) | pass |
| | Q5_K_M | 1.84 GB (36%) | **−0.70 (0.002)** | Belebele Hindi −1.89 (0.057) | fail (MMLU) |
| | Q4_K_M | 1.58 GB (31%) | **−2.94 (< 0.001)** | Belebele Kannada **−7.56 (< 0.001)** | fail |
| Granite 3.0 3B-A800M (MoE, English) | Q8_0 | 3.59 GB (53%) | −0.04 (0.84) | ARC-C −0.34 (0.57) | pass |
| | Q6_K, Q5_K_M | not scored | | | guard test stopped both |
| | Q4_K_M | 2.08 GB (31%) | **−2.31 (< 0.001)** | ARC-C −0.26 (0.84) | fail (MMLU) |
| gemma-2-2b (dense, multilingual) | Q8_0 | 2.78 GB (53%) | +0.01 (1.00) | Belebele Bengali −0.78 (0.28) | pass |
| | Q6_K | 2.29 GB (44%) | −0.11 (0.47) | Belebele Kannada −0.56 (0.44) | pass |
| | Q5_K_M | 2.07 GB (40%) | +0.09 (0.68) | Belebele Kannada −1.67 (0.101) | pass |
| | Q4_K_M | 1.85 GB (35%) | **−1.46 (< 0.001)** | Belebele Kannada −2.33 (0.099) | fail (MMLU) |
| Llama-3.2-1B-Instruct (dense) | Q8_0 | 1.32 GB (53%) | +0.11 (0.39) | none (ARC-C and Belebele Hindi are gains) | pass |
| | Q6_K | 1.09 GB (44%) | +0.12 (0.50) | Belebele Hindi −1.33 (0.065) | pass |
| | Q5_K_M | not scored | | | guard test stopped it |
| | Q4_K_M | 0.87 GB (35%) | **−0.89 (0.001)** | Belebele Hindi −1.33 (0.24) | fail (MMLU) |

What it shows:

- **Q8_0 is the one format that is clean on every model.** The lowest p on any task of any model is 0.071 (sarvam-1 Belebele Bengali, −1.22), then 0.11 (sarvam-1 ARC-C, −0.85).
- **Q6_K is clean on three models and not scored on the fourth.** Llama's Q6_K ran on two GPUs where its other formats ran on one; the split's share of its per-item differences is not measured ([`RESULTS_llama-3.2-1b-instruct.md`](../eval_wide/results/RESULTS_llama-3.2-1b-instruct.md), section 6).
- **Q5_K_M splits the models:** MMLU −0.70 on sarvam-1 (fail) against +0.09 on gemma-2-2b (pass). A format that passes on some models and not on others cannot be a default.
- **Q4_K_M's loss is small per item but real:** 10% to 18% of MMLU items change answer (sarvam-1 15%, Granite 18%, gemma-2-2b 11%, Llama 10%). On MMLU the point limit alone would pass Q4_K_M on every model (−0.9 to −2.9 against −3); the p < 0.01 rule is what fails it, because the full split is large enough to see a 1 to 3 point loss. Whether that loss is acceptable for the size it saves is the trade-off for TSAI:

| | Q6_K | Q4_K_M |
|---|---|---|
| File size, share of Q8_0's | 78% to 83% | 58% to 67% |
| MMLU against bf16 | 0.0 to 0.1 points (not significant; 3 models) | 0.9 to 2.9 points (p ≤ 0.001; 4 models) |
| Indic reading against bf16 (sarvam-1, pooled) | −0.69 (p 0.028) | −5.09 (p < 0.0001) |

## Indic reading (sarvam-1, Belebele, five languages)

sarvam-1 is the only model with five Indic languages well above chance (51% to 59% at bf16). MILU stays at 28% to 30% there and is a control only. Belebele pooled over the five languages (4,500 items), paired against bf16, by [`eval_wide/pooled.py`](../eval_wide/pooled.py):

| Format | Per language (Kannada, Hindi, Telugu, Tamil, Bengali) | Pooled difference (95% CI) | McNemar p |
|---|---|---|---:|
| Q8_0 | −0.56, −0.33, 0.00, 0.00, −1.22 | −0.42 (−0.92 to +0.08) | 0.118 |
| Q6_K | −0.11, −0.67, −0.56, −0.33, −1.78 | −0.69 (−1.28 to −0.09) | 0.028 |
| Q5_K_M | −1.00, −1.89, +0.89, −0.67, −1.56 | −0.84 (−1.70 to +0.01) | 0.059 |
| Q4_K_M | −7.56, −4.67, −5.00, −4.22, −4.00 | **−5.09 (−6.20 to −3.98)** | < 0.0001 |

Q8_0, Q6_K and Q5_K_M all lean slightly negative (−0.4 to −0.8), which no single 900-item language can show; the pooled test puts Q6_K just under p 0.05 (0.028) and the other two above it. Q4_K_M's loss is an order of magnitude larger and hits Kannada hardest (23% of its margin above chance, against 12% for MMLU). The other models do not test this: gemma-2-2b's Indic reading is weaker (14 to 24 points above chance) and no Belebele language drops under p 0.01 at any format; Llama's Kannada, Telugu, Tamil and Bengali sit at chance, and Granite 3.0 is not trained on Indic languages.

## Drop limit per task

Version 1's limits stand for the tasks it set them for, and a task also fails at McNemar p < 0.01. The limit is on the paired point estimate (format minus bf16); a drop equal to the limit passes; a gain does not fail. The wide pass adds the full-split sizes and one new row.

| Task | Items (v1 / wide) | Drop limit | What the wide pass shows |
|---|---|---|---|
| ARC-C | 500 / 1,172 | −2 | no format on any model is under −1.4 |
| MMLU | 570 / 14,042 | −3 | sarvam-1 Q4_K_M −2.94 is just inside the point limit and fails on p; the p rule decides |
| HellaSwag, GSM8K, TruthfulQA-mc2, HumanEval+, MBPP+ | v1 only | −2, −4, −2, −3, −3 | not rerun on the wide pass; v1's limits and results stand for OLMoE and the other v1 models |
| **Belebele, per language (new, proposed)** | 900 | **−3** | sarvam-1: Q8_0, Q6_K and Q5_K_M are all within 1.9 points of bf16 on every language; every language of Q4_K_M is −4.0 or worse. gemma-2-2b's Q4_K_M reaches −2.33 (Kannada) |

The Belebele limit is a proposal for TSAI: on sarvam-1, the model that tests it best, no format other than Q4_K_M comes within 1.1 points of it. Perplexity ratio and KL are not gates.

## Formats the guard test stopped

`eval_mini/test_llamacpp_lm.py` runs before scoring. Checks 1 and 2 test that scores are not shifted by one position; check 3 that scoring with the prefix cache equals scoring from scratch (limit 0.05 nats); check 4 that the model ranks the right answer of an easy item first. The guard was not loosened for any of these runs.

| Model | Format | Check that stopped it | Value |
|---|---|---|---|
| sarvam-1 | Q3_K_M | 3 (prefix cache), in two runs | 0.0862 |
| Granite 3.0 3B-A800M | Q6_K | 3 and 4 | 0.0784; right −1.60, wrong −1.53 |
| Granite 3.0 3B-A800M | Q5_K_M | 4 | right −1.84, wrong −1.84 |
| Llama-3.2-1B-Instruct | Q5_K_M | 4 | right −0.96, wrong −0.92 |

Granite's check 3 does not follow bit width: 0.0434 (bf16), 0.0097 (Q8_0), 0.0424 (Q4_K_M), 0.0498 (Q5_K_M), 0.0784 (Q6_K). That looks like MoE routing near-ties changing with batch shape, not quantization. Check 4 asks whether the model is strong enough for the prompt, not whether scoring is correct, and bf16 passed it in every case; three of the eight Q6_K and Q5_K_M runs stopped on it by 0.07 nats or less. **Open question for the lead and TSAI:** require check 4 of the bf16 reference only and record it for quantized files, then rerun the formats it stopped; Granite Q6_K also needs check 3 at 0.10, noted in its results. Neither change is applied here. Until Granite Q6_K is scored, Q6_K is not shown on a mixture-of-experts model.

## What this does not show

- **All four models are public stand-ins.** None is the lab's model (KE2/KAS head, 16 routed and 1 shared expert). The first format to ship needs the same pass on the lab's own checkpoint, against its own bf16.
- **One seed, one runtime** (llama.cpp b11669, llama-cpp-python 0.3.36, scored one request at a time), and Kaggle T4 GPUs. Serving speed and throughput are not measured here.
- **Indic evidence is one model.** Llama's four of five Indic languages are at chance, Granite is English-only, gemma-2-2b is weaker in Indic.
- MILU is capped at the first 2,500 items per language; Llama has only Belebele Hindi.

## Serving defaults that follow

- **Weights:** Q8_0, with bf16 as the reference. The Ollama release built in [`ollama/`](../ollama/README.md) ships BF16 and Q8_0 and no Q4_K_M, which matches this proposal.
- **KV cache:** fp8 first. On sarvam-1 in vLLM 0.33.0, an fp8 KV cache gives 1.94 times bf16's tokens with next-token NLL within noise (+0.0006) and 19% more tokens per second; vLLM's TurboQuant presets hold fewer tokens than fp8 on the dry-run-2 shape (T2, 8 layers; computed from vLLM's slot sizes, not run), so they are not a better option there. One model, a proxy, no task scores: [`serving/kv_cache/`](../serving/kv_cache/README.md).

## Before the main run ships

1. Run the wide pass (full MMLU and ARC-C, Belebele in the Indic languages) on the lab's checkpoint for bf16, Q8_0 and Q6_K, with `eval_wide/` as it is. Check the truncated-request count: Llama's Indic scripts needed `n_ctx` 12,288 where sarvam-1 fits in 4,096.
2. Ship Q8_0 if it passes; ship Q6_K only if it also passes and TSAI accepts the size gain; otherwise bf16.
3. Re-measure the Ollama export's cost (head bias, router bias, tokenizer) on the real checkpoint ([`ollama/`](../ollama/README.md)).

## Appendix: version 1 evidence (eval_mini)

Kept for the record. These cells come from eval_mini's 200 to 570 items per task and can hide a 1 to 3 point loss: the wide pass found a significant MMLU loss for Llama-3.2-1B-Instruct where the table below says "yes". The OLMoE ladder and the format limits above still come from this evidence.

### Q4_K_M against the limits, version 1

| Model | Worst drop | Inside the limits |
|---|---|---|
| OLMoE-1B-7B | MMLU −1.9 | yes |
| Llama-3.2-1B-Instruct | GSM8K −4.0 | yes (equal to the GSM8K limit; p 0.23). Wide pass: MMLU −0.89, p 0.001, fails |
| Phi-3-mini-4k-instruct | GSM8K −3.0 | yes |
| Qwen3-4B | MMLU −2.5 | yes on the point limit. The 95% interval is −4.7 to −0.2 (p 0.049), so record the cell. p is not under 0.01, so it is not a reject. |
| Qwen3-4B-Thinking-2507 | GSM8K −2.0 | yes |
| SmolLM3-3B | ARC-C −1.8, MMLU −1.8 | yes. GSM8K is +6.5 (p 0.019), a gain. |
| Sarvam-30B | GSM8K **−9.5** | no (p 0.037; interval −17.9 to −1.1) |
| Granite-4.2-3B | ARC-C **−2.6**, MMLU **−3.7**, TruthfulQA-mc2 **−2.6** | no. MMLU p 0.014, interval −6.5 to −0.9. TruthfulQA interval −4.9 to −0.3. GSM8K is +5.0 and does not offset the other three. |
| Gemma-4-E2B-it | HellaSwag **−3.8**, GSM8K **−14.0** | no. GSM8K p < 0.001, interval −20.8 to −7.2. HellaSwag p 0.028, interval −6.9 to −0.6. TruthfulQA is +5.7 and does not offset GSM8K. |
| Nemotron-3.5-Bolt-E4B | GSM8K **−13.0** | no (p < 0.001; interval −18.9 to −7.1). MMLU is −2.5, inside the point limit, with interval −4.9 to −0.0 (p 0.065). |

### Version 1 limits and OLMoE ladder

| Task | Items | Drop limit | OLMoE Q4_K_M | OLMoE Q3_K_M | What sets the number |
|---|---|---|---|---|---|
| ARC-C | 500 | −2 | +0.2 | −0.2 | Q4 and Q3 are inside the noise. Q2_K is −7.6. |
| HellaSwag | 400 | −2 | −0.2 | −1.2 | Same pattern. Q2_K is −4.5. |
| MMLU | 570 | −3 | −1.9 | **−4.2** | The cut between Q4 and Q3. Q4's interval still includes zero (p 0.13). |
| GSM8K | 200 | −4 | +1.0 | −1.5 | Q2_K is −5.5 (p 0.052). The widest Q4 drop that the paired test does not call a loss is Llama-3.2-1B at −4.0 (p 0.23). |
| TruthfulQA-mc2 | 200 | −2 | −0.5 | −0.6 | No format on the ladder misses this. The largest move is Q2_K at +1.5. |
| HumanEval+ | 164 | −3 | −1.8 | **−3.7** | Q3 is the first significant code loss (p 0.031). Only OLMoE has this task. |
| MBPP+ | 378 | −3 | −0.5 | −1.6 | Same bar as HumanEval+. Q3 fails on HumanEval+, not here. |

Sources: [`eval_mini/results/RESULTS.md`](../eval_mini/results/RESULTS.md), [`eval_code_indic/results/RESULTS.md`](../eval_code_indic/results/RESULTS.md), [`runs/runs.csv`](../runs/runs.csv), [`bitwidth_vs_drop.md`](bitwidth_vs_drop.md) (the OLMoE ladder, bits per weight against drop), and the paired tables under `eval_mini/results/`.
