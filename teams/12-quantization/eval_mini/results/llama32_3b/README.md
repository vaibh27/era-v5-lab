# Llama-3.2-3B-Instruct, bf16 vs Q4_K_M (eval_mini)

Run by chapanduk on an Apple M4 Max (64 GB unified memory, Metal), 12 Nov 2026. Model
`unsloth/Llama-3.2-3B-Instruct` (ungated mirror of the gated `meta-llama/Llama-3.2-3B-Instruct`), revision
`006f5dcd1393c3add266de40994ba96225e9689d`. llama.cpp b11669 for conversion and quantizing, llama-cpp-python 0.3.36
(in-process, all layers on Metal), lm_eval 0.4.13, datasets 5.1.0, G9's `run_eval_mini.py` (sha256 `e91bd8b37975d098`).

Both formats ran in the same session on the same machine, so the paired comparison against bf16 is made directly
(no `combine.py` needed).

## Files

`scores.csv`, `scores_long.csv`, `paired.md`, `environment.json`, per format `summary.json` plus one
`*.samples.jsonl` per task (one row per question), and `logs/`: the two `test_llamacpp_lm.py` checks, the Q4_K_M
quantize log and `llama_bench.md`.

## Results

| Task | bf16 | Q4_K_M | Difference (95% CI) | b / c | McNemar p |
|---|---|---|---|---|---|
| ARC-Challenge (500) | 46.6 | 45.2 | -1.4 (-3.4 to +0.6) | 16 / 9 | 0.230 |
| HellaSwag (400) | 62.3 | 61.8 | -0.5 (-2.0 to +1.0) | 6 / 4 | 0.754 |
| MMLU (570) | 64.4 | 64.6 | +0.2 (-1.9 to +2.2) | 17 / 18 | 1.000 |
| GSM8K 5-shot (200) | 65.0 | 65.5 | +0.5 (-5.8 to +6.8) | 20 / 21 | 1.000 |
| TruthfulQA-mc2 (200) | 52.3 | 53.2 | +0.9 (-0.0 to +1.8) | - | - |

Mean score: bf16 58.12, Q4_K_M 58.05 (-0.07 pp). **No task differs significantly from bf16** (all McNemar p > 0.2).
TruthfulQA-mc2's CI lower bound rounds to -0.0: a small, borderline upward shift in the probability share, not a
loss. b = right in bf16, wrong in Q4_K_M; c = the reverse.

| | bf16 | Q4_K_M |
|---|---|---|
| File size | 6.43 GB (16.00 BPW) | 2.11 GB (5.25 BPW), **3.05x smaller** |
| Quantize time | - | 9.5 s (llama-quantize, M4 Max CPU) |
| Prompt processing, pp512 | 2288 t/s | 2006 t/s (0.88x) |
| Generation, tg128 (10 reps) | 66.7 ± 1.4 t/s | 143.3 ± 7.1 t/s (**2.15x**) |
| eval_mini wall clock (sum of tasks) | 1054 s | 859 s |

Generation is memory-bandwidth bound, so the 3x smaller file generates about 2x faster. Prompt processing is
compute bound and Q4_K_M pays for dequantizing, so it is slightly slower. That is why GSM8K (the only generation
task) took 508 s in bf16 and 333 s in Q4_K_M while the multiple-choice tasks took the same time.

Dense model (no router); tied embeddings (`tie_word_embeddings: true`), so there is no separate `output.weight`
tensor and `--output-tensor-type q8_0` quantizes `token_embd.weight` itself (751.5 MiB -> 399.2 MiB), the same
convention as the Llama-3.2-1B and Qwen2.5-1.5B rows.

## 1B vs 3B: does the 4-bit drop shrink with size?

| | Llama-3.2-1B ([`llama32_1b`](../llama32_1b/)) | Llama-3.2-3B (this run) |
|---|---|---|
| Mean, bf16 -> Q4_K_M | 42.71 -> 42.05 (-0.66) | 58.12 -> 58.05 (-0.07) |
| GSM8K | 30.5 -> 26.5 (-4.0, p 0.229) | 65.0 -> 65.5 (+0.5, p 1.000) |
| Questions flipped, ARC+HellaSwag+MMLU+GSM8K | 138 of 1,670 | 111 of 1,670 |
| Compression | 2.85x | 3.05x |

Neither model shows a significant drop on any task, so this is a direction, not a result: the 3B model's mean moves
less, its GSM8K does not drop, and fewer answers flip (111 vs 138). That fits the usual picture that larger models have
more redundancy to absorb rounding error. The 3B also compresses slightly better because the tied embedding (kept
at Q8_0) is a smaller share of its weights.

## Differences from the notebook runs

- **Not run through `eval_mini.ipynb`.** The notebook's full-run path builds a CUDA wheel and stops without GPU
  offload; its Mac mode is a 2-item dry run. Its cells were executed by hand on macOS with the same pins:
  llama.cpp b11669 built with Metal, `convert_hf_to_gguf.py --outtype bf16` in a separate venv from llama.cpp's
  requirements, `llama-quantize --output-tensor-type q8_0`, `test_llamacpp_lm.py` on both files (both PASS), then the
  committed `run_mini_gguf.py` and `paired.py`. The evaluation venv had no torch or transformers, as the notebook
  checks. `environment.json` therefore has `notebook_version: null`; `combine.py` would refuse this run against
  other members' runs, but nothing here needs combining.
- **`run_meta` is correct without hand edits.** `run_mini_gguf.py` was given `--model` and `--revision`, so
  `summary.json` names this model (the notebook does not pass them, which is why the Phi-3-mini and Llama-3.2-1B
  runs had to fix the OLMoE default by hand).
- **Guard check.** bf16: log p(" Paris") = -0.360, generation -0.360, off-by-one value -9.226, cache max diff 0.0001.
  Q4_K_M: -0.513, -0.513, -9.303, 0.0002. Both PASS.
- `quant_time_s` is the `llama-quantize` wall time alone; `eval_time_s` is `run_mini_gguf.py`'s per-task wall-clock
  sum, excluding quantize and the check. `gen_tok_s` in `runs.csv` is llama-bench tg128.
