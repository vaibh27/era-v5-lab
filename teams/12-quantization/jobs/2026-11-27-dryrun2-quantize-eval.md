# Dry run 2 checkpoint: quantize, score and 100 fixed prompts (Gate 4 evidence)

REQUEST FOR TSAI REVIEW. No approval and no run result is claimed. The authors are the G12 owners below. SreekanthZipsy, the lead, asked for this request (G12 channel, 27 Nov); TSAI runs it.

```text
Team:            12-quantization
Owner:           therahulbhati, s9k96
Lead approval:   SreekanthZipsy
Question:        Does the G12 flow take the dry run 2 checkpoint (KAS V2 head, loss-free router bias) to bf16, Q8_0, Q6_K and Q4_K_M GGUF files with no manual step, is the export wired exactly once the two biases Granite cannot hold are set aside, and how far does each format move from bf16 in score and on 100 fixed prompts?
Script:          teams/12-quantization/scripts/run_flow.py (it runs quantize_eval.py, then flow_fixed_prompts.py), with export_hf.py, check_gguf_logits.py, tokenizer/convert_frozen.py and Team 10's eval/effort_suite.py
Command:         python teams/12-quantization/scripts/run_flow.py --checkpoint ./inputs/dryrun2-1234/model.safetensors --tokenizer ./inputs/tokenizer.json --eval teams/12-quantization/prompts/eval_own_20.json --work ./work --out ./out/dryrun2-g12 --formats bf16,Q8_0,Q6_K,Q4_K_M --manifest teams/10-post-training/eval/effort_manifest.json --source-dir code_contests=./inputs/code_contests --source-dir hendrycks_math=./inputs/hendrycks_math --qe-args --spec ./inputs/dryrun2-1234/manifest.json --verify 16 --drop-router-bias --drop-head-bias
Config:          none. Every setting is on the command line or is a default of the scripts.
Inputs:          the dry run 2 model at step 1,221 as model.safetensors, its manifest.json, the frozen p1536 tokenizer (sha256 e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b), the two Team 10 parquet sets, and a prepared work folder. See "Source and staging".
Hardware:        1 GPU
Max time:        90 minutes (ESTIMATE, see "What I ran before filing")
Precision:       bf16 baseline; Q8_0, Q6_K and Q4_K_M weights, router F32, output layer Q8_0
Measures:        Export test A1 twice: against G5's model as trained (shows how far routing moves without the router bias) and against G5 with router_bias set to zero (the wiring alone; must pass). The KAS head bias's effect on the same prompts (verify.head_bias_effect). Test A3 (llama.cpp against G5 with router_bias zeroed and the folded head without its output bias). Per format: size, bits per weight, score on 20 items, eval time, tokens per second. On the 100 prompts per format: finish reasons, tokens per second, outputs identical to bf16 and mean shared prefix with bf16, reported separately for Team 10's 55 slots and our 45 fill slots.
Expected result: A1 with router_bias zeroed and A3 PASS. A1 against the trained routing FAILS by design; the size of the miss is the result. Q8_0 close to bf16 on the prompts and the score, Q6_K close, Q4_K_M further away. No score level is assumed: the model saw one English shard for 1,221 steps (TSAI's result note).
Outputs:         ./out/dryrun2-g12/{runs.csv,scores.json,manifest_<date>.json,prompts/summary.json,prompts/prompts_<format>.jsonl}, 50 MB or less (the dry run 1 smoke wrote 0.7 MB). GGUF files are deleted at the end. scores.json carries both A1 results and the statistics of the two dropped biases (export_a1, export_a1_zero_router_bias, export_router_bias, export_head_bias).
Test log:        "What I ran before filing" below
```

## One run

| Model | Formats | Prompts | Output directory |
|---|---|---:|---|
| dry run 2, dryrun2.yaml (KAS V2 head, bias_gamma 0.001), frozen p1536, seed 1234, step 1,221 | bf16, Q8_0, Q6_K, Q4_K_M | 100 | ./out/dryrun2-g12/ |

One job, one command, no shell pipe. `run_flow.py` runs `quantize_eval.py` and then `flow_fixed_prompts.py` and stops at the first non-zero exit code. Everything after `--qe-args` goes to `quantize_eval.py`.

## Where each setting comes from

| Setting | Value in this job | Source |
|---|---|---|
| Checkpoint | `./inputs/dryrun2-1234/model.safetensors`, step 1,221 | TSAI's dry run 2 result (G12 channel, 27 Nov): job `dryrun2-1234`, final_model_sha256 `8d1600b278a3d8e15c3d7af408a86e7c47acd987a899bf5eeee6b2c72192eb47`. The checkpoint is in Team 7's store, `/shared/ckpt/dryrun2-1234`. ASSUMPTION on the staging, see below. |
| Model settings | `./inputs/dryrun2-1234/manifest.json` (`config.model`), passed with `--spec` | the run's `./out/dryrun2-1234/manifest.json`. A bare state_dict has no settings beside it, so `export_hf.py` needs the file. |
| Router bias | dropped (`--drop-router-bias`); A3 and the second A1 compare with G5 at router_bias zero | GraniteMoeShared has no selection bias (#762). The exact route is the patched llama.cpp, not ready yet. |
| KAS head bias | dropped (`--drop-head-bias`); A1 and A3 compare with the folded head without it, and its effect is measured | GraniteMoeShared has no output bias (#780). |
| Formats | bf16, Q8_0, Q6_K, Q4_K_M | `docs/bitwidth_proposal.md` v2 (#853): serve Q8_0, Q6_K the candidate "confirm on the lab's own model", Q4_K_M the format that failed on all four public models. Q5_K_M is left out (not to ship). |
| A1 prompts | 16 per A1 (two A1 runs) | `--verify 16`. The default 64 would take twice as long on CPU for a 300M model with two A1 runs; 16 prompts x 64 tokens is 1,024 positions each. |
| Router and output layer | router F32, output layer Q8_0 | `quantize_eval.py` defaults. |
| Items for the score | `prompts/eval_own_20.json`, sha256 `5d4fec16221c3d2dd1aa2695f6f6ded97d5a493afd460bf7f927e19e36fbb069` | same as dry run 1, so the two runs compare. |
| 100 prompts | Team 10's `eval/effort_manifest.json` (30 code and 25 math slots) plus 45 of ours | as in dry run 1. |
| Decoding | greedy (temperature 0, top_k 1), seed 0, 128 new tokens, context 2,048 | `flow_fixed_prompts.py` defaults. |
| llama.cpp | b11669 binaries and checkout | #467, as in dry run 1. |

## Source and staging

- Start from `theschoolofai/era-v5-lab` main with the PR that carries this file merged. It adds `--formats` to `run_flow.py`, the second A1 to `export_hf.py`, and both A1 results and the bias statistics to `quantize_eval.py`'s scores.json.
- Checkpoint: the model weights at step 1,221 from `/shared/ckpt/dryrun2-1234`, written as `./inputs/dryrun2-1234/model.safetensors` with Team 7's `ckpt.export_safetensors` (the same call Team 9's `harness_lab/run_lab_ckpt.py` makes; it checks every key). A `checkpoint.pt` with `{"model": state_dict}` works too: then pass `--checkpoint ./inputs/dryrun2-1234/checkpoint.pt`. Copy `./out/dryrun2-1234/manifest.json` beside it.
- Tokenizer, Python, llama.cpp binaries, the `./work/` folder (`llama.cpp-b11669/`, `.venv-convert/`) and Team 10's parquet files: as in `2026-11-18-dryrun1-quantize-eval.md`, "Source and staging". The dry run 1 staging can be reused as it is. Please start from a `./work/` without dry run 1's `*-bf16.gguf` or `*-scores.json`, so the prompt step finds only this run's files.
- With these in place the run makes no network call.

## How the run is scored and reported

- As in dry run 1: 20 own items for the score (a check that the files work and move little from bf16, not a benchmark), and Team 10's 55 slots and our 45 reported apart, with no prompt or output text for Team 10's slots.
- The two biases are set aside in every GGUF, so every format is compared with bf16 of the same export. The bf16 number is not the checkpoint's own quality: the first A1 and `head_bias_effect` say how far the export is from it on random prompts. What the biases cost on real text comes from two other reports, attached next to this result, not from this job: Team 10's `router_bias_reach.py` in the dry run 2 chain and G12's `ollama/measure_export_gaps.py` (#834).
- One checkpoint, one seed, greedy decoding. No winner is named and no setting is changed.

## What I ran before filing

On an Apple laptop (CPU), with a random-weight model in dry run 2's shape from `make_random_checkpoint.py --head-mode kas_v2 --bias-gamma 0.001 --router-bias-std 0.01` (2 layers, width 128, 10.8M parameters, vocabulary 131,072), saved as a bare `model.safetensors` with its `manifest.json` beside it, as the staging above:

| Step | Result |
|---|---|
| `export_hf.py` without the two flags | refused: `blocks.1.moe.router_bias is not zero (largest absolute value 0.0236, bias_gamma 0.001)` |
| `export_hf.py --verify 16 --drop-router-bias --drop-head-bias` | exit 0. The export itself takes 7 s; the two A1 runs take the rest (95 s in all with `--verify 2`, 15.7 min with `--verify 16` on a busy laptop) |
| A1 against G5 as trained | max logit difference 6.3, same top-1 on 864 of 1,024 positions: FAIL, as expected with the router bias dropped |
| A1 against G5 with router_bias zeroed | max logit difference 2.1e-05 (limit 1e-3), same top-1 on 1,024 of 1,024: PASS |
| `test_export_hf.py`, `test_run_flow.py` | 40 and 5 tests pass |

Not run here: the llama.cpp steps (A3, quantize, scores, prompts). They are the dry run 1 flow, unchanged, which ran end to end in that smoke (`evidence/2026-11-18-dryrun1-quantize-eval/smoke.md`), and A3 on a dropped-both KAS V2 GGUF passed in #780's smoke (16 of 16 prompts, 0.017). The random weights make every A1 number a wiring check, not a prediction for the trained model. I could not test the real checkpoint, a GPU, or the p1536 tokenizer on this machine.

## Open items for TSAI

1. **Checkpoint.** Can the step 1,221 weights be staged as `model.safetensors` (Team 7's `export_safetensors`) or as a `checkpoint.pt` with a `model` key, plus `manifest.json`? If the path differs, I change `--checkpoint` and `--spec`.
2. **Time.** The two A1 runs are on CPU and take most of the export time (see the smoke). 90 minutes is an estimate for a model about 30 times the stand-in's size. If it runs long, `--verify 8` halves the A1 time.
3. **Work folder.** As in dry run 1, `./work/` sits outside `./out/` (llama.cpp checkout, converter venv, the exported model, the GGUFs). Nothing in it needs copying: `scores.json` has the export results.
4. **Dry run 1.** The dry run 1 job (22 Nov, "running it now") has not posted results yet. If it finished, its `runs.csv` and `scores.json` let us put the two runs side by side.
