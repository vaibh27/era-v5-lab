# Job request — dry run 2 DT SFT followed by code RL

Status as of 27 November 2026: dry run 2 has completed. Lead-approved (`saitej123`, 27 Nov 2026), including the 27 November changes (`--keep-weights-dir keep_t7/` for `g10-rl-final`, keeping `rl_state/` if that save fails, and `jobs/2026-11-27-g10-dryrun2-prerun-rehearsal.log`) and the [staged local wrapper log](2026-11-26-dt-then-rl-staged.log). This canary uses the DT arm exported by the [dry run 2 SFT comparison](2026-11-26-g10-dryrun2-dt-vs-fft.md). Execution now waits only on that kept DT export (`keep_canary/dt/`) and its delivered run folder and staged pool/probe inputs. Code-verifier reward before and after RL remains unmeasured on dry run 2.

```text
Team:            10-post-training
Owner:           Hema Varshini Palpandian (Hemaaaa04)
Lead approval:   saitej123 (26 Nov canary + 27 Nov updates: --keep-weights-dir keep_t7/, keeping rl_state/ after failed save, and jobs/2026-11-27-g10-dryrun2-prerun-rehearsal.log approved 27 Nov 2026; runnable once keep_canary/dt/ and dry run 2 run folder are staged)
Question:        Can the dry run 2 DT SFT checkpoint take code-verifier RL updates with its deleted weights held at zero, and does held-out reward rise?
Script:          teams/10-post-training/capstone/rl_job_v1.py (calls rl-loop/train.py)
Command:         G10_STDIO_RUNNER=docker G10_STDIO_SECRET_PATHS=<POOL>/out/v1/cc.jsonl python teams/10-post-training/capstone/rl_job_v1.py --g5-root <G5> --run-folder <RUN> --tokenizer <TOKENIZER> --byte-table <TABLE> --probe-file <PROBE> --origin g5_delivery --expected-head-mode kas_v2 --expected-bias-gamma 0.001 --rl-tasks <POOL>/out/v1/cc.jsonl --train-pool <POOL>/pool/rl-code-v1-train.pool.jsonl --heldout-pool <POOL>/pool/rl-code-v1-heldout.pool.jsonl --sft-arm none --start-weights keep_canary/dt/model.safetensors --keep-mask keep_canary/dt/dt_keep_masks.bin --sampler g5_batch --sampler-max-rows 128 --micro-rows 1 --update-mode auto --time-sync --updates 3 --state-every 1 --screen-n 64 --min-mixed 4 --prompts-per-step 16 --k 8 --max-new 1536 --heldout-n 64 --eval-k 4 --lr 1e-5 --seed 1 --deadline-minutes 25 --device cuda --keep-weights-dir keep_t7/ --out out/task11-code-canary
Config:          rl-loop/config/task11_code_canary.yaml is the trainer template. The wrapper writes the staged config to out/task11-code-canary/rl_config.yaml with actual artifact hashes and deadline.
Inputs:          dryrun2-1234 run folder or Team 7 store reconstruction, p1536 tokenizer and KE2 table bound to its manifest, validated probe, keep_canary/dt/{model.safetensors,dt_keep_masks.bin,dt_keep_masks.bin.json,SHA256SUMS}, pinned RL data v1 cc.jsonl and train/held-out pool files
Hardware:        1 GPU, preloaded Python verifier image; use the delivered run's GPU class when rebuilding the run folder from the store
Max time:        Proposed 30 minutes; wrapper deadline 25 minutes leaves time for final evaluation and reporting. TSAI must enforce the outer limit on the process group.
Precision:       bf16 compute and fp32 master weights, g5_batch native sampler, one row per log-prob/update forward
Measures:        Input hashes and exclusion counts; mixed groups; step-0/final held-out reward and paired prompt bootstrap; wrong-answer and literal-print counts; stored optimizer updates; frozen router and router_bias checks; keep_zero deletion counts; phase seconds, mask-check seconds, mask bytes, peak GPU memory
Expected result: Unknown code reward and mixed groups. A useful update run has a positive initial deletion count, at least one stored optimizer update, zero deleted-weight regrowth, and unchanged router state.
Outputs:         out/task11-code-canary/{pins,pool,plan,screen_survey,report,summary}.json as available, rl_config.yaml and rl/{evals,metrics}.jsonl plus summary.json. Keep paused rl_state/ private for resume. Return aggregate reports; task text, generations and model weights stay private. keep_t7/g10-rl-final/ (Team 7 Checkpointer run folder of the weights after RL, trained tensors in bf16) stays on TSAI storage for Team 9; it is not written when the status is NO_GO_MIXED_BAND or PAUSED_AT_DEADLINE (no completed canary: the DT arm's keep_t7/g10-dr2-dt is then the last kept weights). If that save fails, the rl step fails and rl_state/ is not deleted: free the cause and rerun the same command with the same --out, which resumes the finished state without new updates and saves again. Keep rl_state/ private.
Test log:        jobs/2026-11-26-dt-then-rl-staged.log — actual rl_job_v1.py end to end on a reduced KAS V2 CPU fixture, including exact commands, pins, outputs, exit status and timing. Real Docker verifier: exit 0, 10.409 s, NO_GO_MIXED_BAND. Separate scripted-reward integrity run: exit 0, 11.922 s, three updates and zero regrowth. Production-scale GPU measurements pending. 27 Nov: `jobs/2026-11-27-g10-dryrun2-prerun-rehearsal.log` section 2 runs this file's Command line with `--keep-weights-dir` on a CPU stand-in with a seeded reward: exit 0, three updates, no deleted weight non-zero, `g10-rl-final` written and read by Team 9's pointer functions. Section 4: with the final save blocked the step failed and kept `rl_state/`; the same command run again resumed at step 3 and saved weights equal to the unblocked run's, no deleted weight non-zero.
```

## Dry run 2 handoff

1. Run the preceding SFT comparison with `--keep-canary-state-dir keep_canary/`. Use its `dt/` export with all four files together. The wrapper verifies the checkpoint, mask binary and metadata against `SHA256SUMS` and sends those pins to the trainer. Keep the export read-only until RL finishes. A separate manifest can be passed with `--canary-sha256s`.
2. Use the run folder and probe produced by `capstone/dryrun2_job_chain.sh`. The staged adapter must resolve dry run 2 settings `head_mode: kas_v2` (KE2/KAS `a10_kas_tokdrop`) and `moe.bias_gamma: 0.001`, with p1536 and table hashes from the actual manifest. Team 5 #717 and G10 integration #778 have merged. The two expected-model flags refuse mismatched head or router-bias settings before model work.
3. Pool staging follows the existing RL job pins. The committed 66-ID free-reward list has SHA-256 `c97568952d2374be473e747184ea2b23098601797c936aad35363d8e0539aa2e`. The wrapper excludes those IDs and the 55 effort-suite IDs from training, checks train/held-out overlap, and keeps the first 64 held-out pool IDs in file order. It does not choose held-out prompts based on model scores. The current pinned pools give 3,199 training IDs after exclusions.
4. The wrapper verifies a correct answer, wrong answer and secret-read attack before loading the model. It records a no-skill baseline, checks the time budget, and screens 64 training prompts with eight samples each. Fewer than four mixed groups yields `NO_GO_MIXED_BAND`; an insufficient budget gives a screen-only result. Neither outcome establishes a learning gain.
   Two of the 64 screened ids can never score: `cc-hackerearth-cypher` (index 13) and `cc-hackerearth-generating_sequence_5` (index 16). Each has a test whose expected output, written with one space between tokens, is longer than the scorer's 65,536-character cap per test (`rl-data/stdio_check.py` `MAX_OUTPUT_CHARS`; 75,879 and 92,956 characters), so even a correct program is cut short and fails. In the pinned pool (`cc.jsonl` SHA-256 `91ee9951...`) 15 tasks are like this: 14 of the 3,199 training ids and one held-out id (`cc-aizu-p01246_Land_Mark`, index 201 of 255, outside the first 64 this canary evaluates). The pins stay as they are. Read the screen as at most 62 possible mixed groups out of 64, and keep the held-out item in the denominator of any evaluation that includes it.
5. `g5_batch`, `micro_rows: 1`, and the identical-weight log-probability gate use the existing Week 4 path. `update_mode: auto` resolves KAS dropout behavior and is recorded in the report. The DT mask stays baked; RL trains surviving weights with AdamW and keeps router weights and bias fixed. `keep_zero` checks run at every step. The wrapper reports deletion integrity alongside the held-out change.
6. After the canary, the lead chooses the full-run budget. Use the same held-out split and sampler settings for that run's before/after comparison, record the completed update count and confidence interval, and fill `results/dt_then_rl.md` from returned logs. Three steps validate the handoff; the reward gain can be positive, flat or negative.

## Local wrapper test — 26 November 2026

The wrapper ran from build `2e1e2169` on macOS arm64, CPU fp32, Python 3.12.14 and PyTorch 2.14.1, with Docker 29.9.2.
Team 5's unmodified trainer made a labelled rehearsal checkpoint inheriting dryrun2's KAS V2 head and router-bias configuration.
One actual DT SFT step produced the baked weights and mask, with reload parity passing and 283 deleted entries.
The [log](2026-11-26-dt-then-rl-staged.log) gives the three export hashes and all staged commands.

| Wrapper run | Exit | Wall time | Observed outcome |
| --- | ---: | ---: | --- |
| Real Docker code verifier | 0 | 10.409 s | `NO_GO_MIXED_BAND`, 0 of 2 train prompts mixed, no RL updates |
| Built-in `--stand-in-reward 0.4` fixture | 0 | 11.922 s | Three stored optimizer updates, 283 deleted entries held at zero, router weights and bias fixed |

Both runs used `g5_batch`, the expected KAS/bias flags, checksum-pinned DT files, automatic update mode resolving to
`eval`, and the identical-weight log-probability gate. The reduced geometry was one layer, width 16, two routed experts
and one shared expert, context 128. Pretraining, SFT and code prompts were synthetic fixtures. The test-only
`--no-pack-check` and rehearsal flags allowed those fixture inputs; the production command above retains data pins and
the delivered-artifact checks. These runs verify the wrapper path and mask handoff. They provide no dry-run-2 code-learning
result, production pool validation, or GPU/bf16 cost measurement.

## Historical CPU rehearsal

The following 20 November notes describe the earlier dry-run-1 stand-in. The new dry run 2 results are pending.


## Update 20 Nov 2026 (f20201351h-maker): staged local test log on a stand-in (CPU)

This is not a result on the dry run 1 checkpoint. The model is a Team 5 week2 LanguageModel (2 layers, width 64, 10.6 million parameters, random init, 20 CPU smoke steps) and the canary's expected status on it is NO_GO_MIXED_BAND.

1. DT SFT state, from the job script itself (default settings of the DT flags; rows are the #587 slice settings reduced to 10 steps, batch 16, as in the #587 CPU log):
   `python capstone/sft_job_v1.py --g5-root <worktree> --run-folder <stand-in run> ... --device cpu --method dt_markers --keep-canary-state-dir <dir> --keep-arm a2 --out <out>`
   Result: exit 0, `{"ok": true, "steps": 5, "failed_steps": []}`, 331.0 s wall. `<dir>/a2/` holds `model.safetensors`, `dt_keep_masks.bin`, `dt_keep_masks.bin.json` and `SHA256SUMS` (`sha256sum -c` OK); summary.json records the same three sha256 values.
2. Canary config staged for CPU: the three sha pins filled from that folder, device cpu, local input paths, `allow_fixture_smoke: true`, max_new 48, eval_k 2, reward_pool 4, no autocast, 16 train ids and 6 held-out ids; every other line is the repo file's.
3. Canary run with the trainer's reward replaced by a seeded coin (p 0.4, dry run only, the same patch point as `rl_job_v1.py --stand-in-reward`): exit 0, 72.3 s, 3 optimizer updates, 106,420 deleted entries at the start, deleted-nonzero 0 at every step and at the end, `keep_zero_checks` 0.017 to 0.020 s per step, `persistent_mask_tensor_bytes` 8,355,840, stored-update share 0.930, peak memory footprint 2.8 GB (`/usr/bin/time -l`).
4. The same canary with the real stdio verifier (Docker) on an earlier stand-in (19 Nov, same config shape): exit 0, 22.5 s, `NO_GO_MIXED_BAND`, 0 of 16 prompts mixed, as expected for a random model.
Not shown by any of this: a pass rate or learning signal, real sizes or timings, GPU or bf16 behaviour, the admission and length check of the 16 prompts, the probe check of a delivered run folder.
Still blocked for the real canary: a DT SFT state made from dry run 1 (the job script can now make it, but its DT settings are the lead's and this task's owner's choice, and it needs the #587 result and a probe expectation for the #475 checkpoint; see `2026-11-20-g10-dt-sft-dryrun1.md`).
