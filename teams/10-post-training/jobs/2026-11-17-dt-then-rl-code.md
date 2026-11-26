# Job request draft — dry run 2 DT SFT followed by code RL

Status as of 24 November 2026. This canary uses the DT arm exported by the [dry run 2 SFT comparison](2026-11-26-g10-dryrun2-dt-vs-fft.md). The run needs the delivered dry run 2 checkpoint and its DT SFT output, staged pool files, and lead approval. Code-verifier reward before and after RL remains unmeasured on dry run 2.

```text
Team:            10-post-training
Owner:           Hema Varshini Palpandian (Hemaaaa04)
Lead approval:   pending saitej123 for this DT-then-RL canary
Question:        Can the dry run 2 DT SFT checkpoint take code-verifier RL updates with its deleted weights held at zero, and does held-out reward rise?
Script:          teams/10-post-training/capstone/rl_job_v1.py (calls rl-loop/train.py)
Command:         G10_STDIO_RUNNER=docker G10_STDIO_SECRET_PATHS=<POOL>/out/v1/cc.jsonl python teams/10-post-training/capstone/rl_job_v1.py --g5-root <G5> --run-folder <RUN> --tokenizer <TOKENIZER> --byte-table <TABLE> --probe-file <PROBE> --origin g5_delivery --expected-head-mode kas_v2 --expected-bias-gamma 0.001 --rl-tasks <POOL>/out/v1/cc.jsonl --train-pool <POOL>/pool/rl-code-v1-train.pool.jsonl --heldout-pool <POOL>/pool/rl-code-v1-heldout.pool.jsonl --sft-arm none --start-weights keep_canary/dt/model.safetensors --keep-mask keep_canary/dt/dt_keep_masks.bin --sampler g5_batch --sampler-max-rows 128 --micro-rows 1 --update-mode auto --time-sync --updates 3 --state-every 1 --screen-n 64 --min-mixed 4 --prompts-per-step 16 --k 8 --max-new 1536 --heldout-n 64 --eval-k 4 --lr 1e-5 --seed 1 --deadline-minutes 25 --device cuda --out out/task11-code-canary
Config:          rl-loop/config/task11_code_canary.yaml is the trainer template. The wrapper writes the staged config to out/task11-code-canary/rl_config.yaml with actual artifact hashes and deadline.
Inputs:          dryrun2-1234 run folder or Team 7 store reconstruction, p1536 tokenizer and KE2 table bound to its manifest, validated probe, keep_canary/dt/{model.safetensors,dt_keep_masks.bin,dt_keep_masks.bin.json,SHA256SUMS}, pinned RL data v1 cc.jsonl and train/held-out pool files
Hardware:        1 GPU, preloaded Python verifier image; use the delivered run's GPU class when rebuilding the run folder from the store
Max time:        Proposed 30 minutes; wrapper deadline 25 minutes leaves time for final evaluation and reporting. TSAI must enforce the outer limit on the process group.
Precision:       bf16 compute and fp32 master weights, g5_batch native sampler, one row per log-prob/update forward
Measures:        Input hashes and exclusion counts; mixed groups; step-0/final held-out reward and paired prompt bootstrap; wrong-answer and literal-print counts; stored optimizer updates; frozen router and router_bias checks; keep_zero deletion counts; phase seconds, mask-check seconds, mask bytes, peak GPU memory
Expected result: Unknown code reward and mixed groups. A useful update run has a positive initial deletion count, at least one stored optimizer update, zero deleted-weight regrowth, and unchanged router state.
Outputs:         out/task11-code-canary/{pins,pool,plan,screen_survey,report,summary}.json as available, rl_config.yaml and rl/{evals,metrics}.jsonl plus summary.json. Keep paused rl_state/ private for resume. Return aggregate reports; task text, generations and model weights stay private.
Test log:        results/dt_then_rl_evidence/week4_validation.md covers local integration tests. Before sending, attach the exact fully staged command, final output lines, exit status, wall time, and hardware from an end-to-end run of this wrapper.
```

## Dry run 2 handoff

1. Run the preceding SFT comparison with `--keep-canary-state-dir keep_canary/`. Use its `dt/` export with all four files together. The wrapper verifies the checkpoint, mask binary and metadata against `SHA256SUMS` and sends those pins to the trainer. Keep the export read-only until RL finishes. A separate manifest can be passed with `--canary-sha256s`.
2. Use the run folder and probe produced by `capstone/dryrun2_job_chain.sh`. The staged adapter must resolve dry run 2 settings `head_mode: kas_v2` (KE2/KAS `a10_kas_tokdrop`) and `moe.bias_gamma: 0.001`, with p1536 and table hashes from the actual manifest. Team 5 #717 and G10 integration #778 have merged. The two expected-model flags refuse mismatched head or router-bias settings before model work.
3. Pool staging follows the existing RL job pins. The committed 66-ID free-reward list has SHA-256 `c97568952d2374be473e747184ea2b23098601797c936aad35363d8e0539aa2e`. The wrapper excludes those IDs and the 55 effort-suite IDs from training, checks train/held-out overlap, and keeps the first 64 held-out pool IDs in file order. It does not choose held-out prompts based on model scores. The current pinned pools give 3,199 training IDs after exclusions.
4. The wrapper verifies a correct answer, wrong answer and secret-read attack before loading the model. It records a no-skill baseline, checks the time budget, and screens 64 training prompts with eight samples each. Fewer than four mixed groups yields `NO_GO_MIXED_BAND`; an insufficient budget gives a screen-only result. Neither outcome establishes a learning gain.
   Two of the 64 screened ids can never score: `cc-hackerearth-cypher` (index 13) and `cc-hackerearth-generating_sequence_5` (index 16). Each has a test whose expected output, written with one space between tokens, is longer than the scorer's 65,536-character cap per test (`rl-data/stdio_check.py` `MAX_OUTPUT_CHARS`; 75,879 and 92,956 characters), so even a correct program is cut short and fails. In the pinned pool (`cc.jsonl` SHA-256 `91ee9951...`) 15 tasks are like this: 14 of the 3,199 training ids and one held-out id (`cc-aizu-p01246_Land_Mark`, index 201 of 255, outside the first 64 this canary evaluates). The pins stay as they are. Read the screen as at most 62 possible mixed groups out of 64, and keep the held-out item in the denominator of any evaluation that includes it.
5. `g5_batch`, `micro_rows: 1`, and the identical-weight log-probability gate use the existing Week 4 path. `update_mode: auto` resolves KAS dropout behavior and is recorded in the report. The DT mask stays baked; RL trains surviving weights with AdamW and keeps router weights and bias fixed. `keep_zero` checks run at every step. The wrapper reports deletion integrity alongside the held-out change.
6. After the canary, the lead chooses the full-run budget. Use the same held-out split and sampler settings for that run's before/after comparison, record the completed update count and confidence interval, and fill `results/dt_then_rl.md` from returned logs. Three steps validate the handoff; the reward gain can be positive, flat or negative.

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
