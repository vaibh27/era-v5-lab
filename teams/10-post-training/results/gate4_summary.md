# Team 10 Gate 4 Evidence Summary (live status as of Fri 27 Nov 2026)

Status as of Fri 27 Nov 2026 (Week 4 D6; Gate 4 evidence due tonight Fri 27 Nov, 11:59 PM IST; Go/No-Go Sat 28 Nov). All paths are under `teams/10-post-training/` unless noted. Public text names teams and GitHub handles only. Nothing here claims dry run 2 post-training numbers that have not yet run.

## 0. Dry Run 2 Pretraining Checkpoint Delivered (`dryrun2-1234`, 27 Nov)

TSAI reported `dryrun2-1234` (eager arm) **COMPLETED**: `1,221 / 1,221` steps (`79.8 min` inside the `90 min` cap, `0` NaN/inf, loss `10.27 -> 5.25`, `heldout_v1` `val/bpb` `2.1607` at step 0 -> `2.0561` at step 1,221 on `549,042` scored tokens; `final_model_sha256: 8d1600b278a3d8e15c3d7af408a86e7c47acd987a899bf5eeee6b2c72192eb47`, checkpoint saved at step `1,221`). Note TSAI's two caveats: single seed (`1234`, eval moves ~`0.1` between checks) and trained on one 25M-token English web shard rather than `sample-v0`.

## 1. Gate 4 / Gate 3 Item 7 Required Evidence (SFT & RL Method Decision)

Per TSAI rulings (21 Nov 2026), closing Gate 3 item 7 requires **both** results on the **dry run 2 checkpoint**:

| Required Evidence | Status (27 Nov) | Code / Job Ready | Blocking External Dependency |
|---|---|---|---|
| **(1) Matched-step DT vs LoRA vs Full FT on dry run 2** | **Pre-run ready & 27 Nov lead-approved (0% GPU run)** | `capstone/dryrun2_dt_compare.py`, `capstone/dryrun2_job_chain.sh` (with step 6 `forget_bpb_kept.py --dtype auto`), `capstone/g5_source_resolver.py`, `jobs/2026-11-26-g10-dryrun2-dt-vs-fft.md` (`--slice-arm a4`, `125` min) + dense, KAS V2 + Team 7 store (`2026-11-26-g10-dryrun2-dt-vs-fft-kas.log`), and 27 Nov pre-run rehearsal (`2026-11-27-g10-dryrun2-prerun-rehearsal.log`) logs. Target: [`results/dt_vs_fft.md` §8](dt_vs_fft.md#8-dry-run-2-capstone-comparison-full-ft-vs-lora-vs-deletion-tuning). | TSAI staging of `<T5_OUT>` / `/shared/ckpt/dryrun2-1234`, `#565` shards, and `#564` pool to run the 2h+ job. |
| **(2) DT-then-RL chain on dry run 2 DT checkpoint** | **Pre-run ready & 27 Nov lead-approved (0% GPU run)** | `jobs/2026-11-17-dt-then-rl-code.md` (`--keep-weights-dir keep_t7/`, preserving `rl_state/` on failed save), `rl-loop/config/task11_code_canary.yaml`, `capstone/rl_job_v1.py` (`--sampler g5_batch`, `micro_rows: 1`, `completion_logprobs`), `jobs/2026-11-26-dt-then-rl-staged.log`, and `jobs/2026-11-27-g10-dryrun2-prerun-rehearsal.log` (saves + reads `g10-rl-final`). Target: [`results/dt_then_rl.md`](dt_then_rl.md#dry-run-2-status--27-november-2026). | `keep_canary/dt/` (`model.safetensors`, `dt_keep_masks.bin`, `dt_keep_masks.bin.json`, `SHA256SUMS`) from job (1) above + staged dry run 2 inputs. |
| **(3) Held-out forgetting & Team 9 watcher verdicts on kept arms** | **Tooling & Team 9 fixes ready (0% GPU run)** | `capstone/forget_bpb_kept.py` (step 6 of `dryrun2_job_chain.sh`), `capstone/watcher_rows_to_report.py` (`primary: true` selector for `acc`/`acc_norm` tasks), `capstone/keep_weights.py` (`config.model` reader), `sft/tests/forgetting.py`. | Kept Team 7 run folders `keep_t7/g10-dr2-{full,lora,dt}` from job (1) + Team 9 `eval_mini_v1` ARC-Challenge/HellaSwag rows in `bfloat16`. |

## 2. Kept Checkpoints Handed to Team 9 (`eval_schedule.md` #7)

Once jobs (1) and (2) finish, Team 10 hands over these four Team 7 Checkpointer folders for Team 9 `bfloat16` Tier B scoring (`run_lab_ckpt.py --team5-run`, keyed by each folder's `MANIFEST.json` SHA-256 and the final step recorded in `summary.json`):

| Folder | Role | Precision | Step & Manifest SHA-256 | Forgetting Verdict (`forgetting.py`) |
|---|---|---|---|---|
| `keep_t7/g10-dr2-full` | Dry run 2 Full FT SFT arm | `bfloat16` autocast | *Pending job (1)* | *Pending `forget_bpb_kept.py` + Team 9 rows* |
| `keep_t7/g10-dr2-lora` | Dry run 2 LoRA SFT arm | `bfloat16` autocast | *Pending job (1)* | *Pending `forget_bpb_kept.py` + Team 9 rows* |
| `keep_t7/g10-dr2-dt` | Dry run 2 Deletion Tuning SFT arm | `bfloat16` autocast | *Pending job (1)* | *Pending `forget_bpb_kept.py` + Team 9 rows* |
| `g10-rl-final` | Dry run 2 DT-then-RL final checkpoint | `bfloat16` autocast | *Pending job (2)* | *Pending Team 9 Tier B score* |

## 3. Independent & Supporting Week 4 Evidence Landed (`#746`, `#778`, `#797`, `#835`, `#836` merged)

| Deliverable | Evidence Level | Paths | Key Numbers / Result |
|---|---|---|---|
| **KE2/KAS V2 + `router_bias` SFT/DT/RL stack** | Unit + Team 5 `#717`/`#748` + CPU stand-in | `rl-loop/team5_adapter.py`, `capstone/sft_g5.py`, `capstone/keep_weights.py`, `capstone/g5_source_resolver.py`, `dt/mask.py` | Untied `a10_kas_tokdrop` head, `router_bias_unchanged: true`, Wanda calibration & `RouteRecorder` routing with `router_bias`, layer-by-layer DT regulariser backward, and auto-checkout of exact Team 5 manifest commit. |
| **Batched G5 RL sampler (`g5_batch`) & memory fixes** | Measured (Colab T4, fp16, T2/T3) + unit | `rl-loop/g5_decode.py`, `results/b5_sampler_2026-11-23.md`, `capstone/rl_job_v1.py`, `sft/bench/train.py` | **12.8x speedup** on T4 (`44.3 s -> 3.45 s` per prompt); `micro_rows: 1`; `completion_logprobs` on completion tokens only; `cache_enabled=False` autocast frees BF16 weight copy. |
| **Team 7 Checkpointer <-> Team 9 watcher bridge** | Unit + 27 Nov pre-run rehearsal | `capstone/run_folder_from_store.py`, `capstone/forget_bpb_kept.py`, `capstone/watcher_rows_to_report.py`, `capstone/keep_weights.py`, `sft/tests/forgetting.py` | Rebuilds run folders from `#608` stores; `forget_bpb_kept.py` scores held-out bpb in Team 9's dtype; `watcher_rows_to_report.py` reads `primary: true` rows; `keep_weights.py` reads `config.model` from `manifest.json`; `g10-rl-final` export + resume verified. |
| **7.65B MoE Deletion Tuning study (`Phi-mini-MoE-instruct`)** | Independent public-model study (non-capstone, 1 seed; **not Gate 4 capstone evidence**) | `results/dt_phi_moe.md` | All 1,319 GSM8K (chat): `55.95%` base -> `67.10%` DT (`0.6108%` deleted), `74.98%` LoRA (`lr 2e-4`), `54.36%` Full FT (`lr 1e-5`), `21.83%` Full FT (`lr 3e-5`, `15.9%` HumanEval on 164 problems; not matched to DT at either rate; see HumanEval decode caveat). 4-task prior mean: `64.30 -> 64.46` DT, `64.56` LoRA, `63.68` Full FT (`1e-5`), `58.46` Full FT (`3e-5`). |
| **4B / 8B Qwen3-Base DT vs LoRA vs Full FT ladder job** | Lead-approved job request (awaits TSAI GPU run) | `jobs/2026-11-23-g10-dt-large-model.md`, `.log`, `dt/run_ladder.py`, `dt/ladder_qwen3.yaml` | Pure DT, DT+markers, LoRA, Full FT, and low-LR Full FT at 4B and 8B on H200 (141 GB). |
| **Indic RL human templates (6/6 signed off, `train`)** | Decontam 0/1,200 flagged; Team 2 OK'd on `#781` | `rl/indic/templates/indic_templates.yaml`, `rl/indic_prompts_v0.md` | 2 Tamil (`ta_appalam_sub`, `ta_pen_mult`, `team2_reviewer: truesrini`) + 4 Hindi (`hi_f1_left`, `hi_f1_change`, `hi_f6_minutes`, `hi_f3_weekday`, `team2_reviewer: riteshverma`); all 6 `eligibility: train`. |
| **Verifier hardening, template parity & licence hygiene** | Merged (`#778`, `#797`, `#836`) + Docker/local verified | `rl/js/`, `rl/jsts/`, `rl/math/`, `rl/toolresult/`, `rl/test_tool_call_hardening.py`, `rl/agentic/`, `rl-data/test_stdio_hardening.py`, `tests/test_chat_template_parity.py`, `eval/results/api_bank_nemotron_overlap.md`, `licences/licence_table.md` | 21 Nov exclusions applied (`sft-indic-nemotron-personas-india`, `rl-math-hendrycks`); API-Bank overlap mask enforced; Team 3 `release/tokenizer_config.json` byte-parity + multi-turn/Indic/forged-tool-result tests passing; all 5 `rl/agentic/` tasks (`94 passed`, `10/10` `reward.py --check-all --type agentic`) and `test_stdio_hardening.py` (`13 passed`) verified end-to-end in Docker. |

## 4. Cross-Team Handoffs (Week 4)

| Partner Team | Item | Status (27 Nov) | Next Step |
|---|---|---|---|
| **Team 5 (Architecture) / TSAI** | `dryrun2-1234` checkpoint (`1,221` steps, sha256 `8d1600b2...`) | Dry run 2 completed (`27 Nov`); `#717`/`#720`/`#748` merged on `main`; G10 jobs lead-approved | TSAI stages `<T5_OUT>` / `/shared/ckpt/dryrun2-1234` + `#565`/`#564` inputs and launches `jobs/2026-11-26-g10-dryrun2-dt-vs-fft.md` followed by `jobs/2026-11-17-dt-then-rl-code.md`. |
| **Team 7 (Infrastructure)** | Checkpointer wiring (both ways) | **Ready** (`#608` store read/write verified in `run_folder_from_store.py`, `keep_weights.py`, `forget_bpb_kept.py`) | Export `keep_t7/g10-dr2-{full,lora,dt}` and `g10-rl-final` during jobs (1)–(2). |
| **Team 9 (Evaluation)** | `eval_schedule.md` item #7 | **All 3 Team 9 fixes landed (`#176`)** (`--keep-weights-dir keep_t7/`, `primary: true` selector, `config.model` check) | Hand over kept folders + `summary.json` step / `MANIFEST.json` SHA-256 after jobs (1)–(2); score `forgetting.py` verdicts. |
| **Team 2 (Data B)** | Native-reader sign-off on 6 human Indic RL templates | **Done (`#781`)** (`truesrini` Tamil 26 Nov; `riteshverma` Hindi 26–27 Nov) | All 6 templates `eligibility: train`. |

## 5. Remaining Open Items Before Gate 4 (Fri 27 Nov, 11:59 PM IST)

1. **Dry run 2 GPU runs & forgetting verdicts (Gate 4 required):** TSAI launches `jobs/2026-11-26-g10-dryrun2-dt-vs-fft.md` and `jobs/2026-11-17-dt-then-rl-code.md` on `dryrun2-1234` (`8d1600b2...`); run `watcher_rows_to_report.py` + `forgetting.py` once Team 9's rows return; populate `results/dt_vs_fft.md` §8, `results/dt_then_rl.md`, and section 2 above.
2. **Queued TSAI logs (`#565`, `#587`, `#133`, `#142`):** record returned counts/numbers into their job notes once TSAI posts logs.
3. **G11 Release `#705` sign-off:** reply `OK @saitej123` (or updated Team 10 wording) on [PR #705](https://github.com/theschoolofai/era-v5-lab/pull/705).
