# Results

Measured outputs and reports from G10 runs. Each file says how it was produced and what it does not show.

**Effort evaluation** (`eval/`)

- `effort_probe.md`: the suite, the manifest and the CPU checks, with the Stage A section.
- `effort_stageA_olmoe_2026-11-12/`, `effort_stageB_olmoe_2026-11-12/`, `effort_stageB_qwen_2026-11-12/`: per-cell records, per-sample summaries, the pooled numbers and (for Stage B) `report.md`. All records are `do_not_train`.
- `attainability_survey_code2048.md`: the code arm re-run at a 2048-token cap.

**SWE environments** (`swe-envs/`)

- `swe_gold_check.md`, `swe_gold_check_amd64.md`, `swe_gold_reproduction.md`: gold checks of the environments.
- `swe_env_review.md`: the quality review against the SWE-bench Verified criteria.
- `swe_rebench_pilot.md`: admission results of the SWE-rebench pilot.

**End-to-end SWE tasks** (`swe-task/`, `swe-task-go/`)

- `swe_e2e.md`, `go_swe_e2e.md`, `hindi_swe_e2e.md`, `kannada_swe_e2e.md`, `tamil_swe_e2e.md`.

**RL rewards and tokenizer**

- `reward_pool_bench.md`: rollouts per second with and without the warm pools (`rl/`).
- `render_frozen_tokenizer.txt`: the report from `chat-format/verify_frozen_tokenizer.py` on Team 3's frozen tokenizer.
