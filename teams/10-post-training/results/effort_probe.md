# Effort evaluation (Package T2): suite, manifest and CPU checks

Date: 11 Nov 2026 (IST). Code: `eval/effort_suite.py`, `eval/effort_manifest.json`, `eval/test_effort_suite.py`.
**No model has generated anything for this suite yet.** This file reports the suite and the CPU checks only.
No GPU was used.

> 12 Nov update: the first model run, a **Stage A canary on OLMoE-1B-7B-0924-Instruct**, is in
> [Stage A: OLMoE-1B-7B-0924-Instruct canary](#stage-a-olmoe-1b-7b-0924-instruct-canary-12-nov). The 11 Nov text
> above and below is left as written; it describes the suite and the CPU checks of that date.

## What the suite measures

Each fixed prompt is crossed with the four effort labels `1k, 4k, 16k, 64k`. Every record keeps the
**requested effort** (the plain-text line `Reasoning effort: {label}`, chat-format v0) apart from the
**enforced generation cap** (`max_new_tokens`). A text label is not budget enforcement, so three things are
reported apart: does the output follow the format, does it stay inside the cap, is it correct.

| Stage | Prompts | Cells | Cap | When |
|---|---:|---:|---|---|
| A | 10 (5 code, 5 math; first slot of each code judge and math level, plus one more aizu) | 40 | 1024 for every label | default; cheap canary |
| B | all 55 resolved | 220 | 1024 for every label | isolates the effect of the label text at a fixed cap |
| C | 8 preregistered (4 code, math levels 2 to 5) | 32 | label budget + 1024 reply allowance, bounded by the real context | only with `--stage C` |

The preflight checks prompt + cap against the model's real context. In A/B a prompt that cannot fit 1024 new
tokens gets one lower common cap for all four labels, and the record says so. In C a ceiling below the label
budget cannot test that label, so the cell is recorded as unsupported, not run. Neither the common 1024 cap nor
the C subset shows full four-tier compliance; long-tier work waits for a TSAI job.

Per cell: functional success (where a checker exists), G10-format parse status, tool-call validity, generated
tokens, think-plus-tool trace tokens under the converter's contract (`sft/convert.py` TRACE: spans with their
markers), stop reason, truncation, wall time, prompt and total context tokens. Cells that hit the cap are
**censored** and stay in every denominator as failures. A malformed or unclosed trace is reported as a parser
failure with its raw length; it never gets a trace length of zero. The summary also gives matched per-prompt
changes against `1k` (same prompt, same seed), and `compare` matches two checkpoints on the same manifest hash.

Backends: `hf` runs an open checkpoint with its own chat template (effort line in a system message). That is an
optimization diagnostic only. `team5` runs the G5 model with the G10 template through `rl-loop/team5_adapter.py`
and refuses anything but a READY real bundle (a fixture runs only with `--allow-fixture-smoke`, flagged in the
summary). The final G10 effort comparison uses `team5`.

## The 100 slots

| Category | Declared | Resolved | Source | Licence / decision |
|---|---:|---:|---|---|
| code | 30 | 30 (aizu 8, atcoder 7, codechef 7, hackerearth 8) | deepmind/code_contests train @ `802411c3010cb00d1b05bad57ca77365a3c699d6` | CC-BY-4.0; `rl-code-code-contests` allowed (TSAI, 6 Nov) |
| math | 25 | 25 (5 per level, 1 to 5) | EleutherAI/hendrycks_math train @ `21a5633873b6a120296cce3e2df9d5550074f4a3` | MIT; **no row in `licences/licence_table.md` yet** |
| tool/agentic | 20 | 0 | none approved | unavailable, reason in the manifest |
| Indic | 15 | 0 | none approved | unavailable, reason in the manifest |
| general | 10 | 0 | none fetched | unavailable, reason in the manifest |

Both resolved sources are human-written (generator: none). No prompt was generated with any model. The prompt is
the raw task text; an instruction wrapper is used only if a team member supplies one (`--instruction kind=FILE`),
the same rule as `rl-data/survey.py`, and its sha256 goes into every record.

Why the three categories are empty:
- **tool/agentic**: the tool-use sets in the licence table (Nemotron-Agentic-v1, Nemotron-SFT-Agentic-v2) are
  model-simulated dialogues, not human-written. The 21 `swe-envs` are human and allowed, but they are RL
  environments that need a multi-turn Docker agent harness, and using them here needs a team decision and an RL
  hold-out.
- **Indic**: the human-written candidates (aya_dataset, oasst2) are `TSAI reserves`, not cleared, and not held
  locally. The allowed indic-align `Dolly_T` / `OpenAssistant_T` subsets are IndicTrans2 machine translations,
  not human-written Indic text.
- **general**: databricks-dolly-15k is allowed (CC-BY-SA-3.0, human-written) but no pinned copy was held locally
  and none was fetched. It is in the SFT mix, so its prompts would need an SFT hold-out.
Free-form Indic and general output also needs the team's blinded human rubric; the suite records success as
`None` for them and never uses a model judge.

Selection: candidates are the local RL v1 build of 11 Nov (code_contests non-Codeforces after decontamination v2;
MATH train after decontamination v2 and the MATH-500 13-gram guard). 1,805 ids used by the earlier OLMoE runs
(their train, held-out and survey id lists) were removed first, so that checkpoint is not scored on prompts it
trained or was filtered on. Within each stratum the pick is the lowest `sha256(selection_seed:id)`.

The manifest stores no prompt text. Each slot holds dataset, revision, file and its sha256, row index, field,
the sha256 of the text (and of the tests or reference answer), licence, decision, generator and an
original-problem group. `manifest_sha256` seals the file; a changed field fails `verify` and every run.

## Training exclusion

The 55 resolved tasks are all in the RL v1 candidate files, because that is the approved pool. **RL v1 must drop
them before any training while this suite is in use.** The ids and text hashes are in
`training_exclusion`; `effort_suite.py guard --training FILE` fails on an eval task id, an eval prompt, or any
eval output record (`do_not_train: true`). Run on the local RL v1 files it flags 55 of 55 and exits 1, as
expected until the drop is made.

## What ran (CPU, 11 Nov)

| Check | Result |
|---|---|
| `pytest teams/10-post-training/eval/test_effort_suite.py` | 30 passed |
| `effort_suite.py verify` with the pinned files and the rl-data checkers | 55 prompt texts, 30 test sets and 25 reference answers match their hashes; 31 pinned files match their sha256 |
| Stage A pipeline smoke: real pinned prompts, real `math_check` and Docker `stdio_check`, **scripted completions, no model** | right answers scored 1, wrong answers 0, `print(0)` failed on real tests, a capped unclosed think was censored and malformed with raw length 1024 |
| `guard` on the local RL v1 files | 55 problems (the held-out tasks), exit 1 |

Tests cover: empty completion, over-cap censoring kept in denominators, malformed traces never reported as zero
length (unclosed think, second unclosed span, bad tool JSON, stray marker), the committed manifest hash, tampered
and resealed-bad manifests, deterministic slot-major 4-label matrix with one seed per prompt, Stage C off by
default, lowered common cap and unsupported cells from the context preflight, missing checkers giving `None`
rather than 0, output marking and the training guard, and the resolver's file, text and answer hash checks.

## Stage A: OLMoE-1B-7B-0924-Instruct canary (12 Nov)

**Honesty note.** Stage A only: 10 prompts, 4 labels, 2 samples per cell. The model is
`allenai/OLMoE-1B-7B-0924-Instruct` with its own chat template, sampled at temperature 0.7 with base seeds
1234 and 1235. That makes it an optimization diagnostic on an open stand-in. It is **not the capstone model** and
not the G10 template, and Stage B was not run. No comparison with any other model, checkpoint or label is invited
from these numbers. With 20 cells per label, one cell moves a rate by 5 points.

### What ran

| | |
|---|---|
| Prompts | The manifest's own Stage A list: code-00, code-01 (aizu), code-08 (atcoder), code-15 (codechef), code-22 (hackerearth), math-00 … math-20 (MATH levels 1 to 5). Every text, test set and reference answer was hash-checked against `effort_manifest.json` (sha256 `d6190b2d…`) before generation |
| Prompt form | Raw task text as the user message, `Reasoning effort: {label}` as the system message, the model's native template (the same messages as the suite's `hf` backend). No instruction wrapper, so there is no request for `\boxed{}` or a code block |
| Cells | 10 prompts × 4 labels (`1k, 4k, 16k, 64k`) × 2 samples = 80, all of which ran. None was unsupported: the longest prompt is 964 tokens, so 964 + 1024 fits the 4096 context and no cap was lowered |
| Samples | `run_stage` was called once per sample, with base_seed 1234 (sample 0) and 1235 (sample 1). The seed is `prompt_seed(base_seed, slot)`, so a prompt has the same seed under all four labels |
| Sampling | temperature 0.7, top_p 1.0, common cap 1024 for every label (`max_tokens`), stop on EOS id 50279, `enable_thinking=False` passed to the template (OLMoE's template ignores it) |
| Model | `allenai/OLMoE-1B-7B-0924-Instruct` @ `7f1c97f440f06ce36705e4f2b843edb5925f4498`, Apache-2.0, un-gated, bf16, context 4096 |
| Engine | vLLM 0.32.0 on 1× H100 80GB HBM3 (Modal). Load 94 s; generation 38.5 s (sample 0) + 33.0 s (sample 1), one cell at a time |
| Image | `debian_slim` Python 3.12 + `vllm==0.32.0` + `huggingface_hub[hf_xet]`. Env: `VLLM_USE_DEEP_GEMM=0`, `VLLM_USE_FLASHINFER_SAMPLER=0`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `TOKENIZERS_PARALLELISM=false` |
| Scoring | Local, after generation: `math_check` (math-verify 0.9.0) and `stdio_check` in Docker 29.9.2, both from the rl-data checkers of PR saitej123/era-v5-lab#54 @ 957534c and unchanged. Censored cells stay in every denominator |

The suite code (`effort_suite.py`, manifest, tests) is unchanged. Two new files run it:
`eval/stage_a_modal.py` adds a vLLM `Backend` and calls the suite's `run_stage`, and `eval/stage_a_report.py` fills
in checker success and pools the two samples. Records and summaries are in
`results/effort_stageA_olmoe_2026-11-12/`. Every record carries `do_not_train: true`.

### Numbers per label (20 cells each = 10 prompts × 2 samples)

| Label | Ran | Parse ok (malformed / empty) | Within cap (censored) | ≤ label budget | Correct | Code correct | Math correct | Prompts solved in either sample | Median generated tokens |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1k | 20 | 20 (0 / 0) | 19 (1) | 20 | 0/20 | 0/10 | 0/10 | 0/10 | 559 |
| 4k | 20 | 20 (0 / 0) | 20 (0) | 20 | 0/20 | 0/10 | 0/10 | 0/10 | 474 |
| 16k | 20 | 20 (0 / 0) | 19 (1) | 20 | 1/20 | 1/10 | 0/10 | 1/10 | 541 |
| 64k | 20 | 20 (0 / 0) | 20 (0) | 20 | 1/20 | 1/10 | 0/10 | 1/10 | 460 |

- The two censored cells are both math-20 (MATH level 5), sample 0, at `1k` and `16k`. Each hit 1024 tokens.
- The two correct cells are both code-15 (codechef), sample 1, at `16k` and `64k`, passing all tests. Every other
  cell failed: math 40/40 wrong answer; code 23 runtime error on a test, 13 wrong answer on a test, 2 no code block.
- Checker sanity: the math checker scores each of the five reference answers 1, so the 0/40 on math is the model's
  answers (for example 16 against a reference of 12 on the level 1 item), not the scoring path.
- "Parse ok" is the suite's G10-format check: does the completion parse as one assistant turn, with no
  malformed or unclosed trace? OLMoE writes plain text with no G10 markers, so it passes trivially here. It shows
  that nothing malformed was emitted. It does not show that the format was learned. For a descriptive view only,
  math completions containing `\boxed`: 1k 2/10, 4k 4/10, 16k 4/10, 64k 1/10. Code completions containing a
  ```` ```python ```` block: 9, 10, 10 and 9 out of 10.
- "≤ label budget" is always true under one common cap of 1024, since the smallest budget is 1024. It carries no
  information about the label in Stage A.
- Matched against `1k` (same prompt, same seed): sample 1 has one gain at `16k` and one at `64k` (both code-15),
  with no losses. Sample 0 has no changes. These are two cells, not an effect.

### Short Pass/Fail per label

The rules were fixed in `stage_a_report.py` before the run. **Format obeyed:** PASS if every cell parses ok.
**Cap respected:** PASS if no cell is censored. **Answer correct:** PASS if at least one cell is verified
correct. That last rule is a canary bar for the scoring path, not a quality bar.

| Label | Format obeyed | Cap respected | Answer correct |
|---|---|---|---|
| 1k | PASS (20/20, trivially: plain text) | FAIL (19/20) | FAIL (0/20) |
| 4k | PASS (20/20, trivially: plain text) | PASS (20/20) | FAIL (0/20) |
| 16k | PASS (20/20, trivially: plain text) | FAIL (19/20) | PASS (1/20) |
| 64k | PASS (20/20, trivially: plain text) | PASS (20/20) | PASS (1/20) |

The Cap and Correct verdicts differ between labels by one cell each, and these are draws at temperature 0.7.
They are not a ranking of the labels.

### Command and billing

```bash
cd teams/10-post-training
modal run eval/stage_a_modal.py --out RAW_DIR --checker-dir RL_DATA_DIR \
    --source-dir code_contests=CODE_CONTESTS_DIR --source-dir2 hendrycks_math=HENDRYCKS_MATH_DIR
python eval/stage_a_report.py --raw RAW_DIR --checker-dir RL_DATA_DIR \
    --source-dir code_contests=CODE_CONTESTS_DIR --source-dir hendrycks_math=HENDRYCKS_MATH_DIR \
    --out results/effort_stageA_olmoe_2026-11-12
```

`RL_DATA_DIR` is `teams/10-post-training/rl-data` from PR saitej123/era-v5-lab#54 @ 957534c. The source dirs hold
the pinned `deepmind/code_contests` @ `802411c3…` and `EleutherAI/hendrycks_math` @ `21a56338…` files. The weights
were already cached in the Modal Volume from an earlier run. The modal CLI was 1.6.3.

`modal billing report --for today --json`, 12 Nov 08:35 IST (the day starts 00:00 UTC):

```json
{"object_id": "ap-3Fn43Gp7kITZAdCm7AzLpb", "description": "g10-effort-stagea", "cost": "0.21434245"}
{"object_id": "ap-6nfU0QRhbJt2CYS6FxCizL", "description": "g10-effort-stagea", "cost": "0.00294350"}
```

`ap-3Fn4…` is this run. `ap-6nfU…` is a first attempt that stopped at import, because `render.py` reads the
Team 3 marker files from the repo layout and the image had not copied them yet. Nothing was generated in that
attempt. **Stage A total: $0.217** against a $3 ceiling.

## Not done, and what it needs

- **No generation run.** (11 Nov text. On 12 Nov Stage A ran on OLMoE-Instruct, see above. Stage B has not run.) Next: Stage A on the open OLMoE checkpoint with `--backend hf` on a GPU job (diagnostic),
  then Stage B. The G10-template comparison waits for the G5 bundle (`team5` backend; the adapter returns
  `WAITING_FOR_G5_ARTIFACT` until it arrives).
- The `hf` and `team5` backends have not been exercised on a real model here (no local model, no GPU).
- Checkers come from `rl-data/` (PR saitej123/era-v5-lab#54), which is not in this base branch yet. Without it
  functional success is `None` with the reason, never 0. Code checking needs Docker.
- 45 of 100 slots are unavailable (tool/agentic 20, Indic 15, general 10) until approved human-written sources
  and the team's rubric exist.
- A licence-table row for hendrycks MATH, and the RL v1 drop of the 55 held-out tasks.
- G9's final independent score stays separate; this suite is development evidence once inspected.

## Commands

```bash
cd teams/10-post-training
python eval/effort_suite.py verify --source-dir code_contests=DIR --source-dir hendrycks_math=DIR
python eval/effort_suite.py plan --stage B
python eval/effort_suite.py run --stage A --backend hf --model allenai/OLMoE-1B-7B-0125-DPO \
    --source-dir code_contests=DIR --source-dir hendrycks_math=DIR --out OUT_DIR   # outputs stay out of Git
python eval/effort_suite.py guard --training path/to/training.jsonl
python -m pytest eval/test_effort_suite.py -q -p no:cacheprovider
```
