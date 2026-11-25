# Launch overhead: microbatch size and compiled blocks

**Status:** done. CPU and CUDA tests pass; same-container A/B on A100 below. Microbatch size needs no code.
**Decision asked of TSAI:** two A/Bs against `dryrun2.yaml`, both changing the numeric trace.

## Why

Dry run 2 runs `micro_batch_size: 1` (`configs/runtime.yaml`) with eager kernels: 8 forward/backward passes per step, each made of many small elementwise kernels (fp32 RMSNorm casts, RoPE, router softmax and top-k, gate scatter, SwiGLU). With one sample per pass most of these kernels are tiny, so kernel launches and Python overhead are a large share of the step (unmeasured on the G5 trainer; the G6 dry run 1 profile had the GPU idle 23-25% of the step).

## 1. Microbatch size (no code change)

`run.micro_batch_size` is already a resolution key (`apply_resolution` in `train_week2.py`), so an agreed resolution file can set it without touching the arm or config v1:

```yaml
# resolution overlay for the A/B, on top of the agreed dry run 2 resolution
run:
  micro_batch_size: 2   # then 4
```

- Same batch, same tokens, same loss normalisation (each microbatch is divided by the full-batch token count). Only the summation order changes, so the trace is close but not bit-identical.
- Memory (measured on A100 below, KAS V2 head): 10.4 GB at microbatch 1, 18.1 GB at 2, 33.0 GB at 4. Microbatch 2 fits a 24 GB card; 4 needs a 40 GB or larger card.

## 2. `run.compile: true` (this PR)

- `validate` accepted only `compile: false` ("Compiled backend not verified"), and the trainer never read the key. Now `true` compiles each transformer block in place (`Module.compile`, inductor) right after the model is built.
- In-place compile keeps the parameter names, so checkpoints, `model_hash` and resume use the same keys as an eager run (tested). The run identity already includes the config, so a compiled run cannot resume an eager checkpoint by accident.
- One compiled graph serves all blocks (same code and shapes) and there are no graph breaks (CPU check: 1 unique graph, 0 breaks).
- The KE2 embedding, the head and the loss stay eager (G4 owns the KE2 and KAS-head kernels).
- Arm `configs/arms/dryrun2_compile.yaml` is `dryrun2.yaml` plus `run.compile: true`.

## Expected numeric change

Compiled kernels fuse ops and round differently, so the loss trace and model hash will not match an eager run. On CPU fp32 the loss and every gradient match eager to 1e-5 (tested). Two compiled runs in one container repeat bitwise under the trainer's deterministic settings (A100 and H100, below).

## Measured (Modal A100-SXM4-80GB, torch 2.14.1, 25 Nov 2026)

Branch @ `230d60bb` against `g6/dev` @ `0f2dea62`, `dryrun2.yaml` (KAS V2 head), same container, 20 timed steps after 3 warm-up.

| case | median step | vs base | peak memory | first step | check |
| :--- | ---: | ---: | ---: | ---: | :--- |
| base (`g6/dev`, microbatch 1, eager) | 1,623 ms | 1.000x | 10.39 GB | 4.3 s | |
| `run.micro_batch_size=2` | 1,262 ms | 1.286x | 18.14 GB | 3.8 s | trace diff 1.3e-3 |
| `run.micro_batch_size=4` | 1,177 ms | 1.379x | 32.95 GB | 3.7 s | trace diff 1.5e-3 |
| `run.compile: true` | 1,252 ms | 1.296x | 9.30 GB | 30.9 s | 0 graph breaks; trace diff 6.4e-4 |
| `run.compile: true`, second run | 1,235 ms | 1.314x | 9.30 GB | 6.8 s | bitwise equal to the first compiled run: PASS |

- The first compiled step includes compilation (about 31 s cold, 7 s with a warm cache); it is not step time.
- Microbatch 4 logged one caching-allocator retry after an out-of-memory attempt and finished normally.
- Combined with the other G6 switches (all on, compiled, microbatch 4): 1.43x on A100 and 1.76x on H100, 0 graph breaks, repeat bitwise (`g6/w4-g5-switches`, 25 Nov).

## Tests (`tests/test_week2_compile.py`)

| test | checks | result |
| :--- | :--- | :--- |
| `test_arm_is_valid_and_only_changes_compile` | the arm differs from `dryrun2.yaml` only in `run.compile` | PASS (CPU) |
| `test_non_boolean_compile_is_refused` | validation | PASS (CPU) |
| `test_same_parameter_names_and_hash` | state-dict keys and `model_hash` unchanged by compiling | PASS (CPU) |
| `test_compiled_matches_eager_value_and_gradients` | loss and every parameter gradient vs eager | PASS (CPU) |

Full G5 suite on CPU (torch 2.14.1): 210 passed, 4 skipped. On A100: 4/4 tests in this file pass; the full G5 suite passes (the one Team 10 consumer test needs `teams/10-post-training` in the test tree).

## Open items

- None for this switch. With the step-cache and fused branches, both step aside under compile; the combined branch shows 0 graph breaks.
