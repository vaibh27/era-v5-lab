# Launch overhead: microbatch size and compiled blocks

**Status:** compile switch and CPU tests done; CUDA repeat test and the A/B pending. Microbatch size needs no code.
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
- Memory: dry run 1 peaked at 3.56 GB at microbatch 1 on the RTX 3090 (#620). The full-vocab fp32 logits and their gradient grow by about 1 GiB each per extra sample, so microbatch 2 should be near 6-8 GB and 4 near 12-16 GB. Estimates, not measured; microbatch 8 would not fit in 24 GB with the dense head.

## 2. `run.compile: true` (this PR)

- `validate` accepted only `compile: false` ("Compiled backend not verified"), and the trainer never read the key. Now `true` compiles each transformer block in place (`Module.compile`, inductor) right after the model is built.
- In-place compile keeps the parameter names, so checkpoints, `model_hash` and resume use the same keys as an eager run (tested). The run identity already includes the config, so a compiled run cannot resume an eager checkpoint by accident.
- One compiled graph serves all blocks (same code and shapes) and there are no graph breaks (CPU check: 1 unique graph, 0 breaks).
- The KE2 embedding, the head and the loss stay eager (G4 owns the KE2 and KAS-head kernels).
- Arm `configs/arms/dryrun2_compile.yaml` is `dryrun2.yaml` plus `run.compile: true`.

## Expected numeric change

Compiled kernels fuse ops and round differently, so the loss trace and model hash will not match an eager run. On CPU fp32 the loss and every gradient match eager to 1e-5 (tested). Run-to-run repeatability under the trainer's deterministic settings still has to be shown on CUDA: inductor autotuning can pick different kernels between processes unless it is pinned.

## Tests (`tests/test_week2_compile.py`)

| test | checks | result |
| :--- | :--- | :--- |
| `test_arm_is_valid_and_only_changes_compile` | the arm differs from `dryrun2.yaml` only in `run.compile` | PASS (CPU) |
| `test_non_boolean_compile_is_refused` | validation | PASS (CPU) |
| `test_same_parameter_names_and_hash` | state-dict keys and `model_hash` unchanged by compiling | PASS (CPU) |
| `test_compiled_matches_eager_value_and_gradients` | loss and every parameter gradient vs eager | PASS (CPU) |

Full G5 suite on CPU (torch 2.14.1): 210 passed, 4 skipped.

## Open items

- CUDA: two same-seed compiled runs must match bitwise, and the first-step compile time must be recorded so it is not counted as step time.
- Combining with the other branches: the step-cache branch's custom autograd function and the `grouped_mm` path have not been compiled yet; check graph breaks when they meet.
