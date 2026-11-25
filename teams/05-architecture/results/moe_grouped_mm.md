# MoE `grouped_mm` dispatch (routed experts only)

**Status:** done. CPU and CUDA tests pass on H100; same-container A/B on H100 below. Slower on A100 (no fast path below SM90).
**Decision asked of TSAI:** an A/B of `configs/arms/dryrun2_grouped_mm.yaml` against `dryrun2.yaml`, then whether to adopt it for the main run. It changes the numeric trace.

## Why

The dense path (`model.moe.dispatch: loop`, TSAI's Gate 2 fast path) runs all 16 experts on every token and keeps the top 2 with their gates: about 8x the routed expert FLOPs at top-2 of 16, and an `[N, E, h]` activation per projection. The #536 sparse prototype lost (0.875x) because it read the busiest expert's count back to the host on every call, padded every expert to that count, and gathered through a padded buffer.

## What changes

- `model.moe.dispatch` accepts `loop` (the dense path, unchanged and bit-identical) or `grouped_mm`.
- `grouped_mm`: sort all `N x top_k` routes by expert, compute group offsets on the GPU (`searchsorted`), and run gate, up and down as `torch.nn.functional.grouped_mm` with those offsets. Gates are applied in fp32 and the `k` slots summed in fp32, as on the dense path.
- Every route is kept, padding tokens included with gate 0, so the row count is fixed at `N x top_k`: no host synchronisation, no capacity padding, no data-dependent shapes.
- Idle experts get exact zero gradients, as on the dense path, so the Muon update set is the same.
- Needs bf16 autocast on an SM80+ GPU; fp32 on CUDA is refused with a clear error. CPU works in fp32 for tests. torch 2.14 has the fused grouped GEMM only on SM90+ (H100 and newer); on A100 it falls back to a per-expert loop with host syncs, which is slower than the dense path.
- No new parameters: checkpoints load in either mode.

## Expected numeric change

The loss trace and final model hash will not match a dense run: the dense path sums `E` gated outputs (14 of them exact zeros) in one einsum, `grouped_mm` sums `k` outputs. Same value in real arithmetic; on CPU fp32 the outputs and gradients agree to 1e-5 (tested).

## Tests (`tests/test_week2_grouped_mm.py`)

| test | checks | result |
| :--- | :--- | :--- |
| `test_matches_dense_value_and_gradients` | output, input gradient and every parameter gradient vs dense, top-1 and top-2, with padding tokens | PASS (CPU) |
| `test_idle_experts_get_exact_zero_gradients` | an expert no token picks gets zero gradients, as on dense | PASS (CPU) |
| `test_padding_tokens_get_no_routed_output` | invalid tokens get no routed contribution | PASS (CPU) |
| `test_whole_model_training_tracks_dense` | 5 AdamW steps of the whole model: loss trace within 1e-4 of dense | PASS (CPU) |
| `test_arm_is_valid_and_only_changes_dispatch` | the A/B arm differs from `dryrun2.yaml` only in `moe.dispatch` | PASS (CPU) |
| `test_unknown_dispatch_is_refused` | validation | PASS (CPU) |
| `test_bf16_close_to_dense` | T2 width (d=512, h=768, 16 experts), bf16 | PASS (H100) |
| `test_no_host_synchronisation` | `torch.cuda.set_sync_debug_mode("error")` over forward + backward | PASS (H100); skipped below SM90 (fallback syncs) |
| `test_bf16_repeats_bitwise_under_deterministic_mode` | the trainer runs with `use_deterministic_algorithms(True)` | PASS (H100) |
| `test_fp32_on_cuda_is_refused` | clear error instead of a kernel failure | PASS (H100) |

Full G5 suite on CPU (torch 2.14.1): 212 passed, 8 skipped. On H100: 10/10 tests in this file pass; the full G5 suite passes (the one Team 10 consumer test needs `teams/10-post-training` in the test tree). The dense path is bit-identical to `g6/dev` (logits and all 47 parameter gradients, one-off check).

## Measured (Modal H100 80GB HBM3, torch 2.14.1, 25 Nov 2026)

Branch @ `f98ffc84` against `g6/dev` @ `0f2dea62`, `dryrun2.yaml` (KAS V2 head), eager, same container, 20 timed steps after 3 warm-up.

| case | median step | vs base | peak memory | check |
| :--- | ---: | ---: | ---: | :--- |
| base (`g6/dev`, dense path) | 964 ms | 1.000x | 10.44 GB | |
| `dryrun2_grouped_mm.yaml` | 913 ms | 1.056x | 9.01 GB | trace diff 5.9e-4 |

- A separate 4-round H100 A/B on the earlier merged branch (cases interleaved, 2 containers) gave 1.03x eager and 1.02x compiled, both inside the round-to-round spread, with 1.2-1.4 GB less peak memory.
- Verdict: a memory win, no clear speed win at T2 size. The FLOP saving should grow with more experts or a wider model.

## Open risks

- Deterministic mode: the gather `xe[order // top_k]` has a scatter-add backward; under `use_deterministic_algorithms(True)` it repeats bitwise on H100 (tested).
- Team 10 `capstone/test_keep_weights_real_model.py` says only loop dispatch exists; its note becomes out of date.
