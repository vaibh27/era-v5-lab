# Fused RMSNorm and SwiGLU

**Status:** done. CPU and CUDA tests pass; same-container A/B on A100 below.
**Decision asked of TSAI:** an A/B of `configs/arms/dryrun2_fused.yaml` against `dryrun2.yaml`. Small numeric-trace change.

## What changes

| switch | before | after |
| :--- | :--- | :--- |
| `model.norm_impl: fused` | RMSNorm as five eager ops (cast, square, mean, rsqrt, two multiplies) | `torch.nn.functional.rms_norm` on the fp32 input, same formula, one fused kernel on CUDA |
| `model.swiglu_impl: triton` | `F.silu(g) * u`: two kernels forward, about five backward, three saved tensors | one Triton kernel each way, fp32 math, one rounding, two saved tensors |

- Applies to the block norms, the final norm, the dense routed-expert activation (`[N, E, h]`, the largest elementwise tensor in the step) and the shared expert.
- The SwiGLU kernel is Team 6's `kernels/swiglu.py` (`@ 2c122815`, tested on A100 under torch 2.11 and 2.14), copied into `harness/week2/swiglu_triton.py` like `team7_checkpoints.py`. The G5 harness keeps no import path into another team's tree, and the run identity's source hashes cover the copy.
- Without CUDA or Triton, SwiGLU falls back to the torch ops; the default for both keys is `torch` (unchanged behaviour).
- No new parameters: checkpoints load with either setting.
- If `run.compile: true` lands, inductor fuses both ops anyway; then these switches matter only for eager runs. Under `torch.compile` the SwiGLU wrapper uses the torch ops: the kernel's `once_differentiable` backward and `torch.cuda.device` context are not traceable and split every compiled block at each MoE layer (11 graph breaks in the merged A100 run).

## Expected numeric change

- CPU: `F.rms_norm` gives a bit-identical forward and weight gradient; the bf16 input gradient can differ in the last bit (different backward formula).
- CUDA: the fused RMSNorm kernel and the single-rounding SwiGLU round differently from the eager ops, so the loss trace and model hash will not match.

## Tests (`tests/test_week2_fused_kernels.py`)

| test | checks | result |
| :--- | :--- | :--- |
| `test_arm_is_valid_and_only_changes_kernels` | the arm differs from `dryrun2.yaml` only in the two keys | PASS (CPU) |
| `test_unknown_kernel_choices_are_refused` | validation | PASS (CPU) |
| `test_defaults_stay_torch` | no behaviour change without the keys | PASS (CPU) |
| `test_fused_rms_norm_forward_is_bit_identical_on_cpu` | fp32 and bf16 | PASS (CPU) |
| `test_fused_model_matches_torch_ops` | whole-model loss equal, gradients within 1e-5 | PASS (CPU) |
| `test_swiglu_falls_back_to_torch_without_cuda` | fallback is exactly the torch ops | PASS (CPU) |
| `test_block_traces_as_one_graph_with_the_kernel_enabled` | under Dynamo the kernel path is never traced: 1 graph, 0 breaks | PASS (CPU) |
| `test_swiglu_kernel_close_to_torch_on_3d_expert_activations` | `[256, 16, 768]` bf16, forward and both gradients | PASS (A100) |
| `test_bf16_model_close_to_torch_ops` | whole model under bf16 autocast | PASS (A100) |
| `test_compiled_block_has_no_graph_breaks_on_cuda` | bf16 block under Dynamo on CUDA: 0 breaks | PASS (A100) |
| `test_bf16_model_repeats_bitwise_under_deterministic_mode` | trainer's deterministic settings | PASS (A100) |

Full G5 suite on CPU (torch 2.14.1): 213 passed, 8 skipped. On A100: 11/11 tests in this file pass; the full G5 suite passes (the one Team 10 consumer test needs `teams/10-post-training` in the test tree).

## Measured (Modal A100-SXM4-80GB, torch 2.14.1, 25 Nov 2026)

Branch @ `d3833144` against `g6/dev` @ `0f2dea62`, `dryrun2.yaml` (KAS V2 head), eager, same container, 20 timed steps after 3 warm-up.

| case | median step | vs base | peak memory | check |
| :--- | ---: | ---: | ---: | :--- |
| base (`g6/dev`) | 1,700 ms | 1.000x | 10.39 GB | |
| `dryrun2_fused.yaml` | 1,693 ms | 1.004x | 9.93 GB | trace diff 5.9e-4 |

- Speed: no gain on its own at T2 with the KAS V2 head (inside run-to-run spread). The win is memory: 0.46 GB less peak (one fewer saved tensor per SwiGLU).
- Under `run.compile: true` the switches step aside and inductor fuses the same ops, so they matter only for eager runs. Combined with the other G6 switches: 1.13x eager, 1.24x compiled on A100 (`g6/w4-g5-switches`, 25 Nov).
