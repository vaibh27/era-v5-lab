# Step caches: RoPE angles and stacked expert weights

**Status:** done. CPU and CUDA tests pass; same-container A/B on A100 below.
**Numeric trace:** unchanged. Training is bit-identical to the per-call code (tested), so no config or trace decision is needed.

## What changes

| part | before | after |
| :--- | :--- | :--- |
| RoPE (`GQA.rotary`) | arange, pow, cos and sin rebuilt for q and k in every layer of every microbatch | cos and sin built once per sequence length and device, same ops, so the values are bit-identical |
| dense MoE (`MoE.forward`) | `torch.stack` of 16 expert weights, then a cast to bf16, for gate, up and down, in every layer of every microbatch (8 per step at `micro_batch_size: 1`) | stacked and cast once per weight update; a custom autograd function returns the cached stack and sends its gradient to each expert weight with the same cast and unbind as before |

- The stack cache key holds every expert weight's version counter and storage pointer. An optimizer step and `load_state_dict` both change it (tested), so the cache never serves stale weights.
- Inference-mode tensors are cached separately, so evaluation in `torch.inference_mode()` followed by training works (tested).
- No new state-dict entries; checkpoints are unchanged.
- Under `torch.compile` both caches step aside (`torch.compiler.is_compiling()`): Dynamo cannot trace the inference-mode query or the storage pointers in the key, and inductor fuses the stack, cast and RoPE angles into the graph anyway. Without this each block split into 15 graphs and recompiled after every optimizer step (merged-branch compile check).
- Removes an unused `torch.zeros_like(flat)` allocation in the MoE forward.
- Memory: a persistent bf16 stack of the expert weights (T2: about 38 MB per layer, about 300 MB for 8 layers). In exchange the per-call fp32 stack (about 75 MB per layer per call) and its bf16 copy are gone.

## Measured (Modal A100-SXM4-80GB, torch 2.14.1, 25 Nov 2026)

Branch @ `189f6873` against `g6/dev` @ `0f2dea62`, `dryrun2.yaml` (KAS V2 head), eager, same container, 20 timed steps after 3 warm-up.

| case | median step | vs base | peak memory | check |
| :--- | ---: | ---: | ---: | :--- |
| base (`g6/dev`) | 1,393 ms | 1.000x | 10.39 GB | |
| step caches | 1,358 ms | 1.026x | 10.37 GB | loss trace and model hash bitwise equal to base: PASS |

As estimated: a small gain from fewer memory copies, with no numeric change.

## Tests (`tests/test_week2_step_caches.py`)

| test | checks | result |
| :--- | :--- | :--- |
| `test_training_is_bit_identical_to_the_per_call_code` | 4 Muon + AdamW steps, 2 microbatches each, clip: loss trace and every state-dict tensor equal to the old code | PASS (CPU) |
| `test_stack_rebuilds_after_an_optimizer_step_and_a_reload` | cache follows optimizer steps, in-place edits and `load_state_dict` | PASS (CPU) |
| `test_inference_mode_then_training` | no inference-tensor error | PASS (CPU) |
| `test_stack_gradient_reaches_every_expert_weight` | gradient routing through the cached stack | PASS (CPU) |
| `test_rotary_cache_matches_per_call_angles_bitwise` | full and half rotary, two lengths | PASS (CPU) |
| `test_caches_add_no_state` | state-dict keys unchanged | PASS (CPU) |
| `test_block_traces_as_one_graph` | a block under Dynamo: 1 graph, 0 breaks | PASS (CPU) |
| `test_optimizer_steps_cause_no_recompiles` | compiled blocks (`fullgraph=True`), 3 more Muon steps with `error_on_recompile` | PASS (CPU) |
| `test_bf16_training_is_bit_identical_to_the_per_call_code` | same as the first test, bf16 autocast, deterministic mode | PASS (A100) |

Full G5 suite on CPU (torch 2.14.1): 214 passed, 5 skipped. On A100: 9/9 tests in this file pass; the full G5 suite passes (the one Team 10 consumer test needs `teams/10-post-training` in the test tree).
