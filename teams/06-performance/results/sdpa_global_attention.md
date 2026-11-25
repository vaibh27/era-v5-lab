# Global SDPA attention (follow-up to #536)

**Status:** code, CPU and CUDA tests done. The dry run 2 A/B (loss and held-out bpb) is the TSAI job `dryrun2-sdpa-1234`, not run yet.
**Default:** unchanged (eager). Switching dry run 2 is a Team 5/TSAI call; the `--g6 attn_impl=sdpa` launcher is ready.
**Scope:** WO2 is the G5 trainer that dry run 2 runs. The G6 perfkit default stays eager (perfkit supports `attn_impl: sdpa` as an option).

## What changes

- No file under `teams/05-architecture` changes. `teams/06-performance/g5_switches/train.py` runs G5's `harness/train_week2.py` unchanged; `--g6 attn_impl=eager` (default, unchanged) or `--g6 attn_impl=sdpa`.
- With `sdpa`, global layers call `F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=...)` on the shared K/V heads. No `T x T` score matrix is materialised and K/V are not copied per query head.
- Backend: Flash. Profiled on A100 with no backend forced, under the trainer's deterministic settings: the attention kernels are Flash, not the math fallback, and Flash alone serves the call (GQA, causal, bf16, head dim 64).
- Local layers (window, sinks) always use G5's eager path. Only a run started with `--g6 attn_impl=sdpa` uses SDPA; G5's configs, config v1 included, stay eager.
- No new parameters or state: checkpoints load in either mode.
- The switch goes into the manifest and the run identity (`provenance.g6_switches`), so an SDPA run cannot resume an eager checkpoint. With no `--g6` option the launcher is G5's trainer exactly.
- The patch is pinned to the sha256 of G5's `GQA.forward` source: if G5 changes it, `attn_impl=sdpa` stops until the patch is re-checked.
- Command for the A/B: the G5 job command with `python teams/06-performance/g5_switches/train.py --g6 attn_impl=sdpa` in place of `python teams/05-architecture/harness/train_week2.py`, and `--config` still `dryrun2.yaml`.

## Expected numeric change

**The loss trace and final model hash will not match an eager run.** This is expected: eager materialises the full score matrix in fp32 and runs a separate softmax; the fused SDPA kernel never materialises it and rounds differently. A same-seed SDPA run repeats bitwise under the trainer's deterministic settings (`test_bf16_sdpa_backward_repeats_bitwise`, CUDA).

## Evidence

Official Gate 2 A/B by TSAI (#536 at `91e00cd`, T2 Muon arm, seed 1234, bf16, steps 6-20; `gate2_speedup_ab.md`):

| variant | median step | baseline / variant | peak GPU allocation |
| :--- | ---: | ---: | ---: |
| eager | 2,088.1 ms | 1.000x | 11.12 GiB |
| SDPA | 1,902.3 ms | 1.098x | 9.56 GiB |

Dry run 2 (speed, loss and held-out bpb): TSAI job `dryrun2-sdpa-1234` beside the eager `dryrun2-1234`; pending.

## Tests (`correctness/test_g5_sdpa.py`)

| test | checks | result |
| :--- | :--- | :--- |
| `test_local_layers_keep_the_g5_eager_path_bitwise` | local-with-sink layers stay bit-identical to G5's eager code | PASS (CPU) |
| `test_sdpa_matches_eager_value_and_gradients` | fp32 logits and every parameter gradient match eager, for MQA, GQA and MHA | PASS (CPU) |
| `test_sdpa_adds_no_parameters_or_state` | same state-dict keys and shapes | PASS (CPU) |
| `test_parse_defaults_and_refusals` | the launcher accepts only `eager` and `sdpa` | PASS (CPU) |
| `test_eager_patches_nothing` | the default leaves G5's code untouched | PASS (CPU) |
| `test_install_is_idempotent_and_uninstall_restores` | the patch can be applied twice and undone | PASS (CPU) |
| `test_patch_is_pinned_to_the_current_g5_attention` | G5's `GQA.forward` is the code the patch was checked against | PASS (CPU) |
| `test_changed_g5_attention_stops_the_sdpa_patch` | a changed G5 method stops the patch | PASS (CPU) |
| `test_split_args` | `--g6` options are taken out, G5 arguments pass through | PASS (CPU) |
| `test_launcher_runs_g5_with_the_switch_and_records_it` | G5 smoke run through the launcher: switch in the manifest and run identity | PASS (CPU) |
| `test_bf16_sdpa_close_to_eager` | bf16 autocast output and input gradient close to eager | PASS (CUDA) |
| `test_bf16_sdpa_backward_repeats_bitwise` | two backward passes give identical gradients under deterministic mode | PASS (CUDA) |
| `test_flash_alone_serves_the_trainer_call` | Flash alone runs the call (GQA, causal, bf16, deterministic mode) | PASS (CUDA) |
| `test_default_backend_is_flash` | with no backend forced, the profiled attention kernels are Flash | PASS (CUDA) |

On CUDA (A100-80GB): G6 correctness 980 passed, G5 suite 277 passed.

## Other teams

- Team 7 `layout/parallel_layout_v1.py` and Team 10 `rl-loop/runtime.py` read G5's config, which stays `attn_impl: eager`; nothing for them to change.
