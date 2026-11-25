# Global SDPA attention for the G5 trainer (follow-up to #536)

**Status:** done in `g5_switches/`. CPU and CUDA tests pass on A100. Dry run 2 numbers come from the lab job (pending).
**Default:** unchanged. G5's trainer and configs are not edited; dry run 2 stays eager. SDPA runs only through the
G6 launcher with `--g6 attn_impl=sdpa`. Switching any run is a Team 5/TSAI call.
**Scope:** WO2 is the G5 trainer that dry run 2 runs. The G6 perfkit default stays eager (perfkit supports
`attn_impl: sdpa` as an option).

## How to run

Same command as the G5 job, with the launcher in front and one extra option:

```text
PYTHONPATH=teams/07-infrastructure python teams/06-performance/g5_switches/train.py --g6 attn_impl=sdpa \
    --config teams/05-architecture/configs/arms/dryrun2.yaml ...every other G5 argument unchanged...
```

`g5_switches/train.py` imports `teams/05-architecture/harness/train_week2.py`, patches attention, and calls its
`main()` with the remaining arguments. With no `--g6` option it is G5's trainer exactly (same run identity).

## What changes

- Global layers call `F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=...)` on the shared K/V
  heads. No `T x T` score matrix is materialised and K/V are not copied per query head. The rest of G5's
  `GQA.forward` (projections, RoPE, gate, output projection) is unchanged.
- Local layers (window, sinks) keep G5's own eager forward. Only a run started with `--g6 attn_impl=sdpa` uses
  SDPA; G5's configs, including config v1, stay eager.
- No new parameters or state: checkpoints load in either mode.
- The switch and the sha256 of each launcher file go into the resolved config under `provenance.g6_switches`.
  G5 hashes the config into the run identity and writes it to `manifest.json`, so an SDPA run cannot resume an
  eager checkpoint (or the other way round) and the manifest says which attention ran.
- The patch is pinned to the G5 attention code it was checked against (`G5_GQA_FORWARD_SHA256`, the sha256 of
  G5's `GQA.forward` source). If G5 changes that method, the launcher refuses `attn_impl=sdpa` and
  `test_patch_is_pinned_to_the_current_g5_attention` fails, until the patch is re-checked.
- Backend: Flash. With no backend forced, under the trainer's deterministic settings, the attention kernels are
  Flash, and Flash alone serves the call (GQA, causal, bf16, head dim 64).

## Expected numeric change

**The loss trace and final model hash will not match an eager run.** This is expected: eager materialises the
full score matrix in fp32 and runs a separate softmax; the fused SDPA kernel never materialises it and rounds
differently. A same-seed SDPA run repeats bitwise under the trainer's deterministic settings
(`test_bf16_sdpa_backward_repeats_bitwise`, CUDA).

## Evidence

Lab speed (TSAI's official Gate 2 A/B, #536 at `91e00cd`, T2 Muon arm, seed 1234, bf16, steps 6-20;
`gate2_speedup_ab.md`):

| variant | median step | baseline / variant | peak GPU allocation |
| :--- | ---: | ---: | ---: |
| eager | 2,088.1 ms | 1.000x | 11.12 GiB |
| SDPA | 1,902.3 ms | 1.098x | 9.56 GiB |

Dry run 2 (speed, loss and held-out bpb): from the TSAI job `dryrun2-sdpa-1234` beside the eager `dryrun2-1234`;
pending.

Launcher checks (Modal A100, G5's step on `dryrun2.yaml`, no timing claimed): with no switch the model hash is the
same as G5's own code without the launcher, so the launcher runs G5 unchanged; two SDPA runs give the same model
hash.

## Tests (`correctness/test_g5_sdpa.py`)

| test | checks | result |
| :--- | :--- | :--- |
| `test_patch_is_pinned_to_the_current_g5_attention` | G5's `GQA.forward` is the code the patch was checked against | PASS (CPU, A100) |
| `test_changed_g5_attention_stops_the_sdpa_patch` | a changed G5 method stops the patch | PASS (CPU, A100) |
| `test_parse_defaults_and_refusals` | `--g6` accepts only known switches and values | PASS (CPU, A100) |
| `test_eager_patches_nothing` | the default leaves G5's class untouched | PASS (CPU, A100) |
| `test_install_is_idempotent_and_uninstall_restores` | patch can be applied twice and undone | PASS (CPU, A100) |
| `test_sdpa_matches_eager_value_and_gradients` | fp32 logits and every parameter gradient match eager, MQA, GQA and MHA | PASS (CPU, A100) |
| `test_local_layers_keep_the_g5_eager_path_bitwise` | local layer with window and sink: bit-identical to G5 | PASS (CPU, A100) |
| `test_sdpa_adds_no_parameters_or_state` | same state-dict keys and shapes | PASS (CPU, A100) |
| `test_split_args` | `--g6` options are taken out, G5 arguments pass through | PASS (CPU, A100) |
| `test_launcher_runs_g5_with_the_switch_and_records_it` | G5 smoke run through the launcher: switch in the manifest and identity, G5 modules restored | PASS (CPU, A100) |
| `test_bf16_sdpa_close_to_eager` | bf16 autocast output and input gradient close to eager | PASS (A100) |
| `test_bf16_sdpa_backward_repeats_bitwise` | identical gradients twice under deterministic mode | PASS (A100) |
| `test_flash_alone_serves_the_trainer_call` | Flash alone runs the call (GQA, causal, bf16, deterministic mode) | PASS (A100) |
| `test_default_backend_is_flash` | with no backend forced, the profiled kernels are Flash (fresh process) | PASS (A100) |

Suites with this branch on A100: G6 correctness 980 passed (16 in this file), G5 suite 277 passed.

## Other teams

- No file under `teams/05-architecture` changes; G5's tests are untouched.
- Team 7 `layout/parallel_layout_v1.py` and Team 10 `rl-loop/runtime.py` read G5's config, which stays
  `attn_impl: eager`; nothing for them to change.
