# Team 6 Gate 3 evidence for week 3

Tasks 1-5 and rotation R1/R2.

Gate 3 criterion: dry run 1 has a full profile, and each kernel for dry run 2 passes its test.

> **Status: lab run details still pending.** Every profile, A/B and test result in this PR was measured by G6 on a member GPU, an NVIDIA A100-SXM4-80GB on Modal. None of it comes from the lab GPU. We do not yet have the lab's dry run 1 profile: no torch.profiler traces, no GPU name or run SHA, no per-part split. The only lab numbers used are the step times from #620 (median 1,897.6 ms). The lab-column cells in `profile_dryrun1.md` and `speed_dryrun1.md` are marked _pending_. We will fill them when TSAI returns the lab traces; a job request follows.

| # | Task | Output | Status |
| :--- | :--- | :--- | :--- |
| 1 | Profile the dry run 1 step (torch.profiler) | `results/profile_dryrun1.md` | Done on A100, lab column pending |
| 2 | MoE dispatch, all-to-all, full-size checkpoint save | `results/speed_dryrun1.md` §1-3 | Done on A100, lab column pending |
| 3 | Fast kernels (MoE dispatch, attention) vs reference | `correctness/` pair tests | Done |
| 4 | Step time per rank, slow ranks | `results/speed_dryrun1.md` §4 | Done (single GPU, step time from #620) |
| 5 | Kernel changes for dry run 2, each with its test | `results/kernels_dryrun2.md` | Done on A100, TSAI decides |
| R1/R2 | Rotation correctness tests | `correctness/rotation_*.py` (3 members) | Done |

## Evidence

- **Profile (task 1):** on an A100 (member GPU, not the lab run): torch.profiler + layer profile (repeat 2, PASS) and Nsight Systems on the dry run 1 model.
  - Config: `configs/g5_t2_base.yaml`, matching #620: 8 x 512, T=2048, 16 routed + 1 shared experts, top-2, loop dispatch, Muon, bf16.
  - Run: B=8, commit `1f48dee8`.
  - Traces: gist `vaibh27/e4e60f177d4fe675ba0444054928d341`.
- **Speed (tasks 2 and 4):**
  - Step time comes from the lab run #620 (median 1,897.6 ms, steps 11-1221).
  - Dispatch, all-to-all and checkpoint splits are A100 shares.
- **Tests (task 3):** `pytest correctness -q --all-configs -rs` on A100 gives 1371 passed, 0 failed, 10 skipped, 12 xfailed, and 134 of 134 pairs PASS. Same result on torch 2.11.0 and 2.14.1.
- **Dry run 2 proposals (task 5):** 7 ranked changes from a same-container A100 A/B against a torch reference: batched Muon, SDPA attention, fused CE, Triton RMSNorm, router, RoPE and SwiGLU.
  - Each change has a passing test.
  - All 7 together: -57.7% step, 2.36x tokens/s, peak memory 44.3 to 8.2 GiB.
  - MoE dispatch stays `loop`: `sorted` and `grouped` were not faster (padding, extra Muon work on idle experts).
- **Fix:** Triton norm `tl.cast` in `kernels/layer_norm.py` and `kernels/rms_norm.py`, needed under torch.compile.

## Pins

- Job pin: class `main @ aec75af` (#447, `config_v1.yaml`).
- Profile runs: `g6/week_3 @ 1f48dee8`, plus `configs/g5_t2_base.yaml` (dry run 1 model, loop dispatch, loop Muon, torch kernels).
- `mfu_method=g6-v1`.

## Limits

- **Lab run details pending.** All measurements here are G6 runs on one A100-80GB (Modal). The lab's dry run 1 traces, GPU, run SHA and per-part shares are still needed from TSAI to fill the lab column.
- A100 numbers give shares, ratios and counts only. Absolute step time is the lab's (#620).
- Hardware and software differ: A100 on torch 2.11 here, RTX 3090 on torch 2.14 in the lab. Tests pass on both torch versions, but the size of each gain may differ on the lab GPU.
- Dry run 1 ran on one GPU, so there is no measured all-to-all and no slow ranks. §2 is a forecast for the multi-GPU run.
- The A/B used synthetic data with a collapsed router (18 of 128 expert slots active). A balanced-router group is included.
- The Triton MoE kernels (`kernels/moe_dispatch.py`, `grouped_gemm.py`) are not wired into the model and are not proposed.
- Tests live in `correctness/`, not `tests/`.
- The earlier TSAI runs #485, #486 and #492 profiled config_v1 on small `base.yaml` with `dispatch=sorted` (forced by `jobs/g5_config.py`), not the dry run 1 model. Their shares are in fork PR vaibh27/era-v5-lab#90 and will be added to this branch as a labeled "Earlier TSAI runs" section.

## Reproduce

```bash
cd teams/06-performance
python -m perfkit.bench --config configs/g5_t2_base.yaml --modes clean,layer,trace --level 3 --repeat 2 data.source=synthetic
python -m pytest correctness -q --all-configs -rs
```
