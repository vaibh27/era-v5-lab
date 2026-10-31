# Every part we time, and later make faster

This is the list the code is designed around. If a part is not in this file, the report and the
5% gate cannot see it: **add the row here first, then the code.**

## Design rules

1. **Every row is a `region()`** with the name and level below (`perfkit/regions.py`).
   Names are `<phase>:<part>`, e.g. `fwd:block3.attn.sdpa`, `bwd.recompute:block3`, `opt:opt.muon`.
2. **Every expensive part is a config switch backed by a registry** (like `DISPATCH` in `models/moe.py`).
   The plain PyTorch version stays as `reference`. A faster version is a new entry, never a rewrite.
3. **No host syncs inside the step.** `.item()` only at logging steps.
4. **All shapes come from `ModelConfig`.** G5's full-width config is a different YAML, nothing else.
5. **Every faster version ships with a `@pair` correctness test** and a before/after timing on the same GPU.

Levels: 1 = phases, 2 = per layer and big parts (gated at 5%), 3 = inside a layer (information only).

## 1. Inside the training step

| Part | Region (level) | Switch: week-1 baseline → later | Likely bottleneck | Later optimization |
|---|---|---|---|---|
| Data: batch sampling + host-to-device copy | `data` (1) | `data.loader: simple → prefetch` | CPU, copy overlap | background prefetch thread, pinned buffers |
| KE v1 codec + projection | `embed.ke` (3) | `model.ke_mode: dynamic \| cached` → sparse | memory (8192-dim codec) | gather + sum (`embedding_bag`), fuse the position add |
| Position add | `embed` (2) | | memory | fuse into the KE kernel |
| LayerNorm × 2 per block | `blockN.ln1`, `blockN.ln2` (3) | `model.norm_impl: torch → fused` | memory | fuse with the residual add or the next matmul |
| QKV matmul + head split | `blockN.attn.qkv` (3) | | compute | tensor-core-friendly shapes, fused split |
| Attention kernel | `blockN.attn.sdpa` (3) | `model.attn_impl: sdpa → flash3 / flex` | compute / memory | FlashAttention-3 (H100), FA4 / cuDNN (B200) |
| Output copy + projection | `blockN.attn.proj` (3) | | memory (the copy) | a layout that needs no copy |
| Dense MLP | `blockN.mlp` (3) | | compute | fused GELU |
| MoE router: matmul, softmax, top-k, gates | `blockN.moe.router` (3) | | launch | one fused router kernel |
| MoE dispatch: gather, experts, combine | `blockN.moe.dispatch` (3), `.dispatch.expertE` (3); sorted path adds `.dispatch.grouped` | `model.moe.dispatch: loop → sorted` | **host syncs + launch** | sort routes, pack them per expert, run one batched matmul per projection (one host sync for the largest group) |
| Shared experts | `blockN.moe.shared` (3) | | compute | batch with the routed experts |
| Residual / midpoint arithmetic | inside `blockN` (2) | | memory | fusion |
| Final LN + LM head + cross-entropy | `head+loss` (2) | `model.ce_impl: torch → chunked → fused` | **memory (fp32 logits)** | fused linear + cross-entropy |
| Backward of every part above | `bwd:<part>` | same switches | ~2× forward | follows the forward fixes |
| Reversible: invert, recompute, local backward | `bwd.invert:blockN`, `bwd.recompute:blockN`, `bwd:blockN.grad` (3) | `model.reversible` | ~2× extra forward | fuse invert + recompute, cache cheap pieces |
| fp16 grad unscale | `opt.unscale` (2) | | small | |
| Gradient clipping (global norm) | `opt.clip` (2) | | memory, one reduction | foreach / fused kernel |
| Muon (Newton-Schulz per matrix) | `opt.muon` (2), `opt.muon.ns` (3) | `optim.muon_impl: loop ↔ batched` | launch (many small matrices) | batch same-shaped matrices with `bmm`; compare against the loop reference |
| Polar Express Muon + AdamW (one optimizer) | `opt.polar_muon_adamw` (2), with `opt.muon` / `opt.adamw` and `opt.muon.ns` inside | `optim.name: muon → polar_muon_adamw` | launch | comparison path, not the default |
| AdamW (embedding parameters, output head, norm gains, biases) | `opt.adamw` (2) | `fused=True` | vocabulary-dependent | |
| zero_grad, MoE bias update | `opt.zero_grad`, `opt.moe_bias` (3) | | tiny | |

## 2. Around the step (parts of the run, not of the step)

| Part | Where it is timed | Why it matters |
|---|---|---|
| Startup: model, data loader and optimizer build | `goodput.badput_s.startup` | time to first step (week-4 launch speed check) |
| First step, including `torch.compile` / autotune | `goodput.badput_s.first_step` | once per run, can take minutes |
| Warmup steps | `goodput.badput_s.warmup` | allocator and autotune settle |
| Logging (`.item()`) | `goodput.badput.logging` (planned) | each call is a host sync |
| Eval | `goodput.badput.eval` (planned) | not training time |
| Checkpoint save (later load / resume) | `checkpoint` mode: `save_ms` (= `to_cpu_ms` + `write_ms`), `size_mb` | blocks training; later async or sharded |
| Tokenizer throughput | `tokenizer` mode (planned; not in `bench.MODES` yet) | offline, but feeds the data pipeline (G3's tokenizer later) |

## 3. Measured, but not regions

| What | How | Why |
|---|---|---|
| CPU-GPU syncs per step | `torch.cuda.set_sync_debug_mode("warn")` in trace mode: `trace.host_syncs_per_step`, `trace.sync_sites` | each one stops the CPU from queueing work ahead |
| GPU busy % and idle gaps | profiler trace: `trace.gpu_busy_pct` | shows the GPU waiting for the CPU |
| Kernel launches per step | profiler trace: `trace.kernels_per_step`, `trace.top_ops` | launch-bound vs compute-bound |
| Peak memory | `torch.cuda` memory stats: `memory.peak_allocated_gb`, `peak_reserved_gb` (GiB; `memory.gpu` identifies device) | batch-size limits, reversible savings |
| GPU utilization, clocks, temperature, power, slowdown reasons | `nvidia-smi` sampled through the clean steps (`GpuSampler`): `system`; one-off samples in `meta.telemetry` | explains 5% misses on shared GPUs |

## 4. Weeks 2 to 4 (plan the names and switches now, code later)

| Part | Region | Switch |
|---|---|---|
| Gradient all-reduce / reduce-scatter (data parallel) | `comm.grad` | `dist.strategy: ddp → fsdp` |
| MoE expert-parallel all-to-all (dispatch and combine) | `comm.moe_a2a` (week 1: the payload, timed alone with `--modes comm` under torchrun) | `model.moe.parallel: none → ep` |
| Communication overlap with backward | `bwd` with and without comm | bucket size |
| Load imbalance across GPUs and experts | per-rank step time, tokens per expert | balancing settings |
| Checkpoint sharding | `checkpoint` | `ckpt.mode: sync → async / sharded` |
| FP8 matmuls (H100 / B200) | same regions | `run.dtype: bf16 → fp8` |
| Time to first step | startup | launch speed check |

## Switches already present

| Switch | Values (default first) | Notes |
|---|---|---|
| `model.moe.dispatch` | `loop` \| `sorted` | `DISPATCH` in `models/moe.py`; implementation switch |
| `optim.muon_impl` | `loop` \| `batched` | `batched` runs same-shaped matrices through `torch.bmm`; implementation switch |
| `model.ce_impl` | `torch` \| `chunked` \| `fused` | `chunked` checkpoints position chunks; `fused` tiles vocabulary by `model.ce_vocab_chunk` and recomputes logits in backward |
| `model.ke_mode` | `dynamic` \| `cached` | `cached` keeps a V x 8192 fp32 table (4 GiB at V = 131072) |
| `optim.name` | `adamw` (base) / `muon` (lab) \| `polar_muon_adamw` | a different update, so a different config, not an A/B |
| `run.compile` | `false` \| `true` | regions inside compiled code are not timed reliably |

Implementation switches (`report.IMPL_SWITCHES`: `model.moe.dispatch`, `optim.muon_impl`) are the only config
differences `--against` accepts; the verdict is then labelled A/B. Every other switch in the tables above
(`data.loader`, `model.norm_impl`, `model.attn_impl`, `dist.strategy`, `model.moe.parallel`, `ckpt.mode`, fp8)
is planned and not in `configs/base.yaml` yet.
