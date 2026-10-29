# METRICS — MFU / HFU formulas for G8

Source of truth for the numbers: `perfkit/metrics.py`. Bench writes them into the run JSON
(`perfkit/schema.py` `REQUIRED["metrics"]`); report and G8 read the same fields. Do not invent
constants here — change `metrics.py` first.

## FLOPs per token

`T` = sequence length, `L` = layers, `d` = width, `N_active` from `GPT.param_counts`.

```text
model_flops_per_token = 6 * N_active + 12 * L * d * T
                        (PaLM; attention scores counted non-causal)

hw_flops_per_token    = model
                      + 2 forwards of the stack when reversible (invert + recompute)
                      + 2 * heads again when ce_impl=chunked (it recomputes the head, and the MTP head)
                      + 6 * (num_experts - top_k) * expert params per MoE layer when moe.dispatch=dense
                        (every expert runs on every token, as in G5's week-1 harness; a reversible
                        stack recomputes those too)
                      - 6 * (D - P) * d when ke_mode=sparse (it gathers P columns of the KE
                        projection instead of a [D, d] matmul)
                      + optimizer FLOPs per step (Muon Newton-Schulz) / tokens per step
```

Loss in `meta`: with `model.mtp` on, `loss_end` and `loss_after_first_step` are the next-token loss (what
G5 logs), `loss_end_mtp` the t+2 loss and `loss_end_total` their sum (what backward used).

`N_active` counts the MTP head when `model.mtp` is on (G5 config_v0 trains a second head on t+2):
it is a second LM head every token multiplies with. Dense dispatch and the sparse KE change only
HFU: MFU always counts top_k experts and a dense KE projection, so the two runs stay comparable.

## MFU and HFU

```text
model_tflops_per_s = tokens_per_s * model_flops_per_token / 1e12
MFU                = model_tflops_per_s / peak_tflops

HFU                = tokens_per_s * hw_flops_per_token / 1e12 / peak_tflops
```

MFU uses model FLOPs; HFU uses hardware FLOPs. Both divide by the **dense** vendor peak for the
dtype the run used (`peak_tflops` in the run JSON).

## Peak table (`PEAKS` in `metrics.py`)

`gpu_peak(device_name)` / `peak_tflops(device_name, dtype)` walk a substring table (B200, H200,
H100, A100, L40S, …). First match wins, so specific names come first.

- Known GPU, known dtype peak → `peak_tflops` is set; MFU/HFU can be computed.
- Known GPU, empty peak entry, or unknown GPU → peak is missing; **leave MFU empty** and say so
  in the run (or pass `--peak-tflops` on the bench CLI).

`metrics.mfu` and `metrics.hfu` are `null` when there is no peak: **null means missing, not 0**.
`metrics.mfu_reason` says why, and is `null` whenever MFU is set (a `--peak-tflops` override sets it):

| `mfu_reason` | Meaning |
|---|---|
| `not_cuda` | CPU or Apple MPS run: no vendor peak |
| `unknown_gpu` | the GPU name matches no `PEAKS` entry |
| `no_peak_for_dtype` | known GPU, no peak for the run's dtype (e.g. bf16 on a T4) |
| `no_peak` | `mfu()` called without a reason (not written by bench) |

Older run JSONs have no `mfu_reason`; read a missing key as unknown.

## Tokens per second

`tokens_per_s` comes from the clean median step time and `tokens_per_step` (batch × sequence).
Report and G8 must use the value already stored in the run JSON, not recompute a different one.

## Dirty commit

`meta.dirty` is true when the working tree under `teams/06-performance` (outside `results/`)
differs from `meta.commit`. A dirty run is not a named commit: treat baselines from dirty trees
as non-authoritative until the tree is clean.

## System metrics (`run["system"]`, CUDA runs)

`perfkit.metrics.GpuSampler` runs `nvidia-smi --loop-ms=1000` in a child process while the clean steps run
(the steps that give step time, tokens/s and MFU), so the samples describe the same steps. One sample per
second (`interval_ms` 1000). `metrics` has `min`, `median`, `max`, `mean` of each key over the samples;
`series` keeps every sample, so G8 can roll them up into its own windows:

```text
"start_time": "2026/10/29 07:48:04.100",          nvidia-smi timestamp of the first sample (GPU host local time)
"series": [{"t_s": 0.0, "gpu_util_pct": 59.0, "mem_util_pct": 21.0, "mem_used_mib": 30100.0,
            "sm_clock_mhz": 1410.0, "mem_clock_mhz": 1215.0, "power_w": 300.5, "temp_c": 60.0,
            "slowdown": []}, ...]
```

`t_s` is seconds since the first sample (`null` if the timestamp could not be read); a metric is `null` when
nvidia-smi reports `[N/A]`; `slowdown` lists the active slowdown reasons (`[]` for none, `null` when the
driver reports none). `mem_used_mib` is MiB (2^20 bytes). The allocator's `memory.peak_allocated_gb` and
`peak_reserved_gb` are GiB (2^30 bytes) despite the name.

| Key | nvidia-smi field | Meaning |
|---|---|---|
| `gpu_util_pct` | `utilization.gpu` | % of the sample period in which at least one kernel ran. Not SM occupancy: one tiny kernel at a time reads 100%. Use `trace.gpu_busy_pct` for idle gaps between kernels. |
| `mem_util_pct` | `utilization.memory` | % of the sample period in which device memory was read or written |
| `mem_used_mib` | `memory.used` | device memory in use, all processes (the allocator's own peak is `memory.peak_allocated_gb`) |
| `sm_clock_mhz`, `mem_clock_mhz` | `clocks.sm`, `clocks.mem` | clocks; compare with `meta.telemetry_start["clocks.max.sm"]` |
| `power_w`, `temp_c` | `power.draw`, `temperature.gpu` | board power and GPU temperature |

`slowdown_samples` / `slowdown_reasons`: samples in which the driver reported a power cap, thermal or hardware
slowdown (`SLOWDOWN` in `metrics.py`; idle and application clocks are not slowdowns). Any slowdown sample adds
a run warning: such a run may miss the 5% check. Off CUDA, or without `nvidia-smi`, there is no `system` section.

## Shared run JSON

`perfkit.schema.REQUIRED` lists every field report and G8 expect under `meta`, `config`, `clean`,
and `metrics` (plus `layer` / `trace` when those modes ran, `system` on CUDA). Renaming or removing a required key
breaks consumers — agree it before changing.
