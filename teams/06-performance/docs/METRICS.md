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
                      + 2 * head again when ce_impl=chunked or fused (they recompute the head)
                      + optimizer FLOPs per step (Muon Newton-Schulz) / tokens per step
```

## MFU and HFU

```text
model_tflops_per_s = tokens_per_s * model_flops_per_token / 1e12
MFU                = model_tflops_per_s / peak_tflops

HFU                = tokens_per_s * hw_flops_per_token / 1e12 / peak_tflops
```

MFU uses model FLOPs; HFU uses hardware FLOPs. Both divide by the **dense** vendor peak for the
dtype the run used (`peak_tflops` in the run JSON). The run JSON `metrics` section stores `mfu`, `hfu`,
`model_tflops_per_s`, `model_flops_per_token`, `hw_flops_per_token`, `params_total` and `params_active`;
`mfu` and `hfu` are `null` when unavailable, with a reason recorded alongside each value. The optimizer FLOPs
come from `Optimizers.flops()` in `models/optim.py`.

`metrics.mfu` and `metrics.hfu` describe the single GPU selected by the benchmark process: the token
throughput and peak denominator are both per GPU. They are not node-wide or whole-run aggregates. G8 can
aggregate per-GPU values from separate benchmark processes where that is meaningful. Each metric has a
separate reason field next to it: `mfu_unavailable_reason` and `hfu_unavailable_reason`. A reason is `null`
when its metric is present. Current allowed reason strings are `unknown_gpu` (GPU not in `PEAKS`),
`unsupported_dtype` (known GPU has no peak entry for the run dtype), `peak_tflops_unavailable` (no usable
vendor peak was resolved), `invalid_peak_tflops` (a configured peak is not positive), and
`hw_flops_unavailable` (HFU only; hardware FLOPs are absent). When no peak is available both metrics carry
the peak-related reason; if only the hardware FLOPs are absent, MFU remains available and only HFU has a
reason.

## Correctness coverage

`correctness/test_metrics.py` checks the FLOPs-per-token formulas, MFU/HFU conversion, and known or
unknown GPU/dtype peak lookups. `correctness/test_optimizer.py` checks the Newton–Schulz FLOP count,
including that AdamW-only parameters do not contribute. `correctness/test_report.py` covers the 5% gate
and the Markdown renderers.

Run these focused checks from `teams/06-performance` with:

```bash
python -m pytest correctness/test_metrics.py correctness/test_optimizer.py correctness/test_report.py -q
```

The loop-versus-batched Muon bf16 comparison in `test_optimizer.py` requires CUDA with native bf16
support. Its current acceptance bound is `rtol=0.04, atol=0.02`; validate and record the observed error
on the target GPU before treating that bound as empirically measured.

## Peak table (`PEAKS` in `metrics.py`)

The lookup includes separate dense peaks for H200 NVL and H200 SXM; the specific `H200 NVL` match
must precede `H200`. RTX 3060 BF16/FP16 uses the dense value: NVIDIA's 101 Tensor-TFLOPS figure
includes 2:4 sparsity, so the table uses half.

`test_metrics.py` checks BF16, FP16, FP32, FP8, and FP4 lookups for every listed test GPU. The current
table defines no FP8 or FP4 peaks, so those lookups return `None` until authoritative dense values are added.

`gpu_peak(device_name)` / `peak_tflops(device_name, dtype)` walk a substring table (B200, H200 NVL,
H200, H100, A100, RTX 3060, L40S, …). First match wins, so specific names come first.

- Known GPU, known dtype peak → `peak_tflops` is set; MFU/HFU can be computed.
- Known GPU, empty peak entry (today `H100 NVL`, `A10G`), a dtype with no entry (bf16 on V100 / T4 / P100,
  where bench falls back to fp16 anyway), or unknown GPU → peak is missing; **leave MFU empty**. The bench
  adds a run warning; pass `--peak-tflops` on the bench CLI to set it.

## Tokens per second

`tokens_per_s` comes from the clean median step time and `tokens_per_step` (batch × sequence).
Report and G8 must use the value already stored in the run JSON, not recompute a different one.

## Peak memory and run identity

When present, `memory.peak_allocated_gb` and `memory.peak_reserved_gb` describe the CUDA device used by
that benchmark process. The `memory.gpu` object identifies it by name, local CUDA index, UUID, and PCI bus ID.
The historical `_gb` field names contain **GiB** values (bytes divided by 2^30); `memory.unit` is explicitly
`"GiB"` so consumers do not need to infer the unit from the name.

Every independent G6 benchmark invocation gets its own generated `meta.run_id` unless `--run-id <id>` is
provided. To correlate measurements of the same experiment, pass the exact same ID to G6 and G5, and have the
G5 harness write it into its records. G6 does not choose or transmit an ID to a separate harness. Leave
`--run-id` unset for independent runs. `--repeat N` creates a fresh ID per child run and rejects a shared
`--run-id`, since those are separate measurements.

## Dirty commit

`meta.dirty` is true when the working tree under `teams/06-performance` (outside `results/`)
differs from `meta.commit`, new files included. A dirty run is not a named commit: treat baselines from dirty
trees as non-authoritative until the tree is clean. `perfkit.report.compare` refuses a pair unless both runs
have `dirty: false` and the same commit. `dirty` is `null` when git status cannot run; with no `.git`
at all, `meta.commit` is taken from the `PERF_COMMIT` environment variable (or `null`).

## System metrics (`run["system"]`, CUDA runs)

`perfkit.metrics.GpuSampler` runs one `nvidia-smi --loop-ms=1000` child process per benchmark process while the
clean steps run (the steps that give step time, tokens/s and MFU). It queries all GPUs visible on that node and
emits a separate row for each GPU at each sample interval. The `system.metrics` min/median/max/mean aggregates
in the final JSON remain scoped to the benchmark's selected GPU; `system.gpu` identifies it. The sampler also
streams individual records
to the run's `<tag>.jsonl` sidecar as they arrive. Each line is a JSON object with `schema_version`, `event`,
UTC `timestamp`, `run_id`, `node_id`, `gpu` (`name`, local `index`, `uuid`, `pci_bus_id`), current clean `step`
when available, raw `metrics`, and `slowdown_flags`. `mem_used_mib` remains in MiB for G8 to convert. Reading
the JSONL does not synchronize CUDA; a separate thread consumes `nvidia-smi` output and flushes each event.
By default each invocation gets a unique `run_id`; pass `--run-id <id>` to share an ID across coordinated G5/G6
or multi-node invocations. `--node-id` overrides the default node name (`G8_NODE_ID` or hostname). For a
multi-node run, start one benchmark collector on each node with the same run ID and a distinct node ID; each
collector samples every GPU visible on its node. Use node-local output directories, or distinct `--tag` values
when the output directory is shared, so each node writes its own JSON and JSONL sidecars. For example, on each
node run `python -m perfkit.bench --run-id exp-42 --node-id node-a --tag exp-42-node-a ...` (change the node
ID and tag on the other node). A single invocation therefore covers one node and all its visible GPUs; it does
not launch or coordinate distributed training. The existing `torchrun` path in `perfkit.bench` is for the
isolated communication measurement, not a distributed training benchmark. `--repeat` creates separate runs
and therefore cannot be combined with an explicit shared run ID. The per-run `system` aggregates continue to
describe the GPU selected for the benchmark; JSONL contains separate samples for every visible GPU.

The same JSONL also contains one `step_timing` event per clean step, with `run_id`, `node_id`, global `rank`,
`local_rank`, one-based `step`, `data_ms`, `raw_step_ms`, and a UTC timestamp. A standalone process uses rank 0;
under `torchrun`, rank identifiers come from `RANK` and `LOCAL_RANK`. `raw_step_ms` matches
`clean.raw_step_ms` to four decimal places. `data_ms` is host wall time spent inside `Loader.next_batch()`;
on CUDA, a nonblocking host-to-device copy may still be in flight when that call returns.

Step timing events reuse the step timer's existing CUDA event pair. A background worker polls event completion with
`Event.query()`, computes elapsed time only after completion, and writes the JSONL record. It adds no
`cuda.synchronize()` or per-step event pair to the measured path; disk writes and event reads stay off the
training thread. Events arrive after their GPU work completes, so the JSONL is live with a short asynchronous
delay. Existing `clean.raw_step_ms` and `clean.data_ms` summaries are retained.

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

Besides the sampler, `meta.telemetry_start` and `meta.telemetry` (`after_warmup`, `during_steps`, `end`) hold
single `nvidia-smi` readings (`gpu_telemetry`). The `during_steps` reading is taken while three untimed steps run.
From those, bench warns at 83 C or above, when the driver reports a slowdown, or (when it reports no reasons)
when the SM clock is below 85% of `clocks.max.sm`.

## Goodput (partial)

`run["goodput"]["badput_s"]` has the wall-clock seconds of `startup` (model, loader, optimizer built),
`first_step` (includes `torch.compile` / autotune when on) and `warmup`. Time goodput and token goodput are
not written yet.

## Shared run JSON

`perfkit.schema.REQUIRED` lists every field report and G8 expect under `meta`, `config`, `clean`,
and `metrics` (plus `layer` / `trace` when those modes ran, `system` on CUDA). Renaming or removing a required key
breaks consumers — agree it before changing.
