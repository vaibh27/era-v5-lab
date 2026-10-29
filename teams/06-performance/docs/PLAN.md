# G6 week 1 plan

**Week 1: timing harness, correctness framework, first baseline.**

Scope decisions:
- **Lab architecture:** BrahmicTokenizer-131K from its pinned GitHub source, raw FineWeb sample-10BT retokenized to uint32 shards,
  Kronecker Embedding v1, GQA, MoE with 1 shared + top-2 of 8 routed experts, Muon, reversible midpoint
  stack with pre-LN inside `f` (classic Post-LN cannot be inverted, see `models/gpt.py`). This is `configs/lab_small.yaml`.
- **Model: NanoGPT, driven only by config.** When G5 (Architecture) sends their config details and
  harness, we add **a YAML file, not code**.
- **Embedding: Kronecker Embedding v1** (input side). KE2 (from G4): the output side is decided 14 Nov; the KE2
  decode pair (`models/ke2_decode.py` vs G4's `decode_ref.py`) is in `correctness/test_ke2_decode.py`.
- Lab parts to time: attention, MLP or MoE (top-2 of 8), reversible midpoint stack, KE v1, Muon.

---

## 1. Gate 1: what we must deliver

| Deliverable | File |
|---|---|
| Timing harness | `perfkit/timer.py`, `perfkit/bench.py` |
| Correctness framework + test pairs | `correctness/framework.py`, `test_ke.py`, `test_attention.py`, `test_moe.py`, `test_reversible.py` |
| Baseline with GPU name and commit ID | `results/baseline_v0.md` |
| Five slowest parts, each with a trace and one week-2 idea | `results/slowest5_v0.md` |
| System metrics + MFU method to G8 | `docs/METRICS.md` + hand-off meeting |

Gate 1 checks:
1. `pytest teams/06-performance/correctness -q` gives 0 failures.
2. The lead multiplies one fast output by 1.01, and a test fails.
3. Two baseline runs agree within 5% for each part.
   How we check it: every part at level <= 2 is compared; a part fails when it is more than 5% apart and its
   medians are more than `report.min_abs_ms` (0.05 ms) apart. Below 50 us a gap is CUDA event / launch jitter:
   in a GPU-bound step the `data` part is a 30 us copy queued behind other work (g5_v0 on A100: 0.031 vs
   0.035 ms, 11.8% apart, every other part within 0.4%). Proposed 29 Oct; needs TSAI agreement.

---

## 2. Step 0: freeze the interfaces

One skeleton PR: folders, the interfaces below, the lab model, stubs with their signatures, these rules.
After it, every module is built against the interfaces, never against another module's internals,
so modules can be built in parallel and in any order.

### 2.1 Folder layout

```
teams/06-performance/
  configs/        base.yaml, lab_small.yaml, tiny_cpu.yaml
  models/         config.py, gpt.py, moe.py, reversible.py, ke1.py, optim.py
  dataio/         prepare_fineweb.py, loader.py
  perfkit/        timer.py, regions.py, bench.py, metrics.py, report.py, roofline.py, schema.py, smoke.py
  correctness/    framework.py, test_framework.py, test_attention.py, test_ke.py, test_moe.py, test_reversible.py,
                  test_model.py
  results/        baseline_v0.md, slowest5_v0.md, runs/
  docs/           PLAN.md, PARTS.md, METRICS.md
```

Shared files (`config.py`, `regions.py`, `schema.py`, the `framework.py` registry)
are interfaces: change them only with the whole team agreeing.

### 2.2 Model config (`models/config.py`)

```python
@dataclass
class ModelConfig:
    vocab_size: int = 131072     # exact BrahmicTokenizer-131K vocabulary
    block_size: int = 512        # sequence length
    n_layer: int = 9
    n_head: int = 8
    n_kv_head: int = None        # GQA; None = n_head
    n_embd: int = 256
    mlp_ratio: int = 4
    embedding: str = "table"     # table | ke1
    ke_max_pos: int = 32         # KE v1: 256 bytes x 32 positions = 8192-dim codec
    ffn: str = "mlp"             # mlp | moe
    moe: dict = {num_experts: 8, top_k: 2, num_shared: 0, dispatch: "loop"}   # lab: num_shared 1
    reversible: bool = False     # True: midpoint stack
```

- YAML files with `inherit: base.yaml`. Command-line overrides: `model.n_layer=4 run.batch_size=16`.
- **Rule:** no number about the model is hard-coded anywhere else. Everything reads `cfg`.
- G5's config = `configs/g5_*.yaml` (`g5_v0.yaml` from config_v0, 29 Oct). If G5 has a field we lack, add it to
  `ModelConfig` with a default (done for `attn_impl`, `mtp`; `moe.dispatch: dense`, `ke_mode: sparse` are new values).

### 2.3 Model (`models/gpt.py`)

```python
model = GPT(cfg, token_bytes=None)            # token_bytes only for embedding=ke1
logits, loss = model(idx, targets)            # idx, targets: [B, T] int64
total, active = model.param_counts()          # active = what one token multiplies with (for MFU)
model.moe_layers()                            # list of MoE modules (bias update after each step)
```

### 2.4 Timing markers (`perfkit/regions.py`)

```python
with region("block3.attn", level=2):  ...     # a named piece of work
x = mark(x, "block3")                         # backward boundary: when grad reaches x
with phase("bwd.recompute"):  ...             # prefix, so recompute is not counted as forward
```

- Names: `<phase>:<part>`, e.g. `fwd:block3.attn.qkv`, `bwd:head+loss`, `opt:opt.muon`.
- Levels: 1 = phases (data, fwd, bwd, opt). 2 = per layer and big parts (embed, blockN, head+loss, optimizer).
  3 = inside a layer (qkv, sdpa, router, each expert).
- **When timing is off, `region` returns a shared do-nothing context and `mark` returns its input.**
  The model code keeps its markers forever.

### 2.5 Timing rules

1. CUDA events only. **No `synchronize`, `.item()` or `print` inside a timed step.** Read events after the steps.
2. Warmup steps first (default 10). Then measure (default 50 steps).
3. Report median, p10, p90, IQR. Never the mean.
4. Measure the overhead: steps alternate timing on and off. Above 2%, the numbers are flagged.

### 2.6 Run JSON (written by bench, read by report and G8)

```
meta:      gpu, gpu_spec, torch, cuda, driver, device, commit, dirty, config_file, overrides, tag, time
           [reconstruction_err]                                  cuda / driver are None on CPU
           [dtype, dtype_requested, notes, batch_size, seq_len, tokens_per_step, vocab_size, data_source,
            tokenizer {name, revision, sha256, vocab_size}, lr_schedule, python, bench_version, timing,
            loss_after_first_step, loss_end, compute_capability, sm_count, gpu_mem_gb, gpu_count,
            telemetry_start, telemetry {after_warmup, end}]
config:    run, data, model, optim, bench                        the full merged config
clean:     step_ms {median,p10,p90,iqr,n}, data_ms {median,p10,p90,iqr,n}
           [raw_step_ms, events_created_during_steps]
layer:     level, overhead_pct, step_ms, step_ms_recorder_off,
           regions { "<phase>:<name>": {median,p10,p90,iqr,n,level,calls_per_step} }  [each region: pass]
           [data_host_ms, data_note, events_per_step, steps, events_created_during_steps,
            events_created_while_measuring,
            inner_pass {level, overhead_pct, step_ms, step_ms_recorder_off, events_per_step,
                        events_created_during_steps, events_created_while_measuring}]
trace:     trace_file, top_ops, gpu_busy_pct, kernels_per_step
           [steps, host_syncs_per_step, sync_sites, parts_note,
            parts [{part, ms, pct, instances, level, region, file, bytes, calls_in_step, calls_kept, gpu_events}]]
[system]:  source, interval_ms, during, samples,                 CUDA only, sampled through the clean steps
           metrics {gpu_util_pct, mem_util_pct, mem_used_mib, sm_clock_mhz, mem_clock_mhz, power_w, temp_c:
                    {min, median, max, mean}}  [slowdown_samples, slowdown_reasons]
metrics:   tokens_per_step, tokens_per_s, mfu, params_total, params_active, model_flops_per_token,
           peak_tflops                                           [hfu, hw_flops_per_token, model_tflops_per_s]
[memory]:  peak_allocated_gb, peak_reserved_gb
[goodput]: badput_s {startup, first_step, warmup}                [time_goodput, token_goodput_per_s: to write]
[checkpoint]: save_ms, to_cpu_ms, write_ms {stats}, size_mb, write_mb_per_s, level, repeats, where
[comm]:    moe_a2a {bytes_per_call, payload_mb_per_call, payload_dtype, calls_per_layer, calls_per_step, dtype,
           level, measured, [reason] | [world_size, backend, per_call_ms, per_call_ms_slowest_rank,
           per_step_ms, events_created_during_steps (max across ranks)]}
[warnings]: ["..."]                                               rule checks and noisy parts, in words
```

`perfkit/schema.py` (`REQUIRED`, `REGION_KEYS`) is the source of truth; this is a copy. `[...]` = optional.
meta, config, clean and metrics are always there; layer and trace only when that mode ran; system only on CUDA.
`bwd:stack` is the whole stack backward in both stacks, like `fwd:stack` (`perfkit/bench.py` step 6).
New fields can be added; existing ones are not renamed without telling the team and G8.

### 2.7 Test pair (`correctness/framework.py`)

```python
@pair("moe.loop", tol=1e-4)
def _(device):
    ...
    return Case(ref=ref_fn, fast=fast_fn, inputs=[x], params_ref=[...], params_fast=[...])
```

- The reference is written **independently** from the definition, in fp64. Never by calling the fast code.
- Error = `||fast - ref|| / ||ref||` on outputs and gradients. (Do not use `torch.allclose` defaults: in bf16
  they let a 1% error pass.)
- Tolerance per dtype (`TOLERANCES` in `framework.py`, measured on the full lab model): fp32 outputs 1e-5,
  gradients 5e-5; bf16 outputs 1e-2, gradients 3e-2; fp8 is measured and added when an fp8 path exists.
  fp32 tolerances are below 1e-2 always.
- `test_framework.py` multiplies every fp32 pair's fast output by 1.01 and expects a failure (a bf16 tolerance
  cannot see a 1% error, so bf16 pairs are not in this drill).
- Each pair's worst error is printed at the end of every pytest run (`conftest.py`).

---

### 2.8 The skeleton (what step 0 ships)

Interfaces: `models/config.py`, `perfkit/regions.py` (the API; the recorders were stubs, now in `perfkit/timer.py`), `perfkit/schema.py`
(run JSON + `example_run()`), `correctness/framework.py` (the pair registry; the checks were stubs),
`correctness/conftest.py`.

The lab model, so it runs end to end from day one (MVPs, verified against fp64 references and on an A100):
`models/gpt.py` (GQA, pre-LN, MoE / KE / reversible plug in), `models/ke1.py` (ported from
theschoolofai/kronecker-embeddings), `models/moe.py`, `models/reversible.py`, `models/optim.py` (Muon),
`dataio/loader.py`, `dataio/prepare_fineweb.py`, `perfkit/smoke.py`.

Every other file is a stub: the right signatures, a docstring with the MVP, extras and
"done when". Calling a stub raises `NotImplementedError`, and `correctness/conftest.py` reports such a
test as **skipped** (named, with `-rs`), not failed. So tests can be written before the code they test, and
a module's tests start running the moment it is written. At Gate 1 no skip may be left.

## 3. What we have, and what is left

Status "Stub": the file raises `NotImplementedError`, and tests that call it are skipped until it exists. Each stub's
docstring has the full MVP, the extras and "done when".

### Model and data

| File | Role | Status |
|---|---|---|
| `configs/*.yaml` | `base` (every setting; plain NanoGPT on its own), `lab_small` (the lab model), `tiny_cpu` (laptop) | Implemented |
| `models/config.py` | `ModelConfig`, YAML with `inherit:`, `key=value` overrides; unknown or bad keys fail early | Implemented |
| `models/gpt.py` | GQA attention, pre-LN blocks, plain or reversible stack, untied head for KE, `ce_impl` torch / chunked loss | Verified against fp64 references and on an A100 |
| `models/ke1.py` | Kronecker Embedding v1 (ported from theschoolofai/kronecker-embeddings), dynamic and cached modes, Brahmic source vocabulary byte table | Bit-identical to the upstream package |
| `models/moe.py` | 1 shared + top-2 of 8 routed experts, fp32 router, bias balancing, loop dispatch, routing replay | Verified against a token-by-token fp64 reference |
| `models/reversible.py` | Midpoint stack, custom backward (invert, recompute, local backward), `reconstruction_error` | Exact in fp64; bf16 reconstruction 2.5e-4 on an A100 |
| `models/optim.py` | Muon (2D block weights including routers) + AdamW (embedding parameters, head, norm gains, biases), Newton-Schulz FLOPs | Implemented; new optimizer split needs fresh timings |
| `dataio/prepare_fineweb.py` | Retokenizes raw FineWeb sample-10BT using the pinned GitHub tokenizer; uint32 shards | Brahmic preparation needs fresh validation |
| `dataio/loader.py` | Random windows from the shards (or synthetic), pinned copy, KE byte table | Run on real shards |

### Timing harness

| File | Role | Status |
|---|---|---|
| `perfkit/regions.py` (API) | `region(name, level)` marks a piece of work. `mark(x, name)` marks where the backward reaches a tensor. `phase("bwd.recompute")` prefixes names. When timing is off, these cost nothing. | Implemented |
| Markers in the model | `gpt.py`, `moe.py`, `ke1.py`, `reversible.py`, `optim.py` already call `region` and `mark` with the names in `PARTS.md` | In place |
| Recorders (`perfkit/timer.py`, re-exported by `regions.py`) | `EventRecorder`: puts a CUDA event at each region and syncs once after the step. `ProfilerRecorder`: trace mode. | Implemented |
| `perfkit/timer.py` | `StepTimer` (CUDA event pair per step), `stats()` (median, p10, p90, IQR), `sync()`, backward segments from `mark()` points, per-part aggregation, paired overhead, kernel summary, host-sync count | Implemented; no unit tests of its own yet (run end to end by `test_bench.py`) |
| `perfkit/bench.py` | The command-line tool: builds the model, warms up, runs clean, layer, trace, checkpoint and comm modes, writes the run JSON, `--against` 5% check | Implemented; run on CPU, GPU runs pending |
| `perfkit/schema.py` | The run JSON format, with `validate_run` and `example_run` | Implemented |
| `perfkit/metrics.py` | MFU, HFU, peak TFLOPS table, memory, GPU telemetry, GPU name, versions, commit | Implemented (goodput and NVML sampler are extras still to write) |
| `docs/METRICS.md` | The system metrics and the MFU method, for the G8 hand-off | Not written |
| `perfkit/report.py` | Markdown report, 5% two-run comparison, the baseline and slowest-5 files | Stub |
| `perfkit/roofline.py` | Predicts each part's time from FLOPs and bytes (compute / memory / launch bound) | Stub (extra, not needed for Gate 1) |
| `perfkit/smoke.py` | Stopgap: one CUDA event pair per whole step, no per-part numbers | Produced the A100 reference numbers |

### Correctness

| File | Role | Status |
|---|---|---|
| `correctness/framework.py` (registry) | `Case`, `Pair`, `@pair`, `pairs_in`, `to64`; `TOLERANCES` per dtype, fp32 always < 1e-2 | Implemented |
| `correctness/conftest.py` | `sys.path` (so the Gate command works from the repo root), `device` fixture, stub = skipped | Implemented |
| `correctness/framework.py` (checks) | `rel_err`, `check` (outputs + gradients, same seeded cotangent, one backward per side), `scaled(1.01)`, `RESULTS` error report | Implemented |
| `correctness/test_framework.py` | 1.01 on every pair, wrong gradient, NaN, wrong shape must all fail | Implemented |
| `correctness/test_attention.py` | Naive fp64 attention (GQA, both code paths) vs the model's attention: 7 pairs, causality, path agreement | Implemented; 7 pairs pass (errors ~3e-7) |
| `correctness/test_model.py` | The whole GPT (lab_small, base; fp32 and bf16) vs an fp64 model from the definitions, real Brahmic tokens | Implemented; catches 6 injected model bugs |
| `correctness/test_ke.py` | KE v1 built byte by byte in fp64 vs `ke1.py`, every mode; Brahmic source vocabulary byte table | Stub |
| `correctness/test_moe.py` | Token-by-token fp64 MoE vs every dispatch; routing replay (near-tie test); bias balancing | Stub |
| `correctness/test_reversible.py` | Custom backward vs plain autograd; inversion; bf16 reconstruction property test | Stub |

### Left for Gate 1, in order

1. ~~`timer.py` and `EventRecorder`, then `bench.py` clean and layer modes, end to end on CPU.~~ Done.
2. ~~`framework.py` checks and `test_framework.py` (the 1.01 drill)~~ Done. Then the test pairs: KE, MoE, reversible (attention is done).
3. `docs/METRICS.md`, for the G8 hand-off (`metrics.py` is done).
4. `report.py` with the 5% compare.
5. Two baseline runs on one GPU, then `results/baseline_v0.md` and `results/slowest5_v0.md`.

Counters describe Python Event objects, not guaranteed CUDA runtime handle creation.
StepTimer counters remain cumulative across resets; recorder-pool counts are separate.
Communication reports the maximum allocation count across distributed ranks.
