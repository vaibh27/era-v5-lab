# Team 06-performance

Put your work in a **topic folder**: `teams/06-performance/<topic>/<file>`, for example `teams/06-performance/harness/run.py`.
Team documents named in the week plan (for example `benchmarks_v1.md`) may sit directly in the team folder. The team lead decides the topic names. Do not make personal folders. Do not make new folders under `teams/`.
Open a pull request; the lead reviews and merges it. The pr-guard check must pass.

## What this is

Week-1 performance work for the lab model: a timing harness that times every part of the training step,
a correctness framework that proves fast code gives the right result, and the first baseline.
Everything is driven by config: a new model is a new YAML file in `configs/`, not new code.

## The lab model (`configs/lab_small.yaml`)

| Part | Setting |
|---|---|
| Data | Raw FineWeb sample-10BT retokenized with BrahmicTokenizer-131K, random 512-token windows |
| Tokenizer | [BrahmicTokenizer-131K source repo](https://github.com/theschoolofai/BrahmicTokenizer-131K), exactly 131,072 IDs |
| Embedding | Kronecker Embedding v1: fixed byte codec (256 x 32 = 8192) + `Linear(8192 -> 256)` |
| Attention | GQA: 8 query heads, 2 key/value heads, width 256 |
| MoE | 1 shared + top-2 of 8 routed experts (256 -> 1024 -> 256), bias balancing, fp32 router |
| Stack | Reversible midpoint, 9 layers, pre-LN inside `f`; the backward rebuilds each layer's input |
| Optimizer | Muon on 2D block weights including routers; AdamW on embedding parameters (including KE projection), output head, norm gains and biases |
| Size | Reported from the selected config at runtime with the full 131,072-token vocabulary |

`configs/base.yaml` on its own is the plain NanoGPT comparison model (table embedding, dense MLP, plain stack, AdamW).
`configs/tiny_cpu.yaml` is the lab model shrunk for a laptop.

`optim.name=polar_muon_adamw` selects an additional single-process combined optimizer using Polar Express Muon + AdamW updates (including row equilibration, Muon+ norm correction, variance reduction, and cautious Muon decay). It is a separate comparison path; it does not replace the current `muon` implementation. This first integration uses ordinary PyTorch ops; fused kernels, distributed collectives, and sharded optimizer state are not integrated.

`model.moe.dispatch=loop|sorted` selects the reference per-expert gather or the sorted dispatch. The sorted path orders token/expert assignments, packs them into a padded per-expert buffer and runs each expert MLP as one batched matmul. Its scatters and gathers use unique indices, so gradients are bit-identical across runs on CUDA. It reads back only the largest per-expert route count (one host sync per layer call). For an idle expert it produces zero gradient tensors where the loop leaves gradients as `None`; this can change optimizer-state updates, so `loop` remains the default pending an optimizer-safe treatment and server benchmark. Run the dispatch checks with `python -m pytest correctness/test_moe.py -q`.

`optim.muon_impl=loop|batched` selects the per-matrix Muon reference or shape-batched Newton–Schulz path. The benchmark comparison permits this implementation switch while still requiring the same model, optimizer settings, device, dtype, and data. Compare both paths from one clean commit (A/B commands in "Timing" below).

## Folder layout

```
configs/       base.yaml (every setting; plain NanoGPT on its own), lab_small.yaml, tiny_cpu.yaml
models/        config.py, gpt.py, ke1.py, moe.py, reversible.py, optim.py
dataio/        prepare_fineweb.py, loader.py
perfkit/       regions.py, timer.py, bench.py, metrics.py, report.py, roofline.py, schema.py, smoke.py
correctness/   framework.py and the test files
results/       run JSONs, baseline_v0.md, slowest5_v0.md
docs/          PLAN.md (interfaces, what is done and left), PARTS.md (every part we time)
```

## Status

- **Done for Gate 1:** the lab model and data pipeline on the Brahmic vocabulary; the step timer, event recorders
  and bench (`perfkit/timer.py`, `perfkit/bench.py`) with clean, layer, trace, checkpoint and comm modes, a small
  trace per slowest part and GPU system metrics; `--repeat` with each run in its own process; metrics and MFU
  (`perfkit/metrics.py`, `docs/METRICS.md`); the correctness framework with the 1.01 drill and determinism check,
  and test pairs for attention, KE, MoE, reversible and the whole model (fp32 and bf16, plus chunked loss).
- **Baseline v0 (A100-SXM4-40GB, commit `f1f4986`):** 507 ms per step, MFU 3.3%; two runs agree within 3.6% on
  every gated part ([`results/baseline_v0.md`](results/baseline_v0.md)). The five slowest parts, each with a trace
  and a week-2 idea: [`results/slowest5_v0.md`](results/slowest5_v0.md).
- **G4 / G5 alignment (29 Oct):** `configs/g5_v0.yaml` holds G5's config_v0 settings and how their harness runs
  them (eager attention, a t+2 MTP head, every expert on every token, KE cached); sizes stay lab_small until TSAI
  fills them on 31 Oct. New switches: `model.attn_impl: eager`, `model.mtp`, `model.moe.dispatch: dense`,
  `model.ke_mode: sparse`. The KE2 pair: `models/ke2_decode.py` against G4's `decode_ref.py` on their
  131072-row table (`correctness/test_ke2_decode.py`).
- **G5's own harness timed:** `python -m perfkit.g5_harness` imports `teams/05-architecture/harness/train.py`
  (unchanged) and times its four parts (data, forward, backward of both losses, AdamW step) with CUDA events;
  same losses as their `run()` to the bit (`correctness/test_g5_harness.py`). `--set key=value` for sizes.
- **Out of memory:** with `run.batch_size: auto` the bench halves B and retries until the run fits (the
  fallback is a run note); an explicit `run.batch_size` ends with exit code 3 as before.
- **Left:** the MoE all-to-all timed on 2 GPUs (`torchrun`), `perfkit/roofline.py`.

**Distributed optimizer follow-up:** the current Muon/AdamW implementation is local-process only. A DDP integration is implementable follow-up work: make parameter selection robust to the DDP `module.` prefix (or build from the unwrapped module), handle MoE experts unused on a rank, and add a multi-process correctness/smoke test. The current optimizer should not be treated as tested for DDP or ZeRO-style sharding.

The full table is in [`docs/PLAN.md`](docs/PLAN.md) section 3. Every part we time is in [`docs/PARTS.md`](docs/PARTS.md).

## How to run

Every command runs from `teams/06-performance/`. Give the GPU name with every timing you share.

**1. Get the code**

```bash
git clone <REPO_URL> era-v5-lab          # the URL of the integration branch is posted in the G6 group
cd era-v5-lab && git checkout g6/dev && cd teams/06-performance
```

**2. Install.** Laptop (venv outside the repo): `python -m venv ~/.venvs/g6 && source ~/.venvs/g6/bin/activate`,
then `pip install -r perfkit/requirements.txt`. Colab (torch is already installed, do not reinstall it):
`pip install -q tokenizers datasets pyyaml pytest`.

**3. Tests** (CPU is fine, ~5 s)

```bash
python -m pytest correctness -q -rs              # all pass; unfinished stubs are SKIPPED and named
```

At Gate 1 no skip may be left (`-rs` lists them).

**4. Tokenizer and data:** the tokenizer is loaded exclusively from `tokenizer.json` in the
[source repo](https://github.com/theschoolofai/BrahmicTokenizer-131K), pinned to commit
`7dfcf398172c825c6689ad578e8b484ca5f57bc9` and verified by SHA-256. It is never loaded from
Hugging Face. The default downloads that GitHub file into `~/.cache/era-perf/BrahmicTokenizer-131K/`.
For an offline run, clone the source repo and set `data.tokenizer_path=/path/to/BrahmicTokenizer-131K/tokenizer.json`.
Tests accept the same path via `BRAHMIC_TOKENIZER_PATH`.

Data preparation streams raw text from FineWeb sample-10BT and encodes it with that artifact.
New shards use uint32 (IDs exceed 65,535) and a tokenizer metadata manifest. Existing GPT-2 shards
are incompatible and must be prepared again. Data stays outside git, by default in
`~/.cache/era-perf/fineweb-brahmic131k/`.

```bash
python -m dataio.prepare_fineweb                 # val + 1 train shard, 1M tokens each (~8 MB total)
# Offline tokenizer: append --tokenizer-path /path/to/source-checkout/tokenizer.json
```

For a quick check without FineWeb, add `data.source=synthetic`. Synthetic runs still use the
real Brahmic vocabulary and KE byte table, including special tokens and raw partial UTF-8 bytes.

**5. Smoke test the model** (not the baseline: Gate 1 numbers come from `perfkit.bench` on committed code)

```bash
python -m perfkit.smoke                                      # lab model: configs/lab_small.yaml
python -m perfkit.smoke --config configs/base.yaml          # plain NanoGPT for comparison
python -m perfkit.smoke --config configs/tiny_cpu.yaml       # laptop, seconds; numbers mean nothing
python -m perfkit.smoke model.ce_impl=chunked                # if out of memory; then lower --batch (T4 default is 8)
python -m perfkit.smoke --steps 500 --warmup 0               # longer training run (every step is a real train step)
```

Precision and batch come from the GPU: bf16 and B=32 on A100 / L4 / H100, fp16 and B=8 on T4 / P100 / V100 (the bench uses the same rule through `run.batch_size: auto`).
`key=value` overrides work for any key of `configs/base.yaml`. A good run: reconstruction error below
1e-3 (`OK`), loss starting at about ln(131072) = 11.78 and falling, p10 / p90 within a few % of the median,
`all weights finite: True`.

**6. Training schedule:** timing runs use a constant learning rate by default. To inspect training
loss with cosine decay and optional linear warmup, add optimizer overrides; schedule duration follows
the run's `--steps` plus timing warmup steps.

```bash
python -m perfkit.smoke --steps 20 --warmup 5 optim.schedule=cosine optim.warmup_steps=3
```

Compare the Polar Express MuonAdamW update with the current hybrid optimizer using the same model and data:

```bash
python -m perfkit.smoke --steps 20 --warmup 5 optim.name=muon
python -m perfkit.smoke --steps 20 --warmup 5 optim.name=polar_muon_adamw
```

Timing results for the Brahmic vocabulary: [`results/baseline_v0.md`](results/baseline_v0.md).

**7. Bench** (per-part timing; Gate 1 numbers come from here, on committed code)

```bash
python -m perfkit.bench --config configs/tiny_cpu.yaml                      # laptop: proves it runs; writes to ~/.cache/era-perf/runs-smoke/
python -m perfkit.bench --config configs/lab_small.yaml --level 3 --tag a                              # first baseline run
python -m perfkit.bench --config configs/lab_small.yaml --level 3 --tag b --against results/runs/a.json # 5% check
torchrun --nproc_per_node 2 -m perfkit.bench --config configs/lab_small.yaml --modes comm               # 2+ GPUs
```

A/B of an implementation switch (`model.moe.dispatch`, `optim.muon_impl`): same commit, GPU, batch, dtype and data, only the switch changes. `--against` accepts the switch, still checks everything else, and labels the verdict `(A/B)`: a gated part outside 5% there is the speedup being measured, not a failed check.

```bash
python -m perfkit.bench --config configs/lab_small.yaml --modes clean,layer --level 3 --tag moe_loop model.moe.dispatch=loop
python -m perfkit.bench --config configs/lab_small.yaml --modes clean,layer --level 3 --tag moe_sorted --against results/runs/moe_loop.json model.moe.dispatch=sorted
python -m perfkit.bench --config configs/lab_small.yaml --modes clean,layer --level 3 --tag muon_loop optim.muon_impl=loop
python -m perfkit.bench --config configs/lab_small.yaml --modes clean,layer --level 3 --tag muon_batched --against results/runs/muon_loop.json optim.muon_impl=batched
```

Read the clean step median, tokens/s, and the level-3 `blockN.moe.dispatch` or `opt.muon.ns` rows (information only). Keep the default `report.gate_level=2`: the 5% rule covers the step parts and whole blocks, not the parts inside a block (instructor rule 3a). Use real data: the sorted dispatch pads every expert to the busiest one, and synthetic tokens hide routing skew.

The 5% rule itself is two runs of one implementation. Before switching a default, each side must pass it:

```bash
python -m perfkit.bench --config configs/lab_small.yaml --level 3 --repeat 2 --tag moe_sorted_rep model.moe.dispatch=sorted
python -m perfkit.bench --config configs/lab_small.yaml --level 3 --repeat 2 --tag muon_batched_rep optim.muon_impl=batched
```

Defaults (`configs/base.yaml`, `bench:`): all modes (clean, layer, trace, checkpoint, comm), 150 measured steps
after 10 warmup, 10 checkpoint saves after one warm save that is not counted. `run.batch_size: auto` uses the
same rule as smoke: B=32 where bf16 is native (A100, L4, H100), B=8 otherwise (T4: B=32 runs out of memory);
the run JSON records the B it used, and `--against` refuses two runs with different B. On one GPU the all-to-all
is not timed (its payload is recorded); time it under `torchrun` on 2+ GPUs (Kaggle T4 x2). A level-3 run takes
about 8 minutes on an A100. Run the two baselines back to back in one session.

Each run writes `results/runs/<tag>.json` (the format is in `docs/PLAN.md` 2.6) and prints the parts at level <= 2
(gated) and the parts inside a block (information only). Read the `[warn]` lines: a dirty tree, the wrong vocabulary,
recorder overhead above 2% or a throttled GPU make the numbers not count. Exit code 2 means `--against` found a part
outside 5%.

**Colab** (Runtime → Change runtime type → GPU), one cell each so a failing step is easy to see:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
!git clone -q <REPO_URL> /content/era-v5-lab
%cd /content/era-v5-lab
!git checkout -q g6/dev
%cd teams/06-performance
!pip install -q tokenizers datasets pyyaml pytest
```
```python
!python -m dataio.prepare_fineweb
```
```python
!python -m pytest correctness -q -rs
```
```python
!python -m perfkit.smoke
```

Reference (smoke, A100-SXM4-40GB, bf16, B=32): lab_small 511 ms/step, lab_small with `optim.name=adamw`
358 ms, base 115 ms; peak memory ~25 GB for all three (mostly the fp32 logits of the 131K vocabulary).
Use `%cd`, not `!cd` (`!cd` changes the folder for that line only). Colab deletes `/content` and the data
cache when the session ends.
