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

`optim.name=polar_muon_adamw` selects an additional combined optimizer using Polar Express Muon + AdamW updates (including row equilibration, Muon+ norm correction, variance reduction, and cautious Muon decay). It is a separate comparison path; it does not replace the current `muon` implementation. This integration uses ordinary PyTorch ops; optimizer-owned collectives and sharded optimizer state are not integrated. It can run under DDP, which synchronizes gradients outside the optimizer.

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
- **Left:** the KE2 pair (waits for G4's decode reference), the MoE all-to-all timed on 2 GPUs (`torchrun`),
  `perfkit/roofline.py`.

**Distributed optimizer:** DDP synchronizes gradients while the existing Muon/AdamW optimizers keep replicated state on each rank. `build_optimizer` supports a DDP-wrapped model, and MoE expert-balance counts and GradScaler overflow decisions are synchronized across ranks. This path is covered by a two-rank CPU correctness test. ZeRO-style optimizer-state or parameter sharding is still a separate future backend.

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

**Distributed training (DDP)** uses the same model and Muon/AdamW updates on each rank, with DDP averaging
gradients. It replicates model parameters and optimizer state, so use it when the full model fits on each GPU.
`run.batch_size` is **per rank**; the run JSON records `world_size`, per-rank batch, global batch and slowest-rank
step time. The existing `comm` mode above measures only MoE all-to-all communication; use `ddp_bench` to time
forward, backward and optimizer steps together:

```bash
torchrun --standalone --nproc_per_node=2 -m perfkit.ddp_bench --config configs/lab_small.yaml --steps 20 --warmup 5 data.source=synthetic
torchrun --standalone --nproc_per_node=4 -m perfkit.ddp_bench --config configs/lab_small.yaml --steps 20 --warmup 5 data.source=synthetic
torchrun --standalone --nproc_per_node=8 -m perfkit.ddp_bench --config configs/lab_small.yaml --steps 20 --warmup 5 data.source=synthetic
```

`--nproc_per_node` can match any assigned GPU count. To use a subset, expose only those GPUs, for example
`CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 ...`. For a fixed global batch comparison,
set `run.batch_size` to `global_batch / world_size` in each run; leaving it fixed per rank instead measures scaling
with a growing global batch. The benchmark currently requires `run.compile=false`. When the target model no longer
fits on each GPU, evaluate FSDP or DeepSpeed ZeRO as a separate sharding backend; DDP alone does not shard model
parameters or optimizer state.

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
