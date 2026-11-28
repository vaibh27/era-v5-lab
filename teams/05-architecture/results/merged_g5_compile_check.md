# Merged G5 branch: Section A compile check

**Status:** Section A done on CPU (torch 2.13.0+cpu). GPU tests and A/Bs are still pending approval.
**Branch:** `g6/w4-attn`, base `67d06f10` (`origin/g6/dev`, docs-only ahead of `369fd0d1`).

All five week-4 branches merged, six conflicts resolved. Defaults are unchanged: every switch is off in
`dryrun2.yaml` and `teams/06-performance/configs/base.yaml` is untouched.

## Merges

| # | Branch | Merge commit | Conflicts |
|---|---|---|---|
| 1 | `g6/w4-g5-sdpa` | `2955a9cf` | none |
| 2 | `g6/w4-g5-step-caches` | `35d4cd3e` | none |
| 3 | `g6/w4-g5-moe-grouped-mm` | `2c8a8ca1` | `model.py`: kept `stacked()` caches, added `dispatch`; grouped path takes the cached stacks |
| 4 | `g6/w4-g5-launch-overhead` | `62bdb79f` | none |
| 5 | `g6/w4-g5-fused-norm-swiglu` | `c277d2c7` | `config.py`: combined all four checks; `model.py`: `silu_mul` only on the dense routed path, `grouped_experts` keeps `F.silu(hg) * hu` |

## Full G5 suite

```powershell
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
python -m pytest teams/05-architecture/tests -q
```

**234 passed, 14 skipped, 45 subtests passed** (225 s).

Without the MSVC environment two compile-model tests fail with `CppCompileError: ... omp.h: No such file or
directory`. That is the local MSVC include path, not the merge; with `vcvars64.bat` all four compile tests pass.

Environment notes:

- System Python 3.11, torch 2.13.0+cpu, has `F.grouped_mm`; its inductor build needs `vcvars64.bat`.
- `conda activate pytorch` is torch 2.6.0+cu118 with Triton 3.9.0. It has no `F.grouped_mm`, and torch's
  `_inductor` fails to import against that Triton (`AttrsDescriptor` moved). Do not use it for this check.
- The plan's torch 2.14.1 was not available locally; counts below may differ slightly on the lab version.

## Graph breaks

Setup: 2-layer fixture model (`configs/arms/t1.yaml` + `smoke_resolution.yaml`), every switch on:
`attn_impl=sdpa`, `norm_impl=fused`, `swiglu_impl=triton`, `moe.dispatch=grouped_mm`, `run.compile=true`.
The blocks run bf16 because CPU `grouped_mm` is bf16-only under fake tracing (see below); the embedding,
head and loss stay fp32/eager as in the trainer.

`torch._dynamo.explain(model.blocks[0])(hidden, valid)`:

| Configuration | Graphs | Graph breaks |
|---|---|---|
| compile only (defaults) | 15 | 14 |
| + sdpa | 15 | 14 |
| + sdpa + fused kernels | 15 | 14 |
| + sdpa + fused + grouped_mm (bf16) | 15 | 14 |

Every reported break points into the step-cache branch:

- `torch.is_inference_mode_enabled()` in `GQA.rotary_angles` (`model.py:139`) and `MoE.stacked`
  (`model.py:274`). Dynamo reports this as fundamentally unsupported. The dynamo counter records **26**
  breaks per training step across both blocks (forward and backward).
- `w.data_ptr()` in the `stacked()` cache key (`model.py:275`). On the second optimizer step the key changes,
  and the counter adds **3** `Data pointer comparison` breaks.

`grouped_mm` adds no graph breaks of its own, and inductor does generate code for `aten._grouped_mm`
(the `aten_mm_info` counter shows it in every step). But `F.grouped_mm` with fp32 fails Dynamo's fake-tensor
check (`Expected inputs of BF16 type`), so a compiled CPU run must use bf16 blocks, exactly like CUDA autocast.
Eager fp32 `grouped_mm` on CPU is unaffected.

## Recompiles

Same model, `torch.compile` on each block, AdamW steps at the trainer's shape (3 x 8):

| Point | `frames.total` | `stats.unique_graphs` |
|---|---|---|
| after step 1 | 48 | 32 |
| after step 2 | 55 | 38 |
| after step 3 | 55 | 38 |
| after 8 -> 5 token step | 66 | 48 |
| back to 8 tokens | 66 | 48 |

Step 2 is where the optimizer bumps every parameter's `_version`, invalidating the `stacked()` cache key.
Dynamo recompiles the `stacked` frame every step until it hits `recompile_limit=8` on
`torch_dynamo_resume_in_stacked_at_278`, then stops growing (the frame falls back to eager). Over the whole
run `TORCH_LOGS=recompiles` counted **13 recompile events**, 11 of them in `stacked` or its resume frames.
The 8 -> 5 token change adds 11 frames as expected for a new shape; going back to 8 adds none.

## Reading

- The merge is correct: full CPU suite passes and every switch still defaults to the old behaviour.
- Compile and the step caches do not compose yet. The cache branch inserts a fundamental graph break at every
  layer and recompiles after every optimizer step, so `run.compile: true` currently buys little with the caches
  present. This was the open item in `launch_overhead.md`; it is now measured.
- Likely fix direction (new branch, not done here): keep cache-key work out of the traced region, e.g. compute
  the stacked weights in an eager step hook or a custom op with a fake implementation, and replace
  `torch.is_inference_mode_enabled()` with a value passed in from outside the compiled block.
- CUDA may change the picture: Triton SwiGLU/grouped_mm and inductor's CUDA path need the same measurement
  once a GPU run is approved.

## Reproduce

```python
exp = torch._dynamo.explain(model.blocks[0])(hidden_bf16, valid)
print(exp.graph_count, exp.graph_break_count, exp.break_reasons)

# recompiles: run optimizer steps, then inspect
from torch._dynamo import utils
print(utils.counters["frames"], utils.counters["stats"])

# or capture the log
# $env:TORCH_LOGS = "graph_breaks,recompiles"
```
