"""Time each part of the training step and write the run JSON.  [implemented]

    cd teams/06-performance
    python -m perfkit.bench --config configs/lab_small.yaml [--tag name] [--against run.json] [key=value ...]

Examples
    python -m perfkit.bench --config configs/tiny_cpu.yaml                    # laptop: proves it runs (~30 s)
    python -m perfkit.bench --config configs/lab_small.yaml --level 3 --modes clean,layer,trace,checkpoint
    python -m perfkit.bench --config configs/lab_small.yaml --level 3 --against results/runs/<first>.json
    torchrun --nproc_per_node 2 -m perfkit.bench --config configs/lab_small.yaml --modes comm   # 2+ GPUs

Flags --modes, --level, --steps, --warmup are shortcuts for bench.modes=..., bench.level=..., and so
on; any key of configs/base.yaml can be overridden as key=value. Every override is written to meta.

What a run does (the numbered steps of the original stub)
  1. load_config + model_config; device (run.device auto: cuda, else mps, else cpu); seed everything.
  2. Build Loader, GPT (with the KE byte table when embedding == ke1), build_optimizer, autocast for
     bf16 / fp16, GradScaler for fp16. bf16 on a GPU without native bf16 (T4 / P100 / V100) falls back
     to fp16 and says so in meta (DECISIONS B3).
  3. One training step = data, fwd, bwd, opt as level-1 regions; no sync and no .item() inside it.
     Optimizer order (fp16 needs it): unscale, clip, step, scaler.update, MoE bias update, zero_grad.
  4. Reversible stack: reconstruction_error on one batch under the run dtype (meta.reconstruction_err,
     warning above 1e-3). Then the first step (compile / autotune, timed as startup) and the warmup.
  5. clean: one CUDA event pair per step (StepTimer), nothing else. Step time, tokens/s and MFU come
     from here. The data fetch is timed on the host (perf_counter, no sync).
  6. layer: steps ALTERNATE recorder off / on, so overhead_pct comes from neighbouring steps.
     Backward parts are cut from the mark() points (perfkit.timer.backward_segments): bwd:head+loss,
     bwd:block{i}, bwd:stack (the whole stack in both stacks), bwd:embed. The reversible backward
     times its own blocks: bwd:block{i}, and at level 3 bwd.invert / bwd.recompute / bwd:block{i}.grad.
     Two passes when bench.level >= 3: the gated parts (level <= 2) come from a level-2 pass, the
     parts inside a block from a level-3 pass, so the many level-3 events cannot slow the gated
     numbers. Each region keeps its level ("pass" says which pass measured it).
  7. Metrics (perfkit.metrics when written, else the same formulas here), run JSON to
     <bench.out_dir>/<tag>.json, Markdown from perfkit.report.render when written.
     perfkit.schema.validate_run must return [] (exit code 1 otherwise).

Modes (bench.modes; clean always runs: meta, config, clean and metrics are required)
  clean       step time, data time, memory, tokens/s, MFU / HFU
  layer       every part of the step at its level, recorder overhead
  trace       torch.profiler over bench.trace_steps steps with the region names: Chrome trace in
              ~/.cache/era-perf/traces/ (outside the repo), top GPU ops, GPU busy %, kernels per step,
              host syncs per step and the source lines that cause them
  checkpoint  save model + optimizer + scaler state: ms (GPU -> CPU copy, write + fsync), MB, MB/s.
              Written to ~/.cache/era-perf/ckpt/ and deleted after.
  comm        MoE expert-parallel all-to-all with this config's real payload. Needs 2+ GPUs under
              torchrun (Kaggle T4 x2 works); on one GPU the payload and calls per step are recorded
              and the time is marked as not measured.

Instructor rules this follows (G6 group, 25 Oct)
  1a  Timings must use the real vocabulary: BrahmicTokenizer-131K has 131,072 tokens (the KE table
      depends on it). meta.vocab_size records it and a run with GPT-2 data says so.
  3a  The 5% rule applies to whole blocks and to the step parts (level <= report.gate_level); the
      parts inside a block are timed and reported (bench.level=3) but not gated.
  3b  MoE all-to-all time (comm) and checkpoint save time (checkpoint) are measured parts.
  Compute: everything runs on Colab, Kaggle or your own GPU; tiny_cpu.yaml runs on a laptop.

Exit codes: 0 ok, 1 run JSON incomplete, 2 --against found a part outside tol_pct, 3 out of memory,
4 no data.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import random
import re
import subprocess
import sys
import time
from collections import deque
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataio.loader import Loader  # noqa: E402
from models.config import load_config, model_config  # noqa: E402
from models.gpt import GPT  # noqa: E402
from models.optim import build_optimizer  # noqa: E402
from perfkit import timer as T  # noqa: E402
from perfkit.regions import phase, point, region, use_recorder  # noqa: E402
from perfkit.schema import validate_run  # noqa: E402

BENCH_VERSION = 1
CACHE = Path(os.environ.get("ERA_PERF_CACHE", "~/.cache/era-perf")).expanduser()
AMP = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": None}
MODES = ("clean", "layer", "trace", "checkpoint", "comm")


# ------------------------------------------------------------------------------ build

@dataclass
class Setup:
    cfg: dict
    mcfg: object
    device: torch.device
    dtype: str                    # the dtype actually used: bf16 | fp16 | fp32
    loader: Loader
    core: torch.nn.Module         # the GPT module (param_counts, moe_layers, state_dict)
    model: torch.nn.Module        # core, or torch.compile(core)
    opt: object                   # models.optim.Optimizers
    scaler: object = None
    grad_clip: float | None = None
    notes: list = field(default_factory=list)

    def autocast(self):
        amp = AMP[self.dtype]
        return torch.autocast(self.device.type, dtype=amp) if amp is not None else nullcontext()


def pick_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            name = "cuda"
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            name = "mps"
        else:
            name = "cpu"
    d = torch.device(name)
    if d.type == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("run.device=cuda but no CUDA GPU is visible")
        if d.index is None:
            d = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(d)
    return d


def native_bf16() -> bool:
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # torch < 2.4
        return torch.cuda.get_device_capability()[0] >= 8


def choose_dtype(want: str, device: torch.device, notes: list) -> str:
    want = want or "fp32"
    if want not in AMP:
        raise ValueError(f"run.dtype={want!r}: bf16 | fp16 | fp32")
    if device.type == "cuda" and want == "bf16" and not native_bf16():
        notes.append("run.dtype=bf16 but this GPU has no native bf16: using fp16 + GradScaler (DECISIONS B3)")
        return "fp16"
    if device.type == "cpu" and want == "fp16":
        notes.append("run.dtype=fp16 on CPU: using bf16 autocast instead")
        return "bf16"
    if device.type == "mps" and want != "fp32":
        try:
            with torch.autocast("mps", dtype=AMP[want]):
                pass
        except Exception:
            notes.append(f"run.dtype={want} autocast is not supported on this MPS build: using fp32")
            return "fp32"
    return want


def seed_everything(seed: int):
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build(cfg: dict, device: torch.device) -> Setup:
    mcfg = model_config(cfg)
    run, data = cfg["run"], cfg["data"]
    notes = []
    seed_everything(run["seed"])
    dtype = choose_dtype(run.get("dtype", "bf16"), device, notes)
    loader = Loader(data["source"], run["batch_size"], mcfg.block_size, device, data_dir=data.get("dir"),
                    vocab_size=mcfg.vocab_size, seed=run["seed"])
    core = GPT(mcfg, loader.token_bytes(mcfg.ke_max_pos) if mcfg.embedding == "ke1" else None).to(device)
    core.train()
    model = torch.compile(core) if run.get("compile") else core
    opt = build_optimizer(core, cfg["optim"])
    scaler = torch.amp.GradScaler("cuda") if (device.type == "cuda" and dtype == "fp16") else None
    clip = run.get("grad_clip") or None
    return Setup(cfg, mcfg, device, dtype, loader, core, model, opt, scaler, clip, notes)


# ------------------------------------------------------------------------------ one step

def train_step(s: Setup):
    """One training step. Regions cost nothing when no recorder is active (clean mode).
    Returns (host ms of the data fetch, the loss tensor). Nothing here waits for the GPU."""
    t0 = time.perf_counter()
    with region("data", 1):
        x, y = s.loader.next_batch()
    data_ms = (time.perf_counter() - t0) * 1e3
    with region("fwd", 1), phase("fwd"), s.autocast():
        loss = s.model(x, y)[1]          # with ce_impl=chunked the logits are None: use only [1]
    with region("bwd", 1), phase("bwd"):
        point("__bwd_start__")
        (s.scaler.scale(loss) if s.scaler is not None else loss).backward()
        point("__bwd_end__")
    with region("opt", 1), phase("opt"):
        if s.scaler is not None:
            with region("opt.unscale", 2):
                s.opt.unscale(s.scaler)
        if s.grad_clip:
            with region("opt.clip", 2):
                torch.nn.utils.clip_grad_norm_(s.core.parameters(), s.grad_clip)
        s.opt.step(s.scaler)
        if s.scaler is not None:
            s.scaler.update()
        with region("opt.moe_bias", 3):
            for m in s.core.moe_layers():
                m.update_bias()
        s.opt.zero_grad()
    return data_ms, loss.detach()


def reversible_check(s: Setup) -> float:
    from models.reversible import reconstruction_error
    x, _ = s.loader.next_batch()
    s.core.eval()                         # do not count these tokens for the MoE bias update
    try:
        with torch.no_grad(), s.autocast():
            emb = s.core.tok(x) + s.core.pos(torch.arange(s.mcfg.block_size, device=s.device))
            return reconstruction_error(emb, s.core.blocks)
    finally:
        s.core.train()


# ------------------------------------------------------------------------------ modes

def memory(device) -> dict | None:
    if device.type != "cuda":
        return None
    return {"peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30}


def run_clean(s: Setup, n: int):
    st = T.StepTimer(s.device)
    data, loss = [], None
    if s.device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(s.device)
    for _ in range(n):
        st.start()
        d, loss = train_step(s)
        st.stop()
        data.append(d)
    times = st.times_ms()
    clean = {"step_ms": T.stats(times), "data_ms": T.stats(data),
             "raw_step_ms": [round(t, 4) for t in times]}
    return clean, loss, memory(s.device)


def layer_pass(s: Setup, n: int, max_level: int, warm: int = 2) -> dict:
    """n steps with the recorder, n without, alternating. Returns the layer section for this level."""
    pool = T.EventPool() if s.device.type == "cuda" else None
    n_ev = 0
    for _ in range(warm):                 # the first recorded steps touch every region once
        rec = T.EventRecorder(s.device, max_level, pool)
        with use_recorder(rec):
            train_step(s)
        n_ev = rec.n_events()
        rec.collect()
    if pool is not None:
        pool.reserve(4 * n_ev + 64)       # no event is created inside a measured step
        created = pool.created
    on_t, off_t = T.StepTimer(s.device), T.StepTimer(s.device)
    order, pending, parts = [], deque(), []
    for i in range(2 * n):
        on = i % 2 == 1
        if on:
            rec = T.EventRecorder(s.device, max_level, pool)
            on_t.start()
            with use_recorder(rec):
                train_step(s)
            on_t.stop()
            pending.append(rec)
        else:
            off_t.start()
            train_step(s)
            off_t.stop()
        order.append(on)
        while pending and pending[0].ready():          # never waits: only finished steps
            parts.append(T.step_parts(pending.popleft().collect(), max_level))
    while pending:
        parts.append(T.step_parts(pending.popleft().collect(), max_level))
    on_ms, off_ms = iter(on_t.times_ms()), iter(off_t.times_ms())
    seq = [(o, next(on_ms) if o else next(off_ms)) for o in order]
    out = {
        "level": max_level,
        "overhead_pct": T.paired_overhead(seq),
        "step_ms": T.stats([ms for o, ms in seq if o]),
        "step_ms_recorder_off": T.stats([ms for o, ms in seq if not o]),
        "regions": T.aggregate(parts),
        "events_per_step": n_ev,
        "steps": n,
    }
    if pool is not None and pool.created > created:
        out["events_created_while_measuring"] = pool.created - created
    return out


def run_layer(s: Setup, n: int, level: int) -> dict:
    gated = min(level, 2)
    main = layer_pass(s, n, gated)
    for r in main["regions"].values():
        r["pass"] = gated
    if level < 3:
        return main
    inner = layer_pass(s, n, level)
    regions = {}
    for name, r in inner["regions"].items():          # inner order puts each part under its parent
        if r["level"] <= gated:
            if name in main["regions"]:
                regions[name] = main["regions"][name]
        else:
            r["pass"] = level
            regions[name] = r
    for name, r in main["regions"].items():
        regions.setdefault(name, r)
    out = {**main, "level": level, "regions": regions}
    out["inner_pass"] = {k: inner[k] for k in ("level", "overhead_pct", "step_ms", "step_ms_recorder_off",
                                               "events_per_step")}
    return out


def run_trace(s: Setup, steps: int, tag: str) -> dict:
    from torch.profiler import ProfilerActivity, profile, schedule

    tdir = CACHE / "traces"
    tdir.mkdir(parents=True, exist_ok=True)
    path = tdir / f"{tag}.json"
    cuda = s.device.type == "cuda"
    acts = [ProfilerActivity.CPU] + ([ProfilerActivity.CUDA] if cuda else [])
    rec = T.ProfilerRecorder(max_level=3)

    def export(p):
        try:
            p.export_chrome_trace(str(path))
        except Exception as e:  # a full disk must not lose the rest of the run
            print(f"[bench] could not write the trace: {e}")

    with profile(activities=acts, schedule=schedule(wait=0, warmup=1, active=steps, repeat=1),
                 on_trace_ready=export) as prof:
        for _ in range(steps + 1):
            with use_recorder(rec):
                train_step(s)
            prof.step()
    T.sync(s.device)
    if cuda:
        out = T.kernel_summary(T.profiler_device_events(prof), steps)
    else:
        ka = sorted(prof.key_averages(), key=lambda e: e.self_cpu_time_total, reverse=True)[:15]
        tot = sum(e.self_cpu_time_total for e in prof.key_averages()) or 1
        out = {"top_ops": [{"name": e.key[:160], "ms_per_step": e.self_cpu_time_total / 1e3 / steps,
                            "calls_per_step": e.count / steps, "pct": 100 * e.self_cpu_time_total / tot}
                           for e in ka],
               "gpu_busy_pct": None, "kernels_per_step": 0}
    out["trace_file"] = str(path)
    out["steps"] = steps
    if cuda:
        n, sites = T.count_syncs(lambda: train_step(s), root=ROOT)
        out["host_syncs_per_step"] = n
        out["sync_sites"] = dict(list(sites.items())[:12])
    return out


def _to_cpu(obj, memo):
    if isinstance(obj, torch.Tensor):
        key = id(obj)
        if key not in memo:                # tied weights stay one tensor (and are saved once)
            memo[key] = obj.detach().to("cpu")
        return memo[key]
    if isinstance(obj, dict):
        return {k: _to_cpu(v, memo) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_cpu(v, memo) for v in obj)
    return obj


def run_checkpoint(s: Setup, repeats: int, tag: str) -> dict:
    """What a training checkpoint holds: model, every optimizer's state, the scaler. Timed in two parts:
    GPU -> CPU copy, then serialise + write + fsync (so the page cache does not hide the disk)."""
    cdir = CACHE / "ckpt"
    cdir.mkdir(parents=True, exist_ok=True)
    path = cdir / f"{tag}.pt"
    to_cpu, write, total, size = [], [], [], 0
    try:
        for _ in range(repeats):
            T.sync(s.device)
            t0 = time.perf_counter()
            state = {"model": s.core.state_dict(),
                     "optim": {name: o.state_dict() for name, o in s.opt.named},
                     "scaler": s.scaler.state_dict() if s.scaler is not None else None}
            cpu = _to_cpu(state, {})
            T.sync(s.device)
            t1 = time.perf_counter()
            with open(path, "wb") as f:
                torch.save(cpu, f)
                f.flush()
                os.fsync(f.fileno())
            t2 = time.perf_counter()
            size = path.stat().st_size
            del state, cpu
            to_cpu.append((t1 - t0) * 1e3)
            write.append((t2 - t1) * 1e3)
            total.append((t2 - t0) * 1e3)
    finally:
        path.unlink(missing_ok=True)
    save = T.stats(total)
    return {"save_ms": save, "to_cpu_ms": T.stats(to_cpu), "write_ms": T.stats(write),
            "size_mb": size / 2**20, "write_mb_per_s": size / 2**20 / (T.stats(write)["median"] / 1e3),
            "level": 2, "repeats": repeats, "where": "local disk, ~/.cache/era-perf/ckpt (deleted after)"}


def a2a_plan(cfg: dict, mcfg, dtype: str) -> dict | None:
    """The expert-parallel all-to-all this config would do, from the config alone.

    Each rank sends every token's top-k copies (an upper bound: some stay on their own rank), so one
    call moves B * T * top_k * d values. Per MoE layer and step: dispatch + combine in the forward and
    their backward (4 calls); the reversible stack runs the forward twice more in backward (invert and
    recompute), which adds 4 (8 calls).
    """
    if mcfg.ffn != "moe":
        return None
    B, Tn = cfg["run"]["batch_size"], mcfg.block_size
    k, d = mcfg.moe["top_k"], mcfg.n_embd
    nbytes = 4 if dtype == "fp32" else 2
    per_layer = 8 if mcfg.reversible else 4
    return {"bytes_per_call": B * Tn * k * d * nbytes, "payload_mb_per_call": B * Tn * k * d * nbytes / 2**20,
            "calls_per_layer": per_layer, "calls_per_step": per_layer * mcfg.n_layer, "dtype": dtype,
            "level": 2}


def run_comm(plan: dict, device: torch.device, reps: int = 30) -> dict:
    """Time all_to_all_single with the plan's payload on every rank; reports the slowest rank."""
    import torch.distributed as dist
    W = dist.get_world_size()
    dt = {"fp32": torch.float32, "fp16": torch.float16}.get(plan["dtype"], torch.bfloat16)
    n = plan["bytes_per_call"] // (4 if dt == torch.float32 else 2)
    n -= n % W
    inp = torch.randn(n, device=device).to(dt)
    out = torch.empty_like(inp)
    for _ in range(5):
        dist.all_to_all_single(out, inp)
    T.sync(device)
    dist.barrier()
    st = T.StepTimer(device)
    for _ in range(reps):
        st.start()
        dist.all_to_all_single(out, inp)
        st.stop()
    per_call = T.stats(st.times_ms())
    worst = torch.tensor([per_call["median"]], device=device, dtype=torch.float64)
    dist.all_reduce(worst, op=dist.ReduceOp.MAX)
    worst = worst.item()
    return {**plan, "measured": True, "world_size": W, "backend": dist.get_backend(),
            "per_call_ms": per_call, "per_call_ms_slowest_rank": worst,
            "per_step_ms": worst * plan["calls_per_step"],
            "note": "isolated all-to-all, no overlap with compute; per_step_ms = slowest rank x calls_per_step"}


# ------------------------------------------------------------------------------ metadata and metrics

# Dense (no sparsity) vendor peak TFLOPS per dtype. First substring match wins, so specific names come
# first. None = unknown: MFU is left empty and the run says so (or pass --peak-tflops).
PEAKS = [
    ("B200", {"bf16": 2250, "fp16": 2250}),
    ("H200", {"bf16": 989, "fp16": 989, "fp32": 67}),
    ("H100 PCIe", {"bf16": 756, "fp16": 756, "fp32": 51}),
    ("H100 NVL", {}),
    ("H100", {"bf16": 989, "fp16": 989, "fp32": 67}),
    ("A100", {"bf16": 312, "fp16": 312, "fp32": 19.5}),
    ("L40S", {"bf16": 362, "fp16": 362, "fp32": 91.6}),
    ("L40", {"bf16": 181, "fp16": 181, "fp32": 90.5}),
    ("L4", {"bf16": 121, "fp16": 121, "fp32": 30.3}),
    ("A10G", {}),
    ("A10", {"bf16": 125, "fp16": 125, "fp32": 31.2}),
    ("RTX 4090", {"bf16": 165, "fp16": 165, "fp32": 82.6}),
    ("RTX 3090", {"bf16": 71, "fp16": 71, "fp32": 35.6}),
    ("V100", {"fp16": 125, "fp32": 15.7}),
    ("T4", {"fp16": 65, "fp32": 8.1}),
    ("P100", {"fp16": 18.7, "fp32": 9.3}),
]


def _sh(args, cwd=None):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10, check=True, cwd=cwd).stdout.strip()
    except Exception:
        return None


def git_state():
    commit = _sh(["git", "rev-parse", "HEAD"], cwd=ROOT)
    if commit is None:                    # e.g. a Colab upload without .git
        return os.environ.get("PERF_COMMIT"), None
    status = _sh(["git", "status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results"], cwd=ROOT)
    return commit, bool(status) if status is not None else None


def device_name(device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return platform.processor() or platform.machine() or "cpu"


_SMI = "name,driver_version,pstate,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit"


def gpu_telemetry(device) -> dict | None:
    """SM clock, temperature and power now: explains a 5% miss on a shared or hot GPU."""
    if device.type != "cuda":
        return None
    props = torch.cuda.get_device_properties(device)
    dom, bus, dv = (getattr(props, f"pci_{k}_id", None) for k in ("domain", "bus", "device"))
    if all(isinstance(v, int) for v in (dom, bus, dv)):       # nvidia-smi and CUDA may number GPUs differently
        sel = f"--id={dom:08X}:{bus:02X}:{dv:02X}.0"
    else:
        sel = f"--id={device.index or 0}"
    out = _sh(["nvidia-smi", sel, f"--query-gpu={_SMI}", "--format=csv,noheader,nounits"])
    if not out:
        return None
    vals = [v.strip() for v in out.splitlines()[0].split(",")]
    res = {}
    for k, v in zip(_SMI.split(","), vals):
        try:
            res[k] = v if k in ("name", "driver_version", "pstate") else float(v)
        except ValueError:
            res[k] = v
    return res


def peak_tflops(name: str, dtype: str):
    """(peak TFLOPS for this dtype or None, gpu_spec)."""
    try:
        from perfkit import metrics
        p = metrics.gpu_peak(name)
        val = p.get(dtype, p.get(f"{dtype}_tflops", p.get("tflops")))
        return val, p.get("spec", p.get("name"))
    except (NotImplementedError, AttributeError, TypeError):
        pass
    for key, table in PEAKS:
        if key.lower() in name.lower():
            return table.get(dtype), key
    return None, None


def run_metadata(s: Setup, config_file: str, overrides: list, tag: str) -> dict:
    dev = s.device
    commit, dirty = git_state()
    name = device_name(dev)
    _, spec = peak_tflops(name, s.dtype)
    tel = gpu_telemetry(dev)
    meta = {
        "gpu": name, "gpu_spec": spec,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if dev.type == "cuda" else None,
        "driver": (tel or {}).get("driver_version") if dev.type == "cuda" else None,
        "device": dev.type, "commit": commit, "dirty": dirty,
        "config_file": config_file, "overrides": overrides, "tag": tag,
        "time": _dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "dtype": s.dtype, "dtype_requested": s.cfg["run"].get("dtype"), "notes": list(s.notes),
        "batch_size": s.loader.B, "seq_len": s.loader.T, "tokens_per_step": s.loader.B * s.loader.T,
        "vocab_size": s.mcfg.vocab_size, "data_source": s.cfg["data"]["source"],
        "python": platform.python_version(), "bench_version": BENCH_VERSION,
        "timing": {"cuda": "cuda_events", "mps": "sync_per_boundary", "cpu": "perf_counter"}.get(dev.type),
    }
    if dev.type == "cuda":
        p = torch.cuda.get_device_properties(dev)
        meta.update(compute_capability=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count,
                    gpu_mem_gb=round(p.total_memory / 2**30, 1), gpu_count=torch.cuda.device_count())
        meta["telemetry_start"] = tel
    try:
        from perfkit import metrics
        theirs = metrics.run_metadata(dev)
        meta.update({k: v for k, v in theirs.items() if v is not None})
    except NotImplementedError:
        pass
    return meta


def compute_metrics(s: Setup, step_ms: float, peak) -> dict:
    """MFU from model FLOPs; HFU also counts recompute and Muon (perfkit/metrics.py docstring).

    model_flops_per_token = 6 * N_active + 12 * L * d * T      (PaLM; attention scores non-causal)
    hw_flops_per_token    = model + 2 forwards of the stack (reversible: invert + recompute)
                            + the head again (ce_impl=chunked recomputes it) + Muon Newton-Schulz / tokens
    """
    m, core = s.mcfg, s.core
    B, Tn = s.loader.B, s.loader.T
    tokens = B * Tn
    total, active = core.param_counts()
    L, d = m.n_layer, m.n_embd
    model_fpt = 6 * active + 12 * L * d * Tn
    try:
        from perfkit import metrics
        f = metrics.flops_per_token(core, Tn)
        model_fpt, total, active = f["model_flops_per_token"], f["params_total"], f["params_active"]
    except NotImplementedError:
        pass
    head = core.head.weight.numel()
    ke = core.tok.proj.weight.numel() if m.embedding == "ke1" else 0
    stack_fwd = 2 * (active - head - ke) + 4 * L * d * Tn
    hw = model_fpt + (2 * stack_fwd if m.reversible else 0) + (2 * head if m.ce_impl == "chunked" else 0)
    hw += s.opt.flops() / tokens
    tps = tokens / (step_ms / 1e3)
    out = {"tokens_per_step": tokens, "tokens_per_s": tps, "params_total": total, "params_active": active,
           "model_flops_per_token": model_fpt, "hw_flops_per_token": hw, "peak_tflops": peak,
           "model_tflops_per_s": model_fpt * tps / 1e12, "mfu": None, "hfu": None}
    if peak:
        try:
            from perfkit import metrics
            out.update(metrics.mfu({"model_flops_per_token": model_fpt}, tps, peak))
        except NotImplementedError:
            out["mfu"] = model_fpt * tps / (peak * 1e12)
        out["hfu"] = hw * tps / (peak * 1e12)
    return out


# ------------------------------------------------------------------------------ output

def _f(x, nd=2):
    return "-" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def summary(run: dict, gate_level: int) -> str:
    m, c, me = run["metrics"], run["clean"], run["meta"]
    st = c["step_ms"]
    lines = [
        "",
        f"{me['gpu']} | {me['dtype']} | B={me['batch_size']} T={me['seq_len']} V={me['vocab_size']} | "
        f"commit {str(me['commit'])[:10]}{' (DIRTY)' if me.get('dirty') else ''} | tag {me['tag']}",
        f"step  median {_f(st['median'])} ms  p10 {_f(st['p10'])}  p90 {_f(st['p90'])}  spread {_f(st['spread_pct'], 1)}%"
        f"   data {_f(c['data_ms']['median'], 3)} ms",
        f"      {m['tokens_per_s']:,.0f} tokens/s   MFU {_f(100 * m['mfu'], 1) if m['mfu'] else '-'}%"
        f"   HFU {_f(100 * m['hfu'], 1) if m['hfu'] else '-'}%"
        + (f"   peak mem {run['memory']['peak_allocated_gb']:.2f} GB" if run.get("memory") else ""),
    ]
    lay = run.get("layer")
    if lay:
        base = lay["step_ms"]["median"] or 1
        lines += ["", f"parts (gated: level <= {gate_level})  recorder overhead {_f(lay['overhead_pct'], 2)}%",
                  f"  {'part':<34}{'median ms':>11}{'p10..p90':>19}{'% step':>8}{'spread%':>9}{'lvl':>4}{'calls':>6}"]
        for name, r in lay["regions"].items():
            if r["level"] <= gate_level:
                lines.append(f"  {name:<34}{_f(r['median'], 3):>11}{_f(r['p10'], 2) + '..' + _f(r['p90'], 2):>19}"
                             f"{_f(100 * r['median'] / base, 1):>8}{_f(r['spread_pct'], 1):>9}{r['level']:>4}"
                             f"{r['calls_per_step']:>6}")
        inner = {}
        for name, r in lay["regions"].items():
            if r["level"] > gate_level:
                key = re.sub(r"block\d+", "block*", name)
                inner[key] = inner.get(key, 0.0) + r["median"]
        if inner:
            lines += ["", "inside the blocks (information only, summed over layers), top 15:"]
            for k in sorted(inner, key=inner.get, reverse=True)[:15]:
                lines.append(f"  {k:<44}{_f(inner[k], 3):>10} ms{_f(100 * inner[k] / base, 1):>7}%")
    if run.get("checkpoint"):
        ck = run["checkpoint"]
        lines.append(f"\ncheckpoint save {_f(ck['save_ms']['median'], 1)} ms (to CPU {_f(ck['to_cpu_ms']['median'], 1)}, "
                     f"write {_f(ck['write_ms']['median'], 1)}), {ck['size_mb']:.1f} MB, {ck['write_mb_per_s']:.0f} MB/s")
    a2a = (run.get("comm") or {}).get("moe_a2a")
    if a2a:
        if a2a.get("measured"):
            lines.append(f"MoE all-to-all {_f(a2a['per_call_ms_slowest_rank'], 3)} ms/call x {a2a['calls_per_step']} "
                         f"= {_f(a2a['per_step_ms'], 1)} ms/step on {a2a['world_size']} GPUs ({a2a['backend']})")
        else:
            lines.append(f"MoE all-to-all not measured ({a2a['reason']}); {a2a['payload_mb_per_call']:.1f} MB/call, "
                         f"{a2a['calls_per_step']} calls/step")
    tr = run.get("trace")
    if tr and tr.get("gpu_busy_pct") is None:
        lines.append(f"trace (CPU run: no GPU numbers) -> {tr['trace_file']}")
    elif tr:
        lines.append(f"trace: GPU busy {_f(tr.get('gpu_busy_pct'), 1)}%, {tr['kernels_per_step']:.0f} kernels/step"
                     + (f", {tr['host_syncs_per_step']} host syncs/step" if "host_syncs_per_step" in tr else "")
                     + f"  ->  {tr['trace_file']}")
    return "\n".join(lines)


def quick_compare(a: dict, b: dict, tol_pct: float, gate_level: int) -> tuple:
    """This run (a) against an earlier one (b): the whole step and every part at level <= gate_level,
    plus checkpoint save and all-to-all when both runs have them. diff = slower / faster - 1.
    A quick look on the console; the official check is perfkit.report.compare."""
    problems = []
    for k in ("gpu", "dtype", "batch_size", "seq_len", "vocab_size"):
        if a["meta"].get(k) != b["meta"].get(k):
            problems.append(f"different {k}: {a['meta'].get(k)} vs {b['meta'].get(k)}")
    if a["config"].get("model") != b["config"].get("model"):
        problems.append("different model config")
    pairs = [("step (clean)", a["clean"]["step_ms"]["median"], b["clean"]["step_ms"]["median"])]
    ra, rb = (a.get("layer") or {}).get("regions", {}), (b.get("layer") or {}).get("regions", {})
    for name, r in ra.items():
        if r["level"] <= gate_level:
            pairs.append((name, r["median"], rb[name]["median"] if name in rb else None))
    if a.get("checkpoint") and b.get("checkpoint"):
        pairs.append(("checkpoint save", a["checkpoint"]["save_ms"]["median"], b["checkpoint"]["save_ms"]["median"]))
    ca, cb = (a.get("comm") or {}).get("moe_a2a") or {}, (b.get("comm") or {}).get("moe_a2a") or {}
    if ca.get("measured") and cb.get("measured"):
        pairs.append(("moe all-to-all", ca["per_step_ms"], cb["per_step_ms"]))
    rows, ok = [], not problems
    for name, x, y in pairs:
        if y is None or not x or not y:
            rows.append((name, x, y, None, "MISSING"))
            ok = False
            continue
        diff = (max(x, y) / min(x, y) - 1) * 100
        good = diff <= tol_pct
        ok &= good
        rows.append((name, x, y, diff, "ok" if good else "FAIL"))
    return rows, ok, problems


def write_run(run: dict, out_dir: Path, tag: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{tag}.json"
    i = 2
    while path.exists():
        path = out_dir / f"{tag}-{i}.json"
        i += 1
    path.write_text(json.dumps(_finite(run), indent=1, default=str, allow_nan=False) + "\n")
    return path


def _finite(o):
    """NaN / inf -> None: the run JSON must stay valid JSON for report.py and G8."""
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _finite(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_finite(v) for v in o]
    return o


def _show(p: Path) -> str:
    """A path relative to teams/06-performance when it is inside it, else absolute."""
    try:
        return str(Path(p).resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower().replace("nvidia", "")).strip("-") or "cpu"


# ------------------------------------------------------------------------------ main

def parse(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--tag", default=None, help="run name; default <config>_<gpu>_<time>")
    ap.add_argument("--modes", default=None, help=f"comma list from {','.join(MODES)} (= bench.modes)")
    ap.add_argument("--level", type=int, default=None, help="1 phases, 2 layers, 3 inside layers (= bench.level)")
    ap.add_argument("--steps", type=int, default=None, help="measured steps per mode (= bench.steps)")
    ap.add_argument("--warmup", type=int, default=None, help="warmup steps (= bench.warmup)")
    ap.add_argument("--peak-tflops", type=float, default=None, help="for a GPU not in the peak table")
    ap.add_argument("--against", default=None, help="an earlier run JSON: check every gated part within tol_pct")
    ap.add_argument("overrides", nargs="*", help="key=value, e.g. run.batch_size=16 model.vocab_size=131072")
    a = ap.parse_args(argv)
    ov = list(a.overrides)
    if a.modes:
        ov.append(f"bench.modes=[{a.modes}]")
    for key, val in (("level", a.level), ("steps", a.steps), ("warmup", a.warmup)):
        if val is not None:
            ov.append(f"bench.{key}={val}")
    return a, ov


def main(argv=None):
    a, overrides = parse(argv)
    cfg = load_config(a.config, overrides)
    mcfg = model_config(cfg)              # fail early on a bad config
    b = cfg["bench"]
    modes = list(b.get("modes") or ["clean"])
    unknown = sorted(set(modes) - set(MODES))
    if unknown:
        raise SystemExit(f"unknown bench.modes {unknown}: {', '.join(MODES)}")
    warns = []

    # Distributed only for the all-to-all: every rank joins, then rank 0 carries on alone.
    world = int(os.environ.get("WORLD_SIZE", "1"))
    comm = None
    if world > 1:
        import torch.distributed as dist
        lr = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(lr)
            cfg["run"]["device"] = f"cuda:{lr}"
        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo",
                                timeout=_dt.timedelta(minutes=60))
        device = pick_device(cfg["run"]["device"])
        dtype = choose_dtype(cfg["run"].get("dtype", "bf16"), device, [])
        plan = a2a_plan(cfg, mcfg, dtype)
        if plan is not None and device.type == "cuda":
            comm = {"moe_a2a": run_comm(plan, device)}
        rank = dist.get_rank()
        dist.destroy_process_group()
        if rank != 0:
            return 0
        if "comm" not in modes:
            modes.append("comm")

    device = pick_device(cfg["run"]["device"])
    config_file = os.path.relpath(Path(a.config).resolve(), ROOT)
    buckets = T.Buckets(device)
    try:
        with buckets("build"):
            s = build(cfg, device)
    except FileNotFoundError as e:
        print(f"[bench] {e}\n[bench] or time without the shards: add data.source=synthetic")
        return 4
    tag = a.tag or f"{Path(a.config).stem}_{_slug(device_name(device))}_{_dt.datetime.now():%Y%m%d-%H%M%S}"
    meta = run_metadata(s, config_file, overrides, tag)
    print(f"[bench] {meta['gpu']} | {s.dtype} | B={s.loader.B} T={s.loader.T} V={s.mcfg.vocab_size} | "
          f"modes {','.join(modes)} | level {b['level']} | {b['steps']} steps after {b['warmup']} warmup")
    for n in s.notes:
        print(f"[bench] note: {n}")

    run = {"meta": meta, "config": cfg}
    try:
        if s.mcfg.reversible:
            err = reversible_check(s)
            meta["reconstruction_err"] = err
            if err > 1e-3:
                warns.append(f"reversible reconstruction error {err:.1e} > 1e-3: the backward rebuilds wrong inputs")
        with buckets("first_step"):
            _, loss0 = train_step(s)
        with buckets("warmup"):
            for _ in range(max(0, b["warmup"] - 1)):
                train_step(s)
        meta["loss_after_first_step"] = float(loss0)
        tel0 = gpu_telemetry(device)

        print("[bench] clean ...", flush=True)
        run["clean"], loss, mem = run_clean(s, b["steps"])
        meta["loss_end"] = float(loss)
        if mem:
            run["memory"] = mem
        if not math.isfinite(meta["loss_end"]):
            warns.append(f"loss is {meta['loss_end']}: the numbers time a broken run")

        if "layer" in modes:
            print(f"[bench] layer (level {b['level']}) ...", flush=True)
            run["layer"] = run_layer(s, b["steps"], b["level"])
            lay = run["layer"]
            if lay["overhead_pct"] is not None and lay["overhead_pct"] > b["max_overhead_pct"]:
                warns.append(f"recorder overhead {lay['overhead_pct']:.1f}% > {b['max_overhead_pct']}%: "
                             "layer numbers are flagged")
            if lay.get("events_created_while_measuring"):
                warns.append(f"{lay['events_created_while_measuring']} CUDA events were created inside measured steps")
        if "trace" in modes:
            print("[bench] trace ...", flush=True)
            run["trace"] = run_trace(s, b["trace_steps"], tag)
        if "checkpoint" in modes:
            print("[bench] checkpoint ...", flush=True)
            run["checkpoint"] = run_checkpoint(s, b.get("ckpt_repeats", 5), tag)
        if "comm" in modes:
            if comm is None:
                plan = a2a_plan(cfg, s.mcfg, s.dtype)
                if plan is not None:
                    comm = {"moe_a2a": {**plan, "measured": False,
                                        "reason": "one GPU; run: torchrun --nproc_per_node 2 -m perfkit.bench ... --modes comm"}}
            run["comm"] = comm or {"moe_a2a": None}
        tel1 = gpu_telemetry(device)
    except torch.cuda.OutOfMemoryError:
        Bn, Tn, V = s.loader.B, s.loader.T, s.mcfg.vocab_size
        print(f"[bench] out of GPU memory. The fp32 logits alone are {Bn * Tn * V * 4 / 2**30:.1f} GB "
              f"(B={Bn} x T={Tn} x V={V} x 4 bytes), twice that with their gradient.\n"
              f"[bench] try model.ce_impl=chunked, or run.batch_size={max(1, Bn // 2)}")
        return 3

    peak, _ = peak_tflops(meta["gpu"], s.dtype)
    peak = a.peak_tflops or peak
    run["metrics"] = compute_metrics(s, run["clean"]["step_ms"]["median"], peak)
    if peak is None and device.type == "cuda":
        warns.append(f"no peak TFLOPS for {meta['gpu']} / {s.dtype}: MFU is empty (pass --peak-tflops)")
    run["startup"] = buckets.as_dict()
    if device.type == "cuda":
        meta["telemetry"] = {"after_warmup": tel0, "end": tel1}
        try:
            if tel1 and float(tel1["temperature.gpu"]) >= 83:
                warns.append(f"GPU at {tel1['temperature.gpu']:.0f} C at the end: likely thermal throttling")
            if tel1 and float(tel1["clocks.sm"]) < 0.85 * float(tel1["clocks.max.sm"]):
                warns.append(f"SM clock {tel1['clocks.sm']:.0f} MHz at the end vs max {tel1['clocks.max.sm']:.0f}: "
                             "a shared or throttled GPU misses the 5% check")
        except (TypeError, ValueError, KeyError):
            pass

    warns += rule_checks(run, modes, cfg)
    run["warnings"] = warns
    problems = validate_run(run)

    gate = cfg.get("report", {}).get("gate_level", 2)
    print(summary(run, gate))
    path = write_run(run, ROOT / b["out_dir"], tag)
    print(f"\n[bench] run JSON -> {_show(path)}")
    try:
        from perfkit import report
        md = path.with_suffix(".md")
        md.write_text(report.render(run))
        print(f"[bench] report   -> {_show(md)}")
    except NotImplementedError:
        print("[bench] perfkit.report.render is still a stub: no .md written")
    for w in warns:
        print(f"[warn] {w}")
    for p in problems:
        print(f"[schema] {p}")
    code = 1 if problems else 0

    if a.against:
        other = json.loads(Path(a.against).read_text())
        tol = cfg.get("report", {}).get("tol_pct", 5.0)
        rows, ok, why = quick_compare(run, other, tol, gate)
        print(f"\nagainst {a.against} (tol {tol}%, level <= {gate}):")
        for w in why:
            print(f"  [refuse] {w}")
        for name, x, y, diff, verdict in rows:
            print(f"  {name:<34}{_f(x, 3):>11}{_f(y, 3):>11}{_f(diff, 2):>8}%  {verdict}")
        print(f"  {'PASS' if ok else 'FAIL'}")
        if not ok:
            code = code or 2
    return code


def rule_checks(run: dict, modes: list, cfg: dict) -> list:
    """The instructor's rules (25 Oct) that a run can check about itself."""
    out = []
    me = run["meta"]
    if me.get("dirty"):
        out.append("uncommitted changes: Gate 1 numbers must come from committed code")
    if me.get("commit") is None:
        out.append("no commit ID (no .git): set PERF_COMMIT=<sha> so the run names its code")
    if me["data_source"] == "fineweb10b":
        out.append("rule 1a: FineWeb shards hold GPT-2 token ids; time with the BrahmicTokenizer-131K vocabulary "
                   "(model.vocab_size=131072 data.source=synthetic) until data in that tokenizer exists")
    if "layer" in modes and cfg["bench"]["level"] < 3:
        out.append("rule 3a: parts inside a block were not timed; add --level 3 (reported, not gated)")
    missing = [m for m in ("checkpoint", "comm") if m not in modes]
    if missing:
        out.append(f"rule 3b: {' and '.join(missing)} not measured; add them with --modes")
    if run.get("layer"):
        noisy = [n for n, r in run["layer"]["regions"].items()
                 if r["level"] <= cfg.get("report", {}).get("gate_level", 2) and (r["spread_pct"] or 0) > 10]
        if noisy:
            out.append(f"{len(noisy)} gated part(s) have p10..p90 spread > 10%, so may miss the 5% check: "
                       + ", ".join(noisy[:6]))
    if me["device"] == "mps":
        out.append("MPS has no reliable timing events: parts were timed with a sync at every boundary")
    if cfg["run"].get("compile") and run.get("layer"):
        out.append("run.compile=true: regions inside compiled code are not timed reliably; trust clean and trace")
    return out


if __name__ == "__main__":
    sys.exit(main())
