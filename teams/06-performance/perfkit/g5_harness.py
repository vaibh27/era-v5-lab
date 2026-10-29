"""Time G5's own harness (teams/05-architecture/harness/train.py): data, forward, backward, AdamW step.  [implemented]

G5 asked for it (29 Oct): "Timers: there is no hook. In harness/train.py the step is data, forward, backward of
the two losses, then the AdamW step. Time those four parts inside run()." Their file is theirs, so nothing in
teams/05-architecture is edited: this module imports it and runs the same step as run(), with timers around the
four parts. correctness/test_g5_harness.py checks that the losses equal those of their own run(), to the bit.

    cd teams/06-performance
    python -m perfkit.g5_harness                                   # config_v0 + tiny_moe (G5's CPU smoke test)
    python -m perfkit.g5_harness --device cuda --steps 50 --warmup 10 \\
        --set model.n_layer=9 model.d_model=256 model.n_head=8 model.d_ff=1024 moe.n_expert=8 \\
              data.seq_len=512 tiny.batch=16 tiny.vocab=131072 embedding.pos_dim=32

Parts, each per step (median, p10, p90 over the measured steps):
    data  stream slice -> device          (outside their timer: run() starts it after the copy)
    fwd   zero_grad, model, both cross-entropies (ntp on t+1, mtp on t+2)
    bwd   (ntp + mtp).backward()
    opt   clip_grad_norm_ + AdamW step
CUDA: one CUDA event pair per part, no sync inside the step; CPU: perf_counter. step = the sum of the four.
"harness_tokens_per_second" is what their logger computes (perf_counter from zero_grad to float(loss), no
CUDA sync): on a GPU it misses the queued AdamW kernels, so it reads higher than tokens_per_s here.

MFU (perfkit/metrics.py formula): 6 * N_active + 12 * L * d * T, N_active = attention, router, top_k experts,
KE projection, both heads. Their MoE runs every expert on every token (einsum over all n_expert), so HFU adds
the other n_expert - top_k experts. The harness runs fp32 (no autocast), so the peak is the fp32 one.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from perfkit import metrics as M
from perfkit.timer import stats

ROOT = Path(__file__).resolve().parents[1]
G5 = ROOT.parent / "05-architecture"
HARNESS = G5 / "harness" / "train.py"
CONFIG = G5 / "configs" / "config_v0.yaml"
TINY = G5 / "configs" / "small" / "tiny_moe.yaml"
PARTS = ("data", "fwd", "bwd", "opt")
PINNED_SHA256 = {                                                     # the G5 files these timings were checked on
    "train.py": "705ce1ba6ca4a36243722de431ee5bf57a318271c7a093c55a108383aa850870",
    "check_config.py": "dac6e4c562c6cd7ea859f8d3ea9f6d4e1cdf2d6b6f81191f0813082b215c25ea",
}


def load_harness(path: Path = HARNESS):
    """G5's train.py as a module. Its folder is on sys.path only while it loads (for `from check_config import ...`);
    train.py binds those names at import, so the path entry and the bare `check_config` module are removed after."""
    if not path.is_file():
        raise FileNotFoundError(f"G5 harness not found: {path}")
    folder, had_check = str(path.parent), "check_config" in sys.modules
    sys.path.insert(0, folder)
    try:
        spec = importlib.util.spec_from_file_location("g5_train", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path.remove(folder)
        if not had_check:
            sys.modules.pop("check_config", None)
    return mod


def harness_shas(folder: Path = HARNESS.parent) -> dict:
    """SHA-256 of each pinned G5 file, and whether all of them match PINNED_SHA256."""
    shas = {f: hashlib.sha256((folder / f).read_bytes()).hexdigest() for f in PINNED_SHA256}
    return {"harness_sha256": shas, "harness_pinned": shas == PINNED_SHA256}


def settings(h, config: Path = CONFIG, tiny: Path = TINY, sets=()) -> dict:
    """config_v0 merged with the tiny overrides (their merge and checks), then key=value sets (YAML values)."""
    import yaml
    merged = h.merge(h.load_settings(config), tiny)
    found = h.errors_for(merged)
    if found:
        raise SystemExit("\n".join(found))
    cfg = h.as_map(merged)
    for s in sets:
        key, sep, val = s.partition("=")
        if not sep:
            raise SystemExit(f"--set {s!r} must look like key=value")
        cfg[key] = yaml.safe_load(val)
    return cfg


def active_params(model, cfg: dict) -> dict:
    """(active, idle) matmul weights per token, as GPT.param_counts counts them (no biases, norms, lookups)."""
    d, L = int(cfg["model.d_model"]), int(cfg["model.n_layer"])
    E, k = int(cfg["moe.n_expert"]), int(cfg["moe.top_k"])
    per_expert = 2 * d * int(cfg["model.d_ff"])
    active = model.embed.projection.weight.numel() + model.head.weight.numel() + model.mtp_head.weight.numel()
    active += L * (4 * d * d + d * E + k * per_expert)          # q, k, v, o + router + top_k experts
    return {"active": active, "idle": L * (E - k) * per_expert, "total": sum(p.numel() for p in model.parameters())}


class _Clock:
    """Four parts per step: CUDA events (no sync inside the step) or perf_counter on CPU."""

    def __init__(self, device):
        self.cuda = device.type == "cuda"
        self.marks = []

    def mark(self):
        if self.cuda:
            e = torch.cuda.Event(enable_timing=True)
            e.record()
            self.marks.append(e)
        else:
            self.marks.append(time.perf_counter())

    def parts_ms(self) -> list:
        if self.cuda:
            torch.cuda.synchronize()
            return [a.elapsed_time(b) for a, b in zip(self.marks, self.marks[1:])]
        return [(b - a) * 1e3 for a, b in zip(self.marks, self.marks[1:])]


def run(cfg: dict, device: torch.device, h, steps: int, warmup: int) -> dict:
    """run() of train.py, step for step, with the four parts timed. Returns the result dict (no file written)."""
    cfg = {**cfg, "tiny.steps": warmup + steps}                      # the seeded stream covers every step
    seed = int(cfg["run.seed"])
    torch.manual_seed(seed)
    model = h.TinyMoE(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["optim.lr"]))
    stream = h.stream_for(cfg, seed)
    batch, clip = int(cfg["tiny.batch"]), float(cfg["run.grad_clip"])
    T = int(cfg["data.seq_len"])
    losses, mtps, parts, harness_tps = [], [], {p: [] for p in PARTS}, []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for step in range(warmup + steps):
        clock = _Clock(device)
        clock.mark()
        tokens = stream[step * batch:(step + 1) * batch].to(device)
        clock.mark()
        started = time.perf_counter()                                 # where their timer starts
        optimizer.zero_grad(set_to_none=True)
        logits, mtp_logits = model(tokens)
        ntp = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1))
        mtp = F.cross_entropy(mtp_logits[:, :-2].reshape(-1, mtp_logits.shape[-1]), tokens[:, 2:].reshape(-1))
        clock.mark()
        (ntp + mtp).backward()
        clock.mark()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        clock.mark()
        value, mtp_value = float(ntp.detach()), float(mtp.detach())
        if math.isnan(value) or math.isnan(mtp_value):
            raise SystemExit(f"NaN at step {step}")
        harness_tps.append(tokens.numel() / max(time.perf_counter() - started, 1e-9))    # their logger's number
        losses.append(value)
        mtps.append(mtp_value)
        if step >= warmup:
            for name, ms in zip(PARTS, clock.parts_ms()):
                parts[name].append(ms)
    step_ms = [sum(v) for v in zip(*(parts[p] for p in PARTS))]
    tokens_per_step = batch * T
    st = stats(step_ms)
    tps = tokens_per_step / (st["median"] / 1e3)
    n = active_params(model, cfg)
    L, d = int(cfg["model.n_layer"]), int(cfg["model.d_model"])
    model_fpt = 6 * n["active"] + 12 * L * d * T
    flops = {"model_flops_per_token": model_fpt, "hw_flops_per_token": model_fpt + 6 * n["idle"],
             "params_total": n["total"], "params_active": n["active"]}
    name = M.device_name(device)
    peak, _ = M.peak_tflops(name, "fp32")
    reason = None if peak else M.peak_missing_reason(device, name, "fp32")
    out = {
        "steps": steps, "warmup": warmup, "batch_size": batch, "seq_len": T, "vocab_size": int(cfg["tiny.vocab"]),
        "dtype": "fp32", "timing": "cuda_events" if device.type == "cuda" else "perf_counter",
        "step_ms": st, "parts_ms": {p: stats(parts[p]) for p in PARTS},
        "tokens_per_step": tokens_per_step, "tokens_per_s": tps,
        "harness_tokens_per_second": stats(harness_tps[warmup:])["median"],
        "metrics": {**flops, "peak_tflops": peak, **M.mfu(flops, tps, peak, reason)},
        "loss": {"first_ntp": losses[0], "last_ntp": losses[-1], "last_mtp": mtps[-1]},
        "losses": losses,
    }
    if device.type == "cuda":
        out["memory"] = M.memory(device)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Time the four parts of G5's harness step.")
    ap.add_argument("--config", type=Path, default=CONFIG)
    ap.add_argument("--tiny", type=Path, default=TINY)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--set", nargs="*", default=[], help="key=value over the merged settings, e.g. model.d_model=256")
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "runs" / "g5_harness.json")
    a = ap.parse_args(argv)
    h = load_harness()
    pin = harness_shas()
    if not pin["harness_pinned"]:
        changed = [f for f, v in pin["harness_sha256"].items() if v != PINNED_SHA256[f]]
        print(f"[warn] G5 harness changed since the pin ({', '.join(changed)}): not comparable to earlier runs;"
              " re-check test_g5_harness.py and update PINNED_SHA256")
    cfg = settings(h, a.config, a.tiny, a.set)
    device = torch.device(a.device)
    res = run(cfg, device, h, a.steps, a.warmup)
    res["meta"] = {**M.run_metadata(device), "harness": str(HARNESS.relative_to(ROOT.parent.parent)),
                   **pin,
                   "config": str(a.config), "tiny": str(a.tiny), "sets": list(a.set)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, default=str) + "\n")
    st, m = res["step_ms"], res["metrics"]
    print(f"[g5_harness] {res['meta']['gpu']} | fp32 | B={res['batch_size']} T={res['seq_len']} V={res['vocab_size']} "
          f"| {a.steps} steps after {a.warmup} warmup")
    print(f"step  median {st['median']:.3f} ms  p10 {st['p10']:.3f}  p90 {st['p90']:.3f}")
    for p in PARTS:
        s = res["parts_ms"][p]
        print(f"  {p:<5} {s['median']:9.3f} ms  {100 * s['median'] / st['median']:5.1f}%  p10..p90 {s['p10']:.3f}..{s['p90']:.3f}")
    mfu = "n/a" if m["mfu"] is None else f"{100 * m['mfu']:.2f}%"
    hfu = "n/a" if m["hfu"] is None else f"{100 * m['hfu']:.2f}%"
    print(f"{res['tokens_per_s']:,.0f} tokens/s (harness logger says {res['harness_tokens_per_second']:,.0f})   "
          f"MFU {mfu}   HFU {hfu}" + (f" ({m['mfu_reason']})" if m["mfu_reason"] else ""))
    print(f"loss_ntp {res['loss']['first_ntp']:.4f} -> {res['loss']['last_ntp']:.4f}   memory {res.get('memory')}")
    print(f"[g5_harness] -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
