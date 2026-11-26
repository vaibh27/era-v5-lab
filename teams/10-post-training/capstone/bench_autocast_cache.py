"""Autocast weight-cast cache on vs off, on Team 5's model, through the paths TSAI runs for us (job GPL4, 24 Nov 2026).

    python capstone/bench_autocast_cache.py --g5-root ../.. --spec RUN/manifest.json --checkpoint RUN/checkpoint.pt \
        --byte-table ART/ke2_p1536/raw_table_P32.npz --device cuda --out cache.jsonl [--switch global|site]

torch.autocast keeps the low-precision cast of every FP32 leaf weight that requires grad until the context exits. Rows:
  sft_full     one full-FT step as capstone/sft_g5.py runs it (bench.accumulate on G5ForBench, clip, AdamW), with bench's
               AUTOCAST as sft_g5 leaves it, and a second row with bf16 autocast forced on (what a bf16 SFT would hit)
  rl_old_lp    the trainer's no_grad old log-probs (rl-loop/train.py chunked_logprobs under Policy.autocast): the scoring call
  rl_update    the trainer's update on one group (backward_objective_v1, clip, AdamW) under Policy.autocast
  *_frozen     the same with only block 0's attention trainable (frozen FP32 head and experts, as under LoRA or DT)
For each: peak CUDA memory allocated over the phase (GiB, and MiB above the phase start), median seconds over --reps,
and the result's max abs difference between cache on and cache off. --switch global forces cache_enabled on every
torch.autocast (measures the base code); --switch site uses the switches this change adds (policy.AUTOCAST_CACHE,
bench AUTOCAST["cache"]); a tree without the policy switch must use --switch global.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import importlib.util as iu
import json
import statistics
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl-loop"))
sys.path.insert(0, str(HERE.parent / "sft"))
sys.path.insert(0, str(HERE.parent / "sft" / "bench"))
sys.path.insert(0, str(HERE))

import policy as policy_mod  # noqa: E402
import train as bench  # noqa: E402  sft/bench/train.py
from bench_rl_step_g5 import build_lm, load_spec  # noqa: E402

_ORIG_AUTOCAST = torch.autocast


def set_cache(on: bool, switch: str) -> None:
    if switch == "global":
        torch.autocast = _ORIG_AUTOCAST if on else (lambda *a, **k: _ORIG_AUTOCAST(*a, **{**k, "cache_enabled": False}))
    else:
        if not hasattr(policy_mod, "AUTOCAST_CACHE"):  # this tree switches only bench's cache; RL rows need --switch global
            raise SystemExit("--switch site needs rl-loop/policy.py AUTOCAST_CACHE (not in this tree): use --switch global")
        policy_mod.AUTOCAST_CACHE = on
        bench.AUTOCAST["cache"] = on


def rl_train():
    s = iu.spec_from_file_location("g10_rl_train", HERE.parent / "rl-loop" / "train.py")
    m = iu.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def sync(dev):
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def phase(dev, fn, reps):
    """Median seconds of fn over reps (one warm call first), and the peak allocated during the last call."""
    fn()
    ts = []
    for _ in range(reps):
        sync(dev)
        if dev.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.memory_allocated()
        t0 = time.perf_counter()
        r = fn()
        sync(dev)
        ts.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() if dev.startswith("cuda") else 0
    return r, {"s": round(statistics.median(ts), 5), "peak_gib": round(peak / 2**30, 4),
               "over_start_mib": round((peak - start) / 2**20, 2) if dev.startswith("cuda") else None}


def free(dev):
    """Drop everything a row left (closures hold the model), so each row's peak starts from the same floor."""
    gc.collect()
    if dev.startswith("cuda"):
        torch.cuda.empty_cache()


def flat(ts):
    return torch.cat([t.detach().float().reshape(-1).cpu() for t in ts])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--g5-root", type=Path, required=True)
    ap.add_argument("--spec", type=Path, required=True)
    ap.add_argument("--arm", type=Path)
    ap.add_argument("--checkpoint", type=Path)
    ap.add_argument("--byte-table", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--max-new", type=int, default=384)
    ap.add_argument("--sft-rows", type=int, default=16)
    ap.add_argument("--sft-len", type=int, default=512)
    ap.add_argument("--micro-tokens", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--switch", choices=("global", "site"), default="site")
    ap.add_argument("--label", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    dev = a.device
    spec = load_spec(a.spec, a.arm)
    T = rl_train()
    from sft_g5 import G5ForBench  # noqa: PLC0415
    import train_probe as probe  # noqa: PLC0415
    res = {}

    def out(rec):
        rec = {"label": a.label, "switch": a.switch, "device": dev,
               "gpu": torch.cuda.get_device_name() if dev.startswith("cuda") else "cpu", **rec}
        print(json.dumps(rec), flush=True)
        with open(a.out, "a") as fh:
            fh.write(json.dumps(rec) + "\n")

    g = torch.Generator().manual_seed(1)
    V = spec["vocab_size"]
    sft_rows = [{"input_ids": torch.randint(16, V, (a.sft_len,), generator=g).tolist(),
                 "loss_mask": [0] * (a.sft_len // 4) + [1] * (a.sft_len - a.sft_len // 4)} for _ in range(a.sft_rows)]
    seq = torch.randint(16, V, (a.k, a.prompt_len + a.max_new), generator=g).to(dev)
    cmask = torch.ones(a.k, a.max_new, dtype=torch.bool, device=dev)
    adv = torch.tensor([1.0, -1.0] * (a.k // 2) + [0.5] * (a.k % 2), device=dev)

    for cache in (True, False):
        set_cache(cache, a.switch)
        for variant in ("full", "frozen"):
            # ---- SFT step as sft_g5 runs it (AUTOCAST as sft_g5 leaves it), then with bf16 autocast forced
            for sft_amp in ((None, torch.bfloat16) if variant == "full" else ()):
                lm = build_lm(a.g5_root, spec, a.byte_table, dev, 1, a.checkpoint)
                lm.train()
                for p in lm.parameters():
                    p.requires_grad_(True)
                w = G5ForBench(lm)
                opt = torch.optim.AdamW(lm.parameters(), lr=1e-5, weight_decay=0.0)
                bench.AUTOCAST["dtype"] = sft_amp
                seen = {}
                def spy(m, i):
                    seen.setdefault("amp", torch.is_autocast_enabled(torch.device(dev).type))

                hook = lm.blocks[0].register_forward_pre_hook(spy)
                mbs = [bench.collate(gr, 0) for gr in bench.plan_microbatches(sft_rows, a.micro_tokens)]

                def sft_step():
                    opt.zero_grad(set_to_none=True)
                    loss, _ = bench.accumulate(w, mbs, dev)
                    torch.nn.utils.clip_grad_norm_(list(lm.parameters()), 1.0)
                    opt.step()
                    return loss

                torch.manual_seed(0)
                loss, st = phase(dev, sft_step, a.reps)
                hook.remove()
                key = f"sft_full_amp{'bf16' if sft_amp else 'none'}"
                res.setdefault(key, {})[cache] = (torch.tensor([loss]), flat(lm.parameters()))
                out({"row": key, "cache": cache, "autocast_active": seen.get("amp"), "microbatches": len(mbs), **st})
                bench.AUTOCAST["dtype"] = None
                del lm, w, opt, sft_step, mbs, loss
                free(dev)
            # ---- RL: old log-probs (no_grad scoring) and the update, through the trainer's own functions
            lm = build_lm(a.g5_root, spec, a.byte_table, dev, 1, a.checkpoint)
            pol = policy_mod.module_policy(lm, None, dev)
            pol.amp_dtype = torch.bfloat16  # rl-loop/train.py autocast bf16 (a G5 checkpoint is fp32 throughout)
            for n, p in lm.named_parameters():
                p.requires_grad_((".moe.router." not in n) if variant == "full" else n.startswith("blocks.0.attn."))
            named = [p for p in lm.parameters() if p.requires_grad]
            opt = torch.optim.AdamW(named, lr=1e-5, weight_decay=0.0)

            def old_lp():
                lm.eval()
                with torch.no_grad():
                    return T.chunked_logprobs(pol, seq, a.prompt_len, None)

            old, st = phase(dev, old_lp, a.reps)
            res.setdefault(f"rl_old_lp_{variant}", {})[cache] = (old.float().cpu(),)
            out({"row": f"rl_old_lp_{variant}", "cache": cache, "trainable": sum(p.numel() for p in named), **st})

            def update():
                lm.train()
                opt.zero_grad(set_to_none=True)
                T.backward_objective_v1(pol, (seq, cmask, adv, old, None, a.prompt_len, None), 0.2, int(cmask.sum()), None)
                gn = torch.nn.utils.clip_grad_norm_(named, 1.0)
                opt.step()
                return gn

            torch.manual_seed(0)
            gn, st = phase(dev, update, a.reps)
            res.setdefault(f"rl_update_{variant}", {})[cache] = (gn.reshape(1).float().cpu(), flat(named))
            out({"row": f"rl_update_{variant}", "cache": cache, "trainable": sum(p.numel() for p in named), **st})
            del lm, pol, opt, named, old_lp, update, old, gn
            free(dev)
    set_cache(True, a.switch)
    for key, d in res.items():
        diffs = [float((x - y).abs().max()) for x, y in zip(d[True], d[False])]
        out({"row": key, "identity_max_abs_diff_on_vs_off": diffs})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
