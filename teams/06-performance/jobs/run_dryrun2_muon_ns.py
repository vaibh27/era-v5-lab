"""Job wrapper: G6 Week-5 Design Task 5 — Muon Newton-Schulz optimizer cost on dryrun2 config.

Authority: G6 Week-5 Lab 5 brief, Design Task 5 (Sat 28 Nov 2026).
Owner: Adinath Auti (adinathauti)

Measures the Newton-Schulz (NS) share of step time and compares loop vs batched impl on the
dry run 2 config (T2: 8 layers, d 512, 16 experts top-2, batch 8×2048, Muon + AdamW, bf16).
Also runs a second batch size (B=16×2048) and two seeds to show NS share scales and is stable.

Runs perfkit.bench --modes clean,layer --level 3 for each seed × config × impl combination:
  case A: optim.muon_impl=loop    (reference: one NS chain per matrix)
  case B: optim.muon_impl=batched (same-shape matrices stacked; one NS chain per shape group)

Output fields (per case):
  seed, config, impl, batch_size, block_size
  step_ms_clean, tokens_per_s          — from clean mode (no profiler overhead)
  positions_per_s                      — B×T / step_s (all sequence positions)
  scored_tokens_per_s                  — same as positions on synthetic (no padding)
  bytes_per_s                          — positions_per_s × 2 (bf16, 2 bytes/token)
  peak_mem_gb                          — from metrics.peak_mem_gb (0 if not reported)
  opt_muon_ns_ms, NS%step, NS%Muon     — from layer mode (level 3)
  ns_calls_per_step                    — calls to opt:opt.muon.ns per step

Outputs:
  out/result_muon_ns.json              comparison summary (all cases + speedups)
  out/muon_ns_{loop,batched}_{cfg}_s{seed}.json  perfkit bench record per case

Usage from teams/06-performance/:
  # CPU smoke (exit 0 required before filing PR):
  python -B jobs/run_dryrun2_muon_ns.py --smoke-cpu

  # GPU run (Modal A100-40GB or equivalent, synthetic tokens, 2 seeds):
  python -B jobs/run_dryrun2_muon_ns.py --out out --synthetic
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

PERF_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PERF_DIR.parents[1]

# dryrun2 T2 overrides for the perfkit bench (base.yaml).
# Shape: 8 layers, d 512, 8 heads, 2 KV heads (GQA), block 2048, vocab 131072,
#        16 routed + 1 shared experts, ffn_hidden 768 (SwiGLU), batch 8, bf16.
# Dispatch sorted (gate-2 default via g5_config), bias_gamma 0.001 (dryrun2.yaml).
# KAS V2 head (model.head_mode) is a G5 trainer concept and has no matching
# perfkit key; it is left out. Its matrices are outside blocks.* and do not
# affect the NS cost measured here under the current Muon rule.
DRYRUN2_T2_OVERRIDES = [
    "model.n_layer=8",
    "model.n_embd=512",
    "model.n_head=8",
    "model.n_kv_head=2",
    "model.block_size=2048",
    "model.ffn=moe",
    "model.moe.num_experts=16",
    "model.moe.top_k=2",
    "model.moe.num_shared=1",
    "model.moe.ffn_hidden=768",
    "model.moe.dispatch=sorted",
    "model.moe.bias_gamma=0.001",
    "model.embedding=ke2",
    "model.ke_max_pos=32",
    "model.reversible=false",
    "model.vocab_size=131072",
    "model.ce_impl=chunked",
    "run.batch_size=8",
    "run.dtype=bf16",
    "run.seed=1234",
    "run.grad_clip=1.0",
    "optim.name=muon",
    "optim.muon_lr=0.006",
    "optim.weight_decay=0.0",
]

# Second batch config: same T2 shape but B=16 (double batch).
# NS cost is constant (weight shapes unchanged); fwd+bwd scale 2×, so NS% drops.
DRYRUN2_T2_B16_OVERRIDES = [
    x if not x.startswith("run.batch_size=") else "run.batch_size=16"
    for x in DRYRUN2_T2_OVERRIDES
]

# Config table: (label, batch_size, block_size, overrides)
# block_size is the sequence length used in rate calculations.
CONFIGS = [
    ("B8T2048",  8,  2048, DRYRUN2_T2_OVERRIDES),
    ("B16T2048", 16, 2048, DRYRUN2_T2_B16_OVERRIDES),
]

# Default seeds for the full run (≥2 seeds required by Lab 5 brief).
DEFAULT_SEEDS = [1234, 4321]

# Bytes per token for each dtype (used in bytes_per_s calculation).
DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2}

# Smoke overrides: tiny CPU model that exercises both NS impl paths.
SMOKE_OVERRIDES = [
    "model.ffn=moe",
    "model.moe.num_experts=4",
    "model.moe.num_shared=1",
    "model.moe.top_k=2",
    "model.moe.dispatch=sorted",
    "optim.name=muon",
    "optim.muon_lr=0.001",
    "run.seed=1234",
    "run.grad_clip=1.0",
    "data.source=synthetic",
    "run.device=cpu",
    "run.dtype=fp32",
]


def _run_bench(cmd: List[str]) -> Tuple[int, str, float]:
    t0 = time.perf_counter()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    proc = subprocess.run(cmd, cwd=str(PERF_DIR), env=env, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    out = (proc.stdout or "") + (("\n--- stderr ---\n" + proc.stderr) if proc.stderr else "")
    return proc.returncode, out, wall


def _bench_cmd(tag: str, impl: str, base_cfg: str, modes: str, level: int,
               warmup: int, steps: int, overrides: List[str], out_dir: Path,
               extra: List[str]) -> List[str]:
    return [
        sys.executable, "-B", "-m", "perfkit.bench",
        "--config", base_cfg,
        "--modes", modes,
        "--level", str(level),
        "--tag", tag,
        "--warmup", str(warmup),
        "--steps", str(steps),
        f"optim.muon_impl={impl}",
        *overrides,
        f"bench.out_dir={out_dir}",
        *extra,
    ]


def _ns_share(rec: dict) -> Tuple[float, float, float]:
    """(step_ms, ns_ms, ns_pct) from a bench record; ns_ms is 0 when the region is absent."""
    layer = rec.get("layer", {})
    step_ms = (layer.get("step_ms_recorder_off") or {}).get("median") or \
              (layer.get("step_ms") or {}).get("median") or 0.0
    regions = layer.get("regions", {})
    ns = regions.get("opt:opt.muon.ns", {})
    ns_ms = ns.get("median", 0.0)
    ns_pct = 100.0 * ns_ms / step_ms if step_ms > 0 else 0.0
    return step_ms, ns_ms, ns_pct


def _muon_share(rec: dict) -> Tuple[float, float]:
    """(muon_ms, muon_pct) from a bench record (opt.muon region at level 2)."""
    layer = rec.get("layer", {})
    step_ms = (layer.get("step_ms_recorder_off") or {}).get("median") or \
              (layer.get("step_ms") or {}).get("median") or 0.0
    muon_ms = layer.get("regions", {}).get("opt:opt.muon", {}).get("median", 0.0)
    muon_pct = 100.0 * muon_ms / step_ms if step_ms > 0 else 0.0
    return muon_ms, muon_pct


def _clean_toks(rec: dict) -> Tuple[float, float]:
    """(step_ms_clean, tokens_per_s) from the clean section."""
    clean = rec.get("clean", {})
    step_ms = (clean.get("step_ms") or {}).get("median", 0.0)
    tps = rec.get("metrics", {}).get("tokens_per_s", 0.0)
    return step_ms, tps


def _peak_mem_gb(rec: dict) -> float:
    """Peak allocated GPU memory in GB from the bench record."""
    for section in ("metrics", "clean", "layer"):
        v = rec.get(section, {}).get("peak_mem_gb")
        if v:
            return float(v)
    return 0.0


def _rates(step_ms_clean: float, batch_size: int, block_size: int,
           dtype: str = "bf16") -> dict:
    """Compute throughput rates from clean step time and batch dimensions.

    positions_per_s: all sequence positions (B * T) per second.
    scored_tokens_per_s: positions that contribute to the loss. Equal to
      positions_per_s on synthetic data (no padding). Will differ on real data.
    bytes_per_s: positions_per_s * dtype_bytes (input token bytes per second).
    """
    if step_ms_clean <= 0:
        return {"positions_per_s": 0.0, "scored_tokens_per_s": 0.0, "bytes_per_s": 0.0}
    positions = batch_size * block_size
    pos_per_s = positions * 1000.0 / step_ms_clean
    scored_per_s = pos_per_s  # synthetic: no padding
    bps = pos_per_s * DTYPE_BYTES.get(dtype, 2)
    return {
        "positions_per_s": round(pos_per_s, 0),
        "scored_tokens_per_s": round(scored_per_s, 0),
        "bytes_per_s": round(bps, 0),
    }


def _avg(vals: List[float]) -> float:
    return round(sum(vals) / len(vals), 3) if vals else 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description="G6 W5 Design Task 5: Muon NS cost on dryrun2 config")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--modes", default="clean,layer", help="Bench modes")
    ap.add_argument("--level", type=int, default=3, help="Profiler level")
    ap.add_argument("--steps", type=int, default=20, help="Timed steps per case")
    ap.add_argument("--warmup", type=int, default=5, help="Warmup steps")
    ap.add_argument("--synthetic", action="store_true",
                    help="Synthetic tokens (no shard download needed)")
    ap.add_argument("--smoke-cpu", action="store_true",
                    help="Tiny CPU smoke check; exit 0 required before PR")
    ap.add_argument("--batch", default="all",
                    help="Which batch configs to run: 'all', 'B8T2048', 'B16T2048'")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS),
                    help="Comma-separated seed list (default: '1234,4321')")
    args, extra = ap.parse_known_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_cpu:
        run_configs = [("smoke", 2, 32, list(SMOKE_OVERRIDES))]
        seeds = [1234]  # smoke: single seed only
        base_cfg = "configs/tiny_cpu.yaml"
        warmup, steps = 2, 5
        modes = "clean,layer"
        level = 3
    else:
        run_configs = [
            (label, bs, blk, cfg)
            for label, bs, blk, cfg in CONFIGS
            if args.batch == "all" or args.batch == label
        ]
        seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
        base_cfg = "configs/base.yaml"
        warmup, steps = args.warmup, args.steps
        modes = args.modes
        level = args.level

    all_rows: List[dict] = []
    # by_config_raw[cfg_label][seed] = {"speedup": ..., "ns_speedup": ...}
    by_config_raw: Dict[str, Dict[int, dict]] = {}

    for seed in seeds:
        for cfg_label, batch_size, block_size, cfg_overrides in run_configs:
            # Substitute the seed into the overrides for this run.
            overrides = [
                x if not x.startswith("run.seed=") else f"run.seed={seed}"
                for x in cfg_overrides
            ]
            if not args.smoke_cpu and args.synthetic:
                overrides.append("data.source=synthetic")

            config_rows: List[dict] = []
            impls = [
                ("loop",    f"muon_ns_loop_{cfg_label}_s{seed}"),
                ("batched", f"muon_ns_batched_{cfg_label}_s{seed}"),
            ]

            for impl, tag in impls:
                for ext in (".json", ".md", ".jsonl"):
                    p = out_dir / f"{tag}{ext}"
                    if p.exists():
                        p.unlink()

                cmd = _bench_cmd(tag, impl, base_cfg, modes, level, warmup, steps,
                                 overrides, out_dir, extra)
                print(f"\n[muon_ns] seed={seed} config={cfg_label} impl={impl}", flush=True)
                print(f"[muon_ns] cmd: {' '.join(cmd)}", flush=True)
                rc, log, wall = _run_bench(cmd)
                print(log, flush=True)
                if rc != 0:
                    print(f"[muon_ns] FAILED seed={seed} config={cfg_label} impl={impl} "
                          f"(exit={rc})", file=sys.stderr)
                    return rc

                rec_path = out_dir / f"{tag}.json"
                rec = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
                print(f"[muon_ns] seed={seed} config={cfg_label} impl={impl} done  "
                      f"wall={wall:.1f}s", flush=True)

                step_ms_clean, tps = _clean_toks(rec)
                step_ms_layer, ns_ms, ns_pct = _ns_share(rec)
                muon_ms, muon_pct = _muon_share(rec)
                ns_of_muon_pct = 100.0 * ns_ms / muon_ms if muon_ms > 0 else 0.0
                ns_calls = (rec.get("layer", {}).get("regions", {})
                            .get("opt:opt.muon.ns", {}).get("calls_per_step", 0))
                peak_mem = _peak_mem_gb(rec)
                rates = _rates(step_ms_clean, batch_size, block_size)

                row = {
                    "seed": seed,
                    "config": cfg_label,
                    "impl": impl,
                    "batch_size": batch_size,
                    "block_size": block_size,
                    "step_ms_clean": round(step_ms_clean, 2),
                    "tokens_per_s": round(tps, 0),
                    "positions_per_s": rates["positions_per_s"],
                    "scored_tokens_per_s": rates["scored_tokens_per_s"],
                    "bytes_per_s": rates["bytes_per_s"],
                    "peak_mem_gb": round(peak_mem, 2),
                    "step_ms_layer": round(step_ms_layer, 2),
                    "opt_muon_ms": round(muon_ms, 2),
                    "opt_muon_pct": round(muon_pct, 2),
                    "opt_muon_ns_ms": round(ns_ms, 2),
                    "opt_muon_ns_pct_of_step": round(ns_pct, 2),
                    "opt_muon_ns_pct_of_muon": round(ns_of_muon_pct, 2),
                    "ns_calls_per_step": ns_calls,
                }
                config_rows.append(row)
                all_rows.append(row)

            # Per-seed speedup for this config.
            if len(config_rows) == 2:
                loop_ms = config_rows[0]["step_ms_clean"] or 1.0
                bat_ms  = config_rows[1]["step_ms_clean"] or 1.0
                loop_ns = config_rows[0]["opt_muon_ns_ms"]
                bat_ns  = config_rows[1]["opt_muon_ns_ms"]
                seed_sp = {
                    "speedup_loop_to_batched":    round(loop_ms / bat_ms, 3) if bat_ms > 0 else None,
                    "ns_speedup_loop_to_batched": round(loop_ns / bat_ns, 3) if bat_ns > 0 else None,
                }
                by_config_raw.setdefault(cfg_label, {})[seed] = seed_sp

    # Aggregate: average speedups across seeds per config.
    by_config: Dict[str, dict] = {}
    for cfg_label, seed_data in by_config_raw.items():
        speeds    = [v["speedup_loop_to_batched"]    for v in seed_data.values() if v.get("speedup_loop_to_batched")]
        ns_speeds = [v["ns_speedup_loop_to_batched"] for v in seed_data.values() if v.get("ns_speedup_loop_to_batched")]
        by_config[cfg_label] = {
            "speedup_loop_to_batched":    _avg(speeds),
            "ns_speedup_loop_to_batched": _avg(ns_speeds),
            "per_seed": {str(s): v for s, v in seed_data.items()},
        }

    # Backward compat: top-level speedup = B8T2048 averaged.
    primary = by_config.get("B8T2048") or (list(by_config.values())[0] if by_config else {})

    summary = {
        "job": "dryrun2_muon_ns",
        "owner": "Adinath Auti (adinathauti)",
        "config": "dryrun2 T2 (8L d512, 16+1 experts, B8×2048 + B16×2048, bf16, Muon)",
        "seeds": seeds,
        "note": (
            "NS region is opt:opt.muon.ns at level 3. step_ms, positions/s, bytes/s from clean mode. "
            "NS share from layer mode (step_ms_recorder_off when available). "
            "scored_tokens_per_s = positions_per_s on synthetic (no padding; will differ on real data). "
            "bytes_per_s = positions_per_s x 2 (bf16, 2 bytes/token). "
            "peak_mem_gb from metrics.peak_mem_gb; 0 if not reported by perfkit. "
            "KAS V2 head excluded: no matching perfkit key; outside blocks.*. No invented ms."
        ),
        "cases": all_rows,
        "by_config": by_config,
        # Backward compat: top-level speedup = B8T2048 averaged across seeds.
        "speedup_loop_to_batched": primary.get("speedup_loop_to_batched"),
        "ns_speedup_loop_to_batched": primary.get("ns_speedup_loop_to_batched"),
    }

    result_path = out_dir / "result_muon_ns.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[muon_ns] wrote summary -> {result_path}", flush=True)

    # Print table (grouped by seed for readability).
    print("\n=== Muon Newton-Schulz cost: loop vs batched ===")
    hdr = (f"{'seed':>6} {'config':<12} {'impl':<10} {'step_ms':>8} {'tok/s':>8} "
           f"{'pos/s':>8} {'bytes/s':>9} {'mem_gb':>7} {'ns_ms':>7} "
           f"{'ns%step':>8} {'ns%muon':>8} {'ns_calls':>9}")
    sep = "-" * len(hdr)
    print(hdr)
    print(sep)
    prev_seed = None
    for r in all_rows:
        if r["seed"] != prev_seed:
            if prev_seed is not None:
                print()
            prev_seed = r["seed"]
        bps_str = f"{r['bytes_per_s'] / 1e6:.0f}M" if r["bytes_per_s"] > 0 else "0"
        print(
            f"{r['seed']:>6} {r['config']:<12} {r['impl']:<10} "
            f"{r['step_ms_clean']:>8.1f} {r['tokens_per_s']:>8.0f} "
            f"{r['positions_per_s']:>8.0f} {bps_str:>9} "
            f"{r['peak_mem_gb']:>7.1f} {r['opt_muon_ns_ms']:>7.2f} "
            f"{r['opt_muon_ns_pct_of_step']:>8.1f} {r['opt_muon_ns_pct_of_muon']:>8.1f} "
            f"{r['ns_calls_per_step']:>9}"
        )
    print(sep)
    for cfg_label, sp in by_config.items():
        per_seed_str = "  ".join(
            f"s{s}: {v.get('speedup_loop_to_batched')}x / NS {v.get('ns_speedup_loop_to_batched')}x"
            for s, v in sp.get("per_seed", {}).items()
        )
        print(
            f"{cfg_label}: step {sp['speedup_loop_to_batched']}x  "
            f"NS {sp['ns_speedup_loop_to_batched']}x  "
            f"(avg)  [{per_seed_str}]"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
