"""Job wrapper: G6 Week-5 Design Task 5 — Muon Newton-Schulz optimizer cost on dryrun2 config.

Authority: G6 Week-5 Lab 5 brief, Design Task 5 (Sat 28 Nov 2026).
Owner: Adinath Auti (adinathauti)

Measures the Newton-Schulz (NS) share of step time and compares loop vs batched impl on the
dry run 2 config (T2: 8 layers, d 512, 16 experts top-2, batch 8×2048, Muon + AdamW, bf16).

Runs perfkit.bench --modes clean,layer --level 3 twice:
  case A: optim.muon_impl=loop    (reference: one NS chain per matrix)
  case B: optim.muon_impl=batched (same-shape matrices stacked; one NS chain per shape group)

Outputs:
  out/result_muon_ns.json         comparison summary
  out/muon_ns_loop.json           perfkit bench record, loop
  out/muon_ns_batched.json        perfkit bench record, batched

Usage from teams/06-performance/:
  # CPU smoke (exit 0 required before filing PR):
  python -B teams/06-performance/jobs/run_dryrun2_muon_ns.py --smoke-cpu

  # GPU run (Modal A100-80GB or equivalent, synthetic tokens):
  python -B teams/06-performance/jobs/run_dryrun2_muon_ns.py --out out --synthetic
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
    "run.batch_size=8",
    "run.dtype=bf16",
    "run.seed=1234",
    "run.grad_clip=1.0",
    "optim.name=muon",
    "optim.muon_lr=0.006",
    "optim.weight_decay=0.0",
]

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
    args, extra = ap.parse_known_args()

    out_dir = Path(args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_cpu:
        base_cfg = "configs/tiny_cpu.yaml"
        warmup, steps = 2, 5
        overrides = list(SMOKE_OVERRIDES)
        modes = "clean,layer"
        level = 3
    else:
        base_cfg = "configs/base.yaml"
        warmup, steps = args.warmup, args.steps
        overrides = list(DRYRUN2_T2_OVERRIDES)
        if args.synthetic:
            overrides.append("data.source=synthetic")
        modes = args.modes
        level = args.level

    cases = [("loop", "muon_ns_loop"), ("batched", "muon_ns_batched")]
    records: Dict[str, dict] = {}

    for impl, tag in cases:
        # Clean previous artifacts for this tag.
        for ext in (".json", ".md", ".jsonl"):
            p = out_dir / f"{tag}{ext}"
            if p.exists():
                p.unlink()

        cmd = _bench_cmd(tag, impl, base_cfg, modes, level, warmup, steps,
                         overrides, out_dir, extra)
        print(f"\n[muon_ns] case={impl}  cmd: {' '.join(cmd)}", flush=True)
        rc, log, wall = _run_bench(cmd)
        print(log, flush=True)
        if rc != 0:
            print(f"[muon_ns] case={impl} FAILED (exit={rc})", file=sys.stderr)
            return rc

        rec_path = out_dir / f"{tag}.json"
        records[impl] = json.loads(rec_path.read_text(encoding="utf-8")) if rec_path.exists() else {}
        print(f"[muon_ns] case={impl} done  wall={wall:.1f}s", flush=True)

    # Build comparison table.
    rows = []
    for impl, _ in cases:
        rec = records[impl]
        step_ms_clean, tps = _clean_toks(rec)
        step_ms_layer, ns_ms, ns_pct = _ns_share(rec)
        muon_ms, muon_pct = _muon_share(rec)
        ns_of_muon_pct = 100.0 * ns_ms / muon_ms if muon_ms > 0 else 0.0
        ns_calls = rec.get("layer", {}).get("regions", {}).get("opt:opt.muon.ns", {}).get("calls_per_step", 0)
        rows.append({
            "impl": impl,
            "step_ms_clean": round(step_ms_clean, 2),
            "tokens_per_s": round(tps, 0),
            "step_ms_layer": round(step_ms_layer, 2),
            "opt_muon_ms": round(muon_ms, 2),
            "opt_muon_pct": round(muon_pct, 2),
            "opt_muon_ns_ms": round(ns_ms, 2),
            "opt_muon_ns_pct_of_step": round(ns_pct, 2),
            "opt_muon_ns_pct_of_muon": round(ns_of_muon_pct, 2),
            "ns_calls_per_step": ns_calls,
        })

    # Speedup loop→batched.
    loop_ms = rows[0]["step_ms_clean"] if rows[0]["step_ms_clean"] > 0 else 1.0
    bat_ms = rows[1]["step_ms_clean"] if rows[1]["step_ms_clean"] > 0 else 1.0
    speedup = round(loop_ms / bat_ms, 3) if bat_ms > 0 else None
    loop_ns = rows[0]["opt_muon_ns_ms"]
    bat_ns = rows[1]["opt_muon_ns_ms"]
    ns_speedup = round(loop_ns / bat_ns, 3) if bat_ns > 0 else None

    summary = {
        "job": "dryrun2_muon_ns",
        "owner": "Adinath Auti (adinathauti)",
        "config": "dryrun2 T2 (8L d512, 16+1 experts, B8×2048, bf16, Muon)",
        "note": ("NS region is opt.muon.ns at level 3. step_ms and tokens_per_s from clean mode. "
                 "NS share from layer mode (step_ms_recorder_off when available). "
                 "KAS V2 head excluded: no matching perfkit key; its matrices are outside blocks.* "
                 "and do not affect NS cost under the current Muon rule. No invented ms."),
        "cases": rows,
        "speedup_loop_to_batched": speedup,
        "ns_speedup_loop_to_batched": ns_speedup,
    }

    result_path = out_dir / "result_muon_ns.json"
    result_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[muon_ns] wrote summary -> {result_path}", flush=True)

    # Print table.
    print("\n=== Muon Newton-Schulz cost: loop vs batched ===")
    print(f"{'impl':<10} {'step_ms':>9} {'tok/s':>9} {'opt.muon':>10} {'muon%':>7} "
          f"{'ns_ms':>7} {'ns%step':>8} {'ns%muon':>8} {'ns_calls':>9}")
    print("-" * 90)
    for r in rows:
        print(f"{r['impl']:<10} {r['step_ms_clean']:>9.1f} {r['tokens_per_s']:>9.0f} "
              f"{r['opt_muon_ms']:>10.2f} {r['opt_muon_pct']:>7.1f} "
              f"{r['opt_muon_ns_ms']:>7.2f} {r['opt_muon_ns_pct_of_step']:>8.1f} "
              f"{r['opt_muon_ns_pct_of_muon']:>8.1f} {r['ns_calls_per_step']:>9}")
    print("-" * 90)
    if speedup is not None:
        print(f"step speedup loop->batched: {speedup:.3f}x")
    if ns_speedup is not None:
        print(f"NS speedup  loop->batched: {ns_speedup:.3f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
