"""Modal job: G6 W5 Design Task 5 — Muon NS cost on dryrun2 config.

Wraps run_dryrun2_muon_ns.py on a single A100-40GB.
Streams bench output to Modal logs; writes result_muon_ns.json locally.

Usage from repo root:
  modal run teams/06-performance/jobs/modal_muon_ns.py           # full run (20 steps)
  modal run teams/06-performance/jobs/modal_muon_ns.py --smoke   # quick check (3 steps)
  modal run teams/06-performance/jobs/modal_muon_ns.py --gpu A10G  # cheaper GPU

Name: Adinath Auti
Email: adinathauti@g6.tsai
Pack: G6-Performance
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import modal

BRANCH   = "g6/w5-muon-ns-cost"
REPO_URL = "https://github.com/adinathauti/era-v5-lab.git"

app = modal.App("g6-muon-ns-cost")

image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime")
    .apt_install("git")
    .pip_install("triton", "pyyaml")
)


@app.function(image=image, gpu="A100-40GB", timeout=1800)
def bench(steps: int = 20, warmup: int = 5, smoke: bool = False) -> dict:
    import subprocess, json as _json, os
    from pathlib import Path as P

    repo_dir = P("/tmp/era")
    out_dir  = P("/tmp/muon_ns_out")
    out_dir.mkdir(exist_ok=True)

    # Fresh clone — always gets the latest commit on the branch.
    print(f"[modal] cloning {BRANCH} ...", flush=True)
    subprocess.run(
        ["git", "clone", "--branch", BRANCH, "--depth", "1", REPO_URL, str(repo_dir)],
        check=True, capture_output=True,
    )
    commit = subprocess.check_output(
        ["git", "-C", str(repo_dir), "rev-parse", "--short", "HEAD"], text=True,
    ).strip()
    print(f"[modal] commit {commit}", flush=True)

    import torch
    print(f"[modal] {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB  "
          f"torch {torch.__version__}", flush=True)

    perf_dir = repo_dir / "teams" / "06-performance"

    if smoke:
        extra = ["--steps", "3", "--warmup", "1"]
    else:
        extra = ["--steps", str(steps), "--warmup", str(warmup)]

    cmd = [
        sys.executable, "-B",
        str(perf_dir / "jobs" / "run_dryrun2_muon_ns.py"),
        "--synthetic", "--out", str(out_dir),
        *extra,
    ]
    print(f"[modal] CMD: {' '.join(cmd)}", flush=True)

    proc = subprocess.Popen(
        cmd, cwd=str(perf_dir),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, bufsize=1,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"bench failed exit={proc.returncode}")

    return _json.loads((out_dir / "result_muon_ns.json").read_text())


@app.local_entrypoint()
def main(
    smoke: bool = False,
    steps: int = 20,
    warmup: int = 5,
    gpu: str = "A100-40GB",
    out: str = "result_muon_ns.json",
):
    # Allow GPU override at call time without editing the file.
    fn = bench.with_options(gpu=gpu) if gpu != "A100-40GB" else bench

    result = fn.remote(steps=steps, warmup=warmup, smoke=smoke)

    out_path = Path(out)
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n[modal] wrote {out_path}")

    # Summary table.
    print(f"\n{'impl':<10} {'step_ms':>9} {'tok/s':>9} "
          f"{'NS_ms':>8} {'NS%step':>8} {'NS%muon':>8} {'calls':>7}")
    print("-" * 65)
    for c in result["cases"]:
        print(
            f"{c['impl']:<10} {c['step_ms_clean']:>9.1f} {c['tokens_per_s']:>9.0f} "
            f"{c['opt_muon_ns_ms']:>8.2f} {c['opt_muon_ns_pct_of_step']:>7.1f}% "
            f"{c['opt_muon_ns_pct_of_muon']:>7.1f}% {c['ns_calls_per_step']:>7}"
        )
    print("-" * 65)
    sp  = result.get("speedup_loop_to_batched")
    nsp = result.get("ns_speedup_loop_to_batched")
    print(f"step speedup loop->batched: {sp}x   NS speedup: {nsp}x")
