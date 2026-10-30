"""Measure model training throughput with replicated DistributedDataParallel.

Run from teams/06-performance, for example:
    torchrun --standalone --nproc_per_node=2 -m perfkit.ddp_bench \
        --config configs/lab_small.yaml --steps 20 --warmup 5 data.source=synthetic

``run.batch_size`` is the batch per rank. The JSON records both that value and
the global batch (per-rank batch * process count). This benchmark is separate
from ``perfkit.bench --modes comm``, which measures MoE all-to-all payloads only.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from perfkit import bench as B  # noqa: E402
from perfkit import metrics as M  # noqa: E402
from perfkit import timer as T  # noqa: E402
from perfkit.schema import validate_run  # noqa: E402


def _args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True)
    p.add_argument("--tag", default=None)
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--warmup", type=int, default=None)
    p.add_argument("overrides", nargs="*", help="config overrides such as run.batch_size=2")
    return p.parse_args(argv)


def _rank_max(values, world):
    gathered = [None] * world
    dist.all_gather_object(gathered, values)
    return [max(float(rank_values[i]) for rank_values in gathered) for i in range(len(values))]


def run(args):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        raise SystemExit("distributed training benchmark needs torchrun with at least 2 processes")
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device_name = f"cuda:{local_rank}"
        backend = "nccl"
    else:
        device_name = "cpu"
        backend = "gloo"
    dist.init_process_group(backend, init_method="env://")
    rank = dist.get_rank()
    try:
        cfg = B.load_config(args.config, args.overrides)
        if args.steps is not None:
            cfg["bench"]["steps"] = args.steps
        if args.warmup is not None:
            cfg["bench"]["warmup"] = args.warmup
        cfg["run"]["device"] = device_name
        cfg["run"]["seed"] = int(cfg["run"].get("seed", 0)) + rank
        if cfg["run"].get("compile"):
            raise SystemExit("ddp_bench currently requires run.compile=false")

        device = B.pick_device(device_name)
        setup = B.build(cfg, device)
        # The optimizer owns the original Parameters; DDP wraps that same module.
        setup.model = DDP(setup.core, device_ids=[local_rank] if device.type == "cuda" else None,
                          find_unused_parameters=setup.mcfg.ffn == "moe")
        per_rank_batch = setup.loader.B
        global_batch = per_rank_batch * world
        cfg["distributed"] = {"backend": backend, "world_size": world, "local_rank": local_rank,
                               "per_rank_batch_size": per_rank_batch, "global_batch_size": global_batch,
                               "parallelism": "ddp"}

        for _ in range(max(0, cfg["bench"]["warmup"])):
            B.train_step(setup)
        dist.barrier()

        timer = T.StepTimer(device, expected_steps=cfg["bench"]["steps"])
        local_data = []
        for _ in range(cfg["bench"]["steps"]):
            timer.start()
            data_ms, _ = B.train_step(setup)
            timer.stop()
            local_data.append(data_ms)
        local_step = timer.times_ms()
        # Report the slowest rank per step; that rank determines synchronous DDP throughput.
        step_ms = _rank_max(local_step, world)
        data_ms = _rank_max(local_data, world)
        dist.barrier()

        if rank == 0:
            tag = args.tag or f"ddp_{world}gpu_{dt.datetime.now():%Y%m%d-%H%M%S}"
            config_file = os.path.relpath(Path(args.config).resolve(), ROOT)
            meta = B.run_metadata(setup, config_file, args.overrides, tag)
            meta.update({"world_size": world, "local_world_size": int(os.environ.get("LOCAL_WORLD_SIZE", world)),
                         "visible_devices_per_rank": torch.cuda.device_count() if device.type == "cuda" else 0,
                         "backend": backend, "parallelism": "ddp", "per_rank_batch_size": per_rank_batch,
                         "global_batch_size": global_batch})
            peak, _ = M.peak_tflops(meta["gpu"], setup.dtype)
            total_peak = peak * world if peak is not None else None
            # Optimizer FLOPs are incurred once per replica, so divide by that
            # rank's tokens; global throughput and total peak are both scaled by
            # world size below.
            flops = M.flops_per_token(setup.core, setup.loader.T, setup.opt.flops(), per_rank_batch * setup.loader.T)
            tokens_per_step = global_batch * setup.loader.T
            tokens_per_s = tokens_per_step / (T.stats(step_ms)["median"] / 1e3)
            metrics = {"tokens_per_step": tokens_per_step, "tokens_per_s": tokens_per_s,
                       **flops, **M.mfu(flops, tokens_per_s, total_peak), "peak_tflops": total_peak}
            run = {"meta": meta, "config": cfg,
                   "clean": {"step_ms": T.stats(step_ms), "data_ms": T.stats(data_ms),
                             "raw_step_ms": [round(x, 4) for x in step_ms]},
                   "metrics": metrics,
                   "distributed": {"backend": backend, "world_size": world,
                                   "per_rank_batch_size": per_rank_batch, "global_batch_size": global_batch,
                                   "step_ms_slowest_rank": T.stats(step_ms)}}
            problems = validate_run(run)
            if problems:
                raise RuntimeError("invalid distributed benchmark result: " + "; ".join(problems))
            path = B.write_run(run, ROOT / Path(cfg["bench"]["out_dir"]).expanduser(), tag)
            print(f"[ddp-bench] {world} ranks | {meta['gpu']} | per-rank B={per_rank_batch} | "
                  f"global B={global_batch} | step median={T.stats(step_ms)['median']:.2f} ms | "
                  f"{tokens_per_s:,.0f} tokens/s")
            print(f"[ddp-bench] run JSON -> {B._show(path)}")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def main(argv=None):
    run(_args(argv))


if __name__ == "__main__":
    main()
