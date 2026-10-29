"""The run JSON: what bench writes and what report and G8 read.  [CONTRACT]

bench.py must produce it; report.py and the G8 exporter consume it. REQUIRED below is the source of truth:
docs/PLAN.md section 2.6 is a copy of it.
New fields can be added freely. Renaming or removing one breaks report, metrics and G8: agree it first.

    problems = validate_run(run_dict)      # [] when the run has every required field
    run = example_run()                    # a hand-made run, so the report can be built before bench exists
"""
from __future__ import annotations

STATS = ("median", "p10", "p90", "iqr", "n")

# section -> required keys (dotted paths inside the section). "meta", "config", "clean", "metrics" are
# always required; "layer" and "trace" only when that mode ran, "system" only on CUDA. A key with no value
# (cuda and driver on a CPU run) is present with value None.
REQUIRED = {
    "meta": ["gpu", "gpu_spec", "torch", "cuda", "driver", "device", "commit", "dirty", "config_file", "overrides", "tag", "time"],
    "config": ["run", "data", "model", "optim", "bench"],
    "clean": [*(f"step_ms.{s}" for s in STATS), *(f"data_ms.{s}" for s in STATS)],
    "layer": ["level", "overhead_pct", "step_ms.median", "step_ms_recorder_off.median", "regions"],
    "trace": ["trace_file", "top_ops", "gpu_busy_pct", "kernels_per_step"],
    "system": ["source", "samples", "metrics"],           # CUDA only: perfkit.metrics.GpuSampler, for G8
    "metrics": ["tokens_per_step", "tokens_per_s", "mfu", "params_total", "params_active",
                "model_flops_per_token", "peak_tflops"],
}
REGION_KEYS = ("median", "p10", "p90", "iqr", "n", "level", "calls_per_step")


def _get(d, dotted):
    for k in dotted.split("."):
        if not isinstance(d, dict) or k not in d:
            raise KeyError(dotted)
        d = d[k]
    return d


def validate_run(run: dict) -> list:
    problems = []
    for section, keys in REQUIRED.items():
        if section not in run:
            if section in ("meta", "config", "clean", "metrics"):
                problems.append(f"missing section {section!r}")
            continue
        for k in keys:
            try:
                _get(run[section], k)
            except KeyError:
                problems.append(f"missing {section}.{k}")
    for name, r in run.get("layer", {}).get("regions", {}).items():
        for k in REGION_KEYS:
            if k not in r:
                problems.append(f"region {name!r} missing {k!r}")
        if ":" not in name and name not in ("data", "fwd", "bwd", "opt"):
            problems.append(f"region {name!r} has no phase prefix (expected e.g. 'fwd:{name}')")
    return problems


def _s(median):
    return {"median": median, "p10": median * 0.98, "p90": median * 1.03, "iqr": median * 0.02, "n": 50}


def example_run() -> dict:
    """A made-up but well-formed run (2 layers). Numbers are illustrative only."""
    regions = {}

    def reg(name, ms, level):
        regions[name] = {**_s(ms), "level": level, "calls_per_step": 1}

    reg("data", 0.4, 1)
    reg("fwd", 9.0, 1); reg("bwd", 21.0, 1); reg("opt", 6.0, 1)
    reg("fwd:embed", 0.3, 2); reg("fwd:stack", 5.5, 1); reg("fwd:head+loss", 3.1, 2)
    for i in range(2):
        reg(f"fwd:block{i}", 2.7, 2)
        reg(f"fwd:block{i}.attn.qkv", 0.3, 3); reg(f"fwd:block{i}.attn.sdpa", 0.5, 3)
        reg(f"fwd:block{i}.moe.dispatch", 1.6, 3)
        reg(f"bwd:block{i}", 7.9, 2)
    reg("bwd:head+loss", 4.8, 2); reg("bwd:stack", 15.8, 1); reg("bwd:embed", 0.4, 2)   # bwd:stack = sum of bwd:blockN
    reg("opt:opt.muon", 5.2, 2); reg("opt:opt.adamw", 0.3, 2)
    return {
        "meta": {"gpu": "EXAMPLE-GPU", "gpu_spec": "A100-80GB", "torch": "2.x", "cuda": "12.x", "driver": "5xx.xx",
                 "device": "cuda",
                 "commit": "0" * 40, "dirty": False, "config_file": "configs/lab_small.yaml",
                 "overrides": [], "tag": "example", "time": "2026-10-25 12:00:00 IST"},
        "config": {"run": {"batch_size": 32, "dtype": "bf16"}, "data": {}, "model": {"n_layer": 2, "block_size": 512},
                   "optim": {"name": "muon"}, "bench": {"modes": ["clean", "layer"]}},
        "clean": {"step_ms": _s(36.5), "data_ms": _s(0.4)},
        "layer": {"level": 3, "overhead_pct": 0.8, "step_ms": _s(36.8), "step_ms_recorder_off": _s(36.5),
                  "regions": regions},
        "metrics": {"tokens_per_step": 16384, "tokens_per_s": 448876, "mfu": 0.031, "hfu": 0.052, "mfu_reason": None,
                    "params_total": 53_200_000, "params_active": 24_900_000,
                    "model_flops_per_token": 1.6e8, "hw_flops_per_token": 2.7e8, "peak_tflops": 312},
    }
