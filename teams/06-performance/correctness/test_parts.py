"""perfkit/parts.py (the slowest parts and one trace each) and perfkit.metrics.GpuSampler (system metrics for G8).

  * slowest_parts: leaf regions summed over layers; phases, fwd:stack / bwd:stack and regions with parts inside are
    not parts; <x>.expert<E> is folded into <x>; the plain stack's bwd:block<i> segments are parts.
  * cut_trace on a hand-made GPU trace: only the last profiled step; the CPU events inside the region; the kernels
    it launched (by correlation ID), even when they run after the CPU range ends; not the kernels of other parts;
    launch arrows and thread names kept; bwd:<name> segments cut between mark() points; the size cap keeps
    an evenly spaced sample of a repeated region's calls (first to last, not the front half) before it drops
    CPU detail.
  * GpuSampler: nothing off CUDA; CSV parsing and the summary; end to end against a fake nvidia-smi on PATH.
No GPU needed.
"""
from __future__ import annotations

import json
import os
import sys
import time

import pytest
import torch

from perfkit import metrics as M
from perfkit import parts as P



# ------------------------------------------------------------------------------ slowest_parts

def _r(ms, level):
    return {"median": ms, "level": level}


REGIONS = {
    "data": _r(1, 1), "fwd": _r(100, 1), "bwd": _r(150, 1), "opt": _r(60, 1),
    "fwd:stack": _r(60, 1), "fwd:block0": _r(30, 2), "fwd:block1": _r(30, 2),
    "fwd:block0.moe.dispatch": _r(10, 3), "fwd:block1.moe.dispatch": _r(12, 3),
    "fwd:block0.moe.dispatch.expert0": _r(5, 3), "fwd:block1.moe.dispatch.expert0": _r(6, 3),
    "fwd:block0.attn.sdpa": _r(3, 3), "fwd:block1.attn.sdpa": _r(3, 3),
    "fwd:head+loss": _r(40, 2), "bwd:head+loss": _r(45, 2),
    "bwd:stack": _r(70, 1), "bwd:block0": _r(35, 2), "bwd:block1": _r(35, 2),
    "opt:opt.muon": _r(50, 2), "opt:opt.muon.ns": _r(48, 3),
}


def test_slowest_parts_sums_layers_and_skips_containers():
    got = P.slowest_parts(REGIONS, step_ms=310)
    assert [(p["part"], p["ms"]) for p in got] == [
        ("bwd:block*", 70), ("opt:opt.muon.ns", 48), ("bwd:head+loss", 45), ("fwd:head+loss", 40),
        ("fwd:block*.moe.dispatch", 22)]
    assert got[0]["instances"] == 2 and got[0]["pct"] == pytest.approx(100 * 70 / 310)
    assert got[4]["level"] == 3


def test_slowest_parts_never_lists_phases_parents_or_experts():
    every = {p["part"] for p in P.slowest_parts(REGIONS, top=100)}
    assert not every & {"data", "fwd", "bwd", "opt", "fwd:stack", "bwd:stack", "fwd:block*", "opt:opt.muon"}
    assert not any("expert" in p for p in every)


def test_instance_is_the_middle_layer():
    part = {"names": [f"fwd:block{i}.moe.dispatch" for i in (10, 2, 0, 1, 3)]}
    assert P.instance(part) == "fwd:block2.moe.dispatch"           # 0 1 2 3 10 by number; as text it would be 10
    assert P.instance({"names": ["opt:opt.muon.ns"]}) == "opt:opt.muon.ns"


# ------------------------------------------------------------------------------ cut_trace

def _x(name, ts, dur, cat, **args):
    e = {"ph": "X", "cat": cat, "name": name, "ts": ts, "dur": dur, "pid": 1, "tid": 1}
    if args:
        e["args"] = args
    return e


def _trace():
    ann = "user_annotation"
    ev = [
        {"ph": "M", "name": "thread_name", "pid": 1, "tid": 1, "args": {"name": "main"}},
        _x("ProfilerStep#1", 0, 100, ann), _x("fwd:block1.moe.dispatch", 10, 10, ann),      # an earlier step
        _x("ProfilerStep#2", 200, 300, ann),
        _x("fwd:block1.moe.dispatch", 210, 20, ann),
        _x("aten::index_add_", 212, 5, "cpu_op"),
        _x("cudaLaunchKernel", 214, 2, "cuda_runtime", correlation=7),
        _x("aten::mm", 260, 5, "cpu_op"),                                    # after the region
        _x("cudaLaunchKernel", 262, 2, "cuda_runtime", correlation=8),
        _x("index_add_kernel", 400, 10, "kernel", correlation=7),           # runs long after its launch
        _x("gemm_kernel", 420, 10, "kernel", correlation=8),
        _x("fwd:block1.moe.dispatch", 395, 20, "gpu_user_annotation"),
        {"ph": "s", "cat": "ac2g", "name": "ac2g", "id": 7, "ts": 214, "pid": 1, "tid": 1},
        {"ph": "f", "cat": "ac2g", "name": "ac2g", "id": 7, "ts": 400, "pid": 1, "tid": 7, "bp": "e"},
        {"ph": "s", "cat": "ac2g", "name": "ac2g", "id": 8, "ts": 262, "pid": 1, "tid": 1},
        _x("__point__:__bwd_start__", 300, 1, ann), _x("aten::nll_loss_backward", 310, 5, "cpu_op"),
        _x("__point__:head+loss", 330, 1, ann), _x("__point__:block1", 350, 1, ann),
        _x("__point__:stack", 360, 1, ann), _x("aten::embedding_backward", 365, 5, "cpu_op"),
        _x("__point__:__bwd_end__", 380, 1, ann),
    ]
    return {"schemaVersion": 1, "traceEvents": ev}


def _names(cut):
    return [e["name"] for e in cut["traceEvents"]]


def test_cut_keeps_the_region_and_the_kernels_it_launched():
    cut = P.cut_trace(_trace(), "fwd:block1.moe.dispatch")
    names = _names(cut)
    assert "index_add_kernel" in names and "gemm_kernel" not in names
    assert "aten::index_add_" in names and "aten::mm" not in names
    assert "thread_name" in names and "ProfilerStep#2" in names
    assert [e["id"] for e in cut["traceEvents"] if e.get("cat") == "ac2g"] == [7, 7]
    assert any(e.get("cat") == "gpu_user_annotation" for e in cut["traceEvents"])
    assert cut["era_part"]["windows_us"] == [[210, 230]]                   # the last step only
    assert cut["schemaVersion"] == 1                                       # top-level keys kept


def test_cut_backward_segments_between_points():
    names = _names(P.cut_trace(_trace(), "bwd:head+loss"))                 # from __bwd_start__ to head+loss
    assert "aten::nll_loss_backward" in names and "aten::embedding_backward" not in names
    names = _names(P.cut_trace(_trace(), "bwd:embed"))                     # from the stack point to __bwd_end__
    assert "aten::embedding_backward" in names and "aten::nll_loss_backward" not in names


def test_cut_returns_none_for_a_region_not_in_the_trace():
    assert P.cut_trace(_trace(), "fwd:block7.attn.sdpa") is None


def test_size_cap_samples_calls_across_the_step():
    t = {"traceEvents": [_x("ProfilerStep#1", 0, 30_000, "user_annotation")]}
    for i in range(20):                                                     # 20 calls of one region, ~4 KB each
        t["traceEvents"].append(_x("opt:opt.muon.ns", 1000 * i, 500, "user_annotation"))
        t["traceEvents"] += [_x("aten::mm", 1000 * i + j, 1, "cpu_op", pad="x" * 40) for j in range(50)]
    full = P.cut_trace(t, "opt:opt.muon.ns")
    assert full["era_part"]["calls_kept"] == 20 and full["era_part"]["stride"] == 1
    cap = len(json.dumps(full)) // 3
    small = P.cut_trace(t, "opt:opt.muon.ns", max_bytes=cap)
    info = small["era_part"]
    assert info["calls_in_step"] == 20 and info["calls_kept"] < 20 and info["stride"] > 1
    starts = [a for a, _ in info["windows_us"]]
    assert starts[0] == 0 and starts[-1] >= 15_000                          # first call to near the last
    assert starts == [1000 * i for i in range(0, 20, info["stride"])]
    assert info["cpu_detail"] is True                                       # calls sampled before CPU detail
    assert len(P._dump(small)) <= cap


def test_write_part_traces(tmp_path):
    trace_file = tmp_path / "full.json"
    trace_file.write_text(json.dumps(_trace()))
    parts = [{"part": "fwd:block*.moe.dispatch", "names": ["fwd:block1.moe.dispatch"]},
             {"part": "opt:opt.adamw", "names": ["opt:opt.adamw"]}]
    out = P.write_part_traces(trace_file, parts, tmp_path / "parts")
    assert out[0]["file"].endswith("1_fwd_block1.moe.dispatch.json") and out[0]["gpu_events"] == 1
    assert out[0]["stride"] == 1 and out[0]["calls_kept"] == 1
    assert json.loads(open(out[0]["file"]).read())["traceEvents"]
    assert out[1]["file"] is None                                           # not in the trace: no file


# ------------------------------------------------------------------------------ GpuSampler

def test_sampler_does_nothing_off_cuda():
    with M.GpuSampler(torch.device("cpu")) as s:
        pass
    assert s.summary() is None


def test_sampler_parses_lines_and_summarises():
    s = M.GpuSampler(torch.device("cpu"))
    s.reason, s.started = "clocks_event_reasons.active", True
    s.rows = [s.parse_line(line) for line in (
        "2026/10/29 07:48:04.100, 90, 40, 30000, 1410, 1215, 300.0, 60, 0x0000000000000000",
        "2026/10/29 07:48:05.100, 100, 50, 30100, 1200, 1215, 400.0, 70, 0x0000000000000004",      # SwPowerCap
        "2026/10/29 07:48:06.150, [N/A], 50, 30100, 1300, 1215, 350.0, 65, 0x0000000000000001")]   # GpuIdle: no
    assert s.parse_line("1, 2") is None
    out = s.summary()
    assert out["samples"] == 3 and out["metrics"]["gpu_util_pct"] == {"min": 90.0, "median": 95.0, "max": 100.0,
                                                                     "mean": 95.0}
    assert out["metrics"]["sm_clock_mhz"]["median"] == 1300.0
    assert out["slowdown_samples"] == 1 and out["slowdown_reasons"] == ["SwPowerCap"]
    # every sample, for G8's own windows: seconds since the first, None for [N/A], slowdown names per sample
    assert out["start_time"] == "2026/10/29 07:48:04.100"
    assert [r["t_s"] for r in out["series"]] == [0.0, 1.0, 2.05]
    assert [r["gpu_util_pct"] for r in out["series"]] == [90.0, 100.0, None]
    assert [r["slowdown"] for r in out["series"]] == [[], ["SwPowerCap"], []]
    assert out["series"][0]["mem_used_mib"] == 30000.0


def test_sampler_series_without_reasons_or_a_readable_timestamp():
    s = M.GpuSampler(torch.device("cpu"))
    s.started = True
    s.rows = [s.parse_line("[N/A], 90, 40, 30000, 1410, 1215, 300.0, 60")]
    row = s.summary()["series"][0]
    assert row["t_s"] is None and row["slowdown"] is None and row["gpu_util_pct"] == 90.0


def test_mfu_reason():
    f = {"model_flops_per_token": 1e8, "hw_flops_per_token": 2e8}
    assert M.mfu(f, 1e5, 312)["mfu_reason"] is None
    out = M.mfu(f, 1e5, None, "unknown_gpu")
    assert out["mfu"] is None and out["hfu"] is None and out["mfu_reason"] == "unknown_gpu"
    assert M.mfu(f, 1e5, None)["mfu_reason"] == "no_peak"
    assert M.peak_missing_reason(torch.device("cpu"), "arm", "fp32") == "not_cuda"
    assert M.peak_missing_reason(torch.device("cuda"), "Some New GPU", "bf16") == "unknown_gpu"
    assert M.peak_missing_reason(torch.device("cuda"), "Tesla T4", "bf16") == "no_peak_for_dtype"
    assert set(M.MFU_REASONS) >= {"not_cuda", "unknown_gpu", "no_peak_for_dtype", "no_peak"}


FAKE_SMI = """#!{python}
import sys, time
if not any(a.startswith("--loop-ms") for a in sys.argv):
    print("0x0000000000000000")
    sys.exit(0)
while True:
    print(time.strftime("%Y/%m/%d %H:%M:%S") + ".000, 97, 40, 30000, 1410, 1215, 300.5, 60, 0x0000000000000000",
          flush=True)
    time.sleep(0.02)
"""


@pytest.mark.skipif(os.name == "nt", reason="a shebang script stands in for nvidia-smi")
def test_sampler_end_to_end_with_a_fake_nvidia_smi(tmp_path, monkeypatch):
    smi = tmp_path / "nvidia-smi"
    smi.write_text(FAKE_SMI.format(python=sys.executable))
    smi.chmod(0o755)
    monkeypatch.setenv("PATH", f"{tmp_path}{os.pathsep}{os.environ['PATH']}")
    with M.GpuSampler(torch.device("cuda"), interval_ms=20) as s:
        time.sleep(0.5)
    out = s.summary()
    assert out["samples"] >= 3 and out["metrics"]["power_w"]["median"] == 300.5
    assert out["slowdown_samples"] == 0
    assert len(out["series"]) == out["samples"] and out["series"][0]["t_s"] == 0.0
    assert all(r["power_w"] == 300.5 and r["slowdown"] == [] for r in out["series"])
    assert s._proc is None                                                  # the child was stopped
