"""MFU, HFU, memory, GPU telemetry, run metadata.  [implemented; goodput and the NVML sampler are extras]

The one place these numbers are computed: perfkit.bench calls it, and G8 gets the same formulas (docs/METRICS.md).

    gpu_peak(device_name)          {"spec": "A100", "bf16": 312, "fp16": 312, "fp32": 19.5}, {} when unknown
    peak_tflops(device_name, dtype)  (dense peak TFLOPS for this dtype or None, spec or None)
    run_metadata(device)           the device, version and git fields of the run JSON "meta" (perfkit/schema.py)
    flops_per_token(model, T, opt_flops_per_step=0, tokens_per_step=None)
                                   model_flops_per_token, hw_flops_per_token, params_total, params_active
    mfu(flops, tokens_per_s, peak_tflops, reason=None)
                                   {"mfu", "hfu", "model_tflops_per_s", "mfu_reason"}
    peak_missing_reason(device, device_name, dtype)   why there is no peak: not_cuda | unknown_gpu | no_peak_for_dtype
    memory(device)                 peak allocated / reserved GB (None off CUDA)
    gpu_telemetry(device)          SM clock, temperature, power, throttle reasons from nvidia-smi (None off CUDA)
    telemetry_during(device, work) the same, sampled while work() keeps the GPU busy
    throttle_flags(tel)            slowdown reasons in a sample (power cap, thermal, ...), None if not reported
    GpuSampler(device)             GPU utilization, memory, clocks, power, temperature sampled through the clean
                                   steps (run["system"], for G8): the summary and every 1 s sample

FLOPs per token (T = sequence length, L = layers, d = width, N_active from GPT.param_counts):
    model_flops_per_token = 6 * N_active + 12 * L * d * T      (PaLM; attention scores counted non-causal)
    hw_flops_per_token    = model + 2 forwards of the stack when reversible (invert + recompute)
                            + 2 * heads again when ce_impl=chunked (it recomputes the head, and the MTP head)
                            + 6 * (num_experts - top_k) * expert params per MoE layer when moe.dispatch=dense
                              (every expert runs on every token; a reversible stack recomputes those too)
                            - 6 * (D - P) * d when ke_mode=sparse (it gathers P columns of the KE projection,
                              not a [D, d] matmul)
                            + optimizer FLOPs per step (Muon Newton-Schulz) / tokens per step
N_active counts the MTP head when model.mtp is on (it is a second LM head every token multiplies with).
MFU uses model FLOPs, HFU hardware FLOPs; both against the dense vendor peak for the dtype the run used.

Extra, still to write: goodput (time and tokens, badput buckets), HBM GB/s per GPU for the roofline.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]

# Dense (no sparsity) vendor peak TFLOPS per dtype. First substring match wins, so specific names come
# first. An empty table = known GPU, unknown peak: MFU is left empty and the run says so (or pass --peak-tflops).
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


# ------------------------------------------------------------------------------ device and peak

def gpu_peak(device_name: str) -> dict:
    for key, table in PEAKS:
        if key.lower() in device_name.lower():
            return {"spec": key, **table}
    return {}


def peak_tflops(device_name: str, dtype: str):
    """(peak TFLOPS for this dtype or None, gpu_spec or None)."""
    p = gpu_peak(device_name)
    return p.get(dtype), p.get("spec")


MFU_REASONS = ("not_cuda", "unknown_gpu", "no_peak_for_dtype", "no_peak")


def peak_missing_reason(device, device_name: str, dtype: str) -> str:
    """Why MFU is empty (MFU_REASONS): not a CUDA GPU, a GPU missing from PEAKS, or no peak for this dtype."""
    if device.type != "cuda":
        return "not_cuda"
    p = gpu_peak(device_name)
    return "unknown_gpu" if not p else "no_peak_for_dtype"


def device_name(device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple MPS"
    return platform.processor() or platform.machine() or "cpu"


def _sh(args, cwd=None):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=10, check=True, cwd=cwd).stdout.strip()
    except Exception:
        return None


def git_state():
    """(commit, dirty). Dirty = any change in teams/06-performance outside results/, new files included
    (a new module the bench imports is code the commit ID does not name). Not counted: __pycache__,
    .pytest_cache, .ipynb_checkpoints, .DS_Store (they appear without anyone editing code).
    dirty is None when git status cannot run; commit is then PERF_COMMIT when there is no .git."""
    commit = _sh(["git", "rev-parse", "HEAD"], cwd=ROOT)
    if commit is None:                    # e.g. a Colab upload without .git
        return os.environ.get("PERF_COMMIT"), None
    # Python and pytest caches, notebook checkpoints (Colab) and Finder files (macOS) are not code: they appear
    # without anyone editing a file (no .gitignore covers them, and pr-guard allows none in the team folder).
    status = _sh(["git", "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)results",
                  ":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/.pytest_cache/**",
                  ":(exclude,glob)**/.ipynb_checkpoints/**", ":(exclude,glob)**/.DS_Store"], cwd=ROOT)
    return commit, bool(status) if status is not None else None


_SMI = "name,driver_version,pstate,clocks.sm,clocks.max.sm,clocks.mem,temperature.gpu,power.draw,power.limit"


def _smi_select(device) -> str:
    """The nvidia-smi --id for this device, by PCI bus ID: nvidia-smi and CUDA may number GPUs differently."""
    try:
        props = torch.cuda.get_device_properties(device)
        dom, bus, dv = (getattr(props, f"pci_{k}_id", None) for k in ("domain", "bus", "device"))
    except (AssertionError, RuntimeError):
        dom = bus = dv = None
    if all(isinstance(v, int) for v in (dom, bus, dv)):
        return f"--id={dom:08X}:{bus:02X}:{dv:02X}.0"
    return f"--id={device.index or 0}"


def gpu_telemetry(device) -> dict | None:
    """SM clock, temperature and power now: explains a 5% miss on a shared or hot GPU."""
    if device.type != "cuda":
        return None
    sel = _smi_select(device)
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
    for field in REASON_FIELDS:
        r = _sh(["nvidia-smi", sel, f"--query-gpu={field}", "--format=csv,noheader"])
        if r:
            try:
                res["throttle_reasons"] = int(r.splitlines()[0].strip(), 16)
                break
            except ValueError:
                pass
    return res


REASON_FIELDS = ("clocks_event_reasons.active", "clocks_throttle_reasons.active")   # new name, old name


# nvidia-smi clock event reasons that mean the GPU was slowed down. GpuIdle (0x1), application clocks (0x2),
# sync boost (0x10) and display clocks (0x100) are not slowdowns: a GPU sampled between steps reports GpuIdle.
SLOWDOWN = {0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x20: "SwThermalSlowdown", 0x40: "HwThermalSlowdown",
            0x80: "HwPowerBrakeSlowdown"}


def throttle_flags(tel: dict | None) -> list | None:
    """The slowdown reasons active in a telemetry sample; None when the driver did not report them."""
    if not tel or "throttle_reasons" not in tel:
        return None
    return [name for bit, name in SLOWDOWN.items() if tel["throttle_reasons"] & bit]


def telemetry_during(device, work, delay_s: float = 0.2) -> dict | None:
    """gpu_telemetry sampled from a thread while work() runs (a few untimed steps), so the clock and the
    reasons describe a busy GPU. Sampling after the steps sees an idle GPU at a lowered clock."""
    if device.type != "cuda":
        work()
        return None
    import threading
    box = {}

    def sample():
        time.sleep(delay_s)
        box["tel"] = gpu_telemetry(device)
    t = threading.Thread(target=sample, daemon=True)
    t.start()
    work()
    t.join(timeout=30)
    return box.get("tel")


# nvidia-smi query field -> key in run["system"]["metrics"] (docs/METRICS.md says what each one means)
SAMPLE_FIELDS = {"utilization.gpu": "gpu_util_pct", "utilization.memory": "mem_util_pct",
                 "memory.used": "mem_used_mib", "clocks.sm": "sm_clock_mhz", "clocks.mem": "mem_clock_mhz",
                 "power.draw": "power_w", "temperature.gpu": "temp_c"}


class GpuSampler:
    """System metrics while the measured steps run, for G8 (docs/METRICS.md).

        with GpuSampler(device) as s:
            run_clean(...)
        run["system"] = s.summary()        # None off CUDA or without nvidia-smi

    One nvidia-smi child process in loop mode (--loop-ms) writes one sample per interval to a temp file. Nothing
    runs in this process, so the timed steps pay only the child's own CPU (one sample a second by default).
    summary(): min / median / max / mean of every SAMPLE_FIELDS metric, and how many samples had a slowdown
    reason (SLOWDOWN) active, with their names. "series" keeps every sample (one per interval, 1 s by default;
    G8 rolls them up into its own windows): t_s = seconds since the first sample (nvidia-smi's timestamp, None
    when it cannot be read), the SAMPLE_FIELDS keys (None for [N/A]) and "slowdown" (reason names, [] for
    none, None when the driver gives no reasons). "start_time" is the first sample's nvidia-smi timestamp.
    """

    def __init__(self, device, interval_ms: int = 1000):
        self.device, self.interval_ms = device, interval_ms
        self.fields = list(SAMPLE_FIELDS)
        self.reason = None
        self.ts_field = "timestamp"                  # first column of every sample line
        self.started = False
        self.rows = []
        self._proc = self._file = None

    def __enter__(self):
        if self.device.type != "cuda" or shutil.which("nvidia-smi") is None:
            return self
        sel = _smi_select(self.device)
        self.reason = next((f for f in REASON_FIELDS
                            if _sh(["nvidia-smi", sel, f"--query-gpu={f}", "--format=csv,noheader"])), None)
        query = self._query()
        self._file = tempfile.TemporaryFile(mode="w+")
        self._proc = subprocess.Popen(["nvidia-smi", sel, f"--query-gpu={','.join(query)}",
                                       "--format=csv,noheader,nounits", f"--loop-ms={self.interval_ms}"],
                                      stdout=self._file, stderr=subprocess.DEVNULL, text=True)
        self.started = True
        return self

    def __exit__(self, *exc):
        if self._proc is None:
            return False
        self._proc.terminate()
        try:
            self._proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._file.seek(0)
        self.rows = [r for r in map(self.parse_line, self._file.read().splitlines()) if r]
        self._file.close()
        self._proc = self._file = None
        return False

    def _query(self) -> list:
        return [self.ts_field] + self.fields + ([self.reason] if self.reason else [])

    def parse_line(self, line: str) -> dict | None:
        """One CSV line -> {timestamp: str, field: float, reason field: int}; None for a line with the wrong
        number of values."""
        query = self._query()
        vals = [v.strip() for v in line.split(",")]
        if len(vals) != len(query):
            return None
        row = {}
        for f, v in zip(query, vals):
            try:
                row[f] = v if f == self.ts_field else int(v, 16) if f == self.reason else float(v)
            except ValueError:                        # [N/A], [Not Supported]
                row[f] = None
        return row

    @staticmethod
    def _seconds(ts):
        """nvidia-smi timestamp '2026/10/29 07:48:04.123' -> seconds since the epoch (local time), or None."""
        from datetime import datetime
        try:
            return datetime.strptime(ts, "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except (TypeError, ValueError):
            return None

    def series(self) -> list:
        t = [self._seconds(r.get(self.ts_field)) for r in self.rows]
        t0 = next((v for v in t if v is not None), None)
        out = []
        for r, tv in zip(self.rows, t):
            row = {"t_s": None if tv is None or t0 is None else round(tv - t0, 3)}
            row.update({key: r.get(f) for f, key in SAMPLE_FIELDS.items()})
            flags = throttle_flags({"throttle_reasons": r[self.reason]}) if self.reason and \
                r.get(self.reason) is not None else None
            row["slowdown"] = flags
            out.append(row)
        return out

    def summary(self) -> dict | None:
        if not self.started:
            return None
        from perfkit.timer import stats
        out = {"source": f"nvidia-smi --loop-ms={self.interval_ms}", "interval_ms": self.interval_ms,
               "during": "clean steps", "samples": len(self.rows), "metrics": {}}
        for f, key in SAMPLE_FIELDS.items():
            vals = [r[f] for r in self.rows if r.get(f) is not None]
            if vals:
                st = stats(vals)
                out["metrics"][key] = {k: st[k] for k in ("min", "median", "max", "mean")}
        if self.reason:
            flagged = [f for r in self.rows if r.get(self.reason) is not None
                       and (f := throttle_flags({"throttle_reasons": r[self.reason]}))]
            out["slowdown_samples"] = len(flagged)
            out["slowdown_reasons"] = sorted({n for names in flagged for n in names})
        out["start_time"] = self.rows[0].get(self.ts_field) if self.rows else None
        out["series"] = self.series()
        return out


def run_metadata(device) -> dict:
    """gpu, gpu_spec, torch, cuda, driver, device, commit, dirty, python (+ compute capability, SM count,
    memory, GPU count and telemetry_start on CUDA). The bench adds the run fields (config, tag, dtype, ...)."""
    commit, dirty = git_state()
    name = device_name(device)
    tel = gpu_telemetry(device)
    meta = {
        "gpu": name, "gpu_spec": gpu_peak(name).get("spec"),
        "torch": torch.__version__,
        "cuda": torch.version.cuda if device.type == "cuda" else None,
        "driver": (tel or {}).get("driver_version") if device.type == "cuda" else None,
        "device": device.type, "commit": commit, "dirty": dirty,
        "python": platform.python_version(),
    }
    if device.type == "cuda":
        p = torch.cuda.get_device_properties(device)
        meta.update(compute_capability=f"{p.major}.{p.minor}", sm_count=p.multi_processor_count,
                    gpu_mem_gb=round(p.total_memory / 2**30, 1), gpu_count=torch.cuda.device_count(),
                    telemetry_start=tel)
    return meta


# ------------------------------------------------------------------------------ FLOPs, MFU, memory

def flops_per_token(model, T: int, opt_flops_per_step: float = 0.0, tokens_per_step: int | None = None) -> dict:
    """model: a models.gpt.GPT. See the module docstring for the formulas."""
    m = model.cfg
    total, active = model.param_counts()
    L, d = m.n_layer, m.n_embd
    model_fpt = 6 * active + 12 * L * d * T
    head = model.head.weight.numel() + (model.mtp_head.weight.numel() if model.mtp_head is not None else 0)
    ke = model.tok.proj.weight.numel() if m.embedding == "ke1" else 0
    idle = 0                                        # expert weights that run but are not counted as active
    for moe in model.moe_layers():
        if moe.dispatch == "dense":
            per_expert = moe.experts[0].c_fc.weight.numel() + moe.experts[0].c_proj.weight.numel()
            idle += per_expert * (moe.num_experts - moe.top_k)
    stack_fwd = 2 * (active - head - ke + idle) + 4 * L * d * T
    hw = model_fpt + 6 * idle + (2 * stack_fwd if m.reversible else 0) + (2 * head if m.ce_impl == "chunked" else 0)
    if m.embedding == "ke1" and m.ke_mode == "sparse":
        hw -= 6 * (ke - m.ke_max_pos * d)                    # ke = D * d; sparse runs P * d of it
    if opt_flops_per_step and tokens_per_step:
        hw += opt_flops_per_step / tokens_per_step
    return {"model_flops_per_token": model_fpt, "hw_flops_per_token": hw,
            "params_total": total, "params_active": active}


def mfu(flops: dict, tokens_per_s: float, peak_tflops: float | None, reason: str | None = None) -> dict:
    """flops: from flops_per_token (hw_flops_per_token optional). mfu / hfu are None without a peak, and then
    mfu_reason says why (reason, from peak_missing_reason; "no_peak" when not given). None means missing, not 0."""
    out = {"model_tflops_per_s": flops["model_flops_per_token"] * tokens_per_s / 1e12, "mfu": None, "hfu": None,
           "mfu_reason": None if peak_tflops else (reason or "no_peak")}
    if peak_tflops:
        out["mfu"] = flops["model_flops_per_token"] * tokens_per_s / (peak_tflops * 1e12)
        if flops.get("hw_flops_per_token") is not None:
            out["hfu"] = flops["hw_flops_per_token"] * tokens_per_s / (peak_tflops * 1e12)
    return out


def memory(device) -> dict | None:
    if device.type != "cuda":
        return None
    return {"peak_allocated_gb": torch.cuda.max_memory_allocated(device) / 2**30,
            "peak_reserved_gb": torch.cuda.max_memory_reserved(device) / 2**30}
