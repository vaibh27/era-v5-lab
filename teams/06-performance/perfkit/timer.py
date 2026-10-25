"""Step timing that does not change what it measures.  [implemented]

GPU work is asynchronous: time.time() around loss.backward() measures how fast the CPU *queued* the
kernels, and a synchronize() around every part removes the CPU/GPU overlap the step really has. So
every number here comes from CUDA events recorded on the stream the work runs on, and is read back
after the step. Nothing in this file synchronizes inside a timed step.

  sync(device)                       wait for cuda / mps / cpu (only outside a timed step)
  StepTimer(device)                  one event pair per step: start() / stop(); times_ms() syncs once
  stats(xs)                          median, p10, p90, iqr, min, max, n, spread_pct. Never the mean.
  Buckets(device)                    wall-clock seconds per named bucket (startup, compile, checkpoint)
  EventPool()                        reusable CUDA events, so a timed step allocates none
  EventRecorder(device, max_level)   layer mode: the recorder protocol of perfkit/regions.py
  ProfilerRecorder(max_level)        trace mode: record_function + NVTX ranges with the same names
  backward_segments(points, ...)     bwd:<part> times cut from mark() points (bench.py step 6)
  aggregate(steps)                   per-region stats over steps, with level and calls_per_step
  paired_overhead(seq)               recorder cost from neighbouring on / off steps
  kernel_summary(kernels, steps)     top GPU ops, GPU busy %, kernels per step (trace mode)
  count_syncs(fn)                    host syncs a call makes, and the source lines that make them

Rules (docs/PLAN.md 2.5): no synchronize / .item() / print inside a timed step; read events after
the steps; report medians and percentiles, never means.

On CPU, perf_counter is exact (CPU work is synchronous). On MPS there are no timing events we rely
on, so StepTimer and EventRecorder synchronize at every boundary there; MPS numbers are flagged.
"""
from __future__ import annotations

import math
import threading
import time
import warnings
from collections import Counter, defaultdict
from contextlib import contextmanager
from pathlib import Path

import torch

STAT_KEYS = ("median", "p10", "p90", "iqr", "min", "max", "n", "spread_pct")


# ------------------------------------------------------------------------------ basics

def _dev(device) -> torch.device:
    return device if isinstance(device, torch.device) else torch.device(device)


def sync(device):
    """Wait until the device has finished all queued work. Never inside a timed step."""
    d = _dev(device)
    if d.type == "cuda":
        torch.cuda.synchronize(d)
    elif d.type == "mps":
        torch.mps.synchronize()


def _quantile(sorted_xs, q):
    """Linear interpolation between closest ranks (numpy.percentile's default)."""
    k = (len(sorted_xs) - 1) * q
    lo = math.floor(k)
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (k - lo)


def stats(xs) -> dict:
    """median, p10, p90, iqr (p75 - p25), min, max, n, and spread_pct = (p90 - p10) / median * 100.

    spread_pct is the quick check for the 5% gate: a part whose own p10..p90 spread is above ~5% is
    unlikely to agree within 5% between two runs.
    """
    a = sorted(float(x) for x in xs)
    if not a:
        return {k: (0 if k == "n" else None) for k in STAT_KEYS}
    med = _quantile(a, 0.5)
    p10, p90 = _quantile(a, 0.10), _quantile(a, 0.90)
    return {
        "median": med, "p10": p10, "p90": p90,
        "iqr": _quantile(a, 0.75) - _quantile(a, 0.25),
        "min": a[0], "max": a[-1], "n": len(a),
        "spread_pct": (p90 - p10) / med * 100 if med else None,
    }


class StepTimer:
    """Times whole steps.

        t = StepTimer(device)
        for ...:
            t.start(); step(); t.stop()
        ms = t.times_ms()                 # one sync, then every step's time

    CUDA: an event pair per step on the current stream, no sync until times_ms().
    CPU: perf_counter (exact). MPS: synchronize + perf_counter at start and stop.
    """

    def __init__(self, device):
        self.device = _dev(device)
        self.kind = {"cuda": "event", "mps": "sync"}.get(self.device.type, "wall")
        self._pairs = []
        self._open = None

    def _mark(self):
        if self.kind == "event":
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            return ev
        if self.kind == "sync":
            sync(self.device)
        return time.perf_counter()

    def start(self):
        if self._open is not None:
            raise RuntimeError("StepTimer.start() called twice without stop()")
        self._open = self._mark()

    def stop(self):
        if self._open is None:
            raise RuntimeError("StepTimer.stop() without start()")
        self._pairs.append((self._open, self._mark()))
        self._open = None

    @contextmanager
    def step(self):
        self.start()
        try:
            yield
        finally:
            self.stop()

    def times_ms(self) -> list:
        """Every recorded step's time in ms. Synchronizes once (on the last event)."""
        if self.kind != "event":
            return [(b - a) * 1e3 for a, b in self._pairs]
        if self._pairs:
            self._pairs[-1][1].synchronize()          # events on one stream finish in order
        out = []
        for a, b in self._pairs:
            if not b.query():                         # work on another stream: wait for it too
                b.synchronize()
            out.append(a.elapsed_time(b))
        return out

    def reset(self):
        self._pairs.clear()
        self._open = None

    def __len__(self):
        return len(self._pairs)


class Buckets:
    """Wall-clock seconds per named bucket, for goodput / badput: startup, compile, eval, checkpoint.

        b = Buckets(device)
        with b("startup"):  build()
        b.seconds  -> {"startup": 3.2}

    Synchronizes at bucket edges (the work must really be done), so never use it inside a timed step.
    """

    def __init__(self, device):
        self.device = _dev(device)
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    @contextmanager
    def __call__(self, name: str):
        sync(self.device)
        t0 = time.perf_counter()
        try:
            yield
        finally:
            sync(self.device)
            self.seconds[name] += time.perf_counter() - t0
            self.calls[name] += 1

    def as_dict(self) -> dict:
        return {k: round(v, 4) for k, v in self.seconds.items()}


# ------------------------------------------------------------------------------ layer mode

class EventPool:
    """CUDA timing events to reuse. Creating an event inside a timed step costs CPU time and shows up
    as recorder overhead, so the bench reserves enough events before it measures."""

    def __init__(self):
        self._free = []
        self.created = 0

    def get(self):
        try:
            return self._free.pop()
        except IndexError:
            self.created += 1
            return torch.cuda.Event(enable_timing=True)

    def put(self, events):
        self._free.extend(events)

    def reserve(self, n: int):
        while len(self._free) < n:
            self._free.append(torch.cuda.Event(enable_timing=True))
            self.created += 1

    def __len__(self):
        return len(self._free)


class Clock:
    """A point in time, as ms since the recorder was made. clock.ms_since(other) -> ms."""

    __slots__ = ("ms",)

    def __init__(self, ms: float):
        self.ms = ms

    def ms_since(self, other: "Clock") -> float:
        return self.ms - other.ms

    def __repr__(self):
        return f"Clock({self.ms:.4f} ms)"


class Collected:
    """What EventRecorder.collect() returns. Unpacks as (spans, points), the perfkit/regions.py protocol:
        spans  [(full_name, level, ms)]      points [(name, level, Clock)] in the order they happened
    starts: each span's start, ms since the recorder was made (to list parts in the order they run)."""

    __slots__ = ("spans", "points", "starts")

    def __init__(self, spans, points, starts):
        self.spans, self.points, self.starts = spans, points, starts

    def __iter__(self):
        return iter((self.spans, self.points))


class _Span:
    __slots__ = ("rec", "name", "level", "t0")

    def __init__(self, rec, name, level):
        self.rec, self.name, self.level = rec, name, level

    def __enter__(self):
        self.t0 = self.rec._now()
        return self

    def __exit__(self, *exc):
        t1 = self.rec._now()
        with self.rec._lock:
            self.rec._spans.append((self.name, self.level, self.t0, t1))
        return False


class EventRecorder:
    """Layer mode: the recorder of perfkit/regions.py (protocol: max_level, region, point, collect).

    One recorder per timed step:
        rec = EventRecorder(device, max_level=2, pool=pool)
        with use_recorder(rec):  step()
        ... later: spans, points = rec.collect()

    region(name, level) -> context that records an event at entry and exit. point(name, level) records
    one event (mark() calls it when the gradient reaches a tensor). Both may be called from the autograd
    thread during a CUDA backward: appends are locked, and events go on the calling thread's current
    stream, which autograd sets to the forward's stream.

    collect() waits for this step's last event only (not the whole device) and returns
        spans:  [(full_name, level, ms)]
        points: [(name, level, Clock)] in the order they happened
    then gives the events back to the pool. ready() says, without waiting, whether collect() would wait.
    Cost: a few microseconds of CPU per event, ~0 GPU.
    """

    def __init__(self, device, max_level: int = 2, pool: EventPool | None = None):
        self.device = _dev(device)
        self.max_level = max_level
        self.kind = {"cuda": "event", "mps": "sync"}.get(self.device.type, "wall")
        self.pool = pool if pool is not None else (EventPool() if self.kind == "event" else None)
        self._lock = threading.Lock()
        self._spans, self._points = [], []
        self._last = None
        self._origin = self._now()
        self._result = None

    def _now(self):
        if self.kind == "event":
            ev = self.pool.get()
            ev.record()
            self._last = ev
            return ev
        if self.kind == "sync":
            sync(self.device)
        return time.perf_counter()

    def region(self, name: str, level: int = 2):
        return _Span(self, name, level)

    def point(self, name: str, level: int = 1):
        m = self._now()
        with self._lock:
            self._points.append((name, level, m))

    def n_events(self) -> int:
        return 1 + 2 * len(self._spans) + len(self._points)

    def ready(self) -> bool:
        return self._result is not None or self.kind != "event" or self._last.query()

    def collect(self) -> Collected:
        if self._result is not None:
            return self._result
        if self.kind == "event":
            self._last.synchronize()

            def ms(a, b):
                if not b.query():
                    b.synchronize()
                return a.elapsed_time(b)

            o = self._origin
            spans = [(n, lv, ms(s, e)) for n, lv, s, e in self._spans]
            starts = [ms(o, s) for _, _, s, _ in self._spans]
            points = [(n, lv, Clock(ms(o, m))) for n, lv, m in self._points]
            used = [o] + [m for _, _, m in self._points] + [e for _, _, s, e in self._spans] + \
                   [s for _, _, s, _ in self._spans]
            self.pool.put(used)
        else:
            o = self._origin
            spans = [(n, lv, (e - s) * 1e3) for n, lv, s, e in self._spans]
            starts = [(s - o) * 1e3 for _, _, s, _ in self._spans]
            points = [(n, lv, Clock((m - o) * 1e3)) for n, lv, m in self._points]
        self._spans, self._points, self._last = [], [], None
        self._result = Collected(spans, points, starts)
        return self._result


class _ProfRange:
    __slots__ = ("name", "nvtx", "rf")

    def __init__(self, name, nvtx):
        self.name, self.nvtx = name, nvtx

    def __enter__(self):
        self.rf = torch.profiler.record_function(self.name)
        self.rf.__enter__()
        if self.nvtx:
            torch.cuda.nvtx.range_push(self.name)
        return self

    def __exit__(self, *exc):
        if self.nvtx:
            torch.cuda.nvtx.range_pop()
        self.rf.__exit__(*exc)
        return False


class ProfilerRecorder:
    """Trace mode. region() opens torch.profiler.record_function(name) (+ an NVTX range on CUDA, for
    Nsight Systems); point() records a zero-length range "__point__:<name>" so trace tools can cut
    backward segments. Region names in the trace are the same as in the run JSON."""

    def __init__(self, max_level: int = 3, nvtx: bool | None = None):
        self.max_level = max_level
        self.nvtx = torch.cuda.is_available() if nvtx is None else nvtx

    def region(self, name: str, level: int = 2):
        return _ProfRange(name, self.nvtx)

    def point(self, name: str, level: int = 1):
        with torch.profiler.record_function(f"__point__:{name}"):
            pass


# ------------------------------------------------------------------------------ turning events into parts

# A backward segment takes the level of the forward region with the same name (bench.py step 6).
# These are used only when that forward region was not recorded (its level is above max_level).
_SEGMENT_LEVEL_FALLBACK = {"embed": 2, "head+loss": 2, "stack": 1}


def backward_segments(points, fwd_levels: dict, max_level: int, with_start: bool = False) -> list:
    """Cut bwd:<part> times from the points mark() recorded during one backward.

    points: [(name, level, Clock)] in the order they happened, bracketed by "__bwd_start__" and
    "__bwd_end__". A point of level L closes "bwd:<name>", which starts at the previous point of level
    <= L. "__bwd_end__" closes "bwd:embed", which starts at the last "stack" point.

        plain stack:      __bwd_start__  head+loss  block8 .. block0  stack  __bwd_end__
          -> bwd:head+loss, bwd:block8 .. bwd:block0, bwd:stack (= all blocks), bwd:embed
        reversible stack: __bwd_start__  head+loss  stack  __bwd_end__   (its backward times its blocks)
          -> bwd:head+loss, bwd:stack, bwd:embed

    fwd_levels: {"fwd:<name>": level} from the same step. Returns [(name, level, ms)] with level <= max_level
    (with_start: [(name, level, ms, start ms)]).
    """
    names = [p[0] for p in points]
    if "__bwd_start__" not in names:
        return []
    i0 = names.index("__bwd_start__")
    seq = points[i0:]
    out = []
    for j, (name, lvl, clk) in enumerate(seq):
        if j == 0:
            continue
        if name == "__bwd_end__":
            start = next((p for p in reversed(seq[:j]) if p[0] == "stack"), None)
            if start is not None:
                level = fwd_levels.get("fwd:embed", _SEGMENT_LEVEL_FALLBACK["embed"])
                out.append(("bwd:embed", level, clk.ms_since(start[2]), start[2].ms))
            break
        k = j - 1
        while k > 0 and seq[k][1] > lvl:
            k -= 1
        level = fwd_levels.get(f"fwd:{name}", _SEGMENT_LEVEL_FALLBACK.get(name, lvl))
        out.append((f"bwd:{name}", level, clk.ms_since(seq[k][2]), seq[k][2].ms))
    return [s if with_start else s[:3] for s in out if s[1] <= max_level]


def step_parts(collected: Collected, max_level: int) -> list:
    """All parts of one recorded step, [(name, level, ms, start ms)]: its regions plus the backward
    segments cut from its points."""
    spans, points = collected
    fwd_levels = {}
    for name, level, _ in spans:
        if name.startswith("fwd:"):
            fwd_levels[name] = min(level, fwd_levels.get(name, level))
    parts = [(n, lv, ms, st) for (n, lv, ms), st in zip(spans, collected.starts)]
    return parts + backward_segments(points, fwd_levels, max_level, with_start=True)


def aggregate(steps: list) -> dict:
    """steps: one list of (name, level, ms[, start ms]) per step -> {name: stats + level + calls_per_step}.

    A part called several times in a step (opt.muon.ns once per matrix) is summed within the step,
    then summarised over steps. A step that never entered a part counts as 0 ms for it.
    Parts are listed in the order they run (median first start), a parent before its children.
    """
    order, level, first = {}, {}, defaultdict(list)
    per_step = []
    for parts in steps:
        tot, calls, seen = defaultdict(float), Counter(), {}
        for name, lv, ms, *start in parts:
            order.setdefault(name, len(order))
            level[name] = min(lv, level.get(name, lv))
            tot[name] += ms
            calls[name] += 1
            if start:
                seen[name] = min(start[0], seen.get(name, start[0]))
        for name, st in seen.items():
            first[name].append(st)
        per_step.append((tot, calls))
    meds = {n: stats([t.get(n, 0.0) for t, _ in per_step])["median"] for n in order}

    def key(n):
        if first.get(n):
            return (stats(first[n])["median"], -meds[n], order[n])
        return (math.inf, 0, order[n])
    out = {}
    for name in sorted(order, key=key):
        st = stats([t.get(name, 0.0) for t, _ in per_step])
        c = sorted(cl.get(name, 0) for _, cl in per_step)
        st["level"] = level[name]
        st["calls_per_step"] = c[len(c) // 2]
        out[name] = st
    return out


def paired_overhead(seq) -> float | None:
    """seq: [(recorder_on, ms)] in step order, alternating. Returns the recorder overhead in %.

    Each on-step is compared with the mean of its off-step neighbours, so slow drift (clocks, heat)
    cancels out; the median of those ratios is the overhead.
    """
    ratios = []
    for i, (on, ms) in enumerate(seq):
        if not on:
            continue
        nb = [seq[j][1] for j in (i - 1, i + 1) if 0 <= j < len(seq) and not seq[j][0]]
        if nb and sum(nb) > 0:
            ratios.append(ms / (sum(nb) / len(nb)))
    if not ratios:
        return None
    return (stats(ratios)["median"] - 1.0) * 100.0


# ------------------------------------------------------------------------------ trace mode

def profiler_device_events(prof) -> list:
    """GPU activity (kernels, memcpy, memset) from a finished torch.profiler run: [(name, start_us, end_us)]."""
    cuda = getattr(torch.autograd.DeviceType, "CUDA", None)
    out = []
    for e in prof.events():
        if getattr(e, "device_type", None) == cuda:
            tr = e.time_range
            out.append((e.name, float(tr.start), float(tr.end)))
    return out


def _is_kernel(name: str) -> bool:
    n = name.lower()
    return not (n.startswith("memcpy") or n.startswith("memset"))


def kernel_summary(device_events, n_steps: int, top: int = 15) -> dict:
    """top_ops, gpu_busy_pct and kernels_per_step from [(name, start_us, end_us)] over n_steps steps.

    gpu_busy_pct = time the GPU runs anything / span from the first to the last activity. Gaps inside
    that span are the GPU waiting for the CPU: launch overhead, host syncs (MoE loop dispatch),
    Python. (Gaps between steps are not counted: the trace steps are not back to back.)
    """
    if not device_events:
        return {"top_ops": [], "gpu_busy_pct": None, "kernels_per_step": 0}
    n = max(1, n_steps)
    iv = sorted((s, e) for _, s, e in device_events)
    busy, cur_s, cur_e = 0.0, iv[0][0], iv[0][1]
    for s, e in iv[1:]:
        if s > cur_e:
            busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    busy += cur_e - cur_s
    span = max(e for _, e in iv) - min(s for s, _ in iv)
    total, calls = defaultdict(float), Counter()
    for name, s, e in device_events:
        total[name] += e - s
        calls[name] += 1
    all_us = sum(total.values()) or 1.0
    ops = sorted(total, key=total.get, reverse=True)[:top]
    return {
        "top_ops": [{"name": k[:160], "ms_per_step": total[k] / 1e3 / n, "calls_per_step": calls[k] / n,
                     "pct": 100.0 * total[k] / all_us} for k in ops],
        "gpu_busy_pct": 100.0 * busy / span if span > 0 else None,
        "kernels_per_step": sum(c for k, c in calls.items() if _is_kernel(k)) / n,
    }


def count_syncs(fn, root: Path | None = None) -> tuple:
    """Run fn() with torch.cuda.set_sync_debug_mode("warn") and count host syncs.

    Returns (count, {"file.py:line": count}): every .item(), nonzero(), tok.any() in an if, and so on.
    Each one stops the CPU from queueing work ahead of the GPU (docs/PARTS.md section 3). CUDA only.
    """
    if not torch.cuda.is_available():
        return 0, {}
    prev = torch.cuda.get_sync_debug_mode()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            fn()
        finally:
            torch.cuda.set_sync_debug_mode(prev)
    sites = Counter()
    for w in caught:
        if "synchroniz" not in str(w.message):
            continue
        f = Path(w.filename)
        if root is not None:
            try:
                f = f.resolve().relative_to(root)
            except ValueError:
                f = Path(f.name)
        sites[f"{f.as_posix()}:{w.lineno}"] += 1
    return sum(sites.values()), dict(sites.most_common())
