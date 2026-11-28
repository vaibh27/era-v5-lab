"""Issue #26 (G8 core): fault-injection test for dead experts, GPU memory and speed / MFU, with time to alert.

A pytest fixture writes a mock metrics.jsonl: a seeded synthetic stream sent through the real MetricsWriter
(flush_every = 10), so the file carries the fields the writer derives (dead_experts_count, expert_load_imbalance,
gpu_memory_used in bytes) exactly as a trainer's file would. Two AlarmMonitors read the file line by line after
every write, the way `alarms.py --jsonl` does:

  A  AlarmMonitor(), what `alarms.py --jsonl` runs: no bounds, moe_patience 1,000, tps_patience 100.
  B  tuned: moe_patience 10, plus static_bound on gpu_memory_used (hi 76 GiB), model_flops_utilization
     (lo 0.30) and dead_experts_count (hi 0.5).

Only the file path is measured; no monitor runs inside the trainer. Time to alert runs from MetricsWriter.write()
of the alert step's row until the file-reading monitor fires for it.

Faults (expert faults hit expert 0; the number of experts is a parameter, tested at 8, 16 and 64):
  - expert 0 gets no tokens from step 200 on;
  - expert 0's share falls from a fair share to 0.0 over steps 200-249;
  - GPU memory climbs from 60 to 80 GiB over steps 300-499 and the process dies of OOM at 499;
  - GPU memory jumps to 80 GiB at step 205 and the process dies at 206;
  - tokens_per_second (and MFU, which is proportional to it) falls 30% over 150 steps, or over 300 steps.

Run the test:   python -m pytest teams/08-monitoring/alarms/test_alert_injection_nikshrimali.py -q
Print the rows: python teams/08-monitoring/alarms/test_alert_injection_nikshrimali.py
"""
import json
import math
import random
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "metrics_logger"))
from alarms import AlarmMonitor  # noqa: E402
from metrics_writer import DEAD_EXPERT_SHARE, FLUSH_EVERY, MetricsWriter  # noqa: E402

SEED = 26
STEPS = 1300                     # long enough for moe_routing's default 1,000-step patience to run out
GIB = 2 ** 30
BASE_TPS = 50_000.0
BASE_MFU = 0.38
BASE_MEM_GIB = 60.0
CARD_GIB = 80.0

FAULT_STEP = 200                 # expert faults start here
COLLAPSE_STEPS = 50
ZERO_STEP = FAULT_STEP + COLLAPSE_STEPS - 1        # 249: the first step at exactly 0.0 in the collapse
TUNED_PATIENCE = 10
DEFAULT_PATIENCE = AlarmMonitor().moe_patience     # 1,000

MEM_START, MEM_CLIMB = 300, 200
OOM_STEP = MEM_START + MEM_CLIMB - 1               # 499: memory reaches 80 GiB, the step never finishes
JUMP_STEP = 205
MEM_BOUND_GIB = 76.0

SPEED_START = 300
SPEED_DROP = 0.30
FAST_RAMP, SLOW_RAMP = 150, 300
MFU_BOUND = 0.30
DEAD_BOUND = 0.5
MAXVIO_AS_IMBALANCE = 1.0 + AlarmMonitor().moe_max_vio_limit   # MaxVio > 1.0 is the same line as imbalance > 2.0

TUNED = dict(moe_patience=TUNED_PATIENCE, bounds={
    "gpu_memory_used": (None, MEM_BOUND_GIB * GIB),
    "model_flops_utilization": (MFU_BOUND, None),
    "dead_experts_count": (None, DEAD_BOUND),
})
MONITORS = {"A": {}, "B": TUNED}


# Fault shapes: each maps a step to a factor (expert 0's weight, speed) or a value (memory in GiB).

def sudden_zero(step):
    return 0.0 if step >= FAULT_STEP else 1.0


def collapse(step):
    return 1.0 if step < FAULT_STEP else max(0.0, 1.0 - (step - FAULT_STEP + 1) / COLLAPSE_STEPS)


def mem_climb(step):
    if step < MEM_START:
        return BASE_MEM_GIB
    return BASE_MEM_GIB + (CARD_GIB - BASE_MEM_GIB) * min(1.0, (step - MEM_START + 1) / MEM_CLIMB)


def mem_jump(step):
    return CARD_GIB if step >= JUMP_STEP else BASE_MEM_GIB


def speed_ramp(ramp_steps):
    def factor(step):
        if step < SPEED_START:
            return 1.0
        return 1.0 - SPEED_DROP * min(1.0, (step - SPEED_START + 1) / ramp_steps)
    return factor


def make_rows(steps, n_experts=8, expert0=None, mem_gib=None, speed=None, seed=SEED):
    """One dict of trainer metrics per step. The random draws do not depend on the fault, so the healthy
    part of every stream is the same."""
    rng = random.Random(seed)
    rows = []
    for step in range(steps):
        w = [1.0 + rng.gauss(0.0, 0.05) for _ in range(n_experts)]
        if expert0 is not None:
            w[0] *= expert0(step)
        total = sum(w)
        tps = BASE_TPS * (speed(step) if speed else 1.0) * (1.0 + rng.gauss(0.0, 0.01))
        mem = mem_gib(step) if mem_gib else BASE_MEM_GIB
        rows.append({
            "loss_ntp": 2.2 + 1.5 * math.exp(-step / 60) + rng.gauss(0.0, 0.01),
            "lr": 3e-4,
            "tokens_per_second": tps,
            "model_flops_utilization": BASE_MFU * tps / BASE_TPS,   # MFU = 6 x params x tokens/s / peak
            "mem_used_mib": round(mem * 1024 + rng.gauss(0.0, 20.0)),  # nvidia-smi MiB; the writer stores bytes
            "expert_load": [x / total for x in w],
        })
    return rows


class JsonlTail:
    """Read complete new lines from metrics.jsonl and run them through AlarmMonitor like `alarms.py --jsonl`."""

    def __init__(self, path, **monitor_kw):
        self.path = Path(path)
        self.offset = 0
        self.monitor = AlarmMonitor(**monitor_kw)
        self.alarms = []

    def poll(self, now_step, write_t):
        with open(self.path, "rb") as f:    # binary: the offset is a byte position
            f.seek(self.offset)
            data = f.read()
        end = data.rfind(b"\n") + 1         # only whole lines; a partial line waits for the next poll
        self.offset += end
        for line in data[:end].decode("utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            step = row.pop("step")
            for a in self.monitor.step(step, **row):
                ms = (time.perf_counter() - write_t[a["step"]]) * 1e3
                self.alarms.append(dict(a, seen_at=now_step, flush_to_alert_ms=ms))


def run_stream(path, rows, crash_at=None):
    """Write rows[0..] through MetricsWriter; both monitors tail the file. With crash_at, the process dies
    before step crash_at: no close(), so the rows after the last flush never reach the file."""
    write_t = {}
    writer = MetricsWriter(str(path), flush_every=FLUSH_EVERY)
    tails = {name: JsonlTail(path, **kw) for name, kw in MONITORS.items()}
    last = len(rows) if crash_at is None else crash_at
    for step in range(last):
        write_t[step] = time.perf_counter()
        writer.write(step, **rows[step])
        for t in tails.values():
            t.poll(step, write_t)
    if crash_at is None:
        writer.close()
    else:
        writer._file.close()                # the file handle goes with the process; the buffer is lost
    for t in tails.values():
        t.poll(last - 1, write_t)
    return {name: t.alarms for name, t in tails.items()}


class MockRun:
    def __init__(self, path, alarms):
        self.path = path
        self.alarms = alarms
        self.rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    def fired(self, monitor, rule=None, metric=None):
        return [a for a in self.alarms[monitor]
                if (rule is None or a["rule"] == rule) and (metric is None or a["metric"] == metric)]

    def steps(self, monitor):
        return sorted((a["step"], a["rule"], a["metric"]) for a in self.alarms[monitor])

    def first_row(self, test):
        return next((r["step"] for r in self.rows if test(r)), None)


@pytest.fixture
def mock_metrics_jsonl(tmp_path):
    """Factory: write one mock metrics.jsonl for the given faults and return a MockRun."""
    count = [0]

    def make(steps=STEPS, crash_at=None, **faults):
        count[0] += 1
        path = tmp_path / f"mock_metrics_{count[0]}.jsonl"
        return MockRun(path, run_stream(path, make_rows(steps, **faults), crash_at=crash_at))
    return make


def next_flush(step):
    return -(-step // FLUSH_EVERY) * FLUSH_EVERY


# 1. Control.

def test_clean_stream_has_no_alarms(mock_metrics_jsonl):
    run = mock_metrics_jsonl()
    assert run.alarms == {"A": [], "B": []}
    assert all(r["dead_experts_count"] == 0 for r in run.rows)


# 2. Dead experts.

@pytest.mark.parametrize("n_experts", [8, 16, 64])
def test_expert_drops_to_zero(mock_metrics_jsonl, n_experts):
    run = mock_metrics_jsonl(n_experts=n_experts, expert0=sudden_zero)
    assert run.first_row(lambda r: r["dead_experts_count"] > 0) == FAULT_STEP
    assert run.steps("B") == [(FAULT_STEP, "static_bound", "dead_experts_count"),
                              (FAULT_STEP + TUNED_PATIENCE - 1, "moe_routing", "expert_load")]
    assert run.steps("A") == [(FAULT_STEP + DEFAULT_PATIENCE - 1, "moe_routing", "expert_load")]
    (moe,) = run.fired("B", "moe_routing")
    assert moe["severity"] == "warn" and moe["dead_experts"] == 1


@pytest.mark.parametrize("n_experts", [8, 16, 64])
def test_expert_collapses_over_50_steps(mock_metrics_jsonl, n_experts):
    run = mock_metrics_jsonl(n_experts=n_experts, expert0=collapse)
    writer_dead = run.first_row(lambda r: r["dead_experts_count"] > 0)
    assert FAULT_STEP < writer_dead < ZERO_STEP        # below 10% of a fair share, before the exact zero
    assert run.steps("B") == [(writer_dead, "static_bound", "dead_experts_count"),
                              (ZERO_STEP + TUNED_PATIENCE - 1, "moe_routing", "expert_load")]
    assert run.steps("A") == [(ZERO_STEP + DEFAULT_PATIENCE - 1, "moe_routing", "expert_load")]


def test_finding_moe_routing_waits_for_exact_zero(mock_metrics_jsonl):
    """The writer calls expert 0 dead several steps before moe_routing starts counting its patience."""
    run = mock_metrics_jsonl(expert0=collapse)
    writer_dead = run.first_row(lambda r: r["dead_experts_count"] > 0)
    rows = {r["step"]: r for r in run.rows}
    share = rows[writer_dead]["expert_load"][0]
    assert 0.0 < share < DEAD_EXPERT_SHARE / 8
    assert ZERO_STEP - writer_dead >= 3


def test_finding_imbalance_cannot_see_one_dead_expert(mock_metrics_jsonl):
    """With one dead expert of 8 the busiest expert gets about 8/7 of a fair share, far below the 2.0 line."""
    run = mock_metrics_jsonl(expert0=sudden_zero)
    dead_rows = [r for r in run.rows if r["step"] >= FAULT_STEP]
    assert all(r["dead_experts_count"] == 1 for r in dead_rows)
    assert max(r["expert_load_imbalance"] for r in dead_rows) < 1.5 < MAXVIO_AS_IMBALANCE


# 3. GPU memory.

def test_memory_climb_alarms_before_oom(mock_metrics_jsonl):
    run = mock_metrics_jsonl(steps=OOM_STEP + 1, mem_gib=mem_climb, crash_at=OOM_STEP)
    cross = run.first_row(lambda r: r["gpu_memory_used"] > MEM_BOUND_GIB * GIB)
    assert run.steps("B") == [(cross, "static_bound", "gpu_memory_used")]
    (mem,) = run.fired("B")
    assert mem["seen_at"] == next_flush(cross)
    assert OOM_STEP - mem["seen_at"] >= 30            # the warning reaches the file 30+ steps before the OOM
    assert run.alarms["A"] == []                       # the CLI has no bounds: no memory alarm at all
    assert run.rows[-1]["step"] == (OOM_STEP - 1) // FLUSH_EVERY * FLUSH_EVERY   # rows 491-498 died with the process


def test_finding_sudden_oom_never_reaches_the_file(mock_metrics_jsonl):
    crashed = mock_metrics_jsonl(steps=JUMP_STEP + 1, mem_gib=mem_jump, crash_at=JUMP_STEP + 1)
    assert crashed.rows[-1]["step"] == JUMP_STEP // FLUSH_EVERY * FLUSH_EVERY     # 200: row 205 was in the buffer
    assert crashed.alarms == {"A": [], "B": []}
    flushed = mock_metrics_jsonl(steps=JUMP_STEP + 1, mem_gib=mem_jump)          # same rows, clean close()
    assert flushed.steps("B") == [(JUMP_STEP, "static_bound", "gpu_memory_used")]


def test_gpu_memory_used_is_bytes_from_mib(mock_metrics_jsonl):
    run = mock_metrics_jsonl(steps=20)
    assert all(abs(r["gpu_memory_used"] / GIB - BASE_MEM_GIB) < 0.1 for r in run.rows)
    assert all("mem_used_mib" not in r for r in run.rows)


# 4. Speed and MFU.

def test_speed_and_mfu_fall_30pct_over_150_steps(mock_metrics_jsonl):
    run = mock_metrics_jsonl(steps=700, speed=speed_ramp(FAST_RAMP))
    mfu_low = run.first_row(lambda r: r["model_flops_utilization"] < MFU_BOUND)
    (tps,) = run.fired("A")
    assert tps["rule"] == "throughput_drop" and tps["severity"] == "warn"
    assert [a["step"] for a in run.fired("B", "throughput_drop")] == [tps["step"]]
    mfu = run.fired("B", metric="model_flops_utilization")
    assert mfu[0]["step"] == mfu_low and mfu[0]["severity"] == "warn"
    assert SPEED_START < mfu_low < tps["step"]
    assert tps["step"] > SPEED_START + FAST_RAMP       # throughput_drop fires only after the ramp has ended


def test_finding_mfu_bound_flaps_near_the_line(mock_metrics_jsonl):
    """MFU noise crosses 0.30 several times during the ramp; static_bound re-arms on each return and fires again."""
    run = mock_metrics_jsonl(steps=700, speed=speed_ramp(FAST_RAMP))
    mfu = [a["step"] for a in run.fired("B", metric="model_flops_utilization")]
    assert len(mfu) > 1
    assert all(SPEED_START < s < SPEED_START + FAST_RAMP for s in mfu)   # once at 70% speed it stays below


def test_finding_slow_decline_never_fires_throughput_drop(mock_metrics_jsonl):
    """Steps within 10% of the trailing mean join that mean, so a slow decline drags the baseline down with it."""
    run = mock_metrics_jsonl(speed=speed_ramp(SLOW_RAMP))
    assert run.rows[-1]["tokens_per_second"] < 0.75 * BASE_TPS          # 30% slower for the last 700 steps
    assert run.fired("A") == [] and run.fired("B", "throughput_drop") == []
    assert {a["metric"] for a in run.fired("B")} == {"model_flops_utilization"}   # only the absolute MFU line sees it


# 5. The file path and the CLI.

def test_flush_lag_and_timings(mock_metrics_jsonl):
    run = mock_metrics_jsonl(expert0=collapse)
    for monitor in ("A", "B"):
        for a in run.alarms[monitor]:
            assert a["seen_at"] == next_flush(a["step"])
            assert a["flush_to_alert_ms"] >= 0


def _cli_steps(path, extra=()):
    out = subprocess.run([sys.executable, str(HERE / "alarms.py"), "--jsonl", str(path), *extra],
                         capture_output=True, text=True, check=True)
    return sorted((a["step"], a["rule"], a["metric"]) for a in map(json.loads, out.stdout.splitlines()))


def test_cli_matches_monitor_a(mock_metrics_jsonl):
    run = mock_metrics_jsonl(expert0=collapse)
    assert _cli_steps(run.path) == run.steps("A")


# Issue #33: the same files, with monitor B passed as flags. gpu_memory_used in the file is bytes.
B_CLI = (
    "--moe-patience", str(TUNED_PATIENCE),
    "--bound", f"dead_experts_count::{DEAD_BOUND}",
    "--bound", f"model_flops_utilization:{MFU_BOUND}:",
    "--bound", f"gpu_memory_used::{MEM_BOUND_GIB * GIB:.0f}",
)


def test_cli_matches_monitor_b(mock_metrics_jsonl):
    cases = (
        dict(expert0=sudden_zero),
        dict(steps=OOM_STEP + 1, mem_gib=mem_climb, crash_at=OOM_STEP),
        dict(steps=700, speed=speed_ramp(FAST_RAMP)),
    )
    for faults in cases:
        run = mock_metrics_jsonl(**faults)
        assert _cli_steps(run.path, B_CLI) == run.steps("B")


# Table rows for docs/alert_tests.md (monitor B unless the mode says A).

def _scenarios():
    tuned = "`moe_patience=10`"
    # Sudden dead expert, moe_routing with patience 10: fires 9 steps after the zero.
    yield ("moe_routing", "B", f"expert 0 of 8 to 0.0 at once, {tuned}", FAULT_STEP,
           dict(expert0=sudden_zero), None, "moe_routing")
    # Sudden dead expert, bound on the writer's dead_experts_count: fires on the fault step.
    yield ("static_bound", "B", "expert 0 of 8 to 0.0 at once, `bounds={\"dead_experts_count\": (None, 0.5)}`",
           FAULT_STEP, dict(expert0=sudden_zero), None, "dead_experts_count")
    # 50-step collapse, moe_routing with patience 10: starts counting only at the exact zero (249).
    yield ("moe_routing", "B", f"expert 0 of 8 collapses to 0.0 over steps 200-249, {tuned}", FAULT_STEP,
           dict(expert0=collapse), None, "moe_routing")
    # 50-step collapse, CLI defaults: moe_routing waits its 1,000-step patience.
    yield ("moe_routing", "A", "expert 0 of 8 collapses to 0.0 over steps 200-249, default `AlarmMonitor()`",
           FAULT_STEP, dict(expert0=collapse), None, "moe_routing")
    # 50-step collapse, dead_experts_count bound: fires when the share drops below 10% of fair, before the zero.
    yield ("static_bound", "B", "expert 0 of 8 collapses over steps 200-249, `dead_experts_count` hi 0.5",
           FAULT_STEP, dict(expert0=collapse), None, "dead_experts_count")
    # Memory leak climbing to the card's 80 GiB: the bound fires at 76 GiB, before the OOM crash.
    yield ("static_bound", "B", "`gpu_memory_used` 60 to 80 GiB over steps 300-499, OOM at 499, hi 76 GiB",
           MEM_START, dict(steps=OOM_STEP + 1, mem_gib=mem_climb), OOM_STEP, "gpu_memory_used")
    # Sudden jump to 80 GiB, then a crash: the row dies in the writer's buffer, so nothing fires.
    yield ("static_bound", "B", "`gpu_memory_used` jumps to 80 GiB at step 205, process dies at 206, hi 76 GiB",
           JUMP_STEP, dict(steps=JUMP_STEP + 1, mem_gib=mem_jump), JUMP_STEP + 1, "gpu_memory_used")
    # 30% slowdown over 150 steps, absolute MFU line at 0.30: fires the first step MFU dips below it.
    yield ("static_bound", "B", "speed and MFU fall 30% over steps 300-449, `model_flops_utilization` lo 0.30",
           SPEED_START, dict(steps=700, speed=speed_ramp(FAST_RAMP)), None, "model_flops_utilization")
    # 30% slowdown over 150 steps, throughput_drop at defaults: fires after 100 slow steps in a row.
    yield ("throughput_drop", "A", "speed and MFU fall 30% over steps 300-449, default `tps_patience=100`",
           SPEED_START, dict(steps=700, speed=speed_ramp(FAST_RAMP)), None, "tokens_per_second")
    # 30% slowdown over 300 steps: the trailing mean follows it down, so throughput_drop never fires.
    yield ("throughput_drop", "A", "speed and MFU fall 30% over steps 300-599, default `tps_patience=100`",
           SPEED_START, dict(speed=speed_ramp(SLOW_RAMP)), None, "tokens_per_second")


def table_rows(tmp_dir):
    lines = []
    for i, (rule, monitor, mode, fault, faults, crash_at, key) in enumerate(_scenarios()):
        path = Path(tmp_dir) / f"row_{i}.jsonl"
        steps = faults.pop("steps", STEPS)
        alarms = run_stream(path, make_rows(steps, **faults), crash_at=crash_at)[monitor]
        hit = [a for a in alarms if a["rule"] == rule and key in (a["rule"], a["metric"])]
        if hit:
            a = hit[0]
            cells = (f"n/a (jsonl: {a['seen_at']}) | n/a (jsonl: {a['seen_at'] - fault}) | n/a | "
                     f"{a['flush_to_alert_ms']:.3f} | PASS")
        else:
            cells = "never | never | n/a | n/a | MISS (finding)"
        lines.append(f"| @nikshrimali | `{rule}` (warn) | mock `metrics.jsonl` via `MetricsWriter`, monitor {monitor}: "
                     f"{mode} | {fault} | {cells} |")
    return lines


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        rows = table_rows(d)
    print("| rotator | rule_1 & rule_2 | injection_mode | fault_step | alert_step | delta_steps | in_harness_ms | jsonl_flush_to_alert_ms | verdict |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))
