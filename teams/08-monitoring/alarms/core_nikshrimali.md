# Issue #26: @nikshrimali (G8 core), dead experts, GPU memory and speed / MFU

Issue #26 was a week 3 rotation task. It moved to the core team on 24 Nov, so this file is `core_nikshrimali.md`, not
`rotation_<handle>.md`. The layout follows the rotation files.

| Task | Status | Where |
|---|---|---|
| R1. Test (1) `dead_experts_count` / `expert_load_imbalance` and (2) `gpu_memory_used` OOM / MFU drop with an injected fault | Done | [`test_alert_injection_nikshrimali.py`](test_alert_injection_nikshrimali.py) (17 pytest tests) |
| R2. Record the time to alert | Done | below, and the shared table in [`../docs/alert_tests.md`](../docs/alert_tests.md) |

## Setup

A pytest fixture writes a mock `metrics.jsonl`. It is a seeded synthetic stream (seed 26) that goes through the real
`MetricsWriter` (`flush_every = 10`), so the file carries the fields the writer derives (`dead_experts_count`,
`expert_load_imbalance`, and `gpu_memory_used` in bytes from `mem_used_mib`) as a trainer's file would. Healthy steps
have `tokens_per_second` 50,000 ±1%, MFU 0.38 scaled with the speed, 60 GiB of GPU memory, and n expert shares near 1/n.

Two monitors read the file line by line after every write, the way `alarms.py --jsonl` does:

- **A** is `AlarmMonitor()`, what `alarms.py --jsonl` runs: no `bounds`, `moe_patience=1000`, `tps_patience=100`.
- **B** is tuned: `moe_patience=10`, plus `static_bound` on `gpu_memory_used` (hi 76 GiB), `model_flops_utilization`
  (lo 0.30) and `dead_experts_count` (hi 0.5).

Only the file path is measured, so the in-harness columns are "n/a". Time to alert runs from `MetricsWriter.write()`
of the alert step's row until the file monitor fires for it. A test checks that `alarms.py --jsonl` gives the same
alarms as monitor A.

## R1. Faults and results

The expert faults hit expert 0. The expert count is a test parameter, and every expert test passes at 8, 16 and 64
experts with the same alarm steps.

| Fault | Monitor B fires at | Monitor A fires at | Verdict |
|---|---|---|---|
| none (clean 1,300 steps) | nothing | nothing | PASS |
| expert 0 gets no tokens from step 200 | `static_bound` `dead_experts_count` 200, `moe_routing` 209 | `moe_routing` 1199 | PASS |
| expert 0 falls from a fair share to 0.0 over steps 200-249 | `static_bound` `dead_experts_count` 245, `moe_routing` 258 | `moe_routing` 1248 | PASS |
| memory 60 to 80 GiB over steps 300-499, OOM at 499 | `static_bound` `gpu_memory_used` 459 | nothing | PASS on B |
| memory jumps to 80 GiB at step 205, the process dies at 206 | nothing | nothing | MISS (finding 4) |
| speed and MFU fall 30% over steps 300-449 | `static_bound` MFU 404 (again 406, 411), `throughput_drop` 476 | `throughput_drop` 476 | PASS |
| speed and MFU fall 30% over steps 300-599 | `static_bound` MFU 497 (7 times to 523) | nothing | MISS for `throughput_drop` (finding 5) |

## R2. Time to alert (file path)

`fault_step` is where the fault starts. `seen at` is the step at which the row reached the file and the monitor fired.

| Alert | Monitor | Fault step | Seen at | `delta_steps` | `jsonl_flush_to_alert_ms` |
|---|---|---|---|---|---|
| `static_bound` `dead_experts_count`, sudden zero | B | 200 | 200 | 0 | 0.439 |
| `moe_routing`, sudden zero | B | 200 | 210 | 10 | 0.476 |
| `static_bound` `dead_experts_count`, collapse | B | 200 | 250 | 50 | 0.565 |
| `moe_routing`, collapse | B | 200 | 260 | 60 | 0.479 |
| `moe_routing`, collapse | A | 200 | 1250 | 1050 | 0.480 |
| `static_bound` `gpu_memory_used`, climb | B | 300 | 460 | 160 | 0.581 |
| `static_bound` MFU, 150-step ramp | B | 300 | 410 | 110 | 0.631 |
| `throughput_drop`, 150-step ramp | A | 300 | 480 | 180 | 0.370 |

One run on a laptop CPU (Apple Silicon). The milliseconds change by a few hundredths between runs. They are small
because a synthetic step takes microseconds. In a real run, the time to alert is about `delta_steps` x step time.

## Findings

1. **Dry run 1 wrote none of these metrics.** The dry run 1 file (`dryrun1-live3`, 1,222 rows) has no `expert_load`,
   `dead_experts_count`, `expert_load_imbalance`, `gpu_memory_used` or `model_flops_utilization`. Its only GPU memory
   field is `gpu_alloc_gb`. So none of the alarms tested here could have fired in that run, whatever the rules say.
   For dry run 2, the trainer needs to pass `expert_load`, `mem_used_mib` (or `gpu_memory_used`) and
   `model_flops_utilization` to `MetricsWriter.write()`.
2. **The CLI can't alarm on memory or MFU.** `alarms.py --jsonl` builds `AlarmMonitor()` with no `bounds`, so
   `static_bound` is off. Monitor A fired nothing for the memory climb or the MFU drop. A dead expert waits for
   `moe_patience`: 999 steps after the share hits zero (step 1248 for a collapse that ended at 249).
3. **The writer sees a collapsing expert before `moe_routing` does.** `MetricsWriter` counts an expert as dead below
   10% of a fair share, at step 245 (244 with 16 experts). `moe_routing` counts only an exact 0.0, at step 249, and
   then waits its patience. A `static_bound` on `dead_experts_count` fires at 245, 13 steps before `moe_routing` with
   patience 10. `expert_load_imbalance` can't see one dead expert. With one of 8 at zero, the busiest expert stays
   below 1.35 x the mean, far under the 2.0 line that `moe_routing`'s MaxVio > 1.0 draws.
4. **A sudden OOM leaves no row.** When memory jumps to 80 GiB at step 205 and the process dies at 206, rows 201-205
   are still in the writer's buffer and never reach the file, which ends at step 200. Nothing fires. The same rows with
   a clean `close()` fire `static_bound` at 205, so the rule works and the crash is the gap. Only a slow climb can be
   caught: in the climb test, the alarm reaches the file at step 460, 39 steps before the OOM, and the crash loses rows
   491-498. Also, `metrics.jsonl` has one row per step and no rank field, so it holds only the writing process's
   memory. An OOM on another rank never shows in this file.
5. **A slow slowdown never fires `throughput_drop`.** A 30% drop over 150 steps fires at step 476. The same 30% drop
   over 300 steps never fires, even after 700 more steps at 30% below the start (34,730 tok/s against 50,000). A step
   within 10% of the trailing mean is added to that mean, so a slow decline drags the baseline down with it. An
   absolute MFU line (`static_bound` lo 0.30) catches both ramps.
6. **`static_bound` flaps on a noisy metric near its line.** MFU noise (±1%) crosses 0.30 several times during a
   ramp, and the latch re-arms on every return. It fired 3 times on the 150-step ramp and 7 times on the 300-step
   ramp. This is the re-arm behaviour @swatibansal recorded, seen here on real-looking noise.

## Possible follow-ups

These are not part of this PR, which changes no rule.

- Pass default `bounds` for `gpu_memory_used` and `model_flops_utilization` in `alarms.py --jsonl`, or add flags for them.
- Let `moe_routing` read `dead_experts_count`, so it uses the writer's 10% definition.
- Compare `throughput_drop` against a fixed baseline (for example the first N steps after warm-up) as well as the
  trailing mean.
- Add hysteresis to `static_bound`: re-arm only after the value is back past the line by a margin.

## Reproduce

```bash
python -m pytest teams/08-monitoring/alarms/test_alert_injection_nikshrimali.py -q
python teams/08-monitoring/alarms/test_alert_injection_nikshrimali.py     # prints the table rows
```
