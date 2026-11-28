# Dry run 2: eval-watcher lag alarm, synthetic real-pace replay (Issue #32)

**This file holds no dry run 2 data.** TSAI reported `dryrun2-1234` COMPLETED on Fri 27 Nov 2026 (1,221 of 1,221 steps, 79.8 min, `#705`), but its `metrics.jsonl`, the eval watcher's `eval_mini` rows and the output of the `#833` monitoring job had not been returned when this file was written. So the run below is made up: dry run 1's real log, stretched to dry run 2's reported wall time, with fake Tier A scores. It shows that `eval_lag.py` (`#769`, `#822`) and `watch_jsonl.py` (`#671`) as they are on `main` behave as the `#833` job request expects at dry run 2's pace, cadence and kill drill. It says nothing about how `dryrun2-1234` itself went. Section 6 says how to rerun it on the real logs.

## 1. What was replayed

| Item | Value |
|---|---|
| Code | `main` at `57bdc7b4`: `clickhouse/replay_jsonl.py`, `clickhouse/watch_jsonl.py`, `alarms/eval_lag.py`, `clickhouse/replay_latency.py`, all unchanged |
| ClickHouse | local Docker, 24.8.14.39 |
| When | Fri 27 Nov 2026, 13:39:09 to 15:41:15 UTC, real pace (`--speed 1`), on a laptop under `caffeinate` |
| Training log | `teams/05-architecture/jobs/evidence/2026-11-15-dryrun1-full/metrics.jsonl` (`#620`), steps `0..1221`, with every `perf/step_ms` multiplied by `2.049` (and `tokens_per_second` divided by it), so the run trains for `4,788 s` (79.8 min). Median step `3.89 s`. Step 0 (no `perf/step_ms`) takes `3.92 s`. |
| Commits | the trainer commits step 0, then the first step past each 600 s of training, then the last step: steps `0, 152, 305, 455, 608, 763, 917, 1072, 1221` |
| Tier A scores | MADE UP. A stand-in for Team 9's watcher loads one `eval_mini` row per commit (run_id `<run>:step-NNNNNNNN:tier-a:synth0000000`, `checkpoint_run` = the run, `checkpoint_step` = the commit) once the commit step is in the file, after `U(0, 60) s` poll wait + `207 s` Tier A, plus `159 s` Tier B on every 10th commit (`teams/09-evaluation/eval_schedule.md` section 3). No run reaches a 10th commit, so no score waited for Tier B. No Tier B or Tier C rows are loaded. |
| Alarm | `eval_lag.py --interval-s 600 --poll 60` (limit `1,200 s`, `--live-s 1,200`, `--pause-s 300`, label `tier-a`), one process for all runs, as in `#833` step 3 |

Three runs ran side by side, each with its own `replay_jsonl.py` and `watch_jsonl.py --poll 2 --idle-exit 600`:

| Run | What happens | Expected |
|---|---|---|
| `dryrun2-synth` | healthy: every commit scored | 0 pages |
| `dryrun2-synth-stall` | the Tier A watcher dies after commit 3 (step 455, about 1,800 s in) | exactly 1 page |
| `dryrun2-synth-kill` + `dryrun2-synth-kill-resume-a` | SIGKILL at step 690 (2,700 s), 120 s down, restore from the commit at step 608 into a new folder, which trains steps `609..1221` and commits every 600 s of its own training (steps `763, 918, 1072, 1221`). Mapped with `--run dryrun2-synth-kill-resume-a=dryrun2-synth-kill`, as `#833` maps `-resume-a`. | 0 pages |

## 2. Result

| Run | Checks | Live checks | Max lag | Limit | Over-limit checks | Pages | Last training step | Last Tier A step | Verdict |
|---|---|---|---|---|---|---|---|---|---|
| `dryrun2-synth` | 122 | 99 | `823.3 s` | `1,200 s` | 0 | **0** | 1221 | 1221 | **PASS** |
| `dryrun2-synth-stall` | 122 | 99 | `2,969 s` | `1,200 s` | 49 | **1** | 1221 | 455 | **PASS** |
| `dryrun2-synth-kill` | 122 | 106 | `851.0 s` | `1,200 s` | 0 | **0** | 1221 | 1221 | **PASS** |

- Every check wrote a row to `eval_lag_checks`, one per run each minute from 13:40:09 to 15:41:14 UTC. The status log has 0 errors.
- `./out/monitoring/eval_lag_pages.jsonl` holds one line, the stall page (section 3.2).
- After training ended, each run turned `live = 0` one `--live-s` (1,200 s) after its last batch: `dryrun2-synth` and `-stall` at 15:19:13, `-kill` at 15:26:14 UTC.

## 3. Per run

### 3.1 Healthy

The peak, `823.3 s` at 13:53:10 UTC, comes just before commit 1 (step 152) is scored at 13:53:25. That is the 600 s until commit 1, plus the stand-in's delay for it, measured from the run's first batch. The headroom to the limit is `377 s`, so one score would have to land more than about 600 s late to page, as `eval_lag.py`'s docstring says.

### 3.2 Watcher stall

The last Tier A score is for step 455, loaded at 14:13:18 UTC. The batch that brought step 455 was read at 14:09:28.749, so the lag passed 1,200 s at about 14:29:29. The next check, at 14:30:11.582, paged:

```json
{"step": 770, "metric": "eval_mini", "rule": "eval_lag", "severity": "page", "run_id": "dryrun2-synth-stall", "train_runs": ["dryrun2-synth-stall"], "eval_step": 455, "lag_s": 1210.9, "threshold_s": 1200.0, "eval_loaded_at": "2026-11-27 14:13:18.604", "detail": "tier-a for dryrun2-synth-stall is 1,211 s of training behind (limit 1,200 s): newest scored step 455, newest training step 770. Is the tier-a eval watcher running?"}
```

The lag stayed over the limit for 49 checks, up to 15:18:13, with no second page. It went quiet when the run stopped training (`live = 0` at 15:19:13).

### 3.3 SIGKILL and restore

| Checked at (UTC) | Training runs | Training step | Tier A step | Lag |
|---|---|---|---|---|
| 14:23:11 | `-kill` | 670 | 455 | `821.2 s` |
| 14:24:11 | `-kill` | 680 | 608 | `269.1 s` |
| 14:24:27 | SIGKILL after step 690 | | | |
| 14:25:11 | `-kill` | 690 | 608 | `307.2 s` |
| 14:26:11 | `-kill` | 690 | 608 | `307.2 s` |
| 14:26:35 | `-resume-a` first batch, steps `609..610` | | | |
| 14:27:11 | `-kill`, `-resume-a` | 610 | 608 | `0 s` |
| 14:41:12 | `-kill`, `-resume-a` | 830 | 608 | `851.0 s` (peak) |
| 14:41:16 | Tier A for step 763 (the resumed trainer's first commit) loaded | | | |

The resumed batch starts at step 609, which is at or below the last step before it (690). So `eval_lag.py` treats it as a restore to step 608: it drops steps `609..690` of the killed trainer (`307 s` of training) and the `127 s` of downtime. By the wall clock, the newest batch at the peak check (14:40:46) came `1,285 s` after the batch that brought step 608 (14:19:21), so a lag measured by the wall clock would have paged during the drill. The alarm measured `1,285 - 307 - 127 = 851 s` and did not page.

## 4. Ingestion

`replay_latency.py` on each replay log (`missing` and `duplicate` are (step, name, rank) pairs in `metrics_raw`):

| Run | Steps | Rows | Missing / duplicate | Pauses | File write -> ClickHouse (p50 / p95 / max) | Step finished -> ClickHouse (p50 / p95 / max) |
|---|---|---|---|---|---|---|
| `dryrun2-synth` | 1,222 (`0..1221`) | 19,753 | 0 / 0 | 0 | `1.02 / 1.92 / 2.01 s` | `18.62 / 36.11 / 39.00 s` |
| `dryrun2-synth-stall` | 1,222 (`0..1221`) | 19,753 | 0 / 0 | 0 | `1.03 / 1.90 / 2.00 s` | `18.59 / 36.10 / 38.93 s` |
| `dryrun2-synth-kill` | 691 (`0..690`) | 11,187 | 0 / 0 | 0 | `0.91 / 1.73 / 2.01 s` | `18.60 / 36.14 / 38.95 s` |
| `dryrun2-synth-kill-resume-a` | 613 (`609..1221`) | 9,948 | 0 / 0 | 0 | `0.89 / 1.94 / 2.03 s` | `18.29 / 35.81 / 38.95 s` |

The worst case to Grafana is `39.00 s + 10 s` refresh = `49.0 s`, under the `< 60 s` target. It is not within the "20 s of each step" that `#833` item 1 quotes from dry run 1. That number came from a 1.9 s step, and at about 3.9 s per step the writer's 10-step buffer alone holds the first line of a flush for about 35 s. The loader itself (file write -> ClickHouse) stays near the 2 s poll.

## 5. Limits

- The script that stretched the log, the Tier A stand-in and the script that ran the three runs are local and not in the repo. Everything they drove is on `main`.
- Made-up data. The step times are dry run 1's, scaled. The real run may have a different step-time spread, a different first-step delay, or checkpoint stalls that dry run 1 lacks.
- The Tier A stand-in uses `eval_schedule.md`'s timings, not Team 9's watcher. A real watcher that scores slower than about 600 s past a commit would page, as designed.
- `eval_lag.py` reads `ingest_batches.seen_at`, the time `watch_jsonl.py` read the lines. A replay from returned logs therefore measures the replay's clock, not the job host's.
- `replay_latency.py` indexes step times by step number, not by line, so it fails with `IndexError` on a resumed segment that does not start at step 0. The resume row above passes the full log as the source, whose steps `609..1221` are the same lines.

## 6. Rerun on the real logs

When TSAI returns `./out/dryrun2-1234/metrics.jsonl` (and any `-resume-*` folders) and the Tier A `eval_mini` rows: load the `eval_mini` rows into the same ClickHouse with their real `loaded_at`, replay each `metrics.jsonl` with `replay_jsonl.py --speed 1` under `watch_jsonl.py --run-id dryrun2-1234` (and `--run-id dryrun2-1234-resume-a` for a resume), then run `eval_lag.py` with the `#833` step 3 arguments. Since `eval_lag.py` reads `seen_at`, the eval rows have to be shifted onto the replay's clock: each one loads at its real delay after the commit it scores. If the `#833` job ran live on TSAI's host, its `eval_lag_checks` and `ingest_batches` rows replace this file's sections 2 to 4 without a replay.
