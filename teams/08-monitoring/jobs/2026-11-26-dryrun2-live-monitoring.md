# Dry run 2 live monitoring companion job: `watch_jsonl.py` + `eval_lag.py` (week 4)

REQUEST FOR TSAI REVIEW. No approval and no run result is claimed. Team 8 lead (`sandeepkunkunuru`) wrote this file. TSAI reviews it.

Team 5's dry run 2 training job (`#720`, `dryrun2-1234`) does not start Team 8's live ClickHouse tailer or eval-watcher lag alarm. This companion CPU-only job runs `watch_jsonl.py` (`#671`) and `eval_lag.py` (`#769`, `#822`) beside `dryrun2-1234` and Team 9's eval watcher (`#707`, `#747`) so that:

1. Training metrics stream live into ClickHouse (`metrics_raw` + `ingest_batches`) within 20 s of each step (Grafana `< 30 s` with `10s` refresh, verified on dry run 1 replay in `#703`).
2. `eval_lag.py` monitors Team 9's Tier A `eval_mini` rows against `ingest_batches.seen_at` at dry run 2's 600 s checkpoint interval (1,200 s lag limit), including across Team 7's planned SIGKILL/restore drill (`#738`, `#756`), and records every check in `eval_lag_checks` (Grafana private panels 29–31).
3. After the run completes, `load_run_events.py` (`#774`) loads Team 7's checkpoint/restart event logs and `alarms.py` output into `run_events` (Grafana private panels 32–33).

```text
Team:            08-monitoring
Owner:           Team 8 lead (sandeepkunkunuru). Lab rule 3: no person names in the lab repo.
Lead approval:   sandeepkunkunuru (Team 8 lead). TSAI approves execution and ClickHouse URL on the job host.
Question:        During dry run 2 (including Team 7's planned kill/restore drill), do live metrics reach ClickHouse with 0 missing steps and < 60 s latency to Grafana, and does eval_lag.py run cleanly with 0 false pages across restores while recording every check in eval_lag_checks?
Script:          teams/08-monitoring/clickhouse/watch_jsonl.py (#671), teams/08-monitoring/alarms/eval_lag.py (#769, #822), and teams/08-monitoring/clickhouse/load_run_events.py (#774) on main.
Command:         1) Stream dryrun2-1234 live into ClickHouse (start alongside Team 5's #720 trainer; exits cleanly 600 s after the trainer finishes or is killed):
                    python -I teams/08-monitoring/clickhouse/watch_jsonl.py ./out/dryrun2-1234/metrics.jsonl --run-id dryrun2-1234 --poll 2 --idle-exit 600 --url <ClickHouse URL>
                 2) On each planned kill-drill resume (#738 / #756, which writes to a new --out folder), start a watcher for the resumed metrics.jsonl:
                    python -I teams/08-monitoring/clickhouse/watch_jsonl.py ./out/dryrun2-1234-resume-a/metrics.jsonl --run-id dryrun2-1234-resume-a --poll 2 --idle-exit 600 --url <ClickHouse URL>
                    python -I teams/08-monitoring/clickhouse/watch_jsonl.py ./out/dryrun2-1234-resume-b/metrics.jsonl --run-id dryrun2-1234-resume-b --poll 2 --idle-exit 600 --url <ClickHouse URL>
                 3) Run the eval-watcher lag alarm continuously alongside watch_jsonl.py and Team 9's eval watcher (#707):
                    python -I teams/08-monitoring/alarms/eval_lag.py --url <ClickHouse URL> --interval-s 600 --run dryrun2-1234 --run dryrun2-1234-resume-a=dryrun2-1234 --run dryrun2-1234-resume-b=dryrun2-1234 --poll 60 > ./out/monitoring/eval_lag_pages.jsonl 2> ./out/monitoring/eval_lag_status.log
                 4) Post-run (optional, once dryrun2 finishes): load Team 7 restart/checkpoint events (if V5_EVENTS_DIR is set) and offline alarms.py --jsonl output into run_events:
                    python -I teams/08-monitoring/alarms/alarms.py --jsonl ./out/dryrun2-1234/metrics.jsonl > ./out/monitoring/dryrun2_alarms.jsonl
                    python -I teams/08-monitoring/clickhouse/load_run_events.py --run-id dryrun2-1234 --url <ClickHouse URL> --events <V5_EVENTS_DIR> --alarms ./out/monitoring/dryrun2_alarms.jsonl
Config:          --interval-s 600 (matches Team 5's --checkpoint-interval-s 600 in #720), default --intervals 2 (1,200 s lag limit), --label tier-a, --poll 60.
Inputs:          (1) ./out/dryrun2-1234/metrics.jsonl (and ./out/dryrun2-1234-resume-{a,b}/metrics.jsonl if kill-drill resumes write to separate --out folders), read-only on the shared run filesystem.
                 (2) Reachable ClickHouse HTTP endpoint (<ClickHouse URL>) shared with Team 9's eval watcher (#707) and Grafana.
                 (3) Optional: Team 7's V5_EVENTS_DIR directory for post-run load_run_events.py.
Hardware:        CPU only (0 GPUs, negligible RAM/CPU — pure Python standard library polling every 2 s / 60 s).
Packages:        Python 3.10+ standard library only (zero third-party packages required for watch_jsonl.py, eval_lag.py, and load_run_events.py).
Max time:        150 minutes (matches Team 9's #707 watcher window: Team 5's 90 min run + 60 min tail for final commits). Stop eval_lag.py after both Team 5 and Team 9's Tier A watcher exit.
Precision:       n/a (CPU monitoring sidecar)
Measures:        (1) Step completeness & duplicate check in ClickHouse metrics_raw / ingest_batches (0 missing steps, 0 duplicate pairs).
                 (2) Live ingestion latency (ingest_batches.seen_at vs replay/step wall time).
                 (3) Every 60 s check row in ClickHouse eval_lag_checks (lag_s, threshold_s, paged) and any JSONL page events in ./out/monitoring/eval_lag_pages.jsonl.
Expected result: 0 missing steps in ClickHouse; 0 false pages from eval_lag.py across Team 7's planned kill/restore drill; every Tier A commit from Team 9's watcher reflected in eval_lag_checks and Grafana panels 29–31. If Team 9's Tier A watcher ever stalls > 1,200 s of training time, exactly 1 page event emitted per episode.
Outputs:         ./out/monitoring/eval_lag_pages.jsonl (0 bytes if no pages, < 1 MB), ./out/monitoring/eval_lag_status.log (< 5 MB), and ClickHouse tables metrics_raw, ingest_batches, eval_lag_checks, run_events. Well under 10 MB on disk.
Test log:        (1) Dry run 1 full 1,222-step real-pace replay with watch_jsonl.py + replay_jsonl.py: teams/08-monitoring/alarms_dryrun1.md section 5 (#703) — 1,222 steps, 19,753 rows, 0 missing, 0 duplicates, ClickHouse p50 9.51 s / p95 18.14 s / max 19.81 s, Grafana worst-case 29.8 s < 60 s.
                 (2) eval_lag.py live ClickHouse 24.8 suite: 45+ tests in teams/08-monitoring/alarms/tests/test_eval_lag.py (#769, #822), including watch_jsonl.py outage backlog and post-flush restart coverage; offline Team 8 suite 497 passed, 3 skipped on main (51f832b6).
```

## Notes for TSAI

1. **Run IDs for kill-drill resumes:** If TSAI uses different `--out` folder names or `--run-id` values for the two kill-drill restarts in `#738`/`#756`, pass the exact same IDs to `watch_jsonl.py --run-id <RESUME_ID>` and `eval_lag.py --run <RESUME_ID>=dryrun2-1234`.
2. **Fallback if dry run 2 has already started or finished:** If this sidecar cannot run live beside `dryrun2-1234`, Team 8 will run a real-pace replay of the returned `metrics.jsonl` + `eval_mini` logs on local ClickHouse using `replay_jsonl.py` and record the evidence in `teams/08-monitoring/alarms_dryrun2.md` (Issue `#32`).
