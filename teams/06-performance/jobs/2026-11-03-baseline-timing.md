# G6 baseline timing (Gate 2 job request, v2)

```
Team:            06-performance
Owner:           Raj H (raj-harit-v2)
Lead approval:   raj-harit-v2
Question:        What are step_ms (median, p10..p90), tokens/s, peak memory and the Muon part with its spread for configs/base.yaml (and base.yaml with model.mtp=true) on one GPU with synthetic token IDs, and do two runs agree within 5%?
Script:          teams/06-performance/jobs/run_baseline_timing.py
Command:         python teams/06-performance/jobs/run_baseline_timing.py --out out --synthetic
Config:          teams/06-performance/configs/base.yaml
Inputs:          synthetic (data.source=synthetic: random IDs over the 131,072-token vocabulary); BrahmicTokenizer-131K tokenizer.json @ 7dfcf39 (sha256 01d968a7...), already in the node's ~/.cache/era-perf (the PR 239 synthetic run used it)
Hardware:        1 GPU
Max time:        60
Precision:       bf16
Measures:        per bench run: step_ms median / p10 / p90, tokens/s, peak_allocated_gb, opt:opt.muon and opt:opt.muon.ns median and p10..p90 spread, checkpoint save ms; repeat check (gated parts within 5% or 0.05 ms); correctness tests pass; commit SHA, clean tree
Expected result: repeat check PASS on a GPU that is not slowed down; step_ms a little above the smoke step_ms of the same run; correctness tests all pass
Outputs:         out/result.json, out/SUMMARY.md, out/logs/*.log, out/runs/*.json|md|jsonl (about 5 MB, all redacted)
Test log:        teams/06-performance/jobs/2026-11-03-baseline-timing.log (Colab, one GPU, the same command, --synthetic)
```

## What changed after the first run (TSAI comments on PR 239)

| TSAI comment | Change |
| :--- | :--- |
| First run exit 4: no shards; add `--synthetic` | Added. The command uses `--synthetic`; every run JSON must say `data_source: synthetic` or the job fails. Without the flag the job needs shards and stops with exit 11 (it never switches by itself). |
| Repeat check FAIL; the GPU slowed itself down | Before each bench the job waits (up to 3 min) for an idle GPU below 70 C. It records other GPU processes and slowdown reasons at idle. A FAIL lists every part outside 5% with its spread, and says whether the GPU reported a slowdown, so a lab-node problem and a code problem look different. |
| GPU model, host, MFU and HFU must not be published | The job redacts them itself: logs, run JSON, reports and live JSONL under `out/` have no GPU model, UUID, PCI ID, driver, host name, clocks, power, peak TFLOPS, MFU, HFU or absolute path. A final scan of `out/` fails the job (exit 16) if anything is left. Nothing has to be removed by hand. |
| `checkpoint` not measured (rule 3b) | The bench runs all `bench.modes` of base.yaml (clean, layer, trace, checkpoint, comm). |
| Remove the email from the PR description | Done. |

## Inputs

Synthetic token IDs need no shards and no network. The tokenizer file is still needed (vocabulary and KE2
table checks); the job looks for it in `~/.cache/era-perf/BrahmicTokenizer-131K/tokenizer.json`, or `--tokenizer`,
checks its sha256, and stops with exit 11 if it is missing (no download).

For timing, synthetic and FineWeb data do the same work per step (same B x T, same model); only the host-side
`data` part and the loss values differ. A FineWeb run is possible later with `--data-dir` (shards from
`python -m dataio.prepare_fineweb`) if TSAI can stage them.

## What the job runs (the stages of the G6 Colab baseline)

| stage | command (cwd teams/06-performance) | fails with |
| :--- | :--- | :--- |
| preflight | nvidia-smi, git status, package imports | 10 |
| inputs | tokenizer check (and shards without `--synthetic`) | 11 |
| tests | `python -m pytest correctness -q -rs` | 12 |
| smoke | `python -m perfkit.smoke` (weights finite, reversible OK) | 13 |
| bench_normal | `python -m perfkit.bench --level 3 --repeat 2` | 2 (5% check), 14 (crash, OOM, no data) |
| bench_mtp | the same with `model.mtp=true` | 2, 14 |
| tests_all | `python -m pytest correctness -q --all-configs` | 12 |

A stage that hits its time limit gives exit 15. The job runs the later stages after a failure, so one run
gives everything. `out/result.json` is written after every stage, so a job stopped at the time limit still
shows which stage was running. Exit 0 means every stage passed.

## Pins

| Item | Value |
| :--- | :--- |
| G6 code | the commit of this PR (`result.json` and every run JSON record it; dirty trees fail preflight) |
| Packages | `torch>=2.3`, `numpy`, `pyyaml`, `tokenizers>=0.15`, `pytest` (see `teams/06-performance/perfkit/requirements.txt`) |

## Notes

- Writes only under `./out/`. Traces, checkpoints and caches go to `out/.work/`, which is deleted at the end.
- No network, no keys. `--prepare-data` exists for Colab only; do not use it on the lab node.
