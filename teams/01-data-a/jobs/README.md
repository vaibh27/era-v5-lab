# Data A jobs

| Job | Request / configuration | Execution evidence |
|---|---|---|
| PR #244 sample-v0 rerun (12 Nov) | [Current request](2026-11-12-pr244-rerun.md); [config](2026-11-12-pr244-rerun/tokenizer_sample_v0_clean.json), cleaner `22d818c7…`, optional F2 off | [Fresh validation](2026-11-12-pr244-rerun/logs/validation.txt), [two-part smoke](2026-11-12-pr244-rerun/logs/sample-smoke.txt), [summary](2026-11-12-pr244-rerun/logs/job_summary.json), [activity](2026-11-12-pr244-rerun/logs/rule-activity.json). TSAI confirmed rerun after push/green checks/merge; no full local run |
| Tokenizer sample v0 cleaning | [Request](2026-11-02-tokenizer-sample-v0-clean.md); [job folder](2026-11-02-tokenizer-sample-v0-clean/) with the config, the pinned manifest and the old run notes ([README](2026-11-02-tokenizer-sample-v0-clean/README.md)) | [Local log](2026-11-02-tokenizer-sample-v0-clean/logs/2026-11-02-tokenizer-sample-v0-local.txt), [summary](2026-11-02-tokenizer-sample-v0-clean/logs/2026-11-02-tokenizer-sample-v0-summary.json), [rule counts](2026-11-02-tokenizer-sample-v0-clean/logs/2026-11-02-tokenizer-sample-v0-rule-counts.csv), [validation log](2026-11-02-tokenizer-sample-v0-clean/logs/2026-11-02-tokenizer-sample-v0-validation.txt). TSAI's result (PR #244, 3 Nov): 86 of 86 parts, 1,444,766 documents in, 1,416,236 out, 0 rule errors; the cleaned Parquet parts stay with TSAI |
| Cleaned-sample overlap measurement | Original request/implementation absent from this checkout; provenance is recorded in the result README | [2026-11-04 overlap bundle](results/2026-11-04-overlap/README.md) |

## Layout

```
jobs/
  <date>-<name>.md        the request: JOB_REQUEST.md fixes this path, and TSAI reads it here
  <date>-<name>/          the job's config, pinned manifest, run notes and our local logs
  results/<date>-<name>/  what TSAI returns: the path is also fixed by JOB_REQUEST.md
```

A job keeps its own folder for good: nothing is moved when scripts move. Cleaned document text and Parquet outputs are kept outside the committed job evidence. The existing local log is a selected-part test; completeness must be read from the run's manifest and summary, not inferred from the file name.

## A submitted job is a record

The 2 Nov request, config and run notes are kept exactly as submitted. They name the paths of that day: `spec_v1/tools/run_tokenizer_sample.py` (now [spec_v1/run_tokenizer_sample.py](../spec_v1/run_tokenizer_sample.py)), `spec_v1/jobs/…` and `jobs/logs/…` (both now in the job folder). The config's `manifest` path and `clean_sha256` are also those of that day, so it is not runnable today, and the runner refuses a config whose `clean_sha256` is not the current `clean.py`. To reproduce that run, check out the commit of PR #244.

## Run the sample runner for a new job

Write a request and a job folder `jobs/<date>-<name>/` for it, with a config copied from the 2 Nov one and these fields changed to the new job: `manifest`, `manifest_sha256`, `clean_sha256` (the sha256 of the `clean.py` to run), and `input_root` and `output_root` if they differ. From the repository root, with [spec_v1/requirements.txt](../spec_v1/requirements.txt) installed:

```bash
python -I teams/01-data-a/spec_v1/run_tokenizer_sample.py --config teams/01-data-a/jobs/<date>-<name>/<config>.json
```

For a local smoke test add `--input-root <sample folder> --out out/<name>_smoke` and one or two `--only train/<pack>/part-NNNNN.parquet`. The runner checks every pin first, reads only the parts named in the manifest, writes only under `./out/`, keeps train and heldout apart, and reports counts from actual rule invocations. Packs without rules pass through unchanged and are reported as such. Add a test that the new config pins the current `clean.py`.
