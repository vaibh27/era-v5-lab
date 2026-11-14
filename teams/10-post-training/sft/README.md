# SFT data tools: conversion and decontamination

This README covers the tools that turn a candidate SFT set into chat-format v0 rows and check those rows against Team 9's benchmark fingerprints. The OCR-2 admission scripts, the mix inventory, the SFT probe and the max-sequence proposal in this folder have their own notes and are not described here.

## What is here

| File | What it does |
|---|---|
| `convert.py` | Converts one candidate set to chat-format v0 JSONL with a loss mask and a manifest. Every row goes through `chat-format/render.py`. |
| `test_convert.py` | Tests for `convert.py`, on hand-written rows in each set's schema (no real data). |
| `decontam_run.py` | Runs Team 9's `decontam_check.py` and `canary_scan.py` on one set, writes the set without the flagged and canary rows, and writes a record of counts and ids. |
| `decontam_report.py` | Builds `decontam_report.md` from those records. |
| `decontam_pipeline.py` | Runs download, convert and decontam for several sets in parallel, then merges the part records. |
| `test_decontam_run.py` | Tests for `decontam_run.py` and `decontam_report.py`, on a toy fingerprint set built from made-up items. |
| `decontam_report.md` | The result per set: flagged, canary and kept rows, benchmarks hit, and the inputs to reproduce a run. |
| `decontam_kept_rows.md` | Rows in, removed and kept per set for the sets run on every converted row. |

The RL sets use the same tools. Their report is `rl/decontam_report.md`.

## Convert a set

```bash
python teams/10-post-training/sft/convert.py --set sft-code-codeforces-cots --hf-rows 1000 --out out/codeforces.jsonl
python teams/10-post-training/sft/convert.py --set sft-indic-aya-dataset --input aya_train.parquet --limit 1000 \
    --out out/aya.jsonl --tokenizer tokenizer.json
```

It writes `<out>` and `<out>.manifest.json`.

- The set must have a Stage 1 row in `licences/licence_table.md`. A set marked `excluded`, or missing from the table, is refused. The licence, source, generator model and decision in the manifest come from that row.
- A row that does not fit the chat format is skipped and counted by reason. It is never patched.
- The effort level is the smallest budget (1k, 4k, 16k, 64k) that holds the row's trace. Without `--tokenizer` the token count is an estimate (UTF-8 bytes / 4), and the manifest says so. With `--tokenizer`, each row also gets `input_ids` and a per-token `loss_mask`.
- Source options: `--input` (a local `.parquet` or `.jsonl`), `--hf-rows N` (from the Hugging Face datasets-server) or `--hf-file N` (N rows read from the set's own file in the Hub repo by ranged download; `--file-spec`, `--index-base` and `--roll-mb` are what `decontam_pipeline.py` uses for parallel runs).

## Decontaminate a set

```bash
python teams/10-post-training/sft/decontam_run.py --set sft-code-codeforces-cots --input out/codeforces.jsonl --out-dir decontam_out
python teams/10-post-training/sft/decontam_report.py --records decontam_out --out decontam_out/decontam_report.md
```

Write a trial report to a scratch path as above. `sft/decontam_report.md` is built from the records of all sets with `--sets`, so do not point `--out` at it from a one-set run.

`decontam_run.py` does five things for one set:

1. Runs `decontam_check.py` on the full fingerprint set. It refuses any other scope, and any `set_sha256` other than the pinned one (fingerprints v2, lab PR #280).
2. Runs `canary_scan.py` on the same files. A canary is a leak, never a pass.
3. Writes the set without the flagged and canary documents (`<set>.clean.jsonl` or `.parquet`).
4. Reconciles the counts: rows in = rows kept + rows removed, ids are unique, every flagged id is an input id, and the checker run again on the clean file flags 0.
5. Writes `<set>.record.json` with counts, ids, benchmark item ids, offsets, sha256 values and the hashes of the tools.

Exit code 0 means every check passed (flagged documents may exist: they are removed and recorded). Exit code 1 means a canary was found or a reconciliation check failed: stop using the file and post in G9 with the item or canary id, document id and offset only. Exit code 2 means a tool could not run.

`decontam_report.py` refuses to write when a record failed, when a record has no second pass over its clean file, when the records carry different `set_sha256` values or different Team 9 tool files, or when the `set_sha256` is not the expected one. Use `--sets` to name the sets to include, and `--pending` for sets of the mix that were not run yet.

For several sets at once, `decontam_pipeline.py` converts and checks each downloaded part as it arrives. It writes `progress.log` and `status.txt` in its work folder so a person can watch the run.

## Rules

- **Team 9's tools.** They come from lab main, not from this folder. Point `G10_DECONTAM_TOOLS` at `teams/09-evaluation/decontam` of a lab main checkout that has fingerprints v2 (the report names the commit it used). The folder on an older branch does not have v2, and the tests that need it are skipped.
- **Plain `python`.** Run these scripts with plain `python`, not `python -I`: Team 9's scripts import their neighbours from their own folder.
- **No benchmark text, no document text.** Only counts, document ids, benchmark item ids, character offsets and sha256 values leave these scripts (eval firewall, section 4). Keep the `--out-dir` out of Git: the clean files and the records hold training rows and document ids.
- **A rerun is needed** when Team 9 publishes a new fingerprint set (a different `set_sha256`), and for any set whose rows change.

## Tests

```bash
G10_DECONTAM_TOOLS=<lab main>/teams/09-evaluation/decontam \
  python -m pytest teams/10-post-training/sft/test_convert.py teams/10-post-training/sft/test_decontam_run.py -q
```

The Hub calls in `test_convert.py` are mocked, and `test_decontam_run.py` states that it needs no network. Without `G10_DECONTAM_TOOLS` pointing at a v2 folder, the decontam tests that need Team 9's tools are skipped.
