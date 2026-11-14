# Job request: clean Week-2 tokenizer sample v0

```text
Team:            01-data-a
Owner:           Vijay Patil (patilvijay23)
Lead approval:   Pending team-lead review/approval on the job-request PR
Question:        What cleaned parts and sequential per-rule counts does the pinned clean.py produce on the complete Week-2 tokenizer sample v0?
Script:          teams/01-data-a/spec_v1/tools/run_tokenizer_sample.py (included with this request)
Command:         python teams/01-data-a/spec_v1/tools/run_tokenizer_sample.py --config teams/01-data-a/spec_v1/jobs/tokenizer_sample_v0_clean.json
Config:          teams/01-data-a/spec_v1/jobs/tokenizer_sample_v0_clean.json
Inputs:          Week-2 tokenizer sample v0 only; pinned manifest with 48 train and 38 heldout Parquet parts, 1,444,766 documents
Hardware:        CPU (single cleaning process; request 4 vCPU and 8 GiB RAM; no GPU)
Max time:        120 minutes
Precision:       not used
Measures:        Input/output/drop/error counts per part and split; per-rule keep/fix/drop/error/invocation counts; wall time, peak RSS, output checksums
Expected result: 86 output parts with preserved split/schema/doc_ids; zero execution errors; every input row accounted for; explicit unchanged status for unsupported packs
Outputs:         out/tokenizer_sample_v0_clean/{train,heldout}/<pack_id>/part-NNNNN.parquet; reports/*.json; rule_counts.csv; job_summary.json; cleaned_manifest.json
Test log:        teams/01-data-a/jobs/logs/2026-11-02-tokenizer-sample-v0-local.txt (linked below)
```

## Scope and TSAI clarification

This is the **sample-only job explicitly requested by TSAI**, not a full-corpus cleaning request. TSAI clarified that `JOB_REQUEST.md`'s full-data restriction concerns the full corpus, and asked for cleaned sample parts plus counts per rule, with a local log on one or two parts.

The supplied manifest lists **86 Parquet parts (48 train + 38 heldout)**. With `sample_manifest.json`, that accounts for the clarification's 87-file sample; no extra 87th Parquet part is invented. The runner pins the manifest checksum and refuses other sample counts or input checksum mismatches. It processes only manifest entries, never recursively cleans arbitrary datasets.

Pinned manifest: [tokenizer_sample_v0_manifest.json](../spec_v1/jobs/tokenizer_sample_v0_manifest.json), SHA-256 `7fd647c8d4662d17e2310efbcc20981d70656fdad2fabb7663c266a2153a1b7e`.

Pinned cleaner: `97c1a55b89f4a15ef4985928ef124f29f04a37c21038eb0ea4977598767ae58f` (rebuilt after PR #28 and the local privacy/placeholder fixes; the config checks its exact bytes). The current cleaner has **179 active rule IDs**. It runs actual `clean.clean()`/`clean._run()` with counting wrappers. Optional G9 F2 is **off**: this request is for `clean.py` only.

## Inputs and offline setup

Before execution, make the complete sample available read-only under `inputs/tokenizer_sample_v0/`, relative to the repository root. The runner accepts official `train/<pack_id>/part-NNNNN.parquet` and `heldout/<pack_id>/part-NNNNN.parquet` layouts, or organized `<pack_id>/<split>/part-NNNNN.parquet`. Browser-renamed parts are resolved by checksum, so duplicate downloads do not become extra sample parts.

Install `teams/01-data-a/spec_v1/requirements.txt` before the offline run. The script downloads nothing, needs no keys, disables bytecode writes, streams one part in 128-row batches, and writes only under `./out/`. Split boundaries, column schema, and original document IDs are preserved. No document text/private values appear in logs or JSON/CSV reports. The actual sample content and returned Parquet stay under TSAI's sample-data handling; no sample Parquet is included in this request PR.

The local machine has 80 of the 86 checksum-matched parts. The six local gaps are the train and heldout parts for `cadj-commits`, `cadj-notebooks`, and `cadj-stackexchange`. TSAI must supply the full sample for the full command: preflight verifies all selected inputs before creating output. The attached two-part test has all its required inputs and does not claim a full local run.

## Coverage that must remain visible

`clean.py` currently registers 29 of the sample's 38 packs. Following its defined behavior, the other nine are preserved unchanged, with `unsupported_passthrough` in every affected output report and zero rule applications. They are not reported as successfully cleaned:

```text
cadj-commits
cadj-notebooks
cadj-stackexchange
ind-hi-sangraha-s
ind-hi-sangraha-v
ind-ml-fw2
ind-or-fw2
ind-ta-fw2
ind-ta-sangraha-s
```

This accounts for 18 manifest parts. The other 68 parts run the selected rules. The current candidate includes shared Indic F1 and the documented B2 compatibility wrappers. Non-Indic behavior and the nine unsupported-pack decisions are unchanged. This sample execution is not independent rule approval, proof of privacy completeness, or a claim of full-corpus readiness. In particular, Week-2 code uses the generic `code` language tag and science uses `en`; the existing shared F1 has narrower language-tag checks. This job preserves that behavior rather than silently remapping tags or claiming the shared privacy rule covers every row.

## Indic scope and TSAI intake rules

All **176 ready source cards from PR #240** remain in the consolidated inventory: **108 execute and 68 are explicitly deferred**, with reasons. The seven previously selected source F1 rules are replaced by shared Indic F1; other exclusions cover overlap alternatives, Unicode/idempotence failures, pipeline interactions and unresolved legacy benchmark rules. Source algorithms are not silently enabled merely because an author reported a gold pass.

The **75 borrowed same-language rule/pack scopes** each pass the receiving pack's own nonempty gold. They cover 16 supported Indic packs; another pack's gold is not used as a substitute. Missing or empty own-pack inputs explicitly defer affected rules. The copied contributor functions are unchanged; three generated B2 cards have a labelled compatibility wrapper preserving `<PHONE>`, `<EMAIL>` and `<ADDRESS>` during HTML removal. Native/sibling composed-pipeline checks include shared F1 last.

Pending re-runs, unreadable sheets awaiting author-approved rewrites, flag-only entries, and medicines without usable own-pack gold are not imported by this request. The 4 November 23:59 IST re-run deadline and later validated intake remain separate from today's sample execution. Odia is not enabled without its own gold; the cleaner already receives the whole row, including available URL fields.

Indic F1 covers phone/email masking with organization exceptions; Indian street-address masking and personal-listing drops remain unimplemented. Gold harm passes do not establish complete privacy recall.

Fresh validation: **1,115 tests passed, 1 skipped, 42 subtests passed** across `teams/01-data-a`. All 16 supported Indic gold sets pass F1 and the full cleaner (1,300 gold documents). All 8,000 supported Indic diagnosis documents execute without errors or second-pass changes. The original 6,500 non-Indic diagnosis comparisons are unchanged; three parser-cache errors (one Java, two Python) occur in both cleaners and remain explicitly recorded, not counted as successful cleaning. The two attached sample parts have zero errors.

Evidence: [validation log](logs/2026-11-02-tokenizer-sample-v0-validation.txt), [F1 and full-cleaner own-pack gold](../spec_v1/results/cleaner/f1_indic_harm.json), [native rule decisions](../spec_v1/results/cleaner/indic_integration.json), [sibling scopes](../spec_v1/results/cleaner/indic_sibling_scopes.json), [full cleaner/baseline/idempotence](../spec_v1/results/cleaner/indic_clean_validation.json), and [sheet validation](../spec_v1/results/cleaner/specv1_validation.json). Reports contain counts, document IDs and digests, not source text.

## Local execution evidence

- [Captured execution log](logs/2026-11-02-tokenizer-sample-v0-local.txt).
- [Per-rule counts from that execution](logs/2026-11-02-tokenizer-sample-v0-rule-counts.csv).
- [Machine-readable execution summary](logs/2026-11-02-tokenizer-sample-v0-summary.json).

| Complete test part | Input rows | Kept unchanged | Fixed | Dropped | Output rows | Errors | Seconds |
|---|---:|---:|---:|---:|---:|---:|---:|
| train/code-cpp/part-00001.parquet | 14,429 | 4,342 | 9,828 | 259 | 14,170 | 0 | 22.777 |
| train/ind-bn-sangraha-v/part-00000.parquet | 10,328 | 9,159 | 954 | 215 | 10,113 | 0 | 10.974 |
| Total | 24,757 | 13,501 | 10,782 | 474 | 24,283 | 0 | 33.793 wall |

Peak local process RSS: **182.891 MiB**. At the measured mixed-part rate (~733 documents/s), a simple full-sample extrapolation is about 33 minutes. Other pack lengths/rules can differ substantially; the requested 120-minute cap includes headroom. Dependency versions match the repository pins and are recorded in the log. The exact runner and cleaner used in the log are pinned there by SHA-256.

## Returned artifacts and counts

Return the cleaned parts and count reports to `teams/01-data-a/jobs/results/2026-11-02-tokenizer-sample-v0-clean/` using TSAI's sample-artifact delivery mechanism. The request PR contains metadata/logs only, not the input or cleaned Parquet files.

The complete compressed input is **1,885,334,376 bytes**, so returned cleaned Parquet can exceed the generic 50 MB job-output allowance and individual Git file limits. This is explicitly disclosed because TSAI requested the cleaned sample parts. Metadata/count reports should remain well below 50 MB; binary delivery/storage must follow TSAI's sample-specific instruction rather than silently dropping or truncating outputs to fit that cap.

`rule_counts.csv` includes every active rule ID (including shared F1), even when its invocation count is zero. Counts are **sequential actual invocations**: a fix still counts if a later rule drops the document; rules after a drop are not reached. `eligible_by_pack` is the number of input rows in an applicable pack, not an independent detector hit count. `not_reached` makes short-circuiting explicit. Deferred rules are not executed.

Each part's report records its status, input/output SHA-256, row counts, actions, language tags, timing, and per-rule counts. Errors fail the run, are reported by type and doc_id without values, and do not publish a partially written Parquet part. Existing output folders are refused, so train/heldout or reruns cannot overwrite prior results.
