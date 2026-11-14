# Week-2 tokenizer sample job

The TSAI request is [2026-11-02-tokenizer-sample-v0-clean.md](../../jobs/2026-11-02-tokenizer-sample-v0-clean.md). This topic contains the sample-only config and checksum-pinned public manifest; it contains no Parquet data.

From the repository root, with inputs staged and pinned requirements installed:

```bash
python teams/01-data-a/spec_v1/tools/run_tokenizer_sample.py --config teams/01-data-a/spec_v1/jobs/tokenizer_sample_v0_clean.json
```

For a local test, override the sample mount and select one or two manifest part paths:

```bash
python teams/01-data-a/spec_v1/tools/run_tokenizer_sample.py \
  --config teams/01-data-a/spec_v1/jobs/tokenizer_sample_v0_clean.json \
  --input-root /path/to/tokenizer_sample_v0 \
  --out out/tokenizer_sample_smoke \
  --only train/code-cpp/part-00001.parquet \
  --only train/ind-bn-sangraha-v/part-00000.parquet
```

Without `--only`, all 86 parts must pass preflight. All inputs are checked against the pinned manifest. The output root must be a fresh named subdirectory of `./out/`. The runner streams batches and materializes all output parts, including zero-row outputs with a preserved schema. Unsupported packs pass through explicitly; no new medicines are inferred. Optional F2 is disabled for this request.

Per-rule counting instruments actual factories while retaining the existing `clean.clean` and `_run` execution. Reports distinguish eligible-by-pack, invocations, keep/fix/drop/error and not-reached counts; they contain no document text. Counts are sequential, not independent whole-pack rule evaluations. The cleaned manifest marks a two-part smoke test as incomplete and cannot be mistaken for a full sample run.

The included local log used the pinned requirements and two complete parts, with zero errors. Only execution logs, counts and metadata are committed. Local cleaned test parts remain in ignored `out/`.

The current request includes the rebuilt cleaner after Indic F1 PR #28 and the local privacy/placeholder fixes. There are 179 active rule IDs on 29 supported sample packs. The Indic inventory has 176 source cards: 108 selected, 68 deferred; 75 separately validated sibling rule/pack scopes extend coverage to 16 Indic packs, all followed by shared F1. Copied contributor functions remain unchanged; three B2 cards include a labelled wrapper that preserves privacy tokens during HTML removal.

TSAI's own-pack-gold requirement remains mandatory. Missing or empty gold defers the rules; other packs' gold is not a substitute. Pending re-runs, author-approved format rewrites, flag-only entries and missing-gold medicines are not silently enabled. Gold harm reports contain counts, document IDs and digests only. The request links fresh sheet and complete-cleaner evidence; a gold pass does not establish complete privacy recall, and Indic street-address masking/personal-listing drops remain unimplemented.
