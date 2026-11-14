# PR #244 validation after gold rebuild #362

Completed locally against branch head `2eff94990675e06643021bcce5c94c75867bca8a`,
including merged gold commit `e9550201a5a25f3798a8566fadb6e5d294be031a`.
Cleaner SHA-256:
`22d818c72c9f30bf338de344a0fcda230e2d1de30a1aaec5d92bc525bae99179`.
The ready rule source remains PR #240 at `3bbd31f`.

## Result

- **31 supported packs, including 18 Indic packs; 223 active rule IDs.**
- **18/18 Indic pipelines PASS on all 2,193 current own-pack gold documents.**
- **9,000 Indic diagnosis documents:** zero runtime errors or second-pass changes.
- **6,500 non-Indic diagnosis documents:** identical behavior to pre-refresh
  cleaner `21d95374…`, with zero errors. Non-Indic/shared function ASTs are unchanged,
  preserving the merged English and code speed optimizations.
- **31/31 generated-cleaner harm checks PASS**, on 4,162 own-pack gold documents.
  Every active individual rule, disease-code function, full pipeline, runtime and
  out-of-scope check passes.
- **2,112 tests passed, 1 skipped, 42 subtests passed** in the current workspace.
  The isolated committed-input checkout passes 2,111; the extra local test checks
  the preserved deferred 10 November job snapshot.
- **Optional F2:** the full branch-pinned v1 set passes on all 18 Indic gold sets,
  both alone and after the cleaner. F2 alone drops zero raw gold documents. It
  remains opt-in and does not cover Itihasa.
- A fresh build reproduces the cleaner byte-for-byte.

Results use the existing per-pack policy: at most
`max(1, floor(gold_count / 100))` changed-or-dropped gold documents. PASS does not
mean zero gold changes. Per-pack counts are below.

## Necessary scope changes

**Hindi sangraha-v:** the added Week-1 diagnosis file has 500 unique documents
and contains all 219 rebuilt gold IDs. It now receives 16 validated same-language
borrowed scopes and shared F1. The final pipeline changes two gold documents,
drops none, has no errors, and meets its allowance of two. Runtime/idempotence
checks cover all 500 diagnosis documents.

**Marathi FW2:** the old receiving pipeline changes two of the rebuilt 158 gold
documents, above its allowance of one. Native `DB-C3-024` changes one; borrowed
`DB-C1-012` changes another. The existing integration policy rejects that borrowed
scope because their combined harm exceeds the limit. The donor rule stays enabled
on `ind-mr-sangraha-v`, the native rule stays on FW2, and the final FW2 pipeline
changes one gold document and drops none. The rejected attempt and affected IDs
remain in the sibling report and [pre-refresh trace](marathi_pre_refresh_harm.json).

The native selection stays **152 selected / 108 deferred**, plus **10 retired**
cards. Accepted sibling scopes change from **79 to 94**: 16 added for Hindi
sangraha-v and one removed for Marathi FW2. No contributor algorithm or source
rule ID changed. The rejected native Kannada FW2 combination remains recorded as
FAIL on its rebuilt gold; its final sibling-backed receiving pipeline passes.
`DB-D4-011` remains deferred, and Telugu donor restrictions stay intact.

## Current Indic gold results

| Pack | Gold | Changed | Dropped | Result |
|---|---:|---:|---:|---|
| ind-bn-fw2 | 227 | 1 | 1 | PASS |
| ind-bn-sangraha-v | 137 | 1 | 0 | PASS |
| ind-gu-fw2 | 80 | 0 | 0 | PASS |
| ind-hi-fw2 | 218 | 1 | 1 | PASS |
| ind-hi-hplt | 37 | 0 | 0 | PASS |
| ind-hi-indiccorp | 425 | 1 | 1 | PASS |
| ind-hi-sangraha-s | 24 | 0 | 0 | PASS |
| ind-hi-sangraha-v | 219 | 2 | 0 | PASS |
| ind-kn-fw2 | 115 | 1 | 0 | PASS |
| ind-kn-sangraha-s | 22 | 0 | 0 | PASS |
| ind-kn-sangraha-v | 99 | 0 | 0 | PASS |
| ind-mr-fw2 | 158 | 1 | 0 | PASS |
| ind-mr-sangraha-v | 99 | 0 | 0 | PASS |
| ind-sa-fw2 | 71 | 1 | 0 | PASS |
| ind-ta-sangraha-v | 141 | 1 | 0 | PASS |
| ind-te-fw2 | 17 | 0 | 0 | PASS |
| ind-te-sangraha-s | 24 | 0 | 0 | PASS |
| ind-te-sangraha-v | 80 | 0 | 0 | PASS |

## Evidence and reproduction

[validation_summary.json](validation_summary.json) records commit/cleaner/input/
gold/report hashes, the exact scope rejection, full harm summary and test counts.

- [Native integration](../indic_integration.json) and [receiving scopes](../indic_sibling_scopes.json).
- [Generated-cleaner validation](../indic_clean_validation.json),
  [shared-F1 gold](../f1_indic_harm.json), [sheet validation](../specv1_validation.json).
- [All-pack harm tables](../harm_tests.md) and their per-pack JSON reports.
- [Optional F2 gold validation](../../f2/indic_gold_validation.json).

Execution logs are retained locally. The versioned packet contains this README,
`validation_summary.json` and `marathi_pre_refresh_harm.json`; detailed validation
results are in the canonical reports linked above.

From the repository root, with the pinned requirements installed and the matching
Week-1 diagnosis packs, including Hindi sangraha-v, staged:

```bash
mkdir -p out/pr244_validation
git show 476f026:teams/01-data-a/spec_v1/clean.py > out/pr244_validation/pre_refresh_clean.py
python -I teams/01-data-a/spec_v1/build/integrate_indic.py \
  --packs ../erav5-lab-other-files/packs --gold teams/02-data-b/gold
python -I teams/01-data-a/spec_v1/build/build_clean.py
python -I teams/01-data-a/spec_v1/validate/validate_specv1.py \
  --packs ../erav5-lab-other-files/packs \
  --gold teams/01-data-a/gold --indic-gold teams/02-data-b/gold
python -I teams/01-data-a/spec_v1/validate/check_indic_integration.py \
  --packs ../erav5-lab-other-files/packs --gold teams/02-data-b/gold \
  --baseline out/pr244_validation/pre_refresh_clean.py
python -I teams/01-data-a/spec_v1/validate/harm_f1_indic.py \
  --packs ../erav5-lab-other-files/packs --gold teams/02-data-b/gold
python -I teams/01-data-a/spec_v1/validate/validate_f2_indic.py \
  --packs ../erav5-lab-other-files/packs --gold teams/02-data-b/gold
python -m pytest teams/01-data-a/tests -q
```

Native integration was rerun against #362. Its validated native decisions were
then reused by the sibling extension after the missing Hindi diagnosis file was
supplied. The extension verifies native source/input/F1 digests before reuse.
Full validators were rerun on the final 18-pack cleaner. A temporary own-pack gold
lookup combined Data A and Data B CSVs for all 31 generated-cleaner harm checks;
no gold CSV was modified.

## Remaining external decisions

This completes local refreshed validation. TSAI review/merge and the decision on
a new tokenizer-sample cleaning job remain separate. No new job was submitted or
full tokenizer sample processed. The deferred 10 November request/config/logs
remain the historical `14ce6134` candidate and cannot serve as this cleaner's pin.

The unsupported Indic packs are Tamil FW2, Tamil sangraha-s, Malayalam FW2 and
Odia FW2, which still lack valid own-pack evidence. Privacy-address/listing
coverage and positive-case recall retain their documented policy limits.
