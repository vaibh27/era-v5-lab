# Validation evidence index

These are recorded runs, not automatically refreshed attestations of the current cleaner. Read each report's input, gold, sheet/rule, cleaner, and fingerprint hashes before applying its result to a build. Historical code hashes and provenance paths are preserved during cleanup.

The [post-#362 PR #244 validation](cleaner/pr244_gold_refresh/README.md) refreshes the integration, sibling-scope, generated-cleaner, shared-F1, sheet and Indic F2 reports for cleaner `22d818c7…`. All 18 Indic pipelines pass on 2,193 own-pack gold documents; 9,000 Indic diagnosis documents have zero errors or second-pass changes. All 31 generated-cleaner harm checks pass on 4,162 gold documents. The packet records digests, reproduction commands and the necessary Hindi/Marathi scope changes.

The [12 November sample-v0 rerun request](../../jobs/2026-11-12-pr244-rerun.md) pins the current cleaner and has fresh validation and two-part smoke logs. No new full sample has been processed locally.

**Start here:** the harm test of every rule and of `clean()` on each pack is [cleaner/harm_tests.md](cleaner/harm_tests.md), with the English packs in more detail in [cleaner/harm_tests_english.md](cleaner/harm_tests_english.md). The validation of the sheet itself is [cleaner/specv1_validation.json](cleaner/specv1_validation.json).

| Family / location | Producer | Interpretation |
|---|---|---|
| [PR #244 gold-refresh packet](cleaner/pr244_gold_refresh/README.md) | Integration, build and validation commands in the packet | Post-#362 scope changes, current hashes, all-pack harm, tests and optional Indic F2 |
| [cleaner/harm_tests.md](cleaner/harm_tests.md), `cleaner/specv1_<pack>_harm.json` | [make_harm_tests.py](../../spec_v1/validate/make_harm_tests.py) | Generated-cleaner per-rule/function/combined harm and out-of-scope checks; inspect `rules_file_sha256` |
| [cleaner/specv1_validation.json](cleaner/specv1_validation.json) | [validate_specv1.py](../../spec_v1/validate/validate_specv1.py) | Executable-sheet validation snapshot; code/input/gold digests are recorded separately |
| [cleaner/harm_tests_english.md](cleaner/harm_tests_english.md), `cleaner/clean_*`, `cleaner/clean_stress_*` | [English tools](../../spec_v1/validate/README_english.md) | English cleaner, ordering, privacy, and disputed-gold snapshots |
| [cleaner/harm_tests_f1_math_sci.md](cleaner/harm_tests_f1_math_sci.md) | Shared-F1 validation documented in the [cleaner guide](../../spec_v1/README_clean.md) | Historical math/science privacy harm snapshot |
| [cleaner/indic_integration.json](cleaner/indic_integration.json), [indic_sibling_scopes.json](cleaner/indic_sibling_scopes.json) | [integrate_indic.py](../../spec_v1/build/integrate_indic.py), [extend_indic_siblings.py](../../spec_v1/build/extend_indic_siblings.py) | Source selection, exclusions, input digests, and validated receiving scopes; sibling evidence is also read by tests |
| [cleaner/indic_clean_validation.json](cleaner/indic_clean_validation.json), [f1_indic_harm.json](cleaner/f1_indic_harm.json) | [check_indic_integration.py](../../spec_v1/validate/check_indic_integration.py), [harm_f1_indic.py](../../spec_v1/validate/harm_f1_indic.py) | Cleaner/baseline comparison and Indic privacy checks; use embedded cleaner hashes |
| [sheets/](sheets/): code evidence, manifests, `<pack>_harm.json`, `stress_<pack>_harm.json` | [Sheet reproduction scripts](../../oneoff/README.md) | Contributor selection and official/stress snapshots; manifests record input versions |
| [sheets/evidence_english.json](sheets/evidence_english.json), [equivalence_english.json](sheets/equivalence_english.json), English pack reports | [English guide](../../spec_v1/validate/README_english.md) and English harm runner | Compilation tooling for the evidence/equivalence files is not committed; preserve this reproduction dependency |
| [speed_v1.md](speed_v1.md), [code-pack follow-up](speed_v1_code.md), [full evidence](speed_v1_code.json) | [speed_compare.sh](../../oneoff/speed_compare.sh) and [speed_check.py](../../oneoff/speed_check.py) (before/after run and compare), [run_clean.py](../../spec_v1/run_clean.py) (per-rule profile) | Per-core throughput, output hashes, harm comparisons and remaining bottlenecks. Reports and evidence are canonical here; [TSAI's requested path](../../spec_v1/results/speed_v1.md) is a pointer. |
| [F2 v2 migration](f2/README.md#current-v2-pin-and-validation), [v2 validation](f2/v2_validation.json) | [validate_f2.py](../../spec_v1/validate/validate_f2.py) | Current full-set v2 harm validation on 31 supported packs; older v1 reports retain their hashes |
| [f2/README.md](f2/README.md), `f2/f2_<pack>_harm.json`, [Indic F2 report](f2/indic_gold_validation.json) | [f2_rule.py](../../spec_v1/f2_rule.py), [validate_f2_indic.py](../../spec_v1/validate/validate_f2_indic.py) | Optional full-set G9 evidence; reports bind to a fingerprint-set hash and may bind to an older cleaner |

The ten historical reports formerly directly under `results/` were removed during cleanup. Current writers target these family subdirectories. Official/stress reports can be byte-identical and still have distinct roles; keep both names while readers/writers expect them.

Run dates are available in summary headers or linked execution logs where recorded. Some raw reports contain no date: their hashes are authoritative snapshot identifiers; no date is inferred here. Reproducing real-pack evidence requires original diagnosis packs and the matching gold, not week-2 train/heldout shards.

For future report-layout changes, migrate readers/writers and import-time test inputs together before moving files. Keep job-specific execution logs and overlap evidence in the [jobs index](../../jobs/README.md).
