# Indic integration

The Indic source is pinned to [PR #240 at 3bbd31f](https://github.com/Pritish52718/era-v5-lab/blob/3bbd31f051ae0f714ddf932cbec247c40d475a70/teams/02-data-b/indic_rules/medicine_indic.md), including the final visarga fixes of 6 November. The ready sheet has **260 cards on 17 native packs**. Local review selects **152**, defers **108**, and retains **10 retired local cards** separately: 270 Indic inventory IDs in total. The generated cleaner has **222 executable sheet rules plus shared F1 (223 active IDs)** on **31 packs, including 18 Indic packs**.

Local selections remain CANDIDATE. A source gold pass, compatibility check or deterministic overlap selection does not establish positive-case recall or independent approval.

## Selection and deferred rules

[build/integrate_indic.py](build/integrate_indic.py) checks ordinary prose, both additions and removals of script joiners in retained-word probes, visarga and adjacent spacing, unrelated-pack guards, mutation, runtime and idempotence. It re-runs every ready rule on current own-pack gold and checks full diagnosis-pack interactions with shared F1 last. Missing or empty own-pack inputs are NOT TESTABLE and defer affected rules.

For each of the **74 overlapping pack/disease groups**, the first eligible source candidate is provisionally selected; alternatives are deferred. This preserves the existing one-rule-per-pack/disease policy. It does not mean that the selected rule has better recall.

The new code is retained even when integration defers execution. Two material decisions in this refresh are:

- The narrowed, URL-aware `DB-D4-011` passes its 18-document own-pack gold but acts again after the full pipeline, so it stays deferred. Its source and consolidated card use author commit `a74df32`, the revised host pattern, `marks >= 6 and rate > 30`, and `_AUTHOR_RULE(doc["text"], doc.get("url"))`. Validated sibling alternatives supply missing eligible diseases; the deferred rule is never borrowed.
- The selected native Kannada FW2 pipeline changes 2 of 115 gold documents, exceeding its allowance of 1. Its native selections are deferred. The independently validated receiving pipeline uses sibling rules and passes. The native FAIL is retained in `indic_integration.json`; final receiving-pipeline evidence is in `indic_sibling_scopes.json` and `indic_clean_validation.json`.

[medicine_indic_retired.md](sheets/medicine_indic_retired.md) archives ten IDs from the older local intake with explicit policy reasons. The loader reads their metadata with AST literals and never executes their Python. Consolidated deferred cards use `STATUS = "DEFERRED"`, `DEFER_REASON` and `python deferred` fences. The builder omits them from executable `SCOPE`, `_RULE` and pipelines and records their reasons in `clean.DEFERRED_RULES`.

The retired IDs cover the two G2 rules that add ZWJ, the failing source F1 rule, both legacy F2 rules, and five visarga-conversion rules already present in the older intake. `DB-G4-005` and `DB-A6-018` were never in that local snapshot and remain outside ready intake too. #30/#112 G2, unapproved #58/#83 rewrites, and conditional #177 C6 are not enabled. Approved #46/#70/#112 rewrites are included in the pinned source, subject to the same local gates.

## Visarga, labels and action semantics

A visarga used as a colon stays as written in every script, including Tamil `ஃ`. V1 converts neither direction between a visarga and `:`, deletes no such sign and preserves the space before it. The source contains the final seven retained-rule trims: `DB-A6-004`, `DB-G1-006`, `DB-G1-002`, `DB-G1-010`, `DB-G1-005`, `DB-G1-007`, `DB-G1-014`. Their other approved jobs remain intact.

Every source rule labelled `keep` returns `("keep", doc["text"])` without calling a potentially destructive author function. **All 22 source D3 rules are label-only.** The old Kannada D3 driver is absent. `DB-G1-012` uses the latest code without the false mojibake pattern for NFC `ज़्यादा`; boolean-returning medicines carry corrected `drop` or `fix+drop` action metadata.

[test_indic_pr240.py](../tests/spec_v1/test_indic_pr240.py) checks source provenance against the pinned Git blob, all source visarga cases and selected generated rules, complete receiving pipelines, label-only behavior, Hindi NFC prose, Telugu URLs/thresholds, and non-execution of retired code. A whole-document drop can legitimately remove a sign with its words; it must also occur in the ASCII-colon control. Changes elsewhere, such as `DB-C3-017` removing list lines after a heading, are evaluated separately from visarga preservation.

## Scope and privacy

Scope follows the pack, independently of whether a language tag uses `mr` or `mar_Deva`. Source IDs retain their `DB-` prefix. The builder redirects the source `_PACKS` guard and `SPEC_PACKS` to its central `SCOPE` table, with an isolated closure per rule. F1 remains last; each family otherwise follows source section order. G3 now has an eligible source rule, while legacy F2 stays outside the default cleaner.

The source sheet declares the sibling policy separately from copied Data B code. Integration accepts **94 additional same-language rule/pack scopes**, filling missing diseases after native selections. Every receiving scope passes its own nonempty gold and full-pack runtime/idempotence/combined-pipeline checks. `SOURCE_ORIGIN` preserves the donor; accepted `ORIGIN` and `SPEC_PACKS` determine execution.

Explicit review restrictions override fallback borrowing, even if a receiving native rule is deferred: `DB-D4-011` is allowed only on `ind-te-fw2` and `DB-D5-001` only on `ind-te-sangraha-s`. Telugu sangraha-v uses its native `DB-D4-016` and `DB-D5-004`. Hindi sangraha-s now has native candidates and 24 gold documents, so it is supported.

The #362 refresh adds `ind-hi-sangraha-v`: all 219 gold IDs are present in its 500-document diagnosis pack, and 16 borrowed scopes plus shared F1 pass combined harm and runtime/idempotence checks. On `ind-mr-fw2`, the rebuilt gold rejects borrowed `DB-C1-012` because its combination with native `DB-C3-024` changes two of 158 gold documents (allowance one). The native rule and the donor scope on Marathi sangraha-v remain enabled. See the [refresh evidence](../results/spec_v1/cleaner/pr244_gold_refresh/README.md).

Four Indic packs remain unregistered: `ind-ta-fw2`, `ind-ta-sangraha-s`, `ind-ml-fw2`, `ind-or-fw2`. Their gold is empty; Malayalam and Odia also lack a same-language donor. `run_clean.py` rejects unsupported packs. The sample runner explicitly passes them through rather than calling them cleaned.

Every supported Indic pack runs shared [build/f1_rule.py](build/f1_rule.py) last. It masks private phones/emails with `<PHONE>`/`<EMAIL>` and keeps organisation/public-office contacts; source F1 candidates stay deferred. Indian street-address masking and personal-listing drops remain unimplemented. Labelled Data A adapters on `DB-B2-001`, `DB-B2-003`, `DB-B2-004` protect `<PHONE>`, `<EMAIL>`, `<ADDRESS>` during HTML removal; copied Data B functions are unchanged by those adapters.

The four #19 rules `DB-C3-010`, `DB-C4-009`, `DB-D4-008`, `DB-F1-007` match sample-document hashes. They provide targeted sample coverage and do almost nothing on the full pack; do not infer general recall. Source `DB-F1-007` is replaced locally by shared F1.

## Rebuild and validate

Use Python 3 with [requirements.txt](requirements.txt) installed. Run from the repository root:

```bash
python3 teams/01-data-a/spec_v1/build/integrate_indic.py \
  --packs ../erav5-lab-other-files/packs \
  --gold teams/02-data-b/gold
python3 teams/01-data-a/spec_v1/build/build_clean.py
python3 -m pytest teams/01-data-a/tests -q
python3 teams/01-data-a/spec_v1/validate/validate_specv1.py \
  --packs ../erav5-lab-other-files/packs \
  --gold teams/01-data-a/gold \
  --indic-gold teams/02-data-b/gold
python3 teams/01-data-a/spec_v1/validate/harm_f1_indic.py \
  --packs ../erav5-lab-other-files/packs --gold teams/02-data-b/gold
```

Integration refreshes native decisions, archived cards and sibling scopes in the consolidated spec. The builder reads only `sheets/specv1.md`; running it alone cannot import updated source sheets. Standalone `extend_indic_siblings.py` reuses native decisions only when source-code and native input digests still match. Rebuild after changing executable sheet code and verify the sample config's cleaner hash.

The validator applies `max(1, floor(gold_count / 100))` changed-or-dropped documents per rule and combined pipeline, separately per pack. `--allow-missing` records NOT TESTABLE and exits 2; it is not a complete pass. All 18 nonempty Indic gold CSVs have matching local week-1 diagnosis inputs. Week-2 shards do not replace diagnosis packs or gold IDs.

For an independent comparison, save the pre-refresh cleaner before editing, then run:

```bash
python3 teams/01-data-a/spec_v1/validate/check_indic_integration.py \
  --packs ../erav5-lab-other-files/packs --gold teams/02-data-b/gold \
  --baseline /path/to/saved-clean.py
```

The refreshed report compares with the pre-gold-refresh cleaner from `476f026` (SHA-256 `21d95374…`). All 6,500 non-Indic documents have identical behavior and zero errors; all 9,000 Indic diagnosis documents have zero errors or second-run changes. Matching baseline exceptions would be recorded separately, never counted as successful cleaning.

## Optional F2 validation

[validate/validate_f2.py](validate/validate_f2.py) measures optional G9 F2 alone and after the cleaner on all 31 supported packs, using each pack's own gold and the full pinned **v2** set (`2826a3b8…`, normalizer `g9-v2`). Its [current report](../results/spec_v1/f2/v2_validation.json) includes all 18 Indic gold sets (2,193 documents): F2 alone drops zero raw Indic gold documents, and both stages pass the per-pack harm limits. [validate/validate_f2_indic.py](validate/validate_f2_indic.py) remains available for an Indic-only rerun. The earlier [Indic v1 report](../results/spec_v1/f2/indic_gold_validation.json) and [post-#362 packet](../results/spec_v1/cleaner/pr244_gold_refresh/README.md) retain their original hashes as historical evidence. See the [v2 migration findings](../results/spec_v1/f2/README.md#current-v2-pin-and-validation) and [F2 guide](../results/spec_v1/f2/README.md).

G9 F2 remains separate and opt-in through `run_clean.py --f2`, not a default `clean.py` function. It does not cover Itihasa and does not enable the two retired Sanskrit detectors. Optional F2 stays off in the sample job.

## Measured verification and evidence

The [post-#362 refresh for PR #244](../results/spec_v1/cleaner/pr244_gold_refresh/README.md) uses **2,193 current gold documents across 18 Indic packs**, and checks **9,000 Indic diagnosis documents**. The generated cleaner hash is `22d818c72c9f30bf338de344a0fcda230e2d1de30a1aaec5d92bc525bae99179`. All 31 generated-cleaner harm checks pass on 4,162 own-pack gold documents; the current workspace suite passes 2,112 tests with one skipped and 42 subtests. Exact per-pack results and digests are recorded in:

- [indic_integration.json](../results/spec_v1/cleaner/indic_integration.json): all 260 ready and 10 retired decisions, 74 overlap groups, native gold and interaction findings.
- [indic_sibling_scopes.json](../results/spec_v1/cleaner/indic_sibling_scopes.json): every accepted/rejected receiving scope, policy restrictions and final combined checks with shared F1.
- [indic_clean_validation.json](../results/spec_v1/cleaner/indic_clean_validation.json): current generated-cleaner gold/full-pack/idempotence results and the 6,500-document non-Indic comparison.
- [f1_indic_harm.json](../results/spec_v1/cleaner/f1_indic_harm.json): shared F1 alone and the complete generated cleaner on current Indic gold.
- [specv1_validation.json](../results/spec_v1/cleaner/specv1_validation.json): executable sheet checks, per-rule/per-pack harm and out-of-pack guards.
- [Refresh packet](../results/spec_v1/cleaner/pr244_gold_refresh/README.md): scope changes, hashes, per-pack outcomes and reproduction commands.
- [Current sample-v0 rerun request](../jobs/2026-11-12-pr244-rerun.md): current `22d818c7…` pin, fresh validation and two-part Hindi/Telugu smoke; optional F2 off.

JSON reports contain IDs, counts, digests and error types, not document text. The other four family source sheets and normalization fingerprints are unchanged. A gold pass does not establish privacy completeness or positive-case recall; these remain review limits.
