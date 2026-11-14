# PR #240 change plan

Current follow-up: [PR #244 validation after gold rebuild #362](../results/spec_v1/cleaner/pr244_gold_refresh/README.md) is complete locally, including the new Hindi diagnosis pack and the necessary Marathi receiving-scope correction. It supersedes the validation counts below. TSAI review/merge and the sample-job decision remain pending.

Historical status: implemented and validated on 10 November 2026 after the pull to
`88e22653781904bdb82d663bfc3825ff1d089c67`. The changes remain local for review.

## Rechecked inputs and review decisions

[PR #240](https://github.com/theschoolofai/era-v5-lab/pull/240) still has 26
conversation/review entries, no inline review threads, and source head
`3bbd31f051ae0f714ddf932cbec247c40d475a70`. Its latest discussion is the
[6 November visarga follow-up](https://github.com/theschoolofai/era-v5-lab/pull/240#issuecomment-6680263871).
[PR #244](https://github.com/theschoolofai/era-v5-lab/pull/244) still has five
entries; its latest is the already-incorporated
[Telugu URL/scope request](https://github.com/theschoolofai/era-v5-lab/pull/244#issuecomment-6687939357).
No new source commits or review comments superseded the earlier plan.

The ready source is pinned to Data B's Git blob
`f972241202febc587a95953f27397721669e6243`: 260 ready rules on 17 native
packs and 74 overlapping pack/disease groups. Local selection remains one
eligible candidate per pack/disease in source order, with alternatives retained
as deferred inventory. No superior-recall or approval claim follows from that choice.

## Completed implementation

1. **Pinned intake and exclusions.** Refreshed [medicine_indic.md](sheets/medicine_indic.md)
   with preserved IDs, source order, authors, code and harm provenance. The
   separate Data A sibling policy and three B2 privacy-placeholder adapters remain.
   [medicine_indic_retired.md](sheets/medicine_indic_retired.md) archives ten
   local IDs as non-executable history. Their metadata is read using AST
   literals; their Python is never executed.
2. **Visarga.** Imported all seven retained-rule trims: `DB-A6-004`,
   `DB-G1-006`, `DB-G1-002`, `DB-G1-010`, `DB-G1-005`, `DB-G1-007`,
   `DB-G1-014`. Per-rule and complete-pipeline regressions preserve visarga
   and adjacent spacing, including Tamil `ஃ`, with ASCII-colon controls.
3. **Keep/action semantics.** All 22 D3 rules are label-only and keep the
   original text without invoking a destructive author function. Carried
   corrected action metadata for `DB-G1-012`, `DB-A6-003`, `DB-C3-008`,
   `DB-C3-016`, removed the old Kannada D3 driver, and retained the G1 code
   without the false mojibake match on NFC `ज़्यादा`.
4. **Telugu code and scopes.** Imported `DB-D4-011` from author commit
   `a74df32`, with URL forwarding and the narrowed thresholds. The sibling
   policy and dispatcher enforce `DB-D4-011` only on `ind-te-fw2` and
   `DB-D5-001` only on `ind-te-sangraha-s`, even when a receiving native
   rule is deferred. Telugu sangraha-v uses native `DB-D4-016` and `DB-D5-004`.
5. **Selections and generated output.** Reapplied changes to
   [build/integrate_indic.py](build/integrate_indic.py) and
   [build/extend_indic_siblings.py](build/extend_indic_siblings.py), preserving
   the incoming paths/imports. Recomputed native selection and 79 sibling
   scopes, regenerated [specv1.md](sheets/specv1.md), and built
   [clean.py](clean.py) through [build_clean.py](build/build_clean.py).
   Shared F1 remains last; joiner probes reject additions and removals.
6. **Tests, reports and docs.** Moved the saved regression coverage into
   [tests/spec_v1](../tests/spec_v1/), including
   [test_indic_pr240.py](../tests/spec_v1/test_indic_pr240.py). Regenerated
   integration, sibling, cleaner/baseline, shared-F1 and sheet-validation reports
   in [results/spec_v1/cleaner](../results/spec_v1/cleaner/). Updated tool,
   sheet, cleaner, evidence and job documentation for the current layout.

The archived IDs are `DB-G2-001`, `DB-G2-002`, `DB-F1-005`, `DB-F2-001`,
`DB-F2-002`, `DB-G4-001`, `DB-A6-010`, `DB-A6-011`, `DB-A6-012`,
`DB-A6-014`. The other excluded conversion IDs, `DB-G4-005` and `DB-A6-018`,
remain outside ready intake. No excluded rule is executable or borrowable.

The narrowed `DB-D4-011` still acts again after the complete pipeline and is
therefore deferred despite passing its own gold. The native Kannada FW2
combination changes 2 of 114 gold documents (allowance 1), so it is rejected;
the final receiving pipeline uses independently validated sibling alternatives.
Both decisions are retained in the reports.

## Validation on the pulled tree

Cleaner SHA-256:
`14ce613473e691b67cc4d46598c8cb7bec6b7fbd49ef2c2563380f8d88a55de9`.
The saved pulled baseline has SHA-256
`97c1a55b89f4a15ef4985928ef124f29f04a37c21038eb0ea4977598767ae58f`.

| Check | Fresh result |
|---|---|
| All Data A tests | 1,514 passed; 1 skipped; 42 subtests passed |
| Ready/retired inventory | 260 ready: 152 selected, 108 deferred; 10 retired |
| Generated coverage | 223 active rule IDs on 30 packs, including 17 Indic |
| Own-pack Indic gold | All 17 pass; 1,957 documents; shared F1 alone also passes |
| Indic diagnosis pipelines | 8,500 documents; zero runtime errors or second-pass changes |
| Non-Indic baseline comparison | 6,500 documents; zero behavior differences or exceptions |
| Executable sheet and out-of-pack guards | Pass on all 30 supported packs |
| Build/source/config consistency | Reproducible clean.py; pinned source blob; current cleaner/manifest pins |

Validation used pinned requirements and seven prefetched tree-sitter grammars
in a temporary cache. The older three parser-cache exceptions were not
reproduced. All required validation commands exited 0, without `--allow-missing`.
See the [current reports](../results/spec_v1/cleaner/) for per-pack counts and input/code digests.

## Review and execution boundaries

Selections remain CANDIDATE. Positive-case review and the four #19 medicines'
sample-hash-specific scope remain explicit. Gold passes do not establish
privacy completeness: Indian street-address masking and personal-listing drops
remain unimplemented. Five empty-gold Indic packs remain unregistered; source
F1 is replaced by shared F1, and optional G9 F2 stays off. No new Itihasa coverage,
full-sample run, remote execution, commit or PR comment was performed.

## Recovery

The original stash `64883b3d3c949e636634ab7bd78a974329f1d717` remains intact.
Its untracked-file parent contains the full earlier planning/handoff document:

```bash
git show 64883b3^3:teams/01-data-a/spec_v1/PR240_change_plan.md
```

Unrelated audit/viewer work remains saved there. Root `.gitignore` is unchanged,
untracked and unstashed. The removed `spec_v1/tools`, `spec_v1/tests`,
`spec_v1/results` and `spec_v1/jobs` layouts were not recreated.
