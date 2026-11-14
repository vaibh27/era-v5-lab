# Code-pack speed follow-up

Measured on 11 November 2026, after merging Smita’s PR #34. Baseline: `9299ff20afaa3f6511ad1751a42511576de95e2b`; baseline cleaner `11da63a27cca46556a5736a136cc9cd9b4af8f5e9261268ebb298bca9f1237b4`. Optimized cleaner `21d95374ca5522dfe0bd964142ab571962e4186901054e38da730d91fad5ce47`.

Full input/output digests, action/error counts, all three timings, cold-pass throughput, own-pack gold sources and final rule profiles are in [speed_v1_code.json](speed_v1_code.json). No corpus text is included.

## Changes

- **DA-E3-004:** check necessary literal prefixes before known-key substitutions and a Unicode-aware name hint before the expensive assigned-secret pattern. Keep the regexes, entropy checks, placeholders and substitution order unchanged.
- **Shared F1, code path:** skip impossible contact scans and defer organisation context until a phone/address could be masked. Preserve previous-line context, Unicode digits and tree-sitter comment/string boundaries. The English, prose and Indic paths keep their behavior.
- **Acceptance tools:** require the exact expected pack inventory, positive matching document counts and zero errors; propagate pytest/profile/count failures; compare complete harm-result sets. Regression tests cover the former empty/error-run false passes.

## Throughput per core

One process pinned to logical CPU 0, three passes per cleaner, best throughput reported. Packs are loaded before timing; hashing is outside the clock. Baseline and optimized runs were sequential, with no competing validation jobs. CPU: Intel(R) Core(TM) Ultra 5 226V; Python 3.12.3. These results compare against the merged cleaner, not TSAI’s original pre-PR34 baseline.

| Pack | Diagnosis: before → after docs/s | Gain | 5,000-doc slice: before → after docs/s | Gain |
|---|---:|---:|---:|---:|
| code-cpp | 438.7 → 948.8 | 2.16× | 650.6 → 1055.4 | 1.62× |
| code-go | 598.7 → 1071.0 | 1.79× | 805.7 → 1063.9 | 1.32× |
| code-java | 669.1 → 1347.6 | 2.01× | 896.2 → 1316.1 | 1.47× |
| code-js-ts | 942.4 → 2086.4 | 2.21× | 1475.1 → 2273.0 | 1.54× |
| code-python | 407.0 → 928.2 | 2.28× | 917.9 → 1442.8 | 1.57× |
| code-rust | 578.6 → 1451.8 | 2.51× | 802.8 → 1314.8 | 1.64× |
| en-dclm | 831.5 → 840.4 | 1.01× | 980.1 → 967.5 | 0.99× |
| en-fineweb-edu | 1078.3 → 1132.5 | 1.05× | 1145.2 → 1129.4 | 0.99× |

Diagnosis packs contain 500 documents each. Larger inputs are the first 5,000 documents of train part 00000 for each listed pack, streamed into fixed Parquet slices before either timing run. The JSON records source-part and slice hashes. They are selected slices, not a full-corpus run. Small changes on untouched English packs are timing variation.

## Verification

- **30/30 diagnosis packs:** identical input/output hashes and actions, zero errors, all repeats deterministic. This includes all 13 original TSAI-required packs and 17 supported Indic packs.
- **8/8 larger slices:** identical output hashes and actions, zero errors; 40,000 additional input documents. Total checked population: 55,000 documents, three passes per cleaner.
- **30/30 harm reports:** every result field identical except the cleaner hash; 3,926 own-pack gold documents. Both baseline and optimized harm runs exited 0. A temporary gold lookup combined Data A and Data B files without changing them.
- **Suite:** 2,090 passed, 1 skipped, 42 subtests passed. This includes 420 new differential/tool cases. Frozen references come from merged commit `9299ff2` and cover Unicode contact/name cases, key-length boundaries, entropy policy, parser boundaries, scope and randomized positive/negative documents.
- A fresh build reproduces the optimized `clean.py` exactly.

The deferred 10 November refresh config remains pinned to its recorded `14ce6134` candidate. Its test now verifies that config against the original smoke evidence; it is not a config for this newer cleaner.

## Documents changed by the two rules

| Pack | E3 fixed, 500 / 5,000 docs | Shared F1 fixed, 500 / 5,000 docs |
|---|---:|---:|
| code-cpp | 0 / 1 | 0 / 0 |
| code-go | 0 / 3 | 0 / 0 |
| code-java | 1 / 2 | 1 / 0 |
| code-js-ts | 2 / 7 | 0 / 0 |
| code-python | 0 / 6 | 2 / 0 |
| code-rust | 1 / 1 | 0 / 0 |

These are sequential per-rule impacts inside the full cleaner; earlier-dropped documents never reach these rules. Neither rule drops a document in these samples. The full call/fix/drop counts are in the JSON.

## Rule times and remaining bottlenecks

| Pack | Shared F1 ms/call, before → after | E3-004 ms/call, before → after | Slowest after optimization |
|---|---:|---:|---|
| code-cpp | 0.908 → 0.139 | 0.800 → 0.274 | DA-E3-004 25.6%, DA-E3-003 16.6%, f1_rule.py 13.0% |
| code-go | 0.577 → 0.075 | 0.527 → 0.181 | DA-C3-002 31.0%, DA-E3-004 17.5%, DA-E3-003 9.9% |
| code-java | 0.555 → 0.093 | 0.474 → 0.163 | DA-E3-004 21.1%, DA-E3-003 13.5%, f1_rule.py 11.9% |
| code-js-ts | 0.433 → 0.066 | 0.365 → 0.136 | DA-E3-004 26.1%, DA-E1-008 16.9%, DA-E3-003 15.7% |
| code-python | 1.046 → 0.166 | 0.917 → 0.365 | DA-E3-004 32.1%, DA-E3-003 17.4%, f1_rule.py 14.5% |
| code-rust | 0.716 → 0.106 | 0.605 → 0.184 | DA-E3-004 25.8%, DA-E3-003 18.8%, f1_rule.py 14.8% |

Shares above are of measured rule execution time; ms/call excludes documents an earlier rule dropped. A rule can remain the largest cost even after its absolute time falls.

## Reproduce the 13-pack acceptance check

From the repository root, with the pinned requirements installed and a directory containing exactly the 13 original diagnosis packs. Keep generated Parquet and scratch results outside committed evidence.

```bash
PY=python3 teams/01-data-a/oneoff/speed_compare.sh \
  --before 9299ff2 --after worktree \
  --packs /path/to/original-13-packs \
  --work "$PWD/out/code_speed_check" \
  --rules DA-E3-004,f1_rule.py \
  --profile-packs "code-cpp code-go code-java code-js-ts code-python code-rust"
```

Use a fresh scratch folder per run. The saved JSON preserves the wider 30-pack, larger-slice and own-pack-gold comparisons. TSAI’s separate FineWeb ~30,000 docs/s baseline discrepancy still needs its exact original input/runner; it is not resolved by these code-only changes.
