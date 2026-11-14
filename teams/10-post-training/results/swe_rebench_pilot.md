# SWE-rebench pilot: admission results

Five environments rebuilt from the upstream PRs behind rows of `nebius/SWE-rebench` at revision `89cdfbab4ab1bd8f5a658bb212d1b63624f4f881`, split `test` (CC-BY-4.0; attribution in `licences/licence_table.md`). Written by `swe-envs/rebench_pilot.py --report` from each `rebench-pilot/<id>/admission.json`.

Each side ran in a fresh `docker run --rm` container with `--network=none --memory=2g --cpus=2 --pids-limit=256 --cap-drop=ALL --security-opt=no-new-privileges`. `before` = base commit + hidden tests (no patch); `after` = base + upstream fix + hidden tests. A run counts only if every listed test is reported, there is no collection error and verify.py's report-integrity checks pass. The negative control is a comment-only change to the first file the fix touches.

| instance_id | base commit | fix (PR head / merge) | platform | no patch fails | gold passes | negative reward | gold reward | isolation | max run (s) | admitted |
|---|---|---|---|---|---|---|---|---|---|---|
| `andialbrecht__sqlparse-676` | `9d2cb6fc9503` | `38d42107e2eb` / `3d3df9dcfb68` | linux/arm64 | 5/5 | 5/5 | 0 | 1 | pass | 0.6 | yes |
| `andialbrecht__sqlparse-676` | `9d2cb6fc9503` | `38d42107e2eb` / `3d3df9dcfb68` | linux/amd64 | 5/5 | 5/5 | 0 | 1 | pass | 1.1 | yes |
| `astanin__python-tabulate-139` | `b2c26bcb70e4` | `46e985e2dbf9` / `adc5630c8cd4` | linux/arm64 | 5/5 | 5/5 | 0 | 1 | pass | 0.9 | yes |
| `astanin__python-tabulate-139` | `b2c26bcb70e4` | `46e985e2dbf9` / `adc5630c8cd4` | linux/amd64 | 5/5 | 5/5 | 0 | 1 | pass | 1.1 | yes |
| `h2non__filetype.py-135` | `3bd71d081207` | `0828418ff3b2` / `49b0c5c9c2d0` | linux/arm64 | 5/5 | 5/5 | 0 | 1 | pass | 0.5 | yes |
| `h2non__filetype.py-135` | `3bd71d081207` | `0828418ff3b2` / `49b0c5c9c2d0` | linux/amd64 | 5/5 | 5/5 | 0 | 1 | pass | 1.6 | yes |
| `keis__base58-46` | `941894563948` | `2d6554263fee` / `67af5fef898f` | linux/arm64 | 5/5 | 5/5 | 0 | 1 | pass | 0.5 | yes |
| `keis__base58-46` | `941894563948` | `2d6554263fee` / `67af5fef898f` | linux/amd64 | 5/5 | 5/5 | 0 | 1 | pass | 2.5 | yes |
| `lark-parser__lark-1120` | `5772e1878729` | `75d65d72416a` / `0e9edd1453b6` | linux/arm64 | 5/5 | 5/5 | 0 | 1 | pass | 0.6 | yes |
| `lark-parser__lark-1120` | `5772e1878729` | `75d65d72416a` / `0e9edd1453b6` | linux/amd64 | 5/5 | 5/5 | 0 | 1 | pass | 1.8 | yes |

## Source record versus rebuild

| instance_id | issue opened | PR merged | licence | same base | same FAIL_TO_PASS | in-file PASS_TO_PASS ids, cut at the first space (source / rebuild) | variant |
|---|---|---|---|---|---|---|---|
| `andialbrecht__sqlparse-676` | 2020-06-25 | 2022-08-16 | BSD-3-Clause | True | True | 47 / 47 | same base, FAIL_TO_PASS and in-file PASS_TO_PASS as the source row |
| `astanin__python-tabulate-139` | 2021-08-19 | 2022-06-22 | MIT | True | True | 107 / 101 | G10 rebuilt variant: 6 in-file PASS_TO_PASS tests of the source row are not PASS_TO_PASS in the rebuild (they fail on both sides here, see env.json build_notes) |
| `h2non__filetype.py-135` | 2022-07-21 | 2022-08-06 | MIT | False | True | 17 / 17 | G10 rebuilt variant: base is merge-base(PR base, PR head) `3bd71d081207`, source row has `436ade4e8d70` |
| `keis__base58-46` | 2020-01-13 | 2020-01-14 | MIT | True | True | 15 / 15 | same base, FAIL_TO_PASS and in-file PASS_TO_PASS as the source row |
| `lark-parser__lark-1120` | 2022-02-26 | 2022-02-27 | MIT | True | True | 6 / 6 | same base, FAIL_TO_PASS and in-file PASS_TO_PASS as the source row |

## Rejected candidates

| candidate | stage | reason |
|---|---|---|
| `jaraco__inflect-138` | discovery (GitHub) | GitHub reports the repo licence as None; not pursued for the pilot (needs a manual LICENSE read before any build) |
| `kjd__idna-112` | discovery (GitHub) | the PR closes no GitHub issue, so there is no human issue text for the problem statement |
| `ahawker__ulid-455` | build | dependency footprint: the repo's requirements/test.txt pins mypy 0.780, which pulls typed-ast 1.4.1; it has no linux/arm64 wheel for Python 3.8 and the lock build failed without a C compiler |
| `kjd__idna-83` | discovery (GitHub) | the PR closes no GitHub issue, so there is no human issue text for the problem statement |
| `jd__tenacity-179` | build | build configuration: pbr with a flat layout (packages 'tenacity' and 'releasenotes'); setuptools 61+ refuses the automatic package discovery and the editable install needs setuptools 64+, so there is no build without changing upstream files |
| `jaraco__inflect-137` | discovery (GitHub) | GitHub reports the repo licence as None; not pursued for the pilot (needs a manual LICENSE read before any build) |
| `kiorky__croniter-13` | discovery (GitHub) | the PR closes no GitHub issue, so there is no human issue text for the problem statement |
| `astanin__python-tabulate-21` | discovery (GitHub) | the PR closes no GitHub issue, so there is no human issue text for the problem statement |
| `jd__tenacity-201` | discovery (GitHub) | passed the discovery checks; not built because the pilot stops at 5 admitted environments |
| `jaraco__zipp-60` | discovery (GitHub) | GitHub reports the repo licence as None; not pursued for the pilot (needs a manual LICENSE read before any build) |
| `Turbo87__utm-62` | build passed, held | built (FAIL_TO_PASS 1, PASS_TO_PASS 17), but gold.patch carries AUTHORS context lines with an email address; held for the same reason as dgilland__pydash-164 |
| `eliben__pycparser-346` | discovery (GitHub) | GitHub reports the repo licence as 'NOASSERTION'; not pursued for the pilot (needs a manual LICENSE read before any build) |
| `alecthomas__voluptuous-380` | discovery (GitHub) | the PR closes no GitHub issue, so there is no human issue text for the problem statement |
| `Infinidat__munch-54` | discovery (GitHub) | the PR closes no GitHub issue, so there is no human issue text for the problem statement |
| `lark-parser__lark-1013` | discovery (GitHub) | passed the discovery checks; not built because the pilot stops at 5 admitted environments |
| `coady__multimethod-26` | discovery (GitHub) | GitHub reports the repo licence as 'NOASSERTION'; not pursued for the pilot (needs a manual LICENSE read before any build) |
| `dgilland__pydash-164` | build passed, held | built (FAIL_TO_PASS 1, PASS_TO_PASS 255), but gold.patch carries AUTHORS.rst context lines with an email address and real names; the lab rules allow GitHub handles only and pr-guard rejects email addresses. The upstream patch is not edited, so the env is held |
| `kvesteri__intervals-54` | discovery (GitHub) | passed the discovery checks; not built because the pilot stops at 5 admitted environments |
| `carpedm20__emoji-128` | discovery (GitHub) | GitHub reports the repo licence as 'NOASSERTION'; not pursued for the pilot (needs a manual LICENSE read before any build) |
| `SethMMorton__natsort-146` | build | tests/conftest.py imports hypothesis, which no requirements file of the repo lists; pytest exits with code 4 before any test runs. Adding the dependency by hand would be a guess, not the repo's own configuration |
| `cdown__srt-53` | discovery (GitHub) | the PR closes no GitHub issue, so there is no human issue text for the problem statement |
| `devsnd__tinytag-156` | discovery (GitHub) | passed the discovery checks; not built because the pilot stops at 5 admitted environments |
| `eliben__pycparser-364` | discovery (GitHub) | the PR closes no GitHub issue, so there is no human issue text for the problem statement |
