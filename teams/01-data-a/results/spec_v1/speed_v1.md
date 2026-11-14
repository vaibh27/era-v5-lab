# clean.py speed log

Code-pack follow-up measurements and checks are recorded in [speed_v1_code.md](speed_v1_code.md).

Spec v1.1, Gate 2. A log of the slow rules in `clean.py` that were sped up: for each rule, what the fix was, docs/s per core before and after, and the checks showing that nothing else changed. Last updated 12 Nov 2026. Reports counts, hashes and timings only, never document text.

Every fix passes three checks before it is logged:

1. **Output sha256 unchanged.** `clean.py` before and after gives the same output sha256 on all 13 sample packs. The sha256 covers every document in file order: its `doc_id`, the final action (keep, fix or drop) and the final cleaned text, so equal means byte-identical output.
2. **Harm tests unchanged.** Every rule and `clean()` on each pack gives the same harm result and the same gold changed/dropped counts.
3. **Pytest suite passes.**

## 1. Speed, docs/s per core, before and after each fix

500-doc sample packs, one core, `clean()` loop only. Each rule's before and after were timed back to back on the same machine (steps 1 to 3: best of 5 passes; steps 4 to 7: best of 3). The rule time is the per-document time of that rule alone, from the per-rule profile.

| # | rule (what it checks) | en-dclm docs/s | en-fineweb-edu docs/s | rule time en-dclm, ms/doc | rule time en-fineweb-edu, ms/doc |
|---|---|---|---|---|---|
| 1 | `DA-F1-005` (PII) | 47.9 to 250.0 (5.2x) | 94.7 to 333.4 (3.5x) | 17.284 to 0.435 | 7.653 to 0.338 |
| 2 | `DA-D3-003` (foreign-script lines) | 249.0 to 300.3 (1.2x) | 326.0 to 408.0 (1.3x) | 0.872 to 0.211 | 0.690 to 0.130 |
| 3 | `DA-C5-002` (keyword spam) | 300.3 to 341.6 (1.1x) | 408.0 to 461.3 (1.1x) | 0.664 to 0.272 | 0.513 to 0.202 |
| 4 | `DA-B4-010` (template and code-like text) | 352.4 to 397.2 (1.13x) | 481.2 to 546.2 (1.14x) | 0.509 to 0.194 | 0.383 to 0.139 |
| 5 | `DA-B4-008` (detokenised quotes) | 400.5 to 465.4 (1.16x) | 559.2 to 648.7 (1.16x) | 0.334 to 0.007 | 0.268 to 0.003 |
| 6 | `DA-B2-002` (HTML tags) | 467.1 to 545.8 (1.17x) | 662.5 to 769.9 (1.16x) | 0.270 to 0.003 | 0.205 to 0.003 |
| 7 | `DA-C3-003` (cut-off texts) | 548.4 to 617.3 (1.13x) | 762.6 to 833.8 (1.09x) | 0.226 to 0.002 | 0.109 to 0.001 |
| | **all seven** | **47.9 to 617.3 (12.9x)** | **94.7 to 833.8 (8.8x)** | | |

The "all seven" row is the baseline (first row's before) against the last row's after. The rows were timed in separate runs, so it is approximate. The code, math and science packs are not slower (1.0x; none of these rules run on them).

## 2. Status of each fix

| # | rule | output sha256 | harm tests | pytest | documents changed, 500 docs each: en-dclm / en-fineweb-edu / en-nemotroncc / en-nemotroncc-synth | `clean.py` after | commit |
|---|---|---|---|---|---|---|---|
| 1 | `DA-F1-005` | same, 13 of 13 | unchanged | 1138 passed | 2 / 0 / 1 / 2 fixed | `7a76c7eddb51` | `730253d` |
| 2 | `DA-D3-003` | same, 13 of 13 | unchanged | 1576 passed | 3 / 1 / 3 / 0 fixed | `fc6d1a2deea4` | `8a74553` |
| 3 | `DA-C5-002` | same, 13 of 13 | unchanged | 1596 passed | 4 / 0 / 0 / 0 dropped | `622b0d48c65f` | `2d03ebb` |
| 4 | `DA-B4-010` | same, 13 of 13 | unchanged | 1618 passed | 0 / 1 / 0 / 0 fixed | `742787d7c310` | `a187284` |
| 5 | `DA-B4-008` | same, 13 of 13 | unchanged | 1630 passed | 2 / 0 / 0 / 18 fixed | `08e8fe016458` | `04b68e6` |
| 6 | `DA-B2-002` | same, 13 of 13 | unchanged | 1648 passed | 1 / 1 / 0 / 0 fixed | `7836cb354f09` | `2a89f71` |
| 7 | `DA-C3-003` | same, 13 of 13 | unchanged | 1669 passed | 2 / 1 / 12 / 1 dropped | `11da63a27cca` | this commit |

The baseline is `clean.py` `97c1a55b89f4` (commit `730253d^`). Pytest counts are the whole suite at that commit (1 test skipped in each).

## 3. What each fix did

1. **`DA-F1-005`, PII (emails, phones, addresses).** The rule ran the ORGANISATION regex on a joined 13-line window for every line, even lines with no email, phone or address. It now runs ORGANISATION once per line, caches the result and builds the window from the cached results, and a line with no `@` or digit never reaches it.
2. **`DA-D3-003`, lines in a foreign script.** The rule looked up `unicodedata.name()` for every letter of every line. An ASCII line is never foreign (an ASCII letter is always Latin), so it is skipped, and the script of each distinct letter is looked up once and cached.
3. **`DA-C5-002`, keyword-spam pages.** The rule searched nine case-insensitive spam patterns on every document, and needs two different patterns to match before it drops. One search of the union of the nine now comes first, and the nine run one by one only when it matches; the keyword-line pattern runs only when the text has 6 or more `|`, which it needs. The rule drops few documents and is a hardcoded list of spam themes from en-dclm; it does no harm on the gold sets, so it was kept as it is (whether to keep it is the sheet author's and the team's decision).
4. **`DA-B4-010`, template expressions and code-like text.** The rule ran a large token regex on every line to decide whether more than half of the text is code, then searched the whole text for template expressions. It now stops reading lines as soon as the answer cannot change (a prose document is settled halfway through), and runs the template regex only when the text contains one of `{{`, `{%`, `${`, a backtick, `undefined`, `NaN` or `null`, which every alternative of that regex needs. It runs on nearly every document and almost never changes one, which is why a cheap pre-check pays off.
5. **`DA-B4-008`, detokenised opening quotes.** The rule is one regex that puts back the space in `said'sure it is'`. It can only match at an apostrophe, but it begins with look-behinds, so Python tried it at every character position. It now finds the apostrophes with `str.find` and asks the unchanged regex about each one in place (`DETOK.match(text, i)`), then rebuilds the text.
6. **`DA-B2-002`, HTML tags.** The rule removes HTML tags from a text that has two or more of them and is not about HTML. It ran a case-insensitive search for "html", "tag", "element", `class=`, a code fence and `<!DOCTYPE` on every document first (0.16 ms/doc), although only 3 of the 2,000 sample documents have two tags. It now counts `<` first (a tag holds exactly one, so fewer than two means fewer than two tags), then the tags, and asks the about-HTML question only about the few texts that have two or more.
7. **`DA-C3-003`, cut-off texts.** The rule drops a text that ends in a `[...]` marker or is cut off after a dangling word (the, and, of, because ...). A regex searched the whole text from the start for the last word, which was most of the rule's time. The last word is now read backwards from the end of the text, stopping once it is longer than any dangling word (7 letters), and the marker is searched for only when the text ends in `]`.

The tests for fixes 1 to 7 are in `tests/spec_v1/test_speed_equivalence.py`: each rule is compared with a frozen copy of its old code on hand-written edge cases and on random documents.

## 4. Next slowest rules

After fix 7, on en-dclm / en-fineweb-edu: `DA-F1-005` is still the slowest at 25.5% / 27.9% of the time (0.435 / 0.341 ms/doc), then `DA-C5-002` (16.2% / 16.3%), `DA-D3-003` (11.7% / 10.8%) and `DA-B4-010` (10.8% / 11.3%), all already sped up once. The rules not yet touched are each below 7%: `DA-A3-002`, `DA-B4-007` and `DA-D4-001` (about 7% each) and `DA-C2-003` (about 4.5%).

## 5. Method, and adding the next entry

- **Machine:** CPU, Python 3.12.15. Each timing run is pinned to one core (`taskset -c 0`), one run at a time. CPUs 0 and 1 are two threads of one physical core, so never time two runs on them together. Timings vary by about 5% between runs, so treat ratios under 1.1x as about equal.
- **Versions compared:** every row compares two `clean.py` from git; only `clean.py` differs. Step 2's before is `553fcfe32796` (commit `d956c6a`), which is fix 1 plus the Indic refresh merge (`3b91add`, Indic rule scopes only); it runs at the same speed as fix 1 and gives the same output sha256 on the 13 packs.
- **Rule time and documents changed:** `spec_v1/run_clean.py` records the seconds each rule takes; the documents changed are counted by wrapping each rule while `clean.py` runs, in pipeline order.
- **A discarded run:** the first timing of fix 5 gave 0.88x on two packs although the rule time had dropped 48x and the other two packs sped up in the same run; it was disturbed and the speed step was re-run.
- **Tools:** [speed_compare.sh](../../oneoff/speed_compare.sh) runs the whole check, [speed_check.py](../../oneoff/speed_check.py) hashes and times, [make_harm_tests.py](../../spec_v1/validate/make_harm_tests.py) runs the harm tests. The full routine is in [oneoff/speed_handoff.md](../../oneoff/speed_handoff.md).

To add a fix:

1. Edit the rule in `spec_v1/sheets/specv1.md` and `medicine_english.md`, rebuild with `python -I spec_v1/build/build_clean.py`, add tests to `test_speed_equivalence.py`.
2. Before committing, run `teams/01-data-a/oneoff/speed_compare.sh --rules <RULE_ID>`.  It checks the fresh build, the full pytest suite, the output sha256 on the 13 packs, the harm tests, the profile and the documents the rule changes, and ends with a SUMMARY. It must say `RESULT: PASS`, `output sha256: the same on 13 of 13 packs` and `harm: unchanged`; its SUMMARY has the figures for the tables.
3. Add a row to the tables in sections 1 and 2 and a line in section 3, and update section 4. Commit the fix and this log together; write `this commit` in the commit column, or fill in its id with the next commit.

To replay the timings of the fixes in order, from the repository root (about 3 to 5 minutes each). Each command builds its own before and after from git; only the `trees speed profile` steps are used, because `build` and `pytest` check your working copy, not the commits:

```
S=teams/01-data-a/oneoff/speed_compare.sh
$S --before 730253d^ --after 730253d --rules DA-F1-005 trees speed profile
$S --before d956c6a   --after 8a74553 --rules DA-D3-003 trees speed profile
$S --before 8a74553   --after 2d03ebb --rules DA-C5-002 trees speed profile
$S --before 2d03ebb   --after a187284 --rules DA-B4-010 trees speed profile
$S --before a187284   --after 04b68e6 --rules DA-B4-008 trees speed profile
$S --before 04b68e6   --after 2a89f71 --rules DA-B2-002 trees speed profile
$S --before 2a89f71   --after <fix 7 commit> --rules DA-C3-003 trees speed profile
```

- The scripts are run from your current checkout; the commits only supply `clean.py` and the tree it runs in.
- Absolute docs/s depend on the machine, so compare the ratios. The replay times best of 3 passes (the table's fixes 1 to 3 used best of 5), so expect differences of a few percent.
- The input is the 13 packs in `week1-packs` (`<pack>.parquet`, use `--packs` if yours are elsewhere). The output sha256 values in section 6 show you have the same packs: the SUMMARY of each run must say `output sha256: the same on 13 of 13 packs`, and the `clean.py` line must show the shas of the status table. To check a `clean.py` by hand: `git show <commit>:teams/01-data-a/spec_v1/clean.py | sha256sum` (first 12 characters).

## 6. Output sha256 of the 13 sample packs (identical before and after every fix)

| pack | output sha256 (first 16) | pack | output sha256 (first 16) |
|---|---|---|---|
| code-cpp | `86df31e9a4b16abb` | en-nemotroncc | `f111b95a31e6d718` |
| code-go | `6c2c667bdbd4e89e` | en-nemotroncc-synth | `3dc350e529a4139d` |
| code-java | `0ae70771c29dbe0b` | math-finemath | `4e79a08fd226f614` |
| code-js-ts | `f4437f3607e8ddbe` | math-megamath | `51c5b8452f45d195` |
| code-python | `b0e431e0431423eb` | sci-finepdfs | `f67b3fa5b7a92b53` |
| code-rust | `ad6141d053488d1c` | en-dclm | `3fdbc3056421bbe7` |
| | | en-fineweb-edu | `4c8be607c9590b8e` |

## 7. Notes

- **TSAI's figures:** en-dclm about 45 docs/s matches what we measured before the fixes. The en-fineweb-edu figure of about 30,000 docs/s does not: we measure 94.7 docs/s on the same pack before the fixes. How it was obtained is unknown (a different sample, or rules not running); it should be checked with TSAI.
- **Indic packs** are not measured here (the 13 packs exclude `ind-mr-sangraha-v`). The Indic rules are being sped up separately.
- **Harm tests:** for fixes 1 and 2 together, `harm_tests.md` has extra Indic lines for `ind-hi-sangraha-s` (not on this machine); they come from the Indic refresh in the "after" file, not from the fixes. Fixes 1 and 2 were checked together for the harm tests.

## 8. Code-pack follow-up after merging PR #34

The [code-pack report](speed_v1_code.md) compares merged baseline `9299ff2`
with the code-only secret-masking and shared-F1 fast paths. Across six code packs,
the 500-document diagnosis checks improve **1.79–2.51×** and the 5,000-document
raw sample slices improve **1.32–1.64×** on one core, best of three passes.
All 30 supported diagnosis packs and eight larger slices retain identical outputs
with zero errors. All 30 harm reports are unchanged on 3,926 own-pack gold documents;
2,090 tests and 42 subtests pass, with one test skipped. Full hashes, repetitions
and final bottlenecks are in [speed_v1_code.json](speed_v1_code.json).
