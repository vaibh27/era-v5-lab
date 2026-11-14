# Spec v1: structure, requirements and how to run it

Status: **DRAFT / CANDIDATE.** This is the current layout and review policy for the five-family cleaner. The [consolidated spec](sheets/specv1.md) records executable and deferred selections; inclusion does not grant production approval. Family sources and review decisions are indexed in [sheets/README.md](sheets/README.md).

## Layout

```
teams/01-data-a/
  README.md                 pipeline and navigation
  harmtest/                 harm-test harness and medicine-review tools
  tools/                    shared gold-building and viewer tools
  tests/                    tests for harmtest/ and tools/
  gold/                     own-pack gold sets, coverage, and removal provenance
  rules/                    historical contributor registry and viewer snapshot
  jobs/                     request, log, and execution-result index
  spec_v1/
    README.md               this file
    sheets/                 five family medicine sheets, decisions, science review
      specv1.md             reviewed consolidated input to build/build_clean.py
    clean.py                generated cleaner with its scope table
    README_clean.md         how to build, test and run clean.py, step by step
    run_clean.py            runs the cleaner on one pack, or on one part of a pack with --pack-id
    run_tokenizer_sample.py runs the cleaner on the tokenizer sample, for a TSAI job: see ../jobs/README.md
    requirements.txt        the pinned packages, for pip install -r
    build/                  sheets to clean.py: build_clean.py, f1_rule.py, Indic integration
    validate/               checks on the sheets and on clean.py: harm tests, Indic and F2 validation
    normalize.py, f2_rule.py   the normaliser (shared with G9) and the F2 rule, next to clean.py
    ../tests/spec_v1/       cleaner tests on made-up values
    ../results/spec_v1/README.md   report producers, snapshot hashes, reproduction limits
    ../results/spec_v1/cleaner/harm_tests.md   each rule on the gold sets of its packs, each clean.py function, and clean()
    ../results/spec_v1/cleaner/specv1_<pack>_harm.json   the reports behind harm_tests.md: IDs and digests only
    ../results/spec_v1/sheets/<pack>_harm.json   harm_test.py --report on the sheet: IDs and digests only
    ../results/spec_v1/sheets/stress_<pack>_harm.json   the same, with disputed docs left in
    ../results/spec_v1/sheets/evidence.json, inputs_manifest.json   every medicine on every pack; digest of every input
    ../oneoff/              the scripts that made the code sheet, kept for the record: see ../oneoff/README.md
```

Families: `code`, `english`, `math`, `sci`, and `indic`. Indic covers 18 packs through native candidates and validated same-language sibling scopes; see [Indic integration](README_indic.md) for deferred rules and validation limits. The ready Indic intake is pinned to PR #240 at `3bbd31f`; retired local cards remain separate, non-executable history. The [post-#362 validation packet](../results/spec_v1/cleaner/pr244_gold_refresh/README.md) records the refreshed gold checks, Hindi sangraha-v registration and Marathi receiving-scope correction.

**Source flow:** family `sheets/medicine_<family>.md` → reviewed consolidated `sheets/specv1.md` → `build/build_clean.py` → `clean.py`. The builder reads only the consolidated spec: it does not automatically import family-sheet edits. Indic has a separate [integration and sibling-scope review](README_indic.md); the original English compilation tooling remains an external reproduction dependency. Numbers and qualification claims must identify the hashes of their evidence in [results](../results/spec_v1/README.md).

The shared [run_harm_suite.py](../harmtest/run_harm_suite.py) is a legacy registry-inventory runner. Its old `spec/spec_v1.md` input and builder are absent; its parser does not accept the current consolidated sheet. See [registry prerequisites](../rules/README.md). Use the tools below for the current cleaner.

## What every rule must have

A block in the sheet with these fields, and a `medicine(doc) -> (action, text)`. `harm_test.py --rule all` and `medicine_conflicts.py --sheet` already read this form.

| field | meaning |
|---|---|
| `RULE_ID` | Stable `DA-<code>-<nnn>` or `DB-<code>-<nnn>`. Never reused. |
| `CODE` | Disease code from `DISEASES.md`. |
| `ACTION` | `fix`, `drop`, `fix+drop`, or `keep` for a label-only rule. Indic D3 always keeps the original text. |
| `STAGE` | Stage below. The run order is section order. |
| `STATUS` | `CANDIDATE`, `APPROVED`, `FAILED`, `BLOCKED` or `DEFERRED`. |
| `LANGS` | Language tags it runs on. A row outside passes unchanged, except a code row whose `pack_id` is in `ORIGIN`: some code packs label every doc `code`. |
| `ORIGIN` | Packs it runs on, comma separated. A code or math rule reads it, and its scope is the packs in both `SPEC_PACKS` and `ORIGIN`. |
| `SOURCE` | Pack whose medicine it came from. |
| `AUTHOR` | Code field for `medicine_conflicts.py`. Not used in any text. |

Around the block: disease with counts and `doc_id` examples, medicine in words, action, harm test per pack, side effects. Examples give `doc_id` and offset only. E3 and F1 give type and offset only, never the value.

## Stages

| stage | what | why here |
|---|---|---|
| 00 guard | `LANGS` and the pack scope. | A rule must not touch a language it was not tested on. The language tag is taken as correct, except in code packs: a code row whose pack is in `ORIGIN` is taken even when its tag is `code`. |
| 10 characters | A1, A2, A4, A6. | Later patterns need clean characters. |
| 20 markup | B3. | Later rules expect the file to start at its real first line. |
| 30 source | Drops first (E1, E2, C3, C4), then strips (E4, C1). | A generated file with its banner inside the licence survives if the licence goes first. |
| 40 data safety | B4, E3, and F1 if approved. | A6 can turn an obfuscated address into plain text, so this runs after it. |
| 90 post-clean | A test, not a rule: no empty docs, no doc left with no code. | Catches what the gold test cannot. |

## When a rule is APPROVED

All of these, on every pack in `ORIGIN`. Anything else stays `CANDIDATE`, or `BLOCKED` when it waits on an answer.

1. **Within the limit,** alone and in the whole sheet: changed + dropped at most `max(1, floor(1% of gold))`. An empty gold set is NOT TESTABLE, never PASS. This is the limit in `harm_test.py`; §2.5 of the template still says 1.0% and needs updating.
2. **Treats real rows:** labelled rows treated out of labelled rows, per pack. A rule that does nothing also passes the limit.
3. **Does not damage what it keeps:** comment rules leave code unchanged, which is checked with comments stripped. Gold cannot show this. `DA-E4-004` passed gold and deleted `#include`.
4. **Idempotent:** a second run changes nothing.
5. **Not resting on a dispute:** it still passes with the disputed docs left in, or the reason is written down. A reader's own dispute must not be what makes their rule pass.
6. **Reviewed** by someone other than its author, with commit and digests recorded.

## Assumptions

- **Language tags are correct.** No detector runs. D1 rows wait for a language check at scale.
- **F2 is G9's fingerprint check** (`f2_rule.py`, see "F2" below). The word "leetcode" alone is not a leak (instructor, 30 Oct).
- **H3 has no rule.** A vendored copy is not a disease by itself. The same file twice in a pack is H1 or H2 on the second copy (instructor, 30 Oct).
- **A 0 result is a bound, not proof.** With n gold docs and 0 hits, the harm rate is only shown to be below about 3/n. Show this next to the result. About 300 docs are needed to show 1%.

## What to report per pack, never pooled

Gold size, limit, changed, dropped, affected IDs, pack fixed / dropped / untouched, labelled rows treated, and the stress result with disputed docs left in. Each pack passes or fails on its own.

## Run the harm test on a sheet

```
python -I teams/01-data-a/harmtest/harm_test.py --pack <pack>.parquet --gold teams/01-data-a/gold/<pack>_gold.csv \
  --rules teams/01-data-a/spec_v1/sheets/medicine_code.md --rule all --report teams/01-data-a/results/spec_v1/sheets/<pack>_harm.json
python -I teams/01-data-a/harmtest/medicine_conflicts.py --packs code- --sheet teams/01-data-a/spec_v1/sheets/medicine_code.md --out <file>
```

These tools are available in [../harmtest](../harmtest/README.md) and [../tools](../tools/README.md). Some committed code-sheet evidence was produced using the historical `bae5163` checkout; its reproduction steps remain in [oneoff/README.md](../oneoff/README.md). Stage the original diagnosis packs and matching gold before rerunning.

## Review and reproduction boundaries

| Area | Current behavior / outstanding work |
|---|---|
| Executable source | The reviewed consolidated sheet drives builds; the former draft-inventory builder is not shipped here. |
| Combined validation | `validate_specv1.py` checks the executable sheet; `make_harm_tests.py` checks generated cleaner rules/functions/pipelines. The legacy registry runner does neither. |
| Evidence | A historical gold pass applies to its recorded code/input hashes. Missing inputs are not a current pass. |
| Approval | Follow each family's review decisions and deferred status; a successful build or unit test does not freeze or approve rules. |

## The cleaner

`clean.py` is the cleaner. `build/build_clean.py` generates it from `sheets/specv1.md`, which consolidates the five family sheets, so **add rules to a sheet, never to `clean.py`**. Update the reviewed consolidated input through the appropriate family process before rebuilding.

A rule names the packs it applies to (instructor, 31 Oct): `SPEC_PACKS` in its block, and for a code or math rule its `ORIGIN` packs too. The build puts those pack IDs in one table, `SCOPE`, at the top of `clean.py`, and each rule reads its entry there. A document goes to the family of its `pack_id` and runs that family's rules in sheet order. A document with a missing or unknown `pack_id` passes unchanged and is not eligible for release. `run_clean.py` takes `pack_id` from the file name when a pack has no such column. `make_harm_tests.py` tests each rule only on the gold sets of the packs in its `SCOPE` entry. It also runs every rule that does not apply to a pack on that whole pack, where it must change nothing.

| where | what | who changes it |
|---|---|---|
| `sheets/medicine_<family>.md` | The rules of one family: `code`, `english`, `math`, `sci`, `indic` | The PR for that family |
| `sheets/specv1.md` | Five families consolidated, with selected/deferred status and scope | The pipeline owner / reviewed integration step |
| `sheets/decisions_<family>.md` or family review report | Why each rule was picked or not | The same PR |
| `build/f1_rule.py` | The shared F1 rule; `SCOPE` runs it on the code, math, science and Indic packs | A PR that names the family it adds |
| `clean.py`, `results/spec_v1/cleaner/harm_tests.md`, `results/spec_v1/cleaner/specv1_*` | Generated | Only the rebuild step, after a merge |

## Run it

Step by step, with every option, output and error: [README_clean.md](README_clean.md).

```
python3 -m pip install -r spec_v1/requirements.txt
python spec_v1/build/build_clean.py
python -m pytest tests/spec_v1
python spec_v1/run_clean.py --pack <pack>.parquet --out out
python spec_v1/run_clean.py --pack <split>/<pack_id>/part-00000.parquet --pack-id <pack_id> --out out
python spec_v1/run_clean.py --pack <pack>.parquet --out out --f2      # then F2, G9's benchmark fingerprints
python spec_v1/validate/make_harm_tests.py ../.. <folder with the packs> [<another folder with packs> ...]
```

Paths start at `teams/01-data-a/`; `../..` points to the repository root for the shared harm runner. `make_harm_tests.py` lists unavailable packs/gold as not run and currently looks for gold in Data A's `gold/`. Use [Indic validation](README_indic.md#rebuild-and-validate) with Data B's own-pack gold for Indic coverage. `requirements.txt` pins the dependencies (Python 3.10 or later). The second `run_clean.py` line cleans a tokenizer sample part, whose filename is not a pack ID. Original packs must be staged separately; week-2 shards cannot replace diagnosis packs for harm tests.

## Add your medicines without double work

1. **Check first.** Look at the open PRs to `g1-specv1`. If a PR already covers your family and code, comment there instead of starting again.
2. **Claim it.** Open a draft PR to `g1-specv1` early, titled `medicine_<family>: <codes>`. Only one open PR per family sheet at a time, so sheets do not conflict.
3. **Put each rule in the existing family sheet under `sheets/`,** in the form of `medicine_code.md`, at the place its stage says. Use the [sheet index](sheets/README.md) for all five families and their review decisions. Indic changes also require the explicit integration/sibling-scope process.
4. **Resolve conflicts before you ask for review.** Run `medicine_conflicts.py --packs <family prefix> --sheet <your sheet>`, then for each code with more than one rule:

   | what you find | what to do |
   |---|---|
   | Same text on every doc | Keep one. Say so in the decisions file. |
   | They fix different rows | Keep both, as two blocks. |
   | They disagree on the same rows | Keep the one that treats the most labelled rows within the limit. On a tie, read the code. |
   | One passes only because its supporter's dispute took docs out of gold | Hold it as `BLOCKED`. |
   | It needs an answer from TSAI | Hold it as `BLOCKED`, with the question. |

5. **Do not edit someone else's rule to change what it does.** Add a new block and say why in the decisions file.
6. **In your PR, change only your sheet, your decisions file and your tests.** Leave `clean.py` and everything in `results/spec_v1/cleaner/` out: two PRs regenerating them will always conflict. Paste your harm results in the PR text.
7. **After a merge,** whoever merges reruns the steps in "Run it" and commits the generated files.

Each rule needs the fields in "What every rule must have" above, and has to meet the bar in "When a rule is APPROVED".

**Rule IDs:** the consolidated builder requires unique IDs across its complete inventory, including deferred cards. Preserve the source `DA-` / `DB-` identity; do not infer a link to the historical contributor registry merely from a matching ID.

## Regex: which diseases, and how

| use | diseases | how |
|---|---|---|
| Regex alone is enough | A1, A2, A4, A5, A6, B1, B3 | A fixed character or token. Replace only that. |
| Regex, with care | B2, B4, E1, E2, E4, C2 | Anchor it to the header or to a known marker. Check it does not touch valid code or markup. |
| Regex plus tree-sitter, in code | F1, and anything that should only touch comments or strings | The regex finds the value and the tree says where it sits. Leave it alone if it is in code. |
| Regex finds some, not all | C1, C3, C5, E3 | Phrase lists and shapes fit the sites seen so far. Report what is left. |
| Library | A3 | `ftfy`, as the English sheet does. |
| Regex does not work | C4, D1, D3, D4, D5, F3, H1, H2, H3 | Label it. These need a whole-pack check, a language model or a classifier. |
| G9 fingerprints | F2 | `f2_rule.py`: G9's check, not a regex. See "F2" below. |

Try a pattern before you write the rule. This prints counts and doc_ids, never text:

```
python tools/regex_check.py --pack <pack>.parquet --gold <gold>.csv --labels <diagnosis/submissions/pack> --code E4 --regex "(?i)copyright"
python tools/regex_check.py --pack <pack>.parquet --gold <gold>.csv --labels <diagnosis/submissions/pack> --code E3 --regex "AKIA" --where string
```

Then:
1. **Read the misses, not only the hits.** A rule can match every labelled row and still delete real code. `DA-E4-004` passed gold and removed `#include`.
2. **Keep the match narrow.** Anchor it, set `LANGS`, and change only the bad part.
3. **Check it is idempotent.** A second run must change nothing.
4. **Add a unit test** with made-up values in `tests/spec_v1/` (or the topic's folder under `tests/`): one sick case and one look-alike that must stay.
5. **Run the harm test.** The limit is `max(1 document, 1%)` of the pack's gold, per pack, never pooled. An empty gold set is not testable.

## F1

`build/f1_rule.py` follows the TSAI decision: `<PHONE>`, `<EMAIL>` and `<ADDRESS>` for a private person, drop a personal listing, keep organisations and public offices.

- **Where it runs:** on the code packs, with tree-sitter, and on the math and science packs, with the prose path (`_F1_PROSE`: `math-*` with `en`, `sci-*` with `eng_Latn`), as TSAI approved on 1 Nov. `SHARED_F1_FAMILIES` in `build/build_clean.py` puts those packs in its `SCOPE` entry; on science it runs after `DA-F1-006`, so the reviewed drops stay. English keeps its own F1, `DA-F1-005`, and never gets the shared rule. The prose path is keyed by `pack_id` because English shares the `en` tag; a doc with no `pack_id` never takes it. To add a family, add it to `SHARED_F1_FAMILIES` and to `_F1_LANGS` (code) or `_F1_PROSE` (prose), run the harm test, and put the results in your PR. On the Indic packs (`ind-*`, any language tag) it takes the Indic path: a mobile number (+91 / 0 and ten digits from 6-9, in Latin or any Indic digits) becomes `<PHONE>` unless its block names an organisation or public office (English or Indian-language words); a landline only with a private label (`Home:`, `घर:`, `मोबाइल`); toll-free 1800/1860 numbers stay; an email becomes `<EMAIL>` unless it is a role mailbox or an Indian organisation domain (`.gov.in`, `.nic.in`, `.ac.in`, ...). It replaces Data B's source F1 rules, which are deferred. It does not treat street addresses on Indic packs.
- **Prose path:** a value is private when a private label sits on it (`Home:`, `Cell:`, `Daytime Phone`), when it is in a person's contact card (each field on its own line, two personal fields or more), or in a `| Name | Telephone |` table. It is kept when its block or the heading above names an organisation or public office, and for toll-free numbers, role mailboxes and example domains. A page mostly of `SURNAME, Forenames, age, address` entries is dropped. It misses a private phone in running prose with no label, non-North-American numbers outside people tables, and addresses that are not "number + name + street type".
- **Never write a private value** in a commit, a PR or chat. Use doc_ids and offsets, as `regex_check.py` does.
- **Known limits:** it misses lower-case addresses, and it reads "8 Bit Drive" as an address. An email at a company domain counts as personal unless it is a role address like `support@`. Names are not in the TSAI list, so it leaves them.
- **B4:** in `clean.py`, the code B4 rule leaves `<EMAIL>` alone, because it is now the F1 tag, and still replaces `<NAME>`, `<KEY>` and `<PASSWORD>`. Before this, a second run turned the tag into `user@example.com`. `build/build_clean.py` makes this change, and `medicine_code.md` still has the old rule until the team agrees. 26 docs labelled B4 hold only `<EMAIL>`, so they keep the token.

## F2

`f2_rule.py` is G9's decontamination check (`teams/09-evaluation/decontam`, PR 31) as a rule: a doc that shares one run of words (13 words, 20 tokens for code) with an item of G9's fingerprint set, or holds a benchmark canary, is dropped. It uses G9's code and set unchanged.

- **Where it runs:** `run_clean.py --f2`, on every pack, after `clean.py`, on the text `clean.py` keeps. It is opt-in: without `--f2` nothing changes, and the tokenizer sample job does not use it. It is not in `clean.py`: the full set needs about 1.1 GB and 20 s to load. The summary gets `f2`: the `set_sha256`, and each dropped `doc_id` with its benchmark items and offset; and `final`: kept, dropped by `clean.py`, dropped by F2 and errors, which add up to the input rows. `actions` stays `clean.py`'s.
- **Harm test:** `python -I harmtest/harm_test.py ... --rules spec_v1/f2_rule.py --rule F2 --timeout 300`, and `clean.py` then F2 on the same gold sets. Current full-set validation covers all 31 supported packs: [v2 migration](../results/spec_v1/f2/README.md#current-v2-pin-and-validation). Earlier 13-pack results and their drop reviews remain historical evidence in [results/spec_v1/f2/](../results/spec_v1/f2/).
- **The set is pinned:** `SET` and `SET_SHA256` in `f2_rule.py`. A missing set, or one with another `set_sha256`, stops the run before anything is written. `F2_FINGERPRINTS` can point to a copy of the same set. The current pin is **v2**, set SHA-256 `2826a3b883210c653979103c09ce1c79e40fc4f7034b54027f2836482370d22a`, using G9's `g9-v2` normalizer. See the [migration evidence](../results/spec_v1/f2/README.md#current-v2-pin-and-validation). Future sets need their own harm validation before changing both constants.
- **Known limits (G9's README):** it finds copies only. A reworded, translated or very short item is not found, nor a benchmark outside the set (`not_covered` in G9's `MANIFEST.json`). A flag can be shared public text that a benchmark also quotes.
