# Data A harm test: medicine checks

> **Formerly in `tools/`.** The harm-test harness moved here from `teams/01-data-a/tools/`: `harm_test.py`, `medicine_extract.py`, `medicine_conflicts.py`, `rule_report.py`, `run_harm_suite.py` and `registry_view.py`. Use the `harmtest/` paths. `tools/harm_test.py` is a forwarder kept until Week 4, so old commands still run. Tests: `tests/harmtest/`.

Current cleaner operation is documented in [../spec_v1/README_clean.md](../spec_v1/README_clean.md). This guide also covers the older contributor-registry review workflow. Its source prerequisites and preserved viewer are described in [../rules/README.md](../rules/README.md); registry snapshots alone cannot run rules whose extracted files are absent.

## For annotators: test your own medicine.md

Before you commit your `medicine.md`, check that each rule changes at most 1% of the clean (gold) documents of your pack, and never less than 1 document: the limit is `max(1, floor(0.01 × n))` documents, where n is the number of rows in your pack's gold set. For every gold set under 200 rows this is 1 document.

**You need**
- Your pack's Parquet file (the one you annotated) and its gold file, `teams/01-data-a/gold/<pack_id>_gold.csv` (ask the Data A team if it is not there yet).
- Your `medicine.md`.
- Python 3.10 or newer and `pip install pyarrow`. Linux, macOS and Windows work. Each rule call on a document has a time limit (`--timeout`, default 5 s). On Linux and macOS a call over the limit is reported as an error for that document. Windows has no SIGALRM, so there a watchdog ends the whole test (exit code 1) and prints where the rule was stuck.

**Run** (from the repo root; `B3` is the disease code of the rule you want to test)

```
python -I teams/01-data-a/harmtest/harm_test.py \
  --pack <your_pack>.parquet \
  --gold teams/01-data-a/gold/<pack_id>_gold.csv \
  --rules diagnosis/submissions/<pack_id>/<your-handle>/medicine.md \
  --rule B3
```

Use `--rule all` to test every section of your file, and then all of them together in file order. You do not need a rule ID.

**Read the result**
- `documents changed` and `documents dropped`: how many gold documents your rule touched. Together they must be at most the number on the `limit:` line, `max(1, floor(1% of the gold set))`. The last line is `result: PASS` or `result: FAIL`. If your pack's gold set is empty, the last line is `result: NOT TESTABLE` and the rule cannot be tested yet.
- If it fails, add `--show 10` to list the first 10 `doc_id`s, or `--debug` to list every document that was changed, dropped or crashed, with the length and the first offset that differs. Your fix then needs to leave those documents alone. Text is never printed.
- Exit code: 0 PASS, 1 FAIL, 2 the test could not run, or the gold set is empty (the message says why).

**Your `medicine.md` must have**
- One `**Disease:** <code>` section for each disease code.
- In it, a ```` ```python ```` block with one function that takes the text and returns the new text. The name `medicine_<code>` (for example `medicine_b3`) is a suggestion, not a requirement: you can keep the names you already have. The function is found by that name (or `cure_<code>`), or else it is the only function in the section whose name does not start with `_`. So if the section has other functions, start their names with `_`. `re` is already imported.
- Every python block must run on its own. Remove or complete a snippet that does not.
- The same name must not be defined with different code in two blocks.

**If the test stops with an error**

| Message | What to do |
|---|---|
| `2 sections for B3 ...` | Keep one section for the code. Put both fixes in one function, using helpers. |
| `block N is a fragment ...` or `has a syntax error` | Complete the block or remove it from the file. |
| `no rule for B3`, or `no rule could be matched` | The section has no function, or has several whose names do not start with `_` and none is named `medicine_b3`. Rename the helpers with a leading `_`, or name the main function `medicine_b3`. |
| `_x is defined with different code ...` | Rename one of the two helpers. |
| `no '**Disease:** <code>' line found` | Add the `**Disease:**` line to each section. |

The test runs the code in your `medicine.md`. Do not run it on files you have not read.

The rest of this file is for the Data A team.

---

Tools to test the cleaning rules (medicines) from the annotators' `medicine.md` files. Steps 1 to 3 need `pip install pyarrow`. Run them from the repo root with `python -I`.

## The scripts

Run them in this order. Steps 2 and 3 can use the rules and IDs that step 1 makes; step 2 can also load a rules file directly. Step 4 audits registered rules against the draft parent Spec v1 and refreshed gold sets.

| Step | Script | Input | Output |
|---|---|---|---|
| 0 | `fetch_medicines.py` | the `medicine.md` files in your checkout, and in the open PRs | the files in `teams/01-data-a/medicines/submissions/<pack_id>/<handle>/` |
| 1 | `medicine_extract.py` | one `medicine.md`, or a directory (eg submissions dir) | the rules as a Python file, and a rule ID for each rule |
| 2 | `harm_test.py` | one pack, one gold file, one rule ID | documents changed, documents dropped, the percent, PASS or FAIL at max(1, 1%) documents |
| 3 | `rule_report.py` | one pack, its gold file, all its rules | `REPORT.md`: harm test of every rule for the pack, and advice on which rules to keep |
| 4 | `run_harm_suite.py` (legacy) | legacy inventory, staged registry/source files, G1 pack list, original packs, gold | per-rule evidence and legacy inventory coverage/status summary |
| any time | `medicine_conflicts.py` | medicine sheets, gold files, and registry or `--sheet` | `MEDICINE_CHECK.md`: coverage and conflict review; nothing is run |
| any time | `registry_view.py` | the registry | `registry.html`: a page to search the rules and see each rule's function |

**Legacy review inputs absent from this checkout.** The shared tools and registry/viewer snapshots are present. Stage the following before running the corresponding registry workflow:

| Absent input | How it was produced | Needed by |
|---|---|---|
| `medicines/submissions/<pack_id>/<handle>/medicine.md` | step 0, `fetch_medicines.py` | step 1 with `--dir`, `medicine_conflicts.py` |
| `rules/<pack_id>/<handle>.py`, the rules files that `registry.json` points to | step 1, `medicine_extract.py` | step 2 with a rule ID (and no `--rules`), steps 3 and 4, `registry_view.py` (the function of each rule), `medicine_conflicts.py` (to tell whether a rule can run) |
| Legacy nonexecutable inventory (`spec/spec_v1.md` was the historical path) | Historical `spec/build_spec_v1.py`, also absent here | step 4; supply a compatible inventory with `--spec` |

For direct sheet testing, use `harm_test.py --rules <medicine.md> --rule <code>` (see "For annotators" above). It does not need these registry inputs. All 24 active registry source-file paths are currently absent; extraction in a review checkout must reconstruct them before registry-based execution.

## 0. Fetch the medicines: `fetch_medicines.py`

```
python -I teams/01-data-a/tools/fetch_medicines.py --include-open-prs
```

Collects the current `medicine.md` of every annotator into `teams/01-data-a/medicines/submissions/<pack_id>/<handle>/medicine.md`. It only saves the files as text. Nothing in them is run. It reads your local `diagnosis/submissions` and, with `--include-open-prs`, the open pull requests too. For each pack and handle the newest version wins: an open PR beats the local file, and a later PR beats an earlier one. Run it again to refresh. Each file is listed as new, updated or unchanged.

`--include-open-prs` needs a GitHub token: set `GH_TOKEN` or `GITHUB_TOKEN`, or run `gh auth login`. Without that option no token is needed. Never commit a token.

Then run step 1 with `--dir teams/01-data-a/medicines/submissions`.

| Option | Meaning |
|---|---|
| `--include-open-prs` | also read the open pull requests. Needs a GitHub token |
| `--pack-id <id>` | only this pack. Can be given more than once |
| `--pack-prefixes <list>` | the packs to take when `--pack-id` is not given. Default: `en-,code-,math-,sci-,cadj-` |
| `--all-packs` | every pack, not only those prefixes |
| `--out <folder>` | where to save. Default: `teams/01-data-a/medicines/submissions`. Keep the last folder named `submissions`, because the extractor reads the pack and handle from the path |
| `--repo <owner/repo>` | the GitHub repo to read the PRs from. Default: `theschoolofai/era-v5-lab` |
| `--repo-root <folder>` | your local checkout. Default: found from the script's location |

Exit code: 0 done, 1 it could not run (for example a missing folder or token).

## 1. Extract the rules: `medicine_extract.py`

```
python -I teams/01-data-a/harmtest/medicine_extract.py diagnosis/submissions/<pack_id>/<handle>/medicine.md
```

Reads the `medicine.md` (it does not run it). Each `**Disease:** <code>` line starts a section, and the Python in that section is its rule. It writes:

- `teams/01-data-a/rules/<pack_id>/<handle>.py`: the rules as one Python file
- `teams/01-data-a/rules/registry.json`: one entry per rule, with its ID

To do all the files in one run, give a folder laid out as `<pack_id>/<handle>/medicine.md` with `--dir <folder>`. It extracts every file and saves the registry once. A file that gives no rule is listed at the end, and the others are still done.

```
python -I teams/01-data-a/harmtest/medicine_extract.py --dir teams/01-data-a/medicines/submissions
```

| Option | Meaning |
|---|---|
| `medicine` | one `medicine.md` (give this or `--dir`, not both) |
| `--dir <folder>` | a folder of `<pack_id>/<handle>/medicine.md`: extract all of them |
| `--out <file>` | where to write the rules file. One `medicine.md` only. Default: `rules/<pack_id>/<handle>.py` |
| `--registry <file>` | the registry to read and update. Default: `rules/registry.json`. With `--dir` the rules files go next to it |

Exit code: 0 done, 1 no rule could be made (with `--dir`: for at least one file), 2 the command could not run.

The extractor finds the function for a code in this order: a function named `medicine_b3` (or `cure_b3`), then the only public function in the section. If none fits, it says so and writes nothing for that rule. 

**Rule IDs** look like `DA-B3-001`: team (`DA` for the Data A packs, `DB` for all others), disease code, number. A rule keeps its ID when you run the extractor again. A rule removed from a `medicine.md` is marked `withdrawn`, and its ID is not reused.

## 2. Harm test: `harm_test.py`

```
python -I teams/01-data-a/harmtest/harm_test.py --pack <pack.parquet> --gold <pack_gold.csv> --rule DA-B3-001
```

Runs one rule on the gold set. Prints documents changed, documents dropped, the percent, the limit, and PASS or FAIL. The exit code is 0 for PASS, 1 for FAIL and 2 for invalid inputs, rule errors or an empty gold set.

**The limit (TSAI decision for small gold sets).** Each rule is checked against the gold set of its own pack; there is no vote or pooling across packs. A rule passes if changed + dropped is at most

```
max(1, floor(0.01 × n))   documents, where n is the number of rows in that pack's gold set
```

At a plain 1%, a gold set under 100 rows (en-nemotroncc 34, math-finemath 30, sci-finepdfs 26, ...) would allow no change at all, so at least 1 document is always allowed until the gold sets grow. For every gold set under 200 rows the limit is 1 document; at 200 to 299 rows it is 2, and so on. `--limit` changes the percent in the same formula: `max(1, floor(limit/100 × n))`. The output shows it, for example `limit: 1 document changed or dropped (max(1, 1% of 34))`, and the `--report` JSON records it as `limit_documents`.

**Empty gold set.** If the gold set has no rows (for example math-megamath), the pack is not testable: the last line is `result: NOT TESTABLE (<file>: the gold set is empty)` and the exit code is 2.

**Where the cleaning code comes from.** `harm_test.py` is a **test runner**, not a collection of hardcoded cleaning rules. It loads executable Python supplied by annotators in one of two ways:

1. **By rule ID (registry):** requires staged extracted source files, currently absent; use `--rules` for direct sheet testing. With `--rule DA-B3-001` and no `--rules`, it looks up the ID in `teams/01-data-a/rules/registry.json`. The registry points to a generated `teams/01-data-a/rules/<pack_id>/<handle>.py` file written by `medicine_extract.py` from an annotator's `medicine.md`. The runner loads that file and selects the matching `RULES_BY_ID` entry.
2. **By explicit file:** With `--rules <rules.py or medicine.md> --rule B3` (or a full rule ID), it loads the supplied file directly, without consulting the registry. A Markdown sheet can contain separate executable `RULE_ID` / `medicine(doc)` blocks; A plain `medicine.md` (one `**Disease:**` section per code) is read by the same code as `medicine_extract.py`, so the rules that run are the rules the extractor writes to `rules/<pack_id>/<handle>.py`. A fragment (a block that does not run on its own) or a block with a syntax error stops the test: fix or remove it. The extractor leaves such a block out and reports it. Two sections for one code, or a name defined with different code in two blocks, is an error too. Python files can define `RULES_BY_ID`, `RULES`, or `cure_<code>`.

**The data comes from different files:** `--pack` supplies the original Parquet documents; `--gold` supplies independently reviewed clean `doc_id`s (with `reader_1` and `reader_2`). Neither is a source of cleaning functions. `harm_test.py` does **not** discover or test all files under `medicines/submissions` automatically; `--rule all` runs only the rules in the **one file** given by `--rules`. The extractor and `rule_report.py` handle multi-submission workflows.

| Option | Meaning |
|---|---|
| `--pack <file>` | the pack Parquet file (required) |
| `--gold <file>` | the gold CSV of that pack (required) |
| `--rule <id or code>` | a rule ID such as `DA-B3-001`, or a disease code such as `B3` (required) |
| `--rules <file>` | a rules `.py` or a `medicine.md`. Not needed for a rule ID |
| `--registry <file>` | where to look up a rule ID. Default: `rules/registry.json` |
| `--limit <percent>` | the failing limit, as a percent of the gold set. Default `1`. At least 1 document is always allowed: `max(1, floor(limit/100 × n))` |
| `--show <N>` | also list the first N `doc_id`s that were changed or dropped. Default `0` |
| `--debug` | print on stderr what the test does (gold and pack sizes, the rules file, the function chosen for each section) and one line for each gold document that was changed, dropped or crashed: its `doc_id`, the kind of change, the lengths and the first offset that differs. Never the text |
| `--timeout <seconds>` | the time allowed for one document. Default `5` |
| `--report <file>` | optional JSON report with source hashes, counts, rule results, and affected IDs (no raw text) |

**Medicine sheets.** A sheet such as `code-medicines/medicine_code.md` has one rule per ```python block. Each block has its own `RULE_ID = "DA-B3-001"` and a function `medicine(doc)`. Each block runs in its own namespace, so blocks may reuse names. A single rule is found by its ID:

```
python -I teams/01-data-a/harmtest/harm_test.py --pack <pack.parquet> --gold <pack_gold.csv> --rules <sheet.md> --rule DA-B3-008
python -I teams/01-data-a/harmtest/harm_test.py --pack <pack.parquet> --gold <pack_gold.csv> --rules <sheet.md> --rule all
```

`--rule all` tests every rule in the supplied file on its own, then the whole file in order. In that last line, `ALL IN ORDER`, a drop ends the document and a fix passes the new text to the next rule. The exit code is the worst result. Not every annotator's Markdown examples are executable rules; malformed Python blocks are errors, not silently skipped.

Two more kinds of rule are accepted:
- **Detectors** that return `True` (drop) or `False` (keep).
- **Rules whose first parameter is named `doc`.** They get the document dictionary (`doc_id`, `text`, and available `url` / `lang_label` metadata). If a medicine requires metadata that the pack lacks, the test reports an error rather than treating the missing value as empty. Rules that look at the file path (H3, F2) or at the language (`LANGS`) need this.

**Safety checks:** malformed/duplicate gold IDs, missing Parquet gold documents, missing required metadata, incomplete/duplicate executable rule definitions, exceptions, and in-place document mutations must not produce a PASS. The runner executes trusted Python from the specified rules file: review the code before running it.

## 3. Report: `rule_report.py`

```
python -I teams/01-data-a/harmtest/rule_report.py --pack-id <pack_id> --pack-file <pack.parquet> --gold <pack_gold.csv>
```

Runs every rule of the pack and writes `teams/01-data-a/rules/<pack_id>/REPORT.md`:

- the harm test result of each rule
- how many of the labelled-sick rows each rule fixes (from the pack's `labels.csv` files)
- for a code with more than one rule: advice to KEEP or DROP each, and why. A rule that does nothing also passes the harm test, so a rule is kept only if it fixes sick rows that the other kept rules do not.
- flags for problems

The advice is for a person to check. Nothing records the decision yet.

| Option | Meaning |
|---|---|
| `--pack-id <id>` | the pack ID, for example `code-rust` (required). Its rules come from the registry |
| `--pack-file <file>` | the pack Parquet file (required) |
| `--gold <file>` | the gold CSV of that pack |
| `--limit <percent>` | the failing limit, as a percent of the gold set. Default `1`. At least 1 document is always allowed: `max(1, floor(limit/100 × n))` |
| `--registry <file>` | the registry to read. Default: `rules/registry.json` |
| `--submissions <folder>` | where the `labels.csv` files are. Default: `diagnosis/submissions` |
| `--out <file>` | where to write the report. Default: `rules/<pack_id>/REPORT.md` |
| `--timeout <seconds>` | the time allowed for one document. Default `5` |

Exit code: 0 done, 2 the report could not be made.

## Check the medicines: `medicine_conflicts.py`

> Registry mode requires staged `medicines/submissions/` and extracted rules, currently absent. Direct `--sheet spec_v1/sheets/medicine_english.md` mode can inspect an existing family sheet without those inputs. Missing sheets in registry mode produce `NO MEDICINE`; that is a missing-input finding, not evidence against the current cleaner.

```
python -I teams/01-data-a/harmtest/medicine_conflicts.py [--packs en-]
```

Reads the medicine sheets, the gold files and the registry, and writes `teams/01-data-a/rules/MEDICINE_CHECK.md`: a section for each gold file, then the codes that several packs of one language treat. **Nothing is run.** The sheets and the rules files are only read (with `ast`), so it is safe to run before anyone has read the annotators' code. It does not run the harm test (`harm_test.py` does that) and it does not choose between rules (`rule_report.py` does that). Run it after step 1.

For each gold file it flags:

| Flag | Meaning |
|---|---|
| `NO MEDICINE` | no medicine sheet, or only blank templates |
| `NOT ENOUGH` | no rule that can run: none was extracted, or each one needs a name that its code never defines |
| `ONE ANNOTATOR` | only one annotator wrote medicines, so there is nothing to compare them with |
| `NOT EXTRACTED` | a sheet has medicine sections but no rule in the registry (run `medicine_extract.py`) |
| `CANNOT RUN` | a rule needs names that its code never defines: it fails with `NameError` |
| `NO RULE` | a code that a sheet says was found more than five times has no rule that can run |
| `BLANK SHEET` | an annotator's sheet is the blank template, while other medicines cover the pack |
| `SMALL GOLD SET` | fewer than 200 gold documents: the limit is 1 document (`max(1, floor(1% of n))`), so a second touched document fails a rule |

For each code with two or more medicines in one pack: `ACTIONS DIFFER` (they disagree on fix, drop or keep), `RULE VS HAND` (one annotator wrote a rule, another edits named rows by hand) and `CHOOSE` (two or more rules can run: `rule_report.py` says which to keep). The medicines of each such code are listed side by side, because a clash in what they do (cut a cut-off text, or join it to the next chunk) cannot be found by a program: a person compares them.

| Option | Meaning |
|---|---|
| `--packs <text>` | only the packs whose ID starts with this, for example `en-` |
| `--medicines <folder>` | the medicine sheets, as `<pack_id>/<handle>/medicine.md` or `<pack_id>-medicine/<handle>-medicine.md`. Default: `medicines/submissions` |
| `--gold-dir <folder>` | the gold files, `<pack_id>_gold.csv`. Default: `gold` |
| `--registry <file>` | the registry to read. Default: `rules/registry.json` |
| `--sheet <file>` | take the rules from a standard sheet (one `RULE_ID` block per rule, with `ORIGIN` and `AUTHOR`), for example `spec_v1/sheets/medicine_english.md`, instead of the registry. A rule whose `ORIGIN` lists several packs counts for each of them |
| `--out <file>` | where to write the report. Default: `rules/MEDICINE_CHECK.md` |

Exit code: 0 done, 1 a gold file has no medicine or no rule that can run (`NO MEDICINE`, `NOT ENOUGH`), 2 the report could not be made.

## 4. Run the legacy inventory harm suite: `run_harm_suite.py`

> **Legacy workflow.** This runner needs a nonexecutable inventory, extracted registry source files, and matching contributor submissions. Those inputs are absent. Its default `spec/spec_v1.md` is historical; pass `--spec` for a staged legacy inventory. Do not pass `spec_v1/sheets/specv1.md`: that executable-sheet format is incompatible. For current validation use [validate_specv1.py](../spec_v1/validate/validate_specv1.py) and [make_harm_tests.py](../spec_v1/validate/make_harm_tests.py).

Run the **batch audit** after collecting/extracting medicines and generating the draft parent Spec v1. Unlike `harm_test.py --rule all`, this script evaluates **individual active registry rules** for each applicable pack; it does **not** execute the `spec_v1.md` itself. The parent spec contains review-only fenced `text` code examples, not executable `python` rule blocks. The suite reads its medicine inventory, proposal statuses, and conflict ledger, then maps candidate rules from `rules/registry.json` by pack, contributor, and disease code.

From the repository root, first inspect the planned coverage (does not execute contributor code):

```bash
uv run python teams/01-data-a/harmtest/run_harm_suite.py \
  --parquet-dir /path/to/original-packs --spec /path/to/legacy-inventory.md --dry-run
```

Then run harm tests against the original Parquet files and current official gold CSVs:

```bash
uv run python teams/01-data-a/harmtest/run_harm_suite.py \
  --parquet-dir /path/to/original-packs --spec /path/to/legacy-inventory.md
```

| Option | Meaning |
|---|---|
| `--parquet-dir <folder>` | original pack files named `<pack_id>.parquet` (**required**) |
| `--gold-dir <folder>` | official gold CSV directory; default `teams/01-data-a/gold` |
| `--report-dir <folder>` | output directory; default `teams/01-data-a/reports` |
| `--spec <file>` | draft parent specification; default `teams/01-data-a/spec/spec_v1.md` |
| `--registry <file>` | executable rule-ID registry; default `teams/01-data-a/rules/registry.json` |
| `--pack-ids <file>` | core G1 pack list; default `teams/01-data-a/gold/g1_pack_ids.txt` |
| `--limit <percent>` | maximum individual-rule harm percentage; default `1`. Passed to `harm_test.py`, so at least 1 document is always allowed |
| `--timeout <seconds>` | time limit for each rule/document call; default `5` |
| `--dry-run` | show coverage and statuses without running any medicine code |

**Outputs:** `reports/g1_harm_summary.json` records each pack's scope, proposed versus registered coverage, unresolved conflicts, rule results, and reason for incomplete status. `reports/g1_harm_metadata.json` records Spec/registry hashes, G2 status, and combined-pipeline status. Per-rule reports are written to `reports/rules/<pack_id>_<RULE_ID>_harm.json` with gold/pack hashes and affected document IDs, not document text. **All serialized JSON paths are relative to the repository root**, including paths in missing-input messages. External original packs are represented with `../` segments where needed; command-line paths may still be absolute. The suite also includes the CADJ packs referenced in Spec v1, marked `SPEC_EXTRA` rather than part of the core G1 pack list.

**Interpretation:** Individual rules report `PASS`, `FAIL`, or `ERROR` as `harm_status`; pack-level `status` is a separate review/coverage state. A pack may be `NOT_EVALUATED` (missing input), `SPEC_STALE` (submission differs from Spec v1), `PENDING_RULES`, `PARTIAL_COVERAGE`, `BLOCKED_CONFLICT`, `HARM_FAIL`, `ERROR`, `DRY_RUN`, or `PENDING_APPROVAL`. A technical PASS is **not** medicine approval. Missing or unmatched proposals are reported rather than counted as PASS, and conflicting candidates are **not** combined or ordered automatically.

**Exit codes:** `1` means the G1 release audit remains **INCOMPLETE**, including successful dry-run audits; that is expected while Spec v1 is a draft, G2 is pending, and no combined pipeline has been approved. `2` indicates invalid suite-level inputs. Check the JSON and printed per-pack statuses rather than treating exit code `1` alone as a test failure. For individual rule failures, review their per-rule JSON reports. The suite runs contributor Python through `harm_test.py`: review the generated rules before execution.

## The registry and Spec v1

> **Snapshot status.** Registry/viewer files are preserved, but their extracted source files, the legacy inventory, and collected contributor submissions are absent. See [reconstruction prerequisites](../rules/README.md). Current five-family cleaner selection and Indic integration are documented in [spec_v1/README.md](../spec_v1/README.md).

`teams/01-data-a/rules/registry.json` has one entry for every rule that the extractor found:

| Field | Meaning |
|---|---|
| `id` | the rule ID, for example `DA-B3-001` |
| `pack`, `handle` | where the rule came from |
| `code`, `n` | the disease code, and which medicine of that code in the annotator's file (1, 2, ...) |
| `action` | `fix` or `drop`, as the annotator wrote it |
| `file` | the rules file that holds the code, relative to the registry |
| `withdrawn` | `true` if the rule is no longer in the `medicine.md` |
| `code_sha256` | fingerprint of the code in the `medicine.md` when it was extracted. `harm_test.py` prints the same value as `code sha256` for a plain `medicine.md`, so you can check that the code you run is the code the annotator tested. Comments, blank lines and spacing do not change it |

To look at the preserved source, open [registry.html](../rules/registry.html). After reconstructing a matching registry and its source files, `python -I teams/01-data-a/harmtest/registry_view.py --out <new-viewer.html>` generates a new viewer without executing rule code or requiring `pyarrow`. Avoid overwriting the preserved viewer from this incomplete source tree. `--show DA-B3-001` prints a staged rule in the terminal.

| Option | Meaning |
|---|---|
| `--registry <file>` | the registry to read. Default: `rules/registry.json` |
| `--out <file>` | where to write the page. Default: `registry.html` next to the registry |
| `--show <rule ID>` | print the source of one rule in the terminal, and write no page |

Historically, the absent `spec/build_spec_v1.py` compiled a nonexecutable parent inventory from collected submissions into `spec/spec_v1.md`, `spec/spec_v1_sources.csv`, and `spec/spec_v1_conflicts.csv`. That was a review inventory, not the current cleaner build. For a reconstructed legacy review:

1. Run the extractor on every `medicine.md`, then `rule_report.py` for every pack.
2. Take the rules that are not withdrawn, PASS the harm test, and that the report advises to KEEP.
3. For each one, write the ID, the code, the action (from the registry), the rule in words (the **Medicine** text in that annotator's `medicine.md`), the rule in code (the function in `file`), and the harm test result (from the report).
4. Record which handoff and review snapshot the inventory describes. Current Indic selections are maintained separately through the [Indic integration process](../spec_v1/README_indic.md).
5. Decide the order the rules run. The registry does not hold an order, so the team decides it.

The generated `spec_v1.md` is a **draft**: reviewers must resolve clashes and verify harm on regenerated official gold sets before approving any executable pipeline. Use the `run_harm_suite.py` batch audit above to collect current per-rule evidence and identify missing coverage; it does not approve conflicts or run the combined pipeline.

## Writing a medicine.md so the extractor can read it

- Put `**Disease:** <code>` and `**Action:** fix` (or `drop`) in each section. Use one section for each disease code.
- Put the Python in a ```` ```python ```` block in that section. Every python block must run on its own: complete it, or remove it from the file.
- Name the main function `medicine_<code>` (for example `medicine_b3`; `cure_b3` also works). This is a suggestion, not a requirement: if the section has only one function whose name does not start with `_`, any name works. It takes the text and returns the new text. Do not name it just `medicine`: every section would then define the same name.
- To drop a document, return `("drop", "")`. To fix, return a string, or `("fix", new_text)`.
- **Two fixes for one disease code** (for example a regex replacement and removing header lines): write one section and one main function. Put each fix in a helper function in the same code block, name it with a leading `_` (for example `_strip_header`), and call the helpers from `medicine_<code>`. Helpers are copied into the rules file with the rule and get no ID of their own.

## Notes

- The tools **run the annotators' code** (`harm_test.py`, `rule_report.py`). Read it first.
- Gold files go in `teams/01-data-a/gold/<pack_id>_gold.csv`: `doc_id`, `reader_1`, `reader_2` (no raw text). A `_DUMMY` gold file is for testing.
- The pack Parquet files are not in the repo.
- **Regression-test code is different from medicine code:** `test_harm_test.py` and `test_medicine_extract.py` create small synthetic rules, mock/test Parquet rows, and gold CSV fixtures to check the **harness**. Their pytest PASS results do **not** establish that annotators' real cleaning rules pass the harm test (`max(1, floor(1% of n))` documents). For real harm results, run `harm_test.py` with the relevant medicine/registered rule, original pack Parquet, and refreshed official gold CSV.
- Tests are in `teams/01-data-a/tests/harmtest/`. Run the suite with `python -m pytest teams/01-data-a/tests/harmtest`; standard-library unittest files such as `test_harm_test.py`, `test_medicine_extract.py` and `test_rule_report.py` also support direct `python -I teams/01-data-a/tests/harmtest/<file>` invocation. The legacy inventory suite needs pytest.
- `registry.json` is shared. Decide who runs the extractor and commits it, to avoid merge conflicts.
