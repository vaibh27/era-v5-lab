# Scripts behind Spec v1.1

Spec v1.1 adds medicines and harm tests for the new diseases that the Week 2 reading audits found in 1% or more of the docs read in a domain. These scripts make `results/` from those audits.

| file | what it does |
|---|---|
| `collect_reading_prs.py` | Collects every `reading_<pack>_<reader>.csv` and `.md` from the open reading PRs and from `g1-reading` on `mittalutkarsh/era-v5-lab`. Counts each disease per domain (english, code, math, science) and lists the Spec v1.1 candidates. Tests: `tests/test_collect_reading_prs.py`. |
| `show_examples.py` | For each Spec v1.1 candidate, writes every reader's row and the cleaned text at their offsets, read from the cleaned sample parts. For the hand check and for writing the medicine. The file holds pack text, so it is written outside the repo. Tests: `tests/test_show_examples.py`. |
| `build_reading_check.py` | Writes `teams/01-data-a/reading/reading_check_v1.md`, the reading consolidation, from the collector's output and `sheets/NEW_DISEASES.md`. It covers all four domains: the docs read, the diseases Spec v1 missed, the new diseases and harm rows with their rates (docs / docs read), and the new diseases at 1% or more. Run it after the collector. Tests: `tests/test_reading_check.py`. |
| `reading_recall.py` | Runs each rule of a medicine sheet on the cleaned docs of the reading rows with its code, and counts the docs it changes or drops: the "reading rows the rule treats" figure on each card. Loads the sheet the way `tools/harm_test.py` does. Prints doc_ids and counts only. Tests: `tests/test_reading_recall.py`. |

## Run

From `spec_v1_1/scripts/`:

```
python3 -I collect_reading_prs.py --docs-read pyru=199              once
python3 -I collect_reading_prs.py --docs-read pyru=199 --watch 30   again every 30 minutes, until Ctrl-C
```

`--docs-read READER=N` is for a reader who read more or fewer than 100 docs; pyru read 199. `--alias CODE=CODE` counts two readers' names for one disease as one. `--min-percent` sets the cut (default 1, so 2 of 200 docs counts and 2 of 299 does not).

Then, with the cleaned sample ("tokenizer sample v0-clean", Train parts) unzipped outside the repo:

```
python3 -I show_examples.py --domain math --parts <folder>/<math parts>.parquet --out <outside the repo>/examples_math.md
```

It refuses an `--out` inside the repo, never shows F1 or E3 rows, and prints only counts. `--context N` sets the characters shown on each side of a reader's span (default 300).

The spec v1 rules column is read from `../../spec_v1/sheets/specv1.md`. A GitHub token comes from `GH_TOKEN`, `GITHUB_TOKEN` or `gh auth token`; without one GitHub allows 60 requests an hour, and a run uses about 10.

## Output: `results/reading/`

| file | what it holds |
|---|---|
| `summary.md` | The open reading PRs, the README rows with no reading yet, the docs read per domain, and the Spec v1.1 list. Missed and harm rows are listed too, but they are about spec v1 rules, not new diseases. |
| `diseases_by_domain.csv` | Per domain, kind and code: distinct docs, % of docs read, readers, spec v1 rules for that family, the Spec v1.1 flag and example doc_ids. |
| `findings_all.csv` | Every row of every reading CSV, with domain, pack, reader and the PR it came from. The quote is left empty for F1 and E3. |
| `raw/`, `state.json` | A local cache of the downloaded files and the PR heads of the last run. Not needed in a commit. |

Every reading row so far is AI-suggested, not a signed human review. Check the example docs before fitting a rule to them.
