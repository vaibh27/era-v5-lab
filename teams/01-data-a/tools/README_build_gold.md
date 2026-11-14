# G1 Data A — gold set builder

Build reproducible, per-pack gold CSVs from diagnosis labels. Designed for
`theschoolofai/era-v5-lab`. It **does not** download or commit the Parquet packs,
private information, medicine text, or diagnosis evidence.

## Install and run

The builder is [build_gold.py](build_gold.py), with tests in
[../tests/tools/test_build_gold.py](../tests/tools/test_build_gold.py). Run the commands below
from a local checkout's repository root.

Requires Python 3.9+ (standard library only for the main task).

To include the **current open PRs** as well as merged/local submissions, log into GitHub:

```bash
gh auth login
python teams/01-data-a/tools/build_gold.py \
  --repo-root . \
  --include-open-prs
```

Alternatively provide a read-only GitHub token via `GH_TOKEN` or `GITHUB_TOKEN`.
The token is never written to disk. The open-PR mode uses GitHub's REST API,
reads the latest complete `labels.csv` blob (NOT the truncated PR diff), and
requires a network connection. It does not modify any PRs.

Local/offline run (only labels already present in your checkout):

```bash
python teams/01-data-a/tools/build_gold.py --repo-root .
```

**Scope:** By default this includes pack names beginning `en-`, `code-`,
`math-`, `sci-`, or `cadj-`. The `cadj-` category is **provisional** until
TSAI confirms Data A/Data B ownership; inspect the generated list. A CSV or
one-per-line text list of confirmed G1 pack IDs is preferable:

```bash
python teams/01-data-a/tools/build_gold.py \
  --repo-root . --include-open-prs \
  --pack-list teams/01-data-a/gold/g1_pack_ids.txt
```

Example entries from `g1_pack_ids.txt`:

```text
en-dclm
code-go
math-finemath
sci-finepdfs
```

A CSV pack list must contain a `pack_id` column. Any listed pack missing labels
still receives a header-only gold CSV and shows zero coverage (no fabricated votes).

Optional: validate IDs against the actual downloaded Parquet packs (requires
`pip install pyarrow`):

```bash
python teams/01-data-a/tools/build_gold.py \
  --include-open-prs --pack-list teams/01-data-a/gold/g1_pack_ids.txt \
  --pack-dir /path/to/parquet_packs
```

The files must be named `<pack_id>.parquet`. This validation reads only the
`doc_id` column. Packs do not belong in git.

## Output

Files under `teams/01-data-a/gold/`:

- `<pack_id>_gold.csv`: `doc_id,reader_1,reader_2` (only IDs and GitHub handles)
- `gold_coverage.csv`: per-pack counts (`label_files`, reviewers, unique IDs,
  full reads, 2-reader overlaps, two-clean candidates, disputes, gold count,
  target shortfall, skipped label files)

Previous gold files for packs no longer selected are left untouched by default;
use `--remove-stale` to delete obsolete `*_gold.csv` outputs deliberately.

The output is sorted and deterministic for unchanged local files and PR heads.
It contains **all** eligible gold IDs; `--target 200` is a reporting target,
not an arbitrary cutoff. A shortfall is reported rather than hidden.

## Eligibility

1. Use one current labels file per `(pack_id, GitHub handle)`; the latest open
   PR version supersedes the merged/local version for that handle. Two PRs
   from one person **do not** make two readers.
2. Parse `doc_id,read,verdict,diseases,evidence,hex,note` using CSV rules.
3. A positive vote requires `read=full`, `verdict=clean`, and no disease
   codes **other than D3**. TSAI clarification: **D3 (natural mixed language)
   is label-only**. A full-read `clean` row with an empty diseases cell or
   only D3 counts as a clean vote. A row with `verdict=fix` and only D3
   **does not count as clean yet** and remains a dispute/veto; we will decide
   that case separately. `partial` does not count. A `drop` verdict or any
   other disease (including `D3;C1`) cannot count as clean.
   Source CSV labels and verdicts are not modified.
4. Require **two distinct positive reviewers for the SAME doc_id in the SAME pack**.
5. Conservatively exclude candidates on which *any* additional reviewer
   explicitly said `fix` or `drop`, including `fix,D3` for now. Use
   `--allow-disputed` only if the team confirms strict
   "2 clean votes regardless of dissent" semantics.
6. Conflicting duplicate rows for a doc ID from one reviewer disqualify that
   reviewer's clean vote **and mark the document disputed**. This prevents a
   `fix`/`drop` from vanishing when other reviewers provide two clean votes.
   Duplicate conflicts cannot enter gold even with `--allow-disputed`.
7. Bad CSV sheets are skipped with warnings, counted in `skipped_label_files`
   (not partially included), and do not stop other packs. The parser accepts
   fields up to 16 MiB; `--fail-on-warnings` makes anomalies fail the command.
   A newer PR replaces, rather than concatenates with, the older file from
   the same reviewer.

Use `--fail-under-target` for an optional nonzero exit code when a listed pack
has fewer than the reporting target, and `--fail-on-warnings` to require
manual review of duplicate/anomalous inputs.

**Gold tests only catch likely harm to reviewed-clean documents.** They do not
prove a cleaner is effective or that a gold set is representative. This script
builds gold CSVs only; `harm_test.py` is a separate deliverable.

## Tests

```bash
python -m unittest discover -s teams/01-data-a/tests/tools -v
```

## Sharing / PR

After examining `gold_coverage.csv`, submit your branch and PR to the original
repository. Do not commit downloaded raw Parquet packs or other sensitive data.

```bash
git checkout -b g1-gold-sets
git add teams/01-data-a/tools/build_gold.py \
        teams/01-data-a/tests/tools/test_build_gold.py \
        teams/01-data-a/tools/README_build_gold.md \
        teams/01-data-a/gold/
git commit -m "G1: build audited Data A gold sets"
git push -u origin g1-gold-sets
gh pr create --repo theschoolofai/era-v5-lab --fill
```

If the gold directory contains other generated artifacts, inspect `git status`
before committing. Do not commit raw labels, notes, API credentials, or pack
contents as part of the G1 gold-set PR.
