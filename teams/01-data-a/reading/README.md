# Week 2: cleaned-pack audit (reading)

Read 100 cleaned documents, log every problem you find, and put two files in this folder.
This covers rows 7-10 of the Week 2 assignments tab.

## Branches

Work on your own branch off `g1-reading`. Never push to `main`, never push to `g1-reading`.

```
git checkout g1-reading
git pull
git checkout -b g1-reading-<pack>-<yourname>        # e.g. g1-reading-en-dclm-aswani
```

Keep the downloaded data **outside** the repo. Only your `.md` and `.csv` go in here.

## The data

Axiom -> Courses -> ERA V5 -> CAP - Capstone Lab 1 -> "Week 2: cleaned tokenizer sample v0-clean".
This is tokenizer sample v0 after our spec v1 cleaner (`clean.py`, PR #244). Same packs, parts
and columns as v0.

Download the Train **Part 1** file for each of your packs, Part 2 as well if your pack has one,
and `rule_counts.csv` for context. Do **not** use the Heldout files.
The links expire Wed 11 Nov, 12:17 IST.

## Who reads what

100 docs per person, no overlap with your partner.

| Row | Reader | Pack(s) |
|---|---|---|
| 7 | Aswani Keerthi | en-dclm (50) + en-fineweb-edu (50) |
| 7 | Harini Jasti | en-nemotroncc (50) + en-nemotroncc-synth (50) |
| 8 | Mahesh Sv | code-cpp (34) + code-go (33) + code-java (33) |
| 8 | Revant Gupta | code-js-ts (34) + code-python (33) + code-rust (33) |
| 9 | Ramesh | math-finemath (100) |
| 9 | Satyanarayana Murthy Toleti | math-megamath (100) |
| 10 | Utkarsh Mittal | sci-finepdfs, docs 1-100 of the shuffled list |
| 10 | Aditeya | sci-finepdfs, docs 101-200 of the shuffled list |

## Picking your docs

Load Part 1 then Part 2 in that order, shuffle with seed **20261101**, take the first N.
Use `pick_docs.py` in this folder so everyone shuffles the same way:

```
python teams/01-data-a/reading/pick_docs.py \
    --parts ~/packs-v0-clean/sci-finepdfs_part1.parquet ~/packs-v0-clean/sci-finepdfs_part2.parquet \
    --rows 1-100 \
    --out ~/packs-v0-clean/sci-finepdfs_utkarsh_docids.txt
```

Science: Utkarsh takes `--rows 1-100`, Aditeya takes `--rows 101-200`, off the same shuffle.
Record the file names, the seed and the doc_ids you read; they go in your `.md`.

## What to read against

Under `teams/01-data-a/spec_v1/sheets/`:

- `specv1.md` - the consolidated spec, all disease codes and rules
- your domain sheet: `medicine_english.md`, `medicine_code.md`, `medicine_math.md`, `medicine_sci.md`
  (science: also `medicine_sci_clash_resolution_and_harm_report.md`)
- `decisions_english.md`, `decisions_code.md`, `decisions_math.md` for decisions already taken

## What to log

Read the full cleaned text of each doc. Every problem is one of three kinds:

- **missed** - spec v1 has a rule for this disease (C1, C2, B4, B5 ...) and the cleaned doc still has it
- **new** - a problem spec v1 has no rule for. From the math audit: missing formulas, flattened
  formulas, SEO filler, template-generated pages, merged table cells, missing spaces between words
- **harm** - a spec v1 rule removed or damaged real content. Give the rule ID.
  Example: DA-C2-001 deleted program output in 4 docs.

A clean doc still counts as read; it just has no rows.

## What to submit

Two files in this folder.

**`reading_<pack>_<yourname>.csv`** - one row per finding, these columns exactly:

```
doc_id,kind,code,offsets,quote,why,source
```

- `doc_id` - the doc's id from the data file
- `kind` - missed / new / harm
- `code` - the disease or rule code (e.g. `C1`, or `DA-C2-001` for harm), or
  `NEW-<short-name>` for a new disease (e.g. `NEW-seo-filler`)
- `offsets` - character start-end in the cleaned text, e.g. `1204-1388`
- `quote` - a short snippet showing the problem
- `why` - one line
- `source` - `manual`, or `ai` if an AI tool suggested it and you checked it

**Never paste personal data or secrets into the quote column.** For F1 and E3 give the type and
the offset only, for example `email at offset 64`, and leave `quote` empty.

**`reading_<pack>_<yourname>.md`** with these sections:

- How it was read: pack file(s), seed 20261101, number of docs, cleaner version/commit, AI used or not
- Findings table: code | kind | docs affected | % of docs read
- New diseases: name, one-line description, number of docs
- Harm: rule ID, what it deleted, number of docs
- Open questions for the lead

Reading two or three packs: either one pair of files per pack, or one pair named after your domain
(`reading_english_aswani.md` / `.csv`) with a `pack` column added to the CSV.

## Submitting

```
git add teams/01-data-a/reading/reading_<pack>_<yourname>.md \
        teams/01-data-a/reading/reading_<pack>_<yourname>.csv
git commit -m "reading: <pack> audit, 100 cleaned docs (task <row>, <yourname>)"
git push origin g1-reading-<pack>-<yourname>
```

Add only those two files. Do not use `git add .`

Then open a PR in this repo, base `g1-reading`, compare your branch. Check that "Files changed"
shows only your two files; if it shows more, stop and ping me. Title:
`reading: <pack> audit, 100 cleaned docs (task <row>, <yourname>)`.
Description follows PR #27 on patilvijay23/era-v5-lab (Files / How it was read / Main findings /
Checks). Add `mittalutkarsh` as reviewer.

## Dates (IST)

| By | What |
|---|---|
| Sat 7 Nov | GitHub username to me, invite accepted |
| Tue 10 Nov | Data downloaded (links die Wed 11 Nov, 12:17) |
| Thu 12 Nov, 23:59 | PR raised |
| Fri 13 Nov | Aditeya and I consolidate into `reading_check_v1.md`, one combined PR onward |

Any new disease above 1% (2+ docs out of 200 in a domain) goes to Smita and Aneesha for spec v1.1.
