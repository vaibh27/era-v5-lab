# Spec v1.1 (Data A)

Spec v1.1 adds medicines and harm tests for the **new** diseases that the Week 2 reading audits found in **1% or more of the docs read in a domain** (english, code, math, science). Each one needs a medicine and a harm test. A `drop` medicine counts as a medicine, and the harm test sets its status.

The flow follows spec v1:
1. One `medicine_<domain>.md` sheet per domain.
2. The sheets are collated, with conflicts resolved.
3. The final selection goes into `sheets/specv1_1.md`.

## Layout

| folder | what is in it |
|---|---|
| `scripts/` | `collect_reading_prs.py` (the candidate list) and `show_examples.py` (the text behind each reading row). See `scripts/README.md`. |
| `tests/` | One test file per script. Run from `teams/01-data-a`: `python3 -m unittest discover -s spec_v1_1/tests`. |
| `results/reading/` | The collector's output: `summary.md`, `diseases_by_domain.csv` and `findings_all.csv`. |
| `results/sheets/` | Harm-test reports (JSON with IDs only), one per pack, and `summary.md`, a summary of the results, the gold contradictions and what is open, with the commit they belong to. |
| `sheets/` | `medicine_<domain>.md`, `decisions_<domain>.md`, and later `specv1_1.md`. |

## Steps

1. **Final candidate list.** After the readers' deadline (12 Nov 23:59 IST), from `scripts/`: `python3 -I collect_reading_prs.py --docs-read pyru=199`. Add `--alias` for any codes agreed to be one disease.
2. **Examples for the hand check.** Run `scripts/show_examples.py` on the cleaned sample parts (tokenizer sample v0-clean, Train). Write its output outside the repo, because it holds pack text. A person reads the examples before a rule is fitted to them.
3. **Sheets.** Fill `sheets/medicine_<domain>.md` and `sheets/decisions_<domain>.md`, one card per candidate, in the spec v1 card form.
4. **Unit tests** for each medicine go in `tests/`.
5. **Harm tests**, with the commands below. Write the reports to `results/sheets/`, never to `rules/`.
6. **Conflicts:** `../tools/medicine_conflicts.py`, with `--out` pointing into `results/sheets/`.
7. **Collate** into `sheets/specv1_1.md`.
8. **Push** to `g1-spec-v1-1` on patilvijay23/era-v5-lab. This updates theschoolofai/era-v5-lab#365.

## Harm tests

**Command.** Run from the repo root. One run per pack, never pooled. `--rule all` tests each section on its own, then all of them in file order.

```
python3 -I teams/01-data-a/tools/harm_test.py \
  --pack <pack Parquet> \
  --gold teams/01-data-a/gold/<pack>_gold.csv \
  --rules teams/01-data-a/spec_v1_1/sheets/medicine_<domain>.md \
  --rule all \
  --report teams/01-data-a/spec_v1_1/results/sheets/<pack>_harm.json
```

**Limit.** Gold docs changed plus dropped must be at most `max(1, floor(1% of gold))`. A pack with no gold set is NOT TESTABLE. A result of 0 is only an upper bound on the harm (about 3/n).

**Packs.** `W1` is `/Users/aneeshadas/ERA5/Capstone/Week1-G-work`. The pack files stay outside the repo.

| pack | pack Parquet | gold docs | limit |
|---|---|---|---|
| math-finemath | `W1/medicine-spec/Maths-parquet-files/math-finemath.parquet` | 86 | 1 |
| math-megamath | `W1/medicine-spec/Maths-parquet-files/math-megamath.parquet` | 26 | 1 |
| sci-finepdfs | `W1/medicine-spec/Science-parquet-files/sci-finepdfs.parquet` | 202 | 2 |
| code-cpp | `W1/medicine-spec/Code-parquet-files/code-cpp.parquet` | 122 | 1 |
| code-go | `W1/medicine-spec/Code-parquet-files/code-go.parquet` | 163 | 1 |
| code-java | `W1/medicine-spec/Code-parquet-files/code-java.parquet` | 116 | 1 |
| code-js-ts | `W1/medicine-spec/Code-parquet-files/code-js-ts.parquet` | 204 | 2 |
| code-python | `W1/medicine-spec/Code-parquet-files/code-python.parquet` | 141 | 1 |
| code-rust | `W1/medicine-spec/Code-parquet-files/code-rust.parquet` | 234 | 2 |
| en-dclm | `W1/english_parquet_files/en-dclm.parquet` | 62 | 1 |
| en-fineweb-edu | `W1/medicine-spec/English-parquet-files/en-fineweb-edu.parquet` | 158 | 1 |
| en-nemotroncc | `W1/medicine-spec/English-parquet-files/en-nemotroncc.parquet` | 129 | 1 |
| en-nemotroncc-synth | `W1/medicine-spec/English-parquet-files/en-nemotroncc-synth.parquet` | 326 | 3 |

## Open decisions

- **Domain split.** Who writes which domain's sheet.
- **Folder layout.** This folder follows main's `spec_v1/` layout. `g1-specv1` has moved to `tests/spec_v1`, `results/spec_v1`.
- **Codes for new diseases.** DISEASES.md has no codes for the `NEW-…` diseases.
- **Which harm run counts.** Is it the v1.1 rules alone, or all v1 + v1.1 rules in order?
