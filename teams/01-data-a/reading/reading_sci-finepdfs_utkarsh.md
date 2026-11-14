# Science reading audit (task 10): sci-finepdfs

Reader: Utkarsh Mittal. Row 10 of the Week 2 assignments, shuffled rows 1-100 (Aditeya reads 101-200).
The findings are in `reading_sci-finepdfs_utkarsh.csv`: 47 findings in 41 docs. The other 59 docs were read and had no finding, so they have no rows; all 100 doc_ids are listed at the end.

## How it was read

- **Pack file:** tokenizer sample v0-clean, `train/sci-finepdfs/part-00000.parquet` (Part 1; sci-finepdfs has no Part 2).
  6,011 docs, sha256 `db4736df225c8931…`.
- **Cleaner:** spec v1 `clean.py` as run by TSAI on #244. The sci family runs 12 rules (`DA-B4-013` to `DA-D3-004`) plus the shared `f1_rule.py`.
- **Seed and pick:** 20261101, `pick_docs.py`, `--rows 1-100`. The list is at the end of this file.
- **Docs read:** 100. Doc [16] `1644d240cedb1491` (1.86M chars, mostly tables and Urdu text) was read in part.
- **AI used:** 
- **Offsets:** code points in the cleaned text, end exclusive. `[N]` below is the position in the shuffled list.
- **Private data:** F1 and personal-listing rows give the type and offsets only, with `quote` left empty.

## Method

The findings were pre-screened with pattern scans over the 100 docs and then read by me. Each row was then
checked against `specv1.md`, `medicine_sci.md` and `medicine_sci_clash_resolution_and_harm_report.md` and the sci rules
in `clean.py`: a finding a sci rule covers is `missed`, one no rule covers is `new`.

- Read fully: <!-- TODO: list docs read fully -->
- Skimmed: <!-- TODO: list docs skimmed --> (doc [16] at least)

## Findings table

17 missed findings in 15 docs, 30 new in 29 docs, 0 harm. 59 docs had no finding.

| code | kind | docs affected | % of docs read |
|---|---|---|---|
| A5 | missed | 4 | 4% |
| F1 | missed | 4 | 4% |
| A1 | missed | 2 | 2% |
| C1 | missed | 2 | 2% |
| C2 | missed | 2 | 2% |
| D1 | missed | 2 | 2% |
| B4 | missed | 1 | 1% |
| NEW-page-artifact | new | 9 | 9% |
| NEW-form-blanks | new | 5 | 5% |
| NEW-drop-cap-split | new | 2 | 2% |
| NEW-missing-figures | new | 2 | 2% |
| NEW-missing-spaces | new | 2 | 2% |
| NEW-near-empty | new | 2 | 2% |
| NEW-small-caps-split | new | 2 | 2% |
| NEW-empty-table-rows | new | 1 | 1% |
| NEW-huge-tables | new | 1 | 1% |
| NEW-latex-residue | new | 1 | 1% |
| NEW-personal-names-ages | new | 1 | 1% |
| NEW-prose-in-code-fence | new | 1 | 1% |
| NEW-scrambled-vertical-text | new | 1 | 1% |

## Top findings

- **F1 not masked:** phone numbers and home addresses of private people survive in [41], [49], [61], [67]. In [41] emails were masked but phones were kept. `DA-F1-006` only quarantines listed IDs from the old pack, and `f1_rule.py` did not catch these.
- **Wrong language in an English pack (D1):** [16] carries ~380k Arabic-script chars of Urdu debate text and [11] opens with the Spanish half of a bilingual treaty. `DA-D1-001` only quarantines two listed IDs.
- **C2 running headers kept:** [3] repeats a date footer on every page, [72] a bid title 49 times. The running-header code in `DA-C1-007` is gated to six old-pack IDs.
- **New candidates:** page artifacts (9 docs), form blanks (5), drop-cap and small-caps splits (4), missing figures (2). Ligatures (A5) are in 4 docs; spec v1 folds them only in math (`DA-A5-002`), so they count as missed here.

## New diseases

| name | description | docs |
|---|---|---|
| `NEW-page-artifact` | bare page numbers, 'Page N' / 'Page N of M' footers and slide date footers left in the text | 9 |
| `NEW-form-blanks` | fill-in blanks from PDF forms: underscore or dotted lines, empty field labels | 5 |
| `NEW-missing-figures` | markdown image links or placeholders where a figure was; the figure content is gone | 2 |
| `NEW-drop-cap-split` | a drop-cap initial split onto its own line or lost ("t. Thomas") | 2 |
| `NEW-small-caps-split` | small caps extracted as a split first letter ("F OODS L IMITED") | 2 |
| `NEW-missing-spaces` | words or fields run together with no space | 2 |
| `NEW-near-empty` | a doc of under ~250 chars with only a title or header | 2 |
| `NEW-latex-residue` | well-formed LaTeX commands left in prose (\textsuperscript) | 1 |
| `NEW-huge-tables` | thousands of markdown table lines dominating a doc | 1 |
| `NEW-empty-table-rows` | markdown table rows with every cell empty | 1 |
| `NEW-scrambled-vertical-text` | vertical letterhead extracted one word or letter per line, out of order | 1 |
| `NEW-prose-in-code-fence` | prose wrapped in code fences | 1 |
| `NEW-personal-names-ages` | names of private people with ages or birthdays, no contact data | 1 |

Over the 1% line (2+ docs of 200 in the domain, with Aditeya's 100): page artifacts, form blanks, missing figures, drop-cap splits, small-caps splits, missing spaces, near-empty docs.

## Harm

None found. No row in this read shows a spec v1 rule removing or damaging real content.

## Open questions for the lead

1. Most sci rules are gated to doc IDs from the original 500-doc pack, so on this sample they mostly do nothing. Is that expected for v0-clean?
2. `cleaned_manifest.json` lists `cleaning_rules_applied: []` for every file and `rule_counts.csv` is header-only. Which `clean.py` commit ran, and did it change any sci doc?
3. [11] (bilingual treaty) and [16] (English tables plus Urdu): D1 or D3? Spec v1 keeps natural mixing (`DA-D3-004`).
4. Ligatures: count as missed A5 (rule exists, math-only) or new for sci?

## Doc IDs read (reading order = shuffle order)

```
1b288da3558387bb
69bec7d4c7c5b24a
3f77b536af11bdbd
6e91bdbbe1990ddb
762c7fe6a66685eb
f59d8025602fb6fe
6bd9cff2867dde6b
ad508b9540cae798
8bed71bfbd032ac7
323d03a0c69a993a
75d3197ef1eebe14
ceea9671dd61d0b1
7c8d999405e165c6
bf7287e7e6c32f33
fefc24dc3b193463
1644d240cedb1491
897fae5366f5b7ee
e54d3904f1e15ffb
d28a54190b6172f4
e4f8b426552194c9
d20e4affc1d318fe
a3cb814a378ce805
e58d6a06c9323550
a98b897ff00b0312
e5c998e626aeaef9
f09fa390dfc047a5
469a6695b57d9130
94f54a2af0a8c897
92c5a95d0a91a30f
eadf39964af0daef
c8c25f100776816b
c700ec66afb3d047
67567a036e8d4fd1
a3d95f26c9633d06
e77b0eeab987db14
762ce1e85df1342d
af56bfd768132ef9
f8dbaec0629f41c8
16cad048dbebdcc9
5379152bb87b7ecf
e21c6a671baba032
5922156bd4540fca
164786e5556a0e4d
43645d783dfd0464
602dce872e8503a9
a5a9266112d5a2fb
f05b43fbbbe8521b
1b8d81ff2aaf0f84
24bebaf6ee460f90
a3f43f77758fc654
ded2213a089b6cec
989d3db03130079b
0fefaf2cff6b7ed2
8d2f5962911c666c
0aa110d861217aa0
4a6a549c3ead2ceb
0cfac7d22a005480
fa41800a38a7ef77
98d972c1b0b93185
48ffcdfea0bdbaf7
d14d57d2f7ae06c9
c7ff8fbe7f989839
ee09990471c8982e
f123f03d5b0db749
6431e058711852b7
441073cee66a7233
b128a5439be2b15d
1f9a9dec9551af98
3573a1f8671fc2bd
184464394a872e72
4768a98dfcf18b45
24eaaeb743718ed2
5877b6d32cedd1f5
b1c389beb28ee89f
8051a39408b38832
3f4118fde7d8d94b
d0ddef17c0bbb2b4
8602f8407bada7ee
57b3e0b1dbee7a09
3037356ebc3cacac
1cacb47f6b4869d6
7d96994fa5a2d678
7cb795516f8208f2
c5bd1e247b1ee501
5d35542043b6cc3b
feaaf0b7ad58777d
5b348d724c744d07
5d0d7f5ff717fd16
e84763c824faa7f9
768ce30e6fac2306
b77af8eafcdfabba
866a36a0781b6ed1
e19ea5618c4a4b2e
7d2c7a0354c9bf1b
766c83b5e63a0192
28b7763d4a13a37e
77f152ce4336c6d9
35041611f4dc7ec5
2f28fd430f65033c
2f16d7464105bfb1
```
