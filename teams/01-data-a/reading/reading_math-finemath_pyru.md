# Math reading audit (task 9): math-finemath

Reader: pyru. Row 9 of the Week 2 assignments, 100 docs (math-megamath is read by the partner).
The findings are in `reading_math-finemath_pyru.csv`: one row per finding, 184 rows, positions only.

## How it was read

- **Pack file:** tokenizer sample v0-clean, `train/math-finemath/part-00000.parquet` (Part 1; math-finemath has no Part 2).
  28,522 docs, sha256 `7b3366baf9b8b756…`, as listed in `cleaned_manifest.json`.
- **Cleaner:** spec v1 as run by TSAI on #244: `clean.py` sha256 `97c1a55b…` at `8da2ac4`. Rule IDs in this file
  are the ones that ran in that job (for example `DA-C2-001`, `DA-D3-005`), not the v1.1 sheet numbers.
- **Seed and pick:** 20261101, the `pick_docs.py` method, rows 1-100: the doc_ids in file order, shuffled with
  `random.Random(20261101)`, first 100. The list is at the end of this file.
- **Docs read:** 100. The cleaner changed 33 of them: `DA-C2-001` 17, `DA-C1-001` 14, `DA-A1-005` 3, and
  `DA-A3-001`, `DA-A5-002`, `DA-A6-003`, `DA-B2-001`, `DA-D3-005` 1 each. These come from a rule-by-rule trace with the
  same math rules, and the trace output matches the job output on all 100 docs.
- **Method: AI-assisted.** Claude (Anthropic) made a first pass over all 100 docs and suggested 184 findings. In a
  local reading page I accepted all 184; I rejected none and added none. Every row therefore has `source` = `ai`.
  The 9 docs with no suggestion have no rows.
- **Offsets:** code points in the cleaned text, end exclusive. A whole-doc finding is `0-<length>`. A harm row gives
  the point in the cleaned text where the rule removed the text (`p-p`).
- **quote:** left empty in every row. The sample holds personal data, so no document text goes into the repo;
  `why` is written in my own words.
- **Replaces PR #27** on patilvijay23/era-v5-lab, which read a different 100 picked with `random.sample` and is now
  closed. Only 1 doc is in both samples (`2842cc032aca51f3`).

## Findings table

184 findings in 91 docs; the other 9 docs have no finding. Kinds: missed 117, new 64, harm 3.

| code | kind | docs affected | % of docs read |
|---|---|---|---|
| C1 | missed | 68 | 68% |
| NEW-missing-formulas | new | 15 | 15% |
| NEW-flattened-formulas | new | 14 | 14% |
| B5 | missed | 8 | 8% |
| C2 | missed | 8 | 8% |
| NEW-merged-table-cells | new | 7 | 7% |
| NEW-seo-filler | new | 7 | 7% |
| A3 | missed | 5 | 5% |
| C5 | missed | 5 | 5% |
| NEW-template-pages | new | 5 | 5% |
| NEW-ai-generated-text | new | 4 | 4% |
| NEW-missing-spaces | new | 3 | 3% |
| B2 | missed | 2 | 2% |
| C3 | missed | 2 | 2% |
| D4 | new | 2 | 2% |
| DA-C2-001 | harm | 2 | 2% |
| A4 | new | 1 | 1% |
| B1 | new | 1 | 1% |
| D2 | new | 1 | 1% |
| DA-D3-005 | harm | 1 | 1% |
| E2 | new | 1 | 1% |
| NEW-hyphen-split-words | new | 1 | 1% |
| NEW-interleaved-columns | new | 1 | 1% |
| NEW-orphan-figure-labels | new | 1 | 1% |

Where the existing rules miss (from the `missed` rows):

- **C1 (68 docs):** forum post headers and user-info blocks, Q&A comment timestamps, related-question and
  related-document blocks, breadcrumbs, adverts and sign-up widgets.
  - The C1 line rule needs a whole-line match, so it misses chrome lines that carry a markdown heading prefix (a
    similar-questions heading, tag toggles) in `4ffb288e0df06071` and `885bc9229d4bb0bf`.
  - In `c63a6dd114e246eb` its line filter removes the reply/view labels before its own reply-view block pattern
    runs, so bare counts are left.
- **C2 (8 docs):** quoted forum posts that repeat earlier posts. The quote is joined to the reply text or split into
  short blocks, so neither the line rule nor the passage rule sees it.
  - Also, the keep-first copy of a UI button line stays (`053c7e6dfbdb4e46`, `fcc79d8f03cf663e`, counted as C1).
- **A3 (5 docs):** mojibake from other code pages: Cyrillic cp1251, Greek cp1253, Thai cp874 and Central
  European cp1250. The rule repairs only cp1252 and cp866. In `6647e054e451c4dc` about 20 emojis became a lone pi
  sign, which reads as maths.
- **B5 (8 docs):** units and exponents broken by a site's math converter (3 docs), site-specific math delimiter tags,
  unescaped currency dollars between prices, and prose wrapped whole in a code fence (2 docs).
- **C5 (5 docs):** listing and search-result pages, plus a lower-case keyword list and a see-more link in the middle of
  a line. The C5 rule needs Title-Case phrases at the start of a line.

## New diseases

Two or more docs out of 100 is at or above the 1% bar, so the first eight go to spec v1.1.

| name | what it is | docs |
|---|---|---|
| NEW-missing-formulas | formula or number images and symbols dropped by the scraper; the text points to a formula that is not there | 15 |
| NEW-flattened-formulas | superscripts, subscripts and fractions flattened to plain digits, or split one token per line | 14 |
| NEW-merged-table-cells | tables run into one line or broken one cell per line, so values lose their row and column | 7 |
| NEW-seo-filler | SEO pages and paragraphs: question headings with pasted answers, site-wide filler text | 7 |
| NEW-template-pages | pages generated from a template for one number or unit value | 5 |
| NEW-ai-generated-text | machine-written articles or AI summaries and FAQs, some with false statements | 4 |
| NEW-missing-spaces | words and sentences run together where line breaks were lost | 3 |
| D4 (no math rule) | spun text: a rewriter swapped words for odd synonyms | 2 |
| A4 (no math rule) | symbols replaced by question marks | 1 |
| B1 (no math rule) | HTML entity left in the text | 1 |
| D2 (no math rule) | romanised Hindi comments | 1 |
| E2 (no math rule) | hash values as a data blob | 1 |
| NEW-hyphen-split-words | print line-end hyphens left inside words | 1 |
| NEW-interleaved-columns | two-column PDF pages read across both columns, so sentences of two exercises alternate | 1 |
| NEW-orphan-figure-labels | bullet lists of figure labels left from an interactive scene | 1 |

## Harm

| rule | what it deleted | docs |
|---|---|---|
| `DA-C2-001` | real lines that repeat by design. In `f85486e0d9ad0a50` it deleted 30 survey trial lines, leaving 8 trials empty. In `3bc7339b3365a201` it deleted 7 of 8 copies of a curve equation, including a working step and the line a later reply points to. | 2 |
| `DA-D3-005` | 98 formula and number lines (hypotheses, test statistics, data values) in `1eadf4cf278db7e6`. The PDF text is cp1251 mojibake, so its Greek letters and signs count as Cyrillic. | 1 |

The earlier, closed read of a different 100 (PR #27) also saw `DA-C2-001` delete program output and numbered-problem
lines in 4 docs.

## Open questions for the lead

1. **A3 and D3:** extend the A3 repair to cp1251, cp1253, cp874 and cp1250? A3 already runs before D3. While A3
   cannot repair cp1251, D3 deletes those lines as Cyrillic, and that is where the harm in `1eadf4cf278db7e6` comes
   from.
2. **C2 line rule:** limit it (for example skip lines with digits or `$`, or lines inside numbered items)? It harms 2
   docs here, and 4 more in the earlier read.
3. **C1:** allow an optional markdown heading prefix in the whole-line phrases, and run the reply-view block pattern
   before the line filter?
4. **Listing, template, SEO and AI-written pages:** drop them by rule, or leave them to the quality filter?
5. **quote column:** left empty because of the personal-data rule. Is that fine for the consolidation?

## Doc IDs read (reading order = shuffle order)

```
f85486e0d9ad0a50 02265dbc8fa20259 7aa25f450cbc9b70 1ab888d9366a43fd fe843a72092064cd
0610be9612d18bc6 ee9ac5fd44a100b7 56aadb5db0327910 9e9292036542a1b0 9373e806c8ee0ac8
54656567cb9c8e38 32b1f031d8c464e4 0815e685a1a74bda f19cd0cc87cb399a a48d37861a2ab782
8f098ccbede3a46d 920808a906ef229b dcd6058ecb2cfc18 743e8c255952e983 4ffb288e0df06071
8d724d02426ad3ff 7872171b9bf54929 2334d216e03d1058 789dfcb4222e401e 767037306d97bc86
42c81cfd52a5849d d72a2a98549b9142 c63a6dd114e246eb 9453359091e63dfc a77ec4895195a433
facfd603bc865298 afa476465dcc2169 25e224253f9a5c0a 26bf10df32e94547 d805befbd984308c
9dd2fc49a23b7e73 10d547035dd610e2 053c7e6dfbdb4e46 fd804060f7a299ac 4ac56ea0cf68007b
6647e054e451c4dc b1925e5c2908520f e7211e16ef44de51 4e9a082228b0c45e 885bc9229d4bb0bf
e492f103e239bec0 f09378316ce939e1 a8004bba3cd5bbfb dc76a8ad4d35952e ec20a4ca6692ddb7
63783f04778f7000 8d53fca95d65362b 26fc94c2c763ffaf 291cff41d0672085 2fb9464195bf2edc
cb9a946cd4eba389 322fa2d2c8fd6470 3567192e9fc75b00 90d9ff5ef9c45a37 6712725af9da5453
a91801046f79b716 4783c724f8fbfa52 31a477d3965aebee a345d577da510a26 d68e2a6365d972da
d3c81797986b7e17 473eccb431e6029a 0499a6d6c38ef86f 8d88bdd73268baaa b47ab81de0a41d92
02b5f8e058694237 06f2d75b8c5610f1 fdb1238e2b5a4d0c 9c393ad230dc838b a919c1b20d1178f0
e71f555ad2f37d57 28f56e454b2ade39 1eadf4cf278db7e6 1f2abff491b96656 3bc7339b3365a201
3825a12534054bb3 0a34dcef3fe6b07a 6aea632beaa6aacf 9b89f0605fe52eea f3a2d01395300022
fcc79d8f03cf663e 379ab92860d593f5 d0693c18056bae02 4f2da70883d8d65f ec4ea0009df33e1a
0614bc6ecf034495 c7e8da53587e0e2a 6f08675347237529 74b959b3e8c2b31b 2842cc032aca51f3
033ab42a96110ff9 cc0bf2c48f0be1a8 ec7ecf0ffc763747 5073e2edc460a575 cdccda71cb0f4ff3
```

Docs with no finding (no rows): `56aadb5db0327910`, `b1925e5c2908520f`, `e7211e16ef44de51`, `26fc94c2c763ffaf`,
`322fa2d2c8fd6470`, `31a477d3965aebee`, `0499a6d6c38ef86f`, `02b5f8e058694237`, `fdb1238e2b5a4d0c`.
