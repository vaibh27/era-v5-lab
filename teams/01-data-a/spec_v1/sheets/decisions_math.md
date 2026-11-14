# Math medicines: final decision

## Verdict

**Ready for team review. Not approved or frozen.** All 14 math rules (`specv1.md` medicines 45-58) are `CANDIDATE`, and
the shared F1 runs on both packs. Every rule meets criteria 1-5 of the spec README on every pack it runs on. What stands
between them and `APPROVED` is criterion 6: a review by someone other than the author, recorded with commit and digests.
B2 also has an open semantic question (below).

**Proposed reviewers.** 13 rules are SomaKorada07's functions (`AUTHOR = "SomaKorada07"`): kunal2016 compiled them and did
not write them, so he can review them. `DA-C5-001` (megamath drop) is kunal2016's: SomaKorada07 can review it. Her check on
theschoolofai/era-v5-lab#61 (1 Nov: same output as her functions on all 500 finemath docs, every scope as she intended) is
an author's check of the port, not the independent review.

**How the decision is made.** A rule is `APPROVED` when all six criteria below are met on every pack in its `ORIGIN`, and
the reviewer records it here with the commit; otherwise it stays `CANDIDATE`, or `BLOCKED` while it waits on an answer.

### Result per pack

`clean()` on each pack's own gold set, the same run as `results/cleaner/harm_tests.md` (2 Nov). Limit: changed + dropped at
most `max(1, floor(1% of gold))`. "Stress" adds the docs the gold builder left out as disputed (two clean readers and a third
who disagreed); it is not the official test.

| pack | gold harm | limit | result | stress | fixed / dropped / untouched | labelled rows treated |
|---|---|---|---|---|---|---|
| finemath | 0 / 86 | 1 | PASS | 0 / 86 PASS (no disputed docs: 2 readers) | 177 / 13 / 310 | 190 / 414 |
| megamath | 0 / 26 | 1 | PASS | 0 / 59 PASS (33 disputed docs) | 62 / 10 / 428 | 72 / 474 |

- **Every rule alone** is also 0 on both gold sets (`results/cleaner/harm_tests.md`, section 2, math).
- **0 hits is not 0 harm.** A gold set of n docs with 0 hits only bounds the harm rate below about 3/n: finemath 3.5%,
  megamath 11.5%. About 300 docs are needed to show 1%. This is why C1 on megamath waits for a spot-check.
- **Labelled rows treated** is low on megamath (72 / 474) by design: C1, C2 and C3 are off there (375, 159 and 282 rows), and
  most of its B5 (235) and G3/G4 rows cannot be repaired by a rule; those documents are `drop` by verdict.
- **A second `clean()` changes nothing** on any doc of either pack. No fix leaves a document empty or removes more than half
  of its text.
- Inputs: finemath labels SomaKorada07 #61, uz9ai #41; megamath labels mukund-23 #102, asraj #146, kunal2016 #151,
  AditeyaAItronics #172, curioustushar #175, mittalutkarsh #181 (latest in each open PR, 2 Nov). Gold: `teams/01-data-a/gold/`
  (sha256 finemath `633f6b6c370a32b4`, megamath `fe865ce345441423`). Packs: finemath `99aec0d9ece5d6c7`, megamath
  `07cdf1aa40a57f86`. `clean.py` `d3464bdaa394d1fb`, on `g1-specv1` `e7531f9`.

### Against the criteria

| criterion | source | result |
|---|---|---|
| Disease, medicine in words and code, action, harm test, side effects | `DISEASES.md` §4 | **Met.** Every section of `medicine_math.md` has all 5. Examples give `doc_id` and offset. |
| 1. Within the limit, alone and combined, per pack | spec README; TSAI `max(1 document, 1%)` | **Met.** 0 gold docs for every rule and for `clean()`, on both packs. |
| 2. Treats real rows | spec README | **Met.** Every rule treats labelled rows on each pack it runs on (table below); A6 is off on megamath because it treats none there. |
| 3. Does not damage what it keeps | spec README | **Met on these packs, one open question.** A5 never applies full NFKC (`x²` stays); D3 leaves Greek; C2 keeps repeated answer options; B2 skips documents with code; no fix empties a doc or removes over half of it. **Open:** B2 would turn `x<sup>2</sup>` into `x2`. It removes no `<sup>`/`<sub>` on either pack, but `specv1.md` lists this as a release gate for the full data. |
| 4. Idempotent | spec README | **Met.** Every rule alone and `clean()`: a second run changes nothing. C3 on megamath and the C3 order on finemath were chosen for this. |
| 5. Not resting on a dispute | spec README | **Met.** megamath 0 / 59 with its 33 disputed docs left in; finemath has none. |
| 6. Reviewed by someone other than its author | spec README | **Pending** for all 14. Proposed: kunal2016 reviews SomaKorada07's 13, SomaKorada07 reviews `DA-C5-001`. |
| One F1 rule, following the instructor's ruling (31 Oct) | TSAI rulings, 31 Oct and 1 Nov | **Met.** The sheet has no F1 block; the shared `tools/f1_rule.py` runs last on both packs (see F1 below). |
| No secrets or personal data | `DISEASES.md` §2a, pr-guard | **Met.** IDs, offsets and made-up test values only. |
| Digests, owner, reviewer, commit | Spec v1 §1 and §2.9 | **Partly.** Digests above; `AUTHOR` in every block. Reviewer and commit come at review. |

### F1 on the math packs

The shared rule, `spec_v1/tools/f1_rule.py`, extended to math and science by TSAI's decision of 1 Nov (#22). On math it takes
the prose path: a private person's phone, email and address become `<PHONE>`, `<EMAIL>`, `<ADDRESS>`; a page that is mostly
a personal listing is dropped; organisation and public-office contacts are kept. It runs last, after the math rules.

| pack | gold harm | docs it treats | F1 rows it treats |
|---|---|---|---|
| finemath | 0 / 86 PASS | 4 | 3 / 81 |
| megamath | 0 / 26 PASS | 26 | 26 / 43 |

Most finemath F1 labels were given before the ruling to organisation and course-site contacts, which the ruling keeps; those
rows should be relabelled. Known misses: a private phone in running prose with no label, numbers outside North-American form
and outside people tables, and addresses not in "number + name + street type" form.

### What changed in this round

1. **v1.1:** the finemath rules are SomaKorada07's own functions (#61, `cc74c7e`), copied unchanged; her A6 and D3 added (#20).
2. **Gold refreshed** to 86 / 26 (31 Oct), and the sheets updated to it (#22).
3. **Shared F1 on math** (#22), and **SomaKorada07's review of the port** on #61.
4. **This verdict** added in the form of `decisions_code.md` and `decisions_english.md`: stress run, criteria, F1 section.

### Questions for the team

| question | what the answer changes |
|---|---|
| B2 on maths markup: is `x<sup>2</sup>` to `x2` acceptable, or should B2 turn `<sup>`/`<sub>` into `^`/`_`? | B2 can be `APPROVED` as it is, or needs a small change first. No doc in either pack is affected today. |
| C1 on megamath for v1.2 | After the spot-check of 15 of the 72 docs it would change (SomaKorada07's condition on #61). |
| Does Gate 2 need `APPROVED` statuses, or `CANDIDATE` with this verdict? | Whether the reviews above must be recorded before 14 Nov. |

### Problems to report to the team

- **G3/G4 on megamath:** hyphens and minus signs are lost across the pack (16 G3, 67 G4 rows). No rule can restore them.
- **F1 labels from before the ruling** inflate the F1 row count on finemath (81 rows, 3 treated). Relabelling would also add
  gold docs.

### Limits

- 500-doc samples, and both gold sets are far under the 200 target (86, 26): finemath because its 2 readers disagree on 276
  docs, megamath because the pack is genuinely dirty (183 docs have even one clean vote).
- The finemath rules were written on finemath. On the full data, site lists (C1) and keyword lines (C5) will miss sites not in
  the sample.

## The decision for each disease

Cell numbers: labelled rows the rule treats / labelled rows with that code, per pack, from the 2 Nov labels; gold = gold docs changed or dropped.

| code | rule(s) | packs | decision | why |
|---|---|---|---|---|
| A3 | `DA-A3-001` | both | keep | finemath 10/11 (v1: 9; her cp866 repair and the extra lead characters), megamath 8/35, 0 gold. Runs first: a repaired run can hold an invisible character, which A1 then removes. |
| A1 | `DA-A1-001` | finemath | keep | 14 of 15 labelled rows, 0 gold. megamath has no A1 rows. |
| A6 | `DA-A6-001` | finemath | keep (new) | finemath 5/9, 0 gold: the Symbol-font map for 46 codes. Changes nothing on megamath (0 of its 6 A6 rows), so it is off there. |
| A5 | `DA-A5-001` | both | keep | finemath 6/6, megamath 9/12, 0 gold. Ligatures and fullwidth punctuation only; never full NFKC (x² must stay). |
| B2 | `DA-B2-001` | both | keep | finemath 3/8, megamath 6/137, 0 gold. Skips documents that hold code; most megamath B2 rows do. It removes no `<sup>`/`<sub>` tag on either pack. |
| B4 | `DA-B4-001` | both | keep | finemath 9/18, megamath 2/19, 0 gold. Leaves `<EMAIL>`, the F1 tag. |
| B5 | `DA-B5-001` | both | keep | finemath 5/28 (7 docs changed: case-insensitive BBCode), megamath 1/235, 0 gold. Formulas stripped by the scraper cannot be repaired; those documents are `drop` by verdict. |
| C5 | `DA-C5-001` (drop), `DA-C5-002` (fix) | drop: megamath; fix: both | keep both | They fix different rows: the drop takes answers.com "related questions" pages (megamath 10); the fix removes "See more:" and title-case keyword lines. C5 together: finemath 5/50, megamath 12/180. |
| D3 | `DA-D3-001` | both | keep (new) | finemath 6/12, megamath 3/20, 0 gold. Removes lines that are mostly Cyrillic, Indic, CJK, kana, Hangul or Arabic; drops a page where those hold over 30% of the letters. Greek is left alone: it is maths. |
| C3 | `DA-C3-001` (drop), `DA-C3-002` (fix) | finemath | keep, as two blocks of one medicine | finemath 15/49, 0 gold. The preview drop runs before C1, which would otherwise delete the banner it reads; the teaser trim runs after the strips. Off on megamath: there the trim is not idempotent on one doc. |
| C1 | `DA-C1-001` | finemath | keep | finemath 109/329 (v1: 47; her full `C1_LINE`, `C1_SITE` and phpBB block), 0 gold. Off on megamath because, as she says, the site list was fitted on finemath; on megamath (375 C1 rows) it would change 72 docs, all labelled C1, with 0 of 26 gold docs changed. She agreed (#61) to try it in v1.2 after a spot-check of the deleted lines on 10-15 of the 72 docs, mainly whole-line phrases such as Share, Views, Recommended, Tags:. |
| C2 | `DA-C2-001` | finemath | keep on finemath, off on megamath | finemath 40/187 (v1: 39; keeps repeated answer options and code-like passages, ruling 13), 0 gold. On megamath it changes 1 of 26 gold docs: inside the limit, but it would use all of it. |
| F1 | shared `../tools/f1_rule.py` | both | shared | One F1 for all families, per the TSAI ruling. On math it runs the prose path, last (TSAI, 1 Nov): 0 of 86 and 0 of 26 gold docs; it treats 4 finemath and 26 megamath docs: finemath 3/81, megamath 26/43 F1 rows (most finemath F1 labels are organisation contacts, kept under the ruling). |
| C4, D4 | none | | no rule | finemath C4 7, D4 22; megamath C4 58, D4 7 rows. Her sheet: a tested C4 rule caught 0 labelled cases and changed 2 clean docs; D4 (spun text) needs a language model. |
| G3, G4 | none | | open | megamath G3 16 and G4 67 rows. The pack loses hyphens and minus signs pack-wide (`2^n1` for `2^n-1`). No rule can restore them; documents whose maths became wrong are `drop` by verdict. |

Reasons for every scope and every rule left out are in the table above; the rule code is in `medicine_math.md` and `specv1.md` medicines 45-58.
