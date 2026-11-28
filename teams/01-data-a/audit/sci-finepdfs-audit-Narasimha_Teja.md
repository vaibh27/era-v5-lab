# sci-finepdfs audit: issues to fix

This file lists the problems found while auditing the sci-finepdfs pack, as of 2026-11-22. Reviewer: Narasimha Teja (`narasimhatejav`). Assignment 12: 35 docs from Part_1.

35 docs were read side by side: the text before cleaning (`Cap1-packs/sci-finepdfs/Part_1/part-00000.parquet`) and after cleaning (`tokenizer_sample_v0_clean/train/sci-finepdfs/part-00000.parquet`). Each doc was labelled in `sci-finepdfs-Part_1-Narasimha_Teja.csv` as `clean`, `fix` or `drop`, with a disease code and a note where one was given. Disease codes follow `era-v5-lab/diagnosis/DISEASES.md`.

**How the 35 docs were chosen.** The cleaner changed only 52 of the 6,011 docs in this pack and dropped none. The 35 were drawn at random from those 52 (`random.seed(20261122)`), as "Only changed" and "Random next" in the viewer ask.

Docs labelled `clean` are left out. What remains is every doc that needs attention: 6 docs, 4 marked `fix` and 2 marked `drop`. The other 29 are `clean`.

The issues fall into four groups:
- **Licence lines and cover pages left at the top.** This is the largest group.
- **A repeated copy left behind by the cleaner's repeat cut.** This is the only damage caused by cleaning.
- **Soft hyphens left inside words.**
- **Docs to drop.** One is a menu with almost no prose, and one is a Japanese doc labelled English.

## Summary

| Issue | Docs |
|---|---:|
| Licence lines and cover pages left (C1) | 3 |
| Repeated copy left by the C2 cut (C2) | 1 |
| Soft hyphen U+00AD left in words (A1) | 1 |
| Menu with no prose (C1, drop) | 1 |
| Wrong language label (D1, drop) | 1 |
| **Total** | **6** |

One doc, `079ce87c3d323ea7`, is in two groups (C1 and A1), so the rows add up to 7 for 6 docs. The C2 issue is the only damage caused by cleaning. It is small in the sample but comes from how the rule cuts text, so it is the most important to fix (see below).

## Licence lines and cover pages left (C1) (3)

**What it is.** Archive and journal front matter is left in the cleaned text:
- `c31993065f746189`: the cleaner removed the HAL cover page (`DA-C1-007`) but not the line printed just after it, so the doc now starts with `Distributed under a Creative Commons Attribution 4.0 International License`. It also has a lone `(Continued on next page)` line at L103.
- `c52dd9e603defa7d`: it starts with a university-repository cover page from a different archive (CURIS), which `DA-C1-007` does not match. The cover runs from L1 to L25 (`Document version … Document license … Download date: 21. apr.. 2021`). It also has a stray journal-header fragment at L27, the Springer page marker `1 3` at L199, and licence and "Publisher's Note" lines at L184 and L216.
- `079ce87c3d323ea7`: the journal's running footer `Rev Panam Salud Publica` / `43, 2019 | www.paho.org/journal | https://doi.org/…` repeats at L165-167 and L195-197, and a licence paragraph sits at L34.

**How big it is.** Across the cleaned pack:
- 2 docs start with the left-over `Distributed under a Creative Commons …` line;
- 6 docs have a `Download date:` repository cover in their first 2,000 characters;
- 3 docs have `(Continued on next page)`.

**What to do.**
- Extend the HAL cover rule so it also removes a Creative Commons line that directly follows the cover (Medicine 2 below).
- Add the CURIS cover layout (`Download date:` near the start) to the cover patterns of `DA-C1-007`.

| Pack | doc_id | Verdict |
|---|---|---|
| sci-finepdfs / Part_1 | `c31993065f746189` | fix |
| sci-finepdfs / Part_1 | `c52dd9e603defa7d` | fix |
| sci-finepdfs / Part_1 | `079ce87c3d323ea7` | fix |

## Repeated copy left by the C2 cut (C2) (1)

**What it is.** The doc is a wine shelf talker: one tasting note of about 200 characters, printed 4 times in the raw. The C2 rule `DA-C2-004` (`_sci_c2_prefix` in `spec_v1_1/clean.py`) finds a second copy of the first 64 characters and removes it character by character, up to the first character where the two copies differ. It does not check where that point falls. Here the cut runs past the end of one copy and takes the heading of the next. After cleaning, 3 copies are still there, and one has no heading (L8 blank, then `ALEXANDER VALLEY` at L9).

**How big it is.** Across the whole pack, `DA-C2-004` changes 10 docs, and 3 of its cuts end in the middle of a line:
- `f0bf0c7911813d28` keeps the fragment `ten 10 day grace period, the absence will be declared illegal/unexcused.`;
- `6c949f549ff6672a` loses the "I" of `I ADD THE HEMP HEARTS…`;
- `0f3c6cb6760a400d` is cut inside a table row.

The first two are in this sample but are labelled `clean` by the reviewer, as low-value form and survey text.

**What to do.** Replace `_sci_c2_prefix` with Medicine 1 below, which removes only whole paragraphs. For this doc, keep one copy of the note and remove the others.

| Pack | doc_id | Verdict |
|---|---|---|
| sci-finepdfs / Part_1 | `335a56d0e87a73a9` | fix |

## Soft hyphen U+00AD left in words (A1) (1)

**What it is.** Invisible soft hyphens from the PDF are left inside words and URLs, for example `cola­ boradores` (L274) and `refe­ rente` (L284). The doc has 5 of them, the same number as in the raw. The sci A1 rule `DA-A1-006` removes only U+200B and U+FEFF.

**How big it is.** 87 docs in the cleaned pack contain U+00AD, 3,155 characters in total.

**What to do.** Add U+00AD to `DA-A1-006`. Where a soft hyphen is followed by a space (`cola­ boradores`), remove the space too, so the word is joined back together.

| Pack | doc_id | Verdict |
|---|---|---|
| sci-finepdfs / Part_1 | `079ce87c3d323ea7` | fix |

## Menu with no prose (C1, drop) (1)

**What it is.** The doc is a saloon catering "price list" with no prices. From L23 to the end (L100) it is item names in capitals (`APPETIZERS`, `CHICKEN TAQUITOS`, `-BUFFALO`, `-CHIPOTLE GLAZED`). Only L13-19 are sentences: four lines of venue marketing. The phone number and address at L21 belong to the business, not a private person.

**What to do.** Drop it. No rule in the pack catches menus, and this is the only one in the sample that the reviewer kept as `drop`.

| Pack | doc_id | Verdict |
|---|---|---|
| sci-finepdfs / Part_1 | `f5332b3e8dc5894a` | drop |

## Wrong language label (D1, drop) (1)

**What it is.** The doc is labelled `en`, but it is mostly Japanese: 6,480 Japanese characters against 4,858 Latin letters. It is a 1914 Foreign Ministry archive of diplomatic telegrams in kanji and katakana, with a few English memoranda.

**How big it is.** It is the only doc in the cleaned pack where Japanese, Chinese or Korean characters outnumber Latin letters.

**What to do.** Drop it from the English science pack, or move it to the right language.

| Pack | doc_id | Verdict |
|---|---|---|
| sci-finepdfs / Part_1 | `29c7e818ebe025b0` | drop |

---

## Medicines

DISEASES.md asks for a medicine for each disease found more than five times. No disease reaches that in the final labels. These two medicines are given anyway, because one fixes the only damage caused by cleaning in this pack and the other fixes the most common leftover.

**Harm test.** The test needs the 500-doc `sci-finepdfs.parquet` that the gold set was read from. That file is not in this checkout, and the week-1 copy in `Cap1-packs` holds none of the 202 gold documents. As a check, each medicine was run on the whole Part_1 pack (6,011 docs) to count the docs it changes. The sheet also loads and runs in `harm_test.py`. **The official gold-set test is still to run:**

```
python -I teams/01-data-a/harmtest/harm_test.py \
  --pack <path>/sci-finepdfs.parquet \
  --gold teams/01-data-a/gold/sci-finepdfs_gold.csv \
  --rules teams/01-data-a/audit/sci-finepdfs-audit-Narasimha_Teja.md \
  --rule all
```

### Medicine 1

1. **Disease:** C2. Three docs show the problem, all damaged by the current `DA-C2-004`:
   - `335a56d0e87a73a9`: one copy has lost its heading;
   - `f0bf0c7911813d28`: the cut leaves the fragment `ten 10 day grace period, …`;
   - `6c949f549ff6672a`: L13 starts ` ADD THE HEMP HEARTS`.
2. **Medicine:** a replacement for `_sci_c2_prefix` that removes only whole paragraphs. It keeps the same search: the first 64 characters must appear again within the first 2,500, and the copy must run for at least 200 characters. Two checks are added:
   - the copy must start at the beginning of a line;
   - if the copies differ before the end of the doc, the cut moves back to the last blank line inside the copy.

   ```python
   def medicine_c2(text):
       """Remove a long exact second copy of the opening passage (as DA-C2-004), but only whole paragraphs:
       the copy must start at a line start, and if the copies diverge the cut goes back to the last blank line."""
       seed_len, min_repeat, search_limit = 64, 200, 2500
       if len(text) < 2 * min_repeat:
           return text
       second = text.find(text[:seed_len], seed_len, min(len(text), search_limit))
       if second < 0 or text[second - 1] != "\n":
           return text
       m, mx = 0, min(second, len(text) - second)
       while m < mx and text[m] == text[second + m]:
           m += 1
       end = second + m
       if end < len(text) and text[end - 2:end] != "\n\n":
           brk = text.rfind("\n\n", second, end)
           if brk < 0:
               return text
           end = brk + 2
       if end - second < min_repeat:
           return text
       return text[:second] + text[end:]
   ```
3. **Action:** fix the text. Use it in place of `_sci_c2_prefix` in `DA-C2-004`.
4. **Harm test:** not run yet (see above). On the whole raw pack it changes 9 docs, where `DA-C2-004` changes 10:
   - it leaves the table row in `0f3c6cb6760a400d` alone;
   - in the other 9 docs every cut ends on a blank line;
   - `335a56d0e87a73a9` keeps every heading;
   - `f0bf0c7911813d28` keeps its last paragraph whole, with no fragment;
   - `6c949f549ff6672a` keeps "I ADD THE HEMP HEARTS…".
5. **Side effects:**
   - It removes less than `DA-C2-004`: a copy that differs from the first one in its first paragraph is now kept whole.
   - It still removes only the second copy of the opening, so repeats further down a doc are not touched. For example, the repeated abstract in `c18182b91c9c6c3a` stays.

### Medicine 2

1. **Disease:** C1. Three examples:
   - `c31993065f746189`: L1 is `Distributed under a Creative Commons Attribution 4.0 International License`, left after the HAL cover was removed;
   - `8b2f0d9c5a7b3d41`: L1 is the same kind of line;
   - `7a2569f0a3cbed33`: the BMJ running footer `Saville C, et al.` / `BMJ Qual Saf` / `2021;` / `30` / `:7–16. doi:10.1136/bmjqs-2019-010569` appears 9 times.

   The last two are labelled `clean` in this review: the reviewer judged them acceptable. The footer part of this medicine, rule (b), is therefore optional.
2. **Medicine:** two narrow rules, in one function.
   - **(a) Licence line.** Remove a `Distributed under a Creative Commons … License` line only when it is the very first line of the doc, which is where `DA-C1-007` leaves it.
   - **(b) Running footer.** Remove a `:pages. doi:10.…` line that repeats 3 or more times. With it, remove up to 4 short lines (40 characters or less) just above it, but only those that also repeat 3 or more times: author "X, et al.", journal, year, volume.

   ```python
   _C1_CC_LEAD = re.compile(r"\A\s*Distributed under a Creative Commons[^\n]{0,120}License\s*\n")
   _C1_DOI_FOOT = re.compile(r"^:\d+[–-]\d+\. doi:10\.\S+\s*$")

   def medicine_c1(text):
       """(a) a Creative Commons line left at the very start after the HAL cover page is cut (DA-C1-007);
       (b) BMJ-style running page footers: the ':7–16. doi:...' line, repeated 3+ times, and the short lines
       just above it (author 'X, et al.', journal, year, volume) when each of them also repeats 3+ times."""
       text = _C1_CC_LEAD.sub("", text, count=1)
       lines = text.split("\n")
       counts = {}
       for l in lines:
           counts[l.strip()] = counts.get(l.strip(), 0) + 1
       anchors = [i for i, l in enumerate(lines) if _C1_DOI_FOOT.match(l.strip()) and counts[l.strip()] >= 3]
       if not anchors:
           return text
       kill = set()
       for i in anchors:
           kill.add(i)
           j, taken = i - 1, 0
           while j >= 0 and taken < 4:
               s = lines[j].strip()
               if not s:
                   kill.add(j); j -= 1; continue
               if len(s) <= 40 and counts[s] >= 3:
                   kill.add(j); taken += 1; j -= 1; continue
               break
       return "\n".join(l for i, l in enumerate(lines) if i not in kill)
   ```
3. **Action:** fix the text. Run it after `DA-C1-007`.
4. **Harm test:** not run yet (see above). On the whole pack:
   - **Raw pack:** it changes 2 docs, the two BMJ articles (`cd6b4946f9dc02b1`, `7a2569f0a3cbed33`). Rule (a) cannot fire on raw text, because the HAL cover still comes first.
   - **Cleaned pack:** it changes 4 docs, the 2 BMJ articles and the 2 HAL docs.
   - **Checked by hand:** in the BMJ articles it removed only the footer lines (31 and 45 non-blank lines); in the HAL docs it removed only the licence line.
5. **Side effects:**
   - Rule (b) only knows the BMJ layout, which ends in `:pages. doi:`. Other running headers are not touched, such as `Rev Panam Salud Publica` in `079ce87c3d323ea7` or the Springer `1 3`.
   - Bare page numbers are left, because a line of digits can be table data.
   - A real line of 40 characters or less that repeats 3 or more times just above a footer would be removed. None was seen.

This report includes no personal data. Emails, phone numbers and the names of private people are referred to by line only.
