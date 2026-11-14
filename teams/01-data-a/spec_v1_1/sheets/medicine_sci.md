# medicine_sci: science medicines for the Spec v1.1 new diseases

**Draft, 12 Nov.** This sheet has one rule for each of the 6 new science diseases in `NEW_DISEASES.md`. These are diseases that aren't in `DISEASES.md` and that turn up in 2 or more docs, so they stay above the bar after Utkarsh's reading. Reasons and choices are in `decisions_sci.md`.

- **Pack.** sci-finepdfs has 202 gold docs (Data A gold).
- **Limit.** Docs changed plus dropped: at most `max(1, floor(1% of 202))` = 2 docs.
- **Rule form.** As the v1 sci rules: `doc = {doc_id, text, url, lang_label, pack_id}`, and `medicine(doc)` returns `(action, text)`. Each rule checks `pack_id` only, not `lang_label`, because sci-finepdfs holds more than one language. A doc with no `pack_id` passes unchanged.
- **Codes and IDs.** They come from `NEW_DISEASES.md`. Every code here is a proposed new code.
- **Run order.** These rules run after the sci family's v1 rules, because the reader found the diseases in v1-cleaned text. Within this sheet they run in section order: stage 10 (characters), then stage 30 (source), with the drops first and then the strips.
- **Every reading row so far is AI-suggested.** The hand check of the examples comes before a rule is final.

The last column counts the reading docs with the rule's code that it changes or drops, on the cleaned sample the reader read (`scripts/reading_recall.py`).

| # | ID | disease | action | stage | sci-finepdfs | reading docs treated |
|---|---|---|---|---|---|---|
| 1 | `DA-A7-001` | A7 word split at a line-end hyphen (NEW-hyphen-split-words) | fix | 10 | 0.50% (1/202) PASS | 3 / 3 |
| 2 | `DA-C9-002` | C9 SEO filler (NEW-seo-filler) | drop | 30 | 0.00% (0/202) PASS | 2 / 2 |
| 3 | `DA-B6-001` | B6 blank form fields (NEW-underscore-blanks) | drop | 30 | 0.00% (0/202) PASS | 2 / 2 |
| 4 | `DA-C7-003` | C7 content lost in extraction: table body or list (NEW-missing-table-data) | fix | 30 | 0.00% (0/202) PASS | 1 / 3 |
| 5 | `DA-C8-004` | C8 layout lost in extraction: reading order (NEW-reading-order) | fix | 30 | 0.00% (0/202) PASS | 1 / 2 |
| 6 | `DA-C8-003` | C8 layout lost in extraction: merged table cells (NEW-merged-table-cells) | fix | 30 | 0.00% (0/202) PASS | 1 / 3 |
| | all in order | sections 1–6 | | | **0.50% (1/202) PASS** | |

**Harm run, 12 Nov.** Report: `results/sheets/sci-finepdfs_harm.json`. Every rule passes, alone and in order.
- **The one gold doc touched.** `920855df51f37db8` (`DA-A7-001`) is a Finnish abstract with a word split at a line-end hyphen. So the disease itself is sitting in the gold set. This is within the limit, so it needs no ruling. It is recorded as a gold contradiction, like the two in finemath (aneeshads, 13 Nov), and flagged for a re-read at the next gold refresh.
- **An earlier fix.** The first version of `DA-B6-001` dropped 4 gold docs at a 10% threshold. It was raised to 30%.
- **Taken out.** A ligature rule (A5) and a table-of-contents rule (C1) came out on 12 Nov, because both diseases are in `DISEASES.md`.

Each section below follows the spec v1 card:
1. **Disease:** code, count and examples by doc_id.
2. **Medicine:** in words, then the python block.
3. **Action** and the packs it runs on.
4. **Harm test.**
5. **Side effects.**
6. **Spec v1.1:** status, author, source.

## 1. `DA-A7-001` · A7 word split at a line-end hyphen · fix · stage 10

1. **Disease:** NEW-hyphen-split-words, 3 docs, Aditeya. The line-end hyphen was kept and the line break became a space. Examples:
   - `5a201c19575522f9`: `company- owned`, `performance- based`
   - `75768b1b6b165557`: `stereo- scopic`
   - `93412124dc441459`: `day- to-day`
2. **Medicine:**
   - **What it joins:** a word of 2 or more letters, then `- `, then a lowercase word. The space goes and the hyphen stays: `stereo-scopic`. With no word list it can't tell `stereoscopic` from a real compound, so it never drops the hyphen.
   - **What it leaves alone:** suspended hyphens, where the next word is `and`, `or`, `nor`, `to`, `through`, `versus` or `vs` followed by a space (`pre- and post-test`, `two- to three-fold`).
   - **`day- to-day`** becomes `day-to-day`, because there `to` is followed by a hyphen.

```python
RULE_ID = "DA-A7-001"
CODE = "A7"  # proposed new code: a word split at a line-end hyphen (NEW_DISEASES.md)
ACTION = "fix"
STAGE = "10"
STATUS = "CANDIDATE"
ORIGIN = "sci-finepdfs"  # packs it runs on
PACKS = set(ORIGIN.split(", "))
SOURCE = "sci-finepdfs"  # pack whose reading found it (Aditeya, Week 2 reading audit)
AUTHOR = "aneeshads"
import re

A7_SPLIT = re.compile(r"\b([A-Za-z]{2,})- ([a-z]{2,})\b(-?)")
A7_SUSPENDED = {"and", "or", "nor", "to", "through", "versus", "vs"}  # "pre- and post-test", "two- to three-fold"


def _a7_join(m):
    left, right, hyphen = m.groups()
    if right in A7_SUSPENDED and not hyphen:
        return m.group()
    return left + "-" + right + hyphen


def medicine_a7(text):
    return A7_SPLIT.sub(_a7_join, text)


def medicine(doc):
    if doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    new = medicine_a7(doc["text"])
    return ("fix" if new != doc["text"] else "keep", new)
```

3. **Action:** fix. **Runs on:** sci-finepdfs. **Reading docs it treats:** 3 of 3, all fixed.
4. **Harm test:** sci-finepdfs 0.50% (1/202) PASS. The one gold doc it changes, `920855df51f37db8`, has the disease itself.
5. **Side effects:**
   - A hyphen used as a dash with no space before it is joined to the next word: `it was late- we left` becomes `late-we`.
   - A word that was really split, such as `stereo-scopic`, keeps a hyphen it shouldn't have.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Source: Aditeya's sci-finepdfs reading, mittalutkarsh/era-v5-lab#3. Drafted as `DA-A7-101`.

## 2. `DA-C9-002` · C9 SEO filler · drop · stage 30

1. **Disease:** NEW-seo-filler, 2 docs, Aditeya. These are "ebook" pages made by a text spinner: the keyword title is repeated, there's no real content, and some carry `Page n/N` markers. Examples:
   - `d8a1ed9b1607a1dd`: "If you ally dependence such a referred … book"
   - `365b8539483e5a3f`: "access free ebook", `Page n/33`

   Math has 15 more, with their own rule, `DA-C9-001`.
2. **Medicine:** drop a doc that matches either of these:
   - it has a sentence that only the ebook spinner writes: "If you ally dependence such a referred …", "You may not be perplexed to enjoy …", "every ebook collections … that we will extremely offer", "access/download/read free ebook", "acquire the enormously best seller from us", "Right here, we have countless books …";
   - it has 5 or more `Page n/N` markers and mentions "ebook".

```python
RULE_ID = "DA-C9-002"
CODE = "C9"  # proposed new code: SEO filler (NEW_DISEASES.md)
ACTION = "drop"
STAGE = "30"
STATUS = "CANDIDATE"
ORIGIN = "sci-finepdfs"  # packs it runs on
PACKS = set(ORIGIN.split(", "))
SOURCE = "sci-finepdfs"  # pack whose reading found it (Aditeya, Week 2 reading audit)
AUTHOR = "aneeshads"
import re

# sentences that only the "ebook" text spinner writes
C9S_SPINNER = re.compile("|".join((
    r"if you ally (?:dependence|need|obsession|craving|habit) such a referred",
    r"you may not be perplexed to enjoy",
    r"\bebook collections?\b[^\n]{0,120}\bthat we will (?:extremely|unconditionally|very|totally) offer",
    r"\b(?:access|download|read) (?:free|this) ebook\b",
    r"acquire the (?:enormously|extremely) best seller from us",
    r"right here, we have countless (?:books?|ebooks?)\b",
)), re.I)
C9S_PAGE = re.compile(r"\bPage \d+/\d+\b")
C9S_MIN_PAGES = 5


def medicine_c9s(text):
    if C9S_SPINNER.search(text):
        return True
    return len(C9S_PAGE.findall(text)) >= C9S_MIN_PAGES and "ebook" in text.lower()


def medicine(doc):
    if doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    return ("drop", doc["text"]) if medicine_c9s(doc["text"]) else ("keep", doc["text"])
```

3. **Action:** drop. **Runs on:** sci-finepdfs. **Reading docs it treats:** 2 of 2, both dropped.
4. **Harm test:** sci-finepdfs 0.00% (0/202) PASS.
5. **Side effects:**
   - A real paper that quotes one of these sentences would be dropped.
   - SEO pages that don't use one of the listed sentences are missed.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Source: Aditeya's sci-finepdfs reading, mittalutkarsh/era-v5-lab#3.

## 3. `DA-B6-001` · B6 blank form fields · drop · stage 30

1. **Disease:** NEW-underscore-blanks, 2 docs, Aditeya: `3b2c0420a33ccaab` (a blank questionnaire) and `d8e3f724095e4aaa` (a blank permit form). In both, long runs of underscores mark the blanks and make up much of the text.
2. **Medicine:** drop the doc when underscore runs (5 or more `_` in a row) make up 30% or more of its non-space characters. A blank form has labels but no content. The two reading docs are at 31% and 63%. The highest gold doc is at 26%: the first version, at 10%, dropped 4 gold docs.

```python
RULE_ID = "DA-B6-001"
CODE = "B6"  # proposed new code: blank form fields (NEW_DISEASES.md)
ACTION = "drop"
STAGE = "30"
STATUS = "CANDIDATE"
ORIGIN = "sci-finepdfs"  # packs it runs on
PACKS = set(ORIGIN.split(", "))
SOURCE = "sci-finepdfs"  # pack whose reading found it (Aditeya, Week 2 reading audit)
AUTHOR = "aneeshads"
import re

B6_RUN = re.compile(r"_{5,}")
B6_SHARE = 0.30  # the reading docs are at 0.31 and 0.63; the highest gold doc is at 0.26


def medicine_b6(text):
    chars = sum(1 for c in text if not c.isspace())
    blanks = sum(len(m.group()) for m in B6_RUN.finditer(text))
    return chars > 0 and blanks >= B6_SHARE * chars


def medicine(doc):
    if doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    return ("drop", doc["text"]) if medicine_b6(doc["text"]) else ("keep", doc["text"])
```

3. **Action:** drop. **Runs on:** sci-finepdfs. **Reading docs it treats:** 2 of 2, both dropped.
4. **Harm test:** sci-finepdfs 0.00% (0/202) PASS.
5. **Side effects:**
   - A worksheet with many fill-in lines would be dropped as well.
   - The margin is small: 31% for one reading doc against 26% for the highest gold doc.
   - A form whose blanks are boxes or dots, not underscores, is missed.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Source: Aditeya's sci-finepdfs reading, mittalutkarsh/era-v5-lab#3.

## 4. `DA-C7-003` · C7 content lost in extraction: table body or list · fix · stage 30

1. **Disease:** NEW-missing-table-data, 3 docs, Aditeya. A table or list was an image or a PDF object and was dropped, while the line that introduces it remained. Examples:
   - `0863ac75c9ec153e`: a table heading with no numbers
   - `4a33733776358fac`: "… is constituted with the following members." with no members after it
   - `fac0d63aba9fab0c`: action-plan column headings with no rows
2. **Medicine:** when a line ends by announcing a list of people ("the following members." / "the following persons:" and the like), and the next non-blank line is not a list, the dangling clause is cut and the sentence ends there: "… is constituted with the following members." becomes "… is constituted.". The next line counts as a list when it starts with a bullet, number, letter, `|`, or is shorter than 60 characters, as a name would be. Headings with no data can't be told from real headings, so they are left.

```python
RULE_ID = "DA-C7-003"
CODE = "C7"  # proposed new code: content lost in extraction (NEW_DISEASES.md)
ACTION = "fix"
STAGE = "30"
STATUS = "CANDIDATE"
ORIGIN = "sci-finepdfs"  # packs it runs on
PACKS = set(ORIGIN.split(", "))
SOURCE = "sci-finepdfs"  # pack whose reading found it (Aditeya, Week 2 reading audit)
AUTHOR = "aneeshads"
import re

C7T_NOUNS = r"(?:members|persons|names|people|officials|officers|staff|students|participants|candidates)"
C7T_INTRO = re.compile(r"[ \t]*\b(?:(?:with|of|are|is|as|by)[ \t]+)?the following " + C7T_NOUNS + r"[ \t]*[.:][ \t]*$", re.I)
C7T_LIST = re.compile(r"^[ \t]*(?:[-*|•●▪]|\(?\d{1,2}[.)]|\(?[A-Za-z][.)][ \t])")
C7T_SHORT = 60


def _c7t_is_list(line):
    return bool(C7T_LIST.match(line)) or len(line.strip()) < C7T_SHORT


def medicine_c7t(text):
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if not C7T_INTRO.search(line):
            continue
        nxt = next((l for l in lines[i + 1:] if l.strip()), None)
        if nxt is None or not _c7t_is_list(nxt):
            lines[i] = C7T_INTRO.sub(".", line)
    return "\n".join(lines)


def medicine(doc):
    if doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    new = medicine_c7t(doc["text"])
    return ("fix" if new != doc["text"] else "keep", new)
```

3. **Action:** fix. **Runs on:** sci-finepdfs. **Reading docs it treats:** 1 of 3 (`4a33733776358fac`).
4. **Harm test:** sci-finepdfs 0.00% (0/202) PASS.
5. **Side effects:**
   - A list that really does follow, written as running prose, loses its introducing clause.
   - Table headings with no data and other lost lists are left as they are.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Source: Aditeya's sci-finepdfs reading, mittalutkarsh/era-v5-lab#3.

## 5. `DA-C8-004` · C8 layout lost in extraction: reading order · fix · stage 30

1. **Disease:** NEW-reading-order, 2 docs, Aditeya:
   - `ef812364b6cca7a0`: numbered steps extracted as 3, 5, 4
   - `e4d1400f9236b8c9`: a sentence split into parts that are scrambled across lines
2. **Medicine:** in a block of 3 or more lines in a row that each start with a step number (`3. …`), check the numbers. If they are a set of consecutive numbers, each used once, but are neither ascending nor descending, the lines are put back in number order. Scrambled sentences can't be put back, so they are left.

```python
RULE_ID = "DA-C8-004"
CODE = "C8"  # proposed new code: layout lost in extraction (NEW_DISEASES.md)
ACTION = "fix"
STAGE = "30"
STATUS = "CANDIDATE"
ORIGIN = "sci-finepdfs"  # packs it runs on
PACKS = set(ORIGIN.split(", "))
SOURCE = "sci-finepdfs"  # pack whose reading found it (Aditeya, Week 2 reading audit)
AUTHOR = "aneeshads"
import re

C8O_STEP = re.compile(r"^[ \t]*(\d{1,2})\.[ \t]+\S")
C8O_MIN_BLOCK = 3


def _c8o_reorder(block):
    nums = [int(C8O_STEP.match(line).group(1)) for line in block]
    if sorted(nums) != list(range(min(nums), min(nums) + len(nums))):
        return block  # not one run of consecutive step numbers
    if nums == sorted(nums) or nums == sorted(nums, reverse=True):
        return block  # already in order, or counting down on purpose
    return [line for _, line in sorted(zip(nums, block))]


def medicine_c8o(text):
    lines, out, i = text.split("\n"), [], 0
    while i < len(lines):
        j = i
        while j < len(lines) and C8O_STEP.match(lines[j]):
            j += 1
        if j - i >= C8O_MIN_BLOCK:
            out += _c8o_reorder(lines[i:j])
        else:
            out += lines[i:max(j, i + 1)]
        i = max(j, i + 1)
    return "\n".join(out)


def medicine(doc):
    if doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    new = medicine_c8o(doc["text"])
    return ("fix" if new != doc["text"] else "keep", new)
```

3. **Action:** fix. **Runs on:** sci-finepdfs. **Reading docs it treats:** 1 of 2 (`ef812364b6cca7a0`).
4. **Harm test:** sci-finepdfs 0.00% (0/202) PASS.
5. **Side effects:**
   - A numbered list that is out of order on purpose (a list of rankings, say) is put back in number order.
   - Steps separated by blank lines or other text are not checked.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Source: Aditeya's sci-finepdfs reading, mittalutkarsh/era-v5-lab#3.

## 6. `DA-C8-003` · C8 layout lost in extraction: merged table cells · fix · stage 30

1. **Disease:** NEW-merged-table-cells, 3 docs, Aditeya. Examples:
   - `6cdcf2807cbf5156`: `| Female | 332 50.4 |  | 48.5 |`, where the count and the percent ended up in one cell and the next cell was left empty
   - `680400a6c14bde86`: a row with one cell too many
   - `6deaff0b02c57734`: a spec table flattened into separate lines

   Math has 11 more, with their own rule, `DA-C8-002`.
2. **Medicine:** in a Markdown table row, a cell that holds exactly two numbers and is followed by an empty cell is split, with the second number moved into the empty cell: `| 332 50.4 |  |` becomes `| 332 | 50.4 |`. Shifted rows and tables flattened into lines can't be put back, so they are left.

```python
RULE_ID = "DA-C8-003"
CODE = "C8"  # proposed new code: layout lost in extraction (NEW_DISEASES.md)
ACTION = "fix"
STAGE = "30"
STATUS = "CANDIDATE"
ORIGIN = "sci-finepdfs"  # packs it runs on
PACKS = set(ORIGIN.split(", "))
SOURCE = "sci-finepdfs"  # pack whose reading found it (Aditeya, Week 2 reading audit)
AUTHOR = "aneeshads"
import re

C8C_NUM = r"-?\d[\d,]*(?:\.\d+)?%?"
C8C_PAIR = re.compile(r"\|[ \t]*(" + C8C_NUM + r")[ \t]+(" + C8C_NUM + r")[ \t]*\|[ \t]*(?=\|)")


def medicine_c8c(text):
    out = []
    for line in text.split("\n"):
        if line.lstrip().startswith("|"):
            line = C8C_PAIR.sub(lambda m: "| " + m.group(1) + " | " + m.group(2) + " ", line)
        out.append(line)
    return "\n".join(out)


def medicine(doc):
    if doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    new = medicine_c8c(doc["text"])
    return ("fix" if new != doc["text"] else "keep", new)
```

3. **Action:** fix. **Runs on:** sci-finepdfs. **Reading docs it treats:** 1 of 3 (`6cdcf2807cbf5156`).
4. **Harm test:** sci-finepdfs 0.00% (0/202) PASS.
5. **Side effects:**
   - A cell that really holds two numbers, such as a range `10 20`, followed by an empty cell would be split.
   - Shifted rows (`680400a6c14bde86`) and flattened tables (`6deaff0b02c57734`) are not fixed.
6. **Spec v1.1:** status CANDIDATE, reviewer pending. Source: Aditeya's sci-finepdfs reading, mittalutkarsh/era-v5-lab#3.
