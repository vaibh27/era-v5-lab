# code-java audit: issues to fix

This file lists the problems found while auditing the code-java pack, as of 2026-11-23, and the medicines proposed for two of them.

35 docs were read side by side: the text before cleaning (`audit_labels/data/pre/code-java/`) and after cleaning (`audit_labels/data/post/code-java/`). Part_1 has 25 docs and Part_2 has 10. Only docs that cleaning changed or dropped were sampled (the viewer's default), so unchanged docs are not covered. Each doc was labelled in `audit_labels/code-java_<part>.csv` (copies in `teams/01-data-a/audit/code-java_<part>-smitasasindran.csv`). The verdict answers "was the cleaner's output right?": `clean` (yes), `fix` (no, something is wrong in the output), or `drop` (the doc was removed, and that was right). Disease codes follow `era-v5-lab/diagnosis/DISEASES.md`.

Docs labelled `clean` are left out: 33 docs. What remains is 2 docs, both marked `fix`. No doc in this sample was dropped by cleaning.

Cleaning did these things correctly in the sample, and they are not issues: it removed the dataset marker line (`<gh_stars>…`, `<reponame>…` or `<filename>…`) at the top of 28 docs; it removed a long licence header from 15 docs; it removed a `Created by … on …` or author-only comment from 6 docs. The dataset's `<NAME>` placeholder is gone from all 6 docs that had one (3 became `AUTHOR`, 3 went with a removed header), and no `<EMAIL>` or `<NAME>` marker is left in any of the 35 cleaned docs.

All medicines below are **proposals**. The running cleaner (`spec_v1/clean.py`, frozen for the current run) and the spec v1.1 sheet are not changed. Rule IDs are provisional until the team checks the shared registry.

## Summary

| Issue | Docs in the sample | Medicine |
|---|---:|---|
| Template header comment left at the top, with the author's handle and a creation date (C1) | 1 | `DA-C1-012` (proposal, replaces `DA-C1-005`) |
| Second licence header left (E4) | 1 | none for this doc's shape (6 docs in the pack); `DA-E4-012` (proposal, replaces `DA-E4-009`) for the short headers found in the pack, or an explicit decision to keep them |
| **Total** | **2** | |

Both issues are small in the sample. The pack-wide counts below show that each one affects hundreds of docs, with a small effect per doc (a few comment lines).

## Template header comment left at the top (C1) (1 in the sample, 295 in the pack)

**What it is.** The IDE inserts a comment at the top of a new file: an author line, a creation date, sometimes a file name. In `18451aef9bf14f36` the cleaned file still starts with

```text
/*
 * Author <handle>
 * Created on <date and time>
 */
```

(the handle and date are not copied here). The comment says nothing about the code, and it carries the author's user name. This is boilerplate (C1).

**Where.** The rule for IDE templates, `DA-C1-005` (java only, medicine 21 in `spec_v1_1/sheets/specv1_1.md`), removes `Created by … on <date>`, `Created with IntelliJ IDEA`, `To change this template` and similar lines. It does not know the wording `Created on <date>`, so the comment is not recognised as a template and stays. In the same doc the dataset marker line was removed correctly.

**How big it is.** On the cleaned code-java output (84,679 docs), `DA-C1-005` leaves 295 docs (0.35%) with a comment block that holds a `Created on <date>` line and that the card in Medicine 1 below recognises as a template; on the raw text before cleaning (85,213 docs) it is 296. In 286 of the 295 the `Created on` line is itself in the part that is removed; the rest are the other template lines of the same block. The sample doc is one of them. 14 other docs have a `Created on` line in a block that also has a copyright line; the card leaves those alone.

**What to do.** Replace `DA-C1-005` by `DA-C1-012` (Medicine 1 below): one new regex, `CREATED_ON`, that makes a block with a `Created on <date>` line a template block, so the existing `_clean_ide_block` removes its template lines (author, date, version, since, e-mail) and keeps any other line. Validation is in the Validation section.

A question for the lead, not decided here: an author handle is not in the F1 list of `DISEASES.md` (emails, phone numbers, addresses, ID numbers), and `DA-C1-005` removes author lines only inside template blocks or in a Javadoc with an author line and nothing else. That is why this doc is labelled C1 and not F1. A plain `@author <handle>` tag in a normal Javadoc is kept: `5fc227e0c45569e7` and `0b88c1632688dba2` have one and are labelled `clean`. Blocks that hold only author, created or date lines are in about 2,200 cleaned docs (2.6%) before `DA-C1-012`; removing them all would need that policy decision first.

| Pack | doc_id | Verdict |
|---|---|---|
| code-java / Part_1 | `18451aef9bf14f36` | fix |

## Second licence header left (E4) (1 in the sample, about 400 short headers in the pack)

**What it is.** `37124eab71fe111d` had two licence headers in different styles: a 17-line star-style Apache block at the top, then a `//Changed by …` line, then a second Apache text that opens with `///*Copyright 2017 …`. Cleaning removed the first block (offsets 12 to 612 before cleaning) and kept the second one: it is at offset 643 before cleaning and offset 30 after. In the second one the dataset's `<NAME>` placeholders became `AUTHOR`. The second block is not a valid comment (`///*` is a line comment, so the text after it is code until the `*/`), which is why the rules that look for comment blocks miss it. This is E4, repeated licence headers.

**How big it is.** After cleaning, 6 code-java docs have two or more licence blocks of 4 or more lines before the first `package` or `import`. That is too few, and each has its own shape, for a medicine. No new rule is proposed for this doc; it is listed here so the second reader can see it.

**The same disease, in a different form.** The code-go audit found 3-line licence headers that the go rule leaves in place (`DA-E4-011` in the code-go report). The java rules do the same by design: `DA-E4-009` removes a licence block of 7 or more lines and `DA-E4-010` one of 5 or more lines that is at least 40% licence text, and the card for `DA-E4-009` says "Short headers stay". None of the 35 sampled docs has a short header (at the observed rate of 0.5% one would expect fewer than one), so this part is not from the sample; it comes from counting the whole pack. On the cleaned code-java output, 3,212 docs (3.8%) start with a comment that has copyright or licence words, and 1,844 of those have 4 lines or fewer. The rule in Medicine 2 changes 400 docs (0.47%): 91 with the "Use of this source code is governed by … LICENSE file" sentence, 186 with "Licensed under the … License", and 123 with an SPDX line, 224 in `//` form and 176 in `/* */` form. Another 1,486 docs keep a short licence comment in a form the rule does not match (table in the validation section). Each header is about 100 to 200 characters, so the harm per doc is small.

**What to do.** The review of the code-go audit asked for one of two things for the same disease: handle these forms, or decide explicitly to keep them. Medicine 2 handles the forms below, with negative cases that protect substantive comments. If the lead decides to keep short notices, `DA-E4-009` stays as it is, and that decision should be written next to it; it should be the same decision as for `DA-E4-011` in the code-go report.

| Pack | doc_id | Verdict |
|---|---|---|
| code-java / Part_1 | `37124eab71fe111d` | fix |

## Other notes

- **Javadoc HTML.** `<p>` and `<br>` tags in Javadoc comments (`9cb77e447db4ee48`, `b8003102c989a4d0`, `2736e7f64ee0aaab`, `24852455a7586b63`) are normal Java and are not a disease. Notes in the label files say so.
- **Chinese comments.** Chinese comments next to ASCII code (`0b88c1632688dba2`, `f5af76f477baf13e`, `1158639fa76e3297`) are natural, not look-alike mixing. The code-go audit withdrew two A6 labels for the same reason.
- **Generator placeholders.** `1158639fa76e3297` is a code-generator output: its Javadoc has `${comments}` and `$column.comments` placeholders in the source. The cleaner removed the `@author` and `@date` lines of that block and left the placeholders and an empty `@email`. Harmless, so the verdict is `clean`.
- **Placeholders are replaced in different ways.** `<NAME>` became `AUTHOR` in 3 docs (`2736e7f64ee0aaab`, `37124eab71fe111d`, `8be0d5379d2edc49`). Same finding as in the code-go report.
- **The A6 fold.** The code-go card `DA-A6-004` already lists code-java in its scope. No sampled code-java doc was changed by the A6 rule, so there is nothing to add for it here.

## Medicine proposals

The cards follow the spec v1.1 form (`spec_v1_1/sheets/specv1_1.md`). Each block is standalone and is loaded by `harmtest/harm_test.py` from this file. Each copies the replaced rule as it is in spec v1.1 and makes the change described in its card.

### Medicine 1 DA-C1-012 C1 template header with `Created on`

1. **Disease:** C1 boilerplate (an IDE template header). The audited doc `18451aef9bf14f36`; 295 docs of the cleaned code-java output are changed by the card. Two more examples from the pack (not in the sample): `8e9f527e60ab3674` (a block with only a `Created on <date>` line, before `package`) and `5123c6a5ac814057` (a block with a file name and `Created on <date>`, at offset 29). **Replaces:** `DA-C1-005` on code-java. **Status:** PROPOSAL.
2. **Medicine:**
   - **What it removes, new.** The template lines of a comment block that has a line `Created on <date>` (the word `Created on` and a digit within 40 characters), and the whole block when nothing else is in it. The template lines are the ones `DA-C1-005` already removes: author, date, version, since, e-mail and similar.
   - **What it keeps.** A block with a licence word, as before. A block that also has a copyright line (new). `Created on` without a digit (`Created on the fly`). Every line of the block that is not a template line, for example a description.
   - **Unchanged.** Everything else in `DA-C1-005`.

```python
REPLACES = "DA-C1-005"
RULE_ID = "DA-C1-012"
SOURCE_RULE_ID = "DA-C1-005"
CODE = "C1"
ACTION = "fix"
STAGE = "30"
STATUS = "PROPOSAL"
LANGS = {'java'}
ORIGIN = "code-java"  # packs it runs on
PACKS = set(ORIGIN.split(", "))  # a code pack may label every doc "code", so the pack is checked too
SOURCE = "code-java"  # pack whose medicine this came from
AUTHOR = "vaibh27"
import re

LIC_WORDS = re.compile(r"licen[cs]e|permission is hereby granted|redistribution and use", re.I)

NOT_LICENCE = re.compile(r"to change this licen[cs]e header|choose licen[cs]e headers", re.I)

IDE_LINE = re.compile(r"(?im)^[ \t]*//[ \t]*TODO Auto-generated (?:method|constructor|catch block|stub)[^\n]*\n")

BLOCK = re.compile(r"[ \t]*/\*.*?\*/[ \t]*\n?", re.S)

IDE_MARK = re.compile(r"(?im)TODO Auto-generated|To change this (?:licen[cs]e header|template)|choose Tools \| Templates|"
                      r"Created (?:by|with) IntelliJ IDEA|Created by [^\n]{1,80}? on \d|File \| Settings \| File Templates|"
                      r"^\s*\*?\s*User\s*:.*\n\s*\*?\s*Date\s*:|^\s*\*?\s*@(?:project|package)\s*[:：]|\$\{[\w.]{1,40}\}|"
                      r"^\s*\*?\s*<p>\s*Title\s*[:：]")

# The IDE header "Created on <date>". It is a template mark unless the block also has a copyright line:
# a notice is not template text, so that block is left as it is.
CREATED_ON = re.compile(r"(?is)\A(?!.*copyright).*?Created on [^\n]{0,40}\d")

EMPTY_DESC = re.compile(r"(?im)^\s*\*?\s*@?desc(?:ription)?\s*[:：]*\s*(?:TODO)?\s*$")

AUTHOR_DATE = re.compile(r"(?im)^\s*\*?\s*@?(?:author|user)\b(?:.*\n)*?\s*\*?\s*@?(?:date|create|created)\s*[:：]")

TEMPLATE_LINE = re.compile(r"(?i)^\s*(?:/\*+|\*+)?\s*(?:<p>\s*)?(?:@?(?:author|user|date|time|class|project|package|create|created|"
                           r"version|since|title)\b\s*[:：]?.*|desc(?:ription)?\s*[:：]?\s*(?:TODO)?\s*(?:</p>)?|Created (?:by|with) .*|"
                           r"To change this .*|choose .*Templates.*|and open the template.*|TODO Auto-generated.*|"
                           r"(?:copyright|e-?mail|blog|company)\b.*)\s*$")

AUTHOR_ONLY = re.compile(r"/\*\*(?:\s*\*)*\s*@author\b[^\n]*(?:\s*\*)*\s*/")

def _clean_ide_block(block: str) -> str:
    if LIC_WORDS.search(block) and not NOT_LICENCE.search(block):
        return block  # a licence goes with M4
    if AUTHOR_ONLY.fullmatch(block.strip()):  # empty Javadoc with one @author line (labelled B4 only for <NAME>)
        return ""
    if not (IDE_MARK.search(block) or CREATED_ON.match(block) or (EMPTY_DESC.search(block) and AUTHOR_DATE.search(block))):
        return block
    body = block.rstrip("\n")
    lines, end = body.split("\n"), block[len(body):]  # end: the newline after */
    inner = [l for l in lines[1:-1] if not TEMPLATE_LINE.match(l) and l.strip(" \t*")]
    if len(lines) < 3 or not inner:
        return ""  # nothing but the template
    return "\n".join([lines[0]] + [l for l in lines[1:-1] if not TEMPLATE_LINE.match(l)] + [lines[-1]]) + end

def m6_strip_ide_templates(t: str) -> str:
    t = IDE_LINE.sub("", t)
    return BLOCK.sub(lambda m: _clean_ide_block(m.group()), t)


def _std(out, text):
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, str): return ("fix" if out != text else "keep", out)
    action, new = out
    if action == "drop": return ("drop", text)
    return ("fix" if new != text else "keep", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS and doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    return _std(m6_strip_ide_templates(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'code-cpp', 'code-go', 'code-java', 'code-js-ts', 'code-python', 'code-rust'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** fix, stage 30, as `DA-C1-005`. **Runs on:** java.
4. **Harm test:** gold 0 changed and 0 dropped of 116 on code-java; PASS (validation below).
5. **Side effects:** The template-line test is the one of `DA-C1-005`, so in a block with a `Created on` line it also removes a line that starts with `Class`, `Project`, `Package` or `Title` (in the 295 docs, 2 such lines start with `Project` and 1 with `Class`); a block that holds a description and a `Created on` line keeps the description. A comment that holds a `Created on` line inside other text on one line (`Some text. Created on 2010-1-1.`) is not removed. The removed author handle is a person's user name; the rule does not need it.
6. **Integration:** reviewer pending. Apply only through the spec v1.1 sheet and a rebuild of the cleaner; `spec_v1/clean.py` is frozen for the current run.
7. **For the teammate who collects the medicines:** java only, so there is nothing to merge with a code-go card. The same check on code-go finds 10 docs with a `Created on <date>` line near the top, too few to propose anything there. If a `Created on` header shows up in another pack, `CREATED_ON` can be reused as it is.

### Medicine 2 DA-E4-012 E4 short licence header (java)

1. **Disease:** E4 repeated licence headers. The audited doc `37124eab71fe111d` is a different shape and is **not** changed by this card (see the section above); the evidence for this card is the pack: 400 docs of the cleaned code-java output have one of the three forms. Examples from the pack (not in the sample): `8942544aebab8fc0` (3 `//` lines, "governed by … LICENSE file", offset 0), `9ccedceaad7727e6` (2 `//` lines, "Licensed under the Universal Permissive License", offset 0), `78eac33318ad53c6` (a `/* */` block, copyright and an SPDX line, offset 0). **Replaces:** `DA-E4-009` on code-java. **Status:** PROPOSAL.
2. **Medicine:**
   - **What it removes, new.** A header at the top of the file (after dataset marker lines and blank lines only) of 2 or 3 comment lines, in `//` form or as one `/* */` block: a copyright line, then one licence sentence or one SPDX line. The sentence forms are "Use of this source code is governed by … that can be found in the LICENSE file" (the sentence can wrap over two lines) and "Licensed under the … License"; the SPDX form is `SPDX-License-Identifier: <id>`. Stacked short headers go in one pass.
   - **What it keeps.** A block whose first line is not a copyright line; a block with a second copyright line, a second sentence or extra words; a block with only a copyright line; an SPDX line alone; a comment that is not at the top; a comment after code or after another comment that is kept.
   - **Unchanged.** The drop on a restrictive licence and the removal of a licence block of 7 or more lines stay exactly as in `DA-E4-009`.

```python
REPLACES = "DA-E4-009"
RULE_ID = "DA-E4-012"
SOURCE_RULE_ID = "DA-E4-009"
CODE = "E4"
ACTION = "fix+drop"
STAGE = "30"
STATUS = "PROPOSAL"
LANGS = {'java'}
ORIGIN = "code-java"  # packs it runs on
PACKS = set(ORIGIN.split(", "))  # a code pack may label every doc "code", so the pack is checked too
SOURCE = "code-go"  # pack whose medicine this came from
AUTHOR = "RaviNaik"
import re

_LIC_WORDS = re.compile(
    r"(?i)licensed under|permission is hereby granted|without warranty|"
    r"you may not use this file|redistribut|general public license|all rights reserved|"
    r"warranties of|distributed under the licen"
)

_RESTRICTIVE = re.compile(
    r"(?i)free for personal use|commercial use requires|non-?commercial use only|commons clause|"
    r"not for commercial use|for evaluation purposes only"
)

def _comment_runs(lines, limit=60, tail=False):
    """Yield (start, end) of contiguous // runs, /* */ blocks and runs of one-line /* ... */ comments."""
    n = len(lines)
    rng = range(max(0, n - 40), n) if tail else range(0, min(n, limit))
    lo, hi = rng.start, rng.stop
    i = lo
    while i < hi:
        s = lines[i].strip()
        if s.startswith("/*") and s.endswith("*/") and len(s) > 4:
            j = i
            while j < n and lines[j].strip().startswith("/*") and lines[j].strip().endswith("*/"):
                j += 1
            yield i, j
            i = j
        elif s.startswith("/*"):
            j = i
            while j < n and "*/" not in lines[j]:
                j += 1
            yield i, j + 1
            i = j + 1
        elif s.startswith("//") and not s.startswith(("//go:", "// +build", "//+build")):
            j = i
            while j < n and lines[j].strip().startswith("//"):
                j += 1
            yield i, j
            i = j
        else:
            i += 1

def strip_licence_header(text: str, min_lines: int = 7):
    """Drop files whose header carries a restrictive licence. Otherwise remove the first
    comment block in the first 60 lines (or the last 40 lines) that has >= min_lines lines and
    licence grant text. Run this AFTER fix_b3 so the file starts at the real first line."""
    lines = text.split("\n")
    head = "\n".join(lines[:60])
    if _RESTRICTIVE.search(head):
        return "drop", text
    for tail in (False, True):
        for a, b in _comment_runs(lines, tail=tail):
            block = lines[a:b]
            if len(block) >= min_lines and sum(1 for x in block if _LIC_WORDS.search(x)) >= 2:
                kept = lines[:a] + lines[b:]
                return "fix", "\n".join(kept).strip("\n") + ("\n" if text.endswith("\n") else "")
    return "keep", text


# The short header at the top of the file: a copyright line, then one licence sentence or an SPDX line, in 2 or 3 comment lines
# (`//` lines or one `/* */` block). The same shape and the same two sentence forms as DA-E4-011 in the code-go report; SPDX is added.
# Only that exact shape is removed: extra words, a second sentence, or a first line that is not a copyright line keep the block.
_COPYRIGHT = re.compile(r"(?i)(?:copyright|\(c\)|\u00a9)\b")
_SENTENCE = re.compile(
    r"(?is)use of this source code is governed by .{3,80}? (?:that|which) can be found in (?:the |a |this )?"
    r"(?:[\w.-]+ )?(?:LICENSE|LICENCE|COPYING)[\w.-]*(?: file)?(?: in [\w .,-]{0,60})?\.?"
    r"|licensed under (?:the |an? )?.{3,80}?licen[cs]e[\w .,;:()/-]{0,25}(?: see [\w .,-]{0,80}\.)?")
_SPDX = re.compile(r"(?i)spdx-license-identifier:\s*[\w.+-]+(?:\s+(?:OR|AND|WITH)\s+[\w.+-]+)*")
_MARKS = re.compile(r"(?:<(?:reponame|filename|gh_stars)>[^\n]*\n|[ \t]*\n)*")
_TOP_BLOCK = re.compile(r"[ \t]*/\*.*?\*/[ \t]*(?:\n|\Z)|(?:[ \t]*//[^\n]*(?:\n|\Z))+", re.S)
_BLANKS = re.compile(r"(?:[ \t]*\n)*")

def _is_short_licence(block):
    b = block.strip()
    if b.startswith("/*"):
        lines = [re.sub(r"^[ \t]*\*+[ \t]?", "", l).strip() for l in b[2:-2].split("\n")]
    else:
        lines = [re.sub(r"^[ \t]*//+[ \t]?", "", l).strip() for l in b.split("\n")]
    lines = [l for l in lines if l]
    if not 2 <= len(lines) <= 3 or not _COPYRIGHT.match(lines[0]) or any(_COPYRIGHT.match(l) for l in lines[1:]):
        return False
    rest = " ".join(lines[1:])
    return bool(_SENTENCE.fullmatch(rest) or _SPDX.fullmatch(rest))

def strip_short_headers(text: str):
    """Remove short licence headers at the very top (after dataset marker lines and blank lines); stacked ones go in one pass."""
    top = _MARKS.match(text).end()
    pos = top
    while True:
        m = _TOP_BLOCK.match(text, pos)
        if not m or not _is_short_licence(m.group()):
            break
        pos = _BLANKS.match(text, m.end()).end()
    return text if pos == top else text[:top] + text[pos:]

def medicine_e4(text):
    action, t = strip_licence_header(text)
    if action == "drop":
        return ("drop", text)
    t = strip_short_headers(t)
    return ("fix" if t != text else "keep", t)


def _std(out, text):
    # the author's result in the shared form: str -> fix/keep, None or True -> drop, False -> keep
    if out is None or out is True: return ("drop", text)
    if out is False: return ("keep", text)
    if isinstance(out, str): return ("fix" if out != text else "keep", out)
    action, new = out
    if action == "drop": return ("drop", text)
    return ("fix" if new != text else "keep", new)


def medicine(doc):
    if doc.get("lang_label") not in LANGS and doc.get("pack_id") not in PACKS:
        return ("keep", doc["text"])
    return _std(medicine_e4(doc["text"]), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'code-cpp', 'code-go', 'code-java', 'code-js-ts', 'code-python', 'code-rust'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** fix+drop, stage 30, as `DA-E4-009`. **Runs on:** java.
4. **Harm test:** gold 0 changed and 0 dropped of 116 on code-java; PASS (validation below). On the cleaned output it only removes the header and never drops a doc.
5. **Side effects:** It removes the notice that the licence asks to keep, as `DA-E4-009` does: record the licence in metadata first. A copyright line alone (663 docs) is kept, and so are the 1,486 docs with other short forms (listed below). Only a comment that starts the file is looked at.
6. **Integration:** reviewer pending, and only if the lead decides to remove short headers; otherwise record the decision to keep them. Apply through the spec v1.1 sheet and a cleaner rebuild.
7. **For the teammate who collects the medicines:** this is the java counterpart of `DA-E4-011` in `code-go_audit_smitasasindran.md`. `COPYRIGHT`, `SENTENCE` and the shape test (a copyright line, then one sentence, in 2 or 3 lines) are the same code in both cards. The differences are `/* */` blocks, the SPDX form, and that this card changes the java-only `DA-E4-009` where `DA-E4-011` changes the go-only `DA-E4-005`. They could become one shared helper, or one card for `code-go, code-java`. I did not merge them because the two replaced rules differ in what else they do (`DA-E4-005` also removes a block at the end of a file, `DA-E4-009` has a 7-line threshold). Please decide whether to merge them; the adopt-or-keep decision should be the same for both packs.

## Validation

All results are on this machine, 2026-11-23. The two rules are loaded from this file by `harmtest/harm_test.py`. Pre-cleaning text is `audit_labels/data/pre/code-java/Part_1` and `Part_2` (62,834 and 22,379 docs; 85,213 in all); cleaned output is `audit_labels/data/post/code-java/` (84,679 docs). The rules were run alone on those files, never the whole cleaner on the full data. No document text is in this report.

### DA-C1-012 against DA-C1-005

| Check | Result |
|---|---|
| The sample doc `18451aef9bf14f36` | the 4-line comment is removed and the file starts at `package`; nothing else changes |
| Cleaned code-java output, 84,679 docs: docs the new rule changes that the old one leaves alone | 295 (0.35%) |
| What was removed in those docs | only comment lines (498 lines in all): `Created on` lines in 286 docs, `@author` lines in 159, `@version` or `@since` in 40, e-mail lines in 28; no doc dropped. No copyright line is removed (the guard) |
| Raw pre-cleaning text, 85,213 docs | 296 docs differ from the old rule's output; the same kind of change; no doc dropped; 6 of them show a line replaced by an empty line only because of how the diff pairs lines, and in one the comment sits on the `package` line (`package …;/* … */`) and only the comment goes |
| Idempotent | applying the new rule to its own output changes nothing, on every changed doc |
| Docs the new rule changes that the old one leaves alone, and that have a licence word | 0 (the licence guard is unchanged) |
| Gold harm, code-java | 0 changed, 0 dropped of 116; PASS |
| Time, rule alone on all docs, ms/doc, old to new | 0.131 to 0.136 (cleaned), 0.141 to 0.145 (raw); about 4% slower, 0.005 ms/doc |

### DA-E4-012 against DA-E4-009

| Check | Result |
|---|---|
| Cleaned code-java output, 84,679 docs: docs the new rule changes that the old one leaves alone | 400 (0.47%): 91 "governed by … LICENSE file", 186 "Licensed under …", 123 SPDX; 224 in `//` form, 176 in `/* */` form |
| What was removed in those docs | only the comment lines of a 2 or 3-line header at the top and the blank line after it, in every doc; no doc dropped |
| Raw pre-cleaning text, 85,213 docs | 447 docs (0.52%) differ from the old rule's output; in every one the only difference is deleted comment lines and blank lines; no other change, no doc dropped |
| Idempotent | applying the new rule to its own output changes nothing |
| Gold harm, code-java | 0 changed, 0 dropped of 116; PASS |
| Time, rule alone on all docs, ms/doc, old to new | 0.087 to 0.086 (cleaned), 0.094 to 0.093 (raw); no change |

Short licence comments at the top that the new rule does not match (1,486 docs on the cleaned output, after the rule): the copyright line alone (663), a copyright line and a "WPILib BSD license file" sentence (61), a copyright line with "for details. All rights reserved. Use of this source code is governed by …" over two lines (31), a copyright line and "All rights reserved." (29) or "All Rights Reserved" (27), a copyright line and the Apache licence URL (26). They are kept; widening the rule to them needs more cases and a new harm run.

Negative cases (made-up text, in `teams/01-data-a/tests/audit/test_code_java_medicines.py`): for `DA-E4-012`, the rule keeps a package comment that quotes the licence sentence (no copyright first line); a copyright line plus a substantive sentence; a copyright line alone; an SPDX line alone; two copyright lines; a header after code; a header after another comment; a header inside a class; a 4-line block; other packs. It removes the 3-line and the 2-line `//` forms, the `/* */` forms, a header after dataset marker lines, and a short header behind a long one in the same pass. For `DA-C1-012`, the rule keeps a block with a licence word, a block with a copyright line, `Created on the fly`, and a description in the same block as a `Created on` line; it removes the sample's header shape, a block with only `Created on`, and the Javadoc form.

### Worst-case texts

`stress` (built-in set, ms, old rule to new rule; the rule alone on one text). `DA-C1-005`: one 2 MB line 27.7 to 28.2, 200,000 short lines 18.8 to 19.7, 20,000 comment blocks separated by blank lines 5.2 to 5.0, 20,000 short paragraphs 9.6 to 9.6, 2 MB of one repeated character 29.3 to 29.3. `DA-E4-009`: one 2 MB line 72.1 to 69.3, 200,000 short lines 7.5 to 7.3, 20,000 comment blocks 0.45 to 0.46, 20,000 short paragraphs 0.59 to 0.60, 2 MB of one repeated character 105.3 to 67.2 (noise between runs). Three cases of this report's own are pinned by timing bounds (2 seconds, about 0.03 to 0.1 s in practice) in `test_code_java_medicines.py`: 2,000 stacked short headers, one 2 MB comment block with no closing `*/`, and 20,000 `Created on` blocks without a date plus one block of 150,000 of them.

### Speed and whole-cleaner harm

Method: the project's `oneoff/speed_compare.sh`, driven by `audit/medicine_replacement_check.py` (below). Before is the committed `spec_v1_1/clean.py` (`6a57276c0e83`); after is the same sheet with the two cards swapped in under their old IDs, rebuilt (`d6a5b5925a1b`). The sheet diff is 52 changed lines in 4 hunks, all inside the two cards. The 13 week-1 sample packs (the 6 code packs, 4 English, 2 math, 1 science), 500 docs each, one core pinned to CPU 0, before and after timed back to back. Timings vary by about 5%.

| Measure | Before | After |
|---|---:|---:|
| code-java, docs/s | 1467.8 | 1532.7 |
| code-cpp, docs/s (no medicine here) | 1432.1 | 1468.0 |
| `DA-C1-005` on code-java, ms/doc | 0.138 | 0.145 |
| `DA-E4-009` on code-java, ms/doc | 0.104 | 0.107 |

Every difference is inside the noise: the medicines do not slow the cleaner. Docs/s after is between 0.9 and 1.1 times the before value on all 13 packs, including the 12 packs where the output is identical (for example en-dclm 641.0 to 597.8 and code-js-ts 2977.4 to 3175.8), so the difference comes from the order and the noise of the two runs, not from the rules. The two rules got 5% and 3% slower in the profile; the rule-alone timing over all docs above gives about 4% slower (0.005 ms per doc) and 2% faster.

- **Output sha256:** the same on 12 of 13 packs. It differs on code-java only, as expected: 5 of the 500 sample docs change (`d1e50dcbfa47a388`, `d9bca72f1f4841f5`, `750a1f9e926ed904`, `fc6b0564755d6d4f` stay `fix` with more lines deleted, `a7c46686fe59cb25` goes from keep to `fix`), in every case with only comment lines deleted. `speed_compare.sh` therefore prints `RESULT: FAIL`: it tests for a pure speed-up. That is not a harm failure.
- **Whole-cleaner harm (13 packs, gold):** every harm result file is identical before and after, except `code-java`, where only the 500-doc sample's "fixed" count changes (349 to 350) and the cleaner's hash. Gold for code-java stays 0 changed, 0 dropped of 116. The script also reports that one pack fails its own harm limit; that is the same before and after, so it is not caused by these medicines.

### Reproduce

Everything below runs from the repository root with a Python that has pyarrow (`/home/forest/miniconda3/envs/era5/bin/python` on the author's machine) and writes nothing into the repository. `medicine_replacement_check.py` is not specific to this report: it finds every card with `REPLACES` and `RULE_ID` lines in the report it is given and checks those. `--packs` is the folder of the week-1 sample packs (`<pack_id>.parquet`; default `../week1-packs`). `--work` is a scratch folder (default `../scratchpad/medicine-replacement`). The raw data and gold files are not in git.

```bash
CHECK=teams/01-data-a/audit/medicine_replacement_check.py
REPORT=teams/01-data-a/audit/code-java_audit_smitasasindran.md
python -I $CHECK prepare --report $REPORT                       # sheet with the 2 cards swapped in, diff, clean.py before and after
python -I $CHECK harm    --report $REPORT --packs <week1-packs> # harm_test.py on code-java's gold set, cards loaded from this file
python -I $CHECK docs    --report $REPORT --packs <week1-packs> # which documents differ, before and after, and how
python -I $CHECK stress  --report $REPORT
python -I $CHECK speed   --report $REPORT --packs <week1-packs> # speed_compare.sh: gold harm on 13 packs, doc counts, profile, docs/s (~10 min)
python -m pytest teams/01-data-a/tests/audit/test_code_java_medicines.py teams/01-data-a/tests/audit/test_medicine_replacement_check.py -q
```

The pack-wide counts above come from running each rule alone, old and new, over the Parquet files of the audit (`audit_labels/data/pre/code-java/Part_*/*.parquet` and `audit_labels/data/post/code-java/*.parquet`) and comparing the two outputs line by line; the loop is a few lines of Python around the two `medicine(doc)` functions (a doc is a dict with `doc_id`, `text`, `url`, `lang_label`, `pack_id`). A single card can also be run on a gold set by hand:

```bash
python -I teams/01-data-a/harmtest/harm_test.py --pack <week1-packs>/code-java.parquet \
  --gold teams/01-data-a/gold/code-java_gold.csv --rules $REPORT --rule DA-E4-012
```

To use the medicines in the spec, either keep the new IDs and the "replaces" line (the way proposals are added), or paste each code block into its card in `specv1_1.md` after resetting `REPLACES`/`RULE_ID`/`STATUS` to the old rule's values. `prepare` builds exactly that second form, and `specv1_1_changes.diff` in `--work` is the exact change.

## Decisions for the review

1. **`DA-C1-012`:** adopt it to remove template headers that say `Created on <date>`, or leave them. Separately, decide whether an author handle in a comment is personal data (F1) or boilerplate (C1): the cards assume boilerplate, as `DA-C1-005` does.
2. **`DA-E4-012`:** adopt it to remove the three short header forms, or record an explicit decision to keep them. Make the same decision for `DA-E4-011` in the code-go report. The two cards share their sentence tests; the lead may merge them.
3. **`37124eab71fe111d`:** no medicine; a second licence block behind another comment is in 6 docs of the pack.
4. **New IDs or an update in place:** the cards are proposals with new IDs and a "replaces" line, the way PR #75 and PR #76 add behaviour changes; the team decides whether `DA-C1-005` and `DA-E4-009` are updated in place instead.
5. **Second reader:** the 2 `fix` docs and the 2 `clean` docs with a plain `@author` handle left (`5fc227e0c45569e7`, `0b88c1632688dba2`) need a second pair of eyes for the C1 or F1 question.
