# code-go audit: issues to fix

This file lists the problems found while auditing the code-go pack, as of 2026-11-22, and the medicines proposed for two of them.

45 docs were read side by side: the text before cleaning (`Cap1-packs/`) and after cleaning (`tokenizer_sample_v0_clean/train/`). Part_1 has 35 docs and Part_2 has 10. Only docs that cleaning changed or dropped were sampled (the viewer's default), so unchanged docs are not covered. Each doc was labelled in `audit_labels/code-go_<part>.csv` (copies in `teams/01-data-a/audit/code-go_<part>-smitasasindran.csv`). The verdict answers "was the cleaner's output right?": `clean` (yes), `fix` (no, something is wrong in the output), or `drop` (the doc was removed, and that was right). Disease codes follow `era-v5-lab/diagnosis/DISEASES.md`.

Docs labelled `clean` are left out: 33 docs. What remains is 12 docs: 4 marked `fix` and 8 marked `drop`. The 8 dropped docs need no action; they are listed so a second reader can check them.

Every one of the 45 docs also lost a dataset header line (`<gh_stars>…`, `<reponame>…` or `<filename>…`) at the top. That was done correctly in every doc and is not an issue.

All medicines below are **proposals**. The running cleaner (`spec_v1/clean.py`, frozen for the current run) and the spec v1.1 sheet are not changed. Rule IDs are provisional until the team checks the shared registry.

## Summary

| Issue | Docs in the sample | Medicine |
|---|---:|---|
| Cyrillic letters changed by a look-alike fold where they should stay (G4) | 1 | `DA-A6-004` (proposal, replaces `DA-A6-001`) |
| Short licence header left at the top (E4) | 3 | `DA-E4-011` (proposal, replaces `DA-E4-005`), or an explicit decision to keep these headers |
| Dropped by cleaning, correctly (E1, E2) | 8 | none |
| **Total** | **12** | |

The first issue is the smallest group in the sample, but the pack-wide run (below) shows it is bigger than the sample suggests, and it changes what the code or the text means. The second is minor and affects many docs across the pack.

## Cyrillic letters changed by a look-alike fold (G4) (1 in the sample, 9 wrong changes in the pack)

**What it is.** The rule `DA-A6-001` turns a Cyrillic look-alike letter into the Latin letter inside any word that mixes the two scripts. That is right for a typo such as `retriеve`, but it also changes text where the Cyrillic is deliberate. In the sample: `dabdf7ecc74cd614` is a test fixture for a Go linter that flags non-ASCII identifiers. Before cleaning it declares `type TеstStruct struct{}` with a Cyrillic `е` (U+0435). Cleaning changed it to a Latin `e`. The comment on the same line still says the identifier "contain non-ASCII character: U+0435", so the cleaned file contradicts itself and the test no longer tests anything. This is damage from cleaning: G4, over-normalised.

**How big it is.** The rule is `DA-A6-001` in `spec_v1/clean.py` (medicine 3 in `spec_v1_1/sheets/specv1_1.md`, with a speed guard that does not change its output). Run alone on the whole code-go pack (83,267 docs before cleaning), it changes **18 docs**. The first version of this report counted 11: it only counted docs where no mixed word was left after cleaning. The other 7 keep a mixed word after the fold, because they are real Russian words, and that is the damage.

| Group | Docs | Verdict |
|---|---:|---|
| Accidental look-alike letter in a Latin word (typo) | 9 | the change is right |
| Test file with deliberate Cyrillic (a `testdata/` file or a `_test.go` file) | 2 | the change is wrong |
| Real Russian word stuck to a Latin character (`\nИли`, `IP_сервера`, `%sПогода`, `test1фыва`) | 7 | the change is wrong |

All 18 docs, by `doc_id`. The "Word" column gives the changed word before and after the fold, except for the two copyright comments, where the word is part of an obfuscated address and is not copied here (DISEASES.md section 2a).

| Pack | doc_id | File | Word | Group | Change right? |
|---|---|---|---|---|---|
| code-go / Part_1 | `21157a287d0d7f6d` | `fs.go` | in a copyright comment | typo | yes |
| code-go / Part_1 | `74bfb60c19c028e0` | `host-zone.go` | in the same copyright comment | typo | yes |
| code-go / Part_1 | `46b904f2ef8e5319` | `ts_schema_manager.go` | `retriеve` to `retrieve` | typo | yes |
| code-go / Part_1 | `f9eaf956f28fcb71` | `cluster.go` | `PVСReport` to `PVCReport` | typo | yes |
| code-go / Part_1 | `48ba5986aedab756` | `config.go` | `EnableH2С` to `EnableH2C` | typo | yes |
| code-go / Part_1 | `c93f242e8a9a6222` | `followerList.go` | `followersListСollectioName` to `followersListCollectioName` | typo | yes |
| code-go / Part_1 | `bec15fe3f9cbb8c8` | `rpc.go` | `AcсBandwidth` to `AccBandwidth` | typo | yes |
| code-go / Part_2 | `0735eb1ab62fe34f` | `upload.go` | `сheckLength` to `checkLength` | typo | yes |
| code-go / Part_2 | `c19c5ba211d773b2` | `gzip.go` | `сType` to `cType` | typo | yes |
| code-go / Part_1 | `dabdf7ecc74cd614` | `testdata/struct_name.go` | `TеstStruct` to `TestStruct` | test file | **no** |
| code-go / Part_2 | `c95ad16bf96bf0d6` | `symbols_test.go` | test input `оjо` and others | test file | **no** |
| code-go / Part_1 | `5999cbfd41bf0f0d` | `service_test.go` | `test1фыва` to `test1фывa` | Russian word | **no** |
| code-go / Part_1 | `7c33d05933b542b8` | `initjson.go` | `IP_сервера` to `IP_cepвepa` | Russian word | **no** |
| code-go / Part_1 | `3a7fe5367f2d7506` | `ban.go` | `nЕсли` to `nEcли` | Russian word | **no** |
| code-go / Part_1 | `91c4a6915779fdc7` | `finder.go` | a long Russian identifier ending in `Windows`, half of its letters turned Latin | Russian word | **no** |
| code-go / Part_1 | `6447774f597c7fe5` | `weather.go` | `sПогода` to `sПoгoдa` | Russian word | **no** |
| code-go / Part_2 | `5e3fab815e8db5a8` | `bashorg.go` | `nПодождите` to `nПoдoждитe` | Russian word | **no** |
| code-go / Part_2 | `3657802746c284b0` | `get.go` | `nИспользуйте` to `nИcпoльзyйтe` | Russian word | **no** |

The `n`, `s` and `1` that start the Russian words are the end of a `\n` escape, a `%s` verb and a digit in a Go string. The damage is still in the cleaned text of all 7 Russian-word docs and of the 2 test files. The same rule on the other five code packs (same method, the week-2 train parts) changes 15 docs in code-cpp, 19 in code-java, 40 in code-js-ts, 43 in code-python and 5 in code-rust. The word lists of the ones it would now keep are mostly the same pattern (`nВыберите`, `nЭто`).

**What to do.** Replace `DA-A6-001` by `DA-A6-004` (Medicine 1 below). It adds two guards: it skips Go test files (a `testdata/` folder or a `_test.go` name, from the `url` column), and it folds a word only when its Cyrillic letters are single and separate, so a word with two Cyrillic letters in a row is kept as real Cyrillic. Each guard alone is not enough (validation below). Validation covers the 2 wrong test changes, the 7 wrong Russian changes and the 9 right changes.

| Pack | doc_id | Verdict |
|---|---|---|
| code-go / Part_1 | `dabdf7ecc74cd614` | fix |

## Short licence header left at the top (E4) (3)

**What it is.** The file still starts with a 3-line copyright and licence comment after cleaning, for example "Copyright 2014 … / Use of this source code is governed by a MIT-style / license that can be found in the LICENSE file." The same wording repeats across thousands of Go files. This is disease E4, repeated licence headers.

**Where.** The licence rule for code-go, `DA-E4-005`, only removes a comment block that has licence words and at least 4 lines (`block.count("\n") >= 4` in `_is_licence`). Longer Apache and MIT blocks were removed in this sample (seven docs). The 3-line form passes through. In `05809df4` the header also shows `Copyright 2018 AUTHOR`, where the original had a `<NAME>` placeholder.

**How big it is.** On the cleaned code-go output (76,955 docs), 2,051 docs (2.7%) start with a short header that the rule in Medicine 2 matches: 1,838 of the "governed by … LICENSE file" form and 213 of the "Licensed under the … License" form. Another 228 docs (0.3%) start with a short licence comment in a form it does not match (table in the validation section). Each header is about 150 characters of boilerplate, so the harm per doc is small.

**What to do.** The review asked for one of two things: handle these forms, or decide explicitly to keep them. Medicine 2 below handles the two forms above, with negative cases that protect substantive comments. If the lead decides to keep 3-line notices instead, `DA-E4-005` stays as it is and that decision should be written next to it. `88d0c261` was labelled `fix` without the E4 code; it now has it.

| Pack | doc_id | Verdict |
|---|---|---|
| code-go / Part_1 | `4a379ae4228af6e2` | fix |
| code-go / Part_1 | `88d0c26177f7877c` | fix |
| code-go / Part_1 | `05809df4f85761c5` | fix |

## Dropped by cleaning, correctly (E1, E2) (8)

**What it is.** Cleaning removed these docs completely. Each starts with a `Code generated … DO NOT EDIT` header, so they are generated code (E1). Two of them (`d0b99f0f8a8caaeb` and `12450bcb472f6610`, both `.pb.go` files) also end in a large gzipped descriptor as a byte array (E2). The notes in the label files say "Correctly dropped by cleaning". The viewer shows "No matching doc_id" for any dropped doc, which is not a reason and is not used in the notes.

**What to do.** No new medicine. The generated-code drop rules already do this, and their documented disposition is kept as it is. If a second reader disagrees with any row, change its verdict to `fix` and say what was good about the doc.

| Pack | doc_id | Verdict |
|---|---|---|
| code-go / Part_1 | `7643d5cdbac0d774` | drop |
| code-go / Part_1 | `9506fec46784a06e` | drop |
| code-go / Part_1 | `fe6a2b7719890828` | drop |
| code-go / Part_1 | `95e214af76eb5d16` | drop |
| code-go / Part_1 | `b86a73c3741ef72f` | drop |
| code-go / Part_1 | `d0b99f0f8a8caaeb` | drop |
| code-go / Part_2 | `12450bcb472f6610` | drop |
| code-go / Part_2 | `87e58585404b217d` | drop |

## Other notes

- **Withdrawn labels.** Two docs (`0be515fddd3bb1cc`, `dbbb98d4c8762c0c`) were first labelled A6 for "CJK and Latin in one word". They are Chinese comments next to ASCII code (`//构造url`), which is normal, so they are now `clean`.
- **Placeholders are replaced in different ways.** The dataset's `<PASSWORD>` became `EXAMPLE_PASSWORD`, `<KEY>` became `EXAMPLE_KEY`, and `<NAME>` became `AUTHOR`. None of these changes broke the sampled code, but the rule is worth a line in the medicine sheet.
- **CRLF.** `a310b1f78ba5bb79` had 121 carriage returns removed and nothing else changed. That is fine.

## Medicine proposals

The cards follow the spec v1.1 form (`spec_v1_1/sheets/specv1_1.md`). Each block is standalone and is loaded by `harmtest/harm_test.py` from this file. Each copies the replaced rule as it is in spec v1.1 and makes the change described in its card.

### Medicine 1 DA-A6-004 A6 look-alike letters, without test files and real Cyrillic words

1. **Disease:** A6 mixed script (look-alike letters), and the G4 damage that the old rule does. 18 code-go docs changed by `DA-A6-001`: 9 right, 9 wrong (the 18 doc_ids are in the table above). **Replaces:** `DA-A6-001` on the six code packs. **Status:** PROPOSAL.
2. **Medicine:**
   - **What it fixes.** A Cyrillic look-alike letter inside a Latin word becomes the Latin letter, as before.
   - **What it leaves alone, new.** (a) A Go test file or a file under `testdata/`, where the Cyrillic can be the point of the test. (b) A word with two or more Cyrillic letters in a row: that is a real Cyrillic word stuck to a Latin character (`\nИли`, `IP_сервера`, `%sПогода`), not a typo.

```python
REPLACES = "DA-A6-001"
RULE_ID = "DA-A6-004"
SOURCE_RULE_ID = "DA-A6-001"
CODE = "A6"
ACTION = "fix"
STAGE = "10"
STATUS = "PROPOSAL"
LANGS = {'cpp', 'go', 'java', 'javascript', 'python', 'rust', 'typescript'}
ORIGIN = "code-go, code-rust, code-js-ts, code-java, code-cpp, code-python"  # packs it runs on
PACKS = set(ORIGIN.split(", "))  # a code pack may label every doc "code", so the pack is checked too
SOURCE = "code-go"  # pack whose medicine this came from
AUTHOR = "TEJASKUMAR-REDDY-J"
import re

HOMO = str.maketrans("аеорсухіјѕАВЕКМНОРСТХІЈЅ", "aeopcyxijsABEKMHOPCTXIJS")

# A Go test file or a file under testdata/: its odd letters can be the point of the test, so the rule leaves it alone.
TEST_FILE = re.compile(r"(?:^|/)testdata/|_test\.go$")

# Only a Latin word with single, separate Cyrillic letters in it (a typo for a look-alike). A word with two Cyrillic letters in a row is
# real Cyrillic text stuck to a Latin character (`\nИли`, `IP_сервера`, `%sПогода`) and is left alone.
MIXED = re.compile(r"\b(?=\w*[A-Za-z])(?=\w*[Ѐ-ӿ])(?!\w*[Ѐ-ӿ]{2})\w+\b")

CYRILLIC = re.compile("[\u0400-\u04ff]")  # the range MIXED asks for: every match holds one of these

def medicine_a6(doc):
    text = doc["text"]
    if not CYRILLIC.search(text):  # no Cyrillic character, so MIXED has no match: the text is kept
        return ("keep", text)
    t = MIXED.sub(lambda m: m.group().translate(HOMO), text)
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
    if TEST_FILE.search(doc.get("url") or ""):
        return ("keep", doc["text"])
    return _std(medicine_a6(doc), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'code-cpp', 'code-go', 'code-java', 'code-js-ts', 'code-python', 'code-rust'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** fix, stage 10, as `DA-A6-001`. **Runs on:** the six code packs.
4. **Harm test:** gold 0 changed and 0 dropped on all six code packs (validation below). Full-data results are in the validation section too.
5. **Side effects:** A typo made of two adjacent Cyrillic look-alikes in one Latin word (for example `wіkіреdіа` in a code-cpp doc) is no longer fixed. Only Go test paths are skipped: test files of other languages are still folded, because no evidence of damage there was found; a doc with no `url` gets the old behaviour. It can still turn an obfuscated e-mail address back into plain text, so the data-safety stage runs after it (unchanged).
6. **Integration:** reviewer pending. Apply only through the spec v1.1 sheet and a rebuild of the cleaner; `spec_v1/clean.py` is frozen for the current run. Rerun the affected pipeline checks after the rebuild.

### Medicine 2 DA-E4-011 E4 short Go licence header

1. **Disease:** E4 repeated licence headers. The 3 audited docs `4a379ae4228af6e2`, `88d0c26177f7877c` and `05809df4f85761c5`; 2,051 docs of the cleaned code-go output have one of the two forms. **Replaces:** `DA-E4-005` on code-go. **Status:** PROPOSAL.
2. **Medicine:**
   - **What it removes, new.** A header of 2 or 3 `//` lines at the top of the file (after build lines and dataset header lines only): a copyright line, then one licence sentence. The two sentence forms are "Use of this source code is governed by … that can be found in the LICENSE file" (the sentence can wrap over two lines) and "Licensed under the … License".
   - **What it keeps.** A block whose first line is not a copyright line; a block with a second copyright line, a second sentence or extra words; a block with only a copyright line; a comment that is not at the top; build tags and package documentation (as before).
   - **Unchanged.** The 4-line-or-longer rule and the drop on a restrictive licence stay exactly as in `DA-E4-005`.

```python
REPLACES = "DA-E4-005"
RULE_ID = "DA-E4-011"
SOURCE_RULE_ID = "DA-E4-005"
CODE = "E4"
ACTION = "fix+drop"
STAGE = "30"
STATUS = "PROPOSAL"
LANGS = {'go'}
ORIGIN = "code-go"  # packs it runs on
PACKS = set(ORIGIN.split(", "))  # a code pack may label every doc "code", so the pack is checked too
SOURCE = "code-go"  # pack whose medicine this came from
AUTHOR = "TEJASKUMAR-REDDY-J"
import re

LICENCE = re.compile(r"Licensed under|Licensed to the Apache|Permission is hereby granted|Redistribution and use|"
                     r"Use of this source code is governed by|This source code is licensed under|"
                     r"released under the \w+ licen[cs]e|General Public License|Mozilla Public License", re.I)

RESTRICTIVE = re.compile(r"(?i)free for personal use|commercial use requires|non-?commercial use only|"
                         r"commons clause|not for commercial use|for evaluation purposes only")

BLOCK = re.compile(r"(?:[ \t]*/\*[^\n]*\*/[ \t]*\n){2,}|/\*.*?\*/[ \t]*\n?|(?://[^\n]*\n)+", re.S)

TAIL = re.compile(r"\n((?:[ \t]*/\*[^\n]*\*/[ \t]*\n?){2,}|/\*(?:(?!\*/).)*\*/|(?://[^\n]*(?:\n|\Z))+)\s*\Z", re.S)

def _is_licence(block):
    return bool(LICENCE.search(block)) and block.count("\n") >= 4

# The short Go header: a copyright line, then one licence sentence, in 2 or 3 `//` lines. Only that exact shape is removed;
# any extra words, a second sentence, or a first line that is not a copyright line keep the block.
COPYRIGHT = re.compile(r"(?i)(?:copyright|\(c\)|\u00a9)\b")
SENTENCE = re.compile(
    r"(?is)use of this source code is governed by .{3,80}? (?:that|which) can be found in (?:the |a |this )?"
    r"(?:[\w.-]+ )?(?:LICENSE|LICENCE|COPYING)[\w.-]*(?: file)?(?: in [\w .,-]{0,60})?\.?"
    r"|licensed under (?:the |an? )?.{3,80}?licen[cs]e[\w .,;:()/-]{0,25}(?: see [\w .,-]{0,80}\.)?")
BUILD_OR_BLANK = re.compile(r"(?:[ \t]*//[ \t]*(?:go:build|\+build)[^\n]*\n|<(?:reponame|filename|gh_stars)>[^\n]*\n|[ \t]*\n)*")

def _is_short_licence(block, before):
    if not BUILD_OR_BLANK.fullmatch(before):
        return False  # only at the top of the file, after build lines
    if not 2 <= block.count("\n") <= 3:
        return False
    lines = [re.sub(r"^[ \t]*//+[ \t]?", "", l).strip() for l in block.rstrip("\n").split("\n")]
    lines = [l for l in lines if l]
    if len(lines) < 2 or not COPYRIGHT.match(lines[0]) or any(COPYRIGHT.match(l) for l in lines[1:]):
        return False
    return bool(SENTENCE.fullmatch(" ".join(lines[1:])))

def _strip_head(head):
    # BLOCK.sub, with one change: a short header counts as "at the top" when only build lines, blank lines and the blocks
    # removed before it are above it, so two stacked headers go in one pass (apply(apply(doc)) == apply(doc)).
    # `before` is the kept text above the block. It is dropped (None) once it is over 2000 characters: no file top is that long,
    # and it keeps the loop linear when a file has thousands of comment blocks.
    out, pos, before = [], 0, ""
    for m in BLOCK.finditer(head):
        gap = head[pos:m.start()]
        out.append(gap)
        if before is not None:
            before += gap
        block = m.group()
        if _is_licence(block) or (before is not None and _is_short_licence(block, before)):
            pass
        else:
            out.append(block)
            if before is not None:
                before += block
        if before is not None and len(before) > 2000:
            before = None
        pos = m.end()
    out.append(head[pos:])
    return "".join(out)

def medicine_e4(doc):
    t = doc["text"]
    if RESTRICTIVE.search("\n".join(t.split("\n")[:60])):
        return ("drop", t)  # a restrictive licence: stripping the header would not make the code free to use
    body = re.search(r"^(?:import|func|type|var|const)\b", t, re.M)
    end = body.start() if body else len(t)
    head = _strip_head(t[:end])
    if head != t[:end]:
        head = head.lstrip("\n")
    new = head + t[end:]
    tail = TAIL.search(new)
    if body and tail and tail.start() > len(head) and _is_licence(tail.group(1)):
        new = new[:tail.start()] + "\n"
    return ("keep", t) if new == t else ("fix", new)


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
    return _std(medicine_e4(doc), doc["text"])

_spec_source_medicine = medicine

SPEC_PACKS = {'code-cpp', 'code-go', 'code-java', 'code-js-ts', 'code-python', 'code-rust'}

def medicine(doc):
    if doc.get("pack_id") not in SPEC_PACKS:
        return ("keep", doc["text"])
    out = _spec_source_medicine(doc)
    return out
```

3. **Action:** fix+drop, stage 30, as `DA-E4-005`. **Runs on:** go.
4. **Harm test:** gold 0 changed and 0 dropped of 163 (validation below). On the cleaned output it only removes the header and never drops a doc.
5. **Side effects:** It removes the notice that the licence asks to keep, as `DA-E4-005` does: record the licence in metadata first. A copyright line alone is kept, and so are the 228 docs with other short forms (listed below). A block of 4 or more lines that has a licence sentence followed by package documentation is still removed whole by the old 4-line test; that existing side effect is not changed here.
6. **Integration:** reviewer pending, and only if the lead decides to remove short headers; otherwise record the decision to keep them. Apply through the spec v1.1 sheet and a cleaner rebuild.

## Validation

All results are on this machine, 2026-11-22. The two rules are loaded from this file by `harmtest/harm_test.py`. Pre-cleaning text is `week-2/train/<pack>/part-0000{0,1}.parquet`; for code-go that is the same data as the audit's Part_1 and Part_2 (61,393 and 21,874 docs). Cleaned output is the audit's `post/` files (76,955 docs). No document text is in this report.

### DA-A6-004 against DA-A6-001

| Check | Result |
|---|---|
| The 9 right changes (listed above) | the output is identical to `DA-A6-001` for all 9 |
| The 2 wrong changes in test files (`dabdf7ecc74cd614`, `c95ad16bf96bf0d6`) | kept, text unchanged |
| The 7 wrong changes in Russian text | all 7 kept, text unchanged |
| code-go, all 83,267 docs: docs changed, old rule to new rule | 18 to 9; the 9 are exactly the 9 right ones |
| Each guard alone | the path guard alone still changes 15 of the 18 (it misses the 7 Russian-text docs except `5999cbfd41bf0f0d`, which is also a `_test.go` file); the two-letters guard alone still changes 11 (it misses the 2 test files) |
| Idempotent | applying the new rule to its own output changes nothing, on every changed doc |
| New rule changes a doc that the old rule leaves alone | 0 docs, on all six packs |

Docs changed by the rule alone, old rule to new rule, on the other code packs (whole packs, same method): code-cpp 15 to 5 of 55,545; code-java 19 to 12 of 85,213; code-js-ts 40 to 19 of 121,938; code-python 43 to 17 of 74,943; code-rust 5 to 1 of 60,126. The docs the new rule still changes show single look-alike letters in Latin words (`сleaned`, `Сlass`, `market_priсe`). I did not read every one.

Gold harm (`harm_test.py`, limit `max(1, floor(1%))`): code-cpp 0 of 122, code-go 0 of 163, code-java 0 of 116, code-js-ts 0 of 204, code-python 0 of 141, code-rust 0 of 234. All PASS, 0 changed and 0 dropped.

### DA-E4-011 against DA-E4-005

| Check | Result |
|---|---|
| Cleaned code-go output, 76,955 docs: docs the new rule changes that the old one leaves alone | 2,051 (2.7%): 1,838 "governed by … LICENSE file" form and 213 "Licensed under …" form |
| What was removed in those docs | only a 2 or 3-line `//` block at the top, in every doc; no doc dropped |
| Idempotent | applying the new rule to its own output changes nothing |
| Raw pre-cleaning code-go text, 83,267 docs | 2,185 docs (2.6%) differ from the old rule's output. In every one the only difference is deleted `//` comment lines and blank lines (a 2 or 3-line header, or several stacked ones); no other change, no doc dropped, idempotent |
| Gold harm, code-go | 0 changed, 0 dropped of 163; PASS. 3 of the 163 gold docs start with a copyright comment, and none is changed |

Short licence comments at the top that the new rule does not match (228 docs on the cleaned output): the largest forms are a two-sentence Apache-style comment ("or more contributor license agreements. Licensed under …", 31), a copyright line, a blank `//` line and the "governed by" sentence without "that can be found in" (24), a comment with no copyright line (15), and a copyright and a one-line licence sentence on one line (13). They are kept; widening the rule to them needs more cases and a new harm run.

Negative cases (made-up text, in `teams/01-data-a/tests/audit/test_code_go_medicines.py`): the rule keeps a package comment that quotes the licence sentence (no copyright first line); a copyright line plus a substantive sentence; a copyright line alone; two copyright lines; a copyright line plus a comment that only mentions the LICENSE file; a header after code; a header inside a function; other packs. It removes the 3-line form, the 2-line form, a differently wrapped form, a header after `//go:build` lines, and a short header behind a long one in the same pass.

### Speed and whole-cleaner harm

Method: the project's `oneoff/speed_compare.sh`, driven by `audit/medicine_replacement_check.py` (below). Before is the committed `spec_v1_1/clean.py` (`6a57276c0e83`); after is the same sheet with the two cards swapped in under their old IDs, rebuilt (`c106e350ea44`). The sheet diff is 55 changed lines in 3 hunks, all inside the two cards. The 13 week-1 sample packs (the 6 code packs, 4 English, 2 math, 1 science), 500 docs each, one core pinned to CPU 0, best of 3, before and after timed back to back. Timings vary by about 5%.

| Measure | Before | After |
|---|---:|---:|
| code-go, docs/s | 1036.6 | 1083.9 |
| code-rust, docs/s | 2452.7 | 2553.5 |
| `DA-A6-001` on code-go, ms/doc | 0.024 | 0.020 |
| `DA-A6-001` on code-rust, ms/doc | 0.024 | 0.026 |
| `DA-E4-005` on code-go, ms/doc | 0.103 | 0.107 |

Every difference is inside the noise: the medicines do not slow the cleaner. Docs/s after is 1% to 10% higher on all 13 packs, including the 7 packs the medicines do not touch (for example en-nemotroncc-synth 2332.3 to 2577.5), so the gain comes from the order of the two runs, not from the rules.

- **Output sha256:** the same on 11 of 13 packs. It differs on code-cpp (1 doc) and code-go (14 docs), as expected. `docs` lists them: the code-cpp doc is `782bc9665f0acdf7` (the Russian comment, now left unchanged), and 13 code-go docs go from keep to fix and one stays fix, in every case with only `//` comment lines deleted. `speed_compare.sh` therefore prints `RESULT: FAIL`: it tests for a pure speed-up. That is not a harm failure.
- **Whole-cleaner harm (13 packs, gold):** every harm result file is identical before and after, except `code-go`, where only the 500-doc sample's "fixed" count changes (260 to 273) and the cleaner's hash. Gold for code-go stays 0 changed, 0 dropped of 163. The script also reports that one pack fails its own harm limit; that is the same before and after, so it is not caused by these medicines.
- **Worst-case texts** (`stress`, built-in set, ms, old rule to new rule). `DA-E4-005`: 20,000 comment blocks separated by blank lines 24 to 25, one 2 MB line 197 to 196, 200,000 short lines 87 to 90. `DA-A6-001`: one 2 MB line 11 to 11, 200,000 short lines 6 to 6. The first version of the E4 code was quadratic on the 20,000-block case (1,833 ms); the card has the fix, and a unit test pins it. Two cases of this report's own are pinned by timing bounds in the unit tests: 2,000 stacked short headers (about 0.12 s, the old rule 0.025 s) and a 2.4 MB Russian text (about 0.07 s, the old rule 0.10 s).

### Reproduce

Everything below runs from the repository root with a Python that has pyarrow (`/home/forest/miniconda3/envs/era5/bin/python` on the author's machine) and writes nothing into the repository. `medicine_replacement_check.py` is not specific to this report: it finds every card with `REPLACES` and `RULE_ID` lines in the report it is given and checks those. `--packs` is the folder of the week-1 sample packs (`<pack_id>.parquet`; default `../week1-packs`). `--work` is a scratch folder (default `../scratchpad/medicine-replacement`). The raw data and gold files are not in git.

```bash
CHECK=teams/01-data-a/audit/medicine_replacement_check.py
REPORT=teams/01-data-a/audit/code-go_audit_smitasasindran.md
python -I $CHECK prepare --report $REPORT                       # sheet with the 2 cards swapped in, diff, clean.py before and after
python -I $CHECK harm    --report $REPORT --packs <week1-packs> # harm_test.py on each pack's gold set, cards loaded from this file
python -I $CHECK docs    --report $REPORT --packs <week1-packs> # which documents differ, before and after, and how
python -I $CHECK stress  --report $REPORT
python -I $CHECK speed   --report $REPORT --packs <week1-packs> # speed_compare.sh: gold harm on 13 packs, doc counts, profile, docs/s (~10 min)
python -m pytest teams/01-data-a/tests/audit/test_code_go_medicines.py teams/01-data-a/tests/audit/test_medicine_replacement_check.py -q
```

`stress` times the old and new rule on a built-in set of worst-case texts (one 2 MB line, 200,000 short lines, 20,000 comment blocks, and others); `--stress-file` adds a rule's own cases. `--include-all` makes `speed` use every pack in `--packs` (needed for Indic rules; copy the Indic gold in as `oneoff/speed_handoff.md` describes). The build-based steps (`prepare`, `docs`, `speed`) need cards that the sheet builder accepts: a card with a narrower scope than the rule it replaces is rejected by `build_clean.py`; `harm` and `stress` do not need the build. Run `speed` alone in a plain terminal, not in VS Code and with no other timing run. A single card can also be run on a gold set by hand:

```bash
python -I teams/01-data-a/harmtest/harm_test.py --pack <week1-packs>/code-go.parquet \
  --gold teams/01-data-a/gold/code-go_gold.csv --rules $REPORT --rule DA-A6-004
```

Expected: `prepare` says `clean_before.py is the committed spec_v1_1/clean.py: yes`; `harm` prints PASS with 0 changed and 0 dropped on all 7 lines; `docs` lists 1 doc for code-cpp, 14 for code-go and none for the other four; `speed` shows the output sha256 equal on 11 of 13 packs and the gold results equal everywhere but the code-go sample count.

To use the medicines in the spec, either keep the new IDs and the "replaces" line (the way proposals are added), or paste each code block into its card in `specv1_1.md` after resetting `REPLACES`/`RULE_ID`/`STATUS` to the old rule's values. `prepare` builds exactly that second form, and `specv1_1_changes.diff` in `--work` is the exact change.

## Decisions for the review

1. **`DA-A6-004`:** adopt the two guards (Go test paths, and single Cyrillic letters only), or only the test-path guard. The path guard alone leaves the 7 Russian-text docs damaged.
2. **`DA-E4-011`:** adopt it to remove the two short header forms, or record an explicit decision to keep them. Either closes the review comment.
3. **The 8 dropped docs:** no new medicine; the disposition `drop` is kept.
4. **New IDs or an update in place:** the cards are proposals with new IDs and a "replaces" line, the way PR #75 and PR #76 add behaviour changes; the team decides whether `DA-A6-001` and `DA-E4-005` are updated in place instead.
5. **Second reader:** the dropped docs and the 18 `DA-A6-001` changes need a second pair of eyes, since the Russian-text and typo groupings above are my reading of the words.
