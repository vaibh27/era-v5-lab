# en-dclm audit: issues found, day D6

Week 3, D6 (20 Nov 2026). Reviewer: `TEJASKUMAR-REDDY-J`. Pack: `en-dclm`, 35 documents: 25 from Part_1 (`part-00000`) and 10 from Part_2 (`part-00001`).

Each document was read in full, side by side: the text before cleaning (`Cap1-packs/`) and after cleaning (`tokenizer_sample_v0_clean/train/`), matched by `doc_id`. Disease codes follow [DISEASES.md](../../../diagnosis/DISEASES.md). The labels are in `day_D6_en-dclm_Part_1.csv` and `day_D6_en-dclm_Part_2.csv` (the viewer's seven columns; `evidence` and `hex` empty).

## Inputs and method

| Input | File | sha256 |
|---|---|---|
| Raw Part_1 | `Cap1-packs/en-dclm/Part_1/part-00000.parquet` | `67b305d60aa0f409…` |
| Raw Part_2 | `Cap1-packs/en-dclm/Part_2/part-00001.parquet` | `0dec2d4fc09db295…` |
| Cleaned Part_1 | `tokenizer_sample_v0_clean/train/en-dclm/part-00000.parquet` | `34fdfc2879ef81c5…` |
| Cleaned Part_2 | `tokenizer_sample_v0_clean/train/en-dclm/part-00001.parquet` | `0674a7af8be15bdc…` |
| Shard check | `en-dclm-train-000000.bin`, `en-dclm-train-000001.bin` (frozen p1536) | `1fccdcc4c15bb22e…`, `2b6e1f71836f0824…` |

- **Selection:** the viewer's defaults, "Only changed" and "Random next" (`diagnosis/tools/view_packs_hub.py`, branch `g1-web-viewer-pre-post` at `be1571f8`). A doc counts as changed if its text differs after cleaning or it is missing after cleaning: Part_1 has 7,181 changed docs of 46,245 (876 dropped), Part_2 2,375 of 15,122 (279 dropped). The 35 docs are a random sample of the changed ones (`random.seed("en-dclm-<part>-audit")`, `random.sample`).
- **Shard check:** `day_D3.md` warns that the `tokenizer_sample_v0_clean` copy must not stand in for the shards (#576). So every audited doc was also decoded from the frozen-p1536 shards: all 28 kept docs decode to exactly the cleaned text, and the 7 dropped docs are not in the shards. The findings below therefore hold for the shards too.

## Summary

| Verdict | Docs |
|---|---:|
| fix | 25 |
| clean | 2 |
| drop | 1 |
| Dropped by cleaning (no cleaned version) | 7 |
| **Total** | **35** |

| Disease | Docs | Interpretation |
|---|---:|---|
| C1: boilerplate | 25 | Menus, share/follow lines, cookie banners and site footers left after cleaning. The cleaner removes some of them and keeps the rest. |
| C4: two documents joined | 4 | Blog index pages holding several posts. |
| A2: control characters | 1 | Handled wrongly: deleted instead of converted (see below). |
| A3, B3, B5, C3, C5, D1 | 1 each | See the groups below. |

Disease counts overlap: one doc can carry several codes.

## cp1252 quotes deleted instead of converted (A2) (1 in the sample, 49 apostrophe cases in the two parts)

**What it is.** Some web pages store ’ “ ” as the cp1252 bytes 0x92, 0x93, 0x94, which arrive as the C1 control characters U+0092, U+0093, U+0094. Cleaning deletes these characters instead of converting them, so `children’s` becomes `childrens`, `that’s` becomes `thats`, and quotation marks disappear.

**How big it is.** In the two raw parts, 103 docs contain U+0091–U+0097; 99 are kept, and none of those has a control character left after cleaning. Of the 50 kept docs with U+0092 (the apostrophe), 49 lose the apostrophe outright and 1 gets it converted.

**What to do.** Map C1 controls that cp1252 defines to their characters before control characters are stripped (for example ftfy's `fix_c1_controls`), and check that the apostrophe count stays the same.

| Part | doc_id | Verdict |
|---|---|---|
| Part_1 | `a9a7821bb3d942d0` | fix |

## Lines cut inside code (1)

**What it is.** Repeated-line removal also removes repeated lines inside code. In a shell/OpenVPN config, `cd /etc/openvpn` went from 3 lines to 1 and `up ip route add via` from 4 to 1, which changes the scripts. (IP addresses and paths missing in that code are already missing in the raw.)

**What to do.** Skip repeated-line removal inside code blocks, or limit it to lines that look like boilerplate.

| Part | doc_id | Verdict |
|---|---|---|
| Part_2 | `f1b8673c7cfd7aac` | fix |

## Math removed inside `$...$` (B5) (1)

**What it is.** Number lists inside inline math (`$\{8,16,24,...\}$`) were removed, leaving `Multiples of $8$ are: $ Multiples of $14$ are: $` with stray `$` signs and no numbers. The answer no longer follows from the text.

**What to do.** Find the rule that removes the lists (it may treat them as number blobs) and exempt text inside `$...$`.

| Part | doc_id | Verdict |
|---|---|---|
| Part_1 | `bfa86fe32d00c663` | fix |

## Phone number masked once, left on the next line (1)

**What it is.** A business freephone number was replaced by `<PHONE>` on one line and left as it is on the next line, where it follows the same word. The masking is inconsistent within one document.

| Part | doc_id | Verdict |
|---|---|---|
| Part_2 | `ef987c066d1cdf0f` | fix |

## `<s>` and `</s>` left in text (B3) (1)

**What it is.** A wiki talk page uses `<s>` and `</s>` as strikethrough tags. They are kept, and they read as BOS/EOS ghost markers (B3, the most dangerous disease). In the shard they are ordinary text tokens, not special ids, so the shard writer does not leave the doc out.

| Part | doc_id | Verdict |
|---|---|---|
| Part_1 | `b86875f06abe649a` | fix |

## Wrong language, gutted by cleaning (D1, C3) (1)

**What it is.** A Traditional Chinese translation of an MLB interview in the English pack. Cleaning removed most lines (2,732 to 673 characters), leaving English questions without their answers.

| Part | doc_id | Verdict |
|---|---|---|
| Part_2 | `ac9442043a4d4806` | drop |

## Dropped by cleaning (no cleaned version) (7)

The verdict column is empty, as in the task's example; the note says whether the drop looks right.

| Part | doc_id | Drop | Why |
|---|---|---|---|
| Part_1 | `7e774e95e602476e` | right | Spun SEO article promoting an essay-writing service |
| Part_1 | `c241159e7ee00f17` | right | Marketing page, duplicated title, link and share lines |
| Part_1 | `e670f86abf48520d` | acceptable | OCR scan with garbled Greek and soft hyphens |
| Part_1 | `0086c20deb34c1ec` | acceptable | 361-character fragment that ends mid-sentence |
| Part_1 | `ecbf55aa8452ab45` | acceptable | Short joke blog post, if a minimum-length rule applies |
| Part_1 | `b3abeb4533becb1d` | **wrong** | Public-domain Epictetus, Discourses Book Two, chapters 1-15; only site nav at the top and a cut last sentence: trim, do not drop |
| Part_1 | `9edbebb5cf7fc415` | **wrong** | Complete humour column; only the last contact line is boilerplate |

## All 35 documents

| Part | doc_id | Output | Verdict | Diseases | Note |
|---|---|---|---|---|---|
| Part_1 | `7e774e95e602476e` | Dropped | — | — | Drop right: spun SEO essay-service text |
| Part_1 | `bfa86fe32d00c663` | Changed | fix | B5;C1 | Number lists cut inside `$...$`; "Update this answer!" kept |
| Part_1 | `c8c3820547a7ab6f` | Changed | fix | C1 | Author contact footer kept; only the name removed |
| Part_1 | `b61794200f81acab` | Changed | fix | C1 | StackExchange chrome kept (tour, share, votes, "Your Answer") |
| Part_1 | `8da43001d1c5ea60` | Changed | fix | C1 | Accessibility menu and social footer kept |
| Part_1 | `852d6c991a0ed3ed` | Changed | fix | C1 | Call-to-action header; phone masked, business email kept |
| Part_1 | `a9a7821bb3d942d0` | Changed | fix | A2;C1 | cp1252 quotes deleted: "childrens", "thats"; nav line kept |
| Part_1 | `5fed88b562dff56f` | Changed | fix | C1 | Page header kept; report body fine |
| Part_1 | `dff5216e6f15c4dd` | Changed | fix | C1;C4 | Related-article teasers and blank lines after the article |
| Part_1 | `7ccb72daa2438a76` | Changed | fix | C1;C5 | One SEO keyword line left; comment section kept |
| Part_1 | `8bc5cc927d1c4095` | Changed | fix | C1 | BOM removed correctly; blog widgets kept |
| Part_1 | `b3abeb4533becb1d` | Dropped | — | — | Drop wrong: public-domain Epictetus |
| Part_1 | `c241159e7ee00f17` | Dropped | — | — | Drop right: marketing page |
| Part_1 | `c733a2499740aff8` | Changed | fix | C4;C1 | Four blog posts joined; "follow us" block kept once |
| Part_1 | `b86875f06abe649a` | Changed | fix | B3;C1 | `<s>`/`</s>` kept; wiki nav kept |
| Part_1 | `ad11cf73e258fd95` | Changed | fix | A3;C1 | PDF extract: "page 1of 6", U+00FF bullets, split words |
| Part_1 | `1e811184b12bc962` | Changed | fix | C1 | "Return to Top" and "last reviewed" line kept |
| Part_1 | `9edbebb5cf7fc415` | Dropped | — | — | Drop wrong: complete humour column |
| Part_1 | `06f1063ec1748884` | Changed | clean | — | Empty tag line removed correctly |
| Part_1 | `e670f86abf48520d` | Dropped | — | — | Drop acceptable: garbled OCR scan |
| Part_1 | `beb5d368db86fb5e` | Changed | clean | — | Keywords line removed; abstract intact |
| Part_1 | `b8a0a886678197f6` | Changed | fix | C1 | Contact line and copyright footer kept |
| Part_1 | `28e2db94666d860f` | Changed | fix | C1 | Forum chrome kept ("Last Updated:", "Share This Page") |
| Part_1 | `0086c20deb34c1ec` | Dropped | — | — | Drop acceptable: short fragment |
| Part_1 | `ecbf55aa8452ab45` | Dropped | — | — | Drop acceptable: short post |
| Part_2 | `ac9442043a4d4806` | Changed | drop | D1;C3 | Chinese in the English pack; most lines removed |
| Part_2 | `afc83ec7e700b331` | Changed | fix | C4;C1 | Five channelled-message posts joined; blog chrome kept |
| Part_2 | `64c909239beadad9` | Changed | fix | C1 | Market header, related headlines and site footer kept |
| Part_2 | `ef987c066d1cdf0f` | Changed | fix | C1 | Phone masked once, left on the next line |
| Part_2 | `90860d3fa1f13864` | Changed | fix | C1 | Cookie banner and publication metadata around the abstract |
| Part_2 | `6d42ab66ea6a9d2c` | Changed | fix | C1 | Essay-site ads, image file names and site footer kept |
| Part_2 | `f1b8673c7cfd7aac` | Changed | fix | C4;C1 | Blog index; repeated-line removal cut code lines |
| Part_2 | `5a7a2f792c27521e` | Changed | fix | C1 | BOM removed correctly; marketing nav kept |
| Part_2 | `3391ad7aa8aa6e49` | Changed | fix | C1 | Membership banner kept |
| Part_2 | `72320b6dece6c7bf` | Changed | fix | C1 | Cookie message, article count and related links kept |

## Follow-up

1. cp1252 quotes: confirm the rule that strips C1 controls and map them first; check the whole pack, not only these two parts.
2. Code lines: decide whether repeated-line removal should skip code blocks; check the code packs for the same effect.
3. Math: find the rule that removed the number lists inside `$...$`.
4. Phone masking: find why the second occurrence was not masked.
5. `<s>`/`</s>`: decide whether these strikethrough tags are removed or escaped (B3).
6. Drops: the two wrong drops are good documents; find the rule that removed each (run the current `clean.py` on them).
7. Boilerplate (C1) is the most common finding; the cleaner removes some lines of a block and keeps the rest.

This report has no document payloads and no personal data values; the short snippets are evidence for the findings.
