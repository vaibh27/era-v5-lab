# Clean-document audit: code-js-ts (Kunal Sinha)

**Pack:** `code-js-ts` (starcoderdata JavaScript/TypeScript). **Raw:** sample-v0 Part_1 `part-00000.parquet`, Part_2
`part-00001.parquet`. **Cleaned:** the #244 cleaning (`tokenizer_sample_v0_clean/train/code-js-ts/`), compared by `doc_id`.
**Method:** Narasimha's viewer (`g1-web-viewer-pre-post`), *Only changed* and *Random next* on, so the sample is of
documents the cleaner changed or dropped (it measures what the cleaner does, not how common a disease is in the pack).
Each document's raw → cleaned diff was also checked by script (emails, `<Phone>` masks, non-ASCII, invisible and control
characters, long lines, base64-like runs, leftover StarCoder tags).

## Coverage and verdicts

| Part | Reviewed (target) | clean | fix | drop |
|---|---:|---:|---:|---:|
| Part_1 | 35 (25) | 30 | 2 | 3 |
| Part_2 | 10 (10) | 9 | 0 | 1 |
| **Total** | **45 (35)** | **39** | **2** | **4** |

All read in full except one long file read partly. Diseases: E1 ×4 (all correct drops), E4 ×1. 10 extra Part_1 rows were
reviewed by mistake while Part_2 was intended; they are kept.

## What the cleaner does on this pack (all correct)

- **Removes StarCoder's metadata line** (`<reponame>…`, `<filename>…`, `<gh_stars>…`, alone or combined): the change in
  most documents.
- **Replaces StarCoder's privacy placeholders:** `<NAME>` → `AUTHOR` (5 documents), `<KEY>` → `EXAMPLE_KEY` (2),
  `<PASSWORD>` → `EXAMPLE_PASSWORD` (1). No real name, email or secret was seen in any cleaned document.
- **Removes Apache licence headers** worded "Licensed under the Apache License" (DA-E4-007; row 43).
- **Normalises CRLF line endings** to LF (row 29).
- **Drops build output and generated files** (DA-E1-008): files under `node_modules/` (2), a compiled `.d.ts` with a
  `//# sourceMappingURL=` line (1), an OpenAPI-generated model ("auto generated … Do not edit", 1). All 4 drops look right.

## Findings

1. **DA-E4-007 misses the ASF licence wording** (`b55c0a38bd118ca5`, verdict fix, E4). The 18-line header "Licensed to
   the Apache Software Foundation (ASF) under one or more contributor license agreements…" (used in every Apache/Cordova
   file) stays, because DA-E4-007's pattern only has "Licensed under the Apache License". DA-E4-010 (the other code packs)
   already matches "Licensed to the Apache". Proposal (v1.1 backlog; a miss, not a false positive): add
   `Licensed to the Apache` to DA-E4-007's licence pattern, with a harm test on the code-js-ts gold.
2. **Upstream redaction broke code; the cleaner keeps it** (`bb937e91245d322c`, verdict fix). StarCoder's PII filter
   replaced the identifier `bcrypt` with `<PASSWORD>` (the file imports `bcryptjs`); the cleaner's placeholder
   replacement leaves `EXAMPLE_PASSWORD.hashSync(...)`, which calls an undefined name. Proposal (v1.1 backlog): when
   `<PASSWORD>`/`<KEY>` is used as an identifier (followed by `(` or `.`), report the document as damaged by upstream
   redaction instead of replacing the placeholder silently. Prevalence not measured.
3. **Watch:** DA-E1-008's generated-header pattern `auto-?generated` does not match "auto generated" written with a
   space; the OpenAPI file was caught by "Do not edit" anyway. No missed file was seen.

## Files

`code-js-ts-Part_1-Kunal_Sinha.csv` (35 rows), `code-js-ts-Part_2-Kunal_Sinha.csv` (10 rows), in the viewer's format
(`doc_id, read, verdict, diseases, evidence, hex, note`; `evidence` and `hex` empty). No document text, names or secrets
are copied into notes.
