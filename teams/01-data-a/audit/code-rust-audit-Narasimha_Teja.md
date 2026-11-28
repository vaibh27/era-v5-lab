# code-rust audit

This file gives the result of the code-rust audit, as of 2026-11-23. Reviewer: Narasimha Teja (`narasimhatejav`). Assignment 6 (code-rust): 35 docs, 25 from Part_1 and 10 from Part_2.

35 docs were read in full, side by side: the text before cleaning (`Cap1-packs/code-rust/Part_1/part-00000.parquet`, `Part_2/part-00001.parquet`) and after cleaning (`tokenizer_sample_v0_clean/train/code-rust/part-00000.parquet`, `part-00001.parquet`). Each doc is labelled in `code-rust-Part_1-Narasimha_Teja.csv` and `code-rust-Part_2-Narasimha_Teja.csv`. Disease codes follow `era-v5-lab/diagnosis/DISEASES.md`.

## Result

34 of the 35 docs are `clean`, including all 10 in Part_2. One doc in Part_1 is marked `fix`. The cleaning did no damage in the sample. The one issue is a licence header that a rule did not remove.

| Issue | Docs |
|---|---:|
| Short licence header left (E4) | 1 |
| **Total** | **1** |





