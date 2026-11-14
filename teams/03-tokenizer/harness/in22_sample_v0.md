# IN22-Gen fertility: C1–C3 sample-v0 vs BrahmicTokenizer-131K

Tokens / whitespace word (lower is better). Mean = Indic languages present.
Fertility is deterministic; one run.

BrahmicTokenizer-131K Indic mean 3.0792, En 1.3385.
C3 Indic mean −15.6% vs BrahmicTokenizer-131K, En +0.2%.

## Artifacts

| Artifact | sha256 / revision |
|---|---|
| IN22-Gen parquet (`data/train-00000-of-00001.parquet`) | sha256 `e79c7889abe97c78cf03c09f04e50874b7932b381b49a08a4431e58aefc5fb5c`, HF revision `e042ab3d3063110b1a85efa0a59bdbf8553bb928` |
| C1 `tokenizer.json` | `36acf1264248c0f7ed1e3b18cfef6e4e9f489660d749c1e9c2b79e80901c82a2` |
| C2 `tokenizer.json` | `b0d44576012cede10ea197d33251377ee5aac72f246aca954b96a95a20f8488a` |
| C3 `tokenizer.json` | `15468e66457b43af79f695cbe585c0d03e3cfe7a4a5d04d189dd3c6b9f02ddc3` |

## IN22-Gen fertility

| Tokenizer | En | Hi | Bn | Ta | Te | Kn | Ml | Mr | Gu | Pa | Or | As | Mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BrahmicTokenizer-131K | 1.3385 | 1.7557 | 2.5567 | 3.3510 | 3.2814 | 3.6962 | 3.9840 | 2.7294 | 2.5442 | 2.6086 | 4.4142 | 2.9503 | 3.0792 |
| c1 | 1.3413 | 1.7194 | 2.5130 | 2.8881 | 2.8133 | 3.1649 | 3.5509 | 2.2810 | 2.3571 | 2.1059 | 3.0331 | 2.5221 | 2.6317 |
| c2 | 1.3413 | 1.7194 | 2.5130 | 2.8208 | 2.7838 | 3.1331 | 3.5509 | 2.2810 | 2.4827 | 2.1667 | 2.9182 | 2.5221 | 2.6265 |
| c3 | 1.3413 | 1.7188 | 2.4628 | 2.8208 | 2.7838 | 3.1336 | 3.4363 | 2.2805 | 2.4827 | 2.1059 | 2.8606 | 2.5029 | 2.5990 |
