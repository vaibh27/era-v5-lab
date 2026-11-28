# Round trip on the p1536 token shards (week 3, Task 4)

**Result: PASS.** 19,970 documents drawn from 22 shards in 5 domains (up to 1,000,000 per domain, seed 4); 0 failures.

| Field | Value |
|---|---|
| Tokenizer | p1536 `tokenizer.json`, sha256 `e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b` |
| Shards | TSAI's p1536 sample-v0 bundle (16 Nov), `dataset_version: sample` |
| Script | `teams/03-tokenizer/harness/roundtrip_shards.py` at `1081a8bc3426` |
| Environment | Python 3.11.5, tokenizers 0.23.3, macOS-26.3-arm64-arm-64bit |
| Wall time | 36.6 s, including sha256 of every shard |

## Checks per document

| Check | Rule |
|---|---|
| eos | ends with EOS `<\|endoftext\|>` 130822; no other special id inside |
| ids | `encode(decode(ids)) == ids` (ids without the final EOS, `add_special_tokens=False`) |
| normal | `normalizer(decode(ids)) == decode(ids)`: decoded text is already NFC + chillu + nukta form |
| source | `decode(ids) == normalizer(source text)`, source row from `docs.parquet` `input_part` / `input_row` |

Each shard was first checked against its manifest: tokenizer sha256, EOS id, `.bin` header and size, payload and index sha256, and the document index.

## By domain

| Domain | Shards | Docs in shards | Drawn | eos | ids | normal | source | Tokens drawn |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| code | 4 | 3,999 | 3,999 | 3,999 / 3,999 | 3,999 / 3,999 | 3,999 / 3,999 | not run | 6,087,950 |
| en | 1 | 1,997 | 1,997 | 1,997 / 1,997 | 1,997 / 1,997 | 1,997 / 1,997 | not run | 1,928,034 |
| ind | 7 | 10,000 | 10,000 | 10,000 / 10,000 | 10,000 / 10,000 | 10,000 / 10,000 | not run | 10,613,274 |
| math | 2 | 1,976 | 1,976 | 1,976 / 1,976 | 1,976 / 1,976 | 1,976 / 1,976 | not run | 3,260,153 |
| sci | 8 | 1,998 | 1,998 | 1,998 / 1,998 | 1,998 / 1,998 | 1,998 / 1,998 | not run | 14,136,534 |

## By shard

Writer drops: documents `shard_writer.py` already left out because `decode(encode(x)) != normalizer(x)` (from each shard's `quality_report.json`), of rows read.

| Shard | Domain | payload sha256 | Docs | Tokens | Drawn | Failures | Writer drops |
|---|---|---|---:|---:|---:|---:|---:|
| `code-cpp-train-000000` | code | `249ef91abc3d…` | 1,051 | 1,987,220 | 1,051 | 0 | n/a |
| `code-cpp-train-000001` | code | `005e88a90443…` | 949 | 1,786,699 | 949 | 0 | n/a |
| `code-python-train-000000` | code | `adc08e2d5836…` | 1,763 | 1,999,201 | 1,763 | 0 | n/a |
| `code-python-train-000001` | code | `4826006bbeec…` | 236 | 314,830 | 236 | 0 | n/a |
| `en-fineweb-edu-train-000000` | en | `b81c5932ca98…` | 1,997 | 1,928,034 | 1,997 | 0 | n/a |
| `ind-bn-fw2-train-000000` | ind | `f083954d80fc…` | 2,000 | 1,447,195 | 2,000 | 0 | n/a |
| `ind-hi-fw2-train-000000` | ind | `d0f8e1600ae3…` | 2,000 | 1,385,709 | 2,000 | 0 | n/a |
| `ind-ml-fw2-train-000000` | ind | `04b5c9adb0a3…` | 2,000 | 1,434,326 | 2,000 | 0 | n/a |
| `ind-sa-fw2-train-000000` | ind | `ec10189f0701…` | 812 | 1,993,013 | 812 | 0 | n/a |
| `ind-sa-fw2-train-000001` | ind | `b527ab0abccb…` | 781 | 1,999,556 | 781 | 0 | n/a |
| `ind-sa-fw2-train-000002` | ind | `f2fc7a9c2409…` | 407 | 1,071,744 | 407 | 0 | n/a |
| `ind-ta-fw2-train-000000` | ind | `128b8c123078…` | 2,000 | 1,281,731 | 2,000 | 0 | n/a |
| `math-finemath-train-000000` | math | `57dd1a4f19e9…` | 1,137 | 1,998,474 | 1,137 | 0 | n/a |
| `math-finemath-train-000001` | math | `c2a9c2afb430…` | 839 | 1,261,679 | 839 | 0 | n/a |
| `sci-finepdfs-train-000000` | sci | `4211a1a7cea3…` | 341 | 1,998,392 | 341 | 0 | n/a |
| `sci-finepdfs-train-000001` | sci | `79cf82a578ad…` | 179 | 1,987,592 | 179 | 0 | n/a |
| `sci-finepdfs-train-000002` | sci | `7eca4eb24d8b…` | 305 | 1,857,283 | 305 | 0 | n/a |
| `sci-finepdfs-train-000003` | sci | `29b4205120ea…` | 233 | 1,999,420 | 233 | 0 | n/a |
| `sci-finepdfs-train-000004` | sci | `dae0c72de875…` | 332 | 1,996,220 | 332 | 0 | n/a |
| `sci-finepdfs-train-000005` | sci | `f5894fbf99b6…` | 299 | 1,994,114 | 299 | 0 | n/a |
| `sci-finepdfs-train-000006` | sci | `96500b686355…` | 254 | 1,999,066 | 254 | 0 | n/a |
| `sci-finepdfs-train-000007` | sci | `a351e0b6fc00…` | 55 | 304,447 | 55 | 0 | n/a |

## Failures

None.

## Notes

- The round trip is against the normalized text, as the tokenizer card says (section 2): Bn/Pa nukta and Ml old chillu come back in normal form, so `decode(encode(x))` can differ from the raw `x`.
- The source check (`--clean`) did not run: the cleaned parts were not available. `ids` and `normal` together show the shard ids are p1536's own encoding of text already in normal form, and the writer had already checked decode against the normalized source for every kept document.
- No document text is in this file: only IDs, counts, hashes and token positions.

## Command

```
python teams/03-tokenizer/harness/roundtrip_shards.py <shards> --tokenizer <p1536>/tokenizer.json --per-domain 1000000 --seed 4 --out teams/03-tokenizer/results/roundtrip_shards.md
```
