# Deterministic tokenization: two shard-writer runs, p1536 `e6b93a93…`

**Result: PASS.** Made by `harness/determinism_shards.py` (241 s). Input: the first 2,000 documents of one train part of each of 10 packs of Data A's tokenizer sample v0 (checked against its SHA256SUMS), through Data A's synthetic pipeline run and `shard_writer.py`.

| Run | Tokenizer | Settings |
|---|---|---|
| A | release p1536 | `RAYON_NUM_THREADS=1`, `PYTHONHASHSEED=1`, separate process |
| B | release p1536 | all 10 cores, `PYTHONHASHSEED=2`, separate process |
| R | p1536 rebuilt from the pinned inputs (`rebuild_cand.py`) | as B |

| Pack | Shards | Docs | Tokens | Files A vs B | A = B | B = R | eos / sha in manifests | tok/s, 1 thread (writer) |
|---|---:|---:|---:|---:|---|---|---|---:|
| en-fineweb-edu | 1 | 1,997 | 1,928,034 | 7 | identical | identical | 130822 / p1536 | 290,510 |
| code-python | 2 | 1,999 | 2,314,031 | 13 | identical | identical | 130822 / p1536 | 401,253 |
| code-cpp | 2 | 2,000 | 3,773,919 | 13 | identical | identical | 130822 / p1536 | 502,629 |
| math-finemath | 2 | 1,976 | 3,260,153 | 13 | identical | identical | 130822 / p1536 | 464,487 |
| sci-finepdfs | 8 | 1,998 | 14,136,534 | 49 | identical | identical | 130822 / p1536 | 524,780 |
| ind-hi-fw2 | 1 | 2,000 | 1,385,709 | 7 | identical | identical | 130822 / p1536 | 264,145 |
| ind-ta-fw2 | 1 | 2,000 | 1,281,731 | 7 | identical | identical | 130822 / p1536 | 239,664 |
| ind-bn-fw2 | 1 | 2,000 | 1,447,195 | 7 | identical | identical | 130822 / p1536 | 277,789 |
| ind-sa-fw2 | 3 | 2,000 | 5,064,313 | 19 | identical | identical | 130822 / p1536 | 458,552 |
| ind-ml-fw2 | 1 | 2,000 | 1,434,326 | 7 | identical | identical | 130822 / p1536 | 244,871 |
| **All** | 22 | 19,970 | 36,025,945 | 142 | | | | |

The writer's tokens per second include reading parquet, the round-trip check and writing; it is not the bare encoder rate.

Documents the writer left out (the same in A, B and R): secret 29, special 1. By pack: en-fineweb-edu {'secret': 3}; code-python {'special': 1}; math-finemath {'secret': 24}; sci-finepdfs {'secret': 2}.

## The sha is enforced

| Tokenizer given to the writer | sha256 | Exit | Shards written | Message |
|---|---|---:|---:|---|
| p1536 plus one trailing newline (same JSON) | `9c6406275969…` | 1 | 0 | tokenizer.json: sha256 9c6406275969ff89e213fd16b70d540b76b2873700eaca5cf5c038c6db88a846, not the frozen p1536 e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f879 |
| C3 sample-v0 | `15468e66457b…` | 1 | 0 | tokenizer.json: sha256 15468e66457b43af79f695cbe585c0d03e3cfe7a4a5d04d189dd3c6b9f02ddc3, not the frozen p1536 e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f879 |

## Round trip on run A (`roundtrip_shards.py`, every document)

**Result: PASS.** 19,970 documents drawn from 22 shards in 5 domains (up to 1,000,000 per domain, seed 4); 0 failures. Full report: `determinism_shards_roundtrip.md`.

## `structural_identity_ids.py` release vs rebuilt

```
A: release/tokenizer.json (Kaggle sohiyiy/p1536-sample-v0)
B: rebuild_cand.py --cand p1536 out/p1536/tokenizer.json (16 Nov job)
--- token string -> id ---
  |A| 131,072   |B| 131,072
  same string in both:        131,072
  same string AND same id:    131,072
  same string, different id:  0
--- ids whose string differs: 0 ---
--- pipeline pieces ---
  normalizer     identical: True
  pre_tokenizer  identical: True
  decoder        identical: True
  post_processor identical: True
--- encode check ---
  en      10 tokens   identical ids: True
  hi      14 tokens   identical ids: True
  bn       9 tokens   identical ids: True
  ta      10 tokens   identical ids: True
  te       7 tokens   identical ids: True
  or      12 tokens   identical ids: True
  code    17 tokens   identical ids: True
  mixed    7 tokens   identical ids: True
  all samples identical: True
```

## Shard sha256 (run A; B and R are identical)

| Shard | sha256 |
|---|---|
| `en-fineweb-edu-train-000000.bin` | `b81c5932ca98b6abc3d4f75b79078cd3e8d1d1563f1b37ce8d0badb187365884` |
| `code-python-train-000000.bin` | `adc08e2d583650d08f216b99bd743cbdf09e5e91b2ace53ec89268f1bfb90898` |
| `code-python-train-000001.bin` | `4826006bbeece233c5bc5895634edf3904e8b76387b6a3a119a05571d01f7ce8` |
| `code-cpp-train-000000.bin` | `249ef91abc3d8b6ad94157eb43b09b4881007612fed43d541d29b249f5075a6d` |
| `code-cpp-train-000001.bin` | `005e88a90443b07fdffdf0c62052f820a38c6df334d71340995f4cb891930a44` |
| `math-finemath-train-000000.bin` | `57dd1a4f19e9281bcf72f623d448860929fcdcab41f2bbe8c1fc36f19345879c` |
| `math-finemath-train-000001.bin` | `c2a9c2afb4308183314be2720613281b358b9dfaa17101c75baa131416639bd0` |
| `sci-finepdfs-train-000000.bin` | `4211a1a7cea3cca83e9de033ce30d4f159ba3b14cd11b8f29294849d49956ea7` |
| `sci-finepdfs-train-000001.bin` | `79cf82a578adec42a5609c5ec43f92bb05c22b992bf89d8f34f0b9635df9d47d` |
| `sci-finepdfs-train-000002.bin` | `7eca4eb24d8b890e1d256ca502048a6002c5e6458bf61bec3ec6c2d13a80e32e` |
| `sci-finepdfs-train-000003.bin` | `29b4205120ea8b1a6347d2d57ca13289b938808542b99ade09bb1255e338aefb` |
| `sci-finepdfs-train-000004.bin` | `dae0c72de875d93c91f8e335c4031ef12fe0e770608c0387285aaff7c814dc9d` |
| `sci-finepdfs-train-000005.bin` | `f5894fbf99b66454e245e9751db2a5d69123d9e0c7798a67bbd92229b99bb736` |
| `sci-finepdfs-train-000006.bin` | `96500b6863556ab2b44b533932928e31f150bed5b8388f647559f8eb06f0deb9` |
| `sci-finepdfs-train-000007.bin` | `a351e0b6fc0076f7b6e04a22431c3d7930a89a888b5f1bcfa2db56e7ef371974` |
| `ind-hi-fw2-train-000000.bin` | `d0f8e1600ae3451e2058c58adbfa198e149f07a9a2effed480a8bc43f42007c5` |
| `ind-ta-fw2-train-000000.bin` | `128b8c1230780274d5855c095b94a151def8b6bbb66045788ca4ba42d69976e5` |
| `ind-bn-fw2-train-000000.bin` | `f083954d80fc0fcf9196778e6bc9964f034d4f07a14922b2b0f37b9cded6948e` |
| `ind-sa-fw2-train-000000.bin` | `ec10189f0701eee3bca0a2fec4017b0ff33e4a0183516ebbd787bd343597edf3` |
| `ind-sa-fw2-train-000001.bin` | `b527ab0abccbc62ee12eb7c245b70f8b2c3830eb5b98697c369e1c4d936b99c0` |
| `ind-sa-fw2-train-000002.bin` | `f2fc7a9c2409df1a0a4f15474d52eb75ad86aefcae90e68766b412b3342f5a84` |
| `ind-ml-fw2-train-000000.bin` | `04b5c9adb0a3681a35d1398f19ffbd202b85fad3bd2f0f3cf76e89e9ba362018` |

```
python teams/03-tokenizer/harness/determinism_shards.py --sample <sample v0> --tokenizer release/tokenizer.json --rebuilt <rebuild>/p1536/tokenizer.json --c3 <c3>/tokenizer.json --rows 2000 --tokens 2000000
```
