# Special-token test, p1536 `e6b93a93…`

Made by `release/special_slots.py`. Result: **PASS**.

| Check | Result |
|---|---|
| Each of the 356 special tokens encodes to one id, its own (alone, and between words) | 356 / 356 |
| Plain text makes no special id (13 held-out files, 4,270,468 tokens) | 0 special ids |
| Look-alikes make no special id (16 strings: fullwidth, spaces, case, zero-width, cut off) | 0 made one |
| A literal marker in raw text becomes its special id (expected: the shard writer leaves such docs out) | [130822, 129994] |
| G10's `special_tokens_v1.md` markers: special at their id, with G10's string | 15 / 17; differ: 130827 subagent_call (frozen `<|g10_slot_05|>`), 130828 subagent_return (frozen `<|g10_slot_06|>`) |

| File | sha256 | Tokens | Special ids |
|---|---|---:|---:|
| `dev/As.txt` | `6060b60d0164…` | 238,381 | 0 |
| `dev/Bn.txt` | `828be2d10b09…` | 231,232 | 0 |
| `dev/Code.txt` | `3a2e8ff6d799…` | 707,414 | 0 |
| `dev/En.txt` | `b44df0781544…` | 1,077,077 | 0 |
| `dev/Gu.txt` | `87f5bd1ad359…` | 250,971 | 0 |
| `dev/Hi.txt` | `fdcef2982b72…` | 236,682 | 0 |
| `dev/Kn.txt` | `cd757d8c88a3…` | 216,652 | 0 |
| `dev/Ml.txt` | `d436400d5efc…` | 199,508 | 0 |
| `dev/Mr.txt` | `66a567ce58fe…` | 213,839 | 0 |
| `dev/Or.txt` | `3ca261ae7541…` | 224,274 | 0 |
| `dev/Pa.txt` | `2e50c225be42…` | 257,056 | 0 |
| `dev/Ta.txt` | `6093418072d6…` | 190,816 | 0 |
| `dev/Te.txt` | `942f5ae7f86a…` | 226,566 | 0 |
