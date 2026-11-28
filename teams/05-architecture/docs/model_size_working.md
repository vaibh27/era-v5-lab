# Model size: parameter count and size grid (input to the proposal)

This is a working for the model-size proposal. It does not pick a size and does not change `configs/config_v1.yaml`
(the size keys stay null). TSAI decides the size at Gate 4, once the corpus-v1 token count is known. The tests of the
KAS head stay with #717.

Lab main at `821c82a` (25 Nov 2026). Script: `harness/sizing.py`. Tests: `tests/test_sizing.py`.

## 1. The count, checked against the real model

`harness/sizing.py` counts parameters in closed form, term by term from `harness/week2/model.py` and `kas_v2.py`
(w = d_model, L = layers, h = expert FFN, E = 16 routed, S = 1 shared, k = top-2, V = 131,072):

| Part | Parameters |
|---|---|
| Per block: 2 RMSNorm | 2w |
| Per block: attention q, o / k, v (GQA, head dim w / n_head) | 2w² + 2·w·n_kv·(w / n_head) |
| Per block: router | w·E |
| Per block: experts (routed + shared SwiGLU, 3 matrices each) | (E + S)·3·w·h |
| Final RMSNorm | w |
| KAS input projection (`head_mode: kas_v2`) | 2048·w |
| KAS head: W_out, beta, unigram, tok_u, tok_c | 2048·w + 33 + V + 64·V + 64·w |
| Dense untied head (+ KE2 projection 256·32·w) | V·w |
| MTP head, when `mtp: true` | V·w |

Active parameters per token: the same, with (k + S) = 3 experts per block instead of 17. The router, the input
projection and the head are dense, so they count as active.

`python3 harness/sizing.py --check` builds the real `LanguageModel` from each arm config and compares:

| Arm | Head | Formula | Real model | |
|---|---|---:|---:|---|
| t1 | dense, KE2 | 58,080,512 | 58,080,512 | exact |
| t2 | dense, KE2 | 237,052,416 | 237,052,416 | exact |
| t3 | dense, KE2 | 391,529,600 | 391,529,600 | exact |
| dryrun2 | KAS v2 | 176,398,881 | 176,398,881 | exact |

The instructor figures in `experiments/plan_v1.md` (56M, 233M, 386M) use tied embeddings: one V × d table and no
KE2 projection. With that convention the formula gives 55.98M, 232.86M and 386.29M. The harness counts are higher by
the KE2 projection (256·32·w: 2.1M, 4.2M, 5.2M).

## 2. What moves the size

- **Experts are almost all of it.** At d_model 5120, 32 layers: routed experts 60.4B, shared expert 3.8B, attention
  2.0B, KAS input + head 30M, router 2.6M. The model size is set by d_model, layers and the expert FFN width.
- **Active / total is about 0.20 on the Gate 2 MoE** (3 of 17 experts per token): 0.215 at expert FFN 1.0 × d_model,
  0.20 at 1.5 ×, 0.196 at 2.0 ×. So a 30B–120B total model is about 6B–24B active.
- **The head choice is small at this scale.** KAS input + head is 30M at d_model 5120; the dense untied head with the
  KE2 projection would be 713M (about 1% of a 66B model). An MTP head adds V·d_model (0.67B at 5120).
- **Expert FFN ratio.** T1 uses 2.0 × d_model, T2 1.5 ×, T3 1.4 ×. The same d_model and layers give about 2× the
  total at ratio 2.0 as at 1.0 (5120 × 32: 44.8B at 1.0, 66.2B at 1.5, 87.6B at 2.0).

## 3. Size grid inside 30B–120B

Assumptions for the grid (placeholders, not choices): head dim 128, 8 KV heads, expert FFN 1.5 × d_model (rounded to
128), KAS head, no MTP head. Change them with the script's `big()` arguments or `--ffn-ratio`.

Total and active parameters, Gate 2 MoE (16 routed + 1 shared, top-2), KAS head, head dim 128, 8 KV heads, expert FFN 1.5 x d_model. Only shapes with 30B to 120B total are listed.

Tokens at 20 per active parameter is the dense compute-optimal rule of thumb, shown as a reference only. Weights are bf16, 2 bytes per total parameter.

| d_model | Layers | d_model / layers | Heads | Expert FFN | Total | Active | Active / total | Tokens at 20 / active | 6·N·D at those tokens | bf16 weights |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4096 | 24 | 171 | 32 | 6144 | 31.8B | 6.5B | 0.203 | 129.4B | 5.02e+21 | 59 GiB |
| 4096 | 32 | 128 | 32 | 6144 | 42.4B | 8.6B | 0.203 | 172.4B | 8.91e+21 | 79 GiB |
| 4096 | 40 | 102 | 32 | 6144 | 53.0B | 10.8B | 0.203 | 215.3B | 1.39e+22 | 99 GiB |
| 4096 | 48 | 85 | 32 | 6144 | 63.6B | 12.9B | 0.203 | 258.3B | 2.00e+22 | 119 GiB |
| 4096 | 56 | 73 | 32 | 6144 | 74.3B | 15.1B | 0.203 | 301.2B | 2.72e+22 | 138 GiB |
| 4096 | 64 | 64 | 32 | 6144 | 84.9B | 17.2B | 0.203 | 344.2B | 3.55e+22 | 158 GiB |
| 4096 | 72 | 57 | 32 | 6144 | 95.5B | 19.4B | 0.203 | 387.2B | 4.50e+22 | 178 GiB |
| 4096 | 80 | 51 | 32 | 6144 | 106.1B | 21.5B | 0.203 | 430.1B | 5.55e+22 | 198 GiB |
| 4096 | 88 | 47 | 32 | 6144 | 116.7B | 23.7B | 0.203 | 473.1B | 6.71e+22 | 217 GiB |
| 5120 | 16 | 320 | 40 | 7680 | 33.1B | 6.7B | 0.202 | 134.0B | 5.39e+21 | 62 GiB |
| 5120 | 24 | 213 | 40 | 7680 | 49.7B | 10.0B | 0.202 | 200.7B | 1.21e+22 | 93 GiB |
| 5120 | 32 | 160 | 40 | 7680 | 66.2B | 13.4B | 0.202 | 267.4B | 2.15e+22 | 123 GiB |
| 5120 | 40 | 128 | 40 | 7680 | 82.8B | 16.7B | 0.202 | 334.1B | 3.35e+22 | 154 GiB |
| 5120 | 48 | 107 | 40 | 7680 | 99.3B | 20.0B | 0.202 | 400.8B | 4.82e+22 | 185 GiB |
| 5120 | 56 | 91 | 40 | 7680 | 115.9B | 23.4B | 0.202 | 467.5B | 6.56e+22 | 216 GiB |
| 6144 | 16 | 384 | 48 | 9216 | 47.6B | 9.6B | 0.201 | 192.0B | 1.11e+22 | 89 GiB |
| 6144 | 24 | 256 | 48 | 9216 | 71.5B | 14.4B | 0.201 | 287.6B | 2.48e+22 | 133 GiB |
| 6144 | 32 | 192 | 48 | 9216 | 95.3B | 19.2B | 0.201 | 383.3B | 4.41e+22 | 177 GiB |
| 6144 | 40 | 154 | 48 | 9216 | 119.1B | 23.9B | 0.201 | 478.9B | 6.88e+22 | 222 GiB |
| 7168 | 16 | 448 | 56 | 10752 | 64.8B | 13.0B | 0.201 | 260.4B | 2.03e+22 | 121 GiB |
| 7168 | 24 | 299 | 56 | 10752 | 97.2B | 19.5B | 0.201 | 390.1B | 4.57e+22 | 181 GiB |

## 4. Tokens per parameter, when the corpus-v1 count is known

Run `python3 harness/sizing.py --corpus-tokens <corpus-v1 tokens>`. It adds a table with tokens per active
parameter at 1, 2 and 4 epochs, tokens per total parameter, and 6·N·D for one epoch, for every row of the grid.

How to read it:

- At 20 tokens per active parameter (the dense compute-optimal rule of thumb; MoE is not dense, so a reference only),
  30B–120B total needs about 130B–480B training tokens.
- If corpus-v1 is smaller than that, the options are a smaller model, or more than one epoch over the corpus. The
  table shows both.

## 5. Open inputs

- The corpus-v1 token count (Team 1 / Team 2).
- The context length (separate task this week) and the KV-head count. Both change attention and the KV cache but
  hardly change the parameter count.
- The expert FFN ratio at scale.
- Whether the MTP head stays in the final model.
