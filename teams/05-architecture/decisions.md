# Decisions for config v1

Gate 2 proposal, 12 Nov 2026. TSAI signs this on 14 Nov. The numbers are in `experiments/results_v1.md`. The provisional T1 learning rate is 1e-3. No other arm is decided. An arm with no bits per byte keeps the config v0 default. Model sizes are not copied in from the test-size line.

| Setting | Decision | Why |
|---|---|---|
| optim.lr | 0.001, provisional, T1 only | Instructor, 12 Nov 2026. 1e-3 has the lowest whole-file val/bpb on all three seeds and the smallest spread, mean 1.9686, SD 0.0070. The three-group, code, and Indic values were not measured. The choice holds for 245-step T1 runs only. EXP-2 and EXP-3 keep their own test of the rate. |
| optim.name | adamw | EXP-2 has no held-out bits per byte. |
| moe.n_expert | null | EXP-3 has no bits per byte. The test-size count is not copied in. |
| shared expert | unset | EXP-3 has no bits per byte. The test-size line is not copied in. |
| attention.impl | eager | The attention runs have no official bits per byte. RoPE and the window are not copied in. |
| embedding.ke2_output_dim | null | TSAI decides the output head. Every test model uses the dense untied head. |
| model.n_layer, model.d_model, model.n_head, model.d_ff, moe.top_k, data.seq_len | null | These stay empty. The T1, T2, and T3 lines are test sizes, not this file. |
| precision | null | Production precision is not chosen. |

EXP-4 is week 3. It is not in this file.

EXP-2, filed in pull request 354, uses AdamW 0.001 and Muon 0.01 with the comparison marked not final. Those rates are not copied into config v1.
