# Results v1

Gate 2 proposal, 12 Nov 2026. A bits-per-byte number is recorded only when the run is a result. A missing number keeps the config v0 default. The provisional T1 learning rate is recorded below. No other arm is decided.

The EXP-1 numbers are final whole-file val/bpb from the TSAI comment on pull request 310 (7 Nov 2026). Each run reached 245 steps. The value is the summed next-token loss in nats, divided by ln 2, divided by the UTF-8 byte count. It is one number for the whole held-out file. It is not the three-group mean, and it has no English, code, or Indic split.

## EXP-1, whole-file val/bpb

| AdamW rate | Seed 1234 | Seed 1235 | Seed 1236 | Mean | SD | Seed spread |
|---|---|---|---|---|---|---|
| 1e-4 | 2.3730 | 2.3653 | 2.3862 | 2.3749 | 0.0106 | 0.0209 |
| 3e-4 | 2.1357 | 2.1214 | 2.1642 | 2.1405 | 0.0218 | 0.0428 |
| 1e-3 | 1.9757 | 1.9617 | 1.9684 | 1.9686 | 0.0070 | 0.0140 |
| 3e-3 | 2.1000 | 2.0377 | 2.0935 | 2.0771 | 0.0342 | 0.0623 |
| 1e-2 | 2.3222 | 2.1969 | 2.3417 | 2.2869 | 0.0786 | 0.1448 |

Seed spread is the highest seed minus the lowest seed in the posted table. Means and standard deviations are the values TSAI posted.

The lowest mean is 1e-3 at 1.9686. The second is 3e-3 at 2.0771. The gap is 0.1085. The larger of those two seed spreads is 0.0623. 1e-3 is also the lowest on every seed, and its spread, 0.0140, is the smallest of the five.

The instructor set `optim.lr` to 1e-3 on 12 Nov 2026, provisional, so pull request 374 can carry it. The three-group, code, and Indic values were not measured. The choice holds for 245-step T1 runs only. EXP-2 and EXP-3 keep their own test of the rate.

## Arms with no bits per byte

| Arm | Owner | Bits per byte | Why there is no number |
|---|---|---|---|
| exp2-adamw | Vardhan Walavalkar | none | Job request merged in pull request 354. TSAI has not posted a result. AdamW 0.001 on that job is not final. |
| exp2-muon | Vardhan Walavalkar | none | Same job. Muon 0.01 is provisional. The T1 screen has training loss only, not held-out bits per byte. |
| exp3-base | Dharshan Thaigu | none | No job is filed. Draft pull request 360 is not a request. |
| exp3-experts-8 | Dharshan Thaigu | none | Same. |
| exp3-experts-32 | Dharshan Thaigu | none | Same. |
| exp3-shared-0 | Dharshan Thaigu | none | Same. |
| attn-0 | Revant Gupta | none | The fifteen-run result is not posted. |
| attn-1 | Revant Gupta | none | Same. |
| attn-2 | Revant Gupta | none | Same. |
| attn-3 | Revant Gupta | none | Same. |
| attn-4 | Revant Gupta | none | Same. |
| t3-confirm | the owner of a family that has a winner | none | No family has a winner, so T3 does not run. |

The Muon screen comment on pull request 241 gives last training cross-entropy for one seed: 7.086 at 1e-3, 6.626 at 1e-2, and 7.287 at 1e-1. Training loss is not bits per byte. Pull request 356 records that screen and names no winner.

The attention one-seed screen in pull request 345 is marked smoke. It is not the fifteen-run result, so its numbers are not copied here.
