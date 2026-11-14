Paired against **bf16** on the same items. Differences in points (format minus base); b = right in base, wrong in format; c = the reverse.

| Format | Task | n | Base | Format | Difference (95% CI) | b / c | McNemar p |
|---|---|---|---|---|---|---|---|
| Q4_K_M | arc_challenge | 500 | 46.6 | 45.2 | -1.4 (-3.4 to +0.6) | 16 / 9 | 0.230 |
| Q4_K_M | hellaswag | 400 | 62.3 | 61.8 | -0.5 (-2.0 to +1.0) | 6 / 4 | 0.754 |
| Q4_K_M | mmlu | 570 | 64.4 | 64.6 | +0.2 (-1.9 to +2.2) | 17 / 18 | 1.000 |
| Q4_K_M | gsm8k | 200 | 65.0 | 65.5 | +0.5 (-5.8 to +6.8) | 20 / 21 | 1.000 |
| Q4_K_M | truthfulqa_mc2 | 200 | 52.3 | 53.2 | +0.9 (-0.0 to +1.8) | - | - |
