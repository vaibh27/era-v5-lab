# Rows kept per SFT set after decontam

Rows in, removed and kept for the 33 SFT sets that were run on every converted row of the whole source (Team 9 fingerprints v2, full set, `set_sha256` `2826a3b883210c653979103c09ce1c79e40fc4f7034b54027f2836482370d22a`, 0 canaries). The numbers come from the run records behind `decontam_report.md`; every record passed the reconciliation checks in section 5 of that report.

"Rows in" counts the rows that `convert.py` converted. It skips a row whose trace is over the 64k effort budget, so a set can have more source rows than rows in (for example `sft-math-nemotron-proofs-v3-sft`: 414,890 source rows, 293,648 converted and checked). Kept rows are not token shares: the mix shares in `sft_mix_proposal.md` count tokens.

| Set | Rows in | Removed | Removed % | Rows kept |
|---|---:|---:|---:|---:|
| `sft-agentic-nemotron-agentic-v1` | 315,560 | 432 | 0.14% | 315,128 |
| `sft-agentic-nemotron-sft-agentic-v2-tool-calling` | 364,889 | 2,146 | 0.59% | 362,743 |
| `sft-bespoke-stratos-17k` | 16,710 | 1,791 | 10.72% | 14,919 |
| `sft-chat-nemotron-if-v1` | 4,963 | 1 | 0.02% | 4,962 |
| `sft-chat-nemotron-if-v2` | 929,130 | 5,820 | 0.63% | 923,310 |
| `sft-code-codeforces-cots` | 45,734 | 20,796 | 45.47% | 24,938 |
| `sft-code-commitpackft` | 702,062 | 1,001 | 0.14% | 701,061 |
| `sft-code-nemotron-cascade-swe` | 210,582 | 18,248 | 8.67% | 192,334 |
| `sft-code-nemotron-cuda` | 2,273 | 2 | 0.09% | 2,271 |
| `sft-code-nemotron-opencode` | 67,086 | 332 | 0.49% | 66,754 |
| `sft-code-nemotron-terminal` | 31,928 | 9,901 | 31.01% | 22,027 |
| `sft-code-opencodereasoning` | 567,831 | 112,753 | 19.86% | 455,078 |
| `sft-code-opencodereasoning-2` | 542,219 | 115,067 | 21.22% | 427,152 |
| `sft-code-swe-rebench-openhands` | 27,516 | 4,864 | 17.68% | 22,652 |
| `sft-code-swe-zero-openhands` | 317,039 | 22,167 | 6.99% | 294,872 |
| `sft-dolly-15k` | 15,011 | 4 | 0.03% | 15,007 |
| `sft-indic-aya-collection` | 654,530 | 6 | 0.00% | 654,524 |
| `sft-indic-aya-dataset` | 41,261 | 8 | 0.02% | 41,253 |
| `sft-indic-indic-align-dolly-t` | 10,541 | 0 | 0.00% | 10,541 |
| `sft-indic-indic-align-oasst-t` | 19,909 | 9 | 0.05% | 19,900 |
| `sft-math-nemotron-math-v2` | 7,079,896 | 60,665 | 0.86% | 7,019,231 |
| `sft-math-nemotron-proofs` | 920,831 | 2,357 | 0.26% | 918,474 |
| `sft-math-nemotron-proofs-v3-sft` | 293,648 | 3,012 | 1.03% | 290,636 |
| `sft-math-nemotron-sft-math-v4` | 542,117 | 4,776 | 0.88% | 537,341 |
| `sft-math-openr1-math` | 224,882 | 1,803 | 0.80% | 223,079 |
| `sft-math-openr1-math-raw` | 321,100 | 2,494 | 0.78% | 318,606 |
| `sft-oasst2` | 9,548 | 5 | 0.05% | 9,543 |
| `sft-rl-math-openmathreasoning` | 3,200,855 | 65,402 | 2.04% | 3,135,453 |
| `sft-salesforce-xlam-60k` | 60,000 | 182 | 0.30% | 59,818 |
| `sft-science-glm51-phd` | 103,702 | 207 | 0.20% | 103,495 |
| `sft-science-nemotron-science` | 174,155 | 369 | 0.21% | 173,786 |
| `sft-science-nemotron-science-v2` | 96,605 | 286 | 0.30% | 96,319 |
| `sft-stem-code-openthoughts3` | 450,599 | 7,457 | 1.65% | 443,142 |
| **33 sets** | **18,364,712** | **464,363** | **2.53%** | **17,900,349** |

## Notes on the table

- `sft-code-opencodereasoning-2`: all 70 python shards after the question join. 1,398,166 raw rows; 268,752 upstream test rows, 3,629 protected-family rows and 513,809 rows that fail the quality proposal are left out before the scan, and 69,690 `open-r1/codeforces` rows are held. 542,219 rows reach the scan. TSAI admitted the set on 11 Nov 2026 (`allowed (TSAI admitted, 11 Nov 2026)`, CC-BY-4.0, attribution retained), so it is licence-eligible for admission. No kept rows are admitted yet: the pack is blocked on source identity approval and verification, the frozen tokenizer, the context limit (`max_sequence_tokens`), and the Codeforces pool producer revision, as recorded in `licence_table.md` and the pack admission report.

- The total in section 1 of `decontam_report.md` (18,376,709 rows in) is this total plus the `sft-safety-aegis-v1` text column (11,997 rows, 2 removed), which is not training text.

## Not in this table

- `sft-safety-aegis-v1`: the text column only (11,997 rows, 2 removed). It has no assistant turn, so it is not training text.
- The excluded safety and code sets listed in section 7 of `decontam_report.md` are not run.
