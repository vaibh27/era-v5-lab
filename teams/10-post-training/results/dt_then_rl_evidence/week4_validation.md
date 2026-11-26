# Task 11 dry run 2 preparation — 24 November 2026

Validation was run on CPU with Python 3.12.14 and PyTorch 2.14.1. The delivered dry run 2 checkpoint and its DT SFT export were unavailable locally. GPU code-verifier reward and timing measurements remain pending.

```text
$ PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python -m pytest teams/10-post-training/capstone/test_rl_job_v1.py teams/10-post-training/capstone/test_router_bias_kas.py teams/10-post-training/capstone/test_dryrun2_dt_compare.py teams/10-post-training/rl-loop/test_keep_zero.py teams/10-post-training/rl-loop/test_task11_kas_mask.py -q -p no:cacheprovider
63 passed, 3 skipped, 6 warnings in 13.72s
exit=0
```

The three skips require external stand-in/checkpoint fixtures. The warnings come from Team 7 checkpoint save/load tests using a single process without distributed initialization.

The wrapper tests verify the exported checkpoint, mask and metadata against SHA256SUMS and refuse changed bytes. They check generated keep_zero settings and reject missing or violated deletion integrity in a completed report. Additional tests reject the wrong head or router-bias settings and compare the command in the request with the trainer template. The existing pinned-list test checks all 66 free-reward IDs and the 55 effort-suite IDs. Pool tests keep exclusions out of training and preserve the unfiltered held-out subset.

The new KAS integration test constructs Team 5's actual KE2/KAS V2 classes with a nonzero router bias and a baked gate mask, then runs the real objective_v1 trainer for three CPU updates through g5_batch. It checks identical-weight log-probability parity, surviving-weight changes, zero regrowth, and frozen router weights and bias. Its rewards are scripted fixtures used to exercise updates, so this test supplies no code-learning or held-out accuracy result.

Before submission to TSAI, append the exact staged wrapper command, input pins, hardware, exit status, final output and wall time from an end-to-end run. The current job request remains a draft pending those inputs and lead approval.

After merging the concurrent Week 4 update (base `8e5f5389`), the wrapper test additions were combined and the completion-log-probability suite was included to cover the updated trainer path:

```text
$ PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 python -m pytest teams/10-post-training/capstone/test_rl_job_v1.py teams/10-post-training/capstone/test_router_bias_kas.py teams/10-post-training/capstone/test_dryrun2_dt_compare.py teams/10-post-training/rl-loop/test_keep_zero.py teams/10-post-training/rl-loop/test_task11_kas_mask.py teams/10-post-training/rl-loop/test_completion_logprobs.py -q -p no:cacheprovider
67 passed, 7 skipped, 6 warnings in 22.27s
exit=0
```

The four additional skips require the external Team 5 stand-in run folder and its tokenizer/table artifacts. The KAS mask integration test passed on the merged trainer path.
