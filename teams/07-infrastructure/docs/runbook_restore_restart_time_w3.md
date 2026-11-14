# G7 runbook: restore, restart and restore-time runs (week 3 rotation)

Team 07, Infrastructure. Rotation member: Dharshan (week 3). Date: 11 Nov 2026.
Results only: no code was changed. Commits tested: Machine A (Colab), second session, lab `main` at
4fd6c4973e5d40aa17dc63b3a6145e2d2ac9710d (all tests in section 2.1); Machine A, first session, a fresh clone of
`main` the same day whose hash was not recorded (the restore-time sweep in section 3 comes from this session);
Machine B (own PC), `main` at 2de8714.

The commands are the ones in `jobs/restore_test/README.md`, `jobs/restart_test/README.md` and
`jobs/restore_time/README.md`. This file records what happened when a new member ran them on two machines.

## 1. Machines

| | Machine A (Colab) | Machine B (own PC) |
|---|---|---|
| OS | Linux (Colab CPU runtime) | Windows 11, native PowerShell |
| Python | 3.13.18 | 3.14.6 |
| torch | 2.14.1+cpu | 2.15.0+cpu |
| numpy | 2.2.6 (pinned file) | 2.5.4 |
| Install | `pip install -r teams/04-ke2/src/requirements.txt` | pinned file failed (see 4.1); installed torch, numpy, pyyaml and the pinned `kronecker-embeddings` commit by hand |

## 2. Results on Machine A (Colab, CPU)

| Test | Command (from repo root, or `teams/07-infrastructure/` for the last three) | Result |
|---|---|---|
| Restore, 1 process | `python teams/07-infrastructure/jobs/restore_test/restore_test.py` | PASS, steps 101-200 bit-equal, data ids equal |
| Restore, 2 processes | `... restore_test.py --nproc 2` | PASS, steps 101-200 bit-equal, data ids equal |
| Restart (SIGKILL) | `CUDA_VISIBLE_DEVICES="" python jobs/restart_test/restart_test.py --run-id g7-restart-w3-20261111-01 --device cpu` | PASS, status pass, exit 0 |
| Queue rules | `python queue/queue_rules.py check-out ./out --run-id g7-restart-w3-20261111-01` | PASS: `./out` follows queue rules v1 |
| Restore time | `python jobs/restore_time/restore_time.py --note "Colab CPU, Linux, Python 3.13, torch 2.14.1"` | finished, wrote `restore_time.json` |

Restart test numbers (`result.json`): 200 of 200 steps bit-equal, `first_diff` null, killed with SIGKILL after
step 140, restarted once from the COMPLETE checkpoint at step 125, 15 steps redone, run A 10 s, run B 13 s,
elapsed 23 s. Supervisor log: attempt 0 exit -9 (SIGKILL), restart; attempt 1 exit 0, stop.

Checkpoints were captured: the restart run committed steps 25, 50, 75, 100, 125, 150, 175 and 200 for arm b
(the step-150 and later commits are epoch 2, after the restart), and `latest.json` ended at step 200.
No work was lost, and the numbers match run A bit for bit.

### 2.1 Second Colab session (main at 4fd6c49, all tests again)

The Colab session was reset, so the tests were run again on a fresh clone and the commit was recorded.

| Test | Result |
|---|---|
| `pytest ckpt/test_supervisor.py -q` | 19 passed in 3.61 s |
| `pytest support/tests/test_ckpt.py -k restore_job -q` | 2 passed, 36 deselected, in 66.30 s |
| Restore, 1 process | PASS, steps 101-200 bit-equal, data ids equal |
| Restore, 2 processes | PASS, steps 101-200 bit-equal, data ids equal |
| Restart (run ID `g7-restart-w3-20261111-02`) | PASS: SIGKILL after step 140, restarted from COMPLETE checkpoint step 125, steps 1-200 bit-equal, data ids equal; supervisor: attempt 0 exit -9, attempt 1 exit 0 |
| Queue rules check | PASS: `./out` follows queue rules v1 |

The result is the same as in the first session.

## 3. Restore time (Machine A, first session, median of 9 runs)

| Model | Checkpoint | Restore (median) | Min to max |
|---|---|---|---|
| Tiny MoE, 1 process | 1.5 MB | 0.32 s | 0.26 to 0.56 s |
| Tiny MoE, 2 processes | 1.6 MB | 0.31 s | 0.30 to 0.75 s |
| Synthetic, width 512 | 25 MB | 0.86 s | 0.84 to 1.32 s |
| Synthetic, width 1024 | 101 MB | 3.29 s | 3.23 to 4.93 s |
| Synthetic, width 2048 | 403 MB | 15.0 s | 14.4 to 25.0 s |
| Synthetic, width 4096 | 1,611 MB | 116 s | 98 to 131 s |

Phase split at 1,611 MB (medians): verify 5.3 s, local copy 63.9 s, load 0.9 s, the four proofs 45.2 s.
At every synthetic size the copy and the proofs take almost all of the time; loading the weights is under 1 s.

Limits (from the restore-time README): sizes are synthetic, the store is a local folder with a warm page cache,
one process, CPU only. This is not CK-13 (job start to first productive step).

### 3.1 Comparison with the team's numbers (runbook.md, section 8)

These Colab numbers are **much slower** than the team's laptop numbers. They do not reproduce them.

| | Team (runbook section 8, macOS laptops) | This run (Colab CPU) | Ratio |
|---|---|---|---|
| Tiny MoE, 1 process, restore | 0.031 to 0.071 s | 0.32 s | 4.5 to 10 times slower |
| Synthetic 25 MB, total | 0.070 s | 0.86 s | about 12 times slower |
| Synthetic 101 MB, total | 0.256 s | 3.29 s | about 13 times slower |
| Synthetic 403 MB, total | 1.010 s | 15.0 s | about 15 times slower |
| Synthetic 1,611 MB, total | 4.514 s | 116 s | about 26 times slower |
| Local copy at 1,611 MB | 1.08 s | 63.9 s | about 59 times slower |

The team found restore time roughly linear in size (2.5 to 2.8 s per GB). Here, a 64 times larger checkpoint
(25 MB to 1,611 MB) took 135 times longer (about 72 s per GB at the top), so growth is steeper than linear on
Colab. The local-copy phase accounts for most of the gap. The cause was not tested; a slow shared disk and
shared CPU on the Colab runtime are likely but unconfirmed. The restore is still correct (all checks pass, the
losses are bit-equal), only slower. Do not use these Colab times as the team's restore-time figure.

## 4. Results on Machine B (own PC, Windows 11, CPU)

| Test | Result |
|---|---|
| Restore, 1 process | PASS, steps 101-200 bit-equal, data ids equal |
| Restore, 2 processes | PASS, same result. Gloo prints `[c10d] ... socket ... error 10049` warnings; the run still passes |
| Restart (SIGKILL) | did not finish: `OSError(22, 'Incorrect function') ... lost sys.stderr`; no `result.json` was written |
| Queue rules check | FAIL only because the restart test wrote no `result.json` |
| Restore time | width 512: 0.31 s, width 1024: 1.16 s; then crashed at width 2048 with `SystemError: ... tell ... returned a result with an exception set` inside `torch.distributed.checkpoint` |

### 4.1 Notes on Machine B

- `pip install -r teams/04-ke2/src/requirements.txt` fails on Python 3.14 because `numpy==2.2.6` has no
  ready-made build and the source build fails. The restore test needs only `kronecker_embeddings`, torch, numpy and
  pyyaml, so those were installed by hand.
- The restart-test failure is on native Windows with Python 3.14 and torch 2.15, outside the Python 3.13 and
  torch 2.14.1 setup in the READMEs. The cause was not investigated further. The same test passes on Linux
  (section 2).
- The restore-time crash is inside torch's checkpoint reader on this setup. The same script finished on Linux.

## 5. How to reproduce

Use Linux (or Colab), Python 3.13, and the pinned requirements file. Windows works for the restore test only.

```bash
git clone --depth 1 https://github.com/theschoolofai/era-v5-lab.git && cd era-v5-lab
pip install -r teams/04-ke2/src/requirements.txt
python teams/07-infrastructure/jobs/restore_test/restore_test.py
python teams/07-infrastructure/jobs/restore_test/restore_test.py --nproc 2
cd teams/07-infrastructure
CUDA_VISIBLE_DEVICES="" python jobs/restart_test/restart_test.py --run-id <run-id> --device cpu
python queue/queue_rules.py check-out ./out --run-id <run-id>
python jobs/restore_time/restore_time.py --note "<machine>"
```

## 6. Summary

On a second, independent machine (Colab CPU) the restore test, the SIGKILL restart test and the queue-rules
check all pass, with equal loss and equal data ids, and checkpoints were captured at every 25 steps. This
reproduces the team's correctness evidence. It does not reproduce the team's restore times: Colab is 4.5 to 26
times slower, and the gap grows with checkpoint size (section 3.1). Native Windows with Python 3.14 passes the
restore test but not the restart test or the full restore-time sweep.

Both pytest suites also pass (section 2.1). Not run: the GPU variants (no GPU), and a re-run of the restore-time
sweep on the second session's clone.
