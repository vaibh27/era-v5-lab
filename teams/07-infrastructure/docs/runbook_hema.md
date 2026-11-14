# Runbook: CPU checkpoint restore and crash recovery

## When to act

| I see | I do |
|---|---|
| Exit 75 under a supervisor/scheduler | Do nothing; the configured supervisor/scheduler retries automatically. |
| Exit 78 or a held job | Read the cause and follow the [causes and actions](../support/docs/codes_and_actions.md). |
| CK-05 | A restore check failed. Do not continue on that checkpoint; inspect fallback to an older committed checkpoint. |
| CK-13 | Restore is slow. Inspect restore phases and store throughput; see section 3 for alarm thresholds. |
| NaN or loss spike | Roll back using `--from-step`, `--skip` and `--branch`; see the template below. |

All commands run from the repository root. Set `SCRATCH` to a disposable local
directory for logs; it is separate from the real run's checkpoint store.
Slot `rotation_03` is provisional until the team confirms the number.

## 1. Restore safely and prove the selected state

For a real interrupted run:

1. Preserve the failed attempt's logs and identify its run ID, configuration and store.
2. List committed checkpoints and verify the selected step as in section 2.
3. Restart the same trainer command with the same configuration, run ID and store.
   The trainer must call `Checkpointer.restore()` before consuming its next batch.
4. Confirm `restore.done.step`, the first resumed step and the batch identity in
   the trainer's logs. Inspect any fallback to an older checkpoint.
5. If the child exits 78, or every checkpoint is unusable, investigate the reported
   cause before another attempt. Do not disable integrity or environment checks.

Without a scheduler, this is a **template**, not a command measured in this rehearsal:

```bash
RUN_ID="$RUN_ID" V5_REMOTE_ROOT="$REMOTE_ROOT" python teams/07-infrastructure/ckpt/supervisor.py --log "$SCRATCH/supervisor.jsonl" -- python path/to/trainer.py --config path/to/config.yaml
```

Replace the trainer and config paths with the original run's command; its code
must use the same checkpoint environment variables. The measured restart test
in section 5 already wires the supervisor and restoring trainer together.

The trainer calls `Checkpointer.restore()` on startup. It claims a new epoch,
chooses committed candidates, checks file hashes and environment, downloads the
state, checks tensor names/shapes/dtypes, then restores the model, optimizer,
random generators and loader position.

| Restore proof | What must match |
|---|---|
| State digest | Logical model and optimizer tensors versus the saved digest |
| Probe loss | Raw loss bits for the current device class, when recorded |
| Next batch | Saved batch position and token hash |
| Frozen tensors | Each recorded frozen tensor's fingerprint |

These tiny-MoE measurements have `probe_recorded=true` in `restore.done` and
`frozen={}` in the manifests. The frozen proof had no entries to compare. The
full checkpoint suite separately tests rejection of a modified frozen tensor.

The trainer's **`restore.done.step` is the actual restored step**. The
supervisor's `pointer_step` only reports its pointer observation; a corrupt
newest checkpoint can make the trainer fall back to an older valid commit.
Inspect `restore.failed` followed by `restore.done` to detect such fallback.
The restore rehearsal explicitly requests step 100 and must restore that step.

The interrupted part since the last checkpoint is replayed. For a kill after
step K and restore from C, `K - C` completed steps need replaying. That is work
redone; matching final state and batch IDs establishes correct recovery.

| Crash run | Actual kill | Actual restore | Replayed steps | Equal steps |
|---|---:|---:|---:|---:|
| Default | 141 | 125 | 16 | 200/200 |
| Kill point 31 | 33 | 25 | 8 | 200/200 |
| Kill point 99 | 99 | 75 | 24 | 200/200 |
| Kill point 164 | 166 | 150 | 16 | 200/200 |

For failure causes and recovery actions, use the [cause/action reference](../support/docs/codes_and_actions.md).
An explicit `--from-step` fails if that step is invalid; it does not silently fall back.

The supervisor stops on child exit 0 or 78. It retries exit 75 up to its safe-stop
limit; it retries crashes up to `--max-restarts`, stopping earlier for a crash
loop (default three crashes in 30 minutes). It logs each attempt and decision.

Rollback template (trainer integration required; not executed in this rehearsal):

```bash
python path/to/trainer.py --config path/to/config.yaml --from-step K --skip N --branch recovery_1
```

Replace `K` with the last verified good step and `N` with the deliberate batch
skip count. Keep the original run/store and select a new branch. The trainer
must pass these options to `Checkpointer.restore()`; do not run placeholder paths.

## 2. Establish that a checkpoint was committed

Current code uses **COMPLETE as a state**, with no separate `COMPLETE` file:

1. Each rank uploads its shard and extra state, with checksums.
2. Rank 0 verifies every file and writes `MANIFEST.json` as the seal.
3. It conditionally moves `latest.json` to the sealed step; an older attempt
   cannot overwrite a newer attempt's pointer (fencing).
4. It appends the committed step to `commits.jsonl`.

Checkpoint layout:

```text
<run>/latest.json
<run>/commits.jsonl
<run>/<branch>/step_<N>-e<E>/MANIFEST.json
<run>/<branch>/step_<N>-e<E>/extra_rank<r>.json
<run>/<branch>/step_<N>-e<E>/sums_rank<r>.json
<run>/<branch>/step_<N>-e<E>/<checkpoint shard files>
```

After the rehearsal in section 5, the representative restore test's B store is `$SCRATCH/restore_1p_3/ckpt_store/restore-b`;
the restart test's B store is `$SCRATCH/restart/ckpt/b`.

```bash
python teams/07-infrastructure/ckpt/ckpt.py list --run "$SCRATCH/restore_1p_3/ckpt_store/restore-b"
python teams/07-infrastructure/ckpt/ckpt.py verify --run "$SCRATCH/restore_1p_3/ckpt_store/restore-b" --step 100
python teams/07-infrastructure/ckpt/ckpt.py list --run "$SCRATCH/restart/ckpt/b"
python teams/07-infrastructure/ckpt/ckpt.py verify --run "$SCRATCH/restart/ckpt/b" --step 125
```

For the last command, substitute the **observed** restored step if it differs
from 125. `verify` checks one selected checkpoint, latest by default; checking
every historical checkpoint requires repeating `--step` for each listed step.
Record `files`, `bad` and exit code; success has `bad: []` and exit 0.

Retain checkpoint inventory, manifest size/checksums and committed/restore events
as evidence. Keep model and checkpoint binaries in scratch storage.

The audit re-read **240 manifest-listed files across 76 checkpoints in 20 stores**:
all sizes and SHA-256 hashes match, every manifest has a commit record, and every
latest pointer names an existing committed manifest. There were no bad files.
The tiny-model manifests record `git_sha=unknown`; the separately recorded source
commit in section 4 establishes checkout provenance. This rehearsal does not prove
automatic Git provenance capture inside the checkpoint.
Follow-up: [checkpoint commit provenance issue](https://github.com/theschoolofai/era-v5-lab/issues/298).

## 3. Restore time and interpretation

Read `restore.done` from the restore test's `events/` files or the restart test's
`logs/events.jsonl`. Retain repeat count, checkpoint bytes and each phase.
The values below are measurements from these local runs.

| Measurement | 1 CPU process | 2 CPU processes |
|---|---:|---:|
| Repetitions / checkpoint bytes | 3 / 1526402, 1526482, 1526494 | 3 / 1547214, 1547226, 1547286 |
| Checksum `verify_s`, median seconds | 0.0007 | 0.0009 |
| Local copy `download_s`, median seconds | 0.0018 | 0.0067 |
| State load `load_s`, median seconds | 0.0151 | 0.0190 |
| Proofs `checks_s`, median seconds | 0.0087 | 0.0092 |
| Restore `restore_s`, median [min, max] seconds | 0.0345 [0.0335, 0.0357] | 0.0380 [0.0376, 0.0384] |
| Whole test, median [min, max] seconds | 5.0863 [4.9930, 6.6755] | 6.0263 [5.9846, 6.0596] |

`restore_s` measures one candidate inside `_restore_one`: it excludes the epoch
claim, process startup and model construction. Whole-test time also contains
training A/B and comparison. Neither measures CK-13's complete job-start to
first-productive-step recovery time. A local copy is labelled `download_s` by
the API even when no network is involved.

Lower training loss between early and late steps describes learning. Recovery
correctness instead requires **equal** loss bits and batch IDs at matching
steps. Record representative losses from the evidence without treating a loss
difference between process counts as a restore failure.

Representative training losses are retained in the JSON evidence.

Performance comparisons require matching checkpoint size, process count,
software, hardware, storage and cache state. These measurements alone do not
establish a code improvement or production-scale performance.

CK-13 thresholds in [`metrics/catalog.yaml`](../metrics/catalog.yaml) are
**LOW 600 s, MEDIUM 900 s, HIGH 1800 s**, measured from job start to the first
productive step. Compare that full interval to the alarm, not `restore_s` alone.

The roughly 35 ms here is for a **1.5 MB tiny-model checkpoint** on a local CPU
store. It does not scale to the real model. A production estimate needs config
v1 checkpoint sizes and the real store; those measurements remain pending for
Saturday 14 November in this Week 2 plan.

## 4. Environment and result

All commands below run from the **repository root**, with the test virtual
environment active. Check out the recorded source commit before reproducing.

| Item | Recorded value |
|---|---|
| Source commit | `ddfe977ea8f771ba6fa9137c1e9f46aade0125d1` |
| Run date | 4 November 2026 (IST) |
| OS / CPU / memory | macOS 15.7.4 / Apple M4 (arm64) / 16 GiB |
| Python / torch / dependency versions | Python 3.12.14; torch 2.14.1; numpy 2.2.6; pytest 9.1.1; full versions in JSON |
| Device / store / threads | CPU / local folder / one torch thread per trainer process |
| Overall result | PASS; detailed test outcomes in section 5 |

The source requirements are in `teams/04-ke2/src/requirements.txt`; the restore
script requires torch 2.14.1 or newer, numpy, PyYAML and the pinned
`kronecker_embeddings` package. Infrastructure tests also use pytest,
safetensors and boto3. Preserve the actual installed versions in the evidence.
An unavailable dependency is a limitation to report, rather than a reason to
weaken the tests or change their version checks.

In a Python 3.12 virtual environment, install from the repository root:

```bash
python -m pip install -r teams/04-ke2/src/requirements.txt
python -m pip install pyyaml==6.0.3 safetensors==0.8.0 boto3==1.43.123 'moto[s3]==5.1.10'
```

## 5. Run the complete CPU rehearsal

These commands use full 200-step training runs. The two-process case uses CPU
gloo; it does not establish GPU or multi-node performance. Each invocation clears
its own checkpoint folders. Distinct output directories
retain the three repetitions for later comparison.

```bash
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""
export SCRATCH=out/rotation_03
for n in 1 2; do
  for rep in 1 2 3; do
    python teams/07-infrastructure/jobs/restore_test/restore_test.py --nproc "$n" --out "$SCRATCH/restore_${n}p_${rep}"
  done
done
CUDA_VISIBLE_DEVICES="" python teams/07-infrastructure/jobs/restart_test/restart_test.py --out "$SCRATCH/restart" --also-kill-at 31,99,164 --report "$SCRATCH/restart_cpu_raw.txt"
python -m pytest -q -ra teams/07-infrastructure/support/tests teams/07-infrastructure/ckpt/test_supervisor.py teams/07-infrastructure/layout/test_parallel_layout.py
```

The pytest command includes the checkpoint failure scenarios, supervisor,
queue, health, monitoring, cost and parallel-layout tests. The shorter restore
jobs inside pytest supplement the full training runs above. Record failures,
warnings and backend-dependent skips, including their reasons.

| Test | Expected evidence | Measured outcome |
|---|---|---|
| Restore, 1 CPU process | Steps 101–200: both losses bit-equal; data IDs equal | PASS, 3/3 repetitions |
| Restore, 2 CPU processes | Same checks against a 2-process reference | PASS, 3/3 repetitions |
| Restart, default kill point | One SIGKILL restart; steps 1–200 bit-equal; data IDs equal | PASS, 200/200 steps |
| Restart, kill points 31/99/164 | Same recovery/equality checks for each kill point | PASS on each, 200/200 steps |
| Full G7 pytest command | Passed/failed/skipped totals and reasons | 119 passed, 0 failed, 10 skipped, 9 warnings; 62.67 s pytest time |

The ten skips are one FSDP2 case requiring two CUDA GPUs, eight cases requiring
DeepSpeed, and one DeepSpeed fp16 case requiring two CUDA GPUs. The S3 conditional
write case passed with moto 5.1.10; that is emulator coverage. The nine pytest
warnings concern single-process distributed-checkpoint operation. A single note
in the text evidence records PyTorch's repeated macOS redirect notices.

For a restore test, run A trains to step 200 and saves at step 100. Run B saves
at 100, stops at 150, then fresh processes restore 100 and repeat steps 101–200.
The script compares `loss_ntp_bits`, `loss_mtp_bits` and `batch_id` for each step.
Exit 0 means PASS, 1 means a difference, and 78 means a crashed run.

For the restart test, A trains uninterrupted while B runs under the supervisor.
B checkpoints every 25 steps and is killed after the requested step. Scheduling
may move the actual kill later; retain the observed step. Success requires
exactly one restart after exit -9, then exit 0, with the first resumed step equal
to the restored checkpoint step plus one. Every final step 1–200 must match A.

## 6. Scope and retained evidence

The [JSON summary](../results/rotation_03.json) and [text log](../results/rotation_03.txt)
record versions, results, timings, checkpoint audit totals, equality outcomes,
pytest skips/failures and limitations; executed commands remain in the text log.
The 1,400-row steps file is retained locally at
`out/rotation_03/rotation_03_steps.jsonl` relative to the repository root, outside
the committed evidence. It has not been uploaded publicly. The JSON records its
SHA-256, row count and comparison fields. Preserve that original scratch file;
request a copy from the submitter when verifying on another machine.

To check the retained file (set `SCRATCH=out/rotation_03` in your shell):

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
root = Path("teams/07-infrastructure/results")
summary = json.loads((root / "rotation_03.json").read_text())
raw = Path("out/rotation_03/rotation_03_steps.jsonl").read_bytes()
assert hashlib.sha256(raw).hexdigest() == summary["row_comparison"]["sha256"]
rows = [json.loads(line) for line in raw.splitlines()]
assert len(rows) == 1400
assert all(row["a"][key] == row["b"][key] for row in rows
           for key in ("loss_ntp_bits", "loss_mtp_bits", "batch_id"))
print("PASS: 1400 matching step comparisons")
PY
```

The measured restart summary is
[`restart_cpu_rotation_03.txt`](../results/restart_cpu_rotation_03.txt).
This summary is assembled from the four measured results and supervisor events;
the script's raw report stays in scratch storage.

These are local CPU/local-store runs. They do not establish GPU/NCCL,
multi-node, remote-store or config-v1 restore performance. There is no student
lab Slurm access; larger runs use the approved `JOB_REQUEST.md` process and
returned TSAI logs. Do not treat production Slurm examples as commands to run
on a student laptop. Queue `check-out` validation is pending
[queue-rules integration](https://github.com/theschoolofai/era-v5-lab/pull/245).
Add that step when the checker is available; it was not run in this rehearsal.
