# Team 07 runbook: checkpoint restore test

This runbook records the independent checkpoint restore test performed in Google Colab on 12 November 2026. It covers restore correctness for one and two CPU processes and the time taken by the Tiny MoE restore test.

**Author:** Sonu Agarwal  
**Repository:** `theschoolofai/era-v5-lab`  
**Tested commit:** `4fd6c4973e5d40aa17dc63b3a6145e2d2ac9710d`  
**Overall result:** Restore correctness passed for both process configurations.

---

## 1. What was tested

The restore test checks whether training can continue from a saved checkpoint and reproduce the uninterrupted training result.

The test was run with one CPU process and two CPU processes. Tiny MoE restore timings were recorded for both configurations. These are timing measurements for the restore test, not separate scalability experiments.

## 2. Environment

| Parameter | Recorded value |
|---|---|
| Runtime | Google Colab |
| Operating system | Linux 6.6.159+ |
| Architecture | x86_64 |
| Python | 3.13.18 |
| PyTorch | 2.14.1+cpu |
| CPU count | 2 |
| PyTorch CPU threads | 1 |
| CUDA available | No |
| Total RAM | 12.67 GB |
| Available RAM at inspection | 11.28 GB |
| Device | CPU |
| Repository commit | `4fd6c4973e5d40aa17dc63b3a6145e2d2ac9710d` |

## 3. Restore test results

The test compares a run resumed from a checkpoint with an uninterrupted reference run. The checkpoint is saved at step 100. The resumed run continues from that checkpoint, and the results for steps 101–200 are compared against the uninterrupted reference.

### 3.1 One CPU process

**PASS — exit code 0**

| Measurement | Result |
|---|---:|
| Process count | 1 |
| Whole test duration | 28.6720 s |
| Comparison interval | Steps 101–200 |
| `loss_ntp` | Bit-equal |
| `loss_mtp` | Bit-equal |
| Batch data IDs | Equal |

The restored run matched the uninterrupted reference for both loss values and batch data IDs over steps 101–200.

### 3.2 Two CPU processes

**PASS — exit code 0**

| Measurement | Result |
|---|---:|
| Process count | 2 |
| Whole test duration | 44.3425 s |
| Comparison interval | Steps 101–200 |
| `loss_ntp` | Bit-equal |
| `loss_mtp` | Bit-equal |
| Batch data IDs | Equal |

The two-process run also matched the uninterrupted reference for both loss values and batch data IDs over steps 101–200.

### 3.3 Correctness conclusion

Both restore configurations passed. This confirms that, in the tested CPU environment, training resumed from the checkpoint with the expected losses and data sequence.

## 4. Restore time

Restore time is reported as `restore_s` in the `restore.done` event. It is the time spent in the restore operation, including verification, download/copy, loading, and correctness checks.

The whole-script wall time is reported separately because it includes work outside the restore operation.

The Tiny MoE checkpoint was approximately 1.5 MB. Each process configuration was measured over five repetitions.

### 4.1 One CPU process

| Restore phase | Median (s) | Minimum (s) | Maximum (s) |
|---|---:|---:|---:|
| Verification (`verify_s`) | 0.0058 | 0.0058 | 0.0060 |
| Download / copy (`download_s`) | 0.0075 | 0.0071 | 0.0076 |
| Load (`load_s`) | 0.1221 | 0.1197 | 0.1231 |
| Correctness checks (`checks_s`) | 0.0699 | 0.0691 | 0.0713 |
| **Total restore (`restore_s`)** | **0.2249** | **0.2226** | **0.2260** |
| Whole script wall time | 26.8392 | 25.4695 | 28.2718 |

Checkpoint size: **1,536,063 bytes (approximately 1.54 MB).**

### 4.2 Two CPU processes

| Restore phase | Median (s) | Minimum (s) | Maximum (s) |
|---|---:|---:|---:|
| Verification (`verify_s`) | 0.0072 | 0.0070 | 0.0139 |
| Download / copy (`download_s`) | 0.0096 | 0.0090 | 0.0133 |
| Load (`load_s`) | 0.2206 | 0.2069 | 0.4488 |
| Correctness checks (`checks_s`) | 0.0786 | 0.0771 | 0.1323 |
| **Total restore (`restore_s`)** | **0.3536** | **0.3335** | **0.6406** |
| Whole script wall time | 39.9530 | 39.4496 | 41.2230 |

Checkpoint size: **1,557,359 bytes (approximately 1.56 MB).**

### 4.3 Timing observation

For this Tiny MoE workload in the recorded CPU environment:

- The one-process median restore time was 0.2249 seconds.
- The two-process median restore time was 0.3536 seconds.
- Loading and correctness checks accounted for most of the measured restore time.
- Whole-script wall time is not the same as `restore_s` and should not be used as the restore duration.

These are measurements from this environment and workload, not production performance estimates.

## 5. Checkpoint verification status

The restore correctness tests passed and the run recorded a `restore.done` event.

A separate manual checkpoint verification command was attempted using the parent checkpoint-store directory. It returned an `IndexError` because the CLI did not find the requested checkpoint at that directory level.

**Status: NOT VERIFIED.**

The checkpoint-listing attempt also exited with code 0 but did not produce a useful listing when the parent store directory was supplied. The run-specific checkpoint directories were identified by inspecting the directory structure; the inspected `restore-b` checkpoint was at step 100.

This manual CLI attempt must not be reported as a successful standalone integrity verification. It remains a separate pending check and does not change the PASS result of the restore correctness tests.

## 6. Limitations

- The tests were run on CPU; CUDA was unavailable.
- The environment had two CPUs and PyTorch was configured to use one CPU thread.
- The Tiny MoE checkpoint was approximately 1.5 MB.
- The timings were measured using a local folder with a warm page cache.
- The results do not measure remote object-store or production-cluster performance.
- The separate manual checkpoint CLI verification remains unverified.

## 7. Final result

| Requirement covered in this runbook | Status | Result |
|---|---|---|
| Restore correctness — one CPU process | **PASS** | Losses and data IDs matched for steps 101–200 |
| Restore correctness — two CPU processes | **PASS** | Losses and data IDs matched for steps 101–200 |
| Record Tiny MoE restore time — one process | **MEASURED** | Median `restore_s`: 0.2249 s |
| Record Tiny MoE restore time — two processes | **MEASURED** | Median `restore_s`: 0.3536 s |
| Separate manual checkpoint CLI verification | **NOT VERIFIED** | Initial attempt used the parent store directory and returned an `IndexError` |

**Conclusion:** The checkpoint restore correctness test passed for both one-process and two-process CPU configurations. The Tiny MoE restore time was measured for each configuration and is reported separately from whole-script wall time. The standalone manual checkpoint CLI verification remains unverified.
