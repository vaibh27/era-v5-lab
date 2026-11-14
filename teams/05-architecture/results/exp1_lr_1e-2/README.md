# EXP-1 Arm `exp1-lr-1e-2` (`optim.lr = 0.01`) — Pre-Agreement T1 Verification Logs

> **Status: Pre-agreement verification logs — NOT an approved EXP-1 quality result (`status = NOT_ELIGIBLE_NOT_A_QUALITY_RESULT`, `eligible_for_quality_comparison = false`).**
>
> Per lead review on PR #286, these 245-step `NVIDIA H200` `bf16` logs (seeds `1234`, `1235`, `1236`) and 4-step CPU smoke/resume logs are retained as verification evidence, with the quality-result claim removed until the following three items are in place on `main`:
> 1. **Approval & Evaluation Flags (`configs/runtime.yaml`):** On `main`, `approval.status` is `pending` (`approval.reference: null`) and `evaluation.approved` is `false`. All manifests in this directory record those `main` values (`pending`, `null`, `false`). For the 245-step `H200` CUDA runs (`seed_1234`, `seed_1235`, `seed_1236`), `manifest.json` sets `"smoke": false` and `"status": "NOT_ELIGIBLE_NOT_A_QUALITY_RESULT"`, and `summary.json` sets `"status": "NOT_ELIGIBLE_FOR_QUALITY_COMPARISON"` and `"eligible_for_quality_comparison": false`. The 4-step CPU runs (`cpu_smoke_a`, `cpu_smoke_b`, `cpu_resume`) stay `"smoke": true` (`"status": "SMOKE_NOT_A_QUALITY_RESULT"` / `"SMOKE"`, `"eligible_for_quality_comparison": false`).
> 2. **Shared EXP-1 Training File:** These runs used a local 3,247-document FineWeb sample (`10,001,358` bytes, SHA-256 `3647da224420ef7b53d2d0ae4cf9150914519e31a2f373c156f1657f5e26fc77`). The shared EXP-1 training file across all five learning-rate arms has not been named yet.
> 3. **Held-Out Provenance on Fingerprints v2:** `heldout_v1.jsonl` (`1,059` documents) was re-checked on **Fingerprints v2** (`set_sha256: 2826a3b883210c653979103c09ce1c79e40fc4f7034b54027f2836482370d22a`) via `python teams/09-evaluation/decontam/decontam_check.py teams/05-architecture/heldout_v1/data/heldout_v1.jsonl --id-field id --text-field text --exclude heldout_g5_v1` on `2026-11-04`, yielding `docs 1059 flagged 0 (0.00%) by benchmark {}`. Note that `teams/05-architecture/heldout_v1/heldout_report.json` on `main` still records `fingerprint_set_version: "v1"` (`2026-11-02`).

- **Owner:** Varshith Chennuru (`varshith2000`)
- **Arm Config:** `teams/05-architecture/configs/arms/exp1/lr_1e-2.yaml` (inherits `configs/arms/t1.yaml`)
- **Test Size:** `T1` (`d_model=256`, `n_layer=6`, `n_head=4`, `n_kv_head=2`, `num_experts=8`, `num_shared=1`, `top_k=2`, `ffn_hidden=512`, `block_size=1024`, `vocab_size=131072`, `58,080,512` parameters)
- **Harness:** `teams/05-architecture/harness/train_week2.py`
- **Runtime Test Settings (PR #253):** `rope_theta=10000.0`, `norm_eps=1.0e-5`, `mtp=false` (`mtp_weight=0.0`), `moe.bias=false` (`bias_gamma=0.0`), `moe.dispatch=loop`, `attn_impl=eager`, `ce_impl=torch`, `reversible=false`, dense untied output head (`nn.Linear(256, 131072, bias=False)`). The copied `intended.tied_embeddings_intended: true` flag in `manifest.json` (inherited from `configs/arms/t1.yaml`) is stale, and the run used the dense untied head.
- **Optimizer & Budget:** `AdamW` (`lr=0.01`, `betas=[0.9, 0.999]`, `eps=1e-8`, `weight_decay=0.0`, `grad_clip=1.0`, `schedule=constant`), `batch_size=8` (`8,192` tokens/step), `steps=245` (`2,007,040` input positions), `max_time_min=30`

---

## 1. Pre-Run Verification (CPU Gate & Artifact Hashes)

1. **Week-2 Unit Test Suite:**
   - `python -m pytest teams/05-architecture/tests/test_week2_*.py -q` -> **61 passed**
   - `python -m unittest discover -s teams/05-architecture/tests -p 'test_week2*.py' -v` -> **61 passed**
2. **Tokenizer v0 & G4 Format-v2 Raw Byte Table (`P=32`):**
   - `tokenizer_v0.json` SHA-256: `2de039f20d17afd17615095c406c43087359673b767ed0fe2c480b995f2af8e5` (`vocab_size=131072`, `bos_id=130725`, `pad_id=130726`)
   - `raw_table_P32.npz` content SHA-256: `47d411a9c6cae6fa2075935567243e51f4efbc377ccef3eef42c32fce66f869f` (`shape=(131072, 32)`, `n_truncated=0`)
3. **Runtime Schema Validation (`check_week2.py`):**
   - `python teams/05-architecture/harness/check_week2.py teams/05-architecture/configs/arms/exp1/lr_1e-2.yaml` -> **`0 errors`**
4. **CPU Smoke Repeatability & Checkpoint-Resume Verification (`seed=1234`, `steps=4`, `seq_len=64`, `batch_size=2`, `smoke=true`, `status=SMOKE`, `eligible_for_quality_comparison=false`):**
   - `cpu_smoke_a` and `cpu_smoke_b` produce identical `losses.txt` (`first_ce=11.893761 -> last_ce=10.685356`) and identical `final_model_sha256` (`5008e39a95a071f05fb416a1639ee73cd89db2c020bc270010ff46b03b85c11a`).
   - `cpu_resume` (`2 steps + stop + --resume -> 4 steps`) matches the uninterrupted 4-step runs bit-for-bit (`final_model_sha256: 5008e39a95a071f05fb416a1639ee73cd89db2c020bc270010ff46b03b85c11a`).

---

## 2. Held-Out Set & Training Stream Provenance

- **Official Held-Out Set (`heldout_v1`):**
  - File: `teams/05-architecture/heldout_v1/data/heldout_v1.jsonl` (`1,059` documents, `3,775,551` UTF-8 bytes, `585,097` scored tokens across 15 slices)
  - Held-out SHA-256 (`results/heldout_v1.sha256`): `21c6e4b2df8dae9d69e254f4a00fc934d2a3c70f0675065a03724299d94312bb`
  - Sources sidecar SHA-256 (`heldout_v1_sources.jsonl`): `95585a25bdccb077130432d49b6427c5ff5f756d5b43b9c7033803326f65e0d6`
  - **Team 9 Fingerprints v2 Check (`2026-11-04`):**
    - Command: `python teams/09-evaluation/decontam/decontam_check.py teams/05-architecture/heldout_v1/data/heldout_v1.jsonl --id-field id --text-field text --exclude heldout_g5_v1`
    - Output: `fingerprints v2 set_sha256 2826a3b883210c653979103c09ce1c79e40fc4f7034b54027f2836482370d22a (full set except heldout_g5_v1)` / `docs 1059 flagged 0 (0.00%) by benchmark {}`
    - *(Note: `teams/05-architecture/heldout_v1/heldout_report.json` on `main` still records `fingerprint_set_version="v1"`, `fingerprint_set_id="11e290a44d1467875fdf8bebc55eab21f0a6367f411aab98a4a027a546fd372e"`, checked on `2026-11-02`.)*
- **Local Verification Training Stream (`train_exp1_v1.jsonl` — NOT the shared EXP-1 training file):**
  - Source: `HuggingFaceFW/fineweb` (`sample-10BT`, `split="train"`, local sample with zero document overlap with `heldout_v1.jsonl`)
  - Documents: `3,247` (`10,001,358` bytes on disk)
  - SHA-256: `3647da224420ef7b53d2d0ae4cf9150914519e31a2f373c156f1657f5e26fc77`
  - Status: Local sample only (`shared_exp1_training_file = false`). Official EXP-1 comparison requires every arm to run on the same shared training file once named.

---

## 3. Pre-Agreement T1 GPU (`bf16`) Verification Logs (`steps = 245`, `smoke = false`, `eligible_for_quality_comparison = false`)

Seeds `1234`, `1235`, and `1236` each completed `245` steps (`timed_out=false`, `0` NaN/Inf steps) on `NVIDIA H200` in `bf16` under the merged runtime test settings (`norm_eps=1e-5`, `rope_theta=10000.0`, `mtp=false`, `moe.bias=false`) and the dense untied output head (note that the copied `intended.tied_embeddings_intended: true` flag in `manifest.json` is stale; the run used the untied head). Because `approval.status` (`pending`) and `evaluation.approved` (`false`) are not set on `main` and the shared EXP-1 training file is not named yet, these 245-step CUDA runs are recorded with `"smoke": false`, `"status": "NOT_ELIGIBLE_NOT_A_QUALITY_RESULT"` in `manifest.json`, and `"status": "NOT_ELIGIBLE_FOR_QUALITY_COMPARISON"`, `"eligible_for_quality_comparison": false` in `summary.json`.

### Summary Table Across Seeds (Pre-Agreement Verification)

| Metric | Seed `1234` | Seed `1235` | Seed `1236` | 3-Seed Mean | Seed Spread (`max - min`) |
|---|---:|---:|---:|---:|---:|
| **`val/three_group_bpb`** | **3.259499** | **3.258205** | **3.267639** | **3.261781** | **0.009435** |
| `val/groups/english_bpb` | 2.315322 | 2.312464 | 2.257152 | 2.294979 | 0.058170 |
| `val/groups/code_bpb` | 4.339244 | 4.278825 | 4.356545 | 4.324871 | 0.077720 |
| `val/groups/indic_mean_bpb` (11-slice mean) | 3.123932 | 3.183325 | 3.189221 | 3.165493 | 0.065290 |
| `val/bpb` (Whole-file aggregate) | 3.098675 | 3.140200 | 3.141975 | 3.126950 | 0.043300 |
| `train/ce_loss` (Step 1 -> Step 245, nats) | 11.963246 -> 7.520585 | 11.989516 -> 7.407004 | 11.952813 -> 6.930207 | 11.968525 -> 7.285932 | — |
| Wall time (s) | 411.12 s | 414.82 s | 468.50 s | 431.48 s | — |
| `final_model_sha256` | `15701d752a5c...` | `246226fadbc1...` | `927c0110e3fc...` | — | — |
| `smoke` / `eligible_for_quality_comparison` | `false` / `false` | `false` / `false` | `false` / `false` | — | — |

### Per-Slice Held-Out Bits Per Byte at Step 245 (All 15 Slices)

| Slice | UTF-8 Bytes | Scored Tokens | Seed `1234` BPB | Seed `1235` BPB | Seed `1236` BPB | 3-Seed Mean BPB | Seed Spread |
|---|---:|---:|---:|---:|---:|---:|---:|
| `en` (English C4 val) | 247,056 | 53,299 | 2.315322 | 2.312464 | 2.257152 | 2.294979 | 0.058170 |
| `code` (The Vault test) | 250,262 | 62,318 | 4.339244 | 4.278825 | 4.356545 | 4.324871 | 0.077720 |
| `math` (OpenWebMath test) | 259,449 | 71,800 | 3.366067 | 3.382192 | 3.363216 | 3.370492 | 0.018977 |
| `sci` (peS2o v2 val) | 250,710 | 49,492 | 2.273167 | 2.264724 | 2.246424 | 2.261438 | 0.026743 |
| `hi` (Hindi FW2 test) | 255,134 | 31,422 | 2.517169 | 2.591128 | 2.564747 | 2.557682 | 0.073959 |
| `bn` (Bengali FW2 test) | 265,042 | 35,583 | 2.815975 | 2.838291 | 2.865690 | 2.839985 | 0.049715 |
| `ta` (Tamil FW2 test) | 256,545 | 32,584 | 2.605492 | 2.788903 | 2.663792 | 2.686062 | 0.183411 |
| `te` (Telugu FW2 test) | 255,189 | 37,380 | 2.944173 | 3.050235 | 3.009685 | 3.001364 | 0.106062 |
| `kn` (Kannada FW2 test) | 250,459 | 36,397 | 2.960480 | 3.040983 | 3.014740 | 3.005401 | 0.080503 |
| `ml` (Malayalam FW2 test) | 251,018 | 31,840 | 2.579835 | 2.663615 | 2.634423 | 2.625958 | 0.083780 |
| `mr` (Marathi FW2 test) | 250,524 | 35,215 | 2.867656 | 2.969629 | 2.932354 | 2.923213 | 0.101973 |
| `gu` (Gujarati FW2 test) | 250,826 | 36,404 | 2.970832 | 3.121694 | 3.044471 | 3.045665 | 0.150862 |
| `pa` (Punjabi FW2 test) | 254,686 | 49,853 | 3.984098 | 4.003700 | 4.126082 | 4.037960 | 0.141984 |
| `or` (Odia FW2 test) | 225,421 | 50,916 | 4.791237 | 4.615224 | 4.844573 | 4.750345 | 0.229349 |
| `as` (Assamese FW2 test) | 253,230 | 40,078 | 3.326301 | 3.333173 | 3.380877 | 3.346784 | 0.054576 |

### Validation Trajectory Across Evaluation Steps (`eval_every_steps = 50`)

| Seed | Step | `train/ce_loss` | `val/three_group_bpb` | `val/groups/english_bpb` | `val/groups/code_bpb` | `val/groups/indic_mean_bpb` | `val/bpb` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1234 | 0 | — | 3.543433 | 3.724060 | 4.291528 | 2.614710 | 2.988942 |
| 1234 | 50 | 8.285070 | 3.372353 | 2.572048 | 4.341972 | 3.203039 | 3.210911 |
| 1234 | 100 | 8.044210 | 3.294330 | 2.467455 | 4.325041 | 3.090495 | 3.108580 |
| 1234 | 150 | 7.710623 | 3.203843 | 2.404289 | 4.254982 | 2.952258 | 2.983610 |
| 1234 | 200 | 7.308099 | 3.195456 | 2.350375 | 4.241992 | 2.994001 | 3.003088 |
| **1234** | **245** | **7.520585** | **3.259499** | **2.315322** | **4.339244** | **3.123932** | **3.098675** |
| 1235 | 0 | — | 3.546482 | 3.729428 | 4.295963 | 2.614054 | 2.990389 |
| 1235 | 50 | 8.022716 | 3.407820 | 2.579172 | 4.383688 | 3.260599 | 3.256203 |
| 1235 | 100 | 7.870254 | 3.321895 | 2.453493 | 4.266716 | 3.245478 | 3.207123 |
| 1235 | 150 | 7.553059 | 3.350016 | 2.400926 | 4.299817 | 3.349304 | 3.275054 |
| 1235 | 200 | 7.594085 | 3.323902 | 2.349742 | 4.290461 | 3.331503 | 3.253615 |
| **1235** | **245** | **7.407004** | **3.258205** | **2.312464** | **4.278825** | **3.183325** | **3.140200** |
| 1236 | 0 | — | 3.540922 | 3.716208 | 4.284336 | 2.622221 | 2.992535 |
| 1236 | 50 | 8.720010 | 3.492818 | 2.572610 | 4.499094 | 3.406750 | 3.374109 |
| 1236 | 100 | 7.780025 | 3.357289 | 2.422553 | 4.333994 | 3.315322 | 3.260173 |
| 1236 | 150 | 7.251181 | 3.259132 | 2.328348 | 4.306438 | 3.142611 | 3.116686 |
| 1236 | 200 | 7.025902 | 3.287790 | 2.288170 | 4.356040 | 3.219159 | 3.169770 |
| **1236** | **245** | **6.930207** | **3.267639** | **2.257152** | **4.356545** | **3.189221** | **3.141975** |

---

## 4. Files in This Folder

- `seed_1234/` — Pre-agreement 245-step T1 GPU (`bf16`) verification logs for seed `1234` (`manifest.json`, `metrics.jsonl`, `losses.txt`, `total_losses.txt`, `summary.json`; `smoke: false`, `status: NOT_ELIGIBLE_NOT_A_QUALITY_RESULT` / `NOT_ELIGIBLE_FOR_QUALITY_COMPARISON`, `eligible_for_quality_comparison: false`)
- `seed_1235/` — Pre-agreement 245-step T1 GPU (`bf16`) verification logs for seed `1235` (`smoke: false`, `eligible_for_quality_comparison: false`)
- `seed_1236/` — Pre-agreement 245-step T1 GPU (`bf16`) verification logs for seed `1236` (`smoke: false`, `eligible_for_quality_comparison: false`)
- `train_exp1_v1_meta.json` — Metadata & SHA-256 checksum for the local 3,247-document FineWeb sample (`shared_exp1_training_file: false`)
- `cpu_smoke_a/`, `cpu_smoke_b/`, `cpu_resume/` — 4-step CPU determinism and checkpoint-resume smoke verification artifacts (`smoke: true`, `status: SMOKE`, `eligible_for_quality_comparison: false`)
