# G6 Week-3 checkpoint timing — `dryrun1_ckpt`

**Status: FOR LEAD QA — same-script CPU smoke output must be attached before TSAI execution.**

```text
Team:            06-performance
Owner:           Siva Ajjapu (AJJAPUSIVA)
Lead approval:   raj-harit-v2
Question:        What are save_ms, to_cpu_ms, write_ms and size_mb for the signed G5 small-width MoE checkpoint on one lab GPU?
Script:          teams/06-performance/jobs/dryrun1_ckpt.py
Command:         python jobs/dryrun1_ckpt.py --device cuda --repeats 10 --run-id dryrun1_ckpt
Config:          teams/05-architecture/configs/config_v1.yaml + explicit G6 small-width mapping in script
Inputs:          synthetic (no network or tokenizer download)
Hardware:        1 GPU (Linux lab)
Max time:        60
Precision:       fp32 (G5 production precision remains open)
Measures:        checkpoint.save_ms; checkpoint.to_cpu_ms; checkpoint.write_ms; checkpoint.size_mb; GPU; source SHA
Expected result: exit 0; complete JSON under out/dryrun1_ckpt.json; small-width save likely sub-second on local SSD, not a full-width prediction
Outputs:         ./out/dryrun1_ckpt.json (checkpoint temporary file deleted after timing)
Test log:        teams/06-performance/jobs/2026-11-15-dryrun1-ckpt-kaggle.txt (preceding Kaggle rehearsal, not the same-script smoke; pending)
```

## Pins and limitations

- Instructor-main baseline used for the Kaggle rehearsal: `6728a7a02f13fd7e0d2d230241407d9059ffff36`.
- G5 signed config v1: `teams/05-architecture/configs/config_v1.yaml` from G5 PR #447 (`aec75af`).
- G5 signed model choices: Muon, eager attention, 16 routed experts + 1 shared, top-2.
- Dry Run 1 Muon LR is `0.01` per separate instructor instruction. G5 `optim.lr=0.001` remains provisional.
- **Full-width parameters and production precision are not assigned in config v1.** This is a valid small-width timing request, **not** a full-width measurement. Do not enter its values into the official full-width row.
- Run from `teams/06-performance/`. The program uses the existing `perfkit.bench.run_checkpoint()` and writes only under `./out/`.
- The benchmark times GPU-to-CPU transfer plus local `torch.save`, flush and `fsync`; it is *not* the G7 remote commit/restore protocol.
- For official full-width evidence, a separate approved request will be needed once dimensions and layout are fixed.

## Smoke and reviewer checklist

```bash
cd teams/06-performance
python jobs/dryrun1_ckpt.py --device cpu --repeats 2 --run-id dryrun1_ckpt_smoke
cat out/dryrun1_ckpt_smoke.json
```

- [ ] Run and attach the exact-script CPU smoke output above.
- [ ] Confirm the Git SHA and output JSON are recorded.
- [ ] Lead QA approves the small-width scope and one-GPU resources.
- [ ] TSAI schedules the approved job and returns measured logs.
