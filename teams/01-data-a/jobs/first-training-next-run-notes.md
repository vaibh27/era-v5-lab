# First-training job: what the next run should change

`2026-11-19-first-training.md` and its folder are the submitted job and stay as they are. A rerun gets its own job request and
folder, copied from them, with the changes below.

## prepare_run.py: three required worker flags

`--clean-workers`, `--f2-workers` and `--shard-workers`, all required and positive, written into `run.json` as
`stages.clean.workers`, `stages.f2.workers` and `stages.shards.workers`. None may have a default, because each default is slow:
`clean.json` says 4 clean workers, F2 at one worker takes about 3.1 h instead of 12 min, and the shard stage writes one pack at
a time. Values for the job host (32 vCPU, 64 GiB): `--clean-workers 16` (about 0.5 GB each), `--f2-workers 30` (the fingerprint
set is shared), `--shard-workers 8` (each part has a process of its own, 1.2 to 3.7 GB on the sample, so up to about 30 GB).
Tests: each value reaches `run.json`, an invalid value (0, negative, `True`, `"30"`) is refused, a missing flag stops
`main()` before `prepare` runs.

## Job request and README

- The prepare command carries all three flags; the sentence on worker counts names them and why none has a default.
- The MinHash workers are not set by `prepare_run.py`: both MinHash settings files say 16 (#89), which the sizing gate measures.
- The shard stage writes `--shard-workers` packs at once; the largest pack sets the shortest time (about 90 s in the 24 Nov
  sample run). Measured on this laptop: the 12 Indic packs with 2 workers took 120 s against about 163 s one at a time, with
  identical shard data. The scaling at 8 workers on the job host is not measured.
- `stages.shards.workers` is documented in `pipeline/README.md`.
