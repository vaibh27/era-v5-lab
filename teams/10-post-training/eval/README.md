# Effort evaluation (Package T2)

A fixed set of prompts crossed with the four effort labels `1k`, `4k`, `16k` and `64k`. The requested label and the enforced generation cap are separate fields in every record. The suite measures three things apart: whether the output follows the format, whether it stays inside the cap, and whether it is correct. A text instruction is not a budget mechanism.

| File | What it is |
|---|---|
| `effort_manifest.json` | The frozen manifest of 100 slots (30 code, 25 math, 20 tool/agentic, 15 Indic, 10 general). Prompt texts are not stored: each slot points at a row of a pinned upstream file and holds the sha256 of its text. |
| `effort_suite.py` | The suite, with the subcommands `verify`, `plan`, `run`, `guard` and `compare`. |
| `test_effort_suite.py` | CPU tests, no model and no network. |
| `stage_a_modal.py`, `stage_b_modal.py`, `stage_b_qwen_modal.py` | Generation only, on one H100 through Modal (OLMoE-1B-7B-0924-Instruct for Stages A and B, Qwen3.6-35B-A3B-FP8 for the Qwen Stage B). |
| `stage_a_report.py`, `stage_b_report.py`, `stage_b_qwen_report.py` | Local scoring of the generations with the checkers in `rl-data/` (math-verify for math, Docker for stdin/stdout code), then the per-tier readout. |

```bash
python -m pytest teams/10-post-training/eval/test_effort_suite.py -q -p no:cacheprovider
```

The results are in `results/`. The models in these runs are open stand-ins, not the capstone model, and every record is marked `do_not_train`.
