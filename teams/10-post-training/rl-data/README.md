# RL data v1: tasks the model can learn from

GRPO learns only from groups whose rewards differ (see `rl-loop/grpo.py`). A task the starting model never solves, or always solves, gives no signal. So RL data v1 is built in three steps:
1. take human-written problems with tests;
2. add an easier rung for a very small model;
3. keep only the tasks the starting model sometimes solves.

```bash
# 1. stdio tasks from a local copy of deepmind/code_contests (no network; TSAI publishes approved sets as packs)
python teams/10-post-training/rl-data/build.py --source code_contests --parquet <train-*.parquet> --out out/tasks.jsonl

# 2. the easy rung, from templates written by people on the team (empty until TSAI answers; see easy_rung.py)
python teams/10-post-training/rl-data/easy_rung.py --templates teams/10-post-training/rl-data/templates/easy_rung_templates.yaml --out out/easy.jsonl

# 3. keep tasks with 0 < pass rate < 1 for a given starting model (any OpenAI-compatible endpoint, e.g. llama-server)
python teams/10-post-training/rl-data/filter.py --tasks out/tasks.jsonl --server http://127.0.0.1:8080 --k 8 --out out/manifest.json

python -m pytest teams/10-post-training/rl-data/test_rl_data.py -q -p no:cacheprovider
```

## Files

| File | What |
|---|---|
| `stdio_check.py` | Reward for stdin/stdout tasks, with the same rule as `rl/verifier.py`: the sandbox gets the program and the inputs only, and the outputs are compared here. A program can't fake a pass, because it never sees the right answers. Fresh `python:3.12-slim` container, no network, the verifier's limits. |
| `build.py` | `deepmind/code_contests` (Codeforces rows, rating band, up to 20 tests) to task records. Each record keeps source, licence, generator and the licence-table set id. |
| `filter.py` | Samples k answers per task from the starting model, scores them, and writes a manifest of ids, pass rates and keep / too_hard / too_easy. |
| `easy_rung.py` | Program-generated tasks whose answers come from running a human reference solution. **The templates (statement and reference solution) must be written by a person, never by a model**, because they become training text. `templates/easy_rung_templates.yaml` is empty on purpose. |
| `results/rl_data_filter_olmoe.md` | The filter run on 36 Codeforces problems for OLMoE-1B-7B-0125-Instruct. |

## Rules this follows

- **Data rule (TSAI, 3 Nov):** human-written problems and tests only. Every task records its generator and licence. No Claude-written task text: the 10 tasks in `rl/tasks` say "written by Claude" in their `task.json`, so they are used for evaluation only until TSAI says otherwise.
- **Licences:** `deepmind/code_contests` is CC-BY-4.0 and `TSAI reserves` in `licences/licence_table.md` (`rl-code-code-contests`). TSAI picks the RL sets.
- **No data in the repo:** task records and manifests go to `out/`. Only the summary in `results/` is committed: ids and numbers, no problem text.
- **Decontamination:** run the task statements through the decontamination runner from #34 (fingerprints v2, full set) before any training. Codeforces problems overlap with code benchmarks.
