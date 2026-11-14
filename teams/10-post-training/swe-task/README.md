# SWE task demonstrations

Small end-to-end software-engineering tasks that show the chat format and a fail, patch, pass cycle. The Hindi, Kannada and Tamil runners save measured evidence (logs, the trajectory and a summary) to an output folder, by default a temporary one. The written-up results are in `results/`.

| Files | What it does |
|---|---|
| `calc.py`, `test_calc.py`, `Dockerfile`, `run_e2e.py` | A toy bug in `calc.py` (`return a + b + 1`), seeded and then patched inside Docker (results: `results/swe_e2e.md`). |
| `hindi_text_prep.py`, `Dockerfile.hindi`, `run_hindi_e2e.py` | Prepare Hindi (Devanagari) text for the tokenizer without changing what it says (results: `results/hindi_swe_e2e.md`). |
| `kannada_text_prep.py`, `kannada_examples.py`, `preview_kannada.py`, `Dockerfile.kannada`, `run_kannada_e2e.py` | Kannada demonstration: marker checking that removes only ZWSP and BOM, reviewed example words, and an offline browser preview of the rendered text (results: `results/kannada_swe_e2e.md`). |
| `tamil_utf8_io.py`, `run_tamil_e2e.py` | Write Tamil and other Unicode text without changing its code points (results: `results/tamil_swe_e2e.md`). |

The Go version is in `swe-task-go/`. The checks for these files are in `tests/`. The real SWE environments are in `swe-envs/`.
