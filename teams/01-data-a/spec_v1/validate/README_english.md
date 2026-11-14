# Scripts of the English family

What checks the English family of `clean.py` on the English packs. Paths start at `teams/01-data-a/`. `clean.py` is the one cleaner: [../README_clean.md](../README_clean.md) says how to build and run it. The shared harm runner is [../../harmtest/harm_test.py](../../harmtest/harm_test.py); a checkout argument may point to this repository. The diagnosis packs and disputed-gold inputs must be staged separately.

| script | makes |
|---|---|
| `spec_v1/validate/verify_clean_english.py` | `results/spec_v1/cleaner/verify_clean_english.json`: `clean.py` gives the same action and text as the English rules of `specv1.md` on every English doc; a second run; ZWJ and ZWNJ. |
| `spec_v1/validate/f1_eval_english.py` | `results/spec_v1/cleaner/f1_eval_english.json`: F1 on the English packs, real hits and made-up values, as `spec_v1/validate/f1_eval.py` checks code. |
| `spec_v1/validate/order_checks_english.py` | `results/spec_v1/cleaner/order_checks_english.json`: pipeline attribution and rule/stage-swap checks on `sheets/medicine_english.md`. |
| `spec_v1/validate/make_harm_tests_english.py` | `results/spec_v1/cleaner/harm_tests_english.md`, and the reports of `clean.py` (`results/spec_v1/cleaner/`) and `medicine_english.md` (`results/spec_v1/sheets/`), official gold and with the disputed docs left in. |

## Run it, in this order

```
python3 -m pip install -r spec_v1/requirements.txt
python -I spec_v1/build/build_clean.py
python -m pytest tests/spec_v1
python -I spec_v1/validate/verify_clean_english.py <folder with the packs>
python -I spec_v1/validate/f1_eval_english.py <g1-harm-test checkout> <folder with the packs>
python -I spec_v1/validate/order_checks_english.py <folder with the packs>
python -I spec_v1/validate/make_harm_tests_english.py <g1-harm-test checkout> <folder with the packs> <gold made with build_gold.py --allow-disputed>
```

`verify_clean_english.py` takes about 4 minutes, `order_checks_english.py` about 6, the others a minute or two. Every output is IDs, counts and digests, never document text. The packs must carry `pack_id`: `clean.py` picks the English rules by it.

## Removed legacy files

The English-only cleaner and its builder, earlier English runners, and the old English overview have been removed. Use `clean.py` and `run_clean.py`, the [medicine sheet](../sheets/medicine_english.md), and [decisions](../sheets/decisions_english.md). The tokenizer-sample job has been implemented; its config and local evidence are indexed in [jobs](../../jobs/README.md). This does not turn historical English validation snapshots into evidence for every later cleaner build.

The sheet itself (`medicine_english.md`), `decisions_english.md` and `results/spec_v1/sheets/evidence_english.json`, `results/spec_v1/sheets/equivalence_english.json` were compiled from the annotators' sheets by scripts that are not in this repo yet.
