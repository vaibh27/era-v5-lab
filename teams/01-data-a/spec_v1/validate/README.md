# Validate: checks on the sheets and on `clean.py`

Paths and commands start at the repository root. Install [../requirements.txt](../requirements.txt) before running data-dependent tools. The scripts that produce `clean.py` are in [../build](../build/README.md). Report files and their snapshot status are indexed in [../results/README.md](../../results/spec_v1/README.md).

| Purpose | Scripts | Inputs / outputs |
|---|---|---|
| Validate the executable sheet | [validate_specv1.py](validate_specv1.py) | Original packs and own-pack gold → `results/spec_v1/cleaner/specv1_validation.json` |
| Validate generated cleaner behavior and harm | [make_harm_tests.py](make_harm_tests.py) | Checkout, pack folders → per-pack reports and `harm_tests.md` |
| Compare an Indic build with a baseline | [check_indic_integration.py](check_indic_integration.py) | Explicit baseline file and diagnosis/gold packs; see Indic guide |
| Check code equivalence and privacy behavior | [verify_clean.py](verify_clean.py), [f1_eval.py](f1_eval.py) | Code diagnosis packs, source sheet, labels/gold; IDs and counts only |
| Run English equivalence, order, privacy, and stress checks | [verify_clean_english.py](verify_clean_english.py), [order_checks_english.py](order_checks_english.py), [f1_eval_english.py](f1_eval_english.py), [make_harm_tests_english.py](make_harm_tests_english.py) | [English commands and order](README_english.md) |
| Validate optional F2 on all supported packs | [validate_f2.py](validate_f2.py) | [v2 migration](../../results/spec_v1/f2/README.md#current-v2-pin-and-validation); full pinned set, per-pack raw and clean-then-F2 gold checks, missing inputs fail |
| Validate Indic F1 and optional F2 | [harm_f1_indic.py](harm_f1_indic.py), [validate_f2_indic.py](validate_f2_indic.py) | [Indic commands](../README_indic.md#optional-f2-validation); current own-pack gold, full pinned G9 set for F2 |

The [sheet-reproduction scripts](../../oneoff/README.md) have a separate role: reconstructing contributor evidence and explaining selection decisions.
