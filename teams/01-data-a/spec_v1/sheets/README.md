# Medicine sheets and decisions

| Family | Medicine source | Review / decisions |
|---|---|---|
| Code | [medicine_code.md](medicine_code.md) | [decisions_code.md](decisions_code.md), [reproduction scripts](../../oneoff/README.md) |
| English | [medicine_english.md](medicine_english.md) | [decisions_english.md](decisions_english.md), [validation tools](../validate/README_english.md) |
| Math | [medicine_math.md](medicine_math.md) | [decisions_math.md](decisions_math.md) |
| Science | [medicine_sci.md](medicine_sci.md) | [Conflict resolution and harm report](medicine_sci_clash_resolution_and_harm_report.md) |
| Indic | [medicine_indic.md](medicine_indic.md), [retired cards](medicine_indic_retired.md) | [Integration policy, deferred rules, and validation](../README_indic.md); PR #240 at `3bbd31f` |

[specv1.md](specv1.md) is the reviewed consolidated input consumed by [build_clean.py](../build/build_clean.py). The build reads this consolidated file; it does not automatically collect changed family sheets. Indic integration refreshes its own consolidated cards through the separately documented review process. English source compilation has an outstanding reproduction dependency described in [the English guide](../validate/README_english.md#removed-legacy-files).

The former code/English overview documents were removed during cleanup. Use this index, the family decisions, and [the spec guide](../README.md). Review status and historical harm evidence remain subject to their recorded pack, gold, and code snapshots.
