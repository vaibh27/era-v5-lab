# Contributor registry snapshot

[registry.json](registry.json) stores 125 historical rule IDs, including 112 nonwithdrawn entries. The 24 distinct source files referenced by active entries are absent from this checkout. [registry.html](registry.html) embeds source from the same ID inventory and includes 13 error records; retain it as historical review evidence.

Current cleaner builds use the [reviewed medicine sheets](../spec_v1/sheets/README.md) and [build_clean.py](../spec_v1/build/build_clean.py). They do not load this registry. Shared review tools can still use the registry after their source inputs are staged.

## Reconstruct a review workspace

1. Stage the intended contributor `medicine.md` files, using [fetch_medicines.py](../tools/fetch_medicines.py) if needed. Record which checkout/PR snapshot supplied them; current submissions need not reproduce the historical inventory.
2. Use [medicine_extract.py](../harmtest/medicine_extract.py) in a review checkout to generate `rules/<pack_id>/<handle>.py` and its registry. See the [review guide](../harmtest/README.md) for options.
3. Run registry-based comparisons only when the registry and extracted source files match. `harm_test.py --rules <sheet>` supports direct sheet testing without this reconstruction.
4. Generate a viewer only after source reconstruction, or supply `registry_view.py --out <new-file.html>` to inspect a staged inventory. Regenerating the committed viewer from the incomplete source tree would discard embedded historical source.

[run_harm_suite.py](../harmtest/run_harm_suite.py) additionally requires a legacy nonexecutable inventory via `--spec`. The old default `spec/spec_v1.md` and its builder are absent. The current executable consolidated sheet cannot substitute for that format.
