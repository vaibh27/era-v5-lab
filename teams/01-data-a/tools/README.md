# Shared Data A tools

These tools build gold, fetch contributor medicines, and inspect diagnosis packs. The harm-test harness (`harm_test.py`, `medicine_extract.py`, `rule_report.py`, `medicine_conflicts.py`, `run_harm_suite.py`, `registry_view.py`) moved to [harmtest](../harmtest/README.md); `tools/harm_test.py` is a forwarder kept until Week 4. The cleaner's build and validation scripts are indexed in [spec_v1/build](../spec_v1/build/README.md) and [spec_v1/validate](../spec_v1/validate/README.md).

| Task | Scripts | Guide / inputs |
|---|---|---|
| Build gold sets and review their removal history | [build_gold.py](build_gold.py), [gold_notes.py](gold_notes.py) | [Gold guide](README_build_gold.md); [gold inputs](../gold/) |
| Collect and extract contributor medicines | [fetch_medicines.py](fetch_medicines.py), [medicine_extract.py](../harmtest/medicine_extract.py) | [Medicine review guide](../harmtest/README.md); local diagnosis submissions, optionally open PRs |
| Test a medicine sheet or rule | [harm_test.py](../harmtest/harm_test.py) | Original diagnosis pack and its own gold; use `--rules` for current sheets |
| Compare candidate rules and conflicts | [rule_report.py](../harmtest/rule_report.py), [medicine_conflicts.py](../harmtest/medicine_conflicts.py) | Staged contributor inputs; registry mode also needs extracted source files |
| Inspect a registry | [registry_view.py](../harmtest/registry_view.py) | [Registry snapshot and reconstruction limits](../rules/README.md) |
| Audit the legacy registry inventory | [run_harm_suite.py](../harmtest/run_harm_suite.py) | Legacy inventory supplied with `--spec`; current medicine sheets have a different format |
| Probe a regex on labelled documents | [regex_check.py](regex_check.py) | Pack, gold, labels, disease code, and regex |
| View a pack and audit Unicode | [view_pack.py](view_pack.py) | Local Parquet pack; run with `--help` for options |

Tests for these utilities are in [../tests/tools](../tests/tools/). From the repository root, with [pinned dependencies](../spec_v1/requirements.txt) installed:

```bash
python -m pytest teams/01-data-a/tests/tools -q
python -I teams/01-data-a/tests/tools/test_view_pack.py
```

The viewer unittest suite uses made-up rows and mocked loading, so its direct command needs only the Python standard library. Real-pack harm tests require separately staged data. Fetching current open PRs is optional and is not part of local cleanup or unit testing.
