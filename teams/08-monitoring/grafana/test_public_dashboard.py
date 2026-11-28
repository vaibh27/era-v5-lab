"""Validation tests for the read-only public Grafana dashboard (`grafana/public_v1.json`)."""
from __future__ import annotations

import json
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PUBLIC_JSON = HERE / "public_v1.json"

FORBIDDEN_INTERNAL_METRICS = (
    "gpu_power",
    "gpu_temp_c",
    "gpu_thermal_throttle",
    "gpu_memory_used",
    "gpu_alloc_gib",
    "loader_wait_time",
    "node_health",
    "gpu_telemetry_1s",
)


def _load():
    return json.loads(PUBLIC_JSON.read_text(encoding="utf-8"))


def test_public_dashboard_is_readonly_and_distinct_uid():
    d = _load()
    assert d["editable"] is False, "public dashboard must be strictly read-only (editable: false)"
    assert d["uid"] == "era-v5-public-v1"
    assert "public" in d.get("tags", [])


def test_public_dashboard_panel_ids_unique_and_non_overlapping_queries_safe():
    d = _load()
    panels = d["panels"]
    ids = [p["id"] for p in panels]
    assert len(ids) == len(set(ids)), f"duplicate panel IDs: {ids}"

    chart_panels = [p for p in panels if p["type"] != "row"]
    assert len(chart_panels) >= 6

    for p in chart_panels:
        assert p["type"] in ("trend", "table")
        if p["type"] == "trend":
            assert p.get("options", {}).get("xField") == "step"
        targets = p.get("targets", [])
        assert targets, f"panel {p['id']} has no SQL targets"
        for t in targets:
            sql = t.get("rawSql", "")
            assert sql.strip(), f"panel {p['id']} has empty rawSql"
            for forbidden in FORBIDDEN_INTERNAL_METRICS:
                assert forbidden not in sql, f"panel {p['id']} leaks internal metric/table {forbidden}"


def test_public_descriptions_name_teams_only_no_handles():
    d = _load()
    texts = [d.get("title", ""), d.get("description", "")]
    for p in d["panels"]:
        texts.append(p.get("title", ""))
        texts.append(p.get("description", ""))
    blob = "\n".join(texts)
    assert re.search(r"@[A-Za-z0-9_-]+", blob) is None, "public dashboard text must not contain @handles"


# Team 11 Release's approved sentences, one per public panel (TSAI approved in #598).
DASHBOARD_TEXT = HERE.parents[1] / "11-release" / "dashboard" / "dashboard_text.md"


def _approved_sentences():
    rows = re.findall(r"^\|\s*(.+?)\s*\|\s*(This panel .+?)\s*\|\s*$", DASHBOARD_TEXT.read_text(encoding="utf-8"), re.M)
    return dict(rows)


def test_panel_descriptions_are_team11_approved_sentences():
    approved = _approved_sentences()
    charts = [p for p in _load()["panels"] if p["type"] != "row"]
    assert {p["title"] for p in charts} == set(approved), "public panel titles must match dashboard_text.md"
    for p in charts:
        assert p["description"] == approved[p["title"]], f"panel {p['id']} text differs from dashboard_text.md"


def test_no_placeholder_text_or_dashes():
    d = _load()
    texts = [d.get("title", ""), d.get("description", "")]
    texts += [t for p in d["panels"] for t in (p.get("title", ""), p.get("description", ""))]
    for t in texts:
        assert "to be confirmed" not in t.lower(), t
        assert "—" not in t and "–" not in t, f"no em or en dash in public text: {t!r}"


def test_times_shown_in_ist():
    assert _load()["timezone"] == "Asia/Kolkata"


def test_training_pace_queries_cover_harness_and_public_names():
    panels = {p["title"]: p for p in _load()["panels"]}
    for title, names, unit in (
        ("Training Pace: Tokens per Second", ("tokens_per_second", "tokens_per_s"), "short"),
        ("Training Pace: Step Time", ("perf/step_ms", "step_time_ms"), "ms"),
    ):
        panel = panels[title]
        sql = panel["targets"][0]["rawSql"]
        assert all("'" + name + "'" in sql for name in names)
        assert "run_id IN ($run)" in sql and "value IS NOT NULL" in sql
        assert panel["fieldConfig"]["defaults"]["unit"] == unit
        step = next(o for o in panel["fieldConfig"]["overrides"] if o["matcher"]["options"] == "step")
        props = {p["id"]: p["value"] for p in step["properties"]}
        assert props == {"unit": "short", "custom.axisLabel": "step"}


def test_composition_panel_displays_version_and_cleaning_counts():
    panel = next(p for p in _load()["panels"] if p["title"] == "Dataset and Shard Composition")
    assert panel["type"] == "table"
    sql = panel["targets"][0]["rawSql"]
    for field in ("dataset_version", "stage", "pack_id", "domain", "language", "docs_in", "docs_after_clean", "token_count", "drop_rate_pct"):
        assert field in sql
    assert "shard_composition FINAL" in sql


def test_composition_version_selector_and_section():
    dashboard = _load()
    panel = next(p for p in dashboard["panels"] if p["id"] == 31)
    assert "WHERE dataset_version = ${dataset_version:singlequote}" in panel["targets"][0]["rawSql"]
    var = next(v for v in dashboard["templating"]["list"] if v["name"] == "dataset_version")
    assert var["type"] == "query" and var["datasource"]["uid"] == "${ds}"
    assert var["query"] == "SELECT DISTINCT dataset_version FROM shard_composition ORDER BY dataset_version"
    assert not var["multi"] and not var["includeAll"]
    row = next(p for p in dashboard["panels"] if p["title"] == "Dataset Composition")
    assert row["type"] == "row" and row["gridPos"] == {"h": 1, "w": 24, "x": 0, "y": 43}
    assert panel["gridPos"]["y"] == 44


def test_composition_display_labels_and_units():
    panel = next(p for p in _load()["panels"] if p["id"] == 31)
    expected = {
        "dataset_version": "Dataset version", "stage": "Stage", "pack_id": "Pack",
        "domain": "Subject", "language": "Language code", "docs_in": "Documents in",
        "docs_after_clean": "Documents after cleaning", "token_count": "Tokens",
        "drop_rate_pct": "Drop rate (cleaning)",
    }
    overrides = {o["matcher"]["options"]: {p["id"]: p["value"] for p in o["properties"]}
                 for o in panel["fieldConfig"]["overrides"] if o["matcher"]["id"] == "byName"}
    assert {field: config["displayName"] for field, config in overrides.items()} == expected
    for field in ("docs_in", "docs_after_clean", "token_count"):
        assert overrides[field]["unit"] == "short" and overrides[field]["decimals"] == 1
    assert overrides["drop_rate_pct"]["unit"] == "percent"
    assert overrides["drop_rate_pct"]["decimals"] == 2
    assert panel["gridPos"]["h"] == 16


# Public-page rules from docs/dashboard_content_decisions_w4.md section 1: no compute-cost counter, no hardware or
# machine names or specs, numbers from stored tables only, live refresh.
ALLOWED_TABLES = frozenset({"metrics_raw", "eval_mini", "shard_composition", "run_events"})
ROLLUP_TABLE = re.compile(r"^metrics_rollup_(\d+|\$\{bucket\}|\$bucket)$")
# Never in a public query: cost and spend, MFU (implies peak FLOPS, so hardware), GPU and node telemetry, hosts.
FORBIDDEN_SQL_TERMS = (
    "cost", "spend", "price", "usd", "currency", "gpu_hours", "gpu", "mfu", "model_flops_utilization", "hfu",
    "power", "temp_c", "clock", "xid", "ecc", "host", "node_",
)
# Never in public text or queries: accelerator models and vendors, clouds and hosting, machine specs, money.
HARDWARE_OR_COST = re.compile(
    r"\b(h100|h200|h800|a100|a10g|a6000|b200|gb200|l4|l40s?|t4|v100|rtx\s?\d+|nvidia|amd|mi300x?|tpu|cuda|"
    r"runpod|modal|kaggle|colab|aws|gcp|azure|lambda labs|vram|tflops|petaflops|gpu[- ]hours?|hostnames?|"
    r"dollars?|rupees?|usd|inr)\b|[$₹€]\s?\d",
    re.I,
)


def _public_texts(d):
    """Every piece of text a public viewer can read: titles, descriptions, text-panel content, variable labels."""
    texts = [d.get("title", ""), d.get("description", "")]
    for p in d["panels"]:
        texts += [p.get("title", ""), p.get("description", ""), (p.get("options") or {}).get("content", "")]
    texts += [v.get("label") or "" for v in d.get("templating", {}).get("list", [])]
    return [t for t in texts if t]


def _public_sql(d):
    sql = [t.get("rawSql", "") for p in d["panels"] for t in p.get("targets", [])]
    sql += [v.get("query") or "" for v in d.get("templating", {}).get("list", []) if v.get("type") == "query"]
    return [q for q in sql if isinstance(q, str) and q.strip()]


def test_public_queries_read_only_allowed_tables():
    for sql in _public_sql(_load()):
        for table in re.findall(r"\b(?:FROM|JOIN)\s+([A-Za-z_][\w${}]*)", sql, re.I):
            assert table in ALLOWED_TABLES or ROLLUP_TABLE.match(table), f"public query reads {table!r}: {sql[:120]}"


def test_public_queries_have_no_cost_or_hardware_terms():
    for sql in _public_sql(_load()):
        low = sql.lower()
        for term in FORBIDDEN_SQL_TERMS:
            assert term not in low, f"public query uses {term!r}: {sql[:120]}"


def test_no_compute_cost_counter():
    d = _load()
    for p in d["panels"]:
        fc = p.get("fieldConfig") or {}
        units = [(fc.get("defaults") or {}).get("unit")]
        units += [pr.get("value") for o in fc.get("overrides", []) for pr in o.get("properties", []) if pr.get("id") == "unit"]
        for unit in units:
            assert not str(unit or "").lower().startswith("currency"), f"panel {p['id']} shows money ({unit})"


def test_no_hardware_or_machine_names_or_specs():
    d = _load()
    for text in _public_texts(d) + _public_sql(d):
        m = HARDWARE_OR_COST.search(text)
        assert m is None, f"public dashboard names hardware, a machine or money ({m.group(0)!r}): {text[:120]}"


def test_public_refresh_is_live():
    m = re.fullmatch(r"(\d+)(s|m)", _load().get("refresh") or "")
    assert m, "public dashboard must auto-refresh during a live run"
    seconds = int(m.group(1)) * (60 if m.group(2) == "m" else 1)
    assert 0 < seconds <= 30, f"refresh {seconds}s is too slow for the 60 s metric-to-dashboard target"


def test_rule_checks_catch_violations():
    """The checks above must fail on the things they are for, not only pass on today's dashboard."""
    assert HARDWARE_OR_COST.search("Trained on 8x H100 nodes")
    assert HARDWARE_OR_COST.search("Spend so far: $1200")
    assert HARDWARE_OR_COST.search("GPU-hours used")
    # Team 11's honest prose may name cost and machines as things the page does NOT show.
    assert not HARDWARE_OR_COST.search("The dashboard does not show the cost of the run or the machines that it runs on.")
    assert not HARDWARE_OR_COST.search("SELECT step, run_id, value FROM metrics_rollup_${bucket} WHERE run_id IN ($run)")
    assert ROLLUP_TABLE.match("metrics_rollup_${bucket}") and not ROLLUP_TABLE.match("gpu_telemetry_1s")
