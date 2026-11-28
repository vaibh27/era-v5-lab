"""Unit tests for ClickHouse `shard_composition` schema and loader."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from clickhouse.load_shard_composition import SCHEMA_SQL, normalize_row, parse_file


def test_schema_sql_defines_replacing_merge_tree_without_forbidden_columns():
    ddl = SCHEMA_SQL.read_text(encoding="utf-8")
    ddl_body = "\n".join(ln for ln in ddl.splitlines() if not ln.strip().startswith("--"))
    assert "CREATE TABLE IF NOT EXISTS shard_composition" in ddl_body
    assert "ReplacingMergeTree(loaded_at)" in ddl_body
    assert "ORDER BY (dataset_version, stage, pack_id)" in ddl_body
    for col in ("pack_id", "domain", "language", "stage", "docs_in", "docs_kept", "token_count", "drop_rate_pct"):
        assert col in ddl_body
    for forbidden in ("cost", "usd", "gpu_model", "hostname"):
        assert forbidden not in ddl_body.lower()


def test_normalize_row_computes_drop_rate_from_docs():
    row = normalize_row(
        {
            "pack_id": "ind-te-fw2",
            "domain": "web",
            "language": "te",
            "stage": "train",
            "docs_in": 1000,
            "docs_kept": 712,
            "token_count": 450000,
        },
        default_version="corpus-v1",
    )
    assert row == {
        "dataset_version": "corpus-v1",
        "stage": "train",
        "pack_id": "ind-te-fw2",
        "domain": "web",
        "language": "te",
        "docs_in": 1000,
        "docs_kept": 712,
        "token_count": 450000,
        "drop_rate_pct": 28.8,
    }


def test_normalize_row_converts_fractional_drop_rate():
    row = normalize_row(
        {
            "pack_id": "en-wiki",
            "domain": "encyclopedia",
            "language": "en",
            "token_count": 1200000,
            "drop_rate": 0.125,
        },
        default_version="corpus-v1",
    )
    assert row["drop_rate_pct"] == 12.5


def test_normalize_row_rejects_forbidden_hardware_or_cost_fields():
    with pytest.raises(ValueError, match="forbidden public-dashboard fields"):
        normalize_row(
            {"pack_id": "en-wiki", "token_count": 100, "gpu_model": "H200"},
            default_version="corpus-v1",
        )
    with pytest.raises(ValueError, match="forbidden public-dashboard fields"):
        normalize_row(
            {"pack_id": "en-wiki", "token_count": 100, "cost": 42.0},
            default_version="corpus-v1",
        )


def test_normalize_row_rejects_invalid_counts_and_rates():
    with pytest.raises(ValueError, match="docs_kept"):
        normalize_row(
            {"pack_id": "p1", "docs_in": 10, "docs_kept": 12},
            default_version="corpus-v1",
        )
    with pytest.raises(ValueError, match="drop_rate_pct out of"):
        normalize_row(
            {"pack_id": "p1", "drop_rate_pct": 105.0},
            default_version="corpus-v1",
        )


def test_parse_json_packs_manifest_and_csv(tmp_path: Path):
    jpath = tmp_path / "manifest.json"
    jpath.write_text(
        json.dumps(
            {
                "dataset_version": "corpus-v1",
                "packs": [
                    {
                        "pack_id": "ind-te-fw2",
                        "group": "indic-web",
                        "lang_label": "te",
                        "docs_in": 500,
                        "docs_kept": 356,
                        "tokens": 210000,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    jrows = parse_file(jpath)
    assert len(jrows) == 1
    assert jrows[0]["dataset_version"] == "corpus-v1"
    assert jrows[0]["domain"] == "indic-web"
    assert jrows[0]["language"] == "te"
    assert jrows[0]["drop_rate_pct"] == 28.8

    cpath = tmp_path / "summary.csv"
    cpath.write_text(
        "pack_id,domain,language,stage,docs_in,docs_kept,token_count\n"
        "code-python,code,en,stage1,200,180,95000\n",
        encoding="utf-8",
    )
    crows = parse_file(cpath, dataset_version="corpus-v1")
    assert len(crows) == 1
    assert crows[0]["stage"] == "stage1"
    assert crows[0]["drop_rate_pct"] == 10.0


def test_team1_rehearsal_adapter_matches_recorded_totals():
    from clickhouse.adapt_team1_composition import adapt
    root = Path(__file__).resolve().parents[3] / "01-data-a" / "jobs" / "logs" / "2026-11-19-first-training-rehearsal-38packs"
    def read(path):
        return json.loads(path.read_text(encoding="utf-8"))
    summary = root / "inventory-run" / "summaries"
    rows = adapt(read(summary / "ledger_summary.json"), read(summary / "shards_summary.json"),
                 read(root / "config" / "pack_table_job_shaped.json"), "first-training-v1-rehearsal",
                 read(summary / "reconcile_summary.json"))
    reconciliation = read(summary / "reconcile_summary.json")
    assert (reconciliation["ledger_rows"], reconciliation["kept"], reconciliation["in_shards"]) == (1444766, 1407358, 1385709)
    reconciliation["in_shards"] += 1
    with pytest.raises(ValueError, match="reconcile summary does not match ledger/shard summary"):
        adapt(read(summary / "ledger_summary.json"), read(summary / "shards_summary.json"),
              read(root / "config" / "pack_table_job_shaped.json"), "first-training-v1-rehearsal", reconciliation)
    assert len(rows) == 38
    assert sum(r["token_count"] for r in rows) == 1132509915
    assert all(r["stage"] == "train" and r["dataset_version"] == "first-training-v1-rehearsal" for r in rows)
    go = next(r for r in rows if r["pack_id"] == "code-go")
    assert (go["docs_in"], go["docs_kept"], go["drop_rate_pct"]) == (83267, 76955, 7.5804)


def test_team1_adapter_rejects_incomplete_or_inconsistent_inputs():
    from clickhouse.adapt_team1_composition import adapt
    ledger = {"by_pack": {"p": {"train": {"rows": 10, "rows_cleaned": 8, "dropped_by_clean": 2, "kept": 7}}}}
    shards = {"parts": {"train/p/part": {"pack_id": "p", "status": "written", "tokens": 42, "docs": 7}}}
    metadata = {"packs": {"p": {"domain": "code", "language": "en"}}}
    assert adapt(ledger, shards, metadata, "rehearsal")[0]["drop_rate_pct"] == 20
    del shards["parts"]["train/p/part"]["tokens"]
    with pytest.raises(ValueError, match="missing shard totals"):
        adapt(ledger, shards, metadata, "rehearsal")


def test_team1_adapter_rejects_failed_checks():
    from clickhouse.adapt_team1_composition import adapt
    with pytest.raises(ValueError, match="shard summary reports problems"):
        adapt({}, {"problems": ["x"]}, {}, "rehearsal")
    with pytest.raises(ValueError, match="reconcile check did not pass"):
        adapt({}, {}, {}, "rehearsal", {"ok": False})
