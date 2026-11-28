#!/usr/bin/env python3
"""Convert one Team 01 pipeline run's ledger, shard summary and pack metadata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .load_shard_composition import insert_rows, normalize_row
except ImportError:  # Direct script execution.
    from load_shard_composition import insert_rows, normalize_row


def adapt(ledger: dict, shards: dict, metadata: dict, dataset_version: str,
          reconcile: dict | None = None) -> list[dict]:
    if shards.get("problems"):
        raise ValueError("shard summary reports problems")
    if reconcile is not None and reconcile.get("ok") is not True:
        raise ValueError("reconcile check did not pass")
    if not dataset_version:
        raise ValueError("an explicit dataset version is required")
    packs = metadata.get("packs", metadata)
    by_pack = ledger["by_pack"]
    totals = {}
    for path, part in shards["parts"].items():
        split = path.split("/", 1)[0]
        if split != "train":
            continue  # held-out data is not sharded and has no training token count
        pack = part["pack_id"]
        if part["status"] not in ("written", "empty", "all_left_out"):
            raise ValueError(f"unexpected train shard status for {path}")
        if part.get("truncated"):
            raise ValueError(f"truncated shard part: {path}")
        if part["status"] == "written" and ("tokens" not in part or "docs" not in part):
            raise ValueError(f"missing shard totals: {path}")
        totals.setdefault(pack, [0, 0])
        totals[pack][0] += int(part.get("tokens", 0))
        totals[pack][1] += int(part.get("docs", 0))
    if reconcile is not None:
        ledger_totals = ledger.get("totals", {})
        if (reconcile.get("ledger_rows") != ledger_totals.get("rows")
                or reconcile.get("kept") != ledger_totals.get("kept")
                or reconcile.get("in_shards") != sum(t[1] for t in totals.values())
                or "rows" not in ledger_totals or "kept" not in ledger_totals):
            raise ValueError("reconcile summary does not match ledger/shard summary")
    expected = {p for p, splits in by_pack.items() if "train" in splits}
    if set(totals) != expected:
        raise ValueError("train packs differ between ledger and shard summary")
    rows = []
    for pack in sorted(expected):
        counts = by_pack[pack]["train"]
        info = packs[pack]
        raw, cleaned, dropped = (int(counts[k]) for k in ("rows", "rows_cleaned", "dropped_by_clean"))
        if raw <= 0 or cleaned != raw - dropped:
            raise ValueError(f"invalid cleaning counts for {pack}")
        if totals[pack][0] < 0 or totals[pack][1] < 0 or totals[pack][1] > int(counts["kept"]):
            raise ValueError(f"invalid shard totals for {pack}")
        if not info.get("domain") or not info.get("language"):
            raise ValueError(f"missing pack metadata for {pack}")
        rows.append(normalize_row({"pack_id": pack, "domain": info["domain"],
            "language": info["language"], "stage": "train", "docs_in": raw,
            "docs_kept": cleaned, "token_count": totals[pack][0]}, dataset_version))
    if not rows:
        raise ValueError("no training packs found")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True, type=Path)
    ap.add_argument("--shards", required=True, type=Path)
    ap.add_argument("--metadata", required=True, type=Path)
    ap.add_argument("--reconcile", type=Path)
    ap.add_argument("--dataset-version", required=True, help="Public label; explicitly identify rehearsals")
    ap.add_argument("--url", default="http://127.0.0.1:8123")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    read = lambda p: json.loads(p.read_text(encoding="utf-8"))
    rows = adapt(read(args.ledger), read(args.shards), read(args.metadata), args.dataset_version,
                 read(args.reconcile) if args.reconcile else None)
    if args.dry_run:
        for row in rows:
            print(json.dumps(row, sort_keys=True))
    else:
        print(f"inserted {insert_rows(rows, args.url)} shard_composition rows")


if __name__ == "__main__":
    main()
