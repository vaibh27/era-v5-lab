"""Select SWE-bench-format rows from an open pool (default nebius/SWE-rebench) that pass the G10 rules.

This scales the environment supply for the week 3-4 RL loop. It writes a candidate manifest only; a row
becomes an environment only after `make_env.py` rebuilds it from the upstream PR and `verify.py --check`
passes it, exactly like the hand-picked ones.

Why SWE-rebench content passes the 3 Nov rule (arXiv 2505.20411, sections 2.1 to 2.4): the problem statement
is the GitHub issue, and patch and test_patch come from the merged PR, all from permissively licensed repos.
Qwen2.5-72B-Instruct only wrote install recipes and quality labels (metadata). We rebuild every environment
ourselves and do not use those recipes or labels.

TSAI answer (4 Nov): allowed, using only the human parts (issue, patch, tests), with every env rebuilt and
gold-checked by us, and the CC-BY-4.0 attribution kept in licences/licence_table.md. Pre-2023 rows are fine.

Filters (keep_row):
  - created_at (SWE-rebench: the PR creation date) is a valid date before 2023-01-01; a missing or unreadable
    date is rejected. make_env.py then checks the linked issue's own date from GitHub.
  - repo not in benchmark_repos.txt (SWE-bench, Multimodal, Live, Pro)
  - licence named and permissive (MIT, BSD, Apache-2.0, ISC, ...)
  - FAIL_TO_PASS and PASS_TO_PASS non-empty
  - no AI-tool marker in the problem statement

Pinning: the dataset is read at one Hub commit (`--revision`, default PINNED_REVISION), never at "main". Each
parquet file of the split is checked against the sha256 the Hub records for it (its LFS oid) before it is read,
and the manifest records the revision, split and file hashes. `--local-dir` reads files already downloaded at
that revision (same check). The output is a discovery list: a row is admitted only by make_env.py + verify.py.

Usage:
    python import_pool.py --out pool_manifest.json [--dataset nebius/SWE-rebench] [--revision SHA] [--split test]
                          [--local-dir DIR] [--max-patch-chars 3000] [--per-repo 3]
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import re
import sys
import urllib.request
from collections import Counter
from datetime import date, datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
CUTOFF = "2023-01-01"
# nebius/SWE-rebench at the commit read for the 11 Nov 2026 pilot (Hub lastModified 2025-12-23).
PINNED_REVISION = "89cdfbab4ab1bd8f5a658bb212d1b63624f4f881"
AI_MARKERS = re.compile(
    r"\b(claude|anthropic|copilot|codex|chatgpt|gpt-[345]|openai|gemini|devin|llm)\b|cursor[ .](ai|agent|com|sh)\b|"
    r"generated with|ai[- ]assisted", re.I)

# SPDX ids we accept for the upstream repo licence (permissive, commercial use allowed).
PERMISSIVE_SPDX = {"MIT", "MIT-CMU", "BSD-2-Clause", "BSD-3-Clause", "BSD", "Apache-2.0", "ISC", "0BSD", "Zlib",
                   "Unlicense", "CC0-1.0", "PSF-2.0", "BSL-1.0", "MIT OR Apache-2.0"}
# Licence names as SWE-rebench writes them -> SPDX.
NAME_TO_SPDX = {
    "mit license": "MIT", "mit": "MIT", "mit-cmu license": "MIT-CMU", "apache license 2.0": "Apache-2.0",
    "apache software license": "Apache-2.0", "apache-2.0": "Apache-2.0",
    'bsd 3-clause "new" or "revised" license': "BSD-3-Clause", "bsd 3-clause license": "BSD-3-Clause",
    "bsd-3-clause": "BSD-3-Clause", "bsd 3-clause": "BSD-3-Clause", "new bsd license": "BSD-3-Clause",
    "modified bsd license": "BSD-3-Clause", 'bsd 2-clause "simplified" license': "BSD-2-Clause",
    "bsd-2-clause": "BSD-2-Clause", "bsd license": "BSD", "bsd": "BSD", "isc license": "ISC",
    "mit/apache-2.0 dual license": "MIT OR Apache-2.0", "the unlicense": "Unlicense",
    "creative commons zero v1.0 universal": "CC0-1.0",
    "creative commons zero v1.0 universal license (cc0 1.0)": "CC0-1.0",
}


def benchmark_repos(path: Path = HERE / "benchmark_repos.txt") -> set[str]:
    return {line.strip().lower() for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.startswith("#")}


def spdx(name: str | None) -> str | None:
    return NAME_TO_SPDX.get((name or "").strip().lower())


def parse_date(value) -> date | None:
    """A date from a datetime, a date or an ISO string ("2021-05-01", "2021-05-01 13:21:50", "...Z"). None if
    the value is missing or unreadable, so the caller can reject the row instead of letting it pass."""
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value or "").strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}", text):
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError:
            return None


def keep_row(row: dict, bench: set[str]) -> tuple[bool, str]:
    created = parse_date(row.get("created_at"))
    if created is None:
        return False, f"missing or invalid created_at: {row.get('created_at')!r}"
    if created.isoformat() >= CUTOFF:
        return False, "created 2023 or later"
    if row["repo"].lower() in bench:
        return False, "repo is in a SWE benchmark"
    lic = spdx(row.get("license_name"))
    if lic not in PERMISSIVE_SPDX:
        return False, f"licence not accepted: {row.get('license_name')!r}"
    if not row.get("FAIL_TO_PASS") or not row.get("PASS_TO_PASS"):
        return False, "empty FAIL_TO_PASS or PASS_TO_PASS"
    if AI_MARKERS.search(row.get("problem_statement") or ""):
        return False, "AI-tool marker in the issue"
    return True, ""


def split_files(dataset: str, revision: str, split: str) -> list[dict]:
    """Parquet files of one split at one Hub commit, with the sha256 the Hub records for each (LFS oid)."""
    url = f"https://huggingface.co/api/datasets/{dataset}/tree/{revision}/data"
    entries = json.load(urllib.request.urlopen(url))
    files = [{"path": e["path"], "sha256": (e.get("lfs") or {}).get("oid"), "size": e.get("size")}
             for e in entries if e.get("type") == "file" and e["path"].endswith(".parquet")
             and e["path"].rsplit("/", 1)[-1].startswith(f"{split}-")]
    if not files:
        raise SystemExit(f"no parquet files for split {split!r} at {dataset}@{revision}")
    if any(not f["sha256"] for f in files):
        raise SystemExit("a parquet file has no recorded sha256; refusing to read it unpinned")
    return sorted(files, key=lambda f: f["path"])


def read_rows(dataset: str, split: str, columns: list[str], revision: str = PINNED_REVISION,
              local_dir: Path | None = None) -> tuple[list[dict], list[dict]]:
    """Rows of `split` at `revision`, and the file records. Every file must match its recorded sha256."""
    files = split_files(dataset, revision, split)
    rows = []
    for f in files:
        if local_dir is not None:
            data = (local_dir / f["path"].rsplit("/", 1)[-1]).read_bytes()
        else:
            data = urllib.request.urlopen(
                f"https://huggingface.co/datasets/{dataset}/resolve/{revision}/{f['path']}").read()
        got = hashlib.sha256(data).hexdigest()
        if got != f["sha256"]:
            raise SystemExit(f"{f['path']}: sha256 {got} does not match the Hub record {f['sha256']}")
        import pyarrow.parquet as pq  # only needed for the CLI, after the hash check

        rows += pq.read_table(io.BytesIO(data), columns=columns).to_pylist()
    return rows, files


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="nebius/SWE-rebench")
    ap.add_argument("--revision", default=PINNED_REVISION, help="Hub commit of the dataset (default %(default)s)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--local-dir", type=Path, help="read the split's parquet files from here (checked by sha256)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-patch-chars", type=int, default=3000, help="keep small, focused fixes")
    ap.add_argument("--per-repo", type=int, default=3, help="at most this many rows per repo")
    args = ap.parse_args(argv)

    cols = ["instance_id", "repo", "base_commit", "created_at", "license_name", "FAIL_TO_PASS", "PASS_TO_PASS",
            "patch", "problem_statement"]
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise SystemExit("--revision must be a full 40-character commit sha")
    rows, files = read_rows(args.dataset, args.split, cols, args.revision, args.local_dir)
    bench = benchmark_repos()
    reasons: Counter = Counter()
    per_repo: Counter = Counter()
    kept = []
    for row in sorted(rows, key=lambda r: (len(r["patch"] or ""), r["instance_id"])):
        ok, why = keep_row(row, bench)
        if ok and len(row["patch"] or "") > args.max_patch_chars:
            ok, why = False, "patch too large"
        if ok and per_repo[row["repo"]] >= args.per_repo:
            ok, why = False, "per-repo cap"
        if not ok:
            reasons[why.split(":")[0]] += 1
            continue
        per_repo[row["repo"]] += 1
        m = re.search(r"-(\d+)$", row["instance_id"])
        kept.append({"instance_id": row["instance_id"], "repo": row["repo"], "pr": int(m.group(1)) if m else None,
                     "base_commit": row["base_commit"], "created_at": parse_date(row["created_at"]).isoformat(),
                     "licence": spdx(row["license_name"]), "n_fail_to_pass": len(row["FAIL_TO_PASS"]),
                     "n_pass_to_pass": len(row["PASS_TO_PASS"]), "patch_chars": len(row["patch"]),
                     "split": args.split,
                     "source": f"https://huggingface.co/datasets/{args.dataset}/tree/{args.revision}"})
    manifest = {"dataset": args.dataset, "revision": args.revision, "split": args.split, "files": files,
                "rows_read": len(rows), "kept": len(kept),
                "dropped": dict(reasons.most_common()), "cutoff": CUTOFF, "rows": kept}
    args.out.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: manifest[k] for k in ("rows_read", "kept", "dropped")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
