"""Turn a local copy of a human-written contest set into stdio task records (JSONL), with source, licence and
generator recorded per task. No network: point it at a parquet file that TSAI (or you) downloaded.

    python build.py --source code_contests --parquet <path/to/train-*.parquet> --out out/tasks.jsonl
                    [--min-rating 800 --max-rating 1200 --max-tests 20]

Sources (the generator rule, TSAI 3 Nov: human-written data is in; record the generator for every set):
    code_contests   deepmind/code_contests, CC-BY-4.0. Problems and solutions are human (Codeforces, AtCoder,
                    CodeChef, Aizu, HackerEarth); extra tests are programmatic mutations of existing inputs.
                    Licence table: `rl-code-code-contests`, TSAI reserves. By default only Codeforces rows
                    (source == 2) are kept, so each task has a rating. `--sources aizu,atcoder,codechef,hackerearth`
                    keeps the other judges instead (no rating filter; Codeforces is a Team 9 benchmark source, so
                    the non-Codeforces rows are the safer pool; still run the decontamination check on them).

Which sets may be used for training is TSAI's call ("TSAI reserves" in licences/licence_table.md). This script
only prepares the records; the records themselves are data and stay out of the repository (out/ is local).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

SOURCES = {
    "code_contests": {"dataset": "deepmind/code_contests", "licence": "CC-BY-4.0",
                      "generator": "none (human problems and tests; extra tests are programmatic mutations)",
                      "licence_table_set": "rl-code-code-contests"},
}


# deepmind/code_contests `source` codes (dataset card)
JUDGES = {"unknown": 0, "codechef": 1, "codeforces": 2, "hackerearth": 3, "codejam": 4, "atcoder": 5, "aizu": 6}


def other_judge_rows(rows: list[dict], judges: set[int], max_tests: int, seen: set[str] | None = None) -> list[dict]:
    """Non-Codeforces rows: no rating, so the judge's own difficulty label is kept for stratifying."""
    meta = SOURCES["code_contests"]
    names = {v: k for k, v in JUDGES.items()}
    seen = set() if seen is None else seen
    out = []
    for r in rows:
        if r.get("source") not in judges or r.get("input_file") or r.get("output_file") or not r.get("description"):
            continue
        tests = []
        for key in ("public_tests", "private_tests", "generated_tests"):
            t = r.get(key) or {}
            tests += [{"input": i, "output": o} for i, o in zip(t.get("input", []), t.get("output", []))]
        if len(tests) < 3:
            continue
        judge = names[r["source"]]
        base = f"cc-{judge}-" + "".join(c if c.isalnum() else "_" for c in (r.get("name") or "noname"))[:60]
        tid, n = base, 1
        while tid in seen:
            n += 1
            tid = f"{base}-{n}"
        seen.add(tid)
        out.append({"id": tid, "kind": "stdio", "statement": r["description"], "tests": tests[:max_tests],
                    "time_limit_s": 4, "difficulty": r.get("difficulty"), "judge": judge,
                    "source": f"{meta['dataset']} ({judge} {r.get('name')})", "licence": meta["licence"],
                    "generator": meta["generator"], "licence_table_set": meta["licence_table_set"]})
    return out


def code_contests_rows(rows: list[dict], min_rating: int, max_rating: int, max_tests: int) -> list[dict]:
    meta = SOURCES["code_contests"]
    out = []
    for r in rows:
        if r.get("source") != 2 or not (min_rating <= (r.get("cf_rating") or 0) <= max_rating):
            continue
        if r.get("input_file") or r.get("output_file") or not r.get("description"):
            continue  # file-based IO, or no statement
        tests = []
        for key in ("public_tests", "private_tests", "generated_tests"):
            t = r.get(key) or {}
            tests += [{"input": i, "output": o} for i, o in zip(t.get("input", []), t.get("output", []))]
        if len(tests) < 3:
            continue
        cf = f"{r.get('cf_contest_id')}{r.get('cf_index')}"
        out.append({"id": f"cf{cf}", "statement": r["description"], "tests": tests[:max_tests],
                    "time_limit_s": 4, "rating": r["cf_rating"], "source": f"{meta['dataset']} (Codeforces {cf})",
                    "licence": meta["licence"], "generator": meta["generator"],
                    "licence_table_set": meta["licence_table_set"]})
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", choices=SOURCES, required=True)
    ap.add_argument("--parquet", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-rating", type=int, default=800)
    ap.add_argument("--max-rating", type=int, default=1200)
    ap.add_argument("--max-tests", type=int, default=20)
    ap.add_argument("--sources", default="codeforces",
                    help="comma list of judges: codeforces (default, rating-filtered) or any of aizu, atcoder, "
                         "codechef, hackerearth")
    args = ap.parse_args(argv)
    judges = {JUDGES[j.strip()] for j in args.sources.split(",") if j.strip()}
    import pyarrow.parquet as pq  # noqa: PLC0415  (only needed for the CLI)

    cols = ["name", "description", "public_tests", "private_tests", "generated_tests", "source", "cf_rating",
            "cf_contest_id", "cf_index", "input_file", "output_file", "difficulty"]
    tasks, seen = [], set()
    for path in args.parquet:
        rows = pq.read_table(path, columns=cols).to_pylist()
        if JUDGES["codeforces"] in judges:
            tasks += code_contests_rows(rows, args.min_rating, args.max_rating, args.max_tests)
        if judges - {JUDGES["codeforces"]}:
            tasks += other_judge_rows(rows, judges - {JUDGES["codeforces"]}, args.max_tests, seen)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for t in tasks:
            f.write(json.dumps(t, ensure_ascii=False) + "\n")
    print(json.dumps({"tasks": len(tasks), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
