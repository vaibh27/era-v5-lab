#!/usr/bin/env python3
"""Pool several tasks' items and compare each format with bf16 item by item (paired, McNemar), for example Belebele's
five languages in one test of 4,500 items.

    python pooled.py results/raw/sarvam-1 [--tasks belebele_kan_Knda belebele_hin_Deva ...] [--base bf16] [--out pooled.md]

Reads <raw>/<format>/<task>.samples.jsonl (written by run_wide_gguf.py). An item is keyed by task and doc_id, so the
languages stay separate items; the statistics are those of ../eval_mini/paired.py (difference, 95% CI from the paired
variance, McNemar's exact test). Pooling answers "does the format lose anything across the tasks together", where a single
900-item task cannot tell a loss of 0.7 points from noise.
"""
from __future__ import annotations

import argparse, importlib.util, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_p = HERE / "paired_stats.py" if (HERE / "paired_stats.py").exists() else HERE.parent / "eval_mini" / "paired.py"
_spec = importlib.util.spec_from_file_location("paired_stats", _p)
stats = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(stats)

BELEBELE = ["belebele_kan_Knda", "belebele_hin_Deva", "belebele_tel_Telu", "belebele_tam_Taml", "belebele_ben_Beng"]


def load(fmt_dir: Path, tasks: list[str]) -> dict[str, float]:
    items = {}
    for task in tasks:
        f = fmt_dir / f"{task}.samples.jsonl"
        if not f.exists():
            return {}
        for line in open(f):
            r = json.loads(line)
            items[f"{task}|{r['doc_id']}"] = float(r["acc_norm"] if "acc_norm" in r else r["acc"])
    return items


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("raw", type=Path)
    ap.add_argument("--tasks", nargs="+", default=BELEBELE)
    ap.add_argument("--base", default="bf16")
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    base = load(args.raw / args.base, args.tasks)
    if not base:
        raise SystemExit(f"{args.raw / args.base}: not every task in {args.tasks} is there")
    pts = lambda v: f"{100 * v:+.2f}"
    lines = [f"Pooled over {len(args.tasks)} tasks ({len(base):,} items), paired against **{args.base}**. Points, 95% CI, "
             "b = right in base and wrong in the format, c = the reverse.", "",
             "| Format | Base | Format | Difference (95% CI) | b / c | McNemar p |", "|---|---:|---:|---|---:|---:|"]
    for fmt_dir in sorted(p for p in args.raw.iterdir() if p.is_dir() and p.name != args.base):
        other = load(fmt_dir, args.tasks)
        if not other:
            continue
        r = stats.compare(base, other, binary=True)
        lines.append(f"| {fmt_dir.name} | {100 * r['base']:.2f} | {100 * r['fmt']:.2f} | {pts(r['diff'])} "
                     f"({pts(r['lo'])} to {pts(r['hi'])}) | {r['b']} / {r['c']} | "
                     f"{'< 0.0001' if r['p'] < 0.0001 else format(r['p'], '.4f')} |")
    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)


if __name__ == "__main__":
    main()
