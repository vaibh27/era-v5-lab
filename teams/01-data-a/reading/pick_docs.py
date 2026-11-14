"""Pick the doc_ids one reader audits, the same way on every machine.

  python pick_docs.py --parts sci-finepdfs_part1.parquet sci-finepdfs_part2.parquet --rows 1-100 --out docids.txt

Concatenates the parts in the order given, shuffles the doc_ids with seed 20261101, and prints rows
--rows of the shuffled list (1-based, both ends included). Two readers on the same pack pass the same
--parts in the same order and different --rows, so their sets never overlap.

The shuffle is random.Random(SEED).shuffle on the doc_id list, not a pandas or numpy sampler: those
change order between versions, and then two readers on one pack read the same docs.
"""
import argparse, random, sys
from pathlib import Path
import pyarrow.parquet as pq

SEED = 20261101

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--parts", type=Path, nargs="+", required=True, help="Part 1 first, then Part 2")
ap.add_argument("--rows", required=True, help="1-based inclusive range of the shuffled list, e.g. 1-100")
ap.add_argument("--column", default="doc_id")
ap.add_argument("--out", type=Path, help="write the ids here as well as to stdout")
args = ap.parse_args()

try:
    first, last = (int(n) for n in args.rows.split("-", 1))
except ValueError:
    sys.exit(f"--rows {args.rows!r}: use <first>-<last>, 1-based, e.g. 1-100")
if not 1 <= first <= last:
    sys.exit(f"--rows {args.rows!r}: first must be at least 1 and no more than last")

ids = []
for part in args.parts:
    ids.extend(pq.read_table(part, columns=[args.column]).column(args.column).to_pylist())
if len(set(ids)) != len(ids):
    sys.exit(f"{len(ids) - len(set(ids))} duplicate {args.column}s across the parts: check you passed each part once")
if last > len(ids):
    sys.exit(f"--rows {args.rows!r}: the parts hold only {len(ids)} docs")

random.Random(SEED).shuffle(ids)
picked = ids[first - 1:last]
print("\n".join(str(i) for i in picked))
print(f"\n{len(picked)} docs, rows {first}-{last} of {len(ids)} shuffled with seed {SEED}, "
      f"from {', '.join(p.name for p in args.parts)}", file=sys.stderr)
if args.out:
    args.out.write_text("\n".join(str(i) for i in picked) + "\n", encoding="utf-8")
