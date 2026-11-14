"""Speed and byte-identity check for a clean.py change: run it on a BEFORE and an AFTER clean.py, then compare.

Reusable for any cleaner speed-up (English, Indic, ...): it needs only a folder holding a clean.py (stdlib-only, with
clean(doc) -> (action, text) and _FAMILY) and a folder of <pack_id>.parquet packs. Reads the packs, writes only --out,
prints IDs, counts, hashes and timings, never document text. Needs pyarrow.

  # 1. one run per clean.py (one process each, so the two never share a module or cache)
  taskset -c 0 python -I speed_check.py run --clean-dir <before>/teams/01-data-a/spec_v1 --packs <packs> \\
      --exclude ind-mr-sangraha-v --repeat 3 --out before.json
  taskset -c 0 python -I speed_check.py run --clean-dir <after>/teams/01-data-a/spec_v1  --packs <packs> \\
      --exclude ind-mr-sangraha-v --repeat 3 --out after.json

  # 2. compare: require the original 13 packs, zero errors, equal rows/actions and input/output hashes
  python -I speed_check.py compare before.json after.json [--md speed_table.md]

  # A deliberately different population must declare its exact inventory:
  python -I speed_check.py compare before.json after.json --expected-packs code-cpp code-go

  # 3. how many documents a rule fixes or drops, per pack (for the docs-changed figure of a speed-log entry)
  python -I speed_check.py count --clean-dir <tree>/teams/01-data-a/spec_v1 --packs <packs> --rules DA-C5-002 --only en-dclm

The output sha256 of a pack covers, for every document in file order, doc_id, action ("keep"/"fix"/"drop") and the
cleaned text (or the exception type if clean() raised), each length-prefixed. Parquet bytes, summaries and timings are
not hashed, so the same behaviour always gives the same sha. Every --repeat must give the same sha, or the run stops
(a cache that changed the output would show up here).

--limit N uses only the first N docs of each pack (for a quick look at a large part); compare refuses two runs whose limits differ.

Timing: only the clean(doc) loop is timed (the pack is already in memory), best of --repeat, single process. Pin it to
one core with taskset as above. The first repeat is also recorded (`first_docs_per_s`): it is the cold-cache number, and
the best of the later repeats can be a little higher for rules that cache lookups (such as the foreign-letter script cache).
"""
import argparse, hashlib, importlib, json, os, platform, struct, sys, time
from pathlib import Path

ORIGINAL_PACKS = "code-cpp code-go code-java code-js-ts code-python code-rust en-dclm en-fineweb-edu en-nemotroncc en-nemotroncc-synth math-finemath math-megamath sci-finepdfs".split()


def sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_cleaner(clean_dir):
    sys.path.insert(0, str(Path(clean_dir).resolve()))
    c = importlib.import_module("clean")
    if Path(c.__file__).resolve().parent != Path(clean_dir).resolve():
        sys.exit(f"imported {c.__file__}, not a clean.py in {clean_dir}")
    return c


def clean_pass(c, rows, pack, label="", progress=30):
    """One timed pass over the pack. Returns (seconds, sha256 of the output stream, actions, error count).
    Every `progress` seconds it prints the docs done and the docs/s so far to stderr (0 = silent)."""
    h, actions, errors = hashlib.sha256(), {}, 0
    t0 = time.perf_counter()
    next_report = t0 + progress
    results = []
    for i, row in enumerate(rows, 1):
        doc = dict(row, pack_id=pack)
        try:
            results.append((row["doc_id"], *c.clean(doc)))
        except Exception as e:  # a crashing doc is part of the behaviour: record the type, never the text
            results.append((row["doc_id"], "error", type(e).__name__))
            errors += 1
        if progress and time.perf_counter() >= next_report:
            now = time.perf_counter()
            print(f"  [{time.strftime('%H:%M:%S')}] {pack} {label}: {i}/{len(rows)} docs ({i / len(rows):.0%}), "
                  f"{i / (now - t0):.1f} docs/s so far, about {(len(rows) - i) / (i / (now - t0)) / 60:.1f} min left",
                  file=sys.stderr, flush=True)
            next_report = now + progress
    seconds = time.perf_counter() - t0
    for doc_id, action, text in results:  # hashing is outside the clock
        actions[action] = actions.get(action, 0) + 1
        for part in (str(doc_id), action, text):
            b = part.encode("utf-8", "surrogatepass")
            h.update(struct.pack("<Q", len(b)))
            h.update(b)
    return seconds, h.hexdigest(), actions, errors


def run(args):
    import pyarrow.parquet as pq
    if args.repeat < 1 or args.limit < 0 or args.progress < 0:
        sys.exit("repeat must be positive; limit and progress must be nonnegative")
    c = load_cleaner(args.clean_dir)
    packs = sorted(p.stem for p in Path(args.packs).glob("*.parquet"))
    if args.only:
        missing = set(args.only) - set(packs)
        if missing:
            sys.exit(f"requested packs missing: {sorted(missing)}")
        packs = [p for p in packs if p in args.only]
    packs = [p for p in packs if p not in args.exclude]
    skipped = [p for p in packs if p not in c._FAMILY]
    packs = [p for p in packs if p in c._FAMILY]
    if not packs:
        sys.exit("no registered packs selected")
    out = {"clean_py_sha256": sha256_file(c.__file__), "clean_py": str(Path(c.__file__).resolve()),
           "python": platform.python_version(), "machine": platform.processor() or platform.machine(),
           "repeat": args.repeat, "limit": args.limit, "skipped_not_registered": skipped,
           "cpu_affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None, "packs": {}}
    failed = False
    for pack in packs:
        path = Path(args.packs) / f"{pack}.parquet"
        print(f"[{time.strftime('%H:%M:%S')}] {pack}: loading {path.name}", file=sys.stderr, flush=True)
        table = pq.read_table(path)
        if args.limit:
            table = table.slice(0, args.limit)  # the first N docs in file order: the same N for the before and the after run
        rows = table.to_pylist()
        if not rows:
            sys.exit(f"{pack}: empty input cannot be benchmarked")
        passes = [clean_pass(c, rows, pack, f"pass {n}/{args.repeat}", args.progress) for n in range(1, args.repeat + 1)]
        shas = {p[1] for p in passes}
        if len(shas) != 1:
            sys.exit(f"{pack}: repeats gave different output: the cleaner is not deterministic ({len(shas)} different shas)")
        times = [p[0] for p in passes]
        out["packs"][pack] = {"docs": len(rows), "output_sha256": passes[0][1], "actions": passes[0][2], "errors": passes[0][3],
                              "input_sha256": sha256_file(path), "seconds": [round(t, 4) for t in times],
                              "docs_per_s": round(len(rows) / min(times), 1), "first_docs_per_s": round(len(rows) / times[0], 1)}
        failed |= any(p[3] for p in passes)
        e = out["packs"][pack]
        print(f"{pack:<22}{e['docs']:>6} docs  {e['docs_per_s']:>9.1f} docs/s (first {e['first_docs_per_s']:.1f})  "
              f"{e['output_sha256'][:16]}  {e['actions']}", file=sys.stderr, flush=True)
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {args.out}: {len(out['packs'])} packs, clean.py {out['clean_py_sha256'][:12]}")
    if failed:
        sys.exit("cleaning errors occurred; this run is not equivalence evidence")


def count(args):
    """For each pack, how many documents each named rule fixes or drops while clean.py runs (rules in pipeline order)."""
    import pyarrow.parquet as pq
    from collections import defaultdict
    c = load_cleaner(args.clean_dir)
    rules = args.rules.split(",")
    unknown = [r for r in rules if r not in c._RULE]
    if unknown:
        sys.exit(f"not a rule id in clean.py: {unknown}")
    tally = defaultdict(lambda: {"calls": 0, "fix": 0, "drop": 0})

    def counted(rid, make):
        def factory():
            rule = make()

            def run_rule(doc):
                action, text = rule(doc)
                e = tally[rid]
                e["calls"] += 1
                if action == "drop":
                    e["drop"] += 1
                elif text != doc["text"]:
                    e["fix"] += 1
                return action, text
            return run_rule
        return factory

    wrapped = {make: counted(rid, make) for rid, make in c._RULE.items()}
    c._ID.update({w: c._ID[m] for m, w in wrapped.items()})  # _run looks a rule's SCOPE up by its factory
    original_run = c._run
    c._run = lambda doc, *factories: original_run(doc, *(wrapped[make] for make in factories))
    packs = sorted(p.stem for p in Path(args.packs).glob("*.parquet"))
    packs = [p for p in packs if (not args.only or p in args.only) and p in c._FAMILY]
    out = {}
    for pack in packs:
        table = pq.read_table(Path(args.packs) / f"{pack}.parquet")
        if args.limit:
            table = table.slice(0, args.limit)
        tally.clear()
        fixed = dropped = 0
        rows = table.to_pylist()
        for row in rows:
            action, _ = c.clean(dict(row, pack_id=pack))
            fixed += action == "fix"
            dropped += action == "drop"
        out[pack] = {"docs": len(rows), "clean_fixed": fixed, "clean_dropped": dropped, **{r: dict(tally[r]) for r in rules}}
        parts = "; ".join(f"{r}: ran on {tally[r]['calls']}, fixed {tally[r]['fix']}, dropped {tally[r]['drop']}" for r in rules)
        print(f"{pack:<22}{len(rows):>7} docs  (clean() fixed {fixed}, dropped {dropped})  {parts}", flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8", newline="\n")


def compare(args):
    b, a = (json.loads(Path(p).read_text(encoding="utf-8")) for p in (args.before, args.after))
    expected = set(args.expected_packs)
    if not expected or set(b["packs"]) != expected or set(a["packs"]) != expected:
        sys.exit(f"pack inventory must match expected packs: {sorted(expected)}")
    if b.get("limit") != a.get("limit"):
        sys.exit(f"the runs used different --limit values ({b.get('limit')} and {a.get('limit')}): not comparable")
    names = sorted(set(b["packs"]) | set(a["packs"]))
    lines = [f"| pack | docs | output sha256 (before = after) | same | docs/s before | docs/s after | speed-up |",
             "|---|---:|---|---|---:|---:|---:|"]
    bad = 0
    for p in names:
        if p not in b["packs"] or p not in a["packs"]:
            lines.append(f"| {p} | | only in {'before' if p in b['packs'] else 'after'} | **NO** | | | |")
            bad += 1
            continue
        x, y = b["packs"][p], a["packs"][p]
        if x.get("errors") != 0 or y.get("errors") != 0:
            sys.exit(f"{p}: cleaning errors invalidate the comparison")
        if x["docs"] <= 0 or x["docs"] != y["docs"] or x["docs_per_s"] <= 0 or y["docs_per_s"] <= 0:
            sys.exit(f"{p}: document counts and positive timings must be comparable")
        same = x["output_sha256"] == y["output_sha256"] and x["input_sha256"] == y["input_sha256"] and x["actions"] == y["actions"]
        bad += not same
        sha = f"`{x['output_sha256'][:16]}`" if same else f"`{x['output_sha256'][:12]}` vs `{y['output_sha256'][:12]}`"
        lines.append(f"| {p} | {x['docs']} | {sha} | {'yes' if same else '**NO**'} | {x['docs_per_s']:.1f} | {y['docs_per_s']:.1f} | "
                     f"{y['docs_per_s'] / x['docs_per_s']:.1f}x |")
    head = (f"before clean.py `{b['clean_py_sha256'][:12]}`, after clean.py `{a['clean_py_sha256'][:12]}`; "
            f"{len(names) - bad} of {len(names)} packs have the same output sha256.")
    text = head + "\n\n" + "\n".join(lines)
    print(text)
    if args.md:
        Path(args.md).write_text(text + "\n", encoding="utf-8", newline="\n")
    sys.exit(1 if bad else 0)


ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
r = sub.add_parser("run", help="hash and time one clean.py on a folder of packs")
r.add_argument("--clean-dir", required=True, type=Path, help="the folder holding clean.py")
r.add_argument("--packs", required=True, type=Path, help="the folder of <pack_id>.parquet files")
r.add_argument("--only", nargs="*", default=[], help="run just these pack ids")
r.add_argument("--exclude", nargs="*", default=[], help="pack ids to leave out (e.g. ind-mr-sangraha-v for the 13 non-Indic packs)")
r.add_argument("--repeat", type=int, default=3)
r.add_argument("--progress", type=int, default=30, help="print progress to stderr every N seconds (0 = silent)")
r.add_argument("--limit", type=int, default=0, help="use only the first N docs of each pack (0 = all); give the before and after run the same N")
r.add_argument("--out", required=True, type=Path)
r.set_defaults(fn=run)
n = sub.add_parser("count", help="how many documents each named rule fixes or drops, per pack (the 'docs changed' figure)")
n.add_argument("--clean-dir", required=True, type=Path)
n.add_argument("--packs", required=True, type=Path)
n.add_argument("--rules", required=True, help="rule ids, comma separated, e.g. DA-C5-002,DA-D3-003")
n.add_argument("--only", nargs="*", default=[], help="count on just these pack ids")
n.add_argument("--limit", type=int, default=0)
n.add_argument("--out", type=Path, help="also write the counts as JSON")
n.set_defaults(fn=count)
k = sub.add_parser("compare", help="compare two run files; exit 1 if any pack's output sha256 differs")
k.add_argument("before", type=Path)
k.add_argument("after", type=Path)
k.add_argument("--md", type=Path, help="also write the table to this file")
k.add_argument("--expected-packs", nargs="+", default=ORIGINAL_PACKS,
               help="exact required pack inventory (default: the 13 original sample packs)")
k.set_defaults(fn=compare)
args = ap.parse_args()
args.fn(args)
