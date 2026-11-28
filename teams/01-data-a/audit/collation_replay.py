"""Replay the week-3 audit labels through the current cleaner, rule by rule, and count what each rule does on whole packs.

    python -I teams/01-data-a/audit/collation_replay.py docs  --raw <tokenizer_sample_v0/train> --labels <collation_docs.csv> --out <csv>
    python -I teams/01-data-a/audit/collation_replay.py scope --raw <tokenizer_sample_v0/train> --out <csv> [--pack en-dclm ...]

docs   reads pack, doc_id, audit, verdict and diseases from --labels and writes one row per label: what the current
       spec_v1_1/clean.py does with that document (keep, fix or drop), and the rules that acted, in order.
       Each replay is checked against clean() itself; a difference stops the script.
scope  counts, for each pack and rule, the documents the rule fixes and drops at its place in the pipeline, and how many of
       the drops are over 2,000 characters. One job for each row group, run in parallel (--workers).

The outputs hold ids, rule ids and counts only: no document text.
"""
import sys
sys.dont_write_bytecode = True

import argparse
import csv
import hashlib
import importlib.util
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import pyarrow.parquet as pq

CLEAN = Path(__file__).resolve().parents[1] / "spec_v1_1" / "clean.py"
SCOPE_PACKS = ["en-dclm", "en-fineweb-edu", "en-nemotroncc", "en-nemotroncc-synth", "math-finemath", "math-megamath"]
COLUMNS = ["doc_id", "text", "url", "lang_label"]  # every field the rules read, besides pack_id
LONG = 2000
c = None  # the cleaner module, loaded once in each process


def load():
    global c
    spec = importlib.util.spec_from_file_location("clean", CLEAN)
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)


def rules(pack):
    """[(rule id, rule function)] in the order clean() runs them on this pack."""
    pipeline = (*c._PIPELINES[c._FAMILY[pack]], c._shared_f1_rule)
    return [(c._ID[make], make()) for make in pipeline if pack in c.SCOPE[c._ID[make]]]


def trace(doc, pipeline):
    """([(rule id, action, change in characters)], the cleaned text or None for a drop)"""
    steps, doc = [], dict(doc)
    for rid, rule in pipeline:
        action, text = rule(doc)
        if action == "drop":
            return steps + [(rid, "drop", 0)], None
        if text != doc["text"]:
            steps.append((rid, "fix", len(text) - len(doc["text"])))
            doc["text"] = text
    return steps, doc["text"]


def docs(args):
    load()
    labels = list(csv.DictReader(open(args.labels, encoding="utf-8", newline="")))
    found = {}
    for pack in sorted({r["pack"] for r in labels}):
        ids = sorted({r["doc_id"] for r in labels if r["pack"] == pack})
        pipeline = rules(pack)
        for part in sorted((args.raw / pack).glob("*.parquet")):
            for doc in pq.read_table(part, columns=COLUMNS, filters=[("doc_id", "in", ids)]).to_pylist():
                doc["pack_id"] = pack
                steps, text = trace(doc, pipeline)
                action, clean_text = c.clean(doc)
                if (text is None) != (action == "drop") or (text is not None and text != clean_text):
                    sys.exit(f"replay differs from clean(): {pack} {doc['doc_id']}")
                found[pack, doc["doc_id"]] = [action, "; ".join(f"{rid} {act}{'' if act == 'drop' else f' {n:+d}'}"
                                                                for rid, act, n in steps)]
    missing = [(r["pack"], r["doc_id"]) for r in labels if (r["pack"], r["doc_id"]) not in found]
    if missing:
        sys.exit(f"{len(missing)} labelled documents are not in --raw: {missing[:5]}")
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["pack", "doc_id", "audit", "verdict", "diseases", "current_action", "current_rules"])
        for r in labels:
            out.writerow([r[k] for k in ("pack", "doc_id", "audit", "verdict", "diseases")] + found[r["pack"], r["doc_id"]])
    print(f"{len(labels)} labels replayed, each equal to clean():", dict(Counter(v[0] for v in found.values())))


def count_row_group(job):
    part, pack, row_group = job
    pipeline = rules(pack)
    table = pq.ParquetFile(part).read_row_group(row_group, columns=COLUMNS)
    n, counts = table.num_rows, Counter()
    for doc in table.to_pylist():
        doc["pack_id"] = pack
        steps, _ = trace(doc, pipeline)
        for rid, act, _ in steps:
            counts[rid, act] += 1
            if act == "drop" and len(doc["text"]) > LONG:
                counts[rid, "drop_long"] += 1
    return pack, n, counts


def scope(args):
    jobs = []
    for pack in args.pack or SCOPE_PACKS:
        for part in sorted((args.raw / pack).glob("*.parquet")):
            jobs += [(part, pack, g) for g in range(pq.ParquetFile(part).num_row_groups)]
    n, counts = Counter(), defaultdict(Counter)
    with Pool(args.workers, initializer=load) as pool:
        for pack, k, part_counts in pool.imap_unordered(count_row_group, jobs):
            n[pack] += k
            counts[pack].update(part_counts)
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        out = csv.writer(f, lineterminator="\n")
        out.writerow(["pack", "docs", "rule", "fixed", "dropped", "dropped_over_2000_chars"])
        for pack in args.pack or SCOPE_PACKS:
            for rid in sorted({r for r, _ in counts[pack]}):
                out.writerow([pack, n[pack], rid] + [counts[pack][rid, k] for k in ("fix", "drop", "drop_long")])
            dropped = sum(v for (_, k), v in counts[pack].items() if k == "drop")
            print(f"{pack}: {n[pack]} docs, {dropped} dropped")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, run in (("docs", docs), ("scope", scope)):
        p = sub.add_parser(name)
        p.set_defaults(run=run)
        p.add_argument("--raw", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
        if name == "docs":
            p.add_argument("--labels", type=Path, required=True)
        else:
            p.add_argument("--pack", nargs="*")
            p.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    print("clean.py sha256", hashlib.sha256(CLEAN.read_bytes().replace(b"\r\n", b"\n")).hexdigest())
    t = time.perf_counter()
    args.run(args)
    print(f"wall time {time.perf_counter() - t:.1f} s")


if __name__ == "__main__":
    main()
