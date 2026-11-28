"""Tokenization is deterministic: two shard-writer runs on the same input give identical shards (G3, 28 Nov, Tasks A and B).

    python teams/03-tokenizer/harness/determinism_shards.py --sample <sample v0 dir> --tokenizer <p1536>/tokenizer.json \
        --rebuilt <rebuilt p1536>/tokenizer.json --out teams/03-tokenizer/results/determinism_shards

1. Input: the first --rows documents of one train part of each pack in --packs, from Data A's tokenizer sample v0 (each part
   checked against the sample's SHA256SUMS). They go through Data A's synthetic pipeline run (shards/tests/synthetic_run.py:
   run_pipeline.py with made-up dedup and F2 that drop nothing), so the writer reads what a real run writes.
2. Data A's shard_writer.py writes every part twice, in two separate processes with different settings: run A with
   RAYON_NUM_THREADS=1 and PYTHONHASHSEED=1, run B with all cores and PYTHONHASHSEED=2. --tokens is small so parts roll
   over into several shards. Every file of A and B is compared byte for byte (.bin, .idx.npy, .docs.parquet, manifest,
   quality report); a manifest field that differs is named.
3. Run B is written again with --rebuilt, a p1536 rebuilt from the pinned inputs (rebuild_cand.py), and compared too;
   structural_identity_ids.py compares the two tokenizer files id by id.
4. Task B: the writer refuses any tokenizer but p1536 (C3, a copy with one byte changed); every manifest names p1536's sha256
   and eos 130822; roundtrip_shards.py (sha, header, index, eos at the end of every document, encode(decode(ids)) == ids)
   runs on run A.

Writes <out>.json and <out>.md: ids, counts and hashes only, no document text. Needs numpy, pyarrow, tokenizers, torch, pytest.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pyarrow.parquet as pq
import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
WRITER = REPO / "teams/01-data-a/shards/shard_writer.py"
sys.path.insert(0, str(REPO / "teams/01-data-a/shards/tests"))
from synthetic_run import make_run  # noqa: E402

P1536 = "e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b"
EOS = 130822
PACKS = ("en-fineweb-edu", "code-python", "code-cpp", "math-finemath", "sci-finepdfs", "ind-hi-fw2", "ind-ta-fw2",
         "ind-bn-fw2", "ind-sa-fw2", "ind-ml-fw2")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sample_rows(sample, pack, n):
    sums = {name: d for d, name in (l.split(maxsplit=1) for l in (sample / "SHA256SUMS").read_text().splitlines())}
    name = f"train/{pack}/part-00000.parquet"
    path = sample / name
    if sha(path) != sums[name]:
        sys.exit(f"{name}: sha256 does not match the sample's SHA256SUMS")
    t = pq.ParquetFile(path).iter_batches(batch_size=n, columns=["doc_id", "text", "url", "lang_label", "pack_id"])
    rows = next(t).to_pylist()[:n]
    return name, sums[name], rows


def write(run, part, tokenizer, out, env_extra, tokens):
    env = {**os.environ, **env_extra}
    cmd = [sys.executable, str(WRITER), "--input", str(run / "final" / "train" / part), "--manifest",
           str(run / "pipeline_manifest.json"), "--tokenizer", str(tokenizer), "--language", "mixed", "--script", "mixed",
           "--lane", "g3-determinism", "--domain", part.split("-")[0], "--dataset-version", "sample",
           "--tokens", str(tokens), "--out", str(out)]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, capture_output=True, text=True)
    return r, time.perf_counter() - t0


def diff_dirs(a, b):
    files = sorted({p.name for p in a.iterdir()} | {p.name for p in b.iterdir()})
    out = {}
    for f in files:
        pa_, pb = a / f, b / f
        if not (pa_.exists() and pb.exists()):
            out[f] = "missing in " + ("B" if pa_.exists() else "A")
            continue
        if sha(pa_) == sha(pb):
            continue
        if f.endswith(".json"):
            ja, jb = json.loads(pa_.read_text()), json.loads(pb.read_text())
            out[f] = sorted(k for k in set(ja) | set(jb) if ja.get(k) != jb.get(k))
        else:
            out[f] = "bytes differ"
    return files, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sample", type=Path, required=True)
    ap.add_argument("--tokenizer", type=Path, required=True)
    ap.add_argument("--rebuilt", type=Path, required=True)
    ap.add_argument("--c3", type=Path, help="C3 tokenizer.json, to show the writer refuses it")
    ap.add_argument("--rows", type=int, default=2000)
    ap.add_argument("--tokens", type=int, default=1_000_000, help="writer --tokens: small, so parts roll over")
    ap.add_argument("--packs", default=",".join(PACKS))
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    for p in (a.tokenizer, a.rebuilt):
        if sha(p) != P1536:
            sys.exit(f"{p} is not p1536")
    t_start = time.time()
    work = Path(tempfile.mkdtemp(prefix="g3-determinism-"))
    parts, inputs = {}, {}
    for pack in a.packs.split(","):
        name, digest, rows = sample_rows(a.sample, pack, a.rows)
        parts[("train", f"{pack}/part-00000.parquet")] = rows
        inputs[pack] = {"sample_part": name, "sha256": digest, "rows": len(rows)}
    with pytest.MonkeyPatch.context() as mp:
        run = make_run(work / "pipeline", mp, parts)

    settings = {"A": {"RAYON_NUM_THREADS": "1", "PYTHONHASHSEED": "1", "TOKENIZERS_PARALLELISM": "false"},
                "B": {"PYTHONHASHSEED": "2", "TOKENIZERS_PARALLELISM": "true"}}
    report = {"tokenizer_sha256": P1536, "inputs": inputs, "settings": settings, "packs": {}, "problems": []}
    for pack in inputs:
        part = f"{pack}/part-00000.parquet"
        res = {}
        for key, env in settings.items():
            r, secs = write(run, part, a.tokenizer, work / key / pack, env, a.tokens)
            if r.returncode:
                report["problems"].append(f"{pack} run {key}: writer exit {r.returncode}: {r.stderr[-400:]}")
            res[key] = secs
        r, _ = write(run, part, a.rebuilt, work / "R" / pack, settings["B"], a.tokens)
        if r.returncode:
            report["problems"].append(f"{pack} rebuilt: writer exit {r.returncode}: {r.stderr[-400:]}")
        files, d_ab = diff_dirs(work / "A" / pack, work / "B" / pack)
        _, d_br = diff_dirs(work / "B" / pack, work / "R" / pack)
        manifests = sorted((work / "A" / pack).glob("*-train-*.json"))
        manifests = [m for m in manifests if not m.name.endswith(".quality_report.json")]
        ms = [json.loads(m.read_text()) for m in manifests]
        bins = sorted((work / "A" / pack).glob("*.bin"))
        tokens = sum(m.get("token_count", 0) for m in ms)
        left = {}
        for q in (work / "A" / pack).glob("*.quality_report.json"):
            for reason, items in (json.loads(q.read_text()).get("left_out") or {}).items():
                left[reason] = left.get(reason, 0) + (len(items) if isinstance(items, list) else int(items or 0))
        report["packs"][pack] = {
            "shards": len(bins), "docs": sum(m.get("doc_count", 0) for m in ms), "tokens": tokens,
            "left_out": {k: v for k, v in left.items() if v},
            "files_compared": len(files), "differ_A_vs_B": d_ab, "differ_B_vs_rebuilt": d_br,
            "bin_sha256": {b.name: sha(b) for b in bins},
            "manifest_tokenizer_sha256": sorted({m.get("tokenizer_sha256") for m in ms}),
            "manifest_eos": sorted({m.get("eos_token_id") for m in ms}),
            "seconds": {k: round(v, 2) for k, v in res.items()},
            "tokens_per_second_1_thread": round(tokens / res["A"]) if res["A"] else None,
        }
        if d_ab or d_br:
            report["problems"].append(f"{pack}: shards differ: A/B {d_ab}, B/rebuilt {d_br}")
        if report["packs"][pack]["manifest_tokenizer_sha256"] != [P1536] or report["packs"][pack]["manifest_eos"] != [EOS]:
            report["problems"].append(f"{pack}: manifest sha or eos is not p1536 / 130822")

    # Task B: the writer refuses other tokenizers
    refusals = {}
    pack0 = next(iter(inputs))
    bad = work / "edited" / "tokenizer.json"
    bad.parent.mkdir(parents=True)
    bad.write_bytes(a.tokenizer.read_bytes() + b"\n")               # one byte added: the same JSON, another sha256
    for label, tok in (("p1536 plus one trailing newline (same JSON)", bad), ("C3 sample-v0", a.c3)):
        if tok is None:
            continue
        r, _ = write(run, f"{pack0}/part-00000.parquet", tok, work / "refused" / label.replace(" ", "_"), settings["B"], a.tokens)
        wrote = list((work / "refused" / label.replace(" ", "_")).glob("*.bin")) if (work / "refused" / label.replace(" ", "_")).exists() else []
        refusals[label] = {"sha256": sha(tok), "exit": r.returncode, "shards_written": len(wrote),
                           "message": (r.stderr.strip().splitlines() or [""])[-1].replace(str(tok), tok.name)[:160]}
        if r.returncode == 0 or wrote:
            report["problems"].append(f"writer accepted {label}")
    report["refusals"] = refusals

    # structural identity of the two p1536 files, and the round trip on run A
    sid = subprocess.run([sys.executable, str(HERE / "structural_identity_ids.py"), str(a.tokenizer), str(a.rebuilt)],
                         capture_output=True, text=True)
    report["structural_identity_ids"] = [l for l in sid.stdout.strip().splitlines() if l.strip()][-30:]
    flat = work / "A_all"
    flat.mkdir()
    for pack in inputs:
        for f in (work / "A" / pack).iterdir():
            if f.suffix in (".bin", ".npy", ".parquet") or (f.suffix == ".json" and "-train-" in f.name and "quality" not in f.name):
                shutil.copy(f, flat / f.name)
    rt_out = a.out.parent / (a.out.name + "_roundtrip.md")
    rt = subprocess.run([sys.executable, str(HERE / "roundtrip_shards.py"), str(flat), "--tokenizer", str(a.tokenizer),
                         "--out", str(rt_out), "--per-domain", "1000000"], capture_output=True, text=True)
    report["roundtrip_shards"] = {"exit": rt.returncode, "report": rt_out.name,
                                  "summary": next((l for l in rt_out.read_text().splitlines() if l.startswith("**Result"))
                                                  if rt_out.exists() else [], "")}
    if rt.returncode:
        report["problems"].append(f"roundtrip_shards.py exit {rt.returncode}: {rt.stderr[-300:]}")
    report["environment"] = {"python": platform.python_version(), "platform": platform.platform(),
                             "cores": os.cpu_count(), "script_sha256": sha(__file__),
                             "shard_writer_sha256": sha(WRITER)}
    report["result"] = "FAIL" if report["problems"] else "PASS"
    report["seconds"] = round(time.time() - t_start)
    a.out.with_suffix(".json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    shutil.rmtree(work)

    P = report["packs"]
    md = [f"# Deterministic tokenization: two shard-writer runs, p1536 `{P1536[:8]}…`", "",
          f"**Result: {report['result']}.** Made by `harness/determinism_shards.py` ({report['seconds']} s). "
          f"Input: the first {a.rows:,} documents of one train part of each of {len(P)} packs of Data A's tokenizer sample v0 "
          "(checked against its SHA256SUMS), through Data A's synthetic pipeline run and `shard_writer.py`.", "",
          "| Run | Tokenizer | Settings |", "|---|---|---|",
          "| A | release p1536 | `RAYON_NUM_THREADS=1`, `PYTHONHASHSEED=1`, separate process |",
          f"| B | release p1536 | all {os.cpu_count()} cores, `PYTHONHASHSEED=2`, separate process |",
          "| R | p1536 rebuilt from the pinned inputs (`rebuild_cand.py`) | as B |", "",
          "| Pack | Shards | Docs | Tokens | Files A vs B | A = B | B = R | eos / sha in manifests | tok/s, 1 thread (writer) |",
          "|---|---:|---:|---:|---:|---|---|---|---:|"]
    for k, v in P.items():
        md.append(f"| {k} | {v['shards']} | {v['docs']:,} | {v['tokens']:,} | {v['files_compared']} | "
                  f"{'identical' if not v['differ_A_vs_B'] else v['differ_A_vs_B']} | "
                  f"{'identical' if not v['differ_B_vs_rebuilt'] else v['differ_B_vs_rebuilt']} | "
                  f"{v['manifest_eos'][0] if len(v['manifest_eos']) == 1 else v['manifest_eos']} / "
                  f"{'p1536' if v['manifest_tokenizer_sha256'] == [P1536] else v['manifest_tokenizer_sha256']} | "
                  f"{v['tokens_per_second_1_thread'] or '-':,} |")
    tot = sum(v["tokens"] for v in P.values())
    md += [f"| **All** | {sum(v['shards'] for v in P.values())} | {sum(v['docs'] for v in P.values()):,} | {tot:,} | "
           f"{sum(v['files_compared'] for v in P.values())} | | | | |", "",
           "The writer's tokens per second include reading parquet, the round-trip check and writing; it is not the bare encoder rate.", "",
           "Documents the writer left out (the same in A, B and R): " + (", ".join(f"{k} {sum(v['left_out'].get(k, 0) for v in P.values())}" for k in sorted({r for v in P.values() for r in v['left_out']})) or "none") + ". "
           "By pack: " + "; ".join(f"{k} {v['left_out']}" for k, v in P.items() if v["left_out"]) + ".", "",
           "## The sha is enforced", "", "| Tokenizer given to the writer | sha256 | Exit | Shards written | Message |",
           "|---|---|---:|---:|---|"]
    md += [f"| {k} | `{v['sha256'][:12]}…` | {v['exit']} | {v['shards_written']} | {v['message']} |" for k, v in refusals.items()]
    md += ["", "## Round trip on run A (`roundtrip_shards.py`, every document)", "",
           f"{report['roundtrip_shards']['summary']} Full report: `{rt_out.name}`.", "",
           "## `structural_identity_ids.py` release vs rebuilt", "", "```", *report["structural_identity_ids"], "```", "",
           "## Shard sha256 (run A; B and R are identical)", "", "| Shard | sha256 |", "|---|---|"]
    md += [f"| `{n}` | `{s}` |" for v in P.values() for n, s in v["bin_sha256"].items()]
    md += ["", "```", f"python teams/03-tokenizer/harness/determinism_shards.py --sample <sample v0> --tokenizer release/tokenizer.json "
           f"--rebuilt <rebuild>/p1536/tokenizer.json --c3 <c3>/tokenizer.json --rows {a.rows} --tokens {a.tokens}", "```"]
    if report["problems"]:
        md += ["", "## Problems", ""] + [f"- {p}" for p in report["problems"]]
    a.out.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:30]))
    return 1 if report["problems"] else 0


if __name__ == "__main__":
    sys.exit(main())
