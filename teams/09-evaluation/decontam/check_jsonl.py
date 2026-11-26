"""Run the training-text check on JSONL training files in one command (G9).

    python check_jsonl.py ./inputs/train.jsonl --work <dir>             # no network
    python check_jsonl.py a.jsonl b.jsonl --work <dir> --id-field id --text-field text

For training text staged as JSONL (one JSON object per line, as train_week2.py --train-data reads: "text" and an
optional "id"), not as token shards. For each file this runs decontam_check.py (full set: no --benchmarks, --tags or
--exclude) and canary_scan.py, with the scripts unchanged. It needs no network and no tokenizer, so it can run as a job
that reads only its inputs.

Speed. The file is cut into parts of whole lines, in file order (the lines are copied as they are, never parsed or
changed), and decontam_check.py runs on each part in its own process, in one pool of --jobs worker processes. By
default a part is the total size of the files divided by --jobs, at most 64 MB: so a small file still uses every
worker, and each process loads the fingerprint set (about 2 s) as few times as possible. --part-bytes sets a fixed
size. A document is checked alone, so the parts give the same result as one run on the file. canary_scan.py
reads all parts of a file in one process. --jobs defaults as in check_shards.py (one decontam_check.py holds the
fingerprint set, about 1.4 GB). Peak disk use is one copy of the files being checked, in <work>/text.

A line without --id-field is named "<file name>:<line index>" (0-based, blank lines counted), as the two scripts name it
when they read the whole file. The parts are deleted after the checks, and any parts in <work>/text are deleted at the
start and at the end of a run. One run at a time can use a --work folder (the lock of check_shards.py).

Results go to <work>/results/<file name>.json (sha256 of the file, bytes, documents, flagged doc IDs, canary hits; never
text). At the end it prints a markdown table, the fingerprint set_sha256 and the canary count. Exit code: 0 when every
file passed with 0 flagged and 0 canary hits; 1 when a file failed a step or has a hit; 2 for a usage error.
Note: decontam_check.py on its own exits 0 also when it flags documents; this runner reads its "flagged" count.
"""

import argparse
import hashlib
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # python -I leaves the script's folder out of sys.path
    sys.path.insert(0, str(HERE))

from check_shards import (CANARY_COUNT, CANARY_LINE, DECONTAM_LINE, GB_PER_JOB, SET_LINE, CheckError,  # noqa: E402
                          default_fingerprints, default_jobs, fingerprint_set_sha, lock_work, run)

PART_NAME = re.compile(r"^(?:.*[\\/])?(.+)\.part(\d{4,})\.jsonl:(\d+)$")  # 5+ digits past 9,999 parts
MAX_PART_BYTES = 64_000_000
CHUNK = 1 << 24  # bytes read at a time while cutting a file into parts
# A blank line, as the scripts skip it (line.strip() is empty). Other Unicode spaces make the counts differ: a loud failure.
BLANK = re.compile(rb"(?m)^[ \t\r\f\v]*\n")


def auto_part_bytes(total_bytes, jobs):
    """The default part size: one part per worker, at most MAX_PART_BYTES. Measured on sample text: 100 MB with
    --jobs 10 takes 7 s in 10 parts, 23 s in 2 parts (64 MB), and more parts than workers are slower again."""
    return max(1, min(MAX_PART_BYTES, -(-total_bytes // jobs)))


def part_stem(n, name):
    return f"f{n:03d}-{name}"


def part_path(text_dir, n, name, k):
    return text_dir / f"{part_stem(n, name)}.part{k:04d}.jsonl"


def split_jsonl(path, n, part_bytes, text_dir):
    """Cut one JSONL file into parts of whole lines, about part_bytes each. Lines end at b"\n" only, and the bytes are
    copied as they are (a last line without a newline gets one in its part).

    Returns (sha256 of the file, bytes, non-blank lines, parts), parts = [(path, index of its first line)]."""
    h = hashlib.sha256()
    parts, size, docs, line_no = [], 0, 0, 0
    out, out_bytes, carry = None, 0, b""
    try:
        with open(path, "rb") as f:
            while True:
                chunk = f.read(CHUNK)
                h.update(chunk)
                size += len(chunk)
                data = carry + chunk
                if chunk:
                    cut = data.rfind(b"\n") + 1  # whole lines only; the rest waits for the next chunk
                    data, carry = data[:cut], data[cut:]
                elif data and not data.endswith(b"\n"):
                    data += b"\n"
                pos = 0
                while pos < len(data):
                    if out is None or out_bytes >= part_bytes:
                        if out is not None:
                            out.close()
                        p = part_path(text_dir, n, path.name, len(parts))
                        out, out_bytes = open(p, "wb"), 0
                        parts.append((p, line_no))
                    # up to the end of the line that fills the part, or to the end of the data
                    end = data.find(b"\n", min(pos + part_bytes - out_bytes, len(data)) - 1) + 1 or len(data)
                    piece = data[pos:end]
                    out.write(piece)
                    out_bytes += len(piece)
                    lines = piece.count(b"\n")
                    line_no += lines
                    docs += lines - len(BLANK.findall(piece))
                    pos = end
                if not chunk:
                    break
    finally:
        if out is not None:
            out.close()
    return h.hexdigest(), size, docs, parts


def global_id(doc_id, first_lines, name):
    """Map "<part file>:<n>" (a line with no id) back to "<file name>:<line index in the file>"."""
    m = PART_NAME.match(doc_id)
    if not m:
        return doc_id
    key = (m.group(1), int(m.group(2)))
    if key not in first_lines:
        return doc_id
    return f"{name}:{first_lines[key] + int(m.group(3))}"


def check_one(n, path, a, pool):
    """Run both checks on one JSONL file, return the result dict (never document text)."""
    t0 = time.time()
    text_dir = a.work / "text"
    parts = []
    try:
        with a.split_lock:
            sha, size, docs, parts = split_jsonl(path, n, a.part_bytes, text_dir)
        if not parts or docs == 0:
            raise CheckError("the file has no documents")
        first_lines = {(part_stem(n, path.name), k): first for k, (_, first) in enumerate(parts)}
        fields = ["--id-field", a.id_field, "--text-field", a.text_field]
        fp = ["--fingerprints", str(a.fingerprints)] if a.fingerprints else []
        f_dec = [pool.submit(run, [sys.executable, str(HERE / "decontam_check.py"), str(p)] + fields + fp)
                 for p, _ in parts]
        f_can = pool.submit(run, [sys.executable, str(HERE / "canary_scan.py")] + [str(p) for p, _ in parts]
                            + fields + fp)
        dec = [f.result() for f in f_dec]
        code_c, out_c, err_c = f_can.result()
    finally:  # never leave training text behind, also when a step fails
        for p, _ in parts:
            p.unlink(missing_ok=True)

    sets, n_docs, n_flagged, flagged_ids = set(), 0, 0, []
    for code_d, out_d, err_d in dec:
        m_set, m_dec = SET_LINE.search(err_d), DECONTAM_LINE.search(err_d)
        if code_d != 0 or not m_set or not m_dec:
            raise CheckError(f"decontam_check.py failed (exit {code_d}): {err_d.strip()[-300:]}")
        sets.add(m_set.groups())
        n_docs += int(m_dec.group(1))
        n_flagged += int(m_dec.group(2))
        flagged_ids += [global_id(json.loads(line)["doc_id"], first_lines, path.name)
                        for line in out_d.splitlines() if line.strip()]
    if len(sets) != 1:
        raise CheckError(f"the parts used different fingerprint sets: {sorted(sets)}")
    version, set_sha, scope = sets.pop()
    m_cc, m_can = CANARY_COUNT.search(err_c), CANARY_LINE.search(err_c)
    if code_c not in (0, 1) or not m_cc or not m_can:  # 1 = hits (counted below), 2 = a file was skipped
        raise CheckError(f"canary_scan.py failed (exit {code_c}): {err_c.strip()[-300:]}")
    if n_docs != docs or int(m_can.group(1)) != docs:
        raise CheckError(f"document counts differ: lines {docs}, decontam {n_docs}, canary {m_can.group(1)}")
    hits = [json.loads(line) for line in out_c.splitlines() if line.strip()]
    return {
        "file": path.name,
        "sha256": sha,
        "bytes": size,
        "docs": docs,
        "fingerprints_version": version,
        "set_sha256": set_sha,
        "scope": scope,
        "flagged": n_flagged,
        "flagged_doc_ids": flagged_ids,
        "canaries_checked": int(m_cc.group(1)),
        "canary_hits": int(m_can.group(3)),
        "canary_hit_list": [{"canary": h["canary"], "doc_id": global_id(h["doc_id"], first_lines, path.name),
                             "offset": h["offset"]} for h in hits],
        "parts": len(dec),
        "seconds": round(time.time() - t0, 1),
    }


def table(results):
    lines = ["| # | file | sha256 | bytes | docs | flagged | canary hits |", "|---:|---|---|---:|---:|---:|---:|"]
    for n, r in enumerate(results, 1):
        lines.append(f"| {n} | `{r['file']}` | `{r['sha256']}` | {r['bytes']:,} | {r['docs']:,} | {r['flagged']} | "
                     f"{r['canary_hits']} |")
    return "\n".join(lines)


def clean_text(work):
    """Delete parts in <work>/text (called while this run holds the lock)."""
    for old in (work / "text").glob("*.part*.jsonl"):
        old.unlink(missing_ok=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", type=Path, nargs="+", help="JSONL training files (one JSON object per line)")
    ap.add_argument("--work", type=Path, required=True,
                    help="work folder: text/ (copies of the training text while it runs: never commit it) and results/")
    ap.add_argument("--id-field", default="id", help="document id field (default id, as train_week2.py)")
    ap.add_argument("--text-field", default="text", help="text field (default text)")
    ap.add_argument("--fingerprints", type=Path, help="fingerprint set folder (default: the scripts' default, v2)")
    ap.add_argument("--jobs", type=int, default=default_jobs(),
                    help=f"worker processes (default {default_jobs()} here: cores, at most one per "
                         f"{GB_PER_JOB} GB of memory)")
    ap.add_argument("--part-bytes", type=int,
                    help=f"bytes per decontam_check.py part (default: the files' total bytes / --jobs, at most "
                         f"{MAX_PART_BYTES:,})")
    a = ap.parse_args(argv)

    if a.jobs < 1:
        ap.error("--jobs must be 1 or more")
    if a.part_bytes is not None and a.part_bytes < 1:
        ap.error("--part-bytes must be 1 or more")
    for p in a.files:
        if not p.is_file():
            ap.error(f"{p} not found")
        if p.suffix != ".jsonl":
            ap.error(f"{p}: not a .jsonl file")
    if len({p.name for p in a.files}) != len(a.files):
        ap.error("two files have the same name: results are kept by file name")
    if a.part_bytes is None:
        a.part_bytes = auto_part_bytes(sum(p.stat().st_size for p in a.files), a.jobs)
    try:
        set_sha = fingerprint_set_sha(a.fingerprints or default_fingerprints())
    except CheckError as e:
        ap.error(str(e))
    for sub in ("text", "results"):
        (a.work / sub).mkdir(parents=True, exist_ok=True)
    try:
        work_lock = lock_work(a.work)
    except CheckError as e:
        ap.error(str(e))
    try:
        clean_text(a.work)
        return run_checks(a, set_sha)
    finally:
        clean_text(a.work)
        work_lock.close()


def run_checks(a, set_sha):
    a.split_lock = threading.Semaphore(2)  # files cut at a time (disk: one copy of each)
    lock = threading.Lock()

    def task(n, path, pool):
        try:
            r = check_one(n, path, a, pool)
        except Exception as e:  # one bad file must not stop the others
            with lock:
                print(f"{path.name}: FAILED: {e}", file=sys.stderr)
            return None
        rp = a.work / "results" / f"{path.name}.json"
        tmp = rp.with_name(rp.name + ".tmp")
        tmp.write_text(json.dumps(r, indent=1) + "\n")
        tmp.replace(rp)
        with lock:
            print(f"{path.name}: docs {r['docs']} flagged {r['flagged']} canary hits {r['canary_hits']} "
                  f"({r['parts']} part(s), {r['seconds']} s)", file=sys.stderr)
        return r

    t0 = time.time()
    files = sorted(enumerate(a.files), key=lambda x: -x[1].stat().st_size)  # biggest first
    with ThreadPoolExecutor(a.jobs) as pool, ThreadPoolExecutor(min(len(files), a.jobs)) as file_pool:
        out = list(file_pool.map(lambda x: task(x[0], x[1], pool), files))
    results = sorted((r for r in out if r is not None), key=lambda r: r["file"])
    failed = sorted(p.name for (_, p), r in zip(files, out) if r is None)

    print(table(results))
    sets = sorted({(r["fingerprints_version"], r["set_sha256"], r["scope"]) for r in results})
    for version, sha, scope in sets:
        print(f"\nFingerprint set: {version}, set_sha256 `{sha}`, scope `{scope}`")
    canaries = sorted({r["canaries_checked"] for r in results})
    if canaries:
        print(f"Canaries checked per file: {', '.join(map(str, canaries))}")
    print(f"Total: {len(results)} file(s), {sum(r['docs'] for r in results):,} docs, "
          f"{sum(r['bytes'] for r in results):,} bytes, {sum(r['flagged'] for r in results)} flagged, "
          f"{sum(r['canary_hits'] for r in results)} canary hits")
    for r in results:
        for doc_id in r["flagged_doc_ids"][:20]:
            print(f"flagged: {r['file']} {doc_id}", file=sys.stderr)
        for h in r["canary_hit_list"][:20]:
            print(f"canary: {r['file']} {h['doc_id']} offset {h['offset']}", file=sys.stderr)
    print(f"checked {len(results)} of {len(files)} file(s) in {time.time() - t0:.0f} s with --jobs {a.jobs}"
          + (f"; FAILED: {' '.join(failed)}" if failed else ""), file=sys.stderr)

    other_set = sorted(r["file"] for r in results if r["set_sha256"] != set_sha)
    if other_set:
        print(f"set_sha256 is not {set_sha} for: {' '.join(other_set)}", file=sys.stderr)
    bad = bool(failed) or len(sets) > 1 or bool(other_set)
    bad |= any(r["flagged"] or r["canary_hits"] for r in results)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
