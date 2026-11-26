"""Find the Team 5 code a run folder was trained with, before a one-shot job loads it.

    python capstone/g5_source_resolver.py --g5-root <lab checkout> --manifest <run folder>/manifest.json --work <job dir>

Team 5's train_week2.py records the sha256 of its own sources in manifest.json `sources` (by basename). The adapter
(rl-loop/team5_adapter.py, RUN_FOLDER_SOURCE_FILES and RUN_FOLDER_SOURCE_FILES_IF_RECORDED) refuses the run folder when
those differ from --g5-root. That pin is right: a checkpoint loaded with different code is a silent error. But Team 5's
main keeps moving, so the checkout a TSAI job gets can be later (or earlier) than the commit Team 5 trained at.

This resolver runs once per job, before anything loads the checkpoint:
  * the recorded sources equal --g5-root            -> prints the same root; nothing changes;
  * they differ and --g5-root is a git checkout     -> finds the commit in the same repository (all refs and HEAD) whose
    pinned Team 5 files have exactly the recorded sha256, makes a detached worktree of it under --work, checks the
    hashes there, prints a loud block naming the commit and why, and prints that worktree as the G5 root;
  * no commit matches, or --g5-root is not a git checkout -> exits 2 with a message (under 400 characters) naming the
    files, both hashes and the fix. Full hashes go to <work>/g5_source_resolve.json.
It never relaxes the pin: the adapter still checks the resolved tree, file by file. There is no skip flag.
The only output on stdout is the root to use; everything else goes to stderr.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "rl-loop"))

TEAM5 = "teams/05-architecture"


class SourceResolveError(RuntimeError):
    """The run folder's Team 5 sources cannot be matched to a checkout. The job must not start."""


def pinned_files(sources: dict) -> dict[str, str]:
    """{basename: harness-relative path} exactly as the adapter compares them: the always-pinned files, plus the
    if-recorded ones the manifest recorded. Imported from the adapter so the two can never drift apart."""
    from team5_adapter import RUN_FOLDER_SOURCE_FILES, RUN_FOLDER_SOURCE_FILES_IF_RECORDED  # noqa: PLC0415
    files = dict(RUN_FOLDER_SOURCE_FILES)
    files.update({n: r for n, r in RUN_FOLDER_SOURCE_FILES_IF_RECORDED.items() if n in sources})
    return files


def _sha_file(p: Path) -> str | None:
    if not p.is_file():
        return None
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise SourceResolveError(f"git {' '.join(args)} failed in {root}: {r.stderr.strip()[:200]}")
    return r


def tree_hashes(root: Path, files: dict[str, str]) -> dict[str, str | None]:
    return {n: _sha_file(root / TEAM5 / rel) for n, rel in files.items()}


def _mismatch(want: dict, have: dict) -> list[str]:
    return [n for n in want if have.get(n) != want[n]]


def _short(rows: list[str], want: dict, have: dict) -> str:
    return "; ".join(f"{n} run {str(want.get(n))[:12]} vs tree {str(have.get(n))[:12]}" for n in rows)


def _blob_sha256(repo: Path, blobs: set[str], cache: dict[str, str]) -> None:
    """sha256 of each git blob's bytes (one `git cat-file --batch` for all of them)."""
    todo = sorted(b for b in blobs if b not in cache)
    if not todo:
        return
    p = subprocess.run(["git", "-C", str(repo), "cat-file", "--batch"], input="\n".join(todo).encode() + b"\n",
                       capture_output=True, check=True)
    buf, i = p.stdout, 0
    for b in todo:
        nl = buf.index(b"\n", i)
        size = int(buf[i:nl].split()[2])
        cache[b] = hashlib.sha256(buf[nl + 1:nl + 1 + size]).hexdigest()
        i = nl + 1 + size + 1


def candidate_commits(repo: Path, paths: list[str]) -> list[str]:
    """Commits (newest first, HEAD and every ref) that changed any of `paths`: each distinct version of those files
    first appears at one of them."""
    out = _git(repo, "log", "HEAD", "--all", "--date-order", "--format=%H", "--", *paths).stdout.split()
    return list(dict.fromkeys(out))


def find_commit(repo: Path, files: dict[str, str], want: dict[str, str], extra: dict[str, str]) -> tuple[str | None, dict]:
    """First commit (newest first) whose pinned files hash to `want`. Among matches, one whose other recorded week2 files
    (`extra`, report-only) also match is preferred. Returns (commit or None, facts)."""
    cache: dict[str, str] = {}
    rel_of = {f"{TEAM5}/{rel}": n for n, rel in files.items()}
    rel_of.update({f"{TEAM5}/harness/week2/{n}": n for n in extra})
    commits = candidate_commits(repo, list(rel_of))
    first_pin = None
    for c in commits:
        ls = _git(repo, "ls-tree", "-r", c, "--", *rel_of).stdout.splitlines()
        blobs = {}
        for line in ls:
            meta, path = line.split("\t", 1)
            blobs[rel_of[path]] = meta.split()[2]
        _blob_sha256(repo, set(blobs.values()), cache)
        have = {n: cache.get(blobs.get(n, ""), None) for n in rel_of.values()}
        if _mismatch(want, have):
            continue
        if not _mismatch(extra, have):
            return c, {"candidates_walked": commits.index(c) + 1, "candidates_total": len(commits),
                       "other_recorded_week2_files_match": True}
        first_pin = first_pin or c
    return first_pin, {"candidates_walked": len(commits), "candidates_total": len(commits),
                       "other_recorded_week2_files_match": False if first_pin else None}


def resolve(g5_root: Path, manifest: Path, work: Path) -> Path:
    t0 = time.perf_counter()
    g5_root = Path(g5_root).resolve()
    sources = json.loads(Path(manifest).read_text(encoding="utf-8")).get("sources") or {}
    files = pinned_files(sources)
    want = {n: sources.get(n) for n in files}
    have = tree_hashes(g5_root, files)
    bad = _mismatch(want, have)
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    rec = {"g5_root": str(g5_root), "manifest": str(manifest), "recorded": want, "g5_root_hashes": have,
           "mismatched": bad}

    def done(root: Path, **kw) -> Path:
        rec.update(kw, resolved_root=str(root), seconds=round(time.perf_counter() - t0, 3))
        (work / "g5_source_resolve.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        return root

    def fail(msg: str):
        rec.update(error=msg, seconds=round(time.perf_counter() - t0, 3))
        (work / "g5_source_resolve.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
        raise SourceResolveError(msg)

    if not bad:
        print(f"[g5_source_resolver] --g5-root matches the run's recorded Team 5 sources ({len(files)} files); "
              f"using it as is", file=sys.stderr)
        return done(g5_root, action="none")
    unrecorded = [n for n in bad if want[n] is None]
    if unrecorded:
        fail(f"G5 SOURCE REFUSED: manifest.json records no sha256 for {', '.join(unrecorded)}; the run folder cannot be "
             f"matched to any code. Ask Team 5 for the commit they trained at.")
    top = _git(g5_root, "rev-parse", "--show-toplevel", check=False)
    if top.returncode != 0 or Path(top.stdout.strip()).resolve() != g5_root:
        fail(f"G5 SOURCE REFUSED: --g5-root is not a git checkout, and its Team 5 files differ from the run: "
             f"{_short(bad, want, have)}. Fix: check out the lab commit Team 5 trained at (same {', '.join(bad)}) "
             f"and pass it as G5.")
    extra = {n: h for n, h in sources.items()
             if n not in files and (g5_root / TEAM5 / "harness/week2" / n).is_file() and isinstance(h, str)}
    commit, facts = find_commit(g5_root, files, want, extra)
    if commit is None:
        fail(f"G5 SOURCE REFUSED: no commit in {g5_root.name} has the run's Team 5 files: {_short(bad, want, have)}. "
             f"Fix: fetch Team 5's branch or check out the commit they trained at, and pass it as G5.")
    dest = work / f"g5_source_{commit[:12]}"
    if dest.exists():
        head = _git(dest, "rev-parse", "HEAD", check=False).stdout.strip()
        if head != commit:
            fail(f"G5 SOURCE REFUSED: {dest} exists but is at {head[:12] or 'no commit'}, not {commit[:12]}. "
                 f"Remove it and rerun.")
    else:
        _git(g5_root, "worktree", "add", "--detach", str(dest), commit)
    got = tree_hashes(dest, files)
    if _mismatch(want, got) or _git(dest, "status", "--porcelain", "--", TEAM5).stdout.strip():
        fail(f"G5 SOURCE REFUSED: resolved worktree {dest} does not hold the recorded files: "
             f"{_short(_mismatch(want, got), want, got)}")
    subj = _git(g5_root, "log", "-1", "--format=%h %ci %s", commit).stdout.strip()
    bar = "=" * 100
    print("\n".join([
        bar,
        "G5 SOURCE RESOLVED: the run folder was trained with different Team 5 code than this checkout.",
        f"  checkout : {g5_root} at {_git(g5_root, 'rev-parse', '--short=12', 'HEAD').stdout.strip()}",
        f"  differs  : {_short(bad, want, have)}",
        f"  using    : {commit}  ({subj})",
        f"  worktree : {dest}",
        f"  why      : its {len(files)} pinned files ({', '.join(files)}) have exactly the sha256 in the run's "
        f"manifest.json; the adapter's pin is still checked against it",
        f"  other recorded week2 files match too: {facts['other_recorded_week2_files_match']}; "
        f"commits walked {facts['candidates_walked']} of {facts['candidates_total']}",
        bar]), file=sys.stderr)
    return done(dest, action="resolved", commit=commit, commit_subject=subj, **facts)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--g5-root", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True, help="the run folder's manifest.json")
    ap.add_argument("--work", type=Path, required=True, help="job work folder; a resolved worktree is made under it")
    a = ap.parse_args(argv)
    try:
        root = resolve(a.g5_root, a.manifest, a.work)
    except SourceResolveError as e:
        print(str(e), file=sys.stderr)
        return 2
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
