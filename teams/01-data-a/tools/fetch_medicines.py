#!/usr/bin/env python3
"""Collect the current medicine.md files of all annotators into one folder.

Usage:
    python -I fetch_medicines.py [--include-open-prs] [--pack-id <pack_id> ...] [--out <dir>]

Like build_gold.py (which does this for labels.csv), it reads diagnosis/submissions/<pack_id>/<handle>/medicine.md
from the local checkout and, with --include-open-prs, from the open pull requests of the repo. For each pack and
handle the newest version wins: an open PR is newer than the local file, and a later-updated PR is newer than an
earlier one. The files are only saved as text. Nothing in them is run.

The files go to <out>/<pack_id>/<handle>/medicine.md. The default <out> is teams/01-data-a/medicines/submissions.
Keep the folder name "submissions": medicine_extract.py reads the pack and handle from that path.
Then run the extractor on each file:
    python -I medicine_extract.py <out>/<pack_id>/<handle>/medicine.md

Pack selection: the packs from --pack-id, else all packs with the Data A prefixes of build_gold.py, else (with
--all-packs) every pack.

--include-open-prs needs GH_TOKEN or GITHUB_TOKEN, or `gh auth login` (the same as build_gold.py).
"""
from __future__ import annotations

import argparse
import base64
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # python -I does not put the script folder on the path
from build_gold import DEFAULT_PREFIXES, GithubAPI, resolve_token  # noqa: E402  (the same GitHub client)

# Letters, digits, hyphen and underscore only: no dots, so "." and ".." can never be a folder name.
NAME = r"[A-Za-z0-9][A-Za-z0-9_-]*"
MEDICINE_PATTERN = re.compile(rf"^diagnosis/submissions/(?P<pack>{NAME})/(?P<handle>{NAME})/medicine\.md$", re.I)
MAX_BYTES = 5 * 1024 * 1024  # the same limit as pr-guard


@dataclass(frozen=True)
class Medicine:
    pack_id: str
    handle: str
    source: str
    text: str
    priority: Tuple[str, int]  # PR update time and number; the local file is ("", 0), so any PR wins


def decode_markdown(raw: bytes, source: str, warnings: List[str]) -> Optional[str]:
    """UTF-8 if possible. Otherwise Windows-1252 or Latin-1, with a warning. None if the file is unusable."""
    if not raw:
        warnings.append(f"Skipping {source}: empty file")
        return None
    if len(raw) > MAX_BYTES:
        warnings.append(f"Skipping {source}: larger than 5 MB")
        return None
    if b"\x00" in raw:
        warnings.append(f"Skipping {source}: NUL bytes, not a text file")
        return None
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        for encoding in ("cp1252", "latin-1"):
            try:
                text = raw.decode(encoding)
            except UnicodeDecodeError:
                continue
            warnings.append(f"{source}: not UTF-8 (first bad byte at offset {exc.start}); decoded as {encoding}")
            return text
    return None  # unreachable: latin-1 decodes every byte


def local_medicines(repo_root: Path, warnings: List[str]) -> List[Medicine]:
    root = repo_root / "diagnosis" / "submissions"
    if not root.is_dir():
        raise FileNotFoundError(f"Missing directory: {root}")
    found = []
    for path in sorted(root.glob("*/*/medicine.md")):
        rel = path.relative_to(repo_root).as_posix()
        match = MEDICINE_PATTERN.fullmatch(rel)
        if not match:
            warnings.append(f"Skipping main/local: {rel}: unusual folder name")
            continue
        source = f"main/local: {rel}"
        text = decode_markdown(path.read_bytes(), source, warnings)
        if text is not None:
            found.append(Medicine(match.group("pack"), match.group("handle"), source, text, ("", 0)))
    return found


def pr_medicines(api, repo: str, warnings: List[str]) -> List[Medicine]:
    """medicine.md files of the open PRs. The blob of each file is read, not the diff."""
    found, blob_cache = [], {}
    for pr in api.paged(f"repos/{repo}/pulls?state=open"):
        if pr.get("base", {}).get("ref") != "main":
            continue
        number, updated = int(pr["number"]), pr.get("updated_at", "")
        for info in api.paged(f"repos/{repo}/pulls/{number}/files"):
            path = info.get("filename", "")
            match = MEDICINE_PATTERN.fullmatch(path)
            if not match or info.get("status") == "removed":
                continue
            sha = info.get("sha")
            if not sha:
                raise RuntimeError(f"PR #{number}: missing blob SHA for {path}")
            source = f"PR #{number}: {path}"
            if sha not in blob_cache:
                try:
                    blob = api.get(f"repos/{repo}/git/blobs/{sha}")
                except RuntimeError:
                    head_repo = (pr.get("head", {}).get("repo") or {}).get("full_name")  # an unmerged fork object
                    if not head_repo or head_repo == repo:
                        raise
                    blob = api.get(f"repos/{head_repo}/git/blobs/{sha}")
                try:
                    if blob.get("encoding") != "base64":
                        raise ValueError("unsupported GitHub blob encoding")
                    raw = base64.b64decode("".join(blob["content"].split()), validate=True)
                    blob_cache[sha] = decode_markdown(raw, source, warnings)
                except (ValueError, KeyError) as exc:
                    warnings.append(f"Skipping {source}: {exc}")
                    blob_cache[sha] = None
            if blob_cache[sha] is not None:
                found.append(Medicine(match.group("pack"), match.group("handle"), source, blob_cache[sha], (updated, number)))
    return found


def choose_current(items: Sequence[Medicine], warnings: List[str]) -> List[Medicine]:
    """One medicine.md for each pack and handle: the newest."""
    grouped: Dict[Tuple[str, str], List[Medicine]] = defaultdict(list)
    for m in items:
        grouped[(m.pack_id.casefold(), m.handle.casefold())].append(m)
    chosen = []
    for _, files in sorted(grouped.items()):
        ordered = sorted(files, key=lambda m: (m.priority, m.source))
        win = ordered[-1]
        if len({m.text for m in ordered}) > 1:
            warnings.append(f"{win.pack_id}/{win.handle}: {len(files)} versions of medicine.md; using the newest, {win.source}")
        chosen.append(win)
    return chosen


def write_all(chosen: Sequence[Medicine], out: Path) -> List[Tuple[Medicine, Path, str]]:
    """Save each file as <out>/<pack>/<handle>/medicine.md. Returns (medicine, path, 'new'|'updated'|'unchanged')."""
    results = []
    root = out.resolve()
    for m in chosen:
        path = (root / m.pack_id / m.handle / "medicine.md").resolve()
        if root not in path.parents:  # the name pattern should make this impossible
            raise RuntimeError(f"{m.source}: path leaves the output folder")
        old = path.read_bytes() if path.exists() else None
        new = m.text.encode("utf-8")
        state = "new" if old is None else "unchanged" if old == new else "updated"
        if state != "unchanged":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(new)
        results.append((m, path, state))
    return results


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-root", type=Path, default=HERE.parents[2], help="local era-v5-lab checkout (default: inferred)")
    p.add_argument("--repo", default="theschoolofai/era-v5-lab", help="GitHub owner/repo")
    p.add_argument("--include-open-prs", action="store_true", help="also read the open pull requests (needs a GitHub token)")
    p.add_argument("--pack-id", action="append", help="only this pack (can be given more than once)")
    p.add_argument("--all-packs", action="store_true", help="every pack, not only the Data A prefixes")
    p.add_argument("--pack-prefixes", default=",".join(DEFAULT_PREFIXES), help="packs to take when --pack-id is not given")
    p.add_argument("--out", type=Path, help="output folder (default: teams/01-data-a/medicines/submissions in the repo)")
    return p


def main(argv=None, api=None) -> int:
    args = make_parser().parse_args(argv)
    warnings: List[str] = []
    try:
        root = args.repo_root.resolve()
        items = local_medicines(root, warnings)
        if args.include_open_prs:
            if api is None:
                token = resolve_token()
                if not token:
                    raise RuntimeError("To include open PRs, set GH_TOKEN / GITHUB_TOKEN or run 'gh auth login'.")
                api = GithubAPI(token)
            remote = pr_medicines(api, args.repo, warnings)
            print(f"Read {len(remote)} medicine.md files from open PRs; {len(items)} from the local checkout")
            items.extend(remote)
        if args.pack_id:
            wanted = {p.casefold() for p in args.pack_id}
            keep = lambda m: m.pack_id.casefold() in wanted  # noqa: E731
        elif args.all_packs:
            keep = lambda m: True  # noqa: E731
        else:
            prefixes = tuple(p.strip().lower() for p in args.pack_prefixes.split(",") if p.strip())
            keep = lambda m: m.pack_id.lower().startswith(prefixes)  # noqa: E731
        chosen = choose_current([m for m in items if keep(m)], warnings)
        out = args.out or root / "teams" / "01-data-a" / "medicines" / "submissions"
        results = write_all(chosen, out)
    except (FileNotFoundError, RuntimeError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if not results:
        warnings.append("No medicine.md found for the selected packs. Check --pack-id or --pack-prefixes.")
    print(f"\nSaved {len(results)} medicine.md file(s) to {out}")
    if results:
        print(f"{'pack_id':28} {'handle':22} {'state':9} source")
        print("-" * 78)
    for m, _, state in results:
        print(f"{m.pack_id[:28]:28} {m.handle[:22]:22} {state:9} {m.source.split(':')[0]}")
    if results:
        print("\nNext, for each file: python -I teams/01-data-a/harmtest/medicine_extract.py <file>")
    if warnings:
        print(f"\n{len(warnings)} warning(s):", file=sys.stderr)
        for w in warnings[:60]:
            print(f"  - {w}", file=sys.stderr)
        if len(warnings) > 60:
            print(f"  ... {len(warnings) - 60} more", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
