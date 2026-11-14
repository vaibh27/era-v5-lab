"""Tests for fetch_medicines.py. Run: python -I teams/01-data-a/tests/tools/test_fetch_medicines.py"""
import base64
import contextlib
import io
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "tools"))  # python -I: not on the path by itself
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "harmtest"))  # medicine_extract lives here now
import fetch_medicines  # noqa: E402


class FakeAPI:
    """A tiny stand-in for build_gold.GithubAPI. prs is a list of (number, updated_at, {path: (status, bytes)})."""

    def __init__(self, prs, fork_only=()):
        self.prs, self.blobs, self.fork_only = prs, {}, set(fork_only)

    def paged(self, endpoint):
        if endpoint.endswith("pulls?state=open"):
            yield from ({"number": n, "updated_at": u, "base": {"ref": "main"},
                         "head": {"repo": {"full_name": "student/era-v5-lab"}}} for n, u, _ in self.prs)
        else:
            number = int(endpoint.split("/pulls/")[1].split("/")[0])
            for n, _, files in self.prs:
                if n == number:
                    for path, (status, raw) in files.items():
                        sha = f"sha{n}{abs(hash(path)) % 10**6}"
                        self.blobs[sha] = raw
                        yield {"filename": path, "status": status, "sha": sha}

    def get(self, endpoint):
        sha = endpoint.rsplit("/", 1)[1]
        if sha in self.fork_only and "student/" not in endpoint:
            raise RuntimeError("404")
        return {"encoding": "base64", "content": base64.b64encode(self.blobs[sha]).decode()}


class FetchCase(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.out = self.root / "out" / "submissions"

    def local(self, pack, handle, text):
        p = self.root / "diagnosis" / "submissions" / pack / handle / "medicine.md"
        p.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(text, bytes):
            p.write_bytes(text)
        else:
            p.write_text(text)

    def run_fetch(self, *extra, api=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = fetch_medicines.main(["--repo-root", str(self.root), "--out", str(self.out), *extra], api=api)
        return rc, out.getvalue(), err.getvalue()

    def saved(self, pack, handle):
        p = self.out / pack / handle / "medicine.md"
        return p.read_text() if p.exists() else None

    def test_local_files_are_copied_with_the_same_layout(self):
        self.local("code-rust", "alice", "alice v1")
        rc, out, _ = self.run_fetch()
        self.assertEqual(rc, 0)
        self.assertEqual(self.saved("code-rust", "alice"), "alice v1")
        self.assertIn("new", out)

    def test_rerun_says_unchanged_then_updated(self):
        self.local("code-rust", "alice", "v1")
        self.run_fetch()
        _, out, _ = self.run_fetch()
        self.assertIn("unchanged", out)
        self.local("code-rust", "alice", "v2")
        _, out, _ = self.run_fetch()
        self.assertIn("updated", out)
        self.assertEqual(self.saved("code-rust", "alice"), "v2")

    def test_default_prefixes_and_pack_filter(self):
        self.local("code-rust", "alice", "x")
        self.local("ind-hi-fw2", "bob", "y")           # Indic: not in the Data A prefixes
        self.run_fetch()
        self.assertIsNotNone(self.saved("code-rust", "alice"))
        self.assertIsNone(self.saved("ind-hi-fw2", "bob"))
        self.run_fetch("--pack-id", "ind-hi-fw2")
        self.assertIsNotNone(self.saved("ind-hi-fw2", "bob"))
        self.run_fetch("--all-packs")
        self.assertIsNotNone(self.saved("ind-hi-fw2", "bob"))

    def test_open_pr_beats_the_local_file_and_the_later_pr_beats_the_earlier(self):
        self.local("code-rust", "alice", "local")
        self.local("code-rust", "bob", "local bob")
        path = "diagnosis/submissions/code-rust/alice/medicine.md"
        api = FakeAPI([(7, "2026-10-26T10:00:00Z", {path: ("modified", b"pr 7")}),
                       (9, "2026-10-27T10:00:00Z", {path: ("modified", b"pr 9")})])
        rc, out, err = self.run_fetch("--include-open-prs", api=api)
        self.assertEqual(rc, 0)
        self.assertEqual(self.saved("code-rust", "alice"), "pr 9")
        self.assertEqual(self.saved("code-rust", "bob"), "local bob")      # no PR: the local file stays
        self.assertIn("versions of medicine.md; using the newest, PR #9", err)

    def test_files_of_pr_only_authors_are_collected_and_other_files_ignored(self):
        api = FakeAPI([(3, "2026-10-26T10:00:00Z", {
            "diagnosis/submissions/code-go/carol/medicine.md": ("added", b"carol"),
            "diagnosis/submissions/code-go/carol/labels.csv": ("added", b"x"),
            "diagnosis/submissions/code-go/dave/medicine.md": ("removed", b"gone"),
            "teams/01-data-a/README.md": ("modified", b"x")})])
        (self.root / "diagnosis" / "submissions").mkdir(parents=True)
        self.run_fetch("--include-open-prs", api=api)
        self.assertEqual(self.saved("code-go", "carol"), "carol")
        self.assertIsNone(self.saved("code-go", "dave"))

    def test_unsafe_names_never_leave_the_output_folder(self):
        (self.root / "diagnosis" / "submissions").mkdir(parents=True)
        api = FakeAPI([(1, "2026-10-26T10:00:00Z", {
            "diagnosis/submissions/code-go/../evil/medicine.md": ("added", b"x"),
            "diagnosis/submissions/..\\x/y/medicine.md": ("added", b"x"),
            "diagnosis/submissions/code-go/.hidden/medicine.md": ("added", b"x")})])
        self.run_fetch("--include-open-prs", api=api)
        self.assertFalse(any(self.root.rglob("evil")))
        self.assertEqual(list(self.out.rglob("medicine.md")) if self.out.exists() else [], [])

    def test_a_pr_file_that_is_not_readable_from_the_base_repo_is_read_from_the_fork(self):
        (self.root / "diagnosis" / "submissions").mkdir(parents=True)
        path = "diagnosis/submissions/code-go/carol/medicine.md"
        api = FakeAPI([(5, "2026-10-26T10:00:00Z", {path: ("added", b"from fork")})])
        list(api.paged("repos/x/pulls/5/files"))       # register the blob
        api.fork_only = set(api.blobs)
        rc, _, _ = self.run_fetch("--include-open-prs", api=api)
        self.assertEqual((rc, self.saved("code-go", "carol")), (0, "from fork"))

    def test_encodings(self):
        self.local("code-rust", "alice", "﻿utf-8 with a BOM".encode("utf-8"))
        self.local("code-rust", "bob", "caf\xe9".encode("cp1252"))
        self.local("code-rust", "carol", b"binary\x00data")
        self.local("code-rust", "dave", b"")
        _, _, err = self.run_fetch()
        self.assertEqual(self.saved("code-rust", "alice"), "utf-8 with a BOM")     # the BOM is dropped
        self.assertEqual(self.saved("code-rust", "bob"), "caf\xe9")
        self.assertIn("decoded as cp1252", err)
        self.assertIsNone(self.saved("code-rust", "carol"))
        self.assertIsNone(self.saved("code-rust", "dave"))
        self.assertIn("NUL bytes", err)

    def test_open_prs_need_a_token(self):
        (self.root / "diagnosis" / "submissions").mkdir(parents=True)
        fetch_medicines.resolve_token = lambda: None
        rc, _, err = self.run_fetch("--include-open-prs")
        self.assertEqual(rc, 1)
        self.assertIn("GH_TOKEN", err)

    def test_missing_submissions_folder(self):
        rc, _, err = self.run_fetch()
        self.assertEqual(rc, 1)
        self.assertIn("Missing directory", err)

    def test_the_saved_path_works_with_the_extractor(self):
        import medicine_extract
        self.local("code-rust", "alice", "**Disease:** B3, x\n```python\ndef cure_b3(t):\n    return t\n```\n")
        self.run_fetch()
        self.assertEqual(medicine_extract.origin(self.out / "code-rust" / "alice" / "medicine.md"), ("code-rust", "alice"))


if __name__ == "__main__":
    unittest.main()
