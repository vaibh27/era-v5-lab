"""capstone/g5_source_resolver.py on a small git repository shaped like the lab (CPU, no model, about a second).

(a) run trained at an older commit, checkout at a newer one -> that commit's worktree, pinned hashes equal the record
(b) checkout already matches -> the same root, nothing made
(c) recorded sources match no commit -> refused, message under 400 characters with the file, both hashes and the fix
(d) not a git checkout -> refused, same message rules
plus: an if-recorded file (kas_v2.py) takes part in the match; a recorded-but-missing always-pinned file is refused.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

fixture = pytest.fixture  # bare `@pytest.fixture` reads as an e-mail address to pr-guard

sys.path.insert(0, str(Path(__file__).resolve().parent))
import g5_source_resolver as gr  # noqa: E402

FILES = {"harness/train_week2.py": "print('v1')\n", "harness/week2/model.py": "M = 1\n",
         "harness/week2/config.py": "C = 1\n", "harness/week2/artifacts.py": "A = 1\n",
         "harness/week2/kas_v2.py": "K = 1\n", "harness/week2/optim.py": "O = 1\n"}


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def _write(repo: Path, files: dict) -> None:
    for rel, text in files.items():
        p = repo / "teams/05-architecture" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)


def _commit(repo: Path, msg: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def _manifest(tmp: Path, files: dict, name="manifest.json") -> Path:
    m = tmp / name
    m.write_text(json.dumps({"sources": {Path(rel).name: _sha(t) for rel, t in files.items()}}))
    return m


@fixture
def repo(tmp_path):
    r = tmp_path / "lab"
    r.mkdir()
    _git(r, "init", "-q")
    _write(r, FILES)
    (r / "teams/10-post-training").mkdir(parents=True)
    (r / "teams/10-post-training/x.py").write_text("ours\n")
    c1 = _commit(r, "v1")
    v2 = {**FILES, "harness/train_week2.py": "print('v2 diagnostics')\n"}
    _write(r, v2)
    c2 = _commit(r, "v2")
    (r / "teams/10-post-training/x.py").write_text("ours later\n")
    c3 = _commit(r, "unrelated")
    return r, c1, c2, c3, v2


def test_a_older_run_resolves_to_its_commit(repo, tmp_path):
    r, c1, _c2, _c3, _v2 = repo
    root = gr.resolve(r, _manifest(tmp_path, FILES), tmp_path / "work")
    assert root != r.resolve() and root.parent == tmp_path / "work"
    assert _git(root, "rev-parse", "HEAD") == c1
    for rel, text in FILES.items():
        assert gr._sha_file(root / "teams/05-architecture" / rel) == _sha(text)
    rec = json.loads((tmp_path / "work/g5_source_resolve.json").read_text())
    assert rec["action"] == "resolved" and rec["mismatched"] == ["train_week2.py"] and rec["commit"] == c1
    # a second call in the same job reuses the worktree
    assert gr.resolve(r, _manifest(tmp_path, FILES), tmp_path / "work") == root


def test_b_matching_checkout_is_used_as_is(repo, tmp_path):
    r, _c1, _c2, _c3, v2 = repo
    root = gr.resolve(r, _manifest(tmp_path, v2), tmp_path / "work")
    assert root == r.resolve()
    assert not list((tmp_path / "work").glob("g5_source_[0-9a-f]*"))
    assert json.loads((tmp_path / "work/g5_source_resolve.json").read_text())["action"] == "none"


def test_c_no_commit_matches_is_refused(repo, tmp_path):
    r, *_ = repo
    bad = {**FILES, "harness/train_week2.py": "print('a local edit')\n"}
    with pytest.raises(gr.SourceResolveError) as e:
        gr.resolve(r, _manifest(tmp_path, bad), tmp_path / "work")
    msg = str(e.value)
    assert len(msg) < 400, len(msg)
    assert "train_week2.py" in msg and _sha(bad["harness/train_week2.py"])[:12] in msg
    assert _sha("print('v2 diagnostics')\n")[:12] in msg and "Fix:" in msg
    assert not list((tmp_path / "work").glob("g5_source_[0-9a-f]*"))


def test_d_not_a_git_checkout_is_refused(repo, tmp_path):
    r, *_ = repo
    plain = tmp_path / "bundle"
    shutil.copytree(r / "teams", plain / "teams")
    with pytest.raises(gr.SourceResolveError) as e:
        gr.resolve(plain, _manifest(tmp_path, FILES), tmp_path / "work")
    msg = str(e.value)
    assert len(msg) < 400, len(msg)
    assert "not a git checkout" in msg and "train_week2.py" in msg and "Fix:" in msg
    assert _sha(FILES["harness/train_week2.py"])[:12] in msg


def test_d_matching_bundle_without_git_passes(repo, tmp_path):
    r, _c1, _c2, _c3, v2 = repo
    plain = tmp_path / "bundle"
    shutil.copytree(r / "teams", plain / "teams")
    assert gr.resolve(plain, _manifest(tmp_path, v2), tmp_path / "work") == plain.resolve()


def test_if_recorded_file_takes_part(repo, tmp_path):
    r, *_ = repo
    # train_week2 of v1 but a kas_v2.py no commit had: refused on kas_v2.py, not resolved to c1
    odd = {**FILES, "harness/week2/kas_v2.py": "K = 2\n"}
    with pytest.raises(gr.SourceResolveError, match="kas_v2.py"):
        gr.resolve(r, _manifest(tmp_path, odd), tmp_path / "work")


def test_unrecorded_pinned_file_is_refused(repo, tmp_path):
    r, *_ = repo
    m = tmp_path / "m.json"
    m.write_text(json.dumps({"sources": {"train_week2.py": _sha(FILES["harness/train_week2.py"])}}))
    with pytest.raises(gr.SourceResolveError, match="records no sha256"):
        gr.resolve(r, m, tmp_path / "work")


def test_pinned_set_is_the_adapters(tmp_path):
    from team5_adapter import RUN_FOLDER_SOURCE_FILES, RUN_FOLDER_SOURCE_FILES_IF_RECORDED
    got = gr.pinned_files({"kas_v2.py": "x"})
    assert set(got) == set(RUN_FOLDER_SOURCE_FILES) | {"kas_v2.py"}
    assert set(gr.pinned_files({})) == set(RUN_FOLDER_SOURCE_FILES)
    assert set(gr.pinned_files(dict.fromkeys(RUN_FOLDER_SOURCE_FILES_IF_RECORDED, "x"))) == (
        set(RUN_FOLDER_SOURCE_FILES) | set(RUN_FOLDER_SOURCE_FILES_IF_RECORDED))


def test_cli_stdout_is_only_the_root(repo, tmp_path, capsys):
    r, *_ = repo
    assert gr.main(["--g5-root", str(r), "--manifest", str(_manifest(tmp_path, FILES)), "--work",
                    str(tmp_path / "w")]) == 0
    out, err = capsys.readouterr()
    assert out.strip().endswith(next((tmp_path / "w").glob("g5_source_[0-9a-f]*")).name) and "G5 SOURCE RESOLVED" in err
    bad = {**FILES, "harness/train_week2.py": "zzz\n"}
    assert gr.main(["--g5-root", str(r), "--manifest", str(_manifest(tmp_path, bad, "b.json")), "--work",
                    str(tmp_path / "w2")]) == 2
    out, err = capsys.readouterr()
    assert out == "" and "G5 SOURCE REFUSED" in err
