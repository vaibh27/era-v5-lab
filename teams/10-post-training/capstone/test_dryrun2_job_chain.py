"""dryrun2_job_chain.sh's held-out forgetting step, and both dry run 2 job commands against the scripts' own parsers.

    python -m pytest capstone/test_dryrun2_job_chain.py -q

The chain runs in bash with a stand-in for python: it writes what each step would leave behind and records the calls, so
only the shell logic is under test. The scripts themselves have their own tests.
"""
from __future__ import annotations

import json
import os
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
JOBS = HERE.parent / "jobs"
sys.path.insert(0, str(HERE))

STUB = r'''#!__PY__
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
if args[0] == "-":  # the chain's own inline arm check
    sys.argv = args
    exec(compile(sys.stdin.read(), "arm-checks", "exec"), {"__name__": "__main__"})
    sys.exit(0)
name = Path(args[0]).name
with open(os.environ["CALLS"], "a") as f:
    f.write(json.dumps([name, *args[1:]]) + "\n")
def flag(k):
    return args[args.index(k) + 1] if k in args else None
if name == "g5_source_resolver.py":
    print(flag("--g5-root"))
elif name == "make_probe_file.py":
    Path(flag("--out")).write_text("{}")
elif name == "dryrun2_dt_compare.py":
    out = Path(flag("--out"))
    for arm in os.environ["ARMS_DONE"].split():
        (out / f"sft_{arm}").mkdir(parents=True, exist_ok=True)
        (out / f"sft_{arm}" / "sft_report.json").write_text(json.dumps({"router_bias_unchanged": {"pass": True}}))
        if flag("--keep-weights-dir"):
            d = Path(flag("--keep-weights-dir")) / f"g10-dr2-{arm}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "latest.json").write_text("{}")
    (out / "adapter.yaml").write_text("{}")
    sys.exit(int(os.environ.get("WRAPPER_RC", "0")))
elif name == "forget_bpb_kept.py":
    sys.exit(int(os.environ.get("FORGET_RC", "0")))
'''


def _chain(tmp_path, extra, arms="full lora dt", **env):
    stub = tmp_path / "python"
    stub.write_text(STUB.replace("__PY__", sys.executable))
    stub.chmod(0o755)
    (tmp_path / "t5").mkdir()
    for n in ("manifest.json", "checkpoint.pt"):
        (tmp_path / "t5" / n).write_text("{}")
    full = {**os.environ, "G5": str(HERE.parents[2]), "T5_OUT": "t5", "TOKENIZER": "tok", "TABLE": "table",
            "HELDOUT": "held.jsonl", "HELDOUT_SOURCES": "src.jsonl", "WORK": "work/", "PY": str(stub), "DEVICE": "cpu",
            "CALLS": str(tmp_path / "calls"), "ARMS_DONE": arms, **env}
    r = subprocess.run(["bash", str(HERE / "dryrun2_job_chain.sh"), "--", *extra, "--out", "out/"], cwd=tmp_path,
                       env=full, capture_output=True, text=True)
    calls = [json.loads(x) for x in (tmp_path / "calls").read_text().splitlines()]
    return r, [c for c in calls if c[0] == "forget_bpb_kept.py"]


pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
KEEP = ["--keep-weights-dir", "keep_t7/", "--keep-weights-arms", "full,lora,dt"]


def test_the_chain_measures_the_base_and_every_kept_arm_after_the_wrapper(tmp_path):
    r, forget = _chain(tmp_path, KEEP)
    assert r.returncode == 0, r.stdout + r.stderr
    assert len(forget) == 1
    a = forget[0]
    assert [a[i + 1] for i, x in enumerate(a) if x == "--kept"] == [f"keep_t7//g10-dr2-{n}" for n in ("full", "lora", "dt")]
    for flag, value in (("--adapter-config", "out//adapter.yaml"), ("--heldout", "held.jsonl"),
                        ("--heldout-sources", "src.jsonl"), ("--dtype", "auto"), ("--device", "cpu"),
                        ("--out-dir", "out//forget")):
        assert a[a.index(flag) + 1] == value


def test_a_failed_measurement_keeps_the_jobs_exit_code(tmp_path):
    r, forget = _chain(tmp_path, KEEP, FORGET_RC="1")
    assert r.returncode == 0 and len(forget) == 1 and "report only" in r.stdout


def test_only_arms_that_were_kept_are_measured_and_the_wrappers_failure_stays(tmp_path):
    r, forget = _chain(tmp_path, KEEP, arms="full", WRAPPER_RC="1")
    assert r.returncode == 1
    assert [x for x in forget[0] if "g10-dr2-" in x] == ["keep_t7//g10-dr2-full"]


def test_the_default_arm_is_the_wrappers_and_no_keep_dir_means_no_measurement(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, forget = _chain(tmp_path / "a", ["--keep-weights-dir", "keep_t7"])
    assert [x for x in forget[0] if "g10-dr2-" in x] == ["keep_t7/g10-dr2-dt"]
    r, forget = _chain(tmp_path / "b", [])
    assert r.returncode == 0 and forget == []


def test_no_kept_folder_is_logged_and_not_measured(tmp_path):
    r, forget = _chain(tmp_path, KEEP, arms="", WRAPPER_RC="1")
    assert r.returncode != 0 and forget == [] and "skipped" in r.stdout


def _command(name: str) -> list[str]:
    line = re.search(r"^Command:\s+(.*)$", (JOBS / name).read_text(encoding="utf-8"), re.M).group(1)
    return shlex.split(re.sub(r"<([A-Z0-9_]+)>", r"/staged/\1", line))


def test_the_comparison_command_parses_and_keeps_all_three_arms():
    import dryrun2_dt_compare as dc  # noqa: PLC0415
    words = _command("2026-11-26-g10-dryrun2-dt-vs-fft.md")
    rest = words[words.index("--") + 1:]
    a = dc.parse_args(["--g5-root", "g", "--run-folder", "r", "--tokenizer", "t", "--byte-table", "b", *rest])
    assert a.slice_arm == "a4" and a.keep_weights_arms == "full,lora,dt" and str(a.keep_weights_dir) == "keep_t7"
    assert str(a.keep_canary_state_dir) == "keep_canary" and len(a.packs) == 4 and not a.no_pack_check


def test_the_rl_command_parses_keeps_the_final_weights_and_reads_the_comparisons_export():
    import rl_job_v1 as job  # noqa: PLC0415
    words = _command("2026-11-17-dt-then-rl-code.md")
    a = job.parse_args(words[words.index("teams/10-post-training/capstone/rl_job_v1.py") + 1:])
    assert str(a.keep_weights_dir) == "keep_t7" and str(a.start_weights) == "keep_canary/dt/model.safetensors"
    assert a.stand_in_reward is None and not a.no_pack_check and a.origin == "g5_delivery"
