"""Reward for stdin/stdout programming tasks (Codeforces-style), with the same rule as rl/verifier.py:
the expected outputs never enter the sandbox.

A task is a dict: {"id", "statement", "tests": [{"input": str, "output": str}, ...], "time_limit_s", "source",
"licence", "generator"}. The sandbox gets the program and the INPUTS only, runs the program once per input, and
returns what it printed. This process compares the outputs (whitespace-separated tokens, like most judges), so a
program cannot write a fake "all passed" result: it never sees what the right answers are.

    score(task, completion) -> {"reward": 0|1, "passed": n, "total": m, "reason": str}

Code is taken from the completion with rl/verifier.py's extract_code (last python block). The sandbox is a fresh
`python:3.12-slim` container per call with the same limits as the verifier's docker sandbox.
"""
from __future__ import annotations

import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

HERE = Path(__file__).resolve().parent
IMAGE = os.environ.get("G10_RL_IMAGE", "python:3.12-slim")
MAX_TESTS = 20            # per task, for speed; the first ones are the public examples
MAX_OUTPUT_CHARS = 65536  # per test

_spec = importlib.util.spec_from_file_location("g10_rl_verifier_for_stdio", HERE.parent / "rl" / "verifier.py")
_verifier = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_verifier)
extract_code = _verifier.extract_code

# Runs inside the container: reads inputs.json, runs solution.py on each input, writes outputs.json.
RUNNER = r'''
import json, subprocess, sys
work = sys.argv[2] if len(sys.argv) > 2 else "/work"
inputs = json.load(open(work + "/inputs.json"))
limit = float(sys.argv[1])
out = []
for text in inputs:
    try:
        p = subprocess.run([sys.executable, "-B", "-s", work + "/solution.py"], input=text, capture_output=True,
                           text=True, timeout=limit, cwd=work)
        out.append({"status": "ok" if p.returncode == 0 else "error", "stdout": p.stdout[:%d]})
    except subprocess.TimeoutExpired:
        out.append({"status": "timeout", "stdout": ""})
    if out[-1]["status"] != "ok":
        break  # the reward needs every test; stop at the first failure
json.dump(out, open(work + "/outputs.json", "w"))
''' % MAX_OUTPUT_CHARS

DOCKER_FLAGS = ["--network=none", "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "32", "--memory", "512m", "--memory-swap", "512m", "--cpus", "1",
                "--ulimit", "fsize=10000000:10000000", "--ulimit", "core=0",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=16m", "--user", "65534:65534", "-e", "PYTHONHASHSEED=0"]


# Where programs run (G10_STDIO_RUNNER):
#   docker  (default) a fresh python:3.12-slim container per call with DOCKER_FLAGS: the laptop and the job runner.
#   local   a plain subprocess in a temp folder with memory, file-size and core limits and the same per-test
#           timeouts, for places that have no Docker, i.e. inside a Modal container. Use it only where the
#           whole container is disposable and its network is blocked (the Modal survey function sets
#           block_network=True); it is not a sandbox on a machine you care about.
RUNNER_KIND = os.environ.get("G10_STDIO_RUNNER", "docker")
LOCAL_MEMORY_BYTES = 1 << 30   # per process (Python itself needs ~30 MB; the runner and the solution each get this)
LOCAL_FILE_BYTES = 10_000_000


def _local_limits():  # runs in the child before exec (Linux)
    import resource  # noqa: PLC0415
    resource.setrlimit(resource.RLIMIT_AS, (LOCAL_MEMORY_BYTES, LOCAL_MEMORY_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (LOCAL_FILE_BYTES, LOCAL_FILE_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _run_local(w: Path, time_limit_s: float, n_inputs: int):
    cmd = [sys.executable, "-B", "-s", str(w / "run.py"), str(time_limit_s), str(w)]
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONHASHSEED": "0", "HOME": str(w)}
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, cwd=w, env=env,
                            preexec_fn=_local_limits if sys.platform.startswith("linux") else None,
                            start_new_session=True)
    try:
        proc.communicate(timeout=time_limit_s * n_inputs + 30)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.communicate()
        return None
    return proc.returncode


def same_output(actual: str, expected: str) -> bool:
    """Judge-style comparison: the same whitespace-separated tokens."""
    return actual.split() == expected.split()


def run_program(code: str, inputs: list[str], time_limit_s: float) -> list[dict] | str:
    """Run code on each input in a fresh container. Returns per-input results, or an error string."""
    with tempfile.TemporaryDirectory(prefix="g10stdio-", ignore_cleanup_errors=True) as tmp:
        w = Path(tmp)
        os.chmod(w, 0o777)
        (w / "solution.py").write_text(code, encoding="utf-8")
        (w / "inputs.json").write_text(json.dumps(inputs), encoding="utf-8")
        (w / "run.py").write_text(RUNNER, encoding="utf-8")
        if RUNNER_KIND == "local":
            rc = _run_local(w, time_limit_s, len(inputs))
            if rc is None:
                return "timeout"
            if rc != 0:
                return f"runner exited with {rc}"
            try:
                results = json.loads((w / "outputs.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                return "no readable output file"
            if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
                return "output file has the wrong shape"
            return results
        name = f"g10stdio-{uuid.uuid4().hex[:12]}"
        cmd = ["docker", "run", "--rm", "--name", name, *DOCKER_FLAGS, "-v", f"{w}:/work", "-w", "/work", IMAGE,
               "python", "-B", "-s", "/work/run.py", str(time_limit_s)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=time_limit_s * len(inputs) + 30)
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "kill", name], capture_output=True)
            return "timeout"
        if proc.returncode != 0:
            return f"runner exited with {proc.returncode}"
        try:
            results = json.loads((w / "outputs.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return "no readable output file"
    if not isinstance(results, list) or not all(isinstance(r, dict) for r in results):
        return "output file has the wrong shape"
    return results


def score(task: dict, completion: str) -> dict:
    code, why = extract_code(completion)
    out = {"task_id": task["id"], "reward": 0}
    if code is None:
        return {**out, "reason": why}
    tests = task["tests"][:MAX_TESTS]
    results = run_program(code, [t["input"] for t in tests], float(task.get("time_limit_s", 4)))
    if isinstance(results, str):
        return {**out, "reason": results}
    passed = 0
    for r, t in zip(results, tests):
        if r.get("status") != "ok" or not same_output(str(r.get("stdout", "")), t["output"]):
            break
        passed += 1
    reward = int(passed == len(tests))
    if reward:
        reason = "all tests passed"
    elif passed < len(results) and results[passed].get("status") != "ok":
        reason = f"{results[passed].get('status')} on a test"  # error or timeout; which test is not reported
    else:
        reason = "wrong answer on a test"
    return {**out, "reward": reward, "passed": passed, "total": len(tests), "reason": reason}
