#!/usr/bin/env python3
"""TSAI job wrapper: the G6 Gate 2 baseline on one GPU, with real FineWeb Brahmic data, ready to publish.

Usage (repo root):
  python teams/06-performance/jobs/run_baseline_timing.py --out out
  python teams/06-performance/jobs/run_baseline_timing.py --out out --data-dir <shards> --tokenizer <tokenizer.json>
  python teams/06-performance/jobs/run_baseline_timing.py --out out --prepare-data    # Colab only: needs network

The stages are the ones of the G6 Colab baseline, in this order:
  preflight     GPU visible, no other process on it, slowdown reasons at idle, packages, git commit and clean tree
  inputs        FineWeb Brahmic shards (dataio.loader format: header, size, tokenizer manifest) and the pinned
                BrahmicTokenizer-131K artifact (sha256). Never synthetic: missing data stops the job (exit 11).
  tests         python -m pytest correctness -q -rs
  smoke         python -m perfkit.smoke (loss falls, weights finite, reversible reconstruction OK)
  bench_normal  python -m perfkit.bench --level 3 --repeat 2 (base.yaml, all bench.modes)
  bench_mtp     the same with model.mtp=true
  tests_all     python -m pytest correctness -q --all-configs

No network (JOB_REQUEST.md rule 1): the data and the tokenizer are inputs. Make them once on a machine with network:
  cd teams/06-performance && python -m dataio.prepare_fineweb --out <dir>     # val + 1 train shard, 1M tokens each
and pass --data-dir <dir>; the tokenizer lands in ~/.cache/era-perf/BrahmicTokenizer-131K/tokenizer.json.

Writes only under --out:
  out/result.json      verdict, every stage (exit code, meaning, time, log tail), Gate 2 numbers, repeat check
  out/SUMMARY.md       the same for a PR comment
  out/logs/*.log       the output of every stage
  out/runs/*           run JSON, report .md and live .jsonl of every bench run
  out/.work/           raw files while the job runs (traces, checkpoints, raw runs); deleted at the end

Redaction is done here, so nothing needs to be removed by hand: every log, JSON and report under out/ has
no GPU model, GPU UUID or PCI ID, driver, host name, clocks, power, peak TFLOPS, MFU or HFU, and no absolute
path. GPU-architecture names in kernel names (sm80, ampere, ...) are replaced too. At the end every file
under out/ is scanned again; if any hardware detail is still there, the job fails with exit 16 and that file
is deleted (result.json is kept, with the verdict).

Exit codes (result.json says which stage and why):
  0 PASS   2 repeat check FAIL (timings complete, a gated part outside 5%)
  10 preflight   11 inputs missing or wrong   12 correctness tests failed   13 smoke failed
  14 bench did not finish (1 incomplete run JSON / crash, 3 out of memory, 4 no data, killed)
  15 a stage hit its time limit   16 redaction scan found hardware details
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

JOB_DIR = Path(__file__).resolve().parent
PERF_ROOT = JOB_DIR.parent
REPO_ROOT = PERF_ROOT.parents[1]
CONFIG = PERF_ROOT / "configs" / "base.yaml"

LAB_GPU = "one lab GPU"
LAB_NODE = "lab-node"
OK, REPEAT_FAIL = 0, 2
E_PREFLIGHT, E_INPUTS, E_TESTS, E_SMOKE, E_BENCH, E_TIMEOUT, E_LEAK = 10, 11, 12, 13, 14, 15, 16
BENCH_CODES = {0: "ok", 1: "run JSON incomplete or the run crashed", 2: "repeat check: a gated part outside 5%",
               3: "out of GPU memory", 4: "no data (shards missing)"}
SLOWDOWN = {0x4: "SwPowerCap", 0x8: "HwSlowdown", 0x20: "SwThermalSlowdown", 0x40: "HwThermalSlowdown",
            0x80: "HwPowerBrakeSlowdown"}
# Parts reported for Gate 2 next to the step (TSAI, PR 239: "the Muon part with its spread").
GATE2_PARTS = ("opt", "opt:opt.polar_muon_adamw", "opt:opt.muon", "opt:opt.muon.ns", "opt:opt.adamw",
               "fwd", "bwd", "fwd:head+loss", "bwd:head+loss")
STAGE_MINUTES = {"tests": 15, "smoke": 10, "bench_normal": 25, "bench_mtp": 25, "tests_all": 20, "prepare": 30}


# ------------------------------------------------------------------------------ redaction

GPU_MODELS = re.compile(
    r"\b(?:NVIDIA|Tesla|GeForce|Quadro|Instinct|Radeon|RTX ?\w+|A100|A10G?|A30|A40|A[456]000|H100|H200|H800|"
    r"GH200|B100|B200|GB[23]00|L4|L40S?|T4|V100|P100|P4|K80|MI\d{3}X?|SXM\d?|PCIe)\b")
ARCH = re.compile(r"(?i)(?<![a-z0-9])(?:sm_?\d{2,3}[a-z]?|ampere|hopper|turing|volta|ada|lovelace|blackwell|nvjet|"
                  r"cutlass_\d{2,3})(?![a-z])")
UTIL = re.compile(r"(?i)\b(MFU|HFU)\b(\s*[:=]?\s*)[\d.]+\s*%")
CLOCK_POWER = re.compile(r"\b\d+(?:\.\d+)?\s*(MHz|W)\b")
# Keys whose values identify the GPU or the host. Required schema keys are kept with value None.
HW_NONE = {"gpu_spec", "driver", "peak_tflops"}
HW_DROP = {"compute_capability", "sm_count", "gpu_mem_gb", "gpu_count", "gpu_index", "gpu_uuid", "gpu_pci_bus_id",
           "telemetry_start", "uuid", "pci_bus_id", "pci.bus_id", "driver_version", "power.limit", "power.draw",
           "clocks.sm", "clocks.max.sm", "clocks.mem", "sm_clock_mhz", "mem_clock_mhz", "power_w", "pstate"}
SAMPLE_KEEP = {"gpu_util_pct", "mem_util_pct", "mem_used_mib", "temp_c", "utilization.gpu", "utilization.memory",
               "memory.used", "temperature.gpu"}


class Scrubber:
    """Replaces this machine's identifiers (collected before any stage runs) and GPU / architecture names."""

    def __init__(self, identifiers: dict[str, str], paths: dict[str, Path]):
        # Whole words only, longest first: a short host name must not eat parts of ordinary words.
        self.exact = [(re.compile(rf"(?<![\w.-]){re.escape(v)}(?![\w-])"), k)
                      for k, v in sorted(identifiers.items(), key=lambda kv: -len(kv[1] or "")) if v and len(v) >= 4]
        # Longest path first, so <work> wins over <out> and <out> over <repo>.
        self.paths = sorted(((str(p), name) for name, p in paths.items() if p and len(str(p)) > 1),
                            key=lambda x: -len(x[0]))
        self.identifiers = identifiers

    def text(self, s: str) -> str:
        for p, name in self.paths:
            s = s.replace(p, f"<{name}>")
        for pattern, name in self.exact:
            s = pattern.sub(LAB_GPU if name.startswith("gpu_name") else f"<{name}>", s)
        s = GPU_MODELS.sub("<gpu>", s)
        s = ARCH.sub("<arch>", s)
        s = UTIL.sub(lambda m: f"{m.group(1)}{m.group(2)}withheld", s)
        return CLOCK_POWER.sub(lambda m: f"<n> {m.group(1)}", s)

    def obj(self, o, key: str | None = None):
        if isinstance(o, dict):
            out = {}
            for k, v in o.items():
                if k in HW_DROP:
                    continue
                if k in HW_NONE:
                    out[k] = None
                elif k == "node_id":
                    out[k] = LAB_NODE
                elif k == "gpu":
                    out[k] = LAB_GPU if not isinstance(v, dict) else {"name": LAB_GPU}
                elif k in ("mfu_unavailable_reason", "hfu_unavailable_reason"):
                    out[k] = "peak_tflops_unavailable"            # a schema value: MFU / HFU are null below
                elif k in ("mfu", "hfu") or (key == "metrics" and re.search(r"mfu|hfu|tflops", k)):
                    out[k] = None
                elif k == "metrics" and key in ("system", None) and isinstance(v, dict) and _is_sample(v):
                    out[k] = {mk: mv for mk, mv in v.items() if mk in SAMPLE_KEEP}
                elif k == "telemetry" and isinstance(v, dict):
                    out[k] = {phase: _telemetry_flags(t) for phase, t in v.items()}
                elif k == "trace_file":
                    out[k] = "withheld (raw trace not published)"
                elif k == "file" and key == "parts":
                    out[k] = None
                else:
                    out[k] = self.obj(v, k)
            return out
        if isinstance(o, list):
            return [self.obj(v, key) for v in o]
        if isinstance(o, str):
            return self.text(o)
        return o

    def leaks(self, s: str) -> list[str]:
        """Kinds of hardware detail still in s (names only, never the values)."""
        found = [name for pattern, name in self.exact if pattern.search(s)]
        found += [f"path:{name}" for p, name in self.paths if p in s]
        if GPU_MODELS.search(s):
            found.append("gpu model name")
        if ARCH.search(s):
            found.append("gpu architecture name")
        if UTIL.search(s):
            found.append("MFU/HFU value")
        return sorted(set(found))


def _is_sample(d: dict) -> bool:
    return any(k in d for k in ("sm_clock_mhz", "power_w", "gpu_util_pct", "temp_c", "clocks.sm"))


def _telemetry_flags(t):
    """A telemetry sample -> only what says whether the GPU was slowed down (no clocks, power or name)."""
    if not isinstance(t, dict):
        return None
    raw = t.get("throttle_reasons")
    flags = [n for b, n in SLOWDOWN.items() if isinstance(raw, int) and raw & b] if raw is not None else None
    return {"slowdown_flags": flags, "temp_c": t.get("temperature.gpu")}


# ------------------------------------------------------------------------------ machine facts (nvidia-smi, git)

def _sh(cmd, cwd=None, timeout=30) -> str | None:
    try:
        r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return r.stdout.strip() if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def smi_gpus() -> list[dict]:
    """One dict per GPU from nvidia-smi; the parent never initializes CUDA (the bench parent does not either)."""
    for reason in ("clocks_event_reasons.active", "clocks_throttle_reasons.active"):
        out = _sh(["nvidia-smi", f"--query-gpu=name,uuid,pci.bus_id,driver_version,temperature.gpu,{reason}",
                   "--format=csv,noheader,nounits"])
        if out:
            rows = []
            for line in out.splitlines():
                name, uuid, bus, drv, temp, rsn = [x.strip() for x in line.split(",")]
                try:
                    flags = [n for b, n in SLOWDOWN.items() if int(rsn, 16) & b]
                except ValueError:
                    flags = None
                rows.append({"name": name, "uuid": uuid, "pci_bus_id": bus, "driver": drv,
                             "temp_c": float(temp) if temp.replace(".", "").isdigit() else None,
                             "slowdown_flags": flags})
            return rows
    return []


def smi_other_processes() -> int | None:
    out = _sh(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"])
    return None if out is None else len([x for x in out.splitlines() if x.strip()])


def git_state() -> tuple[str | None, bool | None, list[str]]:
    """Same rule as perfkit.metrics.git_state: dirty = any change in teams/06-performance outside results/."""
    commit = _sh(["git", "rev-parse", "HEAD"], cwd=PERF_ROOT)
    status = _sh(["git", "status", "--porcelain", "--untracked-files=all", "--", ".", ":(exclude)results",
                  ":(exclude,glob)**/__pycache__/**", ":(exclude,glob)**/.pytest_cache/**",
                  ":(exclude,glob)**/.ipynb_checkpoints/**", ":(exclude,glob)**/.DS_Store"], cwd=PERF_ROOT)
    files = [ln[3:] for ln in status.splitlines()] if status else []
    return commit, (bool(status) if status is not None else None), files[:20]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------ the job

class Job:
    def __init__(self, a):
        self.a = a
        self.out = Path(a.out).resolve()
        self.work = self.out / ".work"
        self.t0 = time.monotonic()
        self.deadline = self.t0 + a.max_minutes * 60
        self.result = {"verdict": "RUNNING", "exit_code": None, "started_utc": _now(), "finished_utc": None,
                       "job": "G6 baseline timing (Gate 2)", "config": str(CONFIG.relative_to(REPO_ROOT)),
                       "stages": [], "problems": [], "gate2": {}, "repeat_check": {}}
        self.scrub = Scrubber({}, {})
        self.env = {}

    # ---- bookkeeping
    def problem(self, stage: str, code: int, cause: str, detail: str = ""):
        self.result["problems"].append({"stage": stage, "exit_code": code, "cause": cause,
                                        "detail": self.scrub.text(detail)})

    def save(self):
        """result.json after every stage: a job stopped at the time limit still says how far it got."""
        self.out.mkdir(parents=True, exist_ok=True)
        (self.out / "result.json").write_text(json.dumps(self.scrub.obj(self.result), indent=2) + "\n",
                                              encoding="utf-8")

    def minutes_left(self) -> float:
        return (self.deadline - time.monotonic()) / 60

    # ---- running one stage
    def run(self, name: str, cmd: list[str], *, minutes: float, extra_env: dict | None = None) -> dict:
        stage = {"stage": name, "cmd": self.scrub.text(" ".join(cmd)), "status": "running", "exit_code": None,
                 "meaning": None, "seconds": None, "log": f"logs/{name}.log", "tail": []}
        self.result["stages"].append(stage)
        limit = min(minutes, self.minutes_left() - 1)
        if limit < 1:
            stage.update(status="skipped", meaning="no time left under --max-minutes")
            self.problem(name, E_TIMEOUT, "skipped", stage["meaning"])   # a skipped stage is never a PASS
            self.save()
            return stage
        self.save()
        print(f"\n[job] ===== {name} (limit {limit:.0f} min)\n[job] {stage['cmd']}", flush=True)
        log_path = self.out / stage["log"]
        log_path.parent.mkdir(parents=True, exist_ok=True)
        tail: list[str] = []
        t = time.monotonic()
        proc = subprocess.Popen(cmd, cwd=str(PERF_ROOT), env={**self.env, **(extra_env or {})},
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                errors="replace", start_new_session=True)

        def pump():
            with log_path.open("w", encoding="utf-8", newline="\n") as log:
                for line in proc.stdout:
                    clean = self.scrub.text(line.rstrip("\r\n"))
                    print(clean, flush=True)
                    log.write(clean + "\n")
                    tail.append(clean)
                    del tail[:-60]

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        timed_out = False
        try:
            proc.wait(timeout=limit * 60)
        except subprocess.TimeoutExpired:
            timed_out = True
            _kill(proc)
        reader.join(timeout=30)
        stage.update(exit_code=proc.returncode, seconds=round(time.monotonic() - t, 1), tail=tail[-40:],
                     status="timeout" if timed_out else ("ok" if proc.returncode == 0 else "fail"))
        if timed_out:
            stage["meaning"] = f"stopped at the {limit:.0f} min limit"
            self.problem(name, E_TIMEOUT, "time limit", stage["meaning"])
        print(f"[job] {name}: {stage['status']} (exit {proc.returncode}, {stage['seconds']} s)", flush=True)
        self.save()
        return stage

    # ---- stages
    def preflight(self) -> bool:
        gpus = smi_gpus()
        ids = {"host": socket.gethostname(), "host_fqdn": socket.getfqdn()}
        for i, g in enumerate(gpus):
            ids.update({f"gpu_name{i}": g["name"], f"gpu_uuid{i}": g["uuid"], f"gpu_pci{i}": g["pci_bus_id"],
                        f"driver{i}": g["driver"]})
            short = g["name"].replace("NVIDIA ", "")
            ids[f"gpu_name{i}_short"] = short
            ids[f"gpu_name{i}_slug"] = re.sub(r"[^a-z0-9]+", "-", short.lower()).strip("-")
        self.scrub = Scrubber(ids, {"work": self.work, "out": self.out, "repo": REPO_ROOT,
                                    "data": self.data_dir, "tokenizer": self.tokenizer.parent,
                                    "home": Path.home(), "python": Path(sys.executable).parent})
        commit, dirty, changed = git_state()
        others = smi_other_processes()
        pf = {"gpus_visible": len(gpus), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "other_gpu_processes": others,
              "idle_slowdown_flags": gpus[0]["slowdown_flags"] if gpus else None,
              "idle_temp_c": gpus[0]["temp_c"] if gpus else None,
              "python": sys.version.split()[0], "commit": commit, "dirty": dirty, "changed_files": changed,
              "packages": {}}
        for mod in ("torch", "numpy", "yaml", "tokenizers", "pytest"):
            v = _sh([sys.executable, "-c", f"import {mod}; print(getattr({mod}, '__version__', 'ok'))"], timeout=120)
            pf["packages"][mod] = v
        cuda = _sh([sys.executable, "-c", "import torch; print(torch.version.cuda, torch.cuda.is_available())"],
                   timeout=120)
        pf["torch_cuda"], pf["cuda_available"] = (cuda.split() + [None, None])[:2] if cuda else (None, None)
        self.result["preflight"] = pf
        self.result["commit"] = commit
        bad = []
        if not gpus or pf["cuda_available"] != "True":
            bad.append("no CUDA GPU visible to torch")
        missing = [m for m, v in pf["packages"].items() if v is None]
        if missing:
            bad.append(f"missing packages: {', '.join(missing)} (teams/06-performance/perfkit/requirements.txt)")
        if commit is None or dirty is None:
            bad.append("not a git checkout: the numbers must name a commit")
        elif dirty:
            bad.append(f"uncommitted changes in teams/06-performance: {', '.join(changed[:5])}")
        if others:
            self.result.setdefault("environment", []).append(
                f"{others} other process(es) on the GPU before the job: a shared GPU misses the 5% check")
        if pf["idle_slowdown_flags"]:
            self.result.setdefault("environment", []).append(
                f"GPU already slowed down at idle ({', '.join(pf['idle_slowdown_flags'])})")
        self.result["stages"].append({"stage": "preflight", "status": "fail" if bad else "ok",
                                      "exit_code": E_PREFLIGHT if bad else 0, "meaning": "; ".join(bad) or "ok"})
        for b in bad:
            self.problem("preflight", E_PREFLIGHT, "preflight", b)
        self.save()
        print(f"[job] preflight: {'; '.join(bad) or 'ok'} | commit {commit} dirty={dirty} | "
              f"other GPU processes {others}", flush=True)
        return not bad

    def inputs(self) -> bool:
        info = {"data_dir": "<data>", "tokenizer": "<tokenizer>/tokenizer.json", "prepared_by_job": False}
        bad = []
        if self.a.prepare_data and not (self.data_dir / "tokenizer.json.meta").exists():
            st = self.run("prepare", [sys.executable, "-m", "dataio.prepare_fineweb", "--out", str(self.data_dir)],
                          minutes=STAGE_MINUTES["prepare"])
            info["prepared_by_job"] = st["status"] == "ok"
        sys.path.insert(0, str(PERF_ROOT))
        try:
            from dataio.loader import open_shard
            from dataio.tokenizer import REVISION, SHA256, VOCAB_SIZE
        except Exception as e:                                   # noqa: BLE001 - reported, not hidden
            bad.append(f"cannot import dataio: {e!r}")
            open_shard = REVISION = SHA256 = VOCAB_SIZE = None
        if REVISION:
            info["tokenizer_revision"] = REVISION
            if not self.tokenizer.exists():
                bad.append("BrahmicTokenizer-131K tokenizer.json not found: pass --tokenizer (no download, no network)")
            elif sha256(self.tokenizer) != SHA256:
                bad.append("tokenizer.json sha256 differs from the pinned revision")
            meta = self.data_dir / "tokenizer.json.meta"
            if not meta.exists():
                bad.append("no completed FineWeb Brahmic dataset (tokenizer.json.meta missing): stage it with "
                           "python -m dataio.prepare_fineweb --out <dir> and pass --data-dir <dir>")
            else:
                want = {"revision": REVISION, "sha256": SHA256, "vocab_size": VOCAB_SIZE}
                if json.loads(meta.read_text()) != want:
                    bad.append("shard manifest was made with another tokenizer: prepare the data again")
                shards = {}
                for split in ("val", "train"):
                    paths = sorted(self.data_dir.glob(f"fineweb_{split}_*.bin"))
                    if not paths:
                        bad.append(f"no fineweb_{split}_*.bin shard")
                    for p in paths:
                        try:
                            n = len(open_shard(p))
                            shards[p.name] = {"tokens": n, "sha256": sha256(p)}
                        except ValueError as e:
                            bad.append(str(e))
                info["shards"] = shards
        self.result["inputs"] = info
        self.result["stages"].append({"stage": "inputs", "status": "fail" if bad else "ok",
                                      "exit_code": E_INPUTS if bad else 0, "meaning": "; ".join(bad) or "ok"})
        for b in bad:
            self.problem("inputs", E_INPUTS, "inputs", b)
        self.save()
        print(f"[job] inputs: {self.scrub.text('; '.join(bad)) or 'ok'}", flush=True)
        return not bad

    def tests(self, name: str, *extra: str):
        junit = self.work / f"{name}.xml"
        st = self.run(name, [sys.executable, "-m", "pytest", "correctness", "-q", "-rfEs", "-p", "no:cacheprovider",
                             "--basetemp", str(self.work / f"pytest_{name}"), "--junitxml", str(junit), *extra],
                      minutes=STAGE_MINUTES[name])
        st["failed_tests"] = _junit_failures(junit)
        st["summary"] = next((ln for ln in reversed(st.get("tail", [])) if re.search(r"\d+ (passed|failed)", ln)), None)
        if st["status"] == "fail":
            st["meaning"] = f"{len(st['failed_tests'])} test(s) failed" if st["failed_tests"] else "pytest error"
            self.problem(name, E_TESTS, "correctness tests failed", ", ".join(st["failed_tests"][:10]))
        self.save()

    def smoke(self):
        st = self.run("smoke", [sys.executable, "-m", "perfkit.smoke", *self.data_overrides()],
                      minutes=STAGE_MINUTES["smoke"])
        text = "\n".join(st.get("tail", []))
        checks = {"weights_finite": "all weights finite: True" in text,
                  "reconstruction_ok": "(OK)" in text or "reconstruction" not in text}
        m = re.search(r"loss ([\d.]+) -> ([\d.]+)", text)
        checks["loss_falls"] = bool(m and float(m.group(2)) < float(m.group(1)))
        m = re.search(r"step median ([\d.]+) ms.*?([\d,]+) tokens/s", text)
        st["checks"] = checks
        st["step_ms_median"], st["tokens_per_s"] = (float(m.group(1)), int(m.group(2).replace(",", ""))) if m else (None, None)
        if st["status"] == "ok" and not all(checks.values()):
            st["status"] = "fail"
            st["meaning"] = "failed checks: " + ", ".join(k for k, v in checks.items() if not v)
        if st["status"] == "fail":
            self.problem("smoke", E_SMOKE, "smoke failed", st.get("meaning") or _last_error(st))
        self.save()

    def bench(self, name: str, tag: str, *overrides: str):
        cool = self.cooldown()
        st = self.run(name, [sys.executable, "-m", "perfkit.bench", "--config", str(CONFIG), "--tag", tag,
                             "--level", "3", "--repeat", "2", "--node-id", LAB_NODE,
                             f"bench.out_dir={(self.work / 'runs').as_posix()}", *self.data_overrides(), *overrides],
                      minutes=STAGE_MINUTES[name])
        st["cooldown"] = cool
        code = st["exit_code"]
        if st["status"] != "timeout":
            st["meaning"] = BENCH_CODES.get(code, f"killed by signal {-code}" if code and code < 0 else f"exit {code}")
        runs = [self.work / "runs" / f"{tag}.json", self.work / "runs" / f"{tag}-run2.json"]
        loaded = [json.loads(p.read_text()) for p in runs if p.exists()]
        self.result["gate2"][name] = [gate2(r) for r in loaded]
        rc = repeat_check(loaded) if len(loaded) == 2 else {"ran": False}
        rc["environment"] = slowdown_warnings(loaded)
        self.result["repeat_check"][name] = rc
        for r in loaded:
            me = r.get("meta", {})
            if me.get("dirty") is not False or not me.get("commit"):
                self.problem(name, E_BENCH, "provenance", f"run {me.get('tag')}: commit={me.get('commit')} "
                                                          f"dirty={me.get('dirty')}")
        if st["status"] in ("timeout", "skipped"):
            pass
        elif code == 2:
            cause = ("likely the lab GPU (slowdown during the steps): " + "; ".join(rc["environment"])
                     if rc["environment"] else "timing differs between the two runs (no GPU slowdown reported)")
            self.problem(name, REPEAT_FAIL, "repeat check FAIL", f"{cause}. Outside: "
                         + ", ".join(f"{p['part']} {p['change_pct']}%" for p in rc.get("outside", [])))
        elif code != 0:
            self.problem(name, E_BENCH, st["meaning"], _last_error(st))
        self.save()

    def cooldown(self) -> dict:
        """Wait (up to --cooldown-s) until the GPU is below 70 C and not slowed down at idle; record what was seen."""
        t = time.monotonic()
        g = smi_gpus()
        while g and time.monotonic() - t < self.a.cooldown_s and ((g[0]["temp_c"] or 0) >= 70 or g[0]["slowdown_flags"]):
            time.sleep(10)
            g = smi_gpus()
        return {"waited_s": round(time.monotonic() - t), "temp_c": g[0]["temp_c"] if g else None,
                "slowdown_flags": g[0]["slowdown_flags"] if g else None}

    def data_overrides(self) -> list[str]:
        return [f"data.dir={self.data_dir.as_posix()}", f"data.tokenizer_path={self.tokenizer.as_posix()}"]

    # ---- publishing
    def publish(self):
        """Scrubbed run JSON, report .md (rendered again from the scrubbed JSON) and live .jsonl into out/runs/."""
        runs_out = self.out / "runs"
        runs_out.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(PERF_ROOT))
        try:
            from perfkit import report, schema
        except Exception:                                         # noqa: BLE001
            report = schema = None
        for p in sorted((self.work / "runs").glob("*.json")) if (self.work / "runs").exists() else []:
            run = self.scrub.obj(json.loads(p.read_text()))
            if schema:
                probs = schema.validate_run(run)
                if probs:
                    run.setdefault("warnings", []).append(f"after redaction: {probs}")
            (runs_out / p.name).write_text(json.dumps(run, indent=1) + "\n", encoding="utf-8")
            try:
                md = report.render(run)
            except Exception as e:                                # noqa: BLE001
                md = f"# {p.stem}\n\nreport.render failed after redaction: {e!r}\n"
            (runs_out / p.with_suffix(".md").name).write_text(self.scrub.text(md), encoding="utf-8")
            jl = p.with_suffix(".jsonl")
            if jl.exists():
                with jl.open(encoding="utf-8") as src, (runs_out / jl.name).open("w", encoding="utf-8") as dst:
                    for line in src:
                        try:
                            dst.write(json.dumps(self.scrub.obj(json.loads(line))) + "\n")
                        except json.JSONDecodeError:
                            continue

    def leak_scan(self) -> list[dict]:
        """Every file under out/ (not .work), after redaction. A file with a hardware detail is deleted."""
        found = []
        for p in sorted(self.out.rglob("*")):
            if not p.is_file() or self.work in p.parents:
                continue
            kinds = self.scrub.leaks(p.read_text(encoding="utf-8", errors="replace"))
            if kinds:
                found.append({"file": str(p.relative_to(self.out)), "kinds": kinds})
                if p.name != "result.json":                       # result.json stays: it carries the verdict
                    p.unlink()
        return found


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
        proc.wait(timeout=20)
    except (OSError, subprocess.TimeoutExpired, AttributeError):
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (OSError, AttributeError):
            proc.kill()
        proc.wait()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _junit_failures(path: Path) -> list[str]:
    if not path.exists():
        return []
    import xml.etree.ElementTree as ET
    out = []
    for case in ET.parse(path).getroot().iter("testcase"):
        if case.find("failure") is not None or case.find("error") is not None:
            out.append(f"{case.get('classname')}::{case.get('name')}")
    return out


def _last_error(stage: dict) -> str:
    tail = stage.get("tail", [])
    for i in range(len(tail) - 1, -1, -1):
        if "Error" in tail[i] or "Traceback" in tail[i] or tail[i].startswith("[bench]"):
            return " | ".join(tail[i:i + 3])
    return " | ".join(tail[-3:])


# ------------------------------------------------------------------------------ reading the run JSONs

def _region(run: dict, name: str) -> dict | None:
    r = run.get("layer", {}).get("regions", {}).get(name)
    if not r:
        return None
    return {k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k)) for k in ("median", "p10", "p90", "spread_pct")}


def gate2(run: dict) -> dict:
    """What Gate 2 reports (TSAI, PR 239): step time, tokens/s, the Muon part with its spread."""
    st, m, me = run["clean"]["step_ms"], run.get("metrics", {}), run.get("meta", {})
    return {"tag": me.get("tag"), "commit": me.get("commit"), "dirty": me.get("dirty"), "dtype": me.get("dtype"),
            "batch_size": me.get("batch_size"), "seq_len": me.get("seq_len"), "data_source": me.get("data_source"),
            "step_ms": {k: round(st[k], 3) for k in ("median", "p10", "p90") if st.get(k) is not None},
            "step_spread_pct": round(st["spread_pct"], 1) if st.get("spread_pct") is not None else None,
            "tokens_per_s": round(m["tokens_per_s"]) if m.get("tokens_per_s") else None,
            "peak_allocated_gb": round(run["memory"]["peak_allocated_gb"], 2) if run.get("memory") else None,
            "loss_first_to_end": [me.get("loss_after_first_step"), me.get("loss_end")],
            "parts": {p: _region(run, p) for p in GATE2_PARTS if _region(run, p)},
            "checkpoint_ms": (run.get("checkpoint") or {}).get("ms"),
            "warnings": run.get("warnings", [])}


def repeat_check(runs: list[dict]) -> dict:
    """run 2 against run 1 with perfkit.bench.quick_compare (the same rule bench --repeat used)."""
    first, second = runs
    rep = first["config"]["report"]
    sys.path.insert(0, str(PERF_ROOT))
    try:
        from perfkit.bench import PASSED, quick_compare
        rows, ok, problems = quick_compare(second, first, rep["tol_pct"], rep["gate_level"], rep["abs_tol_ms"])
    except Exception as e:                                        # noqa: BLE001
        return {"ran": False, "error": repr(e)}
    regions = second.get("layer", {}).get("regions", {})
    outside = []
    for name, x, y, diff, verdict in rows:
        if verdict not in PASSED:
            spread = (regions.get(name) or {}).get("spread_pct")
            both = x is not None and y is not None
            outside.append({"part": name, "run2_ms": _r(x), "run1_ms": _r(y), "diff_ms": _r(x - y) if both else None,
                            "change_pct": _r(100 * (x - y) / y) if both and y else None, "gate_pct": _r(diff),
                            "run2_spread_pct": _r(spread), "verdict": verdict})
    return {"ran": True, "pass": ok, "gated_parts": len(rows), "outside": outside, "problems": problems,
            "tol_pct": rep["tol_pct"], "abs_tol_ms": rep["abs_tol_ms"]}


def slowdown_warnings(runs: list[dict]) -> list[str]:
    out = []
    for r in runs:
        for w in r.get("warnings", []):
            if re.search(r"slow|throttl|temperature|shared", w, re.I):
                out.append(f"{r['meta'].get('tag')}: {w}")
    return out


def _r(x):
    return round(x, 3) if isinstance(x, (int, float)) else x


# ------------------------------------------------------------------------------ summary

def summary_md(res: dict) -> str:
    lines = [f"# G6 baseline timing: {res['verdict']} (exit {res['exit_code']})", "",
             f"commit `{res.get('commit')}` | config `{res['config']}` | GPU: {LAB_GPU} | "
             f"data: FineWeb Brahmic shards ({len((res.get('inputs') or {}).get('shards') or {})})", "",
             "| stage | status | exit | meaning | s |", "| :--- | :--- | ---: | :--- | ---: |"]
    for s in res["stages"]:
        lines.append(f"| {s['stage']} | {s['status']} | {s.get('exit_code')} | {s.get('meaning') or ''} | "
                     f"{s.get('seconds') or ''} |")
    for name, runs in res["gate2"].items():
        lines += ["", f"## {name}", "", "| run | step median ms | p10..p90 | tokens/s | peak GB | Muon median ms (p10..p90, spread) |",
                  "| :--- | ---: | :--- | ---: | ---: | :--- |"]
        for g in runs:
            mu = g["parts"].get("opt:opt.muon") or {}
            lines.append(f"| {g['tag']} | {g['step_ms'].get('median')} | {g['step_ms'].get('p10')}..{g['step_ms'].get('p90')} "
                         f"| {g['tokens_per_s'] or 0:,} | {g['peak_allocated_gb']} | {mu.get('median')} "
                         f"({mu.get('p10')}..{mu.get('p90')}, {mu.get('spread_pct')}%) |")
        rc = res["repeat_check"].get(name, {})
        if rc.get("ran"):
            lines.append(f"\nrepeat check: {'PASS' if rc['pass'] else 'FAIL'}, {len(rc['outside'])} of "
                         f"{rc['gated_parts']} gated parts outside {rc['tol_pct']}% and {rc['abs_tol_ms']} ms")
            for o in rc["outside"]:
                lines.append(f"- `{o['part']}`: {o['run1_ms']} -> {o['run2_ms']} ms ({o['change_pct']}%), "
                             f"run-2 spread {o['run2_spread_pct']}%")
        for e in rc.get("environment", []):
            lines.append(f"- environment: {e}")
    if res.get("environment"):
        lines += ["", "## Environment before the job", ""] + [f"- {e}" for e in res["environment"]]
    if res["problems"]:
        lines += ["", "## Problems", ""] + [f"- **{p['stage']}** (exit {p['exit_code']}) {p['cause']}: {p['detail']}"
                                           for p in res["problems"]]
    if res.get("redaction_leaks"):
        lines += ["", "## Redaction", ""] + [f"- deleted `{x['file']}`: {', '.join(x['kinds'])}"
                                             for x in res["redaction_leaks"]]
    lines += ["", "MFU and HFU are withheld; the GPU model, host and clocks are removed by the job itself.", ""]
    return "\n".join(lines)


# ------------------------------------------------------------------------------ main

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="out", help="output folder (the only write target)")
    ap.add_argument("--data-dir", default=os.environ.get("ERA_PERF_DATA_DIR"),
                    help="FineWeb Brahmic shards from dataio.prepare_fineweb (default: data.dir of base.yaml)")
    ap.add_argument("--tokenizer", default=os.environ.get("ERA_PERF_TOKENIZER"),
                    help="BrahmicTokenizer-131K tokenizer.json (default: ~/.cache/era-perf/BrahmicTokenizer-131K/)")
    ap.add_argument("--prepare-data", action="store_true",
                    help="run dataio.prepare_fineweb into --data-dir when it is empty (needs network: Colab only)")
    ap.add_argument("--max-minutes", type=float, default=60, help="whole job; later stages are skipped near it")
    ap.add_argument("--cooldown-s", type=int, default=180, help="max wait before each bench for an idle, cool GPU")
    ap.add_argument("--skip-tests", action="store_true", help="skip both pytest stages")
    ap.add_argument("--skip-mtp", action="store_true", help="skip the model.mtp=true bench")
    ap.add_argument("--keep-work", action="store_true", help="keep out/.work (raw, NOT redacted: never publish it)")
    a = ap.parse_args(argv)

    out = Path(a.out).resolve()
    if out.name != "out" and "out" not in out.parts:             # JOB_REQUEST rule 1: write only under ./out
        print("refusing out path that is not under an 'out' folder:", out, file=sys.stderr)
        return E_PREFLIGHT
    job = Job(a)
    cfg_data = _config_data()
    job.data_dir = Path(a.data_dir or cfg_data.get("dir") or "~/.cache/era-perf/fineweb-brahmic131k").expanduser().resolve()
    job.tokenizer = Path(a.tokenizer or cfg_data.get("tokenizer_path")
                         or "~/.cache/era-perf/BrahmicTokenizer-131K/tokenizer.json").expanduser().resolve()
    shutil.rmtree(job.work, ignore_errors=True)
    job.work.mkdir(parents=True)
    # Every cache a stage could write goes under out/.work: traces and checkpoints (perfkit.bench CACHE),
    # Hugging Face (prepare), compile caches, Python bytecode. ERA_PERF_TOKENIZER points the tests at the input.
    job.env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(PERF_ROOT), os.environ.get("PYTHONPATH")])),
               "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "ERA_PERF_CACHE": str(job.work / "cache"),
               "ERA_PERF_TOKENIZER": str(job.tokenizer), "HF_HOME": str(job.work / "hf"),
               "TORCHINDUCTOR_CACHE_DIR": str(job.work / "inductor"), "TRITON_CACHE_DIR": str(job.work / "triton"),
               "MPLCONFIGDIR": str(job.work / "mpl"), "G8_NODE_ID": LAB_NODE}

    if job.preflight() and job.inputs():
        if not a.skip_tests:
            job.tests("tests")
        job.smoke()
        job.bench("bench_normal", "normal_rep")
        if not a.skip_mtp:
            job.bench("bench_mtp", "mtp_rep", "model.mtp=true")
        if not a.skip_tests:
            job.tests("tests_all", "--all-configs")
    job.publish()

    codes = [p["exit_code"] for p in job.result["problems"]]
    hard = [c for c in codes if c not in (OK, REPEAT_FAIL)]
    code = min(hard) if hard else (REPEAT_FAIL if REPEAT_FAIL in codes else OK)
    job.result["exit_code"] = code
    job.result["verdict"] = {OK: "PASS", REPEAT_FAIL: "FAIL (repeat check)"}.get(code, "FAIL")
    job.result["finished_utc"] = _now()
    job.save()
    (job.out / "SUMMARY.md").write_text(summary_md(job.scrub.obj(job.result)), encoding="utf-8")
    leaks = job.leak_scan()
    if leaks:
        job.result["redaction_leaks"] = leaks
        job.problem("redaction", E_LEAK, "hardware detail found after redaction",
                    ", ".join(x["file"] for x in leaks))
        job.result["exit_code"], job.result["verdict"] = E_LEAK, "FAIL"
        job.save()
        (job.out / "SUMMARY.md").write_text(summary_md(job.scrub.obj(job.result)), encoding="utf-8")
        code = E_LEAK
    if not a.keep_work:
        shutil.rmtree(job.work, ignore_errors=True)
    print(f"\n[job] {job.result['verdict']} (exit {code}) -> {job.scrub.text(str(job.out / 'result.json'))}")
    return code


def _config_data() -> dict:
    try:
        import yaml
        return (yaml.safe_load(CONFIG.read_text()) or {}).get("data") or {}
    except Exception:                                             # noqa: BLE001 - defaults then apply
        return {}


if __name__ == "__main__":
    raise SystemExit(main())
