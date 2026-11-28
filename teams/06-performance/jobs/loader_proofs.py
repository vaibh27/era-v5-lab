"""Loader proofs (CPU, loader only): Team 7 contract at world 1/2/4, resume (same world, 4->2, 2->4, separate
processes), and exact shares from the composition log.

  python jobs/loader_proofs.py --out <dir> [--config <yaml>] [--synthetic <dir>] [--batches 40] [--resume-at 25]

Without --synthetic, G6_SHARD_ROOT must point at shards. Writes <out>/proofs.json; exit 0 when all pass.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]                 # teams/06-performance
REPO = PKG.parents[1]
if str(PKG) not in sys.path:
    sys.path.insert(0, str(PKG))

CONTRACT = REPO / "teams" / "07-infrastructure" / "ckpt" / "loader_contract.py"
APPROVED = REPO / "teams" / "01-data-a" / "jobs" / "2026-11-19-first-training" / "approved_inputs.json"
DEFAULT_CONFIG = PKG / "configs" / "mix" / "stage1_placeholder.yaml"


# ---- synthetic shard-v1 -------------------------------------------------------------------------------------------

def write_synthetic(root: Path, tokens_per_pack: int = 600_000, shards_per_pack: int = 2, seed: int = 0) -> dict:
    """Synthetic shard-v1 files for every approved pack (random tokens, lognormal doc lengths)."""
    import numpy as np
    from dataio.prepare_fineweb import write_shard
    from dataio.shards import EOS_ID, P1536_SHA256

    root.mkdir(parents=True, exist_ok=True)
    packs = json.loads(APPROVED.read_text())["packs"]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    total = 0
    for row in packs:
        pack = row["pack_id"]
        rng = np.random.default_rng([seed, int(hashlib.sha256(pack.encode()).hexdigest()[:8], 16)])
        per = tokens_per_pack // shards_per_pack
        for num in range(shards_per_pack):
            lens = []
            while sum(lens) < per:
                lens.append(int(np.clip(rng.lognormal(np.log(600), 1.0), 8, 16384)))
            toks = rng.integers(0, 130_000, size=sum(lens), dtype=np.int64)
            ends = np.cumsum(lens)
            toks[ends - 1] = EOS_ID
            idx = np.concatenate([[0], ends]).astype(np.int64)
            sid = f"{pack}-train-{num:06d}"
            write_shard(root / f"{sid}.bin", toks)
            np.save(root / f"{sid}.idx.npy", idx)
            m = {"schema_version": "shard-v1", "dataset_version": "synthetic-g6", "shard_id": sid,
                 "format": "brahmic_u32_v2+doc_index", "token_dtype": "uint32", "payload_uri": f"{sid}.bin",
                 "payload_sha256": sha(root / f"{sid}.bin"), "payload_bytes": 1024 + 4 * len(toks),
                 "index_uri": f"{sid}.idx.npy", "index_sha256": sha(root / f"{sid}.idx.npy"),
                 "doc_count": len(lens), "token_count": len(toks), "eos_token_id": EOS_ID,
                 "tokenizer_sha256": P1536_SHA256, "pack_id": pack, "language": row["language"],
                 "script": row["script"], "domain": row["domain"], "capability_lane": row["lane"],
                 "license": row.get("license"), "license_source": row.get("license_source"), "split": "train",
                 "created_by": "teams/06-performance/jobs/loader_proofs.py (synthetic)"}
            (root / f"{sid}.json").write_text(json.dumps(m, indent=1))
            total += len(toks)
    info = {"packs": len(packs), "shards": len(packs) * shards_per_pack, "tokens": total,
            "tokens_per_pack": tokens_per_pack, "seed": seed, "approved_inputs_sha256": sha(APPROVED)}
    (root / "synthetic_info.txt").write_text(json.dumps(info, indent=1) + "\n")
    return info


# ---- child process: one rank, some batches ------------------------------------------------------------------------

def child(a) -> int:
    from dataio import mix_loader as ml
    L = ml.MixLoader(a.config, dp_rank=a.rank, dp_world=a.world, log_path=a.log or None)
    if a.state_in:
        L.load_state_dict(json.loads(Path(a.state_in).read_text()))
    start, rows, batch_sha = L.position, [], []
    for _ in range(a.batches):
        b = L.next()
        rows.append([hashlib.sha256(r.numpy().tobytes()).hexdigest() for r in b])
        batch_sha.append(ml.batch_sha256(b))
    if a.state_out:
        Path(a.state_out).write_text(json.dumps(L.state_dict(), sort_keys=True))
    Path(a.result).write_text(json.dumps({"rank": a.rank, "world": a.world, "start": start, "rows": rows,
                                          "batch_sha256": batch_sha, "state": L.state_dict()}))
    return 0


def run_ranks(cfg: Path, out: Path, tag: str, world: int, batches: int, state_in=None, save=False, log=None,
              hash_seed=0) -> list:
    """One process per rank, in parallel. Returns the ranks' result dicts in rank order."""
    procs, results = [], []
    for r in range(world):
        res = out / "runs" / f"{tag}-r{r}.json"
        res.parent.mkdir(parents=True, exist_ok=True)
        cmd = [sys.executable, str(Path(__file__).resolve()), "_child", "--config", str(cfg), "--rank", str(r),
               "--world", str(world), "--batches", str(batches), "--result", str(res)]
        if state_in:
            cmd += ["--state-in", str(state_in)]
        if save:
            cmd += ["--state-out", str(out / "states" / f"{tag}-r{r}.json")]
            (out / "states").mkdir(parents=True, exist_ok=True)
        if log and r == 0:
            cmd += ["--log", str(log)]
        env = dict(os.environ, PYTHONHASHSEED=str(hash_seed + r), PYTHONPATH=str(PKG))
        procs.append((subprocess.Popen(cmd, env=env, cwd=PKG), res))
    for p, res in procs:
        if p.wait() != 0:
            raise RuntimeError(f"{tag}: a rank process failed (exit {p.returncode})")
        results.append(json.loads(res.read_text()))
    return results


def global_digests(results: list) -> list:
    """Per batch: sha256 over the row hashes in global row order."""
    world = len(results)
    out = []
    for i in range(len(results[0]["rows"])):
        parts = [r["rows"][i] for r in results]
        n_rows = sum(len(p) for p in parts)
        out.append(hashlib.sha256("".join(parts[k % world][k // world] for k in range(n_rows)).encode()).hexdigest())
    return out


# ---- proofs -------------------------------------------------------------------------------------------------------

def proof_contract(cfg: Path) -> dict:
    env = dict(os.environ, G6_MIX_CONFIG=str(cfg), PYTHONPATH=str(PKG))
    cli = subprocess.run([sys.executable, str(CONTRACT), "--factory", "dataio.mix_loader:contract_factory"],
                         env=env, cwd=PKG, capture_output=True, text=True, timeout=1800)
    try:
        cli_out = json.loads(cli.stdout)
    except json.JSONDecodeError:
        cli_out = {"pass": False, "problems": [f"no JSON from the CLI: {cli.stdout[-400:]} {cli.stderr[-400:]}"]}
    os.environ["G6_MIX_CONFIG"] = str(cfg)
    spec = importlib.util.spec_from_file_location("loader_contract", CONTRACT)
    lc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(lc)
    from dataio.mix_loader import contract_factory
    by_world = {}
    for w in (1, 2, 4):
        probs = lc.check(contract_factory, world=w)
        by_world[str(w)] = {"pass": not probs, "problems": probs}
    ok = cli.returncode == 0 and cli_out.get("pass") and all(v["pass"] for v in by_world.values())
    return {"pass": bool(ok), "cli": {"exit": cli.returncode, **cli_out}, "check": by_world}


def proof_resume(cfg: Path, out: Path, total: int, cut: int) -> dict:
    log = out / "composition.jsonl"
    if log.exists():
        log.unlink()
    a = run_ranks(cfg, out, "A-w1", 1, total, log=log, hash_seed=100)
    ref = global_digests(a)
    res = {"A": {"world": 1, "batches": total, "digests": ref, "batch_sha256": a[0]["batch_sha256"]}}

    b1 = run_ranks(cfg, out, "B1-w1", 1, cut, save=True, hash_seed=200)
    b2 = run_ranks(cfg, out, "B2-w1", 1, total - cut, state_in=out / "states" / "B1-w1-r0.json", hash_seed=300)
    got = global_digests(b1) + global_digests(b2)
    res["B"] = {"pass": got == ref, "saved_at": cut, "first_diff": _first_diff(got, ref)}

    for name, w_save, w_load in (("C_4_to_2", 4, 2), ("C_2_to_4", 2, 4)):
        s = run_ranks(cfg, out, f"{name}-save-w{w_save}", w_save, cut, save=True, hash_seed=400 + w_save)
        states = {json.dumps(r["state"], sort_keys=True) for r in s}
        l = run_ranks(cfg, out, f"{name}-load-w{w_load}", w_load, total - cut,
                      state_in=out / "states" / f"{name}-save-w{w_save}-r0.json", hash_seed=500 + w_load)
        got = global_digests(s) + global_digests(l)
        res[name] = {"pass": got == ref and len(states) == 1, "states_identical_across_ranks": len(states) == 1,
                     "saved_at": cut, "first_diff": _first_diff(got, ref)}
    res["pass"] = all(v["pass"] for k, v in res.items() if k != "A")
    return res


def _first_diff(got, ref):
    if got == ref:
        return None
    return next((i for i, (x, y) in enumerate(zip(got, ref)) if x != y), min(len(got), len(ref)))


def proof_shares(cfg: Path, log: Path) -> dict:
    from dataio import mix_loader as ml
    L = ml.MixLoader(cfg)
    lines = [json.loads(x) for x in log.read_text().splitlines()]
    head, body = lines[0], lines[1:]
    target: dict = {}
    for p, w in L.weights.items():
        target[L.tables[p].lane] = target.get(L.tables[p].lane, 0.0) + w
    worst = dict.fromkeys(target, 0.0)
    quota_ok = True
    for line in body:
        st = bisect_stage(head["stages"], line["batch"])
        quota_ok &= line["by_pack"] == head["quotas"][st]
        for lane, share in target.items():
            worst[lane] = max(worst[lane], abs(line["by_lane"].get(lane, 0) - share * line["tokens"]))
    lanes_ok = all(v < 1 for v in worst.values())
    return {"pass": bool(quota_ok and lanes_ok and len(body) > 0), "batches": len(body),
            "by_pack_equals_quotas_every_batch": bool(quota_ok),
            "lane_target_share": {k: round(v, 6) for k, v in sorted(target.items())},
            "lane_worst_abs_error_tokens": {k: round(v, 4) for k, v in sorted(worst.items())},
            "tolerance": "each lane and the Indic total within 1 token of tokens x target share, every batch"}


def bisect_stage(stages: list, n: int) -> int:
    i = 0
    for k, s in enumerate(stages):
        if s["from_batch"] <= n:
            i = k
    return i


def environment(cfg: Path) -> dict:
    import numpy as np
    import torch
    from dataio import mix_loader as ml
    L = ml.MixLoader(cfg)
    git = lambda *a: subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True).stdout.strip()
    return {"date": time.strftime("%Y-%m-%d %H:%M:%S %z"), "python": platform.python_version(),
            "numpy": np.__version__, "torch": torch.__version__, "platform": platform.platform(),
            "git_head": git("rev-parse", "HEAD"), "git_dirty_files": len(git("status", "--porcelain").splitlines()),
            "config": str(cfg), "config_sha256": hashlib.sha256(cfg.read_bytes()).hexdigest(),
            "algo": ml.ALGO, "manifest": L.manifest, "mix": L.mix, "packing": L.packing,
            "stages": L.header()["stages"], "packs_read": len(L.packs), "batches_allowed": L.header()["batches_allowed"],
            "limit": L.limit_reason, "weights_file": L.weights_file, "sources_file": L.sources_file, "notes": L.notes}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("_child")
    for k in ("--config", "--result"):
        c.add_argument(k, required=True)
    c.add_argument("--rank", type=int, default=0)
    c.add_argument("--world", type=int, default=1)
    c.add_argument("--batches", type=int, required=True)
    c.add_argument("--state-in")
    c.add_argument("--state-out")
    c.add_argument("--log")
    ap.add_argument("--out")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--synthetic", help="write synthetic shard-v1 shards here and use them")
    ap.add_argument("--tokens-per-pack", type=int, default=600_000)
    ap.add_argument("--batches", type=int, default=40)
    ap.add_argument("--resume-at", type=int, default=25)
    a = ap.parse_args(argv)
    if a.cmd == "_child":
        return child(a)
    if not a.out:
        ap.error("--out is required")
    if not 0 < a.resume_at < a.batches:
        ap.error("need 0 < --resume-at < --batches")
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    cfg = Path(a.config).resolve()
    report = {"schema": "g6-loader-proofs-v1"}
    if a.synthetic:
        root = Path(a.synthetic).resolve()
        report["synthetic"] = write_synthetic(root, a.tokens_per_pack)
        os.environ["G6_SHARD_ROOT"] = str(root)
    elif not os.environ.get("G6_SHARD_ROOT"):
        ap.error("set G6_SHARD_ROOT or pass --synthetic")
    report["shard_root"] = os.environ["G6_SHARD_ROOT"]
    report["environment"] = environment(cfg)
    t = time.time()
    report["contract"] = proof_contract(cfg)
    report["resume"] = proof_resume(cfg, out, a.batches, a.resume_at)
    report["shares"] = proof_shares(cfg, out / "composition.jsonl")
    report["seconds"] = round(time.time() - t, 1)
    report["pass"] = all(report[k]["pass"] for k in ("contract", "resume", "shares"))
    (out / "proofs.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": report["pass"], "contract": report["contract"]["pass"],
                      "resume": {k: v["pass"] for k, v in report["resume"].items() if k not in ("A", "pass")},
                      "shares": report["shares"]["pass"], "seconds": report["seconds"],
                      "out": str(out / "proofs.json")}, indent=1))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
