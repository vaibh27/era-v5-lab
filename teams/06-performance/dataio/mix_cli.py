"""Command line for the mix loader: pick data and weights, then show, check or prove.

  python -m dataio.mix_loader --data synthetic --weights skewed              (show is the default)
  python -m dataio.mix_loader --data <shard dir> --weights equal same
  python -m dataio.mix_loader --data <shard dir> --weights equal --fix-lanes proofs

Actions: summary (log header), show (batch 0), same (determinism checks), proofs (jobs/loader_proofs.py).
Packs or lanes missing from the data are dropped and renormalised (listed in the notes).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch
import yaml

from dataio import mix_loader as ml

PKG = Path(__file__).resolve().parents[1]                 # teams/06-performance
MIX = PKG / "configs" / "mix"
PROOFS = PKG / "jobs" / "loader_proofs.py"
ACTIONS = ("summary", "show", "same", "proofs")


def _abs(value, base: Path) -> str:
    p = Path(os.path.expandvars(str(value))).expanduser()
    return str(p if p.is_absolute() else base / p)


def fix_lanes(src: Path, dst: Path) -> int:
    """Mirror src into dst with symlinks; manifests with capability_lane general_language take their domain instead
    (the sample shards predate the spec's lane names). Returns the number of manifests changed."""
    changed = 0
    for f in sorted(src.rglob("*")):
        if f.is_dir():
            continue
        out = dst / f.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists() or out.is_symlink():
            out.unlink()
        m = json.loads(f.read_text()) if f.suffix == ".json" else None
        if isinstance(m, dict) and m.get("schema_version") == "shard-v1" and m.get("capability_lane") == "general_language":
            out.write_text(json.dumps(dict(m, capability_lane=m["domain"]), indent=1))
            changed += 1
        else:
            out.symlink_to(f.resolve())
    return changed


def build_config(a, work: Path) -> Path:
    """Resolve --config, --data and --weights into one config file with absolute paths; returns its path."""
    base = Path(a.config).resolve().parent
    cfg = yaml.safe_load(Path(a.config).read_text())
    if a.data == "synthetic":
        spec = importlib.util.spec_from_file_location("loader_proofs", PROOFS)
        proofs = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(proofs)
        data = work / "synthetic"
        print("synthetic shards:", json.dumps(proofs.write_synthetic(data)))
    else:
        data = Path(a.data).expanduser().resolve()
        if not data.is_dir():
            raise SystemExit(f"--data {a.data}: not a directory (use a shard folder or 'synthetic')")
    if a.fix_lanes:
        n = fix_lanes(data, work / "fixed_lanes")
        print(f"fix-lanes: {n} manifests took their lane from domain")
        data = work / "fixed_lanes"
    weights = MIX / f"weights_{a.weights}.yaml" if a.weights in ("equal", "skewed") else Path(a.weights).resolve()
    cfg["shards"] = dict(cfg.get("shards") or {}, roots=[str(data)], missing="renormalise")
    cfg["weights"] = {"file": str(weights)}
    if cfg.get("sources_file"):
        cfg["sources_file"] = _abs(cfg["sources_file"], base)
    path = work / "config.yaml"
    path.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return path


def show(cfg: Path) -> None:
    L = ml.MixLoader(str(cfg))
    b, c = L.next(), L.composition(0)
    r = b[0]
    print(f"manifest {L.manifest[:16]}  mix {L.mix[:16]}  packs read {len(L.packs)}")
    print(f"batches allowed {L.batches_allowed} ({L.limit_reason})")
    for note in L.notes:
        print("note:", note)
    print(f"batch 0: {b.shape[0]} rows x {b.shape[2]} tokens, sha {ml.batch_sha256(b)[:16]}")
    print(f"row 0 tokens    {r[0, :12].tolist()}")
    print(f"row 0 piece ids {r[1, :12].tolist()}")
    print(f"row 0 positions {r[2, :12].tolist()}")
    print(f"row 0 holds {int(r[1].max()) + 1} pieces")
    print(f"tokens per lane     {c['by_lane']}")
    print(f"tokens per language {c['by_language']}")
    print(f"pieces not starting at a doc start: {c['nonstart_share']:.0%} of tokens")


def hashes(cfg: Path, n: int) -> list:
    L = ml.MixLoader(str(cfg))
    return [ml.batch_sha256(L.next()) for _ in range(n)]


def same(cfg: Path, n: int) -> bool:
    runs = []
    for k in (1, 2):
        env = dict(os.environ, PYTHONHASHSEED=str(k), PYTHONPATH=str(PKG))
        out = subprocess.run([sys.executable, "-m", "dataio.mix_cli", "_hashes", str(cfg), str(n)], cwd=PKG, env=env,
                             capture_output=True, text=True, check=True).stdout
        runs.append(out.split())
    g = [ml.MixLoader(str(cfg)).global_batch(i) for i in range(n)]
    checks = [(runs[0] == runs[1], f"two separate runs give the same {n} batches"),
              (runs[0] == [ml.batch_sha256(x) for x in g], "random access gives the same batches as reading in order")]
    for W in (2, 4):
        if g[0].shape[0] % W:
            continue
        ok = all(torch.equal(ml.MixLoader(str(cfg), dp_rank=r, dp_world=W).batch(i), g[i][r::W])
                 for r in range(W) for i in range(n))
        checks.append((ok, f"{W} ranks together give the same batches"))
    A, B = ml.MixLoader(str(cfg)), ml.MixLoader(str(cfg))
    for _ in range(n // 2):
        A.next()
    B.load_state_dict(A.state_dict())
    checks.append((all(torch.equal(B.next(), g[i]) for i in range(n // 2, n)),
                   f"resume at batch {n // 2} continues with the same batches"))
    for ok, what in checks:
        print("PASS" if ok else "FAIL", what)
    return all(ok for ok, _ in checks)


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["_hashes"]:                              # child of `same`
        print("\n".join(hashes(Path(argv[1]), int(argv[2]))))
        return 0
    ap = argparse.ArgumentParser(prog="python -m dataio.mix_loader", description=__doc__.split("\n")[0])
    ap.add_argument("action", nargs="?", default="show", choices=ACTIONS, help="default: show")
    ap.add_argument("--data", required=True, help="'synthetic' or a folder of shard-v1 shards")
    ap.add_argument("--weights", default="equal", help="equal, skewed, or a weights file")
    ap.add_argument("--config", default=str(MIX / "stage1_placeholder.yaml"), help="base config (stages, caps)")
    ap.add_argument("--fix-lanes", action="store_true", help="take the lane from domain for general_language manifests")
    ap.add_argument("--batches", type=int, default=20)
    ap.add_argument("--work", help="folder for the resolved config, synthetic shards and proofs (default: a temp dir)")
    a = ap.parse_args(argv)
    work = Path(a.work).resolve() if a.work else Path(tempfile.mkdtemp(prefix="g6-mix-"))
    work.mkdir(parents=True, exist_ok=True)
    cfg = build_config(a, work)
    print("config:", cfg, flush=True)                       # before any child process prints
    if a.action == "summary":
        print(json.dumps(ml.MixLoader(str(cfg)).header(), indent=1, sort_keys=True))
        return 0
    if a.action == "show":
        show(cfg)
        return 0
    if a.action == "same":
        return 0 if same(cfg, a.batches) else 1
    env = dict(os.environ, G6_SHARD_ROOT=yaml.safe_load(cfg.read_text())["shards"]["roots"][0], PYTHONPATH=str(PKG))
    return subprocess.run([sys.executable, str(PROOFS), "--config", str(cfg), "--out", str(work / "proofs"),
                           "--batches", str(max(a.batches, 2)), "--resume-at", str(max(a.batches, 2) * 5 // 8 or 1)],
                          cwd=PKG, env=env).returncode


if __name__ == "__main__":
    sys.exit(main())
