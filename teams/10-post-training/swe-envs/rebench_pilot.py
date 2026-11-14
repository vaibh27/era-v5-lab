"""Admission checks for SWE-rebench pilot environments (G10, D5 pilot: 5 environments).

An environment in `rebench-pilot/<instance_id>/` is built by `make_env.py --out-dir rebench-pilot` from the
upstream PR. This script decides whether it is admitted, and records the evidence in
`rebench-pilot/<instance_id>/admission.json`. It writes no issue text, test or fix; the only content it makes
is a harness-side negative control (a comment-only patch), which is kept in admission.json and never used as
training data.

For each environment and each platform (`--platforms`, default: the Docker host's own and linux/amd64):

  1. Source record. Read the row from the pinned SWE-rebench parquet files (`--source-dir`, checked against
     the sha256 values in `rebench-pilot/source_manifest.json`) and compare it with the rebuild: base commit,
     FAIL_TO_PASS and PASS_TO_PASS. Only the row's id, repo, dates, base commit and test lists are read; the
     release's install recipes and quality labels are not read.
  2. Gold check, N runs per side (default 5), each in a fresh `docker run --rm` container with no network:
       before = base + hidden tests:         every FAIL_TO_PASS test is reported and fails, every PASS_TO_PASS
                                              test passes
       after  = base + gold + hidden tests:  every listed test is reported and passes
     A run with a collection error (a junit case with no class name), a missing listed test or a failed
     report-integrity check (verify.py's sentinel, exit code and summary checks) fails admission.
  3. Reward path: verify.reward() gives 1 for the gold patch and 0 for the negative control.
  4. Isolation probe in the env image with verify.py's Docker limits: no interface is up except `lo` (the
     kernel may list tunnel devices such as sit0 or tunl0, which stay down), there is no IPv4 route, an
     outbound TCP connect and a DNS lookup fail, the repo has exactly one commit, and the user is not root.
  5. No container is left behind after the runs.

Usage:
    python rebench_pilot.py --source-dir DIR [--runs 5] [--platforms native linux/amd64] [ENV_ID ...]
    python rebench_pilot.py --report       # rewrite ../results/swe_rebench_pilot.md from admission.json files
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_env  # noqa: E402
import verify  # noqa: E402

PILOT = HERE / "rebench-pilot"
MANIFEST = PILOT / "source_manifest.json"
REPORT = HERE.parent / "results" / "swe_rebench_pilot.md"
SOURCE_COLUMNS = ["instance_id", "repo", "base_commit", "created_at", "license_name", "FAIL_TO_PASS",
                  "PASS_TO_PASS"]
NEGATIVE_LINE = "# g10 negative control: comment only, no behaviour change"
PROBE = r"""
import json, os, socket, subprocess
net = "/sys/class/net"
names = sorted(i for i in os.listdir(net) if os.path.isdir(os.path.join(net, i)))
state = {i: open(os.path.join(net, i, "operstate")).read().strip() for i in names}
routes = [l for l in open("/proc/net/route").read().splitlines()[1:] if l.strip()]
out = {"interfaces_up": sorted(i for i in names if state[i] != "down"),
       "interfaces_down": sorted(i for i in names if state[i] == "down"), "ipv4_routes": len(routes),
       "uid": os.getuid()}
try:
    socket.create_connection(("1.1.1.1", 443), timeout=3).close(); out["tcp_connect"] = "connected"
except OSError as e:
    out["tcp_connect"] = "failed: " + type(e).__name__
try:
    socket.getaddrinfo("pypi.org", 443); out["dns"] = "resolved"
except OSError as e:
    out["dns"] = "failed: " + type(e).__name__
out["commits"] = int(subprocess.run(["git", "rev-list", "--count", "HEAD"], capture_output=True, text=True,
                                    cwd="/repo").stdout.strip() or -1)
print(json.dumps(out))
"""


# --- source record ----------------------------------------------------------------------------

def norm_test_id(test_id: str) -> str:
    """'tests/test_x.py::Cls::test_a[p]' (SWE-rebench) and 'tests.test_x.Cls::test_a[p]' (junit) -> one form."""
    if ".py::" not in test_id:
        return test_id
    path, rest = test_id.split(".py::", 1)
    parts = rest.split("::")
    return ".".join([path.replace("/", "."), *parts[:-1]]) + "::" + parts[-1]


def cmp_key(test_id: str) -> str:
    """Comparison key: SWE-rebench cuts test ids at the first space ('test_issue26[-- hello]' -> 'test_issue26[--')."""
    return test_id.split()[0] if test_id.split() else test_id


def source_rows(source_dir: Path, ids: set[str]) -> dict[str, dict]:
    import pyarrow.parquet as pq

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = {}
    for f in manifest["files"]:
        path = source_dir / f["path"].rsplit("/", 1)[-1]
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        if got != f["sha256"]:
            raise SystemExit(f"{path}: sha256 {got} is not the pinned {f['sha256']}")
        for row in pq.read_table(path, columns=SOURCE_COLUMNS).to_pylist():
            if row["instance_id"].lower() in ids:
                rows[row["instance_id"].lower()] = row
    return rows


def compare_source(env: dict, row: dict | None) -> dict:
    if row is None:
        return {"found": False, "variant": "not in the pinned split"}
    ours_f2p = {cmp_key(t) for t in env["FAIL_TO_PASS"]}
    ours_p2p = {cmp_key(t) for t in env["PASS_TO_PASS"]}
    src_f2p = {cmp_key(norm_test_id(t)) for t in row["FAIL_TO_PASS"]}
    src_p2p = {cmp_key(norm_test_id(t)) for t in row["PASS_TO_PASS"]}
    # SWE-rebench lists tests from every test file of the repo; the rebuild runs only the PR's test files.
    in_scope = {t for t in src_p2p if any(t.startswith(f[:-3].replace("/", ".") + "::") or
                                         t.startswith(f[:-3].replace("/", ".") + ".")
                                         for f in env["test_files"] if f.endswith(".py"))}
    out = {
        "found": True,
        "source_instance_id": row["instance_id"],
        "source_created_at": str(row["created_at"]),
        "source_licence_name": row["license_name"],
        "base_commit_matches": row["base_commit"] == env["base_commit"],
        "source_base_commit": row["base_commit"],
        "f2p_same": src_f2p == ours_f2p,
        "f2p_only_source": sorted(src_f2p - ours_f2p),
        "f2p_only_rebuild": sorted(ours_f2p - src_f2p),
        "p2p_source_total": len(src_p2p),
        "p2p_source_in_rebuild_files": len(in_scope),
        "p2p_in_files_only_source": sorted(in_scope - ours_p2p),
        "p2p_only_rebuild": sorted(ours_p2p - src_p2p),
    }
    same = out["base_commit_matches"] and out["f2p_same"] and not out["p2p_in_files_only_source"] \
        and not out["p2p_only_rebuild"]
    diffs = []
    if not out["base_commit_matches"]:
        diffs.append(f"base is merge-base(PR base, PR head) `{env['base_commit'][:12]}`, source row has "
                     f"`{row['base_commit'][:12]}`")
    if not out["f2p_same"]:
        diffs.append(f"FAIL_TO_PASS only in source {out['f2p_only_source']}, only in rebuild {out['f2p_only_rebuild']}")
    if out["p2p_in_files_only_source"]:
        diffs.append(f"{len(out['p2p_in_files_only_source'])} in-file PASS_TO_PASS tests of the source row are not "
                     "PASS_TO_PASS in the rebuild (they fail on both sides here, see env.json build_notes)")
    if out["p2p_only_rebuild"]:
        diffs.append(f"{len(out['p2p_only_rebuild'])} PASS_TO_PASS tests only in the rebuild")
    out["differences"] = diffs
    out["variant"] = ("same base, FAIL_TO_PASS and in-file PASS_TO_PASS as the source row" if same else
                      "G10 rebuilt variant: " + "; ".join(diffs))
    return out


# --- negative control -------------------------------------------------------------------------

def negative_patch(env: dict, gold: str) -> str:
    """A comment-only change to the first file the gold patch touches. It applies, but fixes nothing."""
    path = verify.patch_paths(gold)[0]
    owner, name = env["repo"].split("/")
    cache = make_env.CACHE / f"{owner}__{name}"
    text = make_env.sh(["git", "-C", str(cache), "show", f"{env['base_commit']}:{path}"])
    old = text.splitlines(keepends=True)
    new = [NEGATIVE_LINE + "\n", *old]
    body = "".join(difflib.unified_diff(old, new, f"a/{path}", f"b/{path}", n=3))
    return f"diff --git a/{path} b/{path}\n" + body


# --- one platform -----------------------------------------------------------------------------

def collection_errors(results: dict[str, bool]) -> list[str]:
    return sorted(k for k in results if k.startswith("::"))


def side_check(env: dict, side: dict, after: bool) -> list[str]:
    if "error" in side:
        return [side["error"]]
    res = side["results"]
    problems = [f"collection error: {k}" for k in collection_errors(res)]
    for t in env["FAIL_TO_PASS"]:
        if t not in res:
            problems.append(f"FAIL_TO_PASS test missing from the report: {t}")
        elif res[t] != after:
            problems.append(f"FAIL_TO_PASS test {'fails' if after else 'passes'}: {t}")
    for t in env["PASS_TO_PASS"]:
        if t not in res:
            problems.append(f"PASS_TO_PASS test missing from the report: {t}")
        elif not res[t]:
            problems.append(f"PASS_TO_PASS test fails: {t}")
    return problems


def probe(env: dict) -> dict:
    cmd = ["docker", "run", "--rm", *verify.DOCKER_LIMITS,
           *(["--platform", verify.PLATFORM] if verify.PLATFORM else []),
           verify.image_ref(env), "python", "-c", PROBE]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError):
        return {"ok": False, "error": (proc.stdout + proc.stderr)[-500:]}
    out["ok"] = (out["interfaces_up"] == ["lo"] and out["ipv4_routes"] == 0 and out["tcp_connect"].startswith("failed")
                 and out["dns"].startswith("failed") and out["commits"] == 1 and out["uid"] != 0)
    return out


def leftover_containers() -> list[str]:
    out = subprocess.run(["docker", "ps", "-a", "--filter", "name=g10swe-", "--format", "{{.Names}}"],
                         capture_output=True, text=True).stdout
    return out.split()


def check_platform(env_dir: Path, env: dict, runs: int, platform: str | None) -> dict:
    verify.PLATFORM = platform
    t0 = time.monotonic()
    image_id = verify.ensure_image(env_dir, env)
    build_s = round(time.monotonic() - t0, 1)
    arch = subprocess.run(["docker", "image", "inspect", "-f", "{{.Os}}/{{.Architecture}}", verify.image_ref(env)],
                          capture_output=True, text=True).stdout.strip()
    gold = (env_dir / "gold.patch").read_text(encoding="utf-8")
    before, after = [], []
    for i in range(runs):
        b = verify.run_side(env_dir, env, None)
        a = verify.run_side(env_dir, env, gold)
        before.append({"run": i + 1, "ok": not side_check(env, b, False), "problems": side_check(env, b, False),
                       "seconds": b.get("seconds")})
        after.append({"run": i + 1, "ok": not side_check(env, a, True), "problems": side_check(env, a, True),
                      "seconds": a.get("seconds")})
    neg = negative_patch(env, gold)
    neg_reward = verify.reward(env_dir, neg)
    gold_reward = verify.reward(env_dir, gold)
    iso = probe(env)
    left = leftover_containers()
    ok = (all(r["ok"] for r in before) and all(r["ok"] for r in after) and neg_reward["reward"] == 0
          and gold_reward["reward"] == 1 and iso["ok"] and not left and bool(env["FAIL_TO_PASS"]))
    return {"platform": platform or "native", "image_platform": arch, "image": verify.image_ref(env),
            "image_id": image_id, "build_seconds_if_built": build_s, "runs": runs,
            "no_patch_fails": f"{sum(r['ok'] for r in before)}/{runs}",
            "gold_passes": f"{sum(r['ok'] for r in after)}/{runs}",
            "before": before, "after": after,
            "negative_control": {"patch": neg, "reward": neg_reward},
            "gold_reward": gold_reward, "isolation_probe": iso, "leftover_containers": left,
            "docker_flags": verify.DOCKER_LIMITS, "ok": ok}


# --- report -----------------------------------------------------------------------------------

def write_report(path: Path = REPORT) -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    rows = []
    for d in sorted(p.parent for p in PILOT.glob("*/admission.json")):
        rows.append((json.loads((d / "env.json").read_text(encoding="utf-8")),
                     json.loads((d / "admission.json").read_text(encoding="utf-8"))))
    lines = [
        "# SWE-rebench pilot: admission results",
        "",
        f"Five environments rebuilt from the upstream PRs behind rows of `{manifest['dataset']}` at revision "
        f"`{manifest['revision']}`, split `{manifest['split']}` (CC-BY-4.0; attribution in "
        "`licences/licence_table.md`). Written by `swe-envs/rebench_pilot.py --report` from each "
        "`rebench-pilot/<id>/admission.json`.",
        "",
        "Each side ran in a fresh `docker run --rm` container with "
        f"`{' '.join(verify.DOCKER_LIMITS)}`. `before` = base commit + hidden tests (no patch); `after` = base + "
        "upstream fix + hidden tests. A run counts only if every listed test is reported, there is no collection "
        "error and verify.py's report-integrity checks pass. The negative control is a comment-only change to "
        "the first file the fix touches.",
        "",
        "| instance_id | base commit | fix (PR head / merge) | platform | no patch fails | gold passes | "
        "negative reward | gold reward | isolation | max run (s) | admitted |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for env, adm in rows:
        for p in adm["platforms"]:
            secs = [r["seconds"] for r in p["before"] + p["after"] if r["seconds"] is not None]
            lines.append(
                f"| `{env['instance_id']}` | `{env['base_commit'][:12]}` | `{env['pr_head_commit'][:12]}` / "
                f"`{(env.get('merge_commit') or '')[:12]}` | {p['image_platform']} | {p['no_patch_fails']} | "
                f"{p['gold_passes']} | {p['negative_control']['reward']['reward']} | {p['gold_reward']['reward']} | "
                f"{'pass' if p['isolation_probe']['ok'] else 'FAIL'} | {max(secs) if secs else '-'} | "
                f"{'yes' if p['ok'] else 'NO'} |")
    lines += ["", "## Source record versus rebuild", "",
              "| instance_id | issue opened | PR merged | licence | same base | same FAIL_TO_PASS | "
              "in-file PASS_TO_PASS ids, cut at the first space (source / rebuild) | variant |",
              "|---|---|---|---|---|---|---|---|"]
    for env, adm in rows:
        s = adm["source"]
        lines.append(
            f"| `{env['instance_id']}` | {', '.join(d[:10] for d in env['issue_created_at'])} | "
            f"{env['merged_at'][:10]} | {env['licence']} | {s.get('base_commit_matches')} | {s.get('f2p_same')} | "
            f"{s.get('p2p_source_in_rebuild_files')} / {len({cmp_key(t) for t in env['PASS_TO_PASS']})} | "
            f"{s.get('variant')} |")
    lines += ["", "## Rejected candidates", "", "| candidate | stage | reason |", "|---|---|---|"]
    for r in manifest.get("rejected", []):
        lines.append(f"| `{r['instance_id']}` | {r['stage']} | {r['reason']} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ids", nargs="*", help="env folder names under rebench-pilot/ (default: all)")
    ap.add_argument("--source-dir", type=Path, help="folder with the pinned SWE-rebench parquet files")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--platforms", nargs="+", default=["native", "linux/amd64"])
    ap.add_argument("--report", action="store_true", help="only rewrite the results table")
    ap.add_argument("--refresh-source", action="store_true",
                    help="recompute only the source comparison in existing admission.json files, then the report")
    args = ap.parse_args(argv)
    if args.report:
        write_report()
        return 0
    if args.refresh_source:
        if args.source_dir is None:
            ap.error("--refresh-source needs --source-dir")
        dirs = sorted(p.parent for p in PILOT.glob("*/admission.json"))
        src = source_rows(args.source_dir, {d.name for d in dirs})
        for d in dirs:
            adm = json.loads((d / "admission.json").read_text(encoding="utf-8"))
            adm["source"] = compare_source(verify.load_env(d), src.get(d.name))
            adm["admitted"] = all(p["ok"] for p in adm["platforms"]) and adm["source"]["found"]
            (d / "admission.json").write_text(json.dumps(adm, indent=1) + "\n", encoding="utf-8")
        write_report()
        return 0
    if args.source_dir is None:
        ap.error("--source-dir is required to check environments")
    dirs = [PILOT / i for i in args.ids] if args.ids else sorted(p.parent for p in PILOT.glob("*/env.json"))
    src = source_rows(args.source_dir, {d.name for d in dirs})
    all_ok = True
    for d in dirs:
        env = verify.load_env(d)
        adm = {"instance_id": env["instance_id"], "checked_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
               "source": compare_source(env, src.get(env["instance_id"])), "platforms": []}
        for plat in args.platforms:
            adm["platforms"].append(check_platform(d, env, args.runs, None if plat == "native" else plat))
            p = adm["platforms"][-1]
            print(json.dumps({"instance_id": env["instance_id"], "platform": p["image_platform"],
                              "no_patch_fails": p["no_patch_fails"], "gold_passes": p["gold_passes"],
                              "negative": p["negative_control"]["reward"]["reward"],
                              "gold": p["gold_reward"]["reward"], "isolation": p["isolation_probe"]["ok"],
                              "ok": p["ok"]}), flush=True)
        adm["admitted"] = all(p["ok"] for p in adm["platforms"]) and adm["source"]["found"]
        all_ok &= adm["admitted"]
        (d / "admission.json").write_text(json.dumps(adm, indent=1) + "\n", encoding="utf-8")
    write_report()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
