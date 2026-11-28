"""Check the dry run 2 kill drill (job request 2026-11-23-dryrun2-kill-drill.md, Team 07 week 4 task 3).

Reads the files TSAI returns and prints PASS or FAIL for the five pass checks, plus a Markdown table to paste
into results/restore_dryrun2.md. Plain Python, no torch, so it runs on any laptop.

Folders (one per run, as TSAI returns them):
  --run1  out/dryrun2-1234           killed run: metrics.jsonl only (no losses.txt or summary.json after a kill)
  --run2  out/dryrun2-1234-resume-a  first restart
  --run3  out/dryrun2-1234-resume-b  second restart, only if kill (b) was run

Copies TSAI takes right after each kill:
  --latest-a / --commits-a   latest.json and commits.jsonl after kill (a)
  --latest-b / --commits-b   the same after kill (b)

Where the restored step comes from, best first:
  --events DIR               the checkpoint event logs (restore.done, checkpoint.started, checkpoint.committed)
  --restored-a N / --restored-b N   the step from TSAI's note, if the event logs are not returned

Example (PowerShell, one line):
  py check_drill.py --run1 out\\dryrun2-1234 --run2 out\\dryrun2-1234-resume-a --latest-a latest_a.json
      --commits-a commits_a.jsonl --events events
"""
import argparse
import json
import os
import sys


def read_jsonl(path):
    rows = []
    with open(path, encoding="utf8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  note: {path} line {n} is not complete JSON (a kill can cut the last line); skipped")
    return rows


def read_json(path):
    with open(path, encoding="utf8") as f:
        return json.load(f)


def opt(folder, name):
    p = os.path.join(folder, name) if folder else None
    return p if p and os.path.exists(p) else None


def train_losses(metrics_rows):
    """step -> loss, for training rows only (evaluation rows have no 'loss')."""
    out = {}
    for r in metrics_rows:
        if "step" in r and "loss" in r:
            out.setdefault(r["step"], []).append(r["loss"])
    return out


def events_from(folder):
    ev = []
    if folder and os.path.isdir(folder):
        for name in sorted(os.listdir(folder)):
            if name.endswith(".jsonl"):
                ev += read_jsonl(os.path.join(folder, name))
    ev.sort(key=lambda e: e.get("time", ""))
    return ev


class Report:
    def __init__(self):
        self.rows = []

    def add(self, check, ok, detail):
        state = "PASS" if ok is True else ("FAIL" if ok is False else "NOT CHECKED")
        self.rows.append((check, state, detail))
        print(f"[{state}] {check}: {detail}")

    def markdown(self):
        lines = ["| Check | Result | Detail |", "|---|---|---|"]
        for c, s, d in self.rows:
            lines.append(f"| {c} | **{s}** | {d} |")
        return "\n".join(lines)


def compare_rerun(rep, label, before, after, restored):
    """Steps after the restore exist in both runs: the earlier run did them, was killed, the restart redid them."""
    a, b = train_losses(before), train_losses(after)
    redone = sorted(s for s in a if s > restored and s in b)
    if not redone:
        rep.add(f"{label}: re-run steps give the same loss", None,
                f"no overlapping steps after step {restored} (the killed run flushed none past the restore)")
        return
    diffs = [abs(a[s][-1] - b[s][-1]) for s in redone]
    same = sum(1 for d in diffs if d == 0.0)
    worst = max(diffs)
    rep.add(f"{label}: re-run steps give the same loss", worst == 0.0,
            f"steps {redone[0]} to {redone[-1]}: {same} of {len(redone)} equal bit for bit; largest difference {worst:.3e}")
    kept = sorted(s for s in a if s <= restored and s in b)
    if kept:
        bad = [s for s in kept if a[s][-1] != b[s][-1]]
        rep.add(f"{label}: history up to the restore carried over", not bad,
                f"steps {kept[0]} to {kept[-1]}: {len(kept) - len(bad)} of {len(kept)} identical")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run1", required=True)
    p.add_argument("--run2", required=True)
    p.add_argument("--run3")
    p.add_argument("--latest-a")
    p.add_argument("--commits-a")
    p.add_argument("--latest-b")
    p.add_argument("--commits-b")
    p.add_argument("--events")
    p.add_argument("--restored-a", type=int)
    p.add_argument("--restored-b", type=int)
    p.add_argument("--run-id", default="dryrun2-1234")
    a = p.parse_args()

    rep = Report()
    ev = events_from(a.events)
    restores = [e for e in ev if e.get("event") == "restore.done"]
    m1 = read_jsonl(opt(a.run1, "metrics.jsonl")) if opt(a.run1, "metrics.jsonl") else []
    m2 = read_jsonl(opt(a.run2, "metrics.jsonl")) if opt(a.run2, "metrics.jsonl") else []
    last = a.run3 or a.run2

    # Check 1 (kill a): the restart restored the step latest.json pointed to at the kill.
    latest_a = read_json(a.latest_a) if a.latest_a else None
    restored_a = restores[0]["step"] if restores else a.restored_a
    if latest_a is None or restored_a is None:
        rep.add("1. Kill (a): restart lands on the last committed step", None,
                "needs --latest-a and either --events or --restored-a")
    else:
        rep.add("1. Kill (a): restart lands on the last committed step", restored_a == latest_a.get("step"),
                f"latest.json at the kill: step {latest_a.get('step')}; run 2 restored step {restored_a}")
    if a.commits_a and latest_a:
        steps = [c["step"] for c in read_jsonl(a.commits_a)]
        rep.add("1. Kill (a): latest.json is the newest commit", bool(steps) and steps[-1] == latest_a.get("step"),
                f"commits.jsonl steps {steps}")

    # Check 2 (kill b): the half-written save was never committed and was not used.
    if a.run3:
        latest_b = read_json(a.latest_b) if a.latest_b else None
        restored_b = restores[1]["step"] if len(restores) > 1 else a.restored_b
        started = [e["step"] for e in ev if e.get("event") == "checkpoint.started"]
        committed = {e["step"] for e in ev if e.get("event") == "checkpoint.committed"}
        open_saves = [s for s in started if s not in committed]
        half = open_saves[-1] if open_saves else None
        commits_b = [c["step"] for c in read_jsonl(a.commits_b)] if a.commits_b else None
        if latest_b is None or restored_b is None:
            rep.add("2. Kill (b): the half-written save is not used", None,
                    "needs --latest-b and either --events or --restored-b")
        else:
            ok = restored_b == latest_b.get("step")
            detail = f"latest.json at the kill: step {latest_b.get('step')}; run 3 restored step {restored_b}"
            if half is not None:
                in_commits = commits_b is not None and half in commits_b
                ok = ok and restored_b < half and not in_commits
                detail += (f"; the save cut by the kill was for step {half}, "
                           + ("WRONGLY listed in commits.jsonl" if in_commits else "not in commits.jsonl"))
            else:
                detail += "; no unfinished save in the event logs (TSAI's note must say the kill landed mid-save)"
            rep.add("2. Kill (b): the half-written save is not used", ok, detail)
    else:
        rep.add("2. Kill (b): the half-written save is not used", None, "kill (b) was not run (no --run3)")

    # Check 3: one run id everywhere, and commits.jsonl continues across attempts.
    ids = set()
    for folder in (a.run2, a.run3):
        s = opt(folder, "summary.json")
        if s:
            ids.add(read_json(s).get("run_id"))
        m = opt(folder, "manifest.json")
        if m:
            ids.add((read_json(m).get("team7_checkpoint") or {}).get("run_id"))
    ids.discard(None)
    rep.add("3. Same run id in every run", ids == {a.run_id}, f"run ids found: {sorted(ids) or 'none'}")
    final_commits = a.commits_b or a.commits_a
    if final_commits:
        cs = read_jsonl(final_commits)
        steps = [c["step"] for c in cs]
        epochs = [c.get("epoch") for c in cs]
        ok = steps == sorted(set(steps)) and all(x is None or y is None or x <= y for x, y in zip(epochs, epochs[1:]))
        rep.add("3. commits.jsonl continues from run to run", ok,
                f"{len(steps)} commits, steps rising, epochs {sorted(set(e for e in epochs if e is not None))}")

    # Check 4: the steps redone after each restore give the same loss.
    if restored_a is not None and m1 and m2:
        compare_rerun(rep, "4. Kill (a)", m1, m2, restored_a)
    else:
        rep.add("4. Kill (a): re-run steps give the same loss", None, "needs run 1 and run 2 metrics.jsonl and the restored step")
    if a.run3:
        m3 = read_jsonl(opt(a.run3, "metrics.jsonl")) if opt(a.run3, "metrics.jsonl") else []
        rb = restores[1]["step"] if len(restores) > 1 else a.restored_b
        if rb is not None and m2 and m3:
            compare_rerun(rep, "4. Kill (b)", m2, m3, rb)

    # Check 5: the finished run has every step once and is COMPLETED.
    s = opt(last, "summary.json")
    summ = read_json(s) if s else {}
    total = summ.get("steps")
    rep.add("5. Last run finished", summ.get("status") == "COMPLETED",
            f"status {summ.get('status')}, steps {total}, final_model_sha256 {str(summ.get('final_model_sha256'))[:12]}...")
    mf = opt(last, "metrics.jsonl")
    if mf and total:
        tl = train_losses(read_jsonl(mf))
        dup = sorted(st for st, v in tl.items() if len(v) > 1)
        missing = sorted(set(range(1, total + 1)) - set(tl))
        rep.add("5. No lost steps (metrics.jsonl)", not dup and not missing,
                f"{len(tl)} training steps; missing {missing[:10] or 'none'}; twice {dup[:10] or 'none'}")
        lf = opt(last, "losses.txt")
        if lf:
            hexes = [ln.strip() for ln in open(lf, encoding="utf8") if ln.strip()]
            agree = all(float.fromhex(h) == tl.get(i, [None])[-1] for i, h in enumerate(hexes, 1))
            rep.add("5. No lost steps (losses.txt)", len(hexes) == total and agree,
                    f"{len(hexes)} lines for {total} steps; every line equals metrics.jsonl: {agree}")
    else:
        rep.add("5. No lost steps", None, "needs the last run's summary.json and metrics.jsonl")

    print("\n--- paste into results/restore_dryrun2.md ---\n")
    print(rep.markdown())
    return 1 if any(state == "FAIL" for _, state, _ in rep.rows) else 0


if __name__ == "__main__":
    sys.exit(main())
