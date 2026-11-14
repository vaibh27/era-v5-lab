"""Score the Stage B generations from stage_b_modal.py locally and write the per-tier readout.

    python eval/stage_b_report.py --raw RAW_DIR --checker-dir RL_DATA_DIR \
        --source-dir code_contests=DIR --source-dir hendrycks_math=DIR --out OUT_DIR

For each sample directory (sample0, sample1) the records are re-read, prompts and tests are resolved and
hash-checked from the pinned files, and functional success is filled in with the rl-data checkers
(math_check: math-verify; stdio_check: Docker, refused otherwise). Nothing else in a record changes. Then the
suite's own `summarise` runs per sample, and the two samples are pooled per label (k = 2 samples per cell).

Per label (110 cells = 55 prompts x 2 samples), the three things the suite keeps apart:
    parse    completion parses as one assistant turn with no malformed or unclosed trace (suite `analyse`)
    limit    stopped on its own inside the 1024 cap (not censored)
    correct  checker success; censored cells stay in the denominator as failures
    obey     generated tokens <= the requested label budget. With one common cap of 1024 and budgets
             1024..65536 this always holds, so it is reported but says nothing about the label.

Pass/Fail rules (fixed in this file before the run):
    Format obeyed  PASS iff parse == ran (every cell parses)
    Cap respected  PASS iff limit == ran (no cell censored)
    Answer correct PASS iff correct >= 1 (the checker path awards verified credit at least once at this label);
                   a canary bar for the pipeline, not a quality bar
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import effort_suite as es  # noqa: E402


def pool(per_sample: list[list[dict]]) -> dict:
    rows = {}
    for label in es.LABELS:
        rs = [r for recs in per_sample for r in recs if r["requested_effort"] == label and r.get("ran")]
        ok = [r for r in rs if r["functional"]["success"] is not None]
        by_cat = {}
        for cat in sorted({r["category"] for r in rs}):
            c = [r for r in ok if r["category"] == cat]
            by_cat[cat] = {"correct": sum(r["functional"]["success"] for r in c), "checked": len(c)}
        solved_any = {}
        for r in ok:
            solved_any[r["slot"]] = max(solved_any.get(r["slot"], 0), r["functional"]["success"])
        n = len(rs)
        parse = sum(r["parse_status"] == "ok" for r in rs)
        limit = sum(not r["censored"] for r in rs)
        correct = sum(r["functional"]["success"] for r in ok)
        rows[label] = {
            "ran": n, "parse": parse, "malformed": sum(r["parse_status"] == "malformed" for r in rs),
            "empty": sum(r["parse_status"] == "empty" for r in rs), "limit": limit, "censored": n - limit,
            "obey_budget": sum(r["generated_tokens"] <= r["requested_budget_tokens"] for r in rs),
            "correct": correct, "checked": len(ok), "unchecked": n - len(ok),
            "correct_by_category": by_cat,
            "prompts_solved_in_either_sample": sum(solved_any.values()), "prompts": len({r["slot"] for r in rs}),
            "generated_tokens_median": sorted(r["generated_tokens"] for r in rs)[n // 2] if n else None,
            "lama": {"format_obeyed": "PASS" if n and parse == n else "FAIL",
                     "cap_respected": "PASS" if n and limit == n else "FAIL",
                     "answer_correct": "PASS" if correct >= 1 else "FAIL"},
        }
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True)
    ap.add_argument("--checker-dir", required=True)
    ap.add_argument("--source-dir", action="append", default=[])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if os.environ.get("G10_STDIO_RUNNER", "docker") != "docker":
        raise SystemExit("refusing: code must be checked in Docker (G10_STDIO_RUNNER=docker)")
    manifest = es.load_manifest(HERE / "effort_manifest.json")
    checkers, why, last_boxed = es.load_checkers(Path(a.checker_dir), execute_code=True)
    if checkers["math"] is None or checkers["stdio"] is None:
        raise SystemExit(f"checkers unavailable: {why}")
    prompts = es.resolve_prompts(manifest, es._dirs(a.source_dir), last_boxed=last_boxed,
                                 slots=es.stage_slots(manifest, "B"))
    raw, out = Path(a.raw), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads((raw / "run_meta.json").read_text(encoding="utf-8"))
    per_sample = []
    for sdir in sorted(raw.glob("sample*")):
        recs = [json.loads(x) for x in (sdir / "effort_stageB.jsonl").read_text(encoding="utf-8").splitlines()
                if x.strip()]
        for r in recs:
            if r["manifest_sha256"] != manifest["manifest_sha256"]:
                raise SystemExit(f"{sdir}: record from a different manifest")
            if not r.get("ran"):
                continue
            p = prompts[r["slot"]]
            try:
                res = checkers[p.checker](p.task, r["completion"])
                r["functional"] = {"checker": p.checker, "success": int(res["reward"]), "reason": res.get("reason")}
            except Exception as exc:  # noqa: BLE001  a checker crash is recorded, not hidden
                r["functional"] = {"checker": p.checker, "success": None, "reason": f"checker error: {exc!r}"}
        summary = es.summarise(recs, manifest, "B")
        summary.update(backend=recs[0]["backend"], template=recs[0]["template"], model_id=recs[0]["model_id"],
                       context=meta["model"]["max_model_len"], temperature=meta["temperature"],
                       base_seed=next(s["base_seed"] for s in meta["samples"] if f"sample{s['sample']}" == sdir.name),
                       scored_locally=True)
        d = out / sdir.name
        d.mkdir(exist_ok=True)
        (d / "effort_stageB.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs),
                                               encoding="utf-8")
        (d / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        per_sample.append(recs)
    pooled = {"artifact": es.OUTPUT_ARTIFACT, "do_not_train": True, "stage": "B",
              "manifest_sha256": manifest["manifest_sha256"], "samples": len(per_sample),
              "model_id": per_sample[0][0]["model_id"], "temperature": meta["temperature"],
              "sample_base_seeds": meta["sample_base_seeds"], "cap": meta["cap"], "per_label": pool(per_sample),
              "note": "OLMoE-1B-7B-0924-Instruct native template, Stage B only: an optimization diagnostic, not "
                      "the capstone model. No comparison is implied."}
    (out / "stageB_pooled.json").write_text(json.dumps(pooled, indent=1, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    print("label  ran parse limit obey correct  pass@2  format cap correct")
    for lab, r in pooled["per_label"].items():
        print(f"{lab:>5} {r['ran']:4d} {r['parse']:5d} {r['limit']:5d} {r['obey_budget']:4d} "
              f"{r['correct']:4d}/{r['checked']:<2d} {r['prompts_solved_in_either_sample']:3d}/{r['prompts']:<2d}  "
              f"{r['lama']['format_obeyed']:6s} {r['lama']['cap_respected']:4s} {r['lama']['answer_correct']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
