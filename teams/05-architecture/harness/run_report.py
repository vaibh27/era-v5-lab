"""Report on a G5 training run from its metrics.jsonl (train_week2.py output).

Reads only the metrics file. Prints a markdown report: training loss, held-out bits per byte
(overall and per slice), learning rate, real-token throughput and row fill, and MoE expert
load per layer when the run recorded it (--expert-load-every).

    python teams/05-architecture/harness/run_report.py out/dryrun2-1234/metrics.jsonl
    python teams/05-architecture/harness/run_report.py metrics.jsonl --out report.md
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

# TSAI's router comparison with balancing on (05 Architecture, 24 Nov 2026), router on Muon,
# bias_gamma 0.001, 1,221 steps at T2: the reference the dry run 2 expert load is read against.
REFERENCE = {"busiest_low": 0.073, "busiest_high": 0.090, "max_over_mean": 1.44, "cv": 0.105}


def read_rows(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{n}: not JSON ({e})")
    if not rows:
        raise SystemExit(f"{path}: no rows")
    return rows


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(x)


def mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def layer_load(counts):
    """Share of the busiest expert, max/mean, coefficient of variation and dead experts, for one layer."""
    total = sum(counts)
    if total <= 0:
        return None
    m = total / len(counts)
    return {
        "busiest": max(counts) / total,
        "max_over_mean": max(counts) / m,
        "cv": statistics.pstdev(counts) / m,
        "dead": sum(1 for c in counts if c == 0),
        "experts": len(counts),
    }


def report(rows, name):
    out = [f"# Run report: {name}", ""]
    train = [r for r in rows if "loss" in r and r.get("step", 0) > 0]
    evals = [r for r in rows if "val/bpb" in r]
    loads = [r for r in rows if "expert_load" in r]

    # Health: every loss and gradient norm finite.
    bad = [r["step"] for r in train if not finite(r.get("loss")) or not finite(r.get("gradient_norm", 0.0))]
    steps = max((r.get("step", 0) for r in rows), default=0)
    out += ["## Health", "",
            f"- Steps recorded: {len(train)} training rows, last step {steps}.",
            f"- Non-finite loss or gradient norm: {'none' if not bad else f'{len(bad)} rows, first at step {bad[0]}'}.",
            ""]

    if train:
        k = max(1, min(50, len(train) // 10))
        first, last = [r["loss"] for r in train[:k]], [r["loss"] for r in train[-k:]]
        clipped = sum(1 for r in train if r.get("gradient_clipping_applied"))
        norms = [r["gradient_norm"] for r in train if finite(r.get("gradient_norm"))]
        out += ["## Training loss", "",
                f"- Mean of the first {k} steps: {mean(first):.4f}; mean of the last {k}: {mean(last):.4f}; "
                f"lowest single step: {min(r['loss'] for r in train):.4f}.",
                f"- Gradient norm before clipping: median {statistics.median(norms):.3f}, max {max(norms):.3f}; "
                f"clipped on {clipped} of {len(train)} steps ({clipped / len(train):.1%})." if norms else "- No gradient norms.",
                ""]
        lrs = [r["lr"] for r in train if finite(r.get("lr"))]
        if lrs:
            out += ["## Learning rate", "",
                    f"- First {lrs[0]:.3g}, peak {max(lrs):.3g} (first reached at step "
                    f"{next(r['step'] for r in train if r.get('lr') == max(lrs))}), last {lrs[-1]:.3g}.", ""]
        scored = sum(r.get("train/scored_tokens", 0) for r in train)
        positions = sum(r.get("train/input_positions", 0) for r in train)
        ms = sum(r.get("perf/step_ms", 0.0) for r in train)
        if scored and ms:
            out += ["## Throughput", "",
                    f"- Real (scored) tokens per second: {scored / (ms / 1000):,.0f} over {ms / 3.6e6:.2f} h of step time.",
                    f"- Row fill: {scored:,} scored tokens in {positions:,} input positions ({scored / positions:.1%}).",
                    "- The trainer's tokens_per_second counts padded positions; the line above does not.", ""]

    if evals:
        out += ["## Held-out bits per byte", "", "| Step | val/bpb |", "|---:|---:|"]
        out += [f"| {r['step']} | {r['val/bpb']:.4f} |" for r in evals]
        a, b = evals[0], evals[-1]
        slices = sorted({k.split("/")[2] for k in b if k.startswith("val/slices/") and k.endswith("/bpb")})
        if slices:
            out += ["", f"Per slice, step {a['step']} to step {b['step']}:", "",
                    "| Slice | First | Last | Change |", "|---|---:|---:|---:|"]
            for s in slices:
                x, y = a.get(f"val/slices/{s}/bpb"), b.get(f"val/slices/{s}/bpb")
                if finite(x) and finite(y):
                    out.append(f"| {s} | {x:.4f} | {y:.4f} | {y - x:+.4f} |")
        out.append("")

    out += ["## Expert load", ""]
    if not loads:
        out += ["- None recorded (the run did not pass --expert-load-every).", ""]
    else:
        out += [f"- Recorded at {len(loads)} steps (first {loads[0]['step']}, last {loads[-1]['step']}).",
                f"- Reference, TSAI 24 Nov (balanced, router on Muon): busiest expert "
                f"{REFERENCE['busiest_low']:.1%} to {REFERENCE['busiest_high']:.1%} per layer, max/mean "
                f"{REFERENCE['max_over_mean']}, CV {REFERENCE['cv']}.", ""]
        last = loads[-1]
        stats = [layer_load(c) for c in last["expert_load"]]
        n = stats[0]["experts"] if stats and stats[0] else 0
        out += [f"At step {last['step']} (fair share {1 / n:.2%} with {n} experts):" if n else "", "",
                "| Layer | Busiest expert | Max/mean | CV | Dead experts |", "|---:|---:|---:|---:|---:|"]
        for i, st in enumerate(stats):
            out.append(f"| {i} | {st['busiest']:.1%} | {st['max_over_mean']:.2f} | {st['cv']:.3f} | {st['dead']} |"
                       if st else f"| {i} | no tokens | | | |")
        out += ["", "Busiest expert over the run (worst layer at each recorded step):", "",
                "| Step | Worst busiest share | Layers with a dead expert |", "|---:|---:|---:|"]
        for r in loads:
            s = [layer_load(c) for c in r["expert_load"]]
            s = [x for x in s if x]
            out.append(f"| {r['step']} | {max(x['busiest'] for x in s):.1%} | {sum(1 for x in s if x['dead'])} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("metrics", type=Path)
    ap.add_argument("--out", type=Path, help="also write the report here")
    args = ap.parse_args()
    text = report(read_rows(args.metrics), args.metrics.parent.name or args.metrics.name)
    print(text)
    if args.out:
        args.out.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
