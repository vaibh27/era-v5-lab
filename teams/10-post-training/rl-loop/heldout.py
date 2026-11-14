"""Held-out scoring with an honest error bar.

pass@1 per problem = mean reward over its k samples. The run is judged on the change in mean pass@1 between
step 0 and the end, measured on fresh samples each time (never the pass-rate filter's numbers: the filter kept
tasks whose first k samples were mixed, so re-using those numbers would bias step 0).

paired_bootstrap resamples PROBLEMS (not samples) with replacement, because the same problem is scored at both
points and problems differ far more than samples do. The 95% interval is the 2.5th to 97.5th percentile of the
resampled mean differences.
"""
from __future__ import annotations

import torch


def paired_bootstrap(start: list[float], end: list[float], n: int = 10_000, seed: int = 0,
                     alpha: float = 0.05) -> dict:
    if len(start) != len(end) or not start:
        raise ValueError("start and end must be non-empty and score the same problems")
    s = torch.tensor(start, dtype=torch.float64)
    e = torch.tensor(end, dtype=torch.float64)
    d = e - s
    g = torch.Generator().manual_seed(seed)
    idx = torch.randint(0, len(d), (n, len(d)), generator=g)
    means = d[idx].mean(1)
    lo, hi = torch.quantile(means, torch.tensor([alpha / 2, 1 - alpha / 2], dtype=torch.float64)).tolist()
    return {"problems": len(d), "start": round(float(s.mean()), 4), "end": round(float(e.mean()), 4),
            "diff": round(float(d.mean()), 4), "ci_low": round(lo, 4), "ci_high": round(hi, 4),
            "ci_excludes_zero": bool(lo > 0 or hi < 0), "resamples": n}
