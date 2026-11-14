"""Pass-rate filter: keep the tasks a given model sometimes solves.

GRPO learns only from groups whose rewards differ. A task the model never solves (or always solves) adds no
signal, only cost (DAPO's dynamic sampling drops such groups, arXiv 2503.14476). So before RL, sample k answers
per task from the starting model and keep tasks with 0 < pass rate < 1. Rerun it for each new starting model
(OLMoE now, Team 5's model after 18 Nov).

    python filter.py --tasks out/tasks.jsonl --server http://127.0.0.1:8080 --k 8 --out out/manifest.json

The sampler is any OpenAI-compatible chat endpoint (for example llama.cpp's llama-server serving a GGUF of the
starting model). The manifest lists every task with its pass rate and the keep decision, plus the model and
sampling settings; it holds ids and numbers, not task text.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import time
import urllib.request
from pathlib import Path
from typing import Callable

import stdio_check

PROMPT_SUFFIX = ("\n\nWrite a Python 3 program that reads the input from standard input and prints the answer to "
                 "standard output. Reply with the complete program in one ```python block.")


def chat_sampler(server: str, temperature: float, max_tokens: int) -> Callable[[str, int], str]:
    def sample(prompt: str, seed: int) -> str:
        body = {"messages": [{"role": "user", "content": prompt}], "temperature": temperature, "top_p": 0.95,
                "max_tokens": max_tokens, "seed": seed}
        req = urllib.request.Request(server.rstrip("/") + "/v1/chat/completions", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return json.load(r)["choices"][0]["message"]["content"]
    return sample


def decide(passes: list[int]) -> str:
    rate = sum(passes) / len(passes)
    return "keep" if 0 < rate < 1 else ("too_hard" if rate == 0 else "too_easy")


def run_filter(tasks: list[dict], sample: Callable[[str, int], str], k: int,
               score: Callable[[dict, str], dict] = stdio_check.score, workers: int = 4) -> list[dict]:
    jobs = [(t, s) for t in tasks for s in range(k)]

    def one(job):
        t, s = job
        try:
            return t["id"], score(t, sample(t["statement"] + PROMPT_SUFFIX, 1000 + s))["reward"]
        except Exception:  # noqa: BLE001  (a failed generation counts as unsolved, and is rare)
            return t["id"], 0

    passes: dict[str, list[int]] = {t["id"]: [] for t in tasks}
    with cf.ThreadPoolExecutor(workers) as ex:
        for tid, r in ex.map(one, jobs):
            passes[tid].append(r)
    rows = []
    for t in tasks:
        p = passes[t["id"]]
        rows.append({"id": t["id"], "source": t.get("source"), "licence": t.get("licence"),
                     "generator": t.get("generator"), "rating": t.get("rating"), "k": k, "solved": sum(p),
                     "pass_rate": round(sum(p) / k, 3), "decision": decide(p)})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, required=True)
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--model-name", default="unknown", help="recorded in the manifest")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    tasks = [json.loads(l) for l in args.tasks.read_text(encoding="utf-8").splitlines() if l.strip()]
    tasks = tasks[: args.limit or None]
    t0 = time.time()
    rows = run_filter(tasks, chat_sampler(args.server, args.temperature, args.max_tokens), args.k, workers=args.workers)
    counts = {d: sum(r["decision"] == d for r in rows) for d in ("keep", "too_hard", "too_easy")}
    manifest = {"model": args.model_name, "k": args.k, "temperature": args.temperature, "tasks": len(rows),
                **counts, "minutes": round((time.time() - t0) / 60, 1), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k != "rows"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
