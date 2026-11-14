"""Pass-rate survey on a GPU: k samples per task from a vLLM engine, scored by the exact checkers.

    python survey.py --tasks out/math_tasks.jsonl out/cc_tasks.jsonl --model /vol/hf/allenai/OLMoE-1B-7B-0125-DPO \\
                     --k 8 --instruction-math instr_math.txt --instruction-stdio instr_stdio.txt --out out/survey.json

The same decision as filter.py (keep a task when 0 < pass rate < 1, see DAPO's dynamic sampling), but with vLLM's
offline batching instead of one HTTP request per sample, so thousands of tasks fit in one GPU run. Task kinds:
    math    {"problem", "answer"}            scored by math_check.py (math-verify)
    stdio   {"statement", "tests"}            scored by stdio_check.py (runs the program on each test)

Instruction files: one per kind, written by a team member (TSAI 3 Nov data rule: no AI-written prompt text in RL
prompts). The file holds the whole user message with {problem} where the task text goes. Without a file for a
kind the survey refuses, unless --no-instruction is given (then the raw task text is the prompt). The manifest
records each file's sha256, not its text.

The manifest holds ids and numbers only (no task text, no completions), so it can be committed as a result.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import time
from pathlib import Path
from typing import Callable

import math_check
import stdio_check
from filter import decide

KINDS = ("math", "stdio")


def task_kind(task: dict) -> str:
    return task.get("kind") or ("math" if "answer" in task else "stdio")


def render(task: dict, instructions: dict[str, str | None]) -> str:
    kind = task_kind(task)
    body = task["problem"] if kind == "math" else task["statement"]
    tmpl = instructions.get(kind)
    if tmpl is None:
        return body
    if "{problem}" not in tmpl:
        raise ValueError(f"the {kind} instruction file has no {{problem}} placeholder")
    return tmpl.replace("{problem}", body)


def score_one(job: tuple[str, dict, str]) -> tuple[int, str, bool]:
    """(reward, the checker's reason, whether the completion contains a \\boxed answer)."""
    kind, task, completion = job
    boxed = "\\boxed" in completion
    try:
        fn = math_check.score if kind == "math" else stdio_check.score
        r = fn(task, completion)
        return int(r["reward"]), str(r.get("reason", "")), boxed
    except Exception:  # noqa: BLE001  (a checker crash counts as unsolved; rare)
        return 0, "checker error", boxed


def run_survey(tasks: list[dict], texts: list[str], generate: Callable, k: int, workers: int = 8,
               pool: str = "process") -> tuple[list[dict], dict]:
    """generate(texts, k) -> per task a list of k (completion_text, n_tokens, truncated) tuples."""
    t0 = time.time()
    samples = generate(texts, k)
    gen_s = time.time() - t0
    jobs = [(task_kind(t), t, c[0]) for t, cs in zip(tasks, samples) for c in cs]
    t0 = time.time()
    ex_cls = cf.ProcessPoolExecutor if pool == "process" else cf.ThreadPoolExecutor
    with ex_cls(workers) as ex:
        scored = list(ex.map(score_one, jobs, chunksize=16))
    rewards = [x[0] for x in scored]
    score_s = time.time() - t0
    rows, i = [], 0
    for t, cs in zip(tasks, samples):
        rs = rewards[i:i + len(cs)]
        reasons: dict[str, int] = {}
        for x in scored[i:i + len(cs)]:
            reasons[x[1]] = reasons.get(x[1], 0) + 1
        boxed = sum(int(x[2]) for x in scored[i:i + len(cs)])
        i += len(cs)
        lens = [c[1] for c in cs]
        good = [c[1] for c, r in zip(cs, rs) if r]
        rows.append({"id": t["id"], "kind": task_kind(t), "group": t.get("subject") or t.get("judge"),
                     "level": t.get("level") or t.get("difficulty"), "k": len(cs), "solved": sum(rs),
                     "pass_rate": round(sum(rs) / len(cs), 3), "decision": decide(rs),
                     "mean_tokens": round(sum(lens) / len(lens), 1),
                     "mean_tokens_correct": round(sum(good) / len(good), 1) if good else None,
                     "truncated": round(sum(bool(c[2]) for c in cs) / len(cs), 3),
                     "reasons": reasons, "boxed": boxed})
    tokens = sum(c[1] for cs in samples for c in cs)
    return rows, {"generate_s": round(gen_s, 1), "score_s": round(score_s, 1), "sampled_tokens": tokens,
                  "tokens_per_s": round(tokens / max(gen_s, 1e-9), 1)}


def summarise(rows: list[dict]) -> dict:
    out: dict = {}
    for r in rows:
        key = f"{r['kind']}/{r['group']}/{r['level']}"
        d = out.setdefault(key, {"tasks": 0, "keep": 0, "too_hard": 0, "too_easy": 0, "solved": 0, "samples": 0})
        d["tasks"] += 1
        d[r["decision"]] += 1
        d["solved"] += r["solved"]
        d["samples"] += r["k"]
    for d in out.values():
        d["pass_rate"] = round(d["solved"] / max(d["samples"], 1), 4)
    return dict(sorted(out.items()))


def vllm_generate(model: str, temperature: float, top_p: float, max_tokens: int, seed: int,
                  gpu_memory_utilization: float, max_model_len: int) -> tuple[Callable, Callable]:
    """Returns (generate, to_ids): a vLLM-backed generate(texts, k) and the chat-template renderer it uses."""
    from transformers import AutoTokenizer  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    tok = AutoTokenizer.from_pretrained(model)
    llm = LLM(model=model, dtype="bfloat16", seed=seed, gpu_memory_utilization=gpu_memory_utilization,
              max_model_len=max_model_len)

    def to_ids(text: str) -> list[int]:
        ids = tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=True)
        return [int(x) for x in (ids["input_ids"] if hasattr(ids, "keys") else ids)]  # BatchEncoding is not a dict

    def generate(texts: list[str], k: int):
        sp = SamplingParams(n=k, temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
        outs = llm.generate([{"prompt_token_ids": to_ids(t)} for t in texts], sp)
        return [[(c.text, len(c.token_ids), c.finish_reason == "length") for c in o.outputs] for o in outs]

    return generate, to_ids


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tasks", type=Path, nargs="+", required=True)
    ap.add_argument("--ids", type=Path, help="optional file with one task id per line: survey only these")
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-revision", default="unknown", help="recorded in the manifest")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--instruction-math", type=Path)
    ap.add_argument("--instruction-stdio", type=Path)
    ap.add_argument("--no-instruction", action="store_true", help="use the raw task text when a file is missing")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    tasks = [json.loads(line) for f in args.tasks for line in f.read_text(encoding="utf-8").splitlines() if line.strip()]
    if args.ids:
        keep = {line.strip() for line in args.ids.read_text(encoding="utf-8").splitlines() if line.strip()}
        tasks = [t for t in tasks if t["id"] in keep]
    tasks = tasks[: args.limit or None]
    files = {"math": args.instruction_math, "stdio": args.instruction_stdio}
    instructions, shas = {}, {}
    for kind in {task_kind(t) for t in tasks}:
        f = files.get(kind)
        if f is None and not args.no_instruction:
            ap.error(f"no instruction file for {kind} tasks (write one, or pass --no-instruction)")
        instructions[kind] = f.read_text(encoding="utf-8") if f else None
        shas[kind] = hashlib.sha256(instructions[kind].encode()).hexdigest() if f else None
    texts = [render(t, instructions) for t in tasks]
    generate, _ = vllm_generate(args.model, args.temperature, args.top_p, args.max_tokens, args.seed,
                                args.gpu_memory_utilization, args.max_model_len)
    rows, timing = run_survey(tasks, texts, generate, args.k, args.workers)
    counts = {d: sum(r["decision"] == d for r in rows) for d in ("keep", "too_hard", "too_easy")}
    manifest = {"model": args.model, "model_revision": args.model_revision, "k": args.k,
                "temperature": args.temperature, "top_p": args.top_p, "max_tokens": args.max_tokens,
                "seed": args.seed, "instruction_sha256": shas, "tasks": len(rows), **counts, **timing,
                "by_group": summarise(rows), "rows": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in manifest.items() if k not in ("rows", "by_group")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
