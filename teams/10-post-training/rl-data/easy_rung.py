"""The easy rung: program-generated stdio tasks with execution-verified answers, for a model that solves almost
no contest problem yet.

The 3 Nov rule allows "execution-verified synthesis" but not model-written text. So the parts that become training
text, the task STATEMENT TEMPLATES and their reference solutions, must be written by a person on the team, never
by a model (Claude included). This file only has the mechanics: fill the parameters, run the human reference
solution on each generated input to get the expected output (that is the execution check), and emit stdio tasks
in the same format as build.py. Status: waiting for TSAI's answer on this design, and for the templates.

Template file (YAML, a list), written by a human:
    - id: <short id>
      author: <GitHub handle of the person who wrote it>
      statement: <text with {placeholders}; describes the input and output format>
      solution: |
        <a Python program that reads stdin and prints the answer>
      params:                      # how to draw each placeholder
        <name>: {int: [low, high]}
      input: <text with {placeholders}: the stdin for one test>
      tests: <how many tests to generate per task>

    python easy_rung.py --templates templates/easy_rung_templates.yaml --per-template 20 --out out/easy.jsonl
"""
from __future__ import annotations

import argparse
import json
import random
import string
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

REQUIRED = ("id", "author", "statement", "solution", "params", "input", "tests")


def validate(tpl: dict) -> list[str]:
    problems = [f"missing {k}" for k in REQUIRED if k not in tpl]
    if problems:
        return problems
    names = {f for _, f, _, _ in string.Formatter().parse(tpl["statement"] + tpl["input"]) if f}
    unknown = names - set(tpl["params"])
    if unknown:
        problems.append(f"placeholders without params: {sorted(unknown)}")
    for name, spec in tpl["params"].items():
        if not (isinstance(spec, dict) and list(spec) == ["int"] and len(spec["int"]) == 2):
            problems.append(f"param {name}: only {{int: [low, high]}} is supported")
    return problems


def run_solution(code: str, stdin: str, timeout: float = 5.0) -> str:
    """The human reference solution, run locally on generated inputs (trusted code written by the team)."""
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
    p = subprocess.run([sys.executable, "-I", f.name], input=stdin, capture_output=True, text=True, timeout=timeout)
    Path(f.name).unlink(missing_ok=True)
    if p.returncode != 0:
        raise RuntimeError(f"reference solution failed: {p.stderr[-300:]}")
    return p.stdout


def generate(tpl: dict, n_tasks: int, rng: random.Random) -> list[dict]:
    problems = validate(tpl)
    if problems:
        raise ValueError(f"template {tpl.get('id')!r}: {problems}")
    tasks = []
    for i in range(n_tasks):
        values = {k: rng.randint(*v["int"]) for k, v in tpl["params"].items()}
        tests = []
        for _ in range(int(tpl["tests"])):
            test_values = {**values, **{k: rng.randint(*v["int"]) for k, v in tpl["params"].items()}}
            stdin = tpl["input"].format(**test_values)
            tests.append({"input": stdin, "output": run_solution(tpl["solution"], stdin)})
        tasks.append({"id": f"easy-{tpl['id']}-{i}", "statement": tpl["statement"].format(**values), "tests": tests,
                      "time_limit_s": 2, "source": f"G10 easy rung template {tpl['id']} by {tpl['author']}",
                      "licence": "team-written", "generator": "none (human template; answers from the human "
                      "reference solution, execution-verified)"})
    return tasks


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--templates", type=Path, required=True)
    ap.add_argument("--per-template", type=int, default=20)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    templates = yaml.safe_load(args.templates.read_text(encoding="utf-8")) or []
    if not templates:
        print("no templates yet: a team member writes them (see the docstring)")
        return 1
    rng = random.Random(args.seed)
    tasks = [t for tpl in templates for t in generate(tpl, args.per_template, rng)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("".join(json.dumps(t) + "\n" for t in tasks), encoding="utf-8")
    print(json.dumps({"templates": len(templates), "tasks": len(tasks), "out": str(args.out)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
