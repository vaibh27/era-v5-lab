"""Exact-answer checker for math tasks (reward 1 if the model's final answer equals the reference, else 0).

A task is {"id", "problem", "answer"}; the answer is the content of the last \\boxed{...} in the human-written
solution (see build_math.py). The comparison uses Hugging Face's math-verify (Apache-2.0, pinned 0.9.0), which
parses LaTeX and plain expressions and checks equivalence symbolically, so "\\frac{1}{2}" and "0.5" match.

Which part of the completion counts as the final answer is decided by math-verify: a \\boxed{} answer first,
then the last LaTeX or plain expression. The instruction line in the prompt (written by a team member, not by
an AI, per the TSAI 3 Nov data rule) tells the model how to mark its answer.
"""
from __future__ import annotations

import threading

BOXED = ("\\boxed", "\\fbox")


def last_boxed(text: str) -> str | None:
    """Content of the last \\boxed{...} (or \\fbox{...}, or "\\boxed 5"), with nested braces handled."""
    start = max(text.rfind(b) for b in BOXED)
    if start < 0:
        return None
    i = start + (len("\\boxed") if text.startswith("\\boxed", start) else len("\\fbox"))
    if i < len(text) and text[i] == " ":
        end = i + 1
        while end < len(text) and not text[end].isspace() and text[end] not in "$":
            end += 1
        return text[i + 1:end] or None
    if i >= len(text) or text[i] != "{":
        return None
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i + 1:j]
    return None  # unbalanced braces


def score(task: dict, completion: str) -> dict:
    from math_verify import parse, verify  # noqa: PLC0415  (pinned dependency of the survey and the loop)

    # math-verify's timeouts use signal.alarm, which works only on the main thread. The survey scores in worker
    # processes and the trainer scores on its main thread, so both keep the 5 s limit; a thread pool gets none.
    limit = 5 if threading.current_thread() is threading.main_thread() else None
    out = {"task_id": task["id"], "reward": 0}
    gold = parse("$" + task["answer"] + "$", parsing_timeout=limit)
    if not gold:
        return {**out, "reason": "reference answer could not be parsed"}
    pred = parse(completion, parsing_timeout=limit)
    if not pred:
        return {**out, "reason": "no answer found"}
    ok = bool(verify(gold, pred, timeout_seconds=limit))
    return {**out, "reward": int(ok), "reason": "correct" if ok else "wrong answer"}
