#!/usr/bin/env python3
"""
check_ids.py
============

Gate for v2 (#163): a new tokenizer keeps every id a model may have been trained
with. Compares <new> with <old> (v1) and fails unless

  - every special token (added_tokens) has the same id, content and flags, except
    a reserved slot (<|g10_slot_NN|>, <|lang_slot_NN|>, <|ctrl_slot_NN|>,
    <|reserved_N|>) filled by renaming it, with its id and flags kept (v0.1, #132),
  - every id below --base-end maps to the same token string, except the ids that
    the build swapped (o200k pruning by usage: manifest "o200k_swap.swapped_ids",
    passed with --manifest); those must all hold a new token,
  - the kept base merges are identical and in the same order,
  - the vocab size is the same.

Ids at or above --base-end (the script blocks) may differ; the script
reports how many tokens there kept their id.

Usage
-----
    python check_ids.py /path/to/v1/tokenizer.json /path/to/v2/tokenizer.json --manifest v2_manifest.json
"""
import argparse
import json
import re
import sys

RESERVED = re.compile(r"<\|(g10_slot|lang_slot|ctrl_slot|reserved)_\d+\|>\Z")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--base-end", type=int, default=114333, help="first id after the base block (v1: 114333)")
    ap.add_argument("--manifest", help="train_bpe.py manifest of <new>; its o200k_swap.swapped_ids may differ")
    args = ap.parse_args()

    old, new = (json.load(open(p, encoding="utf-8")) for p in (args.old, args.new))
    problems = []
    if len(old["model"]["vocab"]) + len(old["added_tokens"]) != len(new["model"]["vocab"]) + len(new["added_tokens"]):
        problems.append("vocab size differs")
    filled = []
    if len(old["added_tokens"]) != len(new["added_tokens"]):
        problems.append("added_tokens (special ids) differ")
    for a, b in zip(old["added_tokens"], new["added_tokens"]):
        if a == b:
            continue
        if {**a, "content": b["content"]} == b and RESERVED.match(a["content"]):
            filled.append(a["id"])                  # a reserved slot filled by renaming it (README rule, #132)
        else:
            problems.append(f"special id {a['id']} differs: {a['content']} -> {b['content']}")

    o_id = {i: t for t, i in old["model"]["vocab"].items()}
    n_id = {i: t for t, i in new["model"]["vocab"].items()}
    swapped = set()
    if args.manifest:
        swapped = set(json.load(open(args.manifest, encoding="utf-8")).get("o200k_swap", {}).get("swapped_ids", []))
    base_ids = [i for i in o_id if i < args.base_end]
    diff = [i for i in base_ids if n_id.get(i) != o_id[i] and i not in swapped]
    if diff:
        problems.append(f"{len(diff)} base ids differ outside the swap list, first {diff[:5]}")
    same_swapped = [i for i in swapped if n_id.get(i) == o_id.get(i)]
    if same_swapped:
        problems.append(f"{len(same_swapped)} swapped ids still hold their v1 token")

    kept = {o_id[i] for i in base_ids if i not in swapped}
    old_base = [m for m in old["model"]["merges"] if m[0] + m[1] in kept]
    if old_base != new["model"]["merges"][:len(old_base)]:
        problems.append(f"the {len(old_base)} kept base merges differ")

    above = [i for i in o_id if i >= args.base_end and i not in filled]
    same_above = sum(1 for i in above if n_id.get(i) == o_id[i])
    print(f"specials: {len(old['added_tokens']) - len(filled)} identical, {len(filled)} reserved slots filled by rename"
          + (f" (ids {filled[0]}..{filled[-1]})" if filled else ""))
    print(f"base ids 0..{args.base_end - 1}: {len(base_ids) - len(swapped) - len(diff)}/{len(base_ids)} identical, "
          f"{len(swapped)} swapped (unused tokens pruned); kept base merges: {len(old_base)}")
    print(f"ids >= {args.base_end}: {same_above}/{len(above)} keep their v1 token (expected to change)")
    for p in problems:
        print("FAIL:", p)
    print("RESULT:", "FAIL" if problems else "PASS")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
