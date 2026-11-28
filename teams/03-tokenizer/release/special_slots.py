#!/usr/bin/env python3
"""
special_slots.py
================

Writes `special_slots.json`, the machine-readable list of all 356 special tokens of the frozen tokenizer (id,
string, group, meaning, status, who uses it), and runs the special-token test on it:

  1. each special token encodes to exactly one id, its own, alone and between plain words;
  2. plain text never makes a special id: held-out text in 11 Indic languages, English and code;
  3. look-alikes never make one: fullwidth bars, spaces, other case, a zero-width space inside, cut-off markers;
  4. a literal marker string in raw text does become a special id (tokenizers matches added tokens). This is
     expected; the shard writer leaves such documents out and G10's render.py rejects them in message content;
  5. G10's list (teams/10-post-training/special-tokens/special_tokens_v1.md, section 3): each of its markers is
     special at its id, and the frozen string is the one G10 names (two are known not to be: subagent_call/return).

`tokenizer.json` (`added_tokens` with `special: true`) is the source of truth. This script only adds meanings and
users, so it fails if the tokenizer is not the frozen one.

    python teams/03-tokenizer/release/special_slots.py release/tokenizer.json --text heldout/dev \
        --out-json release/special_slots.json --report results/special_tokens_test
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

TEAM = Path(__file__).resolve().parents[1]
G10_SPEC = TEAM.parent / "10-post-training" / "special-tokens" / "special_tokens_v1.md"
FROZEN = "e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b"

# Users: G1/G2 data (shard writer), G4 KE2 (byte rows for every id), G5 trainer, G9 eval, G10 post-training,
# G12 export/serving. Every id, used or not, has a KE2 byte row (G4), so G4 is not repeated below.
CHAT = ["G10", "G12", "G9"]
NAMED = {
    # id: (name, meaning, status, users)
    130822: ("eos", "End of document. Data A's shard writer ends every document with it; generation stops on it.",
             "in use", ["G1", "G2", "G5", "G9", "G10", "G12"]),
    130726: ("pad", "Padding. Never in shards; trainers mask it out of the loss.", "in use", ["G5", "G9", "G12"]),
    130725: ("bos", "Begin of text, named for loaders. Not added: add_bos_token is false and shards have no bos.",
             "named, not emitted", ["G5", "G12"]),
    130727: ("unk", "Unknown, named for loaders. Byte-level BPE never emits it.", "named, not emitted", ["G12"]),
    36: ("legacy_eos", "The baseline's old end of text. Not eos: do not use.", "legacy, do not use", []),
    129994: ("im_start", "Start of a chat turn; the role (system, user, assistant, tool) follows as plain text.",
             "in use", CHAT),
    130816: ("im_end", "End of a chat turn. Generation also stops on it (generation_config.json).", "in use", CHAT),
    130823: ("think", "Start of a reasoning trace.", "in use", CHAT),
    130824: ("think_end", "End of a reasoning trace.", "in use", CHAT),
    130773: ("tool_call", "Start of a JSON tool call.", "in use", CHAT),
    130825: ("tool_call_end", "End of a JSON tool call.", "in use", CHAT),
    130774: ("tool_result", "Start of a tool's output (untrusted).", "in use", CHAT),
    130826: ("tool_result_end", "End of a tool's output.", "in use", CHAT),
    130801: ("fim_prefix", "Fill-in-the-middle: the code before the gap.", "in use", ["G10", "G9"]),
    130803: ("fim_suffix", "Fill-in-the-middle: the code after the gap.", "in use", ["G10", "G9"]),
    130802: ("fim_middle", "Fill-in-the-middle: the gap to write.", "in use", ["G10", "G9"]),
    130804: ("fim_pad", "Fill-in-the-middle padding for batched infilling.", "in use", ["G10", "G9"]),
    130817: ("file_sep", "Boundary between files of a repository.", "in use", ["G10"]),
    130818: ("repo_name", "Repository name anchor.", "in use", ["G10"]),
    130827: ("subagent_call", "G10's sub-agent call marker (special_tokens_v1.md). The frozen string is still "
             "<|g10_slot_05|>; a rename would change the frozen sha, so G10 uses the id with this string.",
             "reserved for G10, string not renamed", ["G10"]),
    130828: ("subagent_return", "G10's sub-agent return marker. The frozen string is still <|g10_slot_06|>.",
             "reserved for G10, string not renamed", ["G10"]),
}
BASE_UNUSED = {
    # role markers and other base names that the chat format does not use (roles are plain text after im_start)
    "<|system|>": "Role marker from the base tokenizer. Unused: ChatML writes the role as plain text after im_start.",
    "<|user|>": "Role marker from the base tokenizer. Unused: ChatML writes the role as plain text after im_start.",
    "<|assistant|>": "Role marker from the base tokenizer. Unused: ChatML writes the role as plain text after im_start.",
    "<|tool|>": "Role marker from the base tokenizer. Unused: ChatML writes the role as plain text after im_start.",
    "<|end_turn|>": "End of turn from the base tokenizer. Unused: turns end with im_end.",
    "<|EOT|>": "End of turn from the base tokenizer. Unused: turns end with im_end.",
    "<|think_begin|>": "Thinking marker from the base tokenizer. Unused: reasoning uses <|think|> and <|/think|>.",
    "<|think_end|>": "Thinking marker from the base tokenizer. Unused: reasoning uses <|think|> and <|/think|>.",
    "<think>": "Thinking tag from the base tokenizer. Unused by the chat format (G10 v1); data cleaning strips it.",
    "</think>": "Thinking tag from the base tokenizer. Unused by the chat format (G10 v1); data cleaning strips it.",
    "<tool_call>": "Tool tag from the base tokenizer. Unused: tool calls use <|tool_call|> and <|/tool_call|>.",
    "</tool_call>": "Tool tag from the base tokenizer. Unused: tool calls use <|tool_call|> and <|/tool_call|>.",
    "<tool_response>": "Tool tag from the base tokenizer. Unused: tool output uses <|tool_result|> and <|/tool_result|>.",
    "</tool_response>": "Tool tag from the base tokenizer. Unused: tool output uses <|tool_result|> and <|/tool_result|>.",
}
SLOT = re.compile(r"<\|(reserved|g10_slot|lang_slot|ctrl_slot)_(\d+)\|>")
SLOT_MEANING = {
    "g10_slot": ("Free slot for G10 (post-training markers).", "G10"),
    "lang_slot": ("Free slot for language or script tags.", "G3"),
    "ctrl_slot": ("Free slot for data, eval or training control.", "G3"),
    "reserved": ("Free slot, unassigned.", "TSAI"),
}
LOOK_ALIKES = ["<｜im_start｜>", "< |im_start| >", "<|IM_START|>", "<|im_​start|>", "<|im_start", "im_start|>",
               "<|endoftext", "＜|endoftext|＞", "<|think |>", "<|/ think|>", "<|tool_call|", "[INST]", "<s>", "</s>",
               "<|endoftext|​>", "<‌|im_end|>"]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def entries(raw):
    out = []
    for a in sorted((a for a in raw["added_tokens"] if a["special"]), key=lambda a: a["id"]):
        i, s = a["id"], a["content"]
        m = SLOT.fullmatch(s)
        if i in NAMED:
            name, meaning, status, users = NAMED[i]
            group = "free slot (named by G10)" if m else "core"
        elif m:
            meaning, owner = SLOT_MEANING[m.group(1)]
            name, group, status, users = f"{m.group(1)}_{m.group(2)}", "free slot", f"free, owner {owner}", []
        elif s in BASE_UNUSED:
            name, group, meaning, status, users = s.strip("<|>/").replace("|", ""), "base, unused", BASE_UNUSED[s], \
                "unused, do not emit", []
        else:
            name, group, status, users = s.strip("<|>"), "base, unused", "unused, do not emit", []
            meaning = "Named special from the base tokenizer (code, data source, vision, reasoning step). " \
                      "Not used by any team's format today; kept because ids never move."
        out.append({"id": i, "token": s, "name": name, "group": group, "meaning": meaning, "status": status,
                    "used_by": users, "bytes": len(s.encode("utf-8"))})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("tokenizer")
    ap.add_argument("--text", nargs="+", required=True, help="folders or files of plain text (held-out splits)")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--report", required=True, help="path stem: writes .json and .md")
    args = ap.parse_args()
    from tokenizers import Tokenizer

    got = sha256(args.tokenizer)
    if got != FROZEN:
        sys.exit(f"{args.tokenizer}: sha256 {got} is not the frozen p1536 {FROZEN}")
    raw = json.loads(Path(args.tokenizer).read_text(encoding="utf-8"))
    tok = Tokenizer.from_file(args.tokenizer)
    rows = entries(raw)
    special = {r["id"] for r in rows}
    checks, problems = {}, []

    # 1. one id each, alone and in context
    bad = [r["token"] for r in rows if tok.encode(r["token"], add_special_tokens=False).ids != [r["id"]]]
    ctx = [r["token"] for r in rows
           if r["id"] not in tok.encode(f"hello {r['token']} world", add_special_tokens=False).ids
           or sum(i in special for i in tok.encode(f"hello {r['token']} world", add_special_tokens=False).ids) != 1]
    checks["one_id_each"] = {"tokens": len(rows), "not_one_id": bad, "not_one_id_in_context": ctx}
    problems += [f"{t} is not one id" for t in bad + ctx]

    # 2. plain text never makes a special id
    files = []
    for p in map(Path, args.text):
        files += sorted(p.glob("*.txt")) if p.is_dir() else [p]
    per_file = {}
    for f in files:
        text = f.read_text(encoding="utf-8")
        ids = tok.encode(text, add_special_tokens=False).ids
        hits = Counter(i for i in ids if i in special)
        literal = sum(text.count(r["token"]) for r in rows)
        per_file[f"{f.parent.name}/{f.name}"] = {"sha256": sha256(f), "bytes": len(text.encode("utf-8")),
                                                  "tokens": len(ids), "special_ids": sum(hits.values()),
                                                  "literal_marker_strings": literal}
        if sum(hits.values()) != literal:
            problems.append(f"{f.name}: {sum(hits.values())} special ids but {literal} literal markers")
    checks["plain_text"] = {"files": per_file, "tokens": sum(v["tokens"] for v in per_file.values()),
                            "special_ids": sum(v["special_ids"] for v in per_file.values())}

    # 3. look-alikes
    la = {s: [i for i in tok.encode(f"a {s} b", add_special_tokens=False).ids if i in special] for s in LOOK_ALIKES}
    checks["look_alikes"] = {"strings": len(la), "made_special_ids": {s: v for s, v in la.items() if v}}
    problems += [f"look-alike {s!r} made a special id" for s, v in la.items() if v]

    # 4. a literal marker in raw text is matched (expected; the writer leaves such docs out)
    lit = tok.encode("Use <|endoftext|> to end and <|im_start|> to start.", add_special_tokens=False).ids
    checks["literal_markers"] = {"ids": [i for i in lit if i in special], "expected": [130822, 129994],
                                 "note": "tokenizers matches added tokens in raw text; shard_writer.py leaves out a "
                                         "doc with a special id inside, render.py rejects markers in content"}
    if checks["literal_markers"]["ids"] != [130822, 129994]:
        problems.append("literal markers did not map to their ids")

    # 5. G10's machine-readable table
    block = G10_SPEC.read_text(encoding="utf-8").split("## 3.")[1].split("```text")[1].split("```")[0]
    by_id = {r["id"]: r["token"] for r in rows}
    g10 = []
    for line in block.strip().splitlines():
        i, name, surface = line.split()
        g10.append({"id": int(i), "name": name, "g10_string": surface, "frozen_string": by_id.get(int(i)),
                    "match": by_id.get(int(i)) == surface})
    unexpected = [g for g in g10 if not g["match"] and g["name"] not in ("subagent_call", "subagent_return")]
    checks["g10_v1"] = {"spec_sha256": sha256(G10_SPEC), "markers": len(g10), "match": sum(g["match"] for g in g10),
                        "differ": [g for g in g10 if not g["match"]]}
    problems += [f"G10 {g['name']} {g['id']}: frozen {g['frozen_string']} vs {g['g10_string']}" for g in unexpected]

    groups = Counter(r["group"] for r in rows)
    status = Counter(r["status"] for r in rows)
    doc = {"_comment": "All special tokens of the frozen tokenizer; made by special_slots.py. tokenizer.json is the "
                       "source of truth for ids and strings; this file adds meaning, status and users. "
                       "used_by lists the teams whose formats emit or read the token; G4 (KE2) has a byte row for "
                       "every id and is not listed.",
           "tokenizer_sha256": FROZEN, "vocab_size": 131072, "special_count": len(rows),
           "roles": {"bos": 130725, "eos": 130822, "pad": 130726, "unk": 130727, "chat_turn_end": 130816},
           "groups": dict(groups), "status": dict(status), "tokens": rows}
    Path(args.out_json).write_text(json.dumps(doc, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    result = {"tokenizer_sha256": FROZEN, "special_slots_json_sha256": sha256(args.out_json),
              "result": "FAIL" if problems else "PASS", "problems": problems, "checks": checks}
    stem = Path(args.report)
    stem.with_suffix(".json").write_text(json.dumps(result, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    pt = checks["plain_text"]
    md = [f"# Special-token test, p1536 `{FROZEN[:8]}…`", "",
          f"Made by `release/special_slots.py`. Result: **{result['result']}**.", "",
          "| Check | Result |", "|---|---|",
          f"| Each of the {len(rows)} special tokens encodes to one id, its own (alone, and between words) | "
          f"{len(rows) - len(set(bad + ctx))} / {len(rows)} |",
          f"| Plain text makes no special id ({len(per_file)} held-out files, {pt['tokens']:,} tokens) | "
          f"{pt['special_ids']} special ids |",
          f"| Look-alikes make no special id ({len(la)} strings: fullwidth, spaces, case, zero-width, cut off) | "
          f"{sum(bool(v) for v in la.values())} made one |",
          "| A literal marker in raw text becomes its special id (expected: the shard writer leaves such docs out) "
          f"| {checks['literal_markers']['ids']} |",
          f"| G10's `special_tokens_v1.md` markers: special at their id, with G10's string | "
          f"{checks['g10_v1']['match']} / {checks['g10_v1']['markers']}; differ: "
          + ", ".join(f"{g['id']} {g['name']} (frozen `{g['frozen_string']}`)" for g in checks['g10_v1']['differ'])
          + " |", "",
          "| File | sha256 | Tokens | Special ids |", "|---|---|---:|---:|"]
    md += [f"| `{k}` | `{v['sha256'][:12]}…` | {v['tokens']:,} | {v['special_ids']} |" for k, v in per_file.items()]
    stem.with_suffix(".md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md[:10]))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
