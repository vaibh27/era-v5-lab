"""Build tokenizer v0.1 from tokenizer v0.

v0.1 = v0 with the first 5 G10 slots renamed to G10's ChatML v0 markers
(teams/10-post-training/special-tokens/special_tokens_v0.md). Rename only: every id,
`special: true`, the vocab size and the merges stay the same. The other 4 G10 markers
already exist in the baseline and keep their ids. See special_slots.md.

Usage:
    python build_v0_1.py <path/to/v0_tokenizer.json> <out/tokenizer.json>
"""
import hashlib
import json
import sys

V0_SHA256 = "2de039f20d17afd17615095c406c43087359673b767ed0fe2c480b995f2af8e5"

# G10 markers that are already baseline specials: content -> id (unchanged).
EXISTING = {
    "<|im_start|>": 129994,
    "<|im_end|>": 130816,
    "<|tool_call|>": 130773,
    "<|tool_result|>": 130774,
}
# New G10 markers: slot -> (id, new content).
RENAMES = {
    "<|g10_slot_00|>": (130822, "<|endoftext|>"),
    "<|g10_slot_01|>": (130823, "<|think|>"),
    "<|g10_slot_02|>": (130824, "<|/think|>"),
    "<|g10_slot_03|>": (130825, "<|/tool_call|>"),
    "<|g10_slot_04|>": (130826, "<|/tool_result|>"),
}


def main(src, dst):
    raw = open(src, "rb").read()
    sha = hashlib.sha256(raw).hexdigest()
    if sha != V0_SHA256:
        sys.exit(f"v0 sha256 mismatch: {sha}")
    t = json.loads(raw.decode("utf-8"))

    by_content = {a["content"]: a for a in t["added_tokens"]}
    for content, i in EXISTING.items():
        assert by_content[content]["id"] == i and by_content[content]["special"], content
    for _, new in RENAMES.values():
        assert new not in by_content and new not in t["model"]["vocab"], new

    m = {old: new for old, (_, new) in RENAMES.items()}
    renamed = 0
    for a in t["added_tokens"]:
        if a["content"] in m:
            assert a["id"] == RENAMES[a["content"]][0] and a["special"], a["content"]
            a["content"] = m[a["content"]]
            renamed += 1
    assert renamed == len(m), renamed

    vocab = t["model"]["vocab"]
    t["model"]["vocab"] = {m.get(k, k): v for k, v in vocab.items()}
    assert len(t["model"]["vocab"]) == len(vocab) == 131072
    assert len({a["content"] for a in t["added_tokens"]}) == len(t["added_tokens"]) == 356

    # Same serialisation as the baseline and v0 files (json.dumps defaults, ensure_ascii=False).
    out = json.dumps(t, ensure_ascii=False).encode("utf-8")
    open(dst, "wb").write(out)
    print(f"renamed {renamed} G10 slots")
    print(f"sha256 {hashlib.sha256(out).hexdigest()}  {dst}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
