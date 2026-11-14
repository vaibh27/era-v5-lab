#!/usr/bin/env python3
"""--normalize-first of gguf_roundtrip.py, tested without the real tokenizer (G12 week 2, task 9 preparation).

    python test_gguf_roundtrip_normalizer.py <llama.cpp b11669 checkout>
    LLAMA_CPP=<checkout> python -m pytest test_gguf_roundtrip_normalizer.py

Builds a tiny BPE tokenizer (a few hundred tokens, trained on a few Indic and English sentences) with the same
pre-tokenizer and decoder as BrahmicTokenizer-131K and Team 3's v1 normalizer (NFC, Malayalam chillu, Bengali / Punjabi
nukta re-join: teams/03-tokenizer/v1/normalizer_experiment_v2.py, variant "nfc+chillu+nukta-BnPa"), converts it to GGUF
with llama.cpp's converter and checks:

  * lines the normalizer does not change (its own output, Latin, code): same ids in the raw pass;
  * lines it does change (NFD text, old-style Malayalam chillu): different ids in the raw pass, because the GGUF has no
    normalizer, and the same ids in the normalized pass;
  * normalize-then-tokenize equals HF's own pipeline on every line;
  * the verdict is 2 (known limit, one cause) with --normalize-first and 1 without it.

The sentences are written for this test; they are not from any benchmark.
"""
from __future__ import annotations

import json, sys, tempfile, unicodedata
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_roundtrip as rt  # noqa: E402

# the split regex of BrahmicTokenizer-131K (o200k's, with \p{M} in the letter classes)
PATTERN = (r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
           r"[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|"
           r"\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+")

RAW_LINES = [
    "সোমবার সে বড় মেয়েকে নিয়ে বাড়ি গেল।", "ঢাকা শহরে আজ অনেক বৃষ্টি হয়েছে।",                       # Bengali
    "திங்கள்கிழமை அவன் கொண்டு வந்தான்.", "கொடுத்த புத்தகம் மேசையில் உள்ளது.",                          # Tamil
    "ಸೋಮವಾರ ಅವನು ಊರಿಗೆ ಹೋದನು.", "ಕೊಡು ಎಂದು ಹೇಳಿದರು.",                                              # Kannada
    "അവൻ ഇന്ന് വീട്ടിൽ വന്നു.", "അവൾ കൂടെ പോയി, അവർ വന്നില്ല.",                                       # Malayalam, atomic chillu
    "ਸਕੂਲ ਵਿੱਚ ਬੱਚੇ ਖ਼ੁਸ਼ ਹਨ।",                                                                       # Punjabi
]
LATIN_CODE = ["The quick brown fox jumps over the lazy dog.", "def add(a, b):\n    return a + b", "1234 + 5678 = 6912"]


def build_tokenizer(path: Path) -> str:
    """Train the tiny tokenizer; returns the normalizer's name for the report."""
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
    sys.path.insert(0, str(HERE.parents[1] / "03-tokenizer" / "v1"))
    from normalizer_experiment_v2 import VARIANTS
    t = Tokenizer(models.BPE())
    t.normalizer = VARIANTS["nfc+chillu+nukta-BnPa"]
    t.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(PATTERN), behavior="isolated"),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False)])
    t.decoder = decoders.ByteLevel()
    t.train_from_iterator((RAW_LINES + LATIN_CODE) * 40, trainers.BpeTrainer(
        vocab_size=700, max_token_length=24, special_tokens=["<|begin_of_text|>", "<|end_of_text|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False))
    j = json.loads(t.to_str())
    j["model"]["ignore_merges"] = True   # as BrahmicTokenizer-131K
    j["post_processor"] = {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": False, "use_regex": True}
    j["decoder"] = {"type": "ByteLevel", "add_prefix_space": True, "trim_offsets": True, "use_regex": True}
    path.write_text(json.dumps(j, ensure_ascii=False), encoding="utf-8")
    return "nfc+chillu+nukta-BnPa"


def sets_for_test() -> dict:
    from normalizer_experiment_v2 import VARIANTS, to_old_chillu
    base = [VARIANTS["nfc+chillu+nukta-BnPa"].normalize_str(x) for x in RAW_LINES]   # in the normalizer's own form
    nfd = [unicodedata.normalize("NFD", x) for x in base]
    ml = [x for x in base if any(c in x for c in "ൻൾർ")]
    mk = lambda lines, kind, src: {"lines": lines, "kind": kind, "source": src}
    return {"normalized_text": mk(base, "script", "written for this test, in the normalizer's output form"),
            "latin_code": mk(LATIN_CODE, "code", "written for this test"),
            "nfd_text": mk([d for d, b in zip(nfd, base) if d != b], "extra", "the lines above, NFD-decomposed (where that differs)"),
            "old_chillu": mk([to_old_chillu(x) for x in ml], "extra", "the Malayalam lines with chillu as consonant+virama+ZWJ")}


def run(llama_cpp: Path, work: Path) -> tuple[dict, dict]:
    from tokenizers import Tokenizer
    tokjson = work / "tokenizer.json"
    build_tokenizer(tokjson)
    model_dir = work / "tiny"
    rt.make_tiny_model(tokjson, model_dir, "<|begin_of_text|>", "<|end_of_text|>")
    gguf = work / "tiny.gguf"
    rt.convert(llama_cpp, model_dir, "granite-embed-multi-97m", gguf)
    hf = Tokenizer.from_file(str(tokjson))
    plain = Tokenizer.from_str(hf.to_str()); plain.normalizer = None
    sets = sets_for_test()
    norm_sets = {n: [hf.normalizer.normalize_str(x) for x in s["lines"]] for n, s in sets.items()}
    r = rt.check(gguf, hf, plain, json.loads(tokjson.read_text(encoding="utf-8")), sets, norm_sets, normalize_first=True)
    r["cli"], r["cli_norm"] = {}, {}
    return r, sets


def check(r: dict) -> list[str]:
    """Problems with a result of run(); [] when the mode behaves as designed."""
    bad = []
    S = r["sets"]
    for name in ("normalized_text", "latin_code"):   # the normalizer changes nothing here
        if S[name]["changed"] or S[name]["id_mismatch"]:
            bad.append(f"{name}: changed {S[name]['changed']}, raw mismatches {S[name]['id_mismatch']}; expected none")
    for name in ("nfd_text", "old_chillu"):   # the normalizer changes every line, the GGUF does not normalize
        if len(S[name]["changed"]) != S[name]["lines"]:
            bad.append(f"{name}: the normalizer changed only {len(S[name]['changed'])} of {S[name]['lines']} lines")
        if not S[name]["id_mismatch"]:
            bad.append(f"{name}: the raw pass shows no difference; the test would not catch a missing normalizer")
        if S[name]["mismatch_unchanged"]:
            bad.append(f"{name}: raw differences on lines the normalizer does not change")
    for name, s in S.items():
        q = s["normalized"]
        if q["id_mismatch"] or q["gguf_text_err"] or q["hf_pipeline_diff"]:
            bad.append(f"{name}: normalized pass {q['id_mismatch']} / decode {q['gguf_text_err']} / pipeline {q['hf_pipeline_diff']}")
        if s["gguf_text_err"]:
            bad.append(f"{name}: raw decode errors {s['gguf_text_err']}")
    if rt.verdict(r, normalize_first=True) != 2:
        bad.append(f"verdict with --normalize-first is {rt.verdict(r, True)}, expected 2")
    if rt.verdict(r, normalize_first=False) != 1:
        bad.append(f"verdict without the flag is {rt.verdict(r, False)}, expected 1")
    return bad


def table(r: dict) -> str:
    rows = ["| Set | Lines | Changed by normalizer | Raw: different ids | Normalized: different ids | HF pipeline differs |",
            "|---|---|---|---|---|---|"]
    for n, s in r["sets"].items():
        rows.append(f"| {n} | {s['lines']} | {len(s['changed'])} | {len(s['id_mismatch'])} | "
                    f"{len(s['normalized']['id_mismatch'])} | {len(s['normalized']['hf_pipeline_diff'])} |")
    return "\n".join(rows)


def test_normalize_first(tmp_path=None):   # pytest entry point
    import os
    import pytest
    if not os.environ.get("LLAMA_CPP") or not (HERE.parents[1] / "03-tokenizer" / "v1").is_dir():
        pytest.skip("set LLAMA_CPP to a llama.cpp b11669 checkout, run inside the lab repo")
    with tempfile.TemporaryDirectory() as t:
        r, _ = run(Path(os.environ["LLAMA_CPP"]), Path(t))
    assert not check(r), check(r)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    with tempfile.TemporaryDirectory() as t:
        res, _ = run(Path(sys.argv[1]), Path(t))
    print(table(res))
    problems = check(res)
    for p in problems:
        print("FAIL", p)
    print("PASS: raw pass differs only on lines the normalizer changes; normalized pass is clean; verdict 2 with the flag, 1 without"
          if not problems else f"{len(problems)} check(s) failed")
    sys.exit(1 if problems else 0)
