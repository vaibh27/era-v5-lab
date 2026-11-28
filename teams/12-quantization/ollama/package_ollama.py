#!/usr/bin/env python3
"""Hugging Face export -> GGUF files that Ollama runs straight from Hugging Face (G12 week 4, TSAI ask of 26 Nov).

    python teams/12-quantization/ollama/package_ollama.py --hf-dir <export_hf.py output> \
        --tokenizer tokenizer.json --tokenizer-sha256 <64 hex> --llama-cpp <b11669 checkout> \
        --convert-python <converter venv python> --quantize llama-quantize --out <release dir> \
        --name era-v5 --repo <hf org>/<repo> [--formats BF16 Q8_0] [--smoke]

The model needs no custom code in Ollama: export_hf.py has already folded the Kronecker (KE2/KAS) embedding and head
into a standard Granite MoE Shared model, which llama.cpp and Ollama run as arch `granitemoe`. This script:
  1. converts the export to a BF16 GGUF with the frozen tokenizer and its forced pre-tokenizer (convert_frozen());
  2. quantizes the other formats (Q8_0 default; the output layer stays Q8_0 or better);
  3. writes <name>-<FORMAT>.gguf (Ollama picks a file by the `:FORMAT` tag of `ollama run hf.co/<repo>:FORMAT`),
     a Modelfile per format (for `ollama create` from a local file), README.md (the Hugging Face model card) and
     release.json (sizes, sha256, source export, tool versions);
  4. --smoke: `ollama create` each format, generate greedily on fixed prompts through Ollama's API and through
     llama.cpp (`llama-completion`) on the same file, and write smoke.json. The two must give the same text.

Nothing is uploaded: TSAI uploads the release folder to its Hugging Face org.
"""
from __future__ import annotations

import argparse, datetime, hashlib, json, shutil, subprocess, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tokenizer"))
FORMATS = ("BF16", "Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M")
SMOKE_PROMPTS = ["The capital of France is", "def fibonacci(n):", "\u092d\u093e\u0930\u0924 \u0915\u0940 \u0930\u093e\u091c\u0927\u093e\u0928\u0940",
                 "\u0c95\u0cb0\u0ccd\u0ca8\u0cbe\u0c9f\u0c95\u0ca6 \u0cb0\u0cbe\u0c9c\u0ca7\u0cbe\u0ca8\u0cbf",
                 # Gurmukhi with SHA and ZA (remapped letters), typed precomposed and decomposed; Bengali with YA+NUKTA
                 "\u0a2a\u0a70\u0a1c\u0a3e\u0a2c \u0a26\u0a40 \u0a2d\u0a3e\u0a36\u0a3e \u0a2c\u0a3e\u0a30\u0a47 \u0a1c\u0a3c\u0a30\u0a42\u0a30\u0a40 \u0a17\u0a71\u0a32",
                 "\u0a2a\u0a70\u0a1c\u0a3e\u0a2c \u0a26\u0a40 \u0a2d\u0a3e\u0a38\u0a3c\u0a3e \u0a2c\u0a3e\u0a30\u0a47 \u0a1c\u0a3c\u0a30\u0a42\u0a30\u0a40 \u0a17\u0a71\u0a32",
                 "\u0986\u09ae\u09be\u09b0 \u09a8\u09be\u09ae \u099c\u09af\u09bc \u098f\u09ac\u0982 \u0986\u09ae\u09bf \u09ac\u09be\u0982\u09b2\u09be\u09af\u09bc \u0995\u09a5\u09be \u09ac\u09b2\u09bf"]
# llama.cpp ids on the remapped GGUF must equal the training ids for Gurmukhi as web text stores it (split letter +
# nukta, the NFC form). Ollama 0.40.18 and llama.cpp 10050 apply no NFC, so precomposed input is recorded, not required.
GURMUKHI_ID_CHECK = SMOKE_PROMPTS[5:6]
GURMUKHI_INFO = SMOKE_PROMPTS[4:5]
OLLAMA = "http://localhost:11434"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def run(cmd: list[str]) -> str:
    print("$", " ".join(cmd), flush=True)
    # errors="replace": generation can stop in the middle of a multi-byte character
    p = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if p.returncode:
        sys.exit(f"failed (exit {p.returncode}): {' '.join(cmd)}\n{p.stdout[-2000:]}\n{p.stderr[-2000:]}")
    return p.stdout


def modelfile(gguf_name: str, ctx: int) -> str:
    # Base (pretrained) model: the prompt goes in as it is; generation stops at the tokenizer's EOS (in the GGUF).
    return (f"FROM ./{gguf_name}\n"
            'TEMPLATE """{{ .Prompt }}"""\n'
            'PARAMETER stop "<|endoftext|>"\n'
            f"PARAMETER num_ctx {ctx}\n")


def ollama_generate(model: str, prompt: str, n: int) -> tuple[str, int]:
    """Greedy text and the number of prompt tokens Ollama counted (its own tokenization)."""
    body = json.dumps({"model": model, "prompt": prompt, "stream": False, "raw": True,
                       "options": {"num_predict": n, "temperature": 0, "top_k": 1, "seed": 0}}).encode()
    req = urllib.request.Request(f"{OLLAMA}/api/generate", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read())
        return d["response"].rstrip("\ufffd"), d.get("prompt_eval_count", -1)


def llamacpp_generate(binary: str, gguf: Path, prompt: str, n: int) -> str:
    out = run([binary, "-m", str(gguf), "-p", prompt, "-n", str(n), "--temp", "0", "--top-k", "1", "-no-cnv",
               "--no-warmup", "--seed", "0", "--no-display-prompt"])
    out = out[:-2] if out.endswith("\n\n") else out   # llama-completion ends its output with exactly "\n\n"
    return out.rstrip("\ufffd")   # an incomplete last character (Ollama holds those bytes back)


def llamacpp_ids(binary: str, gguf: Path, text: str) -> list[int]:
    out = run([binary, "-m", str(gguf), "-p", text, "--ids", "--log-disable", "--no-bos"])
    return json.loads(out.strip().splitlines()[-1])


def card(a, release: dict) -> str:
    q4note = ("; the Q4_K_M file here is for small machines only." if any(f["format"] == "Q4_K_M" for f in release["files"])
              else ", so Q4_K_M is not provided.")
    rows = "\n".join(f"| `{f['file']}` | {f['format']} | {f['bytes'] / 1e9:.2f} GB | `{f['sha256'][:16]}…` |"
                     for f in release["files"])
    default = "Q8_0" if any(f["format"] == "Q8_0" for f in release["files"]) else release["files"][0]["format"]
    return f"""---
license: other
tags: [gguf, ollama, llama.cpp, granitemoe]
---

# {a.name}

ERA V5 lab model, as GGUF files that run in [Ollama](https://ollama.com) and llama.cpp with no custom code.

## Run

```
ollama run hf.co/{a.repo}:{default}
```

This is a **base (pretrained) model**: it continues the text you give it; it is not a chat model.

| File | Format | Size | sha256 |
|---|---|---:|---|
{rows}

**{default}** is the default: in G12's wide benchmark pass (three public models) Q8_0 showed no measurable loss
against BF16, while Q4_K_M lost 1.5 to 2.9 points on MMLU and up to 7.6 points on Indic reading (sarvam-1){q4note}

`ollama run hf.co/...` reads everything it needs from the GGUF (architecture, tokenizer, end-of-text token). The
`Modelfile.<FORMAT>` files are only for a local `ollama create` from a downloaded file.

## What is different from the training checkpoint

- The Kronecker (KE2/KAS) embedding and head are **folded exactly** into ordinary token-embedding and output matrices
  (Granite MoE Shared layout, llama.cpp arch `granitemoe`).
- Parts this layout cannot carry, and what dropping them costs, are listed in `release.json` (`export_report`).
- The tokenizer's normalizer cannot be stored in GGUF beyond NFC; plain-text prompts in some scripts can therefore be
  tokenized differently from training (see `release.json`, `tokenizer`).

Built by `teams/12-quantization/ollama/package_ollama.py` (theschoolofai/era-v5-lab) on {release['built']}.
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf-dir", required=True, type=Path, help="export_hf.py output (config.json, model.safetensors)")
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--tokenizer-sha256", required=True)
    ap.add_argument("--llama-cpp", required=True, type=Path, help="llama.cpp b11669 checkout (the converter)")
    ap.add_argument("--convert-python", required=True, help="python of the converter venv")
    ap.add_argument("--quantize", default="llama-quantize")
    ap.add_argument("--formats", nargs="+", default=["BF16", "Q8_0"], choices=FORMATS)
    ap.add_argument("--out", required=True, type=Path, help="release folder (GGUFs, Modelfiles, README.md)")
    ap.add_argument("--name", default="era-v5")
    ap.add_argument("--repo", default="theschoolofai/era-v5", help="Hugging Face repo the README points to")
    ap.add_argument("--ctx", type=int, help="num_ctx in the Modelfiles (default: the model's context length)")
    ap.add_argument("--remap", nargs="*", default=["gurmukhi"], metavar="SCRIPT",
                    help="remap_vocab.py scripts applied to the GGUF vocabulary (default gurmukhi; none: --remap)")
    ap.add_argument("--smoke", action="store_true", help="ollama create + generate, compared with llama.cpp")
    ap.add_argument("--smoke-tokens", type=int, default=8)
    ap.add_argument("--llama-completion", default="llama-completion")
    ap.add_argument("--llama-tokenize", default="llama-tokenize")
    a = ap.parse_args()

    from convert_frozen import convert_frozen
    sys.path.insert(0, str(HERE))
    from remap_vocab import remap
    a.out.mkdir(parents=True, exist_ok=True)
    tok_file, tok_sha, remap_report = a.tokenizer, a.tokenizer_sha256, None
    if a.remap:
        if sha256(a.tokenizer) != a.tokenizer_sha256.lower():
            sys.exit(f"refusing: {a.tokenizer} is not the tokenizer asked for")
        tok, remap_report = remap(json.loads(a.tokenizer.read_text(encoding="utf-8")), tuple(a.remap))
        tok_file = a.out / "_tokenizer_gguf.json"
        tok_file.write_text(json.dumps(tok, ensure_ascii=False), encoding="utf-8")
        tok_sha = sha256(tok_file)
    cfg = json.loads((a.hf_dir / "config.json").read_text())
    ctx = a.ctx or int(cfg.get("max_position_embeddings", 2048))
    bf16 = a.out / f"{a.name}-BF16.gguf"
    # a rerun must not ship GGUFs made from another export or tokenizer: compare a stamp of the sources
    stamp = {"hf_model_sha256": sha256(a.hf_dir / "model.safetensors"), "gguf_tokenizer_sha256": tok_sha}
    stamp_file = a.out / "_sources.json"
    if stamp_file.exists() and json.loads(stamp_file.read_text()) != stamp:
        for old in a.out.glob(f"{a.name}-*.gguf"):
            old.unlink()
        print("sources changed since the last run: old GGUF files removed", flush=True)
    stamp_file.write_text(json.dumps(stamp))
    if not bf16.exists():
        part = bf16.with_name(bf16.name + ".part")
        convert_frozen(a.hf_dir, tok_file, tok_sha, a.llama_cpp, part, a.out / "_work", python=a.convert_python)
        part.rename(bf16)
        shutil.rmtree(a.out / "_work", ignore_errors=True)
    if tok_file != a.tokenizer:
        tok_file.unlink(missing_ok=True)
    files = []
    for fmt in a.formats:
        gguf = bf16 if fmt == "BF16" else a.out / f"{a.name}-{fmt}.gguf"
        if not gguf.exists():
            part = gguf.with_name(gguf.name + ".part")
            extra = [] if fmt == "Q8_0" else ["--output-tensor-type", "q8_0"]
            run([a.quantize, *extra, str(bf16), str(part), fmt])
            part.rename(gguf)
        (a.out / f"Modelfile.{fmt}").write_text(modelfile(gguf.name, ctx))
        files.append({"format": fmt, "file": gguf.name, "bytes": gguf.stat().st_size, "sha256": sha256(gguf)})
    if "BF16" not in a.formats:
        bf16.unlink()
    report = a.hf_dir / "export_report.json"
    release = {"name": a.name, "repo": a.repo, "built": datetime.date.today().isoformat(), "files": files,
               "hf_export": str(a.hf_dir), "context_length": ctx,
               "export_report": json.loads(report.read_text()) if report.exists() else None,
               "tokenizer": {"sha256": a.tokenizer_sha256, "pre_tokenizer": "granite-embed-multi-97m",
                             "normalizer_in_gguf": "NFC only (tokenizer.ggml.normalizer.nfc)",
                             "gguf_vocab_remap": ({"scripts": a.remap, "gguf_tokenizer_sha256": tok_sha,
                                                   **{k: v for k, v in remap_report.items() if k != "retired"},
                                                   "retired_ids": remap_report["retired"]} if a.remap else None)}}
    (a.out / "release.json").write_text(json.dumps(release, indent=1))
    (a.out / "README.md").write_text(card(a, release))

    if a.smoke:
        results = []
        for f in files:
            model = f"{a.name.lower()}-smoke:{f['format'].lower()}"
            run(["ollama", "create", model, "-f", str(a.out / f"Modelfile.{f['format']}")])
            for p in SMOKE_PROMPTS:
                o, n_prompt = ollama_generate(model, p, a.smoke_tokens)
                l = llamacpp_generate(a.llama_completion, a.out / f["file"], p, a.smoke_tokens)
                n_lc = len(llamacpp_ids(a.llama_tokenize, a.out / f["file"], p))
                # the token count is independent of the weights; the text can diverge on random (flat) weights
                results.append({"format": f["format"], "prompt": p, "ollama": o, "llama_cpp": l,
                                "same": bool(o) and o == l, "prompt_tokens_ollama": n_prompt,
                                "prompt_tokens_llama_cpp": n_lc, "same_tokenization": n_prompt == n_lc})
            run(["ollama", "rm", model])
        # the Gurmukhi remap, in llama.cpp's own tokenizer: plain text must give the training ids
        from tokenizers import Tokenizer
        train = Tokenizer.from_file(str(a.tokenizer))
        id_checks = []
        if "gurmukhi" in (a.remap or []):
            for p in GURMUKHI_ID_CHECK:
                got, want = llamacpp_ids(a.llama_tokenize, bf16 if bf16.exists() else a.out / files[0]["file"], p), \
                    train.encode(p, add_special_tokens=False).ids
                id_checks.append({"text_sha256": hashlib.sha256(p.encode()).hexdigest()[:16], "same_ids": got == want,
                                  "tokens": len(want)})
        info = [{"text_sha256": hashlib.sha256(p.encode()).hexdigest()[:16], "form": "precomposed",
                 "same_ids": llamacpp_ids(a.llama_tokenize, a.out / files[0]["file"], p)
                 == train.encode(p, add_special_tokens=False).ids} for p in GURMUKHI_INFO if id_checks]
        ok = (all(r["same"] for r in results) and all(r["same_tokenization"] for r in results)
              and all(c["same_ids"] for c in id_checks))
        ver = subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip()
        (a.out / "smoke.json").write_text(json.dumps({"ollama": ver, "all_same": ok, "results": results,
                                                      "gurmukhi_id_check": id_checks, "gurmukhi_precomposed_info": info},
                                                     indent=1, ensure_ascii=False))
        print(f"smoke ({ver}): same tokenization {sum(r['same_tokenization'] for r in results)}/{len(results)}, "
              f"same text {sum(r['same'] for r in results)}/{len(results)}, Gurmukhi (split form) ids = training ids "
              f"{sum(c['same_ids'] for c in id_checks)}/{len(id_checks)}")
        if not ok:
            sys.exit(1)
    print("release folder:", a.out)


if __name__ == "__main__":
    main()
