#!/usr/bin/env python3
"""One command for the TSAI job: G5 checkpoint -> Ollama-ready GGUF release + what the GGUF version loses.

    python teams/12-quantization/ollama/run_ollama_job.py --checkpoint ./inputs/dryrun2-1234 \
        --tokenizer ./inputs/tokenizer.json --texts ./inputs/texts --llama-cpp ./work/llama.cpp-b11669 \
        --quantize llama-quantize --work ./work --out ./out/ollama-dryrun2 [--name era-v5-dryrun2]

Steps (stops at the first failure; offline: everything is staged beforehand):
  1. export_hf.py: the checkpoint -> Granite MoE Shared folder, test A1 on 16 prompts. Parts Granite cannot carry
     (router_bias, the KAS head bias) are dropped with --drop-router-bias / --drop-head-bias and recorded;
  2. measure_export_gaps.py: on the staged text, bits per byte of the checkpoint against the checkpoint without
     each dropped part, and the cost of llama.cpp's NFC-only tokenization (with the Gurmukhi vocabulary remap);
  3. package_ollama.py: BF16 + Q8_0 GGUF files with the remapped vocabulary, Modelfiles, README.md, release.json.
The GGUF files (hundreds of MB) go to <work>/release; ./out gets only the small files (under 1 MB).
"""
from __future__ import annotations

import argparse, json, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOKENIZER_SHA = "e6b93a93a60520d7b1af2e9242485f443b5596a1e116b8f8798d83bbad25a64b"   # frozen p1536


def run(cmd: list[str], log: Path) -> None:
    print("$", " ".join(cmd), flush=True)
    with open(log, "w") as f:
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if p.returncode:
        sys.exit(f"failed (exit {p.returncode}), see {log}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--tokenizer-sha256", default=TOKENIZER_SHA)
    ap.add_argument("--texts", required=True, type=Path)
    ap.add_argument("--llama-cpp", required=True, type=Path)
    ap.add_argument("--quantize", default="llama-quantize")
    ap.add_argument("--work", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--name", default="era-v5-dryrun2")
    ap.add_argument("--repo", default="theschoolofai/era-v5-dryrun2")
    ap.add_argument("--tokens", type=int, default=20000)
    a = ap.parse_args()
    py = sys.executable
    for d in (a.work, a.out, a.out / "logs"):
        d.mkdir(parents=True, exist_ok=True)
    hf = a.work / "hf_export"
    run([py, str(HERE.parent / "scripts/export_hf.py"), "--checkpoint", str(a.checkpoint), "--out", str(hf),
         "--tokenizer", str(a.tokenizer), "--verify", "16", "--drop-router-bias", "--drop-head-bias"],
        a.out / "logs/export.log")
    run([py, str(HERE / "measure_export_gaps.py"), "--checkpoint", str(a.checkpoint), "--tokenizer", str(a.tokenizer),
         "--texts", str(a.texts), "--out", str(a.out / "gaps.json"), "--tokens", str(a.tokens)],
        a.out / "logs/gaps.log")
    release = a.work / "release"
    run([py, str(HERE / "package_ollama.py"), "--hf-dir", str(hf), "--tokenizer", str(a.tokenizer),
         "--tokenizer-sha256", a.tokenizer_sha256, "--llama-cpp", str(a.llama_cpp), "--convert-python", py,
         "--quantize", a.quantize, "--out", str(release), "--name", a.name, "--repo", a.repo],
        a.out / "logs/package.log")
    for f in ("README.md", "release.json", *(p.name for p in release.glob("Modelfile.*"))):
        shutil.copy(release / f, a.out / f)
    shutil.copy(hf / "export_report.json", a.out / "export_report.json")
    print(json.dumps({"gguf_files": [p.name for p in release.glob("*.gguf")], "release": str(release),
                      "out": str(a.out)}, indent=1))


if __name__ == "__main__":
    main()
