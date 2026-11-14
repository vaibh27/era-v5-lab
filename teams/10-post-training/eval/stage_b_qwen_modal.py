"""Stage B on Qwen3.6-35B-A3B-FP8, one H100 (Modal): GENERATION ONLY. Scoring is local, see stage_a_report.py.

    modal run eval/stage_b_qwen_modal.py --source-dir code_contests=DIR --source-dir hendrycks_math=DIR \
        --checker-dir RL_DATA_DIR --out results/effort_stageB_qwen_2026-11-12/raw

What runs
    The suite's own `run_stage(manifest, "B", ...)`: the 55 Stage B slots of `effort_manifest.json`, every label
    (1k, 4k, 16k, 64k), common cap 1024, the context preflight, `analyse` and `summarise`, unchanged. The only new
    piece is `VLLMBackend`, a `Backend` that renders the same messages as `HFBackend` (effort line as a system
    message, the raw task text as the user message, the model's own chat template) and samples with vLLM.
    Two samples per cell: `run_stage` is called once per sample with base_seed 1234 and 1235, so every cell keeps
    the suite's rule of one seed per prompt across labels (seed = prompt_seed(base_seed, slot)).
    Checkers are switched off on the GPU box (no Docker there); success is filled in locally afterwards.

Model
    Qwen/Qwen3.6-35B-A3B-FP8 @ c653bba4 (official FP8), dtype auto, context 8192, VLLM_USE_DEEP_GEMM=0 (same pin and
    settings as work/attainability-2026-11-11/gen_modal.py). Native template with enable_thinking=False: an
    above-floor reference, not the capstone model.

Prompts are resolved and hash-checked locally (`resolve_prompts`), and only slot, category, task id, checker and
text go to the GPU. Every output record keeps `do_not_train: true`.
"""
from __future__ import annotations

import json
import time
import traceback
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TEAM = HERE.parent
MODEL = {"repo": "Qwen/Qwen3.6-35B-A3B-FP8", "rev": "c653bba46dfa06e9658dcc0b0dc87fe4685f14a2",
         "dtype": "auto", "max_model_len": 8192}
SAMPLE_BASE_SEEDS = (1234, 1235)  # sample 0, sample 1
TEMPERATURE = 0.7
GPU = "H100!"
GPU_TIMEOUT_S = 6600  # hard stop: about $7.7 of H100 + CPU + memory at the rates below (run cap $9)
# $/s, same table as rl-loop/modal/app.py (modal.com/pricing, read 5 Nov 2026); Modal billing is the truth
GPU_RATE, CPU_RATE, MEM_RATE, CPU_CORES, MEM_GIB = 0.001097, 0.0000131, 0.00000222, 4.0, 32.0

VOL = modal.Volume.from_name("g10-attain", create_if_missing=True, version=2)
image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install("vllm==0.32.0", "huggingface_hub[hf_xet]")
    .env({"HF_XET_HIGH_PERFORMANCE": "1", "HF_HOME": "/vol/hf_cache", "TOKENIZERS_PARALLELISM": "false",
          "VLLM_USE_FLASHINFER_SAMPLER": "0", "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
          "VLLM_USE_DEEP_GEMM": "0"})
)
# render.py reads the Team 3 marker ids from the repo layout, so the files keep their repo-relative paths.
for _rel in ("teams/10-post-training/chat-format/render.py", "teams/10-post-training/eval/effort_suite.py",
             "teams/10-post-training/special-tokens/special_tokens_v1.md", "teams/03-tokenizer/v0/build_v0_1.py",
             "teams/03-tokenizer/v0/special_slots.md"):
    image = image.add_local_file(str(TEAM.parent.parent / _rel), f"/repo/{_rel}")
REMOTE_EVAL = "/repo/teams/10-post-training/eval"
app = modal.App("g10-effort-stagea", image=image)
function, local_entrypoint = app.function, app.local_entrypoint  # plain names keep pr-guard's email regex quiet
WEIGHTS = f"/vol/hf/{MODEL['repo']}@{MODEL['rev']}"


@function(volumes={"/vol": VOL}, timeout=3600, cpu=2.0, memory=4096)
def weights() -> dict:
    """Pull the pinned snapshot into the Volume once (a cached copy is reused)."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    t0 = time.time()
    VOL.reload()
    done = Path(WEIGHTS) / ".complete"
    if done.exists():
        return {"path": WEIGHTS, "cached": True}
    snapshot_download(MODEL["repo"], revision=MODEL["rev"], local_dir=WEIGHTS)
    done.write_text(MODEL["rev"])
    VOL.commit()
    return {"path": WEIGHTS, "cached": False, "seconds": round(time.time() - t0, 1)}


@function(volumes={"/vol": VOL}, gpu=GPU, timeout=GPU_TIMEOUT_S, cpu=CPU_CORES, memory=int(MEM_GIB * 1024))
def run_stage_a(manifest: dict, prompts: list[dict]) -> dict:
    import os  # noqa: PLC0415
    import sys  # noqa: PLC0415
    import tempfile  # noqa: PLC0415

    out: dict = {"stage": "import", "ok": False, "timing": {}, "samples": []}
    t0 = time.time()
    try:
        sys.path.insert(0, REMOTE_EVAL)
        import effort_suite as es  # noqa: PLC0415
        import vllm  # noqa: PLC0415
        from vllm import LLM, SamplingParams  # noqa: PLC0415

        class VLLMBackend(es.Backend):
            """Native chat template, effort line as a system message (same messages as HFBackend)."""

            name, template = "vllm", "native"

            def __init__(self):
                self.llm = LLM(model=WEIGHTS, dtype=MODEL["dtype"], seed=SAMPLE_BASE_SEEDS[0],
                               max_model_len=MODEL["max_model_len"], gpu_memory_utilization=0.90)
                self.tok = self.llm.get_tokenizer()
                self.context = MODEL["max_model_len"]
                gc = json.loads((Path(WEIGHTS) / "generation_config.json").read_text())
                eos = gc.get("eos_token_id", self.tok.eos_token_id)
                self.stop_ids = sorted({eos} if isinstance(eos, int) else set(eos))
                self.model_id = f"{MODEL['repo']}@{MODEL['rev']}"
                self.rendered = {}

            def render(self, text: str, label: str) -> str:
                msgs = [{"role": "system", "content": f"{es.render.EFFORT_PREFIX}{label}"},
                        {"role": "user", "content": text}]
                return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                    enable_thinking=False)

            def prompt_ids(self, text: str, label: str) -> list[int]:
                s = self.render(text, label)
                self.rendered.setdefault(label, s)  # first rendered prompt per label, for the record
                return list(self.tok.encode(s, add_special_tokens=False))

            def generate(self, ids: list[int], cap: int, seed: int, temperature: float) -> es.Generation:
                sp = SamplingParams(n=1, temperature=temperature, top_p=1.0, max_tokens=cap, seed=seed,
                                    stop_token_ids=self.stop_ids, skip_special_tokens=False)
                o = self.llm.generate([{"prompt_token_ids": ids}], sp, use_tqdm=False)[0].outputs[0]
                new = list(o.token_ids)
                if o.finish_reason == "stop" and (not new or new[-1] not in self.stop_ids) \
                        and isinstance(o.stop_reason, int):
                    new.append(o.stop_reason)
                return es._cut(new, self.stop_ids, lambda c: self.tok.decode(c, skip_special_tokens=False))

            def count(self, text: str) -> int:
                return len(self.tok.encode(text, add_special_tokens=False))

        out["vllm_version"] = vllm.__version__
        out["stage"] = "load"
        VOL.reload()
        if not (Path(WEIGHTS) / ".complete").exists():
            raise FileNotFoundError(f"{WEIGHTS} not in the Volume; run weights() first")
        backend = VLLMBackend()
        out["timing"]["load_s"] = round(time.time() - t0, 1)
        out["stop_ids"] = backend.stop_ids
        es.validate(manifest)
        ps = {p["slot"]: es.Prompt(p["slot"], p["category"], p["task_id"], p["text"], p["checker"],
                                   {"id": p["task_id"]}) for p in prompts}
        why = {k: "scored locally after generation (stage_a_report.py; no Docker on the GPU host)"
               for k in ("math", "stdio")}
        why["none"] = "no automatic checker for this category"
        instructions = {"math": None, "stdio": None, "none": None}
        for i, base in enumerate(SAMPLE_BASE_SEEDS):
            out["stage"] = f"generate sample {i}"
            t1 = time.time()
            with tempfile.TemporaryDirectory() as d:
                summary = es.run_stage(manifest, "B", backend, ps, Path(d), {}, why, instructions,
                                       temperature=TEMPERATURE, base_seed=base)
                records = (Path(d) / "effort_stageB.jsonl").read_text(encoding="utf-8")
            out["samples"].append({"sample": i, "base_seed": base, "summary": summary, "records_jsonl": records,
                                   "generate_s": round(time.time() - t1, 1)})
        out["rendered_prompt_per_label_first_slot"] = backend.rendered
        out["stage"], out["ok"] = "done", True
    except BaseException as e:  # noqa: BLE001  report exactly what broke; no retry, no variant
        out["error"] = f"{type(e).__name__}: {e}"
        out["traceback"] = traceback.format_exc()[-6000:]
    out["timing"]["total_s"] = round(time.time() - t0, 1)
    out["gpu_name"] = os.popen("nvidia-smi --query-gpu=name --format=csv,noheader").read().strip()
    return out


@local_entrypoint()
def main(out: str, checker_dir: str, source_dir: str = "", source_dir2: str = ""):
    """--source-dir and --source-dir2 take name=DIR (Modal CLI flags cannot repeat)."""
    import sys  # noqa: PLC0415

    sys.path.insert(0, str(HERE))
    import effort_suite as es  # noqa: PLC0415

    manifest = es.load_manifest(HERE / "effort_manifest.json")
    _, _, last_boxed = es.load_checkers(Path(checker_dir), execute_code=False)
    dirs = es._dirs([x for x in (source_dir, source_dir2) if x])
    resolved = es.resolve_prompts(manifest, dirs, last_boxed=last_boxed, slots=es.stage_slots(manifest, "B"))
    prompts = [{"slot": p.slot, "category": p.category, "task_id": p.task_id, "text": p.text, "checker": p.checker}
               for p in resolved.values()]
    print(f"stage=B prompts={len(prompts)} labels={list(es.LABELS)} samples={len(SAMPLE_BASE_SEEDS)} "
          f"base_seeds={list(SAMPLE_BASE_SEEDS)} T={TEMPERATURE} cap={es.STAGE_CAP} model={MODEL['repo']}"
          f"@{MODEL['rev'][:8]} gpu={GPU} manifest={manifest['manifest_sha256'][:12]}")
    print("weights:", json.dumps(weights.remote()))
    t0 = time.time()
    res = run_stage_a.remote(manifest, prompts)
    wall = time.time() - t0
    dest = Path(out)
    dest.mkdir(parents=True, exist_ok=True)
    for s in res.get("samples", []):
        d = dest / f"sample{s['sample']}"
        d.mkdir(exist_ok=True)
        (d / "effort_stageB.jsonl").write_text(s.pop("records_jsonl"), encoding="utf-8")
        (d / "summary_unscored.json").write_text(json.dumps(s["summary"], indent=1, ensure_ascii=False) + "\n",
                                                encoding="utf-8")
    meta = {k: v for k, v in res.items() if k != "samples"}
    meta.update(samples=[{k: v for k, v in s.items() if k != "summary"} for s in res.get("samples", [])],
                model=MODEL, temperature=TEMPERATURE, top_p=1.0, cap=es.STAGE_CAP, gpu=GPU,
                sample_base_seeds=list(SAMPLE_BASE_SEEDS), manifest_sha256=manifest["manifest_sha256"],
                enable_thinking=False, local_wall_s=round(wall, 1),
                est_usd=round(wall * (GPU_RATE + CPU_CORES * CPU_RATE + MEM_GIB * MEM_RATE), 3))
    (dest / "run_meta.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"ok={res['ok']} stage={res['stage']} gpu={res.get('gpu_name')} vllm={res.get('vllm_version')} "
          f"timing={res['timing']} est_usd={meta['est_usd']} -> {dest}")
    if not res["ok"]:
        print("ERROR:", res.get("error"))
        print(res.get("traceback", ""))
        raise SystemExit(1)
