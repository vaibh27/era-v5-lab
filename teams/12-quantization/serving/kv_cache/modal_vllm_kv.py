"""Real vLLM 0.33.0 run on Modal (one NVIDIA L4 per KV-cache dtype): KV-cache capacity, speed and drift for sarvam-1.

    python kvq_study.py --dump-docs docs.json ...      # the 18 documents as token ids (training text: keep it local)
    modal run modal_vllm_kv.py::main --docs-path docs.json --out-path vllm_capacity_speed.json  # run(): capacity, speed
    modal run modal_vllm_kv.py::drift_main --docs-path docs.json --out-path vllm_drift.json     # drift(): drift
    python vllm_summary.py vllm_capacity_speed.json vllm_drift.json --out results/vllm_summary.json

run(kv_dtype), in its own container:
  - KV capacity: the KV-cache tokens vLLM allocates (num_gpu_blocks x block_size) at gpu_memory_utilization 0.85;
  - speed: 64 prompts (the first 512 tokens of the docs), 256 greedy output tokens each, end-to-end output tok/s;
  - before the timed run it scores the docs once in a single prefill (prompt log-probs). That warms the kernels up, but it
    is NOT a drift measurement for TurboQuant: vLLM's TurboQuant backend runs a first prefill on the uncompressed K/V
    and reads the compressed cache only in decode and continuation chunks. drift() measures drift.
drift(kv_dtype): prompt log-probs of the same docs with chunked prefill in 32-token pieces, so every piece after the
first attends to earlier tokens through the compressed cache with the TurboQuant decode kernel, as decode does.

The image is CUDA 13.0 devel: vLLM 0.33.0 brings torch 2.14 (CUDA 13.0), and fp8 KV on an L4 runs on FlashInfer, which
JIT-compiles its kernels with nvcc. The outputs hold per-position log-probs and top-1 token ids of training text; keep
them local and commit only vllm_summary.py's summary. No benchmark data is involved.
"""
import json
import modal

MODEL, REV = 'sarvamai/sarvam-1', 'e9607337286ddf496d4a2562b194e489dcf3feea'
app = modal.App('era-v5-g12-kvq')
image = (modal.Image.from_registry('nvidia/cuda:13.0.3-devel-ubuntu24.04', add_python='3.12')
         .pip_install('vllm==0.33.0', 'hf_transfer')
         .env({'HF_HUB_ENABLE_HF_TRANSFER': '1', 'VLLM_LOGGING_LEVEL': 'INFO', 'CUDA_HOME': '/usr/local/cuda',
               'VLLM_USE_FLASHINFER_SAMPLER': '0'}))   # PyTorch sampler: greedy and prompt log-probs do not need FlashInfer's
hf_cache = modal.Volume.from_name('era-v5-hf-cache', create_if_missing=True)


@app.function(image=image, gpu='L4', volumes={'/root/.cache/huggingface': hf_cache}, timeout=2400)
def run(kv_dtype: str, docs: list, n_bench: int = 64) -> dict:
    import time, torch, vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    out = dict(kv_dtype=kv_dtype, vllm=vllm.__version__, gpu=torch.cuda.get_device_name(0))
    try:
        t0 = time.time()
        llm = LLM(model=MODEL, revision=REV, dtype='bfloat16', kv_cache_dtype=kv_dtype, max_model_len=2048,
                  gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=False)
        out['load_s'] = round(time.time() - t0, 1)
        cc = llm.llm_engine.vllm_config.cache_config
        out.update(num_gpu_blocks=cc.num_gpu_blocks, block_size=cc.block_size,
                   kv_tokens=(cc.num_gpu_blocks or 0) * cc.block_size)
        res = []
        gens = llm.generate([TokensPrompt(prompt_token_ids=d) for d in docs],
                            SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0))
        for d, g in zip(docs, gens):
            actual, top1 = [], []
            for i, entry in enumerate(g.prompt_logprobs or []):
                if entry is None:
                    continue
                actual.append(entry[d[i]].logprob)
                top1.append(min(entry.items(), key=lambda kv: kv[1].rank)[0])
            res.append(dict(actual=actual, top1=top1))
        out['docs'] = res
        prompts = [TokensPrompt(prompt_token_ids=docs[i % len(docs)][:512]) for i in range(n_bench)]
        t0 = time.time()
        g = llm.generate(prompts, SamplingParams(max_tokens=256, temperature=0, ignore_eos=True))
        dt = time.time() - t0
        n = sum(len(x.outputs[0].token_ids) for x in g)
        out['bench'] = dict(prompts=n_bench, gen_tokens=n, seconds=round(dt, 2), tok_per_s=round(n / dt, 1))
        hf_cache.commit()
    except Exception as e:   # an unsupported preset on this GPU is a result, not a crash
        out['error'] = f'{type(e).__name__}: {str(e)[:800]}'
    return out


@app.function(image=image, gpu='L4', volumes={'/root/.cache/huggingface': hf_cache}, timeout=2400)
def drift(kv_dtype: str, docs: list, chunk: int = 32) -> dict:
    """Prompt log-probs with chunked prefill in `chunk`-token pieces. vLLM's TurboQuant backend runs a first prefill on
    the uncompressed K/V and reads the compressed cache only in decode and in continuation chunks (q_len <= 128 use its
    decode kernel), so a one-piece prefill never sees the quantization. With 32-token pieces every piece after the
    first attends to earlier tokens through the compressed cache, as decode does. Same settings for every dtype."""
    import torch, vllm
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    out = dict(kv_dtype=kv_dtype, vllm=vllm.__version__, gpu=torch.cuda.get_device_name(0), chunk=chunk)
    try:
        llm = LLM(model=MODEL, revision=REV, dtype='bfloat16', kv_cache_dtype=kv_dtype, max_model_len=2048,
                  gpu_memory_utilization=0.85, seed=0, enable_prefix_caching=False, enable_chunked_prefill=True,
                  max_num_batched_tokens=chunk, max_num_seqs=min(chunk, 8))
        sc = llm.llm_engine.vllm_config.scheduler_config
        out.update(max_num_batched_tokens=sc.max_num_batched_tokens, chunked_prefill=sc.enable_chunked_prefill)
        res = []
        gens = llm.generate([TokensPrompt(prompt_token_ids=d) for d in docs],
                            SamplingParams(max_tokens=1, prompt_logprobs=1, temperature=0))
        for d, g in zip(docs, gens):
            actual, top1 = [], []
            for i, entry in enumerate(g.prompt_logprobs or []):
                if entry is None:
                    continue
                actual.append(entry[d[i]].logprob)
                top1.append(min(entry.items(), key=lambda kv: kv[1].rank)[0])
            res.append(dict(actual=actual, top1=top1))
        out['docs'] = res
    except Exception as e:
        out['error'] = f'{type(e).__name__}: {str(e)[:800]}'
    return out


@app.local_entrypoint()
def drift_main(docs_path: str = 'docs.json', out_path: str = 'vllm_drift.json', chunk: int = 32,
               dtypes: str = 'auto,fp8,turboquant_4bit_nc,turboquant_k3v4_nc,turboquant_3bit_nc'):
    docs = json.load(open(docs_path))['ids']
    names = dtypes.split(',')
    results = list(drift.starmap([(d, docs, chunk) for d in names], return_exceptions=True))
    results = [r if isinstance(r, dict) else dict(kv_dtype=d, error=f'{type(r).__name__}: {str(r)[:800]}') for d, r in zip(names, results)]
    json.dump(results, open(out_path, 'w'))
    for r in results:
        print(r['kv_dtype'], '|', r.get('error') or f"chunk {r.get('max_num_batched_tokens')} chunked {r.get('chunked_prefill')} docs {len(r.get('docs', []))}")


@app.local_entrypoint()
def main(docs_path: str = 'docs.json', out_path: str = 'vllm_capacity_speed.json',
         dtypes: str = 'auto,fp8,turboquant_4bit_nc,turboquant_k3v4_nc,turboquant_3bit_nc'):
    docs = json.load(open(docs_path))['ids']
    names = dtypes.split(',')
    results = list(run.starmap([(d, docs) for d in names], return_exceptions=True))   # one crash must not cancel the rest
    results = [r if isinstance(r, dict) else dict(kv_dtype=d, error=f'{type(r).__name__}: {str(r)[:800]}') for d, r in zip(names, results)]
    json.dump(results, open(out_path, 'w'))
    for r in results:
        print(r['kv_dtype'], '|', r.get('error') or f"kv_tokens {r.get('kv_tokens')} | {r.get('bench', {}).get('tok_per_s')} tok/s | load {r.get('load_s')}s")
