# Smoke: the Ollama path on a stand-in in the shape of dry run 2 (26 Nov 2026)

Apple Mac (arm64), CPU for the job, Ollama 0.40.18 for the smoke. llama.cpp b11669 source (converter), its
`requirements-convert_hf_to_gguf.txt` venv (Python 3.12, torch 2.11.0, transformers 4.57.6); `llama-quantize`,
`llama-completion` and `llama-tokenize` from Homebrew (build 10050). Frozen tokenizer p1536, sha256 `e6b93a93…a64b`.

**Result: PASS (flow).** `run_ollama_job.py` ran end to end with exit 0, and the GGUF it made runs in Ollama with the
same greedy text as llama.cpp. (A job request on the dry run 2 checkpoint waits for that checkpoint; dry run 2 had
not started on 26 Nov.)

## Stand-in checkpoint

`../scripts/make_random_checkpoint.py --d-model 512 --n-layer 8 --n-head 8 --n-kv-head 2 --d-ff 768 --n-expert 16
--n-shared 1 --top-k 2 --seq-len 2048 --head-mode kas_v2 --bias-gamma 0.001 --router-bias-std 0.01`: the T2 shape of
`exp2_muon` with dry run 2's `head_mode: kas_v2` and `bias_gamma: 0.001`; random weights (176,398,881 parameters), a
random router_bias in every layer (largest 0.027) and a random head.unigram (std 1). **Its quality numbers mean
nothing**; only the flow is tested.

## Steps

| Step | Result |
|---|---|
| `export_hf.py --verify 16 --drop-router-bias --drop-head-bias` | exit 0; 299,966,976 parameters (bf16). A1 with both parts dropped: same top-1 on 435/1,024 positions, as expected for random weights with a std-1 head bias (the drops are what A1 sees) |
| `measure_export_gaps.py` (Bengali 17 docs, Punjabi 28 docs, 3,000 tokens each) | exit 0; all four variants, bpb, top-1 agreement, KL and the tokenizer cost written to `gaps.json` |
| `package_ollama.py` (Gurmukhi remap) | exit 0; `era-v5-standin-BF16.gguf` 582 MB, `-Q8_0.gguf` 310 MB, Modelfiles, README.md, release.json (192 Gurmukhi tokens renamed, ids 69440 and 100388 retired) |
| `./out` size | 36 KB (GGUF files stay in `work/release`) |
| `package_ollama.py --smoke` (Ollama 0.40.18), after #834 review | `ollama create` both files; 7 prompts (English, code, Hindi, Kannada, Punjabi precomposed and split, Bengali) x 2 formats: **same tokenization 14/14** (Ollama `prompt_eval_count` = llama.cpp ids); same greedy text 12/14 (the 2 misses: the precomposed Punjabi prompt, same token count, random weights give near-flat logits, so greedy picks differ on near-ties); Gurmukhi split-form ids = training ids |
| Ollama reads | arch `granitemoe`, 16 experts + 1 shared (feed-forward 768), top-2, 8 layers, width 512, vocabulary 131,072, pre-tokenizer `granite-embed-multi-97m`, `normalizer.nfc` true |

## Tokenizer remap, measured with llama.cpp's own tokenizer (`llama-tokenize`)

Wikipedia text (CC BY-SA, not committed), lines whose llama.cpp ids differ from the training ids:

| Text | Stock GGUF | Gurmukhi-remapped GGUF |
|---|---:|---:|
| Punjabi, 28 lines | 16 (57.1%) | **3 (10.7%)** |
| Bengali, 60 lines | 52 (86.7%) | 52 (86.7%) |

The same counts come out of the Hugging Face `tokenizers` emulation (NFC-only normalizer) on all 932 Bengali lines:
77.7% before and after, every line unchanged by the Gurmukhi remap.

## Fold variant (`measure_export_gaps.py`, `granite_fold`), same stand-in

Calibration on the first 30% of each language's 4,000 tokens; scores on the rest. Random weights: only the direction of
the change means something.

| | Same top-1 as checkpoint (bn / pa) | Mean KL vs checkpoint, nats (bn / pa) |
|---|---|---|
| `granite` (both biases dropped) | 22.3% / 26.5% | 1.345 / 1.165 |
| `granite_fold` | 42.8% / 46.6% | 0.637 / 0.537 |

Head fit: h.v mean 0.950, std 0.219 (2,559 rows). Router fit, layer 0: x.v mean 0.996, std 0.064; folded router picks
the same experts as the checkpoint on 93.0% of tokens (bias dropped: 89.2%).

## Normalization in today's runtimes (measured)

Token ids for the same text: the training tokenizer, the remapped vocabulary with and without NFC, llama.cpp 10050
(`llama-tokenize`) and Ollama 0.40.18 (`prompt_eval_count`). Both runtimes match the no-NFC column on all four prompts:

| Prompt | Training | Remap + NFC | Remap, no NFC | llama.cpp 10050 | Ollama 0.40.18 |
|---|---:|---:|---:|---:|---:|
| Punjabi, precomposed | 6 | 6 (same ids) | 11 | 11 | 11 |
| Punjabi, split (NFC form) | 6 | 6 (same ids) | 6 (same ids) | 6 (same ids) | 6 |
| Bengali, split য় | 9 | 10 | 10 | 10 | 10 |
| Bengali, precomposed য় | 9 | 10 | 9 (same ids) | 9 (same ids) | 9 |

## Not covered

- The real dry run 2 checkpoint (the point of the job): how much the dropped router bias and head bias cost.
- `ollama run hf.co/...` from Hugging Face (needs TSAI's upload).
- A Linux job machine: the smoke ran on macOS.
