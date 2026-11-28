# G12 Ollama path: the lab model in Ollama, straight from Hugging Face

TSAI's ask (26 Nov 2026): people should download the lab model from Hugging Face and run it in Ollama, which has no
Kronecker (KE2/KAS) embedding. No Ollama change is needed: `../scripts/export_hf.py` folds the Kronecker embedding and
the KAS head **exactly** into a standard Granite MoE Shared model, and Ollama and llama.cpp run that as arch
`granitemoe`.

```
G5 checkpoint --export_hf.py--> HF Granite MoE Shared --package_ollama.py--> <name>-BF16.gguf, <name>-Q8_0.gguf,
                                                                            Modelfiles, README.md (model card)
                                    ollama run hf.co/<TSAI org>/<repo>:Q8_0
```

| File | What |
|---|---|
| [`package_ollama.py`](package_ollama.py) | HF export -> BF16 + Q8_0 GGUF (frozen tokenizer, forced pre-tokenizer, Gurmukhi vocabulary remap), Modelfiles, model card, `release.json`; `--smoke` runs each file in Ollama and in llama.cpp and requires the same greedy text |
| [`remap_vocab.py`](remap_vocab.py) | rewrites token strings so llama.cpp's NFC-only normalization gives the training ids for Gurmukhi; ids never change |
| [`measure_export_gaps.py`](measure_export_gaps.py) | on real text: bits per byte of the checkpoint against the checkpoint without each part Granite cannot carry, and the cost of llama.cpp's tokenization |
| [`stage_texts.py`](stage_texts.py) | stages Wikipedia text (10 languages) for the measurement, once, with network |
| [`run_ollama_job.py`](run_ollama_job.py) | export + measurement + packaging in one command, offline (for the dry run 2 job, once that checkpoint exists) |
| [`smoke_standin.md`](smoke_standin.md) | the whole path on a random checkpoint in dry run 2's shape: Ollama == llama.cpp, remap, fold |

## What the GGUF version cannot carry

| Part | In Granite / Ollama | Handling |
|---|---|---|
| KE2/KAS input embedding, KAS head (W_out, beta, rank-64 correction) | folded exactly | test A1 |
| KAS head bias (`head.unigram`) | no output bias | dropped, or approximated by the fold below; both measured (`no_head`, `granite_fold`) |
| Router selection bias (loss-free balancer, `bias_gamma` > 0) | no selection bias | dropped, or approximated by the fold below; both measured (`no_router`, `granite_fold`) |
| p1536 normalizer (NFC + 15 replacements) | GGUF carries NFC only | Gurmukhi: vocabulary remap; Bengali, Malayalam: measured limit (below) |

**Stock Ollama only.** Ollama ships its own llama.cpp, so a patched llama.cpp (the route to exact bias parity, which
#780's follow-up takes for vLLM) cannot run there until it lands upstream in llama.cpp and then in Ollama. In llama.cpp
b11669, 33 architectures have a selection-only router bias (`exp_probs_b`) and 8 an output bias (`output_b`); none has
both, and none matches our block otherwise. The Ollama release is therefore the Granite export.

**The fold (`granite_fold`).** Still stock Granite weights: fit one vector v with h.v ~= 1 over the final normalized
hidden states h of calibration text, and add unigram (x) v to `lm_head`, so logits ~= W h + unigram. For the router,
G5 selects `topk(softmax(Rx) + beta)` (the bias is on probabilities), so each layer gets a logit shift
d = log(1 + beta / p_k) (p_k: the probability at the top-k boundary on calibration text) folded the same way through
that layer's input. On the random stand-in (no trained structure), the fold halves the gap: mean KL against the
checkpoint 1.35 -> 0.64 (Bengali) and 1.16 -> 0.54 (Punjabi); the folded routers pick the same experts as the checkpoint
93% of the time, against 89% with the bias dropped. The trained checkpoint decides whether it is good enough.

## Tokenizer: the normalizer in llama.cpp

p1536 normalizes NFC, then recomposes Bengali ড় ঢ় য় and six Gurmukhi letters that NFC splits (Unicode composition
exclusions), and joins six Malayalam chillu sequences. A GGUF carries at most the NFC flag (`normalizer.nfc`, applied by
llama.cpp b11669), and **Ollama 0.40.18 and llama.cpp 10050 apply no normalization at all** (measured: their token ids
equal a no-normalizer tokenizer's on every test prompt). So what a user gets depends on how the text is typed:

| Input | Ollama today, remapped GGUF (default) | Ollama today, stock vocabulary | A runtime that applies NFC (b11669+) |
|---|---|---|---|
| Gurmukhi, split letter + nukta (web text, NFC form) | training ids | other ids | training ids (remapped) |
| Gurmukhi, precomposed letter | other ids | training ids | training ids (remapped) |
| Bengali, precomposed য় | training ids | training ids | other ids |
| Bengali, split য + nukta (web text, NFC form) | other ids | other ids | other ids |

The remap is on by default because web and NFC-normalized text (the usual source of pasted prompts, and Wikipedia)
stores these Gurmukhi letters split. Earlier measurement on Wikipedia text with llama.cpp's own tokenizer:

| Text | Lines whose ids differ from training: stock GGUF | With `remap_vocab.py` (Gurmukhi) |
|---|---:|---:|
| Punjabi (28 lines) | 57.1% | **10.7%** |
| Bengali (60 lines) | 86.7% | 86.7% (unchanged; not remapped) |

Bengali cannot be fixed this way: byte-level merges join a vowel sign with the lead bytes of the next letter before
য is complete, so a split য় never reaches a new (য, ়) merge, and reordering merges would change the training ids of
every plain য. Malayalam is not remapped because renaming chillu tokens would break users who type the atomic chillu.
The cost of what remains is measured per language by `measure_export_gaps.py` (`tokenizer_cost_bpb`).

## Rerun on a stand-in

```
python ../scripts/make_random_checkpoint.py --tokenizer tokenizer.json --out standin --d-model 512 --n-layer 8 \
    --n-head 8 --n-kv-head 2 --d-ff 768 --n-expert 16 --n-shared 1 --top-k 2 --seq-len 2048 \
    --head-mode kas_v2 --bias-gamma 0.001 --router-bias-std 0.01
python run_ollama_job.py --checkpoint standin --tokenizer tokenizer.json --texts texts --llama-cpp llama.cpp-b11669 \
    --work work --out out --name era-v5-standin
python package_ollama.py --hf-dir work/hf_export ... --out work/release --smoke     # needs Ollama running
```

Python: the venv of llama.cpp b11669's `requirements/requirements-convert_hf_to_gguf.txt` (torch 2.11.0,
transformers 4.57.6) runs all of it; `export_hf.py` needs transformers 4.57.x (5.x renamed the MoE weights).
