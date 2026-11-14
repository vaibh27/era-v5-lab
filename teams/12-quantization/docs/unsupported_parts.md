# Parts that llama.cpp and vLLM do not support yet (G12, week 2)

**Status:** part 1, 4 Nov 2026: the KE2 embedding, attention and the tokenizer for config v0 and tokenizer v0
(sections 1–6). Part 2, 13 Nov 2026: one plan per custom part of config v1, the Gate 2 proposal that TSAI signs on
14 Nov (sections 7–12). Part 2 is re-checked after the signing; sections 10 and 11 cover the values still open.

**Checked against:**
- llama.cpp **b11669** (= v0.7.0), the build used for every GGUF in this folder. File and line references are to
  `convert_hf_to_gguf.py`, `conversion/base.py` and `src/llama-vocab.cpp` at that tag.
- vLLM: the serving path that Member C sets up in `gpu_path/`. The vLLM version is pinned there; the vLLM rows
  below describe how vLLM loads models in general and are marked "check on the pinned version" where we have not run it.
- Inputs: `teams/04-ke2/output_shape.md` (draft v0, P = 32), `teams/05-architecture/configs/config_v0.yaml`,
  `teams/03-tokenizer/v0/` (BrahmicTokenizer-131K with renamed reserved slots, `tokenizer.json` sha256
  `01d968a7…739fff` for the baseline).

## Summary

| Part | llama.cpp today | vLLM today | What breaks | Plan |
|---|---|---|---|---|
| KE2 input embedding | no: expects a `token_embd` table | no: expects an `nn.Embedding` | the checkpoint has no embedding table, only a `Linear(8192, d_model)` | **fold** KE2 + projection into a normal `V × d_model` table at export. Exact, no runtime change |
| KE2 output side | depends on the design | depends on the design | not decided until 14 Nov | config v1 proposes the dense untied head, which both runtimes load (section 9.2). Tied: fold the same way |
| Attention and MoE block (config v1) | yes, as `granitemoe` | yes, as `GraniteMoeSharedForCausalLM` | nothing, if the export applies three exact rewrites | export as Granite MoE Shared: permute q/k rows for the RoPE layout, write the attention scale, merge shared experts (sections 8, 9) |
| Open attention arms (attn-2, attn-3, attn-4) | attn-2 yes (`granite_swa`); attn-3 needs a 1-line patch; attn-4 no | attn-2 and attn-3 yes; attn-4 no | attn-4 (output gate) needs new runtime code in both | section 10: tell TSAI before it picks attn-4 |
| Tokenizer pre-tokenizer | **converter stops**: hash not recognised | yes: uses `tokenizer.json` directly | the obvious workaround (`gpt-4o`) **splits every Indic word** at its vowel signs | register the tokenizer as `granite-embed-multi`-type pre-tokenizer; prove it with Member E's round trip |
| Tokenizer normalizer (G3 v1 idea, NFC + chillu + nukta) | **no**: the BPE path has no normalizer | yes, if written as `tokenizers` normalizers | llama.cpp would tokenize un-normalized text differently | normalize before tokenizing in the serving app, or keep the normalizer in data prep only |
| MTP head (`loss_mtp`, t+2) | no: converter stops on unknown tensors | no generic support | export fails, or the head is silently ignored | drop the head at export for the demo |
| Router (`ffn_gate_inp`) | kept F32, never quantized | kept bf16 with the `ignore` list | nothing, if the rule is followed | keep checking it in every output file (week-1 rule) |

## 1. KE2 input embedding

**What it is.** KE2 replaces the token-embedding table (`teams/04-ke2/output_shape.md`). A token id becomes a fixed
code of `D = 256 · P = 8,192` numbers (P = 32), built from the token's bytes, with no learned weights. One learned
`Linear(8192, d_model, bias=False)` follows. The output has the same shape as `nn.Embedding`: `(B, T, d_model)`.

**llama.cpp today.** Every llama.cpp architecture reads the input embedding from one tensor, `token_embd.weight`
(`V × d_model`), and looks rows up with `ggml_get_rows`. There is no op that builds a code from token bytes. The
converter maps each checkpoint tensor to a GGUF name and stops on any tensor it does not know
(`conversion/base.py`, `map_tensor_name`: `ValueError("Can not map tensor …")`). A checkpoint with a
`ke2.proj.weight` and no embedding table cannot be converted.

**vLLM today.** Each vLLM model class owns an embedding module (usually `VocabParallelEmbedding`). A KE2 model would
need its own model class. vLLM can also accept precomputed `prompt_embeds`, but that moves the embedding out of the
server and does not cover the tokens the model generates. Check both on the pinned version.

**What breaks.** Nothing works without a change: neither runtime can load the checkpoint as it is.

**Plan: fold KE2 into a normal table at export.** The code depends only on the token id, and the projection is
linear, so for every id the embedding is a fixed vector:

```
E[id] = W · code(id)            W: d_model × 8192,  E: 131,072 × d_model
```

Computing `E` once and writing it as `token_embd.weight` (GGUF) or `embed_tokens.weight` (HF/vLLM) gives a standard
model that both runtimes already load. It is exact, not an approximation. It is also cheap: after z-normalisation
every coordinate of a code has one of two values that depend only on the token length L (KE2 Lemma 1,
`teams/04-ke2/src/ke2/codec.py`, `code_values`), so

```
E[id] = low(L) · Σ_j W[:, j]  +  (high(L) − low(L)) · Σ_{p < L} W[:, b_p · P + p]
```

which needs L column reads per token, not a dense `131,072 × 8,192` product.

| | Before fold (training) | After fold (served) |
|---|---|---|
| Input parameters at `d_model = 768` | 6.29 M (`8,192 × 768`) | 100.66 M (`131,072 × 768`) |
| bf16 size of that part | 12.6 MB | 201 MB |
| Runtime code needed | KE2 codec + matmul | none (table lookup) |

The fold gives up KE2's parameter saving at inference time. That is the price of running on stock llama.cpp and
vLLM. The table can be quantized like any `token_embd` (llama.cpp `--token-embedding-type`); the drop must then be
measured like any other tensor.

**Acceptance test (to run when the first KE2 checkpoint exists):**
1. Fold in float64 with the closed form above; store as bf16.
2. For 10,000 random ids plus all 356 special tokens, compare the folded row with the KE2 forward pass
   (`ke2.codec.encode_ids` then the projection) in float32. Pass: max abs difference at bf16 rounding level.
3. Convert the folded checkpoint to GGUF and compare bf16 GGUF logits with the PyTorch model on 64 prompts.

**Owner / date:** G12 Member D writes `fold_ke2.py` once config v1 fixes `d_model` and G4 confirms the projection's
weight name (week 3).

## 2. KE2 output side

`output_shape.md` says the output side (the next-token layer) is not decided; the KE2 decision is on 14 Nov. Three
cases, with the plan for each:

| Output design | llama.cpp | vLLM | Plan |
|---|---|---|---|
| Normal `lm_head` (`d_model × V`) | yes (`output.weight`) | yes | nothing to do; keep it at Q8_0 or bf16 (week-1 rule) |
| Tied to the KE2 input (`logits = h · Eᵀ`) | yes: if `output.weight` is missing, llama.cpp uses `token_embd` | yes (`tie_word_embeddings`) | fold as in section 1; the folded `E` serves both ends |
| Anything that is not a `V × d_model` matrix (for example a byte-level output) | no | no | needs a custom architecture in both runtimes; plan in part 2 |

## 3. Attention

**What config v0 says.** Only `attention.impl: eager`. All sizes (`n_layer`, `d_model`, `n_head`, `d_ff`,
`seq_len`, `n_expert`, `top_k`) are empty until TSAI fills them. `eager` is a training choice: the harness uses
eager attention so that two runs with the same seed match. It is not part of the model. The runtimes use their own
kernels (llama.cpp flash attention, vLLM FlashAttention or FlashInfer). The numbers differ slightly from eager,
which is why we compare a quantized score only with the bf16 score from the same runtime (week-1 rule).

**llama.cpp and vLLM today.** Both support standard multi-head and grouped-query attention, RoPE with the usual
scaling types, QK-norm (OLMoE, Qwen3-MoE) and sliding-window attention for the architectures that define it. Both
support OLMoE and Qwen3-MoE end to end (`tools/tools.md`). Neither runs an attention variant that no existing
architecture uses.

**What breaks.** Nothing yet; we cannot tell until config v1. The risk is a custom choice that no existing
architecture has.

**Plan.** Export the model under an existing architecture whose attention and MoE block match config v1, so no runtime
code is needed. When config v1 arrives, check each item against OLMoE (`src/models/olmoe.cpp`, vLLM `olmoe.py`) and
Qwen3-MoE:

- heads and KV heads (MHA or GQA), head dimension
- position encoding: RoPE type, theta, scaling for `seq_len`
- QK-norm or not; norm type (RMSNorm or LayerNorm) and its placement
- shared experts, router normalisation, top-k
- any attention bias, logit soft-capping or sliding window

Each item that matches no existing architecture becomes a custom part with its own plan in part 2.

## 4. Tokenizer

**What it is.** Tokenizer v0 is BrahmicTokenizer-131K with renamed reserved slots (`teams/03-tokenizer/v0/`). From its
`tokenizer.json`: byte-level BPE, vocab 131,072, 356 special tokens, `ignore_merges: true`, **no normalizer**, and a
pre-tokenizer regex that is exactly GPT-4o's (o200k) regex, which includes `\p{M}` (combining marks) in its letter classes.

### 4a. Pre-tokenizer: measured

**vLLM today.** vLLM loads the Hugging Face tokenizer from `tokenizer.json`, so it splits text exactly like the
training pipeline. Low risk.

**llama.cpp today.** llama.cpp does not read the regex from `tokenizer.json`. The converter encodes a fixed test string,
hashes the token ids and looks the hash up in a hard-coded list (`conversion/base.py`, `get_vocab_base_pre`). Each
entry maps to a pre-tokenizer type with its own regex in C++ (`src/llama-vocab.cpp`).

Measured on 1 Nov with the baseline `tokenizer.json`:

| Check | Result |
|---|---|
| Our hash | `57e4bd19…2f7748d7` |
| In the b11669 list? | **no**: conversion stops with `BPE pre-tokenizer was not recognized` |

The usual workaround is to add our hash and map it to the type with the same regex, `gpt-4o`. **That breaks
Indic text.** llama.cpp's `gpt-4o` type rewrites the regex with lookaheads that leave out `\p{M}`. Indic vowel signs
and the virama are `\p{M}`, so every Indic word is cut at its vowel signs before BPE runs. The type
`granite-embed-multi-97m` uses the same lookaheads **with** `\p{M}` added (llama.cpp PR #22716), and it also sets
`ignore_merges`, as our tokenizer does.

Pre-token splits of the three regexes, run with Python `regex` on sample sentences:

| Text | HF `tokenizer.json` | llama.cpp `gpt-4o` | llama.cpp `granite-embed-multi` |
|---|---|---|---|
| Hindi | 10 pieces: `हिन्दी`, ` भारत`, … | **23 pieces:** `ह`, `िन`, `्द`, `ी`, … | same as HF |
| Bengali | 6 | **16** | same |
| Tamil | 5 | **13** | same |
| Malayalam | 4 | **7** | same |
| Telugu | 4 | **13** | same |
| English | 10 | same | same |
| Code | 13 | same | same |

A wrong split does not crash. It gives different token ids from training, so the served model sees text it never
saw. This would show up as an Indic-only score drop that has nothing to do with the bit width.

**Plan.**
1. In our converter run, map hash `57e4bd19…` to `granite-embed-multi-97m`: one line in `get_vocab_base_pre`, or
   write `tokenizer.ggml.pre = "granite-embed-multi-97m"` into the GGUF.
2. Member E's `tokenizer/gguf_roundtrip.py` is the proof. Encode 1,000 lines per script and of code with llama.cpp
   and with HF `tokenizers`. Pass: identical ids and lossless decode for every line. This checks llama.cpp's own
   regex engine, not just Python `regex`.
3. Repeat on the frozen tokenizer (14 Nov). If G3 changes the regex, recompute the hash and redo this check.
4. Upstream: once it passes, open a llama.cpp PR adding the hash, so later builds need no local patch.

### 4b. Special tokens and chat template

| Item | llama.cpp | vLLM | Note |
|---|---|---|---|
| 356 specials, renamed reserved slots (`<\|g10_slot_00\|>`, …) | yes: GGUF stores each token's string and type | yes | renames keep ids; nothing to change |
| BOS/EOS, `<\|im_start\|>` / `<\|im_end\|>` (G10 ChatML v0) | yes: set `tokenizer.ggml.eos_token_id` and the end-of-generation tokens correctly, or generation does not stop | yes: from `generation_config.json` | check that the GGUF's EOS and EOT ids are 130,822 (`<\|endoftext\|>`) and 130,816 (`<\|im_end\|>`) |
| Chat template | yes: Jinja in `tokenizer.chat_template` | yes | G10 sends it on 12 Nov; test it with `llama-server` and vLLM chat endpoints |

### 4c. Normalizer (G3 v1 idea #134)

Tokenizer v0 has no normalizer, so there is nothing to break today. G3's v1 idea #134 adds one: NFC, chillu
re-join and Bengali/Punjabi nukta re-join.

| Runtime | Supported? | Detail |
|---|---|---|
| vLLM | yes | it runs whatever `tokenizer.json` contains; the rules can be written as `tokenizers` normalizers (`NFC` + `Replace`) |
| llama.cpp | **no** | its BPE path applies no normalizer at all; at b11669 the only `nfc` flag is used by the Unigram (UGM) tokenizer |

If the frozen tokenizer has a normalizer, the same raw text would give different ids in llama.cpp.

**Plan:** prefer normalizing in data prep and in the serving app, before the text reaches the tokenizer, so the
tokenizer itself has none. If G3 keeps it inside `tokenizer.json`, add normalized and un-normalized lines to the
round trip and list each difference with its cause.

## 5. MTP head

Config v0 trains a second linear head scored at t+2 (`loss_mtp`). It is not a stacked MTP block.

| Runtime | Status |
|---|---|
| llama.cpp | stops on the unknown tensor (`Can not map tensor`) |
| vLLM | uses MTP heads only for some architectures, for speculative decoding; a model class drops or rejects unknown weights depending on its `load_weights` |

**Plan:** drop the head at export for the demo; next-token generation does not use it. Keep it in the training
checkpoint. Reconsider only if TSAI wants speculative decoding, which would need a custom draft path in both
runtimes.

## 6. Router

Already handled; listed so the rule is not lost. llama.cpp never quantizes `ffn_gate_inp`
(`src/llama-quant.cpp`). llm-compressor keeps it in bf16 with `ignore=["re:.*mlp.gate$"]`, and Member C must confirm
that the regex matches the router name in our model (`tools/tools.md`, open checks). Every output file is checked
(week-1 rule).

## What we need from other teams

| From | What | When | Used in |
|---|---|---|---|
| G4 | final `output_shape.md`; the projection's weight name; the output-side decision | 4 Nov; 14 Nov | sections 1, 2 |
| G5 | config v1: sizes, attention and MoE details, custom parts | 14 Nov | section 3, part 2 |
| G3 | frozen tokenizer; whether the normalizer is in `tokenizer.json` | 14 Nov | section 4 |
| G10 | chat template | 12 Nov | section 4b |
| G12 Member E | `results/gguf_roundtrip_v0.md`, then `results/gguf_roundtrip_frozen.md` | 4 Nov, 14 Nov | section 4a |

## Reproduce the pre-tokenizer check

```bash
pip install regex tokenizers
curl -sLO https://huggingface.co/theschoolofai/BrahmicTokenizer-131K/resolve/main/tokenizer.json
curl -sLO https://raw.githubusercontent.com/ggml-org/llama.cpp/b11669/conversion/base.py
curl -sLO https://raw.githubusercontent.com/ggml-org/llama.cpp/b11669/src/llama-vocab.cpp
```

```python
import hashlib, json, re, regex
from tokenizers import Tokenizer

tok = Tokenizer.from_file("tokenizer.json")
base = open("base.py", encoding="utf-8").read()
chktxt = eval(re.search(r"chktxt = ('.*?')\n", base).group(1))
h = hashlib.sha256(str(tok.encode(chktxt, add_special_tokens=False).ids).encode()).hexdigest()
print("hash", h, "known:", h in base)

hf = json.load(open("tokenizer.json", encoding="utf-8"))["pre_tokenizer"]["pretokenizers"][0]["pattern"]["Regex"]
cpp = open("llama-vocab.cpp", encoding="utf-8").read()

def first_regex(case):
    i = cpp.index("regex_exprs = {", cpp.index(case))
    return eval(next(l.strip().rstrip(",") for l in cpp[i:cpp.index("};", i)].splitlines() if l.strip().startswith('"')))

split = lambda p, s: [m.group() for m in regex.finditer(p, s)]
text = "हिन्दी भारत की राजभाषा है।"
for name in ("LLAMA_VOCAB_PRE_TYPE_GPT4O:", "LLAMA_VOCAB_PRE_TYPE_GRANITE_EMB_MULTI:"):
    print(name, split(first_regex("case " + name), text) == split(hf, text))
```

---

# Part 2: one plan per custom part of config v1 (13 Nov 2026)

**Inputs.** `teams/05-architecture/configs/config_v1.yaml` as merged in #374 (`f0c5b84`, main `14e5789`), the
Gate 2 proposal that TSAI signs on 14 Nov; `decisions.md`; `configs/runtime.yaml` and `configs/arms/*.yaml`; and the
model that the config trains, `teams/05-architecture/harness/week2/model.py`.

**Checked against.** llama.cpp **b11669** (`620d222`), as in part 1, and vLLM **0.33.0**, the version in
`gpu_path/results/environment.json`. Every row below was read in the source at those tags. The two weight rewrites in
9.4 and 9.6 were checked on small random matrices with `scripts/check_export_rewrites.py` (max differences 8.5e-14
and 5.7e-14, float64 rounding; the llama.cpp permute restores our rows exactly). **No model was run**: no checkpoint
of the config v1 model exists yet. Section 12 lists the tests to run when one does.

## 7. What config v1 fixes and what it leaves open

| Setting | Config v1 (proposal) | Matters for serving? |
|---|---|---|
| `embedding.type`, `pos_dim` | `kronecker`, 32 | yes: section 9.1 (fold, part 1 §1) |
| `embedding.ke2_output_dim` | null; "test models use the dense untied head" | yes: section 9.2 |
| `loss.train_on` | `ntp`; a second head on t+2 is logged as `loss_mtp` | yes: section 9.3 (drop, part 1 §5) |
| `attention.impl` | `eager`; "RoPE and the window are not copied in" | no: training setting. The attention itself is in `model.py` (section 8) |
| `model.*` sizes, `moe.n_expert`, `moe.top_k`, shared expert, `data.seq_len` | null (TSAI fills) | they set numbers in the export config, not which architecture we use, except shared experts on/off (section 10) |
| `precision` | null | no: we export bf16 whatever the training precision |
| `optim.*`, `run.*` | AdamW, lr 1e-3 (T1 only), seed, clip | no |

Fixed by the harness, not by `config_v1.yaml` (`model.py`, `runtime.yaml`): pre-norm RMSNorm (eps `1e-5`, test
setting), GQA without bias and without QK-norm, RoPE on adjacent pairs (theta `10000`, test setting), attention scale
`1/sqrt(head_dim)`, SwiGLU experts without bias, router softmax in fp32 then top-k then renormalise, shared experts
added with weight 1, final RMSNorm, untied `head`.

## 8. Export target: Granite MoE Shared

Part 1 §3 planned to export under OLMoE or Qwen3-MoE. **Neither fits config v1**:

- Both normalise q and k (QK-norm: llama.cpp `src/models/olmoe.cpp:28-29`, `src/models/qwen3moe.cpp:36`). Our model has
  none, and RMSNorm cannot be switched off by choosing its weights.
- OLMoE does not renormalise the top-k weights (`olmoe.cpp:143`, `norm_w = false`); we do.
- Qwen3-MoE has no shared expert; we have `num_shared` of them (1 in T1 and T2).
- Plain `llama` (Mixtral layout) loads shared-expert tensors but never uses them: `src/models/llama.cpp:84-89` creates
  `ffn_*_shexp` "for Granite MoE Shared", and the MoE branch at `:203-215` does not add them.

IBM's **Granite MoE Shared** matches every part of the default block:

| Our model (`model.py`) | llama.cpp b11669, arch `granitemoe` | vLLM 0.33.0, `GraniteMoeSharedForCausalLM` | Match |
|---|---|---|---|
| Pre-norm RMSNorm, eps from config | RMS norm (`granite.cpp`) | `RMSNorm(rms_norm_eps)` | yes |
| GQA, no bias, no QK-norm | `build_qkv`, no q/k norm | `QKVParallelLinear(bias=False)` | yes |
| RoPE on **adjacent pairs** `(x[2i], x[2i+1])` | NORM rope type = adjacent pairs (`llama-model.cpp`, `GRANITE_MOE → LLAMA_ROPE_TYPE_NORM`) | `is_neox_style=True` = split halves (`granitemoe.py:205`) | yes, after a q/k row permutation (9.4) |
| Scale `1/sqrt(head_dim)` | default when `attention_scale` is absent (`granite.cpp:233`) | `attention_multiplier`; default `head_dim**-1` if None (`granitemoe.py:173-177`) | yes, if we **write** the multiplier (9.5) |
| Router: fp32 softmax → top-k → renormalise | `build_moe_ffn(..., norm_w = true, SOFTMAX)` (`granite.cpp:277-286`) | `FusedMoE(renormalize=True)` (`granitemoe.py:112`) | yes |
| SwiGLU experts, no bias | `LLM_FFN_SILU`, split gate/up | merged gate/up | yes |
| Shared experts added with weight 1 | `ggml_add(moe_out, ffn_shexp)` (`granite.cpp:291-300`) | `moe + shared_mlp(x)` (`granitemoeshared.py:146`) | yes; several shared experts merge into one (9.6) |
| Untied `head`, final RMSNorm | `output.weight` | `lm_head`, `tie_word_embeddings: false` | yes |
| Granite extras: embedding, residual, logit multipliers | applied when set | applied | set all three to `1.0` |

The llama.cpp converter has the class already: `conversion/granite.py:179`,
`@ModelBase.register("GraniteMoeForCausalLM", "GraniteMoeSharedForCausalLM")`. Its arch name is unrelated to the
`granite-embed-multi` pre-tokenizer in part 1 §4a; the two choices are independent.

**Plan.** G12 writes `export_hf.py`: G5 checkpoint → a Hugging Face `GraniteMoeSharedForCausalLM` folder
(`config.json` + safetensors + the tokenizer files). vLLM loads that folder directly. The stock b11669
`convert_hf_to_gguf.py`, with the part 1 §4a pre-tokenizer line, turns it into GGUF. Neither runtime needs new code
for the default config v1.

`config.json`: `architectures: ["GraniteMoeSharedForCausalLM"]`, `model_type: "granitemoeshared"`, `vocab_size:
131072`, `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`, `intermediate_size`
(one expert), `shared_intermediate_size` (all shared experts together, 9.6), `num_local_experts`,
`num_experts_per_tok`, `rms_norm_eps`, RoPE theta, `max_position_embeddings` = `data.seq_len`,
**`attention_multiplier` = `head_dim ** -0.5`**, `embedding_multiplier: 1.0`, `residual_multiplier: 1.0`,
`logits_scaling: 1.0`, `tie_word_embeddings: false`, `hidden_act: "silu"`, and the BOS/EOS ids from part 1 §4b.
Take the exact key names from the installed `transformers` config class, not from this list.

Weight map (G5 `state_dict` → Hugging Face):

| G5 | Hugging Face (Granite MoE Shared) | Transform |
|---|---|---|
| `embedding.projection.weight`, buffers `embedding.rows`, `embedding.lengths` | `model.embed_tokens.weight` | KE2 fold (9.1); drop the buffers |
| `blocks.i.n1.weight` | `model.layers.i.input_layernorm.weight` | none |
| `blocks.i.attn.q.weight`, `.k.weight` | `…self_attn.q_proj.weight`, `…k_proj.weight` | row permutation per head (9.4) |
| `blocks.i.attn.v.weight`, `.o.weight` | `…v_proj.weight`, `…o_proj.weight` | none |
| `blocks.i.n2.weight` | `…post_attention_layernorm.weight` | none |
| `blocks.i.moe.router.weight` | `…block_sparse_moe.router.layer.weight` | none |
| `blocks.i.moe.experts.e.{gate,up,down}.weight` | `…block_sparse_moe.input_linear.weight` `[E, 2h, d]` = cat(gate, up); `…output_linear.weight` `[E, d, h]` | stack, gate before up (`conversion/granite.py` splits it in that order) |
| `blocks.i.moe.shared.s.{gate,up,down}.weight` | `…shared_mlp.input_linear.weight`, `…shared_mlp.output_linear.weight` | merge (9.6) |
| `norm.weight`, `head.weight` | `model.norm.weight`, `lm_head.weight` | none |
| `mtp_head.weight` (only if MTP is on) | — | drop (9.3) |

`export_hf.py` builds the target key list from `transformers`' own `GraniteMoeSharedForCausalLM(config).state_dict()`,
and asserts that the two key sets and every shape match before it writes anything.

## 9. Plan per custom part

### 9.1 KE2 input embedding

Part 1 §1 holds: fold KE2 and its projection into a `V × d_model` table. One change. Build the table by running the
harness module itself, `LanguageModel.embedding` (`model.py`, class `KE2`), in float64 over all 131,072 ids in chunks,
not by re-deriving the closed form. The table then reproduces the trained module exactly, including its `+ 1e-6`
term and unbiased standard deviation (`KE2.projected`). Part 1's acceptance test stays.

### 9.2 Output head

Config v1 keeps `ke2_output_dim` empty, and `model.py` builds `head = nn.Linear(n_embd, vocab_size, bias=False)`,
untied ("Dense untied output head per TSAI review", docstring). Both runtimes load it as `lm_head` / `output.weight`.
Nothing to do. Keep it at Q8_0 or bf16 (week-1 rule).

**Watch:** the T1 and T2 arm files say `tied_embeddings_intended: true`, while `model.py` is untied. If TSAI ties the
head on 14 Nov, part 1 §2 row 2 applies: drop `lm_head`, set `tie_word_embeddings: true`, and the folded table serves
both ends.

### 9.3 MTP head

`runtime.yaml` sets `mtp_weight: 0.0` and every arm sets `mtp: false`, so the test checkpoints have no `mtp_head`.
Config v1 says a second head is trained on t+2. If the production checkpoint has `mtp_head.weight`, `export_hf.py`
drops it (part 1 §5); next-token generation does not use it.

### 9.4 RoPE pair layout (exact rewrite)

**What it is.** `GQA.rotary` rotates adjacent pairs `(x[2i], x[2i+1])` with frequency `theta^(-2i/d)`. vLLM's Granite
path and the Hugging Face format rotate split halves `(x[i], x[i + d/2])` with the same frequencies.

**What breaks.** Loading our q/k weights unchanged into vLLM rotates the wrong pairs. Nothing crashes, but every
attention score is wrong.

**Plan.** In `export_hf.py`, reorder the output rows of `q_proj` and `k_proj` within each head:
`new = cat(old[0::2], old[1::2])`. q and k get the same reordering, so `q·k` keeps its value; v and o are untouched.
The llama.cpp converter then applies its own `LlamaModel.permute` (`conversion/llama.py:176-180`), which interleaves
the halves back, which is exactly our original layout, and llama.cpp's NORM rope rotates adjacent pairs. One rewrite
serves both runtimes.

### 9.5 Attention scale

We scale by `1/sqrt(head_dim)`. Hugging Face's Granite config defaults `attention_multiplier` to 1.0, and vLLM falls
back to `head_dim**-1` when it is None. **Write `attention_multiplier = head_dim ** -0.5`.** The converter copies it
into GGUF as `attention_scale` (`conversion/granite.py:38-40`), so llama.cpp uses the same value.

### 9.6 Shared experts

Granite has one shared MLP. Several SwiGLU shared experts of width `h` sum to one SwiGLU of width `S·h`: stack the
gate rows, stack the up rows (same order), and concatenate the down columns. This is exact. Write
`shared_intermediate_size = S·h`. If EXP-3 sets `num_shared: 0`, write `0`: vLLM then builds no shared MLP
(`granitemoeshared.py:112`), and llama.cpp skips it (`n_ff_shexp > 0` check).

### 9.7 Router precision

`MoE.forward` computes the router in fp32 with autocast off. llama.cpp keeps `ffn_gate_inp` in F32 (part 1 §6). vLLM
runs the router linear in the model dtype (check on 0.33.0). A bf16 router can flip a top-k choice when two scores are
close. Section 12 test A4 measures this; if flips are frequent, keep the router weight in fp32 in the vLLM export.

## 10. Arms that are still open

`decisions.md` adopts no attention arm and no EXP-3 arm for Gate 2; EXP-4 is week 3. If TSAI adopts one, this is the
serving cost.

| Arm | Change (`configs/arms/`) | llama.cpp b11669 | vLLM 0.33.0 | Plan |
|---|---|---|---|---|
| attn-1 MQA | `n_kv_head: 1` | yes | yes | nothing |
| attn-2 window + sink | `LLLGLLLG`, window 128, a learned sink logit per head on L layers | yes: `granite_swa` has a window pattern, a standard window and per-layer sinks (`granite-swa.cpp:17-19, 82`); converter class `GraniteMoeSWAModel` (`conversion/granite.py:124`) | yes: `GraniteMoeSWAForCausalLM` is served by `granitemoeshared` (`registry.py:135`), with window and sinks per layer (`granite.py:95-110`) | export as `GraniteMoeSWAForCausalLM` with `layer_types` and `sliding_window: 128`; see the notes below |
| attn-3 partial RoPE | `rotary_fraction: 0.5` | **no**: `granite.cpp:131` and `granite-swa.cpp:129` assert `n_embd_head == n_rot`, and the converter permutes the whole head | yes: `get_rope` reads `partial_rotary_factor` (`rotary_embedding/__init__.py:69-72`) | vLLM: apply 9.4 to the first `rotary_dim` rows of each head only. llama.cpp: local patch (below) |
| attn-4 output gate | `sigmoid(W_g · x) * attn_out` before `o` | **no** | **no** | needs new runtime code in both (below) |
| EXP-3 experts 8 / 32 | `num_experts` | yes | yes | config value only |
| EXP-3 shared 0 | `num_shared: 0` | yes | yes | 9.6 |
| EXP-2 Muon | optimizer | n/a | n/a | not a model part; the quantization drop is measured as usual |
| router or expert bias | `moe.bias`, `bias_gamma` | the harness rejects both today (`open_choices.json`) | — | if they come back: Granite has neither, so this becomes a custom part |

**attn-2 notes.**
- **Window rule.** Ours keeps keys with `i − j < 128`. llama.cpp masks when `p1 − p0 >= n_swa`
  (`llama-hparams.h:505-508`). That is the same rule.
- **Sinks on every layer.** Both runtimes expect a sink on every layer once `layer_types` is set: llama.cpp requires
  the tensor on every layer, and vLLM sets `has_sink = layer_types is not None`. Our G layers have none. Write
  `-1e4` (fp32) as their sink: `exp(-1e4 − max)` is exactly 0 in fp32, so those layers compute the plain softmax.
  `model.py` treats an all-`-inf` sink as "no sink" for the same reason.
- **Checks.**
  - `transformers` 5.20.2 must have a config class for the SWA model type.
  - vLLM's window must include the current token plus 127 earlier ones.

**attn-3 patch (llama.cpp).** Drop the assert in `granite.cpp` (or `granite-swa.cpp`), and write
`rope.dimension_count = rotary_dim`. `ggml_rope_ext` already rotates only the first `n_rot` dims; `phi2.cpp` and
`stablelm.cpp` use it that way without the assert. In the converter, skip `LlamaModel.permute` (`undo_permute =
False`) and write q/k in our own adjacent-pair layout. Two lines of C++ and a converter subclass, on a local build.
Offer it upstream after it passes section 12.

**attn-4 (output gate).** No architecture in either runtime has this gate on a plain attention block.
- **Patch.** A local llama.cpp patch adds an optional `attn_gate` tensor and computes
  `cur = cur * sigmoid(gate · attn_norm_out)` before `wo`. A vLLM out-of-tree model class, registered with
  `ModelRegistry`, subclasses the Granite attention the same way. Each also needs converter and loader mapping, and
  the section 12 tests.
- **Gate input.** The arm file sets `attn_gate_input: normalized`, which is the attention input we already have. The
  `residual` option would also need the pre-norm residual in the attention call, which is more work.
- **Cost.** Days, not hours, and a fork to maintain in both runtimes.
- **Ask.** **Raise this with TSAI before attn-4 is chosen.** It is the only open arm that cannot be served by stock
  runtimes.

## 11. What we need, and by when

| From | What | When | Used in |
|---|---|---|---|
| TSAI | signed config v1: sizes, `n_kv_head`, `top_k`, shared experts on/off, any arm | 14 Nov | 8, 9.6, 10 |
| TSAI | tied or untied head (the arm files and `model.py` disagree) | 14 Nov | 9.2 |
| G5 | the production model's `state_dict` key names, and confirmation that RoPE stays on adjacent pairs | with the first dry-run checkpoint | 8, 9.4 |
| G5 / G6 | whether the router stays fp32 in the production code | week 3 | 9.7 |
| G12 Member D | `export_hf.py` (sections 8, 9) together with `fold_ke2.py` | week 3, after 14 Nov | all |

## 12. Acceptance tests (when the first config v1 checkpoint exists)

| Test | Compare | Pass |
|---|---|---|
| A1 export | G5 `LanguageModel` (fp32) against `transformers` `GraniteMoeSharedForCausalLM` from the export (fp32), 64 prompts | logits equal to fp32 rounding; same top-1 at every position |
| A2 vLLM | vLLM bf16 prompt logprobs against `transformers` bf16, same prompts | differences at bf16 level; report top-1 agreement |
| A3 llama.cpp | bf16 GGUF (`llama-perplexity` / logits dump) against `transformers` bf16 | same as A2 |
| A4 routing | per layer, the share of tokens whose top-k expert set matches G5's fp32 router, in each runtime | report; investigate below 99% (9.7) |
| A5 attn-2 only | 600-token prompts that cross the 128-token window, A1–A3 | as A1–A3 |

A1 checks `export_hf.py` (9.4–9.6 and the weight map). A2 and A3 check that each runtime runs the architecture the
way we think it does. Only after these pass do score drops from quantization mean anything.

## Reproduce the source checks in part 2

```bash
python teams/12-quantization/scripts/check_export_rewrites.py   # 9.4 and 9.6, numpy only
git clone --depth 1 --branch b11669 https://github.com/ggml-org/llama.cpp.git
grep -n "n_rot\|attention_scale\|build_moe_ffn\|shexp" llama.cpp/src/models/granite.cpp
grep -n "shexp\|build_moe_ffn" llama.cpp/src/models/llama.cpp
grep -n "attn_q_norm\|true,\|false," llama.cpp/src/models/olmoe.cpp llama.cpp/src/models/qwen3moe.cpp
grep -n "register\|permute\|multiplier" llama.cpp/conversion/granite.py llama.cpp/conversion/llama.py
for f in granite.py granitemoe.py granitemoeshared.py registry.py; do
  curl -sLO https://raw.githubusercontent.com/vllm-project/vllm/v0.33.0/vllm/model_executor/models/$f; done
grep -n "scaling\|is_neox_style\|renormalize\|has_sink" granitemoe.py granite.py
```
