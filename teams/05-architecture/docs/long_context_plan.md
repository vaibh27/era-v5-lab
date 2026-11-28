# Production training context and long-context evidence

## Week 4 recommendation · 25 November 2026

**Start main-run pretraining at 2,048 tokens per sequence. Make 4,096 the first extension candidate.** The proposed Gate 4 value is `data.seq_len = 2048`. It describes the initial training window. A supported inference window or a later training phase needs its own evidence. The lead can submit this value for TSAI approval while the production model size is being settled. [Open production settings](../configs/config_v1.yaml) · [Week 4 work order, item 6](https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q?chat=cmvkupxl427gl09l7rv9m4fwy&msg=cmwk405xxxxxxxxxxxxxxx)

Two constraints favor starting at 2K. It is the window in G5’s completed T2 training rehearsal and in the current KAS V2 dry-run arm. The production target spans 30B–120B total parameters while the width, head count, precision and device layout are still open. On the 41 dry-run shards the current loader would fill 35% of training positions at 2K and 19% at 4K. This makes 2K a defensible initial choice on the available stack. **4K from the start is a credible alternative** if the actual corpus, target tasks and device profile support it before launch. At 1K, 43.07% of sample-v0 tokens lie beyond the first window of their document. Use it as a fallback if 2K cannot meet the budget and verify the quality trade-off. Defer 8K until a 4K checkpoint demonstrates useful long-range behavior. [Dry run 1 result](../jobs/evidence/2026-11-15-dryrun1-full/summary.json) · [Dry run 2 arm and corrected inputs](../jobs/2026-11-22-dryrun2.md)

### How often could a longer window help?

**The 41-shard dry-run corpus has a long tail.** We checked the frozen p1536 document offsets for every shard and removed one terminal EOS per document. “Tokens past 2K” is the sum of max(document length − 2,048, 0) divided by all content tokens in that row. It counts where distant history is available. The trainer still scores tokens in later 2K windows, carrying forward only the preceding token at each boundary. [G1 shard inventory and manifest pins](../../01-data-a/audit/shard_decode_patilvijay23/summary.json) · [G9 validation of all 41 shards](../../09-evaluation/results/decontam_shards.md)

| Data | Documents | Content tokens | Documents reaching 2K / 4K | Tokens past 2K / 4K |
|---|---:|---:|---:|---:|
| Sample-v0 train, all shards | 1,200,725 | 1,008,187,260 | 94,486 / 35,111 | 28.40% / 16.80% |
| ↳ Indic | 273,114 | 135,027,125 | 8,625 / 2,685 | 18.02% / 10.75% |
| ↳ Science | 5,993 | 39,464,282 | 2,346 / 1,540 | 80.83% / 71.11% |
| G5 evaluation, all 15 slices | 1,059 | 549,042 | 43 / 15 | 13.76% / 4.93% |
| ↳ Indic | 356 | 310,671 | 31 / 10 | 18.16% / 7.16% |

**Why choose 2K even with those long documents?** The current G5 training loader splits each document into separate windows and pads the last one. Over one pass through sample-v0, 2K would allocate **2.86B positions, 35.22% filled with targets**. At 4K the same documents would occupy **5.18B positions, 19.48% filled**. Only **18.32% of all targets** would land in the second half of a 4K window, where the candidate offers more history than a 2K window can hold. The longer setting would process 1.81× as many padded positions and about **3.62× as many dense attention pairs over the pass**. This calculation assumes the current document-isolated loader and eager kernel. Packing with correct document masking or a different production mix could change the result materially. The corpus-v1 distribution and its final training weights are still needed. [Training window and padding code](../harness/week2/data.py) · [Corrected dry-run input](../jobs/2026-11-22-dryrun2.md)

**What the held-out set can resolve.** The 15 documents reaching 4K comprise five math documents and two each in Assamese, Bengali, Hindi, Punjabi and Telugu. The English, code, science and six other Indic slices contain zero full-length 4K examples. In particular, the training science slice has a long tail that the frozen science evaluation slice cannot test. Keep the frozen short suite and add separate, clean, held-out long examples for the missing slices. [Frozen G5 evaluation set](../heldout_v1/README.md) · [All counts by domain and Indic language](../results/context_length/document_lengths.csv)

To reproduce the counts, use the [frozen p1536 release](../../03-tokenizer/release/README.md), NumPy and `tokenizers==0.23.3`. Place the 41 course-supplied shard manifests and their `.idx.npy` files in one directory. The [profiler](../harness/context_length_profile.py) verifies the manifests against the public inventory and verifies each index hash. It reads no training text. Omit `--sample-indexes` to reproduce just the evaluation rows with the public files.

```sh
python teams/05-architecture/harness/context_length_profile.py \
  --tokenizer /path/to/tokenizer.json --sample-indexes /path/to/sample-v0-indexes
```

**Downstream demand differs by task.** Team 10 counted complete instruction-tuning examples with the same tokenizer. Of 410,538 admitted code-reasoning examples, 18.45% fit at 2K, 41.84% at 4K and 69.63% at 8K. Of 10,541 Indic Dolly examples, 99.86% already fit at 2K. This makes code reasoning a priority for the longer-context test. A 4K training window alone also cannot support a 4K-token answer plus its prompt. It does not tell us the distribution of the pretraining corpus or the quality of answers at these lengths. [Code-reasoning lengths, after the evaluation firewall](../../10-post-training/sft/results/sft_v1_lengths_p1536/sft-code-opencodereasoning-2.json) · [Indic Dolly lengths](../../10-post-training/sft/results/sft_v1_lengths_p1536/sft-indic-indic-align-dolly-t.json)

**What Week 3 measured.** With identical weights, tokenizer, positional method and target tokens, the same SmolLM checkpoint scored **1.695154** mean NLL with the last 2K tokens and **1.696318** with all 4K. Lower is better. Across three OpenWebMath documents the extra prefix gave no mean improvement. YaRN did substantially repair *unscaled* 4K extrapolation (7.125009 → 1.696318). Keep it as one candidate for extending a trained RoPE checkpoint. Native 4K training from initialization is a separate option. [Matched four-arm results](../results/context_extension/approved_summary.csv)

### What each length costs on G5

Eager GQA shares keys and values across query heads, then forms a dense score matrix per query head. A causal mask is applied after the matrix multiplication. For one sequence, one layer and H query heads the matrix has H × L² elements. At a fixed number of input positions per update, doubling L halves the number of sequences and roughly doubles the two attention matrix products’ work per update. Other parts of training have different costs. [Eager attention implementation](../harness/week2/model.py) · [T2 geometry](../configs/arms/t2.yaml)

| Length | Sequences per 16,384 input positions | Attention-product work per update vs 2K | One FP32 attention array, 8 query heads | One FP32 KAS logits array, all targets scored |
|---:|---:|---:|---:|---:|
| 1,024 | 16 | 0.5× | 32 MiB | 0.5 GiB |
| **2,048** | **8** | **1×** | **128 MiB** | **1 GiB** |
| 4,096 | 4 | 2× | 512 MiB | 2 GiB |
| 8,192 | 2 | 4× | 2,048 MiB | 4 GiB |

Memory columns are calculated for microbatch one. The attention column counts one four-byte score-sized array per layer per sequence. Precision and tensor lifetimes determine actual allocations. The head column uses 131,072 vocabulary logits × four bytes × scored positions. Model parameters, other activations, gradients, optimizer state, buffers and parallel communication all affect peak memory. For a production layer with a different query-head count, multiply the attention column by H/8 before accounting for sharding. Request the final shape, worst-rank device memory, parallel layout, recomputation policy, packing policy and usable GPU-hours before making a fit or finish-time claim. [KAS head](../harness/week2/kas_v2.py) · [Trainer loss path](../harness/train_week2.py) · [G7 layout assumptions](../../07-infrastructure/docs/parallel_layout_v1.md)

**Measured reference.** At 2K on one A100-80GB with the T2 KAS V2 arm and synthetic inputs, G6 measured 1,378 ms median eager step time and 10.39 GB peak allocation over 20 timed steps. The earlier real-text T2 dry run took 1,221 steps at 2K. Its log contains 7,206,344 scored tokens in 20,004,864 padded input positions (36.0% utilization of input positions). Longer padded batches can make nominal throughput misleading. The dry run 2 job specifies eager attention. Budget potential SDPA savings only after the separate backend comparison is accepted. G1’s shard format accommodates a later 8K or 16K schedule without reshaping the shards. G5’s current evaluation batch limit is 4,096 tokens. A genuine 8K trial also requires an evaluator change. [Dry run 1 metrics](../jobs/evidence/2026-11-15-dryrun1-full/metrics.jsonl) · [A/B measurements and conditions](../results/sdpa_global_attention.md) · [Shard format](../../01-data-a/shards/shard_format_v1.md) · [Evaluation budget](../harness/week2/evaluation.py)

### How to decide whether to move to 4K

1. **Check the data and budget before scheduling.** Request corpus-v1 counts from Data A by language/source at the frozen p1536 pin. Record how many documents reach 2K and 4K, what fraction of tokens lie past each boundary, the finished production shape, the training token target and the allocated time. Use actual document boundaries. Scarce long Indic material calls for appropriate held-out examples before claiming Indic coverage.
2. **Price the change.** Request a short, separate 2K/4K run on the intended hardware and software stack. Use a separate arm with `model.block_size` set to 4,096. At T2, use microbatch one and 8 × 2,048 versus 4 × 4,096 positions per optimizer update. Measure synchronized full training steps after warm-up, peak allocated/reserved memory on the busiest rank, and median/p95 step time. The trainer logs padded `tokens_per_second`. For throughput use `sum(train/scored_tokens) / (sum(perf/step_ms) / 1000)`, and include evaluation and checkpoint overhead in the wall-time projection. For a fixed allocation of devices, let N be the target scored tokens, f the proposed fraction trained at 4K, and R₂ and R₄ the measured aggregate scored tokens per second at each length. Project training seconds as N × [(1−f)/R₂ + f/R₄], then add measured evaluation, checkpoint and restart allowance. Choose f only after this fits the allocated wall time.
3. **Compare training schedules fairly.** Fork one native 2K checkpoint for separate 2K and 4K continuation runs. Use the same weights, router state, optimizer-state policy, document IDs/order and language mix. The stock trainer resumes an identical configuration and shuffles *windows*, whose indices change with length. A validated fork must rebuild both samplers from a common document boundary and pass a same-length control. Match cumulative non-padding targets and the learning-rate schedule at those boundaries. G5 schedules by step, so match scored tokens per update or add and verify a token-indexed schedule. Keep its full-update loss normalization. For a pre-launch 4K-from-scratch comparison, use the same initial weights and training data. For a later RoPE extension, compare the checkpoint’s native convention first. Using YaRN on G5 requires validating the implementation. Retain short examples during continuation. [Sampler](../harness/week2/data.py) · [Trainer](../harness/train_week2.py)
4. **Measure the value of the extra prefix and compare checkpoints.** Within the 4K candidate, score identical final 512 next-token targets with all 4K and with the last 2K, pinning the positional method, document boundary and target shift. Reset positions per independent window, as in G5, and verify alignment of the shared suffix. This suffix comparison needs a dedicated scorer. Estimate uncertainty across documents. Between checkpoints, evaluate both arms on the same held-out distant-fact questions with their supported context. Put the decisive fact more than 2K tokens before the answer. Test English and native-script Indic, verify that both solve the short-distance version, and remove or move the fact as a control. Separately score both checkpoints with the *same 2K windows* on the frozen short-context suite. Report whole-file, English, code, equal-weighted Indic and per-language bits per original UTF-8 byte. [G5 evaluation](../harness/week2/evaluation.py) · [G5 scoring and seed-spread rule](../harness/week2/scoring.py)
5. **Apply the decision rule.** Keep 2K when the longer setting cannot fit or finish the allotted token budget, when the tested 4K phase lacks long same-document examples in the target languages, when paired long-task results are inconclusive, or when short-language quality regresses beyond the agreed tolerance. Approve a 4K phase when the hardware gate passes, the paired 95% lower confidence bound for the improvement in distant-fact accuracy over the 2K control exceeds zero, and the English, code and Indic retention guards hold. For retention, propose three paired seeds (1234–1236). For each short group, require the candidate’s mean BPB increase to be no greater than the larger within-arm seed spread for that group, extending G5’s existing guard to English explicitly. Agree language sample sizes and any stricter tolerances before unblinding. Use a short run to screen feasibility, then fund replication only for a credible candidate. TSAI dispatches any follow-up separately from dry run 2.

**Published precedents.** OpenMoE-34B trained at 2,048. Qwen3-30B-A3B belongs to a GQA MoE family trained at 4,096 before a 32,768-token stage. Both choices are plausible at this parameter scale. In a controlled dense 7B study, moving from 4K to 32K late reduced compute at equal tokens, with different outcomes by task. Borrow the staged comparison and explicit short-task checks. G5’s own corpus and hardware decide the length. [OpenMoE, Appendix B, Table 12](https://arxiv.org/pdf/2402.01739#page=22) · [Qwen3, §3.2](https://arxiv.org/pdf/2505.09388#page=4) · [Xiong et al., §4.4, Table 9](https://aclanthology.org/2024.naacl-long.260.pdf#page=8)

## Week 3 experiment record · 18 November 2026

The following sections document the merged pilot, its historical configuration and its reproducibility checks. The Week 4 recommendation above governs the proposed production window.

### Assignment and architecture boundary

The Week 3 deliverable was a long-context design and an executed small-model test. [PR #616](https://github.com/theschoolofai/era-v5-lab/pull/616) merged on 18 November. The [executed test log](../results/context_extension/TEST_LOG.md) records that result.

The signed Gate 2 decision retains **v0 eager GQA**, Muon and the selected expert configuration; production sequence length and model dimensions remain open. It explicitly does not copy the experimental RoPE/window settings into `config_v1.yaml`. The Week 2 reference harness contains rotary GQA, but that implementation is not itself a signed production context decision. [Signed configuration][S1] | [Decision rationale][S2] | [Owner allocation][S3]

| Item | Signed team baseline | Completed surrogate test |
|---|---|---|
| Checkpoint | Team model/checkpoint selected by TSAI | Original `HuggingFaceTB/SmolLM-135M`, 134,515,008 parameters |
| Attention execution | Eager v0 GQA | Eager and full vocabulary head for the course-corpus run; SDPA only in the separate exploratory pilot |
| Embeddings/tokenizer | Kronecker input; signed tokenizer v0 baseline, with p1536/current approved release to be frozen for a new trial | SmolLM's pretrained embeddings and original tokenizer |
| Position/window | Production choice open | Original RoPE at 2K; trial static YaRN x2 at 4K |
| Weights/training | Team training process | Fixed pretrained weights; no gradient updates |
| Output metric | Use team-approved evaluation accounting for integration | Natural-log token NLL and exp(mean NLL), within one tokenizer |

The surrogate isolates a positional retrofit on an available pretrained checkpoint. Its results and throughput are not capstone-model benchmarks. This submission does not change the signed configuration, tokenizer, attention kernel or training job settings.

## Why this checkpoint and method

A context-extension method changes how a model represents token positions when reading beyond its original window. **Successful execution**, **retained prediction quality** and **useful retrieval of distant information** are separate claims. This experiment measures the first two on a small, explicitly bounded sample.

The original SmolLM-135M checkpoint is small enough for a fast, replayable evaluation and has an explicit 2,048-token configured window, RoPE and GQA. Extending it to 4,096 is a genuine trial beyond that configured window. A newer checkpoint already configured for long context would answer a different question. Its small size and English-language data limit transfer to the capstone. [Pinned checkpoint configuration][S4]

| Candidate | Main change | Trade-off and use here |
|---|---|---|
| Linear position interpolation | Divide positions, equivalently all rotary frequencies, by the extension factor | Simple baseline for a later comparison; changes local as well as long-range rotations |
| NTK-aware scaling | Change the rotary frequency schedule through its base | Frequency-sensitive alternative; concrete static/dynamic implementation must be specified |
| YaRN | Blend frequency bands and apply attention scaling | Selected as one testable proposal; adds configurable rotary hyperparameters and implementation checks; no new learned weights |

YaRN provides a published context-extension method and a concrete framework implementation. Its paper motivates the candidate; it does not predict a numerical result for SmolLM-135M. No empirical comparison against PI or NTK-aware scaling is claimed here. A broader method sweep and continued training are follow-up work, not prerequisites for this small-model log. [YaRN method][S5]

### Resolved YaRN settings

| Parameter | Value |
|---|---:|
| Original reference length L | 2,048 |
| Trial input length | 4,096 |
| Factor s | 2.0 |
| Rotary base theta | 10,000 |
| beta_fast / beta_slow | 32 / 1 |
| Resolved rotary amplitude factor | 1.0693147181 |
| Frequency-ramp bounds for this head dimension | 8 and 21 |

The experiment sets `max_position_embeddings=4096` and `rope_parameters` on the YaRN configuration while preserving `original_max_position_embeddings=2048`. The unscaled model retains its original rotary frequencies and accepts a complete 4K input. The earlier exploratory notebook includes an independent frequency calculation that matches its loaded YaRN rotary buffer. The course-corpus runner uses the same pinned framework and explicit rotary settings. The stored framework amplitude matches the expected scalar, and the notebook adds no manual attention-temperature multiplier. This check validates configuration, not an end-to-end count of internal multiplications.

## Controlled evaluation

The primary class-aligned self-run check uses **G5 heldout_v1**, a frozen, benchmark-free corpus whose published purpose explicitly allows any team to use it for a perplexity check. The Lab 3 briefing generally directs other testing to Team 3's held-out text; G5's published evaluation corpus supplies the specific documented route used here. No protected benchmark questions or solutions are read. This is a personal small-model run permitted by the job-request guide, not a TSAI-returned run or a claim of lead acceptance. [G5 corpus purpose][S7] | [Lab 3 briefing][S8] | [Job-request scope][S9]

The entire input file has SHA-256 `21c6e4b2df8dae9d69e254f4a00fc934d2a3c70f0675065a03724299d94312bb` and 1,059 document records. Before scoring, scan the C4, OpenWebMath and peS2o prefixes in file order and select the first three **complete documents** with at least 4,096 SmolLM tokens. No concatenation, repetition or outcome-based selection is used. All three qualifying documents are OpenWebMath: `owm-test-000008`, `owm-test-000029` and `owm-test-000036`, with 4,665, 7,537 and 5,872 available tokens. Retain each document's first 4,096 IDs.

This is a small math-domain subset, not the full corpus, the full group-balanced BPB evaluation, or a multilingual validation. Pretraining overlap with the surrogate checkpoint is unknown. The exact document/text/token hashes are recorded in the manifest, and the corpus Git revision is listed in the reproducibility table; raw text is already supplied by the existing lab corpus and is not duplicated into this PR.

The short input is the exact **last 2,048-token suffix** of the 4K input. Every condition predicts the identical final **512 target IDs per article**, or 1,536 targets per condition. Positions restart at zero for each independent input. This avoids comparing losses on different text when changing length.

![Matched 2K and 4K inputs](context_extension_figures/matched_targets.png)

Figure 1. Both windows contain the same scored suffix. The axis identifies token locations in the fixed 4K document; model positions reset independently within each window.

| Run | Input | Rotary setting | Question |
|---|---:|---|---|
| A | 2,048 | Original | Native-window reference |
| B | 4,096 | Original | Unscaled extrapolation |
| C | 4,096 | YaRN x2 | Candidate extension |
| D | 2,048 | Same YaRN configuration as C | Short-context sensitivity |

Run in evaluation and inference modes with no KV cache, no dropout, no parameter updates, batch size three and BF16. Inputs are coherent single-article windows without cross-document packing. For the course-corpus check, compute the full vocabulary logits, select rows `[-513:-1]` and align them with the last 512 input IDs. This avoids making a vocabulary-head speed change in the class-aligned check. All earlier input tokens still pass through every transformer layer and causal attention.

For target losses l_i = -log p(target_i | preceding tokens), report NLL = mean(l_i) and perplexity = exp(NLL). Each document contributes exactly 512 targets, so the mean of the three document NLLs equals the token-weighted mean. Perplexity is computed after aggregation, not by averaging document perplexities.

Correctness screens require identical pretrained parameters, identical scored targets, finite losses, actual 2K/4K inputs, the intended original YaRN reference length and correct next-token alignment. The primary runner uses eager attention for every arm. The separate exploratory notebook records a native-2K BF16 SDPA/eager check with a 0.02-nats/token engineering screen. [Offline course-corpus runner](../harness/context_extension_check.py) | [Primary configuration and checks](../results/context_extension/approved_manifest.json)

## Results and interpretation: frozen G5 corpus subset

| Run | Context | Method | Mean NLL | Perplexity |
|---|---:|---|---:|---:|
| A | 2,048 | Original RoPE | 1.674093 | 5.333954 |
| B | 4,096 | Original RoPE | 7.125009 | 1242.659623 |
| C | 4,096 | YaRN x2 | 1.696318 | 5.453827 |
| D | 2,048 | YaRN x2 | 1.695154 | 5.447483 |

![Class-corpus measured NLL and paired changes](context_extension_figures/approved_quality_results.png)

Figure 2. Eager-attention results on three complete G5 OpenWebMath documents. Every row scores the same 1,536 targets. Changes are descriptive, without a population uncertainty estimate.

| Comparison | Mean NLL change | Interpretation |
|---|---:|---|
| C - B: scaled vs unscaled 4K | -5.428692 | Substantially lower loss with YaRN |
| D - A: static YaRN at 2K | +0.021061 | Higher mean short-window loss on this subset |
| C - A: overall candidate vs native | +0.022225 | Changes length and scaling; 2.25% higher perplexity |
| C - D: additional prefix, fixed YaRN | +0.001164 | Direct length comparison; 0.12% higher perplexity |

**Recommendation:** retain YaRN as a candidate that avoids severe unscaled extrapolation failure, then validate it on the approved team checkpoint before adoption. It does not beat the native-window mean here. C - D holds the positional method fixed and does not show a mean gain from the additional prefix on these three documents. D - A records an observed short-window penalty; it is not a general guarantee of harmless short-context behavior.

All four inputs reached the model at their stated lengths and produced finite losses. Inputs and weights were identical within the intended comparisons. The runner used **eager attention, BF16 and the full vocabulary head**. The exact final script and returned source hash match. [Per-document measurements](../results/context_extension/approved_document_results.csv) | [Run manifest](../results/context_extension/approved_manifest.json) | [Executed log](../results/context_extension/TEST_LOG.md)

## Exploratory WikiText pilot: separate evidence

The earlier reviewed notebook tested three WikiText-103 articles with SDPA/BF16. Its NLLs were A=2.470164, B=6.693264, C=2.482160 and D=2.470355. C - B was -4.211104; C - D was +0.011805. FP32 sensitivity preserved that pilot's broad conclusion, and its original isolated replay reproduced its four BF16 means exactly. These checks concern the WikiText pilot, not an FP32 or isolated rerun of the G5 corpus result. Its external corpus, different backend and different domains are not pooled with the primary measurements. [Pilot notebook](../notebooks/context_extension.ipynb) | [Pilot log](../results/context_extension/PILOT_TEST_LOG.md) | [WikiText source][S6]

Neither small study demonstrates useful distant-fact retrieval, a multilingual outcome, a production training schedule, or a generally safe short-context trade-off.

## Efficiency and reproducibility

The run used an RTX PRO 6000 Blackwell Server Edition with 94.97 GiB visible GPU memory, Python 3.13.11, PyTorch 2.11.0+cu130, CUDA 13.0 and Transformers 5.14.1. The kernel exposed 20 CPU-affinity entries and 160 GiB host RAM; these observations are not a scheduler reservation guarantee. Full versions and configuration are in the manifest.

The primary course-corpus runner keeps eager attention and the full output head. The separate WikiText pilot used BF16, `torch.inference_mode`, SDPA, batched coherent inputs, a single pinned test-data load and the reduced vocabulary-head projection. A direct short-input probe against full logits gave zero loss difference. The native-2K SDPA/eager check's maximum per-document mean-NLL difference was 0.002087. This does not certify backend equivalence for every model, precision or 4K condition.

A separate **WikiText-pilot-only** batch-size sweep repeated the same three fixtures at batch sizes 3, 6, 12, 24 and 48. Batch six was fastest tested: **400,418 input tokens/second**, with **2,475.6 MiB** peak PyTorch allocated GPU memory while both BF16 models were resident. Repeated fixtures do not increase the quality sample count. Larger batches were slower here, so GPU capacity alone is not a reason to increase batch size.

No team-code speed change is proposed here. Any later speed-only integration must satisfy the Lab 3 requirement for byte-identical output. The pilot's BF16 SDPA/eager tolerance check is not that proof.

![Batch-size throughput and memory](context_extension_figures/batch_scaling.png)

Figure 3. Exploratory WikiText-pilot measurements: warmed-up GPU forward plus tail loss, median of three CUDA-event repetitions. Loading, data preparation, warmup and host-side finite checks are excluded. Allocated memory is not device-wide or reserved memory. This is neither training throughput nor cached generation throughput, and it is not a timing of the eager G5 corpus run.

### Reproduce the course-corpus check from the repository root

Use a compatible recent CUDA driver and the recorded Python 3.13 / PyTorch 2.11.0 / Transformers 5.14.1 environment. Before offline execution, stage the exact public SmolLM revision `1d461723eec654e65efdc40cf49301c89c0c92f4` in the Hugging Face cache. The runner uses `local_files_only=True`, accepts only the canonical model ID, reads the existing frozen G5 file and makes no network requests. A missing cache or mismatched input hash fails explicitly. It writes only `./out/`.

```bash
python teams/05-architecture/tests/verify_context_extension.py
python teams/05-architecture/harness/context_extension_check.py --heldout teams/05-architecture/heldout_v1/data/heldout_v1.jsonl
python teams/05-architecture/tests/verify_context_extension.py
```

The offline verifier checks both saved studies, numerical conclusions, the executed primary-runner hash, notebook structure and file integrity without executing a GPU job. Fresh outputs are in `out/`; the committed measurement snapshot is not overwritten. Do not compare these subset token perplexities with published full-corpus BPB numbers or across tokenizers.

| Reproducibility identifier | Value |
|---|---|
| Model/tokenizer revision | `1d461723eec654e65efdc40cf49301c89c0c92f4` |
| Primary runner SHA-256 | `41a8b4b0870d465071fa6cb09e332e5e214e99d8790f46934fe40a3e8e9f3e00` |
| Frozen G5 file SHA-256 | `21c6e4b2df8dae9d69e254f4a00fc934d2a3c70f0675065a03724299d94312bb` |
| Corpus repository revision | `87323846b325997ea2c21737c7816f950b8f1a26` |
| Primary execution | 2026-11-17T21:50:08.066045+00:00 (UTC), 18 November IST |
| Document IDs and hashes | [approved_manifest.json](../results/context_extension/approved_manifest.json) |

The exploratory pilot is supplied as a standard Jupyter notebook with a labelled snapshot of the original results and embedded charts. The notebook now explains the assignment, alternatives, matched comparisons and scoring concepts with comments and visuals. Its 16 code cells were executed in a fresh CUDA kernel after the readability revision; saved outputs include tables, charts and the run manifest. The [rerun receipt](../results/context_extension/readability_rerun.json) binds the executed notebook and code hashes to its input hashes and results. Maximum mean-NLL drift from the historical BF16 pilot was 0 nats/token. Follow the [notebook run instructions](../notebooks/context_extension_README.md) to run it on CUDA; new outputs go to `out/context_extension_pilot/`. Original executed source and replay hashes remain in `source_provenance.json`, and the exact original source is preserved in Git commit `9539c7be6263b3fe429a5096dac8c77ed1428f63`. The complete supporting-evidence download retains both studies' raw losses, exact selected IDs and historical execution logs. Required design/result claims can be inspected from the PR without a private ChatGPT link.

## Week 3 proposal for a team-checkpoint follow-up

1. **Freeze the approved baseline.** Record the team checkpoint, tokenizer, rotary convention, head dimension, original window, eager GQA implementation, precision and evaluation data. The Gate 2 configuration names tokenizer v0, while the Week 3 dry run uses the frozen p1536 release. Pin the p1536 artifact or any subsequently approved team release and its hashes for a new trial; do not silently carry over the older tokenizer reference. Confirm whether the approved positional path actually uses RoPE; YaRN is not a drop-in change for learned absolute embeddings.
2. **Implement only the positional proposal.** Preserve eager GQA, masks, architecture and weights for the first comparison. Use the checkpoint's own rotary convention; the Week 2 harness uses adjacent pairs, whereas a library model may pair dimensions differently. Preserve the approved native reference length. Do not copy SmolLM's head dimensions or its 2K reference into a different checkpoint.
3. **Verify data and alignment.** Ensure a complete extended sequence reaches attention; inspect token counts, positions, document boundaries and loss masks. Use the same scoring text and target-byte accounting for comparisons within the team tokenizer. Recheck full-head/tail alignment and rotary calculations against an independent reference.
4. **Evaluate both lengths.** Repeat A/B/C/D on a larger held-out sample covering the team's language and domain groups. Add distant-fact retrieval at several positions only after verifying that the base model solves the short version. Record per-group and short-context regressions and agree acceptance criteria with TSAI.
5. **Propose training only if needed.** Any continued-training experiment needs an agreed target length, token budget, coherent long-document source, retained short-document mixture, optimizer settings and memory estimate. This inference-only experiment supplies no optimal mixture, learning rate or training budget. Dispatch a production job only through the team-approved process after TSAI's context decision.

The candidate stage is a modest 2x extension, not a proposal to jump directly to the context lengths advertised by larger open-weight models. The broader model-report research is supporting motivation; its budgets and claims are not transferred into this result.

## Sources

[S1]: https://github.com/theschoolofai/era-v5-lab/blob/aec75afc93070f56d891f438ccdd9a791b2c18bd/teams/05-architecture/configs/config_v1.yaml
[S2]: https://github.com/theschoolofai/era-v5-lab/blob/aec75afc93070f56d891f438ccdd9a791b2c18bd/teams/05-architecture/decisions.md
[S3]: https://axiom.theschoolofai.in/courses/cmq97i5kn032208o8xu5dab4q?chat=cmvkupxl427gl09l7rv9m4fwy&msg=cmwg7xjek0ji109o5b03mxagt
[S4]: https://huggingface.co/HuggingFaceTB/SmolLM-135M/blob/1d461723eec654e65efdc40cf49301c89c0c92f4/config.json
[S5]: https://arxiv.org/abs/2309.00071
[S6]: https://huggingface.co/datasets/Salesforce/wikitext/tree/b08601e04326c79dfdd32d625aee71d232d685c3
[S7]: https://github.com/theschoolofai/era-v5-lab/blob/87323846b325997ea2c21737c7816f950b8f1a26/teams/05-architecture/heldout_v1/README.md
[S8]: https://axiom.theschoolofai.in/widgets/cap_lab3_briefing.html
[S9]: https://github.com/theschoolofai/era-v5-lab/blob/87323846b325997ea2c21737c7816f950b8f1a26/JOB_REQUEST.md

- Signed Gate 2 configuration and decisions, 14 November 2026: [configuration][S1] and [decisions][S2].
- G5 Week 3 owner allocation: [assignment][S3].
- SmolLM-135M: [pinned configuration][S4].
- Peng et al., *YaRN: Efficient Context Window Extension of Large Language Models*: [technical paper][S5].
- WikiText-103: [pinned dataset][S6]. Exact public test split and selection rule are recorded above. WikiText derives from Wikipedia; retain source attribution and applicable CC BY-SA terms for redistributed fixtures.

- Primary class-corpus authority: [G5 heldout purpose, frozen bytes and licences][S7]; [Lab 3 rules][S8]; [conditional job-request guide][S9]. The primary selected text is OpenWebMath via Proof-Pile-2, under the corpus's ODC-By attribution. Existing source tracking stays with the lab corpus.
