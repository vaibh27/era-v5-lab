# SFT smoke probe: what it shows

`train_probe.py` trains a tiny random causal transformer on CPU with code-generated integer rows shaped like chat-format v0 (`input_ids` + per-token `loss_mask`).

**What it shows** (`test_train_probe.py`):
- Forward and backward run, and every parameter gets a finite gradient.
- Labels follow the causal shift: logit t is trained on token t+1 only where `loss_mask[t+1]` is 1. Rows with no shifted target are rejected, including all-masked rows and rows truncated past the assistant.
- Masked tool results and context never become targets, and zeroing one mask marker removes exactly that target.
- Padding comes from row lengths. With pad id equal to EOS id, a supervised EOS is still trained.
- Loss is the sum over valid targets divided by the global count. Uneven accumulation matches the one-shot gradient.
- 30 AdamW updates bring the fixed-batch loss down, and the FP32 weights change.

**What it does not show:** anything about OLMoE, G5 or the 131K tokenizer. Not covered: real-tokenizer EOS handling, Indic ZWJ/ZWNJ, CUDA BF16 autocast, admitted data, dev NLL, the 1e-5 learning rate, packing. The 3e-3 learning rate suits only the toy model. Steps 1 and 3–5 of G10_TRAINING.md §3 remain open.
