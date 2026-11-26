"""The model adapter the trainer uses: logits for a batch of token ids, and sampling.

Policy works with any causal LM given as `forward(ids, mask) -> logits [B, T, V]`:
    HFPolicy       a Hugging Face causal LM (OLMoE now): forward = model(input_ids, attention_mask).logits
    ModulePolicy   any nn.Module with its own call, e.g. Team 5's LanguageModel: forward = model(ids, valid)

Sampling generates the k completions of ONE prompt together, so every row has the same prompt length and no left
padding is needed (Team 5's attention has no padding mask). Tokens after an EOS are masked out, never trained on.
"""
from __future__ import annotations

import contextlib
from typing import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class Policy:
    def __init__(self, model: nn.Module, forward: Callable, eos_id: int | None = None, device: str = "cpu",
                 hidden: Callable | None = None, head: Callable | None = None):
        """hidden(model, ids, mask) -> final hidden states [B, T, d] and head(model, h) -> logits [..., V], given together
        only when forward(model, ids, mask) == head(model, hidden(model, ids, mask)) and the head maps each position on
        its own (Team 5's LanguageModel). Then the update runs the head on the completion positions only."""
        if (hidden is None) != (head is None):
            raise ValueError("give hidden and head together, or neither")
        self.model = model.to(device)
        self._forward = forward
        self._hidden, self._head = hidden, head
        # memory mode (train.head_checkpoint, rl_job_v1 --head-checkpoint): under autograd run the head in checkpointed
        # chunks of HEAD_CHUNK tokens. Off by default: one head call is faster when the row fits (job RLM2).
        self.head_checkpoint = False
        self.eos_id = eos_id
        self.device = device
        self.amp_dtype: torch.dtype | None = None  # set by use_fp32_master(); None = no autocast

    def autocast(self):
        """bf16 autocast when the trainable weights are fp32 masters inside a bf16 model, else a no-op."""
        if self.amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=torch.device(self.device).type, dtype=self.amp_dtype)

    def logits(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            mask = torch.ones_like(ids, dtype=torch.bool)
        with self.autocast():
            return self._forward(self.model, ids, mask)

    def completion_logits(self, seq: torch.Tensor, prompt_len: int) -> torch.Tensor:
        """Logits [B, T - prompt_len, V] at the positions that predict seq[:, prompt_len:]. With hidden/head the output
        head runs on those positions only (the prompt's share of the [T, V] logits is never built); otherwise the full
        forward is sliced."""
        mask = torch.ones_like(seq, dtype=torch.bool)
        with self.autocast():
            if self._hidden is None:
                return self._forward(self.model, seq, mask)[:, prompt_len - 1:-1, :]
            return self._head(self.model, self._hidden(self.model, seq, mask)[:, prompt_len - 1:-1, :])

    @torch.no_grad()
    def sample(self, prompt: list[int], k: int, max_new: int, temperature: float = 1.0,
               generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """k completions of one prompt. Returns (sequences [k, P+max_new], completion_mask [k, max_new])."""
        was_training = self.model.training
        self.model.eval()
        seq = torch.tensor([prompt] * k, dtype=torch.long, device=self.device)
        alive = torch.ones(k, dtype=torch.bool, device=self.device)
        cmask = torch.zeros(k, max_new, dtype=torch.bool, device=self.device)
        for t in range(max_new):
            logits = self.logits(seq)[:, -1, :].float()
            if temperature <= 0:
                nxt = logits.argmax(-1)
            else:
                probs = torch.softmax(logits / temperature, -1)
                nxt = torch.multinomial(probs, 1, generator=generator).squeeze(-1)
            cmask[:, t] = alive
            seq = torch.cat([seq, nxt[:, None]], 1)
            if self.eos_id is not None:
                alive = alive & (nxt != self.eos_id)
                if not alive.any():
                    seq = torch.cat([seq, torch.full((k, max_new - t - 1), self.eos_id, device=self.device)], 1)
                    break
        self.model.train(was_training)
        return seq, cmask


def token_logprobs(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """log p(target) from logits [..., V] fp32 and targets [...]."""
    return torch.log_softmax(logits, -1).gather(-1, targets[..., None]).squeeze(-1)


HEAD_CHUNK = 512  # completion tokens per head call in the chunked paths (all rows flattened)
NOGRAD_CHUNKS = True  # no_grad scoring in chunks of HEAD_CHUNK (set from the RLM2 T4 timing; see completion_logprobs)


def _head_logprobs(policy: Policy, h: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    with policy.autocast():
        logits = policy._head(policy.model, h)
    return token_logprobs(logits.float(), targets)


def completion_logprobs(policy: Policy, seq: torch.Tensor, prompt_len: int) -> torch.Tensor:
    """Log-probability of each completion token under the policy: [k, T_completion].

    With hidden/head (Team 5's policy) the trunk runs once over the whole sequence and the head only on the completion
    positions. Under autograd that is one head call, unless policy.head_checkpoint is on: then the head + log-prob run in
    chunks of HEAD_CHUNK tokens, each checkpointed, so the head's [n, V] tensors (Team 5's KAS head builds about four)
    exist for one chunk at a time and are rebuilt in backward (one extra head forward per chunk; job RLM: -41% per-row
    peak on a T4). NOGRAD_CHUNKS decides the no_grad path (old_lp, the gate, kl_to_start), where chunks cost no extra
    forward. The head maps each position on its own, so chunking changes no value beyond float rounding."""
    if policy._hidden is None:
        return token_logprobs(policy.completion_logits(seq, prompt_len).float(), seq[:, prompt_len:])
    mask = torch.ones_like(seq, dtype=torch.bool)
    with policy.autocast():
        h = policy._hidden(policy.model, seq, mask)[:, prompt_len - 1:-1, :]
    targets = seq[:, prompt_len:]
    grad = torch.is_grad_enabled() and h.requires_grad
    if (grad and not policy.head_checkpoint) or (not grad and not NOGRAD_CHUNKS):
        return _head_logprobs(policy, h, targets)
    hf, tf = h.reshape(-1, h.shape[-1]), targets.reshape(-1)
    parts = []
    for i in range(0, hf.shape[0], HEAD_CHUNK):
        if grad:
            parts.append(checkpoint(_head_logprobs, policy, hf[i:i + HEAD_CHUNK], tf[i:i + HEAD_CHUNK],
                                    use_reentrant=False))
        else:
            parts.append(_head_logprobs(policy, hf[i:i + HEAD_CHUNK], tf[i:i + HEAD_CHUNK]))
    return torch.cat(parts).reshape(targets.shape)


class HFPolicy(Policy):
    """Hugging Face causal LM. Samples with model.generate (KV cache), which matters for long completions."""

    @torch.no_grad()
    def sample(self, prompt, k, max_new, temperature=1.0, generator=None):
        if generator is not None:  # generate() has no generator argument; seed torch from ours for reproducibility
            torch.manual_seed(int(torch.randint(0, 2**31 - 1, (1,), generator=generator)))
        was_training = self.model.training
        self.model.eval()
        ids = torch.tensor([prompt] * k, dtype=torch.long, device=self.device)
        pad = self.eos_id if self.eos_id is not None else 0
        with self.autocast():
            out = self.model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new,
                                      min_new_tokens=max_new if self.eos_id is None else 0,
                                      do_sample=temperature > 0,
                                      temperature=temperature if temperature > 0 else None,
                                      top_k=0, top_p=1.0, eos_token_id=self.eos_id, pad_token_id=pad)
        self.model.train(was_training)
        comp = out[:, len(prompt):]
        if comp.shape[1] < max_new:
            comp = torch.cat([comp, torch.full((k, max_new - comp.shape[1]), pad, device=self.device)], 1)
        cmask = torch.ones_like(comp, dtype=torch.bool)
        if self.eos_id is not None:  # keep the EOS token itself, drop everything after it
            after_eos = torch.cumsum((comp == self.eos_id).long(), 1) - (comp == self.eos_id).long() > 0
            cmask = ~after_eos
        return torch.cat([ids, comp], 1), cmask


def use_fp32_master(policy: Policy) -> int:
    """Keep the trainable weights in fp32 and run the model under bf16 autocast.

    AdamW on bf16 weights loses small updates: bf16 keeps about 3 significant digits, so at lr 1e-6 a weight near
    0.02 usually rounds back to itself (measured: one step changed 2% of bf16 weights vs 100% of fp32 weights).
    With fp32 masters the optimiser updates fp32 tensors, and autocast casts them to bf16 for each matmul.
    Frozen weights (router, experts) stay bf16, so memory grows only by the trainable part. Returns the number
    of parameters converted (0 for an fp32 model, where nothing changes).
    """
    n = 0
    for p in policy.model.parameters():
        if p.requires_grad and p.dtype in (torch.bfloat16, torch.float16):
            p.data = p.data.float()
            n += p.numel()
    if n:
        policy.amp_dtype = torch.bfloat16
    return n


def hf_policy(model: nn.Module, eos_id: int | None = None, device: str = "cpu") -> Policy:
    return HFPolicy(model, lambda m, ids, mask: m(input_ids=ids, attention_mask=mask.long()).logits, eos_id, device)


def module_policy(model: nn.Module, eos_id: int | None = None, device: str = "cpu", split_head: bool = False) -> Policy:
    """For models called as model(ids, valid) -> logits, like Team 5's LanguageModel. split_head=True only for a model
    whose forward is head(hidden(ids, valid)) with a per-position head (Team 5's LanguageModel): the update then runs
    the head on the completion positions only."""
    if not split_head:
        return Policy(model, lambda m, ids, mask: m(ids, mask), eos_id, device)
    return Policy(model, lambda m, ids, mask: m(ids, mask), eos_id, device,
                  hidden=lambda m, ids, mask: m.hidden(ids, mask), head=lambda m, h: m.head(h))
