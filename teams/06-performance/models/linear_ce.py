"""Memory-bounded linear cross-entropy with a recomputing custom backward.

The vocabulary dimension is processed in tiles. Forward keeps only per-token log-sum-exp and
target logits; backward recomputes one tile at a time, so an [tokens, vocabulary] logits tensor
is never retained. This is a pure PyTorch implementation intended as a portable fused-op baseline,
not a single custom CUDA kernel.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


class _LinearCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, targets, vocab_chunk: int, ignore_index: int):
        if x.ndim != 2 or weight.ndim != 2:
            raise ValueError("linear cross-entropy expects x [tokens, hidden] and weight [vocab, hidden]")
        if x.shape[1] != weight.shape[1]:
            raise ValueError(f"hidden sizes differ: {x.shape[1]} vs {weight.shape[1]}")
        if vocab_chunk <= 0:
            raise ValueError("vocab_chunk must be positive")
        targets = targets.reshape(-1).long()
        if targets.numel() != x.shape[0]:
            raise ValueError("target count must match the number of input rows")

        x32, w32 = x.float(), weight.float()
        n, vocab = x.shape[0], weight.shape[0]
        log_z = torch.full((n,), -torch.inf, dtype=torch.float32, device=x.device)
        target_logits = torch.zeros((n,), dtype=torch.float32, device=x.device)
        rows = torch.arange(n, device=x.device)
        valid = targets != ignore_index
        for start in range(0, vocab, vocab_chunk):
            end = min(start + vocab_chunk, vocab)
            logits = F.linear(x32, w32[start:end], None if bias is None else bias[start:end].float())
            log_z = torch.logaddexp(log_z, torch.logsumexp(logits, dim=-1))
            in_tile = valid & (targets >= start) & (targets < end)
            local = (targets - start).clamp(min=0, max=end - start - 1)
            selected = logits[rows, local]
            target_logits = torch.where(in_tile, selected, target_logits)

        count = valid.sum().clamp_min(1)
        losses = torch.where(valid, log_z - target_logits, torch.zeros_like(log_z))
        loss = losses.sum() / count
        saved_bias = x.new_empty((0,)) if bias is None else bias
        ctx.save_for_backward(x, weight, targets, log_z, valid, saved_bias, count)
        ctx.bias_is_none = bias is None
        ctx.vocab_chunk = vocab_chunk
        ctx.input_shape = x.shape
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, targets, log_z, valid, saved_bias, count = ctx.saved_tensors
        x32, w32 = x.float(), weight.float()
        n, vocab = x.shape[0], weight.shape[0]
        grad_x = torch.zeros_like(x32)
        grad_weight = torch.empty_like(w32)
        grad_bias = None if ctx.bias_is_none else torch.empty((vocab,), dtype=torch.float32, device=x.device)
        rows = torch.arange(n, device=x.device)
        scale = grad_output.float() / count

        for start in range(0, vocab, ctx.vocab_chunk):
            end = min(start + ctx.vocab_chunk, vocab)
            w_tile = w32[start:end]
            logits = F.linear(x32, w_tile)
            if not ctx.bias_is_none:
                logits = logits + saved_bias[start:end].float()
            probs = torch.exp(logits - log_z[:, None])
            probs.mul_(valid[:, None])
            in_tile = valid & (targets >= start) & (targets < end)
            local = (targets - start).clamp(min=0, max=end - start - 1)
            probs[rows, local] -= in_tile.to(probs.dtype)
            probs.mul_(scale)
            grad_x.add_(probs @ w_tile)
            grad_weight[start:end] = probs.T @ x32
            if grad_bias is not None:
                grad_bias[start:end] = probs.sum(dim=0)

        return grad_x.reshape(ctx.input_shape).to(x.dtype), grad_weight.to(weight.dtype), \
            None if grad_bias is None else grad_bias.to(saved_bias.dtype), None, None, None


def linear_cross_entropy(x, weight, targets, *, bias=None, vocab_chunk=4096, ignore_index=-100):
    """Mean CE for a bias-free or biased linear head, without materializing full logits.

    ``x`` is [tokens, hidden], ``weight`` is [vocabulary, hidden], and ``targets`` is [tokens].
    Ignored targets do not contribute to either the sum or divisor. All-ignored targets return
    zero with zero gradients, matching the existing chunked-loss behavior.
    """
    return _LinearCrossEntropy.apply(x, weight, bias, targets, int(vocab_chunk), int(ignore_index))


class _CachedGradientLinearCrossEntropy(torch.autograd.Function):
    """Token-chunked CE that computes the linear-head gradients while logits are resident.

    This moves the head backward work into forward. The saved state is the resulting input/weight
    gradients (plus bias gradient when needed), rather than inputs and logits for recomputation.
    It is a memory/performance trade-off: the weight-gradient buffer is the size of the head.
    """

    @staticmethod
    def forward(ctx, x, weight, bias, targets, token_chunk: int, ignore_index: int):
        if x.ndim != 2 or weight.ndim != 2:
            raise ValueError("linear cross-entropy expects x [tokens, hidden] and weight [vocab, hidden]")
        if x.shape[1] != weight.shape[1]:
            raise ValueError(f"hidden sizes differ: {x.shape[1]} vs {weight.shape[1]}")
        if token_chunk <= 0:
            raise ValueError("token_chunk must be positive")
        targets = targets.reshape(-1).long()
        if targets.numel() != x.shape[0]:
            raise ValueError("target count must match the number of input rows")

        x32, w32 = x.float(), weight.float()
        y = targets
        n, vocab = x.shape[0], weight.shape[0]
        valid = y != ignore_index
        count = valid.sum().clamp_min(1)
        grad_x = torch.zeros_like(x32)
        grad_weight = torch.zeros_like(w32)
        grad_bias = None if bias is None else torch.zeros_like(bias, dtype=torch.float32)
        loss_sum = torch.zeros((), dtype=torch.float32, device=x.device)
        rows = torch.arange(min(token_chunk, n), device=x.device)

        for start in range(0, n, token_chunk):
            end = min(start + token_chunk, n)
            xc, yc = x32[start:end], y[start:end]
            keep = valid[start:end]
            logits = F.linear(xc, w32, None if bias is None else bias.float())
            log_z = torch.logsumexp(logits, dim=-1)
            safe_y = yc.clamp(min=0, max=vocab - 1)
            target_logit = logits[torch.arange(end - start, device=x.device), safe_y]
            loss_sum += torch.where(keep, log_z - target_logit, 0.0).sum()

            probabilities = torch.softmax(logits, dim=-1)
            probabilities.mul_(keep[:, None])
            probabilities[torch.arange(end - start, device=x.device), safe_y] -= keep.to(probabilities.dtype)
            probabilities.div_(count)
            grad_x[start:end] = probabilities @ w32
            grad_weight.add_(probabilities.T @ xc)
            if grad_bias is not None:
                grad_bias.add_(probabilities.sum(dim=0))

        saved_bias_grad = x.new_empty((0,)) if grad_bias is None else grad_bias
        ctx.save_for_backward(grad_x, grad_weight, saved_bias_grad)
        ctx.bias_is_none = bias is None
        ctx.x_dtype, ctx.weight_dtype = x.dtype, weight.dtype
        ctx.bias_dtype = None if bias is None else bias.dtype
        return loss_sum / count

    @staticmethod
    def backward(ctx, grad_output):
        grad_x, grad_weight, grad_bias = ctx.saved_tensors
        scale = grad_output.float()
        return (grad_x.mul(scale).to(ctx.x_dtype), grad_weight.mul(scale).to(ctx.weight_dtype),
                None if ctx.bias_is_none else grad_bias.mul(scale).to(ctx.bias_dtype),
                None, None, None)


def linear_cross_entropy_cached(x, weight, targets, *, bias=None, token_chunk=16, ignore_index=-100):
    """Token-chunked linear CE that caches head gradients to avoid recomputing logits in backward.

    Unlike :func:`linear_cross_entropy`, this computes ``dh`` and ``dW`` in forward and saves them
    for autograd's backward. ``token_chunk`` bounds the temporary ``[token_chunk, vocab]`` logits;
    the cached ``dW`` is a full-size head-gradient buffer, so benchmark its memory cost as well as time.
    """
    return _CachedGradientLinearCrossEntropy.apply(
        x, weight, bias, targets, int(token_chunk), int(ignore_index)
    )
