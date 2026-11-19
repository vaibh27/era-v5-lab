"""Standalone Triton Newton-Schulz and Muon update kernels.

This module is intentionally not wired into models.optim. It exposes:

  newton_schulz5(G, steps=5, eps=1e-7) -> approximately orthogonalised G
  muon_update_(p, grad, buf, lr, ...) -> in-place Muon update of p and buf

The Newton-Schulz path uses three Triton matmul kernels per iteration:
A = X X^T, B = beta A + gamma A A, and X = alpha X + B X. The public
functions fall back to a torch reference on CPU or when Triton is unavailable.

No optimizer class, config switch, state_dict integration, or AdamW path is
provided here. Those are integration decisions for models/optim.py later.
"""
from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None

NS_COEFFS = (3.4445, -4.7750, 2.0315)
DTYPES = (torch.float32, torch.bfloat16, torch.float16)


def _native_bf16() -> bool:
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:
        return torch.cuda.get_device_capability()[0] >= 8


def _compute_dtype(G: torch.Tensor) -> torch.dtype:
    if G.is_cuda and _native_bf16():
        return torch.bfloat16
    return torch.float32


def supported(G: torch.Tensor) -> bool:
    return triton is not None and G.is_cuda and G.ndim == 2 and G.dtype in DTYPES


def _ns5_torch(G: torch.Tensor, steps: int, eps: float) -> torch.Tensor:
    a, b, c = NS_COEFFS
    X = G.to(_compute_dtype(G))
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transpose else X


if triton is not None:
    @triton.jit
    def _gram_kernel(X, A, M, N, SX0, SX1, SA0,
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                     PRECISION: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        red = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k in range(0, tl.cdiv(N, BLOCK_K)):
            kk = k * BLOCK_K + red
            a = tl.load(X + rows[:, None] * SX0 + kk[None, :] * SX1,
                        (rows[:, None] < M) & (kk[None, :] < N), 0.0)
            b = tl.load(X + cols[:, None] * SX0 + kk[None, :] * SX1,
                        (cols[:, None] < M) & (kk[None, :] < N), 0.0)
            acc = tl.dot(a, tl.trans(b), acc, input_precision=PRECISION)
        tl.store(A + rows[:, None] * SA0 + cols[None, :], acc,
                 (rows[:, None] < M) & (cols[None, :] < M))

    @triton.jit
    def _square_combine_kernel(A, B, M, SA0, SB0, beta, gamma,
                               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                               PRECISION: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        red = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k in range(0, tl.cdiv(M, BLOCK_K)):
            kk = k * BLOCK_K + red
            a_ik = tl.load(A + rows[:, None] * SA0 + kk[None, :],
                           (rows[:, None] < M) & (kk[None, :] < M), 0.0)
            a_kj = tl.load(A + kk[:, None] * SA0 + cols[None, :],
                           (kk[:, None] < M) & (cols[None, :] < M), 0.0)
            acc = tl.dot(a_ik, a_kj, acc, input_precision=PRECISION)
        a_ij = tl.load(A + rows[:, None] * SA0 + cols[None, :],
                       (rows[:, None] < M) & (cols[None, :] < M), 0.0)
        out = beta * a_ij + gamma * acc
        tl.store(B + rows[:, None] * SB0 + cols[None, :], out,
                 (rows[:, None] < M) & (cols[None, :] < M))

    @triton.jit
    def _update_kernel(B, X, Y, M, N, SB0, SX0, SX1, SY0, SY1, alpha,
                       BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                       PRECISION: tl.constexpr):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rows = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        cols = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        red = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        for k in range(0, tl.cdiv(M, BLOCK_K)):
            kk = k * BLOCK_K + red
            b = tl.load(B + rows[:, None] * SB0 + kk[None, :],
                        (rows[:, None] < M) & (kk[None, :] < M), 0.0)
            x = tl.load(X + kk[:, None] * SX0 + cols[None, :] * SX1,
                        (kk[:, None] < M) & (cols[None, :] < N), 0.0)
            acc = tl.dot(b, x, acc, input_precision=PRECISION)
        x_ij = tl.load(X + rows[:, None] * SX0 + cols[None, :] * SX1,
                       (rows[:, None] < M) & (cols[None, :] < N), 0.0)
        out = alpha * x_ij + acc
        tl.store(Y + rows[:, None] * SY0 + cols[None, :] * SY1, out,
                 (rows[:, None] < M) & (cols[None, :] < N))

    @triton.jit
    def _momentum_kernel(Grad, Buf, G, n, momentum, NESTEROV: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n
        g = tl.load(Grad + idx, mask, 0.0)
        b = tl.load(Buf + idx, mask, 0.0)
        b = b + (1.0 - momentum) * (g - b)
        tl.store(Buf + idx, b, mask)
        out = g + momentum * (b - g) if NESTEROV else b
        tl.store(G + idx, out, mask)

    @triton.jit
    def _param_kernel(P, U, n, lr, lr_scale, weight_decay, HAS_WD: tl.constexpr, BLOCK: tl.constexpr):
        idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        mask = idx < n
        p = tl.load(P + idx, mask, 0.0).to(tl.float32)
        u = tl.load(U + idx, mask, 0.0).to(tl.float32)
        if HAS_WD:
            p = p * (1.0 - lr * weight_decay)
        p = p - lr_scale * u
        tl.store(P + idx, p, mask)


def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if G.ndim != 2:
        raise ValueError(f"newton_schulz5 expects a 2D tensor, got shape {tuple(G.shape)}")
    if not isinstance(steps, int) or isinstance(steps, bool) or steps < 0:
        raise ValueError("steps must be a nonnegative integer")
    if not eps > 0:
        raise ValueError("eps must be positive")
    if not supported(G):
        return _ns5_torch(G, steps, eps)

    a, b, c = NS_COEFFS
    X = G.to(_compute_dtype(G)).contiguous()
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T.contiguous()
    M, N = X.shape
    X = X / (X.norm() + eps)
    A = torch.empty((M, M), device=X.device, dtype=X.dtype)
    B = torch.empty_like(A)
    Y = torch.empty_like(X)
    precision = "ieee" if X.dtype == torch.float32 else "tf32"
    bm, bn, bk = 64, 64, 32
    for _ in range(steps):
        _gram_kernel[(triton.cdiv(M, bm), triton.cdiv(M, bn))](
            X, A, M, N, X.stride(0), X.stride(1), A.stride(0),
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, PRECISION=precision, num_warps=4, num_stages=2,
        )
        _square_combine_kernel[(triton.cdiv(M, bm), triton.cdiv(M, bn))](
            A, B, M, A.stride(0), B.stride(0), b, c,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, PRECISION=precision, num_warps=4, num_stages=2,
        )
        _update_kernel[(triton.cdiv(M, bm), triton.cdiv(N, bn))](
            B, X, Y, M, N, B.stride(0), X.stride(0), X.stride(1), Y.stride(0), Y.stride(1), a,
            BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, PRECISION=precision, num_warps=4, num_stages=2,
        )
        X, Y = Y, X
    return X.T if transpose else X


@torch.no_grad()
def muon_update_(p: torch.Tensor, grad: torch.Tensor, buf: torch.Tensor, lr: float,
                 momentum: float = 0.95, nesterov: bool = True, weight_decay: float = 0.0,
                 ns_steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    if p.ndim != 2:
        raise ValueError(f"muon_update_ expects a 2D parameter, got shape {tuple(p.shape)}")
    if grad.shape != p.shape or buf.shape != p.shape:
        raise ValueError("grad and buf must have the same shape as p")
    if not (p.is_contiguous() and grad.is_contiguous() and buf.is_contiguous()):
        raise ValueError("muon_update_ requires contiguous p, grad and buf")
    if not (0.0 <= momentum < 1.0):
        raise ValueError("momentum must be in [0, 1)")
    if lr < 0:
        raise ValueError("lr must be nonnegative")

    g = torch.empty_like(grad)
    if supported(p):
        block = 1024
        _momentum_kernel[(triton.cdiv(grad.numel(), block),)](
            grad, buf, g, grad.numel(), momentum, NESTEROV=nesterov, BLOCK=block,
            num_warps=4,
        )
        u = newton_schulz5(g, steps=ns_steps, eps=eps).contiguous()
        scale = lr * max(1.0, p.size(0) / p.size(1)) ** 0.5
        _param_kernel[(triton.cdiv(p.numel(), block),)](
            p, u, p.numel(), lr, scale, weight_decay, HAS_WD=weight_decay != 0.0, BLOCK=block,
            num_warps=4,
        )
    else:
        buf.lerp_(grad, 1 - momentum)
        g = grad.lerp(buf, momentum) if nesterov else buf
        u = _ns5_torch(g, ns_steps, eps)
        scale = lr * max(1.0, p.size(0) / p.size(1)) ** 0.5
        if weight_decay:
            p.mul_(1 - lr * weight_decay)
        p.add_(u.to(p.dtype), alpha=-scale)
    return p
