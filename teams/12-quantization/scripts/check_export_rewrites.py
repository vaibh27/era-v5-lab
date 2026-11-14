"""Numeric check of the exact export rewrites in docs/unsupported_parts.md, part 2.

9.4: reordering q/k rows to split halves gives the same attention scores under split-half RoPE
     (vLLM / Hugging Face) as G5's adjacent-pair RoPE, and llama.cpp's converter permute restores
     the original rows.
9.6: S SwiGLU shared experts of width h equal one SwiGLU of width S*h (stacked gate/up rows,
     concatenated down columns).

Random small matrices, CPU, numpy only:  python check_export_rewrites.py
Expected: max diffs near 1e-13 (float64 rounding), and True for the permute.
"""
import numpy as np
rng = np.random.default_rng(0)
d, nh, T, w, theta = 16, 2, 7, 32, 10000.0
inv = theta ** (-np.arange(0, d, 2) / d); ang = np.arange(T)[:, None] * inv[None, :]
def rope_adjacent(x):            # G5 model.py: pairs (2i, 2i+1)
    a, b = x[..., 0::2], x[..., 1::2]
    return np.stack((a*np.cos(ang) - b*np.sin(ang), a*np.sin(ang) + b*np.cos(ang)), -1).reshape(x.shape)
def rope_neox(x):                # HF / vLLM is_neox_style: pairs (i, i+d/2)
    a, b = x[..., :d//2], x[..., d//2:]
    return np.concatenate((a*np.cos(ang) - b*np.sin(ang), a*np.sin(ang) + b*np.cos(ang)), -1)
def to_halves(W):                # export rewrite 9.4, per head on output rows
    return W.reshape(nh, d, -1)[:, np.r_[0:d:2, 1:d:2], :].reshape(W.shape)
def llama_permute(W):            # llama.cpp conversion/llama.py:176-180
    return W.reshape(nh, 2, d//2, -1).swapaxes(1, 2).reshape(W.shape)
X = rng.standard_normal((T, w)); Wq = rng.standard_normal((nh*d, w)); Wk = rng.standard_normal((nh*d, w))
heads = lambda W: (X @ W.T).reshape(T, nh, d).transpose(1, 0, 2)
ref = np.einsum('htd,hsd->hts', rope_adjacent(heads(Wq)), rope_adjacent(heads(Wk)))
hf  = np.einsum('htd,hsd->hts', rope_neox(heads(to_halves(Wq))), rope_neox(heads(to_halves(Wk))))
print("9.4 vLLM/HF scores match G5:", np.abs(ref - hf).max())
print("9.4 llama.cpp permute restores G5 rows:", np.array_equal(llama_permute(to_halves(Wq)), Wq))
silu = lambda z: z / (1 + np.exp(-z)); h, S = 5, 3
G = [rng.standard_normal((h, w)) for _ in range(S)]; U = [rng.standard_normal((h, w)) for _ in range(S)]; D = [rng.standard_normal((w, h)) for _ in range(S)]
sep = sum(D[s] @ (silu(G[s] @ X.T) * (U[s] @ X.T)) for s in range(S))
merged = np.hstack(D) @ (silu(np.vstack(G) @ X.T) * (np.vstack(U) @ X.T))
print("9.6 merged shared expert max diff:", np.abs(sep - merged).max())
