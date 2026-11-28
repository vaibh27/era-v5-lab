#!/usr/bin/env python3
"""Parameter count and size grid for the G5 model (input to the model-size proposal; it decides nothing).

The count is closed form and follows harness/week2/model.py and kas_v2.py layer by layer:

  per block   2 RMSNorm weights                         2w
              GQA q, o                                  2w^2
              GQA k, v (head dim w / n_head)            2 w n_kv (w / n_head)
              router                                    w E
              routed + shared SwiGLU experts            (E + S) 3 w h
  input       KAS input projection (head_mode kas_v2)   2048 w
              KE2 projection (dense head)               256 P w     (P = ke_max_pos, 32)
              plain table (tied-embedding convention)   0 extra when tied, else V w
  output      final RMSNorm                             w
              KAS head (kas_v2)                         2048 w + 33 + V + 64 V + 64 w
              dense untied head                         V w
              MTP head (mtp: true)                      V w

Active parameters per token count the same terms with only (top_k + S) experts per block. The input
projection, the head and the router are dense, so they are active. The KAS head computes every logit,
so all of it is active.

    python3 teams/05-architecture/harness/sizing.py --check          # formula vs the real model at T1/T2/T3/dry run 2
    python3 teams/05-architecture/harness/sizing.py --corpus-tokens 6e11 --md out.md
"""
import argparse
import math
import sys

V = 131072          # tokenizer.vocab_size (G3 frozen)
KAS_CODE = 2048     # KAS V2 code width (teams/04-ke2 #765)
KAS_RANK = 64       # KASHead tok_u / tok_c rank
KAS_BETA = 33       # KASHead beta
KE_P = 32           # ke_max_pos

# Gate 2 (config_v1.yaml): 16 routed + 1 shared experts, top-2.
GATE2 = {"num_experts": 16, "num_shared": 1, "top_k": 2}

# Instructor test sizes (experiments/plan_v1.md). "total" is the instructor's figure with tied embeddings.
T_SIZES = {
    "T1": {"d_model": 256, "n_layer": 6, "n_head": 4, "n_kv_head": 2, "num_experts": 8, "ffn_hidden": 512, "total": 56e6},
    "T2": {"d_model": 512, "n_layer": 8, "n_head": 8, "n_kv_head": 2, "num_experts": 16, "ffn_hidden": 768, "total": 233e6},
    "T3": {"d_model": 640, "n_layer": 10, "n_head": 10, "n_kv_head": 2, "num_experts": 16, "ffn_hidden": 896, "total": 386e6},
}


def spec(d_model, n_layer, n_head, n_kv_head, ffn_hidden, num_experts=16, num_shared=1, top_k=2,
         vocab=V, head="kas_v2", embedding=None, mtp=False):
    """One model shape. head: kas_v2 | dense | tied. embedding defaults to the head's pair
    (kas_v2 -> kas_v2 input, dense -> ke2, tied -> table)."""
    if d_model % n_head:
        raise ValueError("d_model must divide by n_head")
    if n_head % n_kv_head:
        raise ValueError("n_head must divide by n_kv_head")
    if top_k > num_experts:
        raise ValueError("top_k > num_experts")
    embedding = embedding or {"kas_v2": "kas_v2", "dense": "ke2", "tied": "table"}[head]
    return dict(d_model=d_model, n_layer=n_layer, n_head=n_head, n_kv_head=n_kv_head, ffn_hidden=ffn_hidden,
                num_experts=num_experts, num_shared=num_shared, top_k=top_k, vocab=vocab, head=head,
                embedding=embedding, mtp=mtp)


def count(s):
    """Total and active parameters, with the groups used by the check against the real model."""
    w, L, h, Vv = s["d_model"], s["n_layer"], s["ffn_hidden"], s["vocab"]
    E, S, k = s["num_experts"], s["num_shared"], s["top_k"]
    hd = w // s["n_head"]
    expert = 3 * w * h
    g = {
        "norms": L * 2 * w + w,
        "attention": L * (2 * w * w + 2 * w * s["n_kv_head"] * hd),
        "router": L * w * E,
        "experts_routed": L * E * expert,
        "experts_shared": L * S * expert,
        "input": {"kas_v2": KAS_CODE * w, "ke2": 256 * KE_P * w, "table": 0 if s["head"] == "tied" else Vv * w}[s["embedding"]],
        "head": {"kas_v2": KAS_CODE * w + KAS_BETA + Vv + KAS_RANK * Vv + KAS_RANK * w,
                 "dense": Vv * w, "tied": Vv * w}[s["head"]],
        "mtp": Vv * w if s["mtp"] else 0,
    }
    total = sum(g.values())
    active = total - g["experts_routed"] + L * k * expert
    # A table lookup does no matrix work, so leave it out of the training-FLOP count.
    compute = active - (g["input"] if s["embedding"] == "table" else 0)
    return {"total": total, "active": active, "compute": compute, "groups": g}


def round_to(x, m):
    return max(m, int(round(x / m)) * m)


def big(d_model, n_layer, ffn_ratio=1.5, head_dim=128, n_kv_head=8, **kw):
    """A large shape on the Gate 2 MoE: head dim 128, n_kv_head 8, expert FFN = ffn_ratio * d_model (multiple of 128)."""
    return spec(d_model, n_layer, d_model // head_dim, n_kv_head, round_to(ffn_ratio * d_model, 128), **GATE2, **kw)


def fmt(n):
    return f"{n / 1e9:.1f}B" if n >= 1e9 else f"{n / 1e6:.2f}M"


def grid(lo=30e9, hi=120e9, widths=(4096, 5120, 6144, 7168), layers=range(16, 97, 8), ffn_ratio=1.5):
    rows = []
    for w in widths:
        for L in layers:
            c = count(big(w, L, ffn_ratio))
            if lo <= c["total"] <= hi:
                rows.append((w, L, c))
    return rows


def check():
    """Formula against the real LanguageModel at T1, T2, T3 (dense head) and dry run 2 (KAS head)."""
    from pathlib import Path
    import torch
    team = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(team / "harness"))
    from week2.config import load_runtime
    from week2.model import LanguageModel
    rows = torch.randint(0, 256, (V, KE_P), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    lengths = torch.randint(1, KE_P + 1, (V,), generator=torch.Generator().manual_seed(1))
    out = []
    for arm in ("t1", "t2", "t3", "dryrun2"):
        m = load_runtime(team / f"configs/arms/{arm}.yaml")["model"]
        kas = m.get("head_mode") == "kas_v2"
        s = spec(m["n_embd"], m["n_layer"], m["n_head"], m["n_kv_head"], m["moe"]["ffn_hidden"],
                 m["moe"]["num_experts"], m["moe"]["num_shared"], m["moe"]["top_k"], m["vocab_size"],
                 head="kas_v2" if kas else "dense", mtp=bool(m["mtp"]))
        if kas:     # KAS initialisation reads tensor values, so build it on CPU
            real = LanguageModel(m, rows, lengths).parameter_count()
        else:
            with torch.device("meta"):
                real = LanguageModel(m, rows, lengths).parameter_count()
        out.append((arm, s, count(s)["total"], real))
    return out


def tokens_table(points, corpus_tokens, epochs=(1, 2, 4)):
    L = ["| Shape | Total | Active | " + " | ".join(f"tokens/active, {e} ep" for e in epochs)
         + " | tokens/total, 1 ep | 6·N·D, 1 ep |",
         "|---|---:|---:|" + "---:|" * len(epochs) + "---:|---:|"]
    for name, c in points:
        L.append(f"| {name} | {fmt(c['total'])} | {fmt(c['active'])} | "
                 + " | ".join(f"{e * corpus_tokens / c['active']:.1f}" for e in epochs)
                 + f" | {corpus_tokens / c['total']:.1f} | {6 * c['compute'] * corpus_tokens:.2e} |")
    return L


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true", help="compare the formula with the real model (needs torch)")
    ap.add_argument("--corpus-tokens", type=float, default=None, help="corpus-v1 token count, when it is known")
    ap.add_argument("--ffn-ratio", type=float, default=1.5)
    ap.add_argument("--md", help="write the grid tables here")
    a = ap.parse_args()
    if a.check:
        ok = True
        for arm, s, f, real in check():
            print(f"{arm:8s} formula {f:>12,d}  model {real:>12,d}  {'OK' if f == real else 'DIFFERENT'}")
            ok &= f == real
        for name, t in T_SIZES.items():
            s = spec(t["d_model"], t["n_layer"], t["n_head"], t["n_kv_head"], t["ffn_hidden"], t["num_experts"], head="tied")
            print(f"{name} tied {count(s)['total'] / 1e6:.2f}M vs instructor {t['total'] / 1e6:.0f}M")
        return 0 if ok else 1
    L = [f"Total and active parameters, Gate 2 MoE (16 routed + 1 shared, top-2), KAS head, head dim 128, "
         f"8 KV heads, expert FFN {a.ffn_ratio} x d_model. Only shapes with 30B to 120B total are listed.", "",
         "Tokens at 20 per active parameter is the dense compute-optimal rule of thumb, shown as a reference only. "
         "Weights are bf16, 2 bytes per total parameter.", "",
         "| d_model | Layers | d_model / layers | Heads | Expert FFN | Total | Active | Active / total "
         "| Tokens at 20 / active | 6·N·D at those tokens | bf16 weights |",
         "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for w, Lr, c in grid(ffn_ratio=a.ffn_ratio):
        s = big(w, Lr, a.ffn_ratio)
        d20 = 20 * c["active"]
        L.append(f"| {w} | {Lr} | {w / Lr:.0f} | {s['n_head']} | {s['ffn_hidden']} | {fmt(c['total'])} | "
                 f"{fmt(c['active'])} | {c['active'] / c['total']:.3f} | {fmt(d20)} | {6 * c['compute'] * d20:.2e} "
                 f"| {2 * c['total'] / 2**30:.0f} GiB |")
    if a.corpus_tokens:
        pts = [(f"d{w} L{Lr}", c) for w, Lr, c in grid(ffn_ratio=a.ffn_ratio)]
        L += ["", f"Corpus: {a.corpus_tokens:.3g} tokens.", ""] + tokens_table(pts, a.corpus_tokens)
    text = "\n".join(L) + "\n"
    if a.md:
        open(a.md, "w", encoding="utf-8").write(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
