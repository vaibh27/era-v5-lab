#!/usr/bin/env python3
"""What the Ollama / GGUF version of a G5 checkpoint loses against the checkpoint itself, measured on real text.

    python measure_export_gaps.py --checkpoint <run dir> --tokenizer tokenizer.json --texts <dir of <lang>.txt> \
        --out gaps.json [--remap gurmukhi] [--tokens 20000] [--window 512] [--device cpu]

The export to Granite MoE Shared (export_hf.py) is exact except for three things, each measured here apart:
  router_bias  the loss-free balancer's selection bias (bias_gamma > 0): Granite picks experts from the raw scores;
  head_bias    the KAS head's per-token output bias (head.unigram): Granite's lm_head has no bias;
  tokenizer    llama.cpp applies only NFC, not p1536's 15 replacements, so plain text can get other ids
               (with --remap, the remapped GGUF vocabulary of remap_vocab.py is used, as package_ollama.py ships it).

Model variants, all G5's own LanguageModel in fp32 (so the KE2/KAS fold, exact by test A1, is not in question):
  reference    the checkpoint as trained
  no_router    router_bias set to 0
  no_head      head.unigram set to 0
  granite      both: what the GGUF computes
  granite_fold both dropped, then approximated in stock-Granite form (still loads in stock llama.cpp / Ollama):
               fit v with h.v ~= 1 over the final normalized hidden states h of calibration text, and put
               unigram (x) v into lm_head, so logits ~= W h + unigram; per layer, fit v_l on the router's input the
               same way and put router_bias (x) v_l into the router weight. Exact if a constant direction exists
               in h (and in each router input); the fit is reported (`fold_fit`). Calibration uses the first
               --calib-frac of every language's tokens, and every score in this file uses only the rest.
For each language: bits per byte (bpb) of the reference and of each variant on the training token ids, the share of
positions where the variant's top-1 token equals the reference's, and the mean KL(reference || variant) in nats.
Tokenizer: bpb of `granite` on the ids llama.cpp would give for the same text (NFC-only normalizer, remapped vocab),
against `granite` on the training ids. bpb is per UTF-8 byte of the text, so different tokenizations compare fairly.
Text is only read; nothing of it is written to the output.
"""
from __future__ import annotations

import argparse, json, math, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "scripts"))
sys.path.insert(0, str(HERE))


def windows(ids: list[int], n: int):
    for i in range(0, len(ids) - 1, n):
        chunk = ids[i:i + n + 1]
        if len(chunk) > 1:
            yield chunk


def fit_constant_direction(rows, ridge: float = 1e-4):
    """v minimising ||H v - 1||^2 + ridge * mean(diag(H^T H)) * ||v||^2 (fp64); returns v and how constant H v is."""
    import torch
    H = torch.cat(rows).double()
    G = H.T @ H
    lam = ridge * G.diagonal().mean()
    v = torch.linalg.solve(G + lam * torch.eye(G.shape[0], dtype=G.dtype), H.T @ torch.ones(len(H), dtype=H.dtype))
    hv = H @ v
    return v.float(), {"rows": len(H), "mean": round(hv.mean().item(), 5), "std": round(hv.std().item(), 5),
                       "p01": round(hv.quantile(0.01).item(), 4), "p99": round(hv.quantile(0.99).item(), 4)}


class Folded:
    """granite_fold: a model with both biases dropped and approximated through fitted constant directions."""

    def __init__(self, model, v_head, unigram):
        self.m, self.v, self.b = model, v_head, unigram

    def __call__(self, x, valid):
        h = self.m.hidden(x, valid)
        return self.m.head(h) + (h @ self.v)[..., None] * self.b

    def to(self, device):
        self.m.to(device); self.v = self.v.to(device); self.b = self.b.to(device)
        return self


def score(models: dict, ids: list[int], window: int, device: str) -> dict:
    """Sum of NLL (nats) of each model on the next tokens, top-1 agreement and KL against models['reference']."""
    import torch
    import torch.nn.functional as F
    out = {k: {"nll": 0.0, "same_top1": 0, "kl": 0.0} for k in models}
    n = 0
    with torch.no_grad():
        for chunk in windows(ids, window):
            x = torch.tensor(chunk[:-1], device=device)[None]
            y = torch.tensor(chunk[1:], device=device)
            valid = torch.ones_like(x, dtype=torch.bool)
            ref_logp = None
            for k, m in models.items():
                logp = F.log_softmax(m(x, valid)[0].float(), -1)
                out[k]["nll"] += -logp.gather(1, y[:, None]).sum().item()
                if ref_logp is None:
                    ref_logp = logp
                out[k]["same_top1"] += (logp.argmax(-1) == ref_logp.argmax(-1)).sum().item()
                out[k]["kl"] += (ref_logp.exp() * (ref_logp - logp)).sum().item()
            n += len(y)
    return {"positions": n, **out}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True, type=Path)
    ap.add_argument("--spec", type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--texts", required=True, type=Path, help="folder of <lang>.txt files (one document per line)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--remap", nargs="*", default=["gurmukhi"], metavar="SCRIPT")
    ap.add_argument("--tokens", type=int, default=20000, help="training-id tokens scored per language")
    ap.add_argument("--window", type=int, default=512)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--calib-frac", type=float, default=0.3, help="share of each language's tokens used to fit the fold")
    ap.add_argument("--no-fold", action="store_true", help="skip the granite_fold variant")
    a = ap.parse_args()

    import copy, torch
    from tokenizers import Tokenizer
    from export_hf import find_checkpoint, load_state, load_spec, resolve_spec, load_reference, HARNESS
    from remap_vocab import remap
    sys.path.insert(0, str(HARNESS))
    from week2.model import LanguageModel

    weights, manifest = find_checkpoint(a.checkpoint)
    state = load_state(weights)
    spec = resolve_spec(state, load_spec(a.spec or manifest))
    ref = LanguageModel(dict(spec, mtp="mtp_head.weight" in state), state["embedding.rows"].numpy(),
                        state["embedding.lengths"].numpy()).float().eval()
    load_reference(ref, state)
    variants = {"reference": ref}
    has_router = any(k.endswith(".router_bias") and v.abs().max() > 0 for k, v in state.items())
    has_head = "head.unigram" in state
    def without(router: bool, head: bool):
        m = copy.deepcopy(ref)
        with torch.no_grad():
            for name, b in m.named_buffers():
                if router and name.endswith(".router_bias"):
                    b.zero_()
            if head:
                m.head.unigram.zero_()
        return m
    if has_router:
        variants["no_router"] = without(True, False)
    if has_head:
        variants["no_head"] = without(False, True)
    variants["granite"] = without(has_router, has_head)
    train = Tokenizer.from_file(str(a.tokenizer))
    tj = json.loads(a.tokenizer.read_text(encoding="utf-8"))
    if a.remap:
        tj, _ = remap(tj, tuple(a.remap))
    tj["normalizer"] = {"type": "NFC"}                    # what a GGUF carries
    llamacpp = Tokenizer.from_str(json.dumps(tj))

    # documents per language, split into calibration (fold fit only) and evaluation (every score below)
    langs = {}
    for path in sorted(a.texts.glob("*.txt")):
        docs = [d for d in path.read_text(encoding="utf-8").split("\n") if d.strip()]
        calib, evald, n = [], [], 0
        for d in docs:
            t = train.encode(d, add_special_tokens=False).ids
            if n + len(t) > a.tokens:
                break
            (calib if n < a.calib_frac * a.tokens and not a.no_fold else evald).append((d, t))
            n += len(t)
        if evald:
            langs[path.stem] = (calib, evald)

    fold_fit = None
    if not a.no_fold and (has_router or has_head):
        def capture(texts):
            """Final normalized hidden states and every router's input of the reference model on `texts`."""
            heads, routers = [], {i: [] for i in range(len(ref.blocks))}
            hooks = [ref.norm.register_forward_hook(lambda mod, i, o: heads.append(o.detach().reshape(-1, o.shape[-1])))]
            for i, blk in enumerate(ref.blocks):
                hooks.append(blk.n2.register_forward_hook(
                    lambda mod, inp, o, i=i: routers[i].append(o.detach().reshape(-1, o.shape[-1]))))
            with torch.no_grad():
                for ids in texts:
                    for chunk in windows(ids, a.window):
                        x = torch.tensor(chunk[:-1])[None]
                        ref(x, torch.ones_like(x, dtype=torch.bool))
            for h in hooks:
                h.remove()
            return heads, routers
        heads, routers = capture([[t for _, d in c for t in d] for c, _ in langs.values()])
        # held-out router inputs: the first calibration-sized slice of each language's evaluation text
        _, held = capture([[t for _, d in ev for t in d][:max(1, int(a.calib_frac * a.tokens))] for _, ev in langs.values()])
        fm = without(has_router, has_head)
        fold_fit = {}
        if has_head:
            v, fold_fit["head"] = fit_constant_direction(heads)
        if has_router:
            fold_fit["router"] = {}
            with torch.no_grad():
                for i, blk in enumerate(fm.blocks):
                    # G5 selects topk(softmax(Rx) + beta) and gates with the unbiased softmax. A logit shift d
                    # multiplies a probability by e^d, so d = log((p_k + beta) / p_k), p_k = mean probability at the
                    # top-k boundary on the calibration text; d * (x . v_l) ~= d goes into the router weight. The
                    # balancer can push beta below -p_k (an overloaded expert): clamp p_k + beta at 1e-3 p_k, so
                    # that expert gets a large negative shift (about -6.9) instead of a NaN.
                    vl, fit = fit_constant_direction(routers[i])
                    moe, beta = ref.blocks[i].moe, ref.blocks[i].moe.router_bias.float()
                    W = moe.router.weight.float()
                    srt = (torch.cat(routers[i]).float() @ W.T).softmax(-1).sort(-1, descending=True).values
                    pk = 0.5 * (srt[:, moe.topk - 1] + srt[:, moe.topk]).mean()
                    d = torch.log((pk + beta).clamp_min(1e-3 * pk) / pk)
                    fit["boundary_prob"] = round(pk.item(), 5)
                    fit["experts_clamped"] = int((pk + beta < 1e-3 * pk).sum())
                    # on held-out tokens: same experts as G5, and how far the gate values move (G5 keeps the
                    # unbiased softmax as gates; a logit shift changes them too), top-k gates renormalized
                    X = torch.cat(held[i]).float()
                    z = X @ W.T
                    p = z.softmax(-1)
                    want = (p + beta).topk(moe.topk, -1).indices.sort(-1).values
                    zf = z + d * (X @ vl)[:, None]
                    got = zf.topk(moe.topk, -1).indices.sort(-1).values
                    same = (got == want).all(-1)
                    g5 = p.gather(-1, want); g5 = g5 / g5.sum(-1, keepdim=True)
                    pf = zf.softmax(-1).gather(-1, got); pf = pf / pf.sum(-1, keepdim=True)
                    fit["held_out_tokens"] = len(X)
                    fit["same_experts"] = round(same.float().mean().item(), 4)
                    fit["same_experts_if_dropped"] = round(
                        (p.topk(moe.topk, -1).indices.sort(-1).values == want).all(-1).float().mean().item(), 4)
                    fit["gate_abs_error_when_same"] = round((g5 - pf).abs()[same].mean().item(), 5) if same.any() else None
                    fold_fit["router"][i] = fit
                    blk.moe.router.weight += d[:, None] * vl[None, :]
        variants["granite_fold"] = Folded(fm, v, state["head.unigram"].float()) if has_head else fm
    for m in variants.values():
        m.to(a.device)

    result = {"checkpoint": str(weights), "variants": list(variants), "remap": a.remap, "window": a.window,
              "calib_frac": None if a.no_fold else a.calib_frac, "fold_fit": fold_fit,
              "router_bias_present": has_router, "head_bias_present": has_head, "languages": {}}
    for name, (calib, evald) in langs.items():
        docs = [d for d, _ in evald]
        ids_t = [t for _, ids in evald for t in ids]
        ids_l = [t for d in docs for t in llamacpp.encode(d, add_special_tokens=False).ids]
        nbytes = sum(len(d.encode("utf-8")) for d in docs)
        s = score(variants, ids_t, a.window, a.device)
        ship = "granite"     # what package_ollama.py ships today (the drop-both export); the fold is measured only
        g = score({ship: variants[ship]}, ids_l, a.window, a.device)[ship]
        bpb = lambda nll: nll / nbytes / math.log(2)
        lang = {"calibration_documents": len(calib), "documents": len(docs), "bytes": nbytes,
                "tokens_training": len(ids_t), "tokens_llamacpp": len(ids_l),
                "docs_with_other_ids": sum(train.encode(d, add_special_tokens=False).ids
                                           != llamacpp.encode(d, add_special_tokens=False).ids for d in docs),
                "bpb": {k: round(bpb(v["nll"]), 4) for k, v in s.items() if isinstance(v, dict)},
                "same_top1_vs_reference": {k: round(v["same_top1"] / s["positions"], 4)
                                           for k, v in s.items() if isinstance(v, dict) and k != "reference"},
                "mean_kl_nats_vs_reference": {k: round(v["kl"] / s["positions"], 4)
                                              for k, v in s.items() if isinstance(v, dict) and k != "reference"},
                "tokenizer_cost_variant": ship, "bpb_llamacpp_ids": round(bpb(g["nll"]), 4)}
        lang["tokenizer_cost_bpb"] = round(lang["bpb_llamacpp_ids"] - lang["bpb"][ship], 4)
        result["languages"][name] = lang
        print(name, json.dumps({k: lang[k] for k in ("bpb", "tokenizer_cost_bpb", "docs_with_other_ids")}),
              flush=True)
    a.out.write_text(json.dumps(result, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
