"""KV-cache quantization, emulated: how much does quantizing the KV cache move a model's next-token predictions?

Every key and value vector (per head, per token, keys after RoPE) is fake-quantized inside attention through a
transformers AttentionInterface, then the logits are compared with the unquantized run on the same text: mean
KL(base || quant), same top-1, and the change in the actual next token's loss. Every position, the current token
included, is quantized, so the setting is pessimistic. A method is `m` or `keys/values`, with an optional `@skipN`
(the first and last N layers stay unquantized, as vLLM's TurboQuant presets do):
  fp8      e4m3 storage of K/V, scale 1.0
  int4     symmetric absmax per vector (a simple 4-bit baseline)
  u<b>     uniform per vector with an fp16 scale and zero point (vLLM's TurboQuant value quantizer)
  tq<b>    TurboQuant-MSE: unit vector -> seeded random Hadamard rotation -> Lloyd-Max codebook for the rotated
           coordinates (TSAI's V4 code: lightninglm/tqp) -> b bits per coordinate + one fp16 norm
  tqn<b>   tq<b> with norm correction (the reconstruction is rescaled to unit length first; vLLM's `_nc`)
  tqp<b>   TurboQuant-prod: (b-1)-bit tq plus 1-bit QJL on the residual (paper Alg. 2), two fp16 norms

    python kvq_study.py --lightninglm <The-School-of-AI/LLM checkout @ 2c18402e> --data <tokenizer_sample_v0_clean> \
        --model sarvamai/sarvam-1 --revision e9607337286ddf496d4a2562b194e489dcf3feea --device mps \
        --packs en-fineweb-edu code-python ind-hi-fw2 ind-bn-fw2 ind-ta-fw2 ind-kn-fw2 --per-pack 3 --tokens 1024 \
        --methods fp8 fp8/u4 tqn4/u4@skip2 tqn3/u4@skip2 tqn3/u3@skip2 --out results/<name>.json

Text: the train split of the cleaned training-sample packs (tokenizer sample v0), never benchmark items; it stays on the
machine that has it.
--dump-docs writes the token ids of the chosen documents (for modal_vllm_kv.py) and exits; keep that file local.
The output JSON holds per-document metrics only (no text, no token ids).
"""
import argparse, json, math, sys, time
from pathlib import Path
import numpy as np, pyarrow.parquet as pq, torch

DATA = None   # --data
STATE = {'method': 'none'}
_CACHE = {}
get_codebook = get_rotation_matrix = None   # from lightninglm.tqp, set by use_lightninglm()


def use_lightninglm(path):
    """TurboQuant's codebook and rotation from TSAI's public V4 repo (github.com/The-School-of-AI/LLM, Apache-2.0)."""
    global get_codebook, get_rotation_matrix
    sys.path.insert(0, str(path))
    from lightninglm.tqp.codebook import get_codebook as cb
    from lightninglm.tqp.turboquant_pretraining_linear import get_rotation_matrix as rot
    get_codebook, get_rotation_matrix = cb, rot


def tq_parts(d, bits, device):
    key = (d, bits, str(device))
    if key not in _CACHE:
        levels, bounds = get_codebook(d, bits)
        R = torch.from_numpy(get_rotation_matrix(d, seed=42)).float().to(device)
        _CACHE[key] = (torch.from_numpy(levels).float().to(device), torch.from_numpy(bounds[1:-1]).float().to(device), R)
    return _CACHE[key]


def qjl_matrix(d, device, seed=7):
    key = ('qjl', d, str(device))
    if key not in _CACHE:
        g = torch.Generator().manual_seed(seed)
        _CACHE[key] = torch.randn(d, d, generator=g).float().to(device)
    return _CACHE[key]


def tq_mse(x, bits):
    levels, inner, R = tq_parts(x.shape[-1], bits, x.device)
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    r = (x / n) @ R.t()
    q = levels[torch.bucketize(r.contiguous(), inner)]
    return (q @ R) * n.half().float()


def tq_mse_renorm(x, bits):
    """TurboQuant-MSE, but the reconstructed unit vector is rescaled to length 1 before the stored norm is applied:
    removes most of the MSE quantizer's shrinkage of inner products at no extra bits."""
    levels, inner, R = tq_parts(x.shape[-1], bits, x.device)
    n = x.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    q = levels[torch.bucketize(((x / n) @ R.t()).contiguous(), inner)] @ R
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12) * n.half().float()


def tq_prod(x, bits):
    """TurboQuant-prod (paper Alg. 2): (bits-1)-bit MSE quantizer, then 1-bit QJL on the residual, so <y, x~> is unbiased.
    x~ = Q_mse(x) + sqrt(pi/2)/d * ||r|| * S^T sign(S r), r = x - Q_mse(x), S a seeded d x d Gaussian. Two fp16 norms."""
    d = x.shape[-1]; xm = tq_mse(x, bits - 1); r = x - xm
    S = qjl_matrix(d, x.device); rn = r.norm(dim=-1, keepdim=True).half().float()
    return xm + math.sqrt(math.pi / 2) / d * rn * (torch.sign(r @ S.t()) @ S)


def fake_quant(x, method):
    if method == 'none':
        return x
    dt = x.dtype; x = x.float()
    if method == 'fp8':
        y = x.clamp(-448, 448).to(torch.float8_e4m3fn).float()
    elif method in ('u4', 'u3', 'u2'):
        b = int(method[1:]); lo = x.amin(-1, keepdim=True); hi = x.amax(-1, keepdim=True)
        s = ((hi - lo) / (2 ** b - 1)).clamp_min(1e-8).half().float(); lo = lo.half().float()
        y = ((x - lo) / s).round().clamp(0, 2 ** b - 1) * s + lo
    elif method == 'int4':
        s = x.abs().amax(-1, keepdim=True).clamp_min(1e-8) / 7
        y = (x / s).round().clamp(-8, 7) * s
    elif method.startswith('tqp'):
        y = tq_prod(x, int(method[3:]))
    elif method.startswith('tqn'):
        y = tq_mse_renorm(x, int(method[3:]))
    elif method.startswith('tq'):
        y = tq_mse(x, int(method[2:]))
    else:
        raise ValueError(method)
    return y.to(dt)


def install():
    from transformers import AttentionInterface
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    base = ALL_ATTENTION_FUNCTIONS['sdpa']

    def kvq_attention(module, query, key, value, attention_mask, **kw):
        m = STATE['method']; skip = 0
        if '@skip' in m:
            m, s = m.split('@skip'); skip = int(s)
        li = getattr(module, 'layer_idx', None)
        if skip and li is not None and (li < skip or li >= STATE['n_layers'] - skip):
            m = 'none'
        km, vm = m.split('/') if '/' in m else (m, m)
        return base(module, query, fake_quant(key, km), fake_quant(value, vm), attention_mask, **kw)
    AttentionInterface.register('kvq', kvq_attention)


def texts(tok, packs, per_pack, T):
    out = {}
    for pack in packs:
        files = sorted((DATA / 'train' / pack).glob('*.parquet'))   # the train split only, never held-out text
        if not files:
            sys.exit(f'{DATA / "train" / pack}: no parquet files')
        got = []
        for f in files:
            for t in pq.read_table(f, columns=['text']).column('text').to_pylist():
                ids = tok(t, add_special_tokens=False)['input_ids']
                if len(ids) >= T:
                    got.append(ids[:T])
                if len(got) >= per_pack:
                    break
            if len(got) >= per_pack:
                break
        out[pack] = got
    return out


def one_bits(m, d):
    if m == 'fp8': return 8.0
    if m == 'int4': return 4 + 16 / d
    if m in ('u4', 'u3', 'u2'): return int(m[1:]) + 32 / d
    if m.startswith('tqp'): return int(m[3:]) + 32 / d
    if m.startswith('tqn'): return int(m[3:]) + 16 / d
    if m.startswith('tq'): return int(m[2:]) + 16 / d
    raise ValueError(m)


def kv_bits(m, d, n_layers=None):
    skip = 0
    if '@skip' in m:
        m, s = m.split('@skip'); skip = int(s)
    km, vm = m.split('/') if '/' in m else (m, m)
    if m == 'none':
        return 16.0
    q = (one_bits(km, d) + one_bits(vm, d)) / 2
    if skip and n_layers:                       # skipped layers keep bf16 (16 bits)
        q = (q * (n_layers - 2 * skip) + 16 * 2 * skip) / n_layers
    return round(q, 4)


def metrics(base, q, ids) -> dict:
    """Per document: KL(base || q) averaged over positions, the share of positions with the same top-1, and the change in
    the mean NLL of the actual next token (q minus base)."""
    lp0, lpq = base.log_softmax(-1), q.log_softmax(-1)
    tgt = torch.tensor(ids[1:], device=base.device)[:, None]
    nll = lambda lp: -lp[:-1].gather(-1, tgt).mean().item()
    return dict(kl=(lp0.exp() * (lp0 - lpq)).sum(-1).mean().item(),
                top1=(base.argmax(-1) == q.argmax(-1)).float().mean().item(),
                dnll=nll(lpq) - nll(lp0))


@torch.no_grad()
def score(model, ids, device):
    x = torch.tensor([ids], device=device)
    return model(input_ids=x, use_cache=False).logits[0].float()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True); ap.add_argument('--tokenizer'); ap.add_argument('--device', default='cpu')
    ap.add_argument('--packs', nargs='+', default=['en-fineweb-edu', 'code-python', 'ind-hi-fw2', 'ind-bn-fw2', 'ind-ta-fw2'])
    ap.add_argument('--per-pack', type=int, default=4); ap.add_argument('--tokens', type=int, default=1024)
    ap.add_argument('--methods', nargs='+', default=['fp8', 'int4', 'tq4', 'tq3', 'tq2'])
    ap.add_argument('--dtype', default='float32'); ap.add_argument('--out')
    ap.add_argument('--revision', help='model revision (Team 12 pins sarvam-1 at e9607337)')
    ap.add_argument('--data', type=Path, required=True, help='tokenizer_sample_v0_clean folder (parquet packs)')
    ap.add_argument('--lightninglm', type=Path, help='checkout of github.com/The-School-of-AI/LLM (needed for tq*)')
    ap.add_argument('--dump-docs', type=Path, help='write the chosen documents as token ids (for modal_vllm_kv.py) and exit')
    a = ap.parse_args()
    global DATA
    DATA = a.data
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer or a.model, revision=a.revision)
    if a.dump_docs:
        data = texts(tok, a.packs, a.per_pack, a.tokens)
        a.dump_docs.write_text(json.dumps(dict(model=a.model, revision=a.revision, tokens=a.tokens,
                                               packs=[p for p in a.packs for _ in data[p]], ids=[x for p in a.packs for x in data[p]])))
        print(f'wrote {sum(len(v) for v in data.values())} documents to {a.dump_docs} (token ids of training text: keep it local)')
        return
    if not a.out:
        ap.error('--out is required unless --dump-docs is given')
    if any('tq' in m for m in a.methods):
        if not a.lightninglm:
            ap.error('tq* methods need --lightninglm')
        use_lightninglm(a.lightninglm)
    install()
    model = AutoModelForCausalLM.from_pretrained(a.model, revision=a.revision, dtype=getattr(torch, a.dtype),
                                                 attn_implementation='kvq').to(a.device).eval()
    cfg = model.config
    STATE['n_layers'] = cfg.num_hidden_layers
    head_dim = getattr(cfg, 'head_dim', None) or cfg.hidden_size // cfg.num_attention_heads
    data = texts(tok, a.packs, a.per_pack, a.tokens)
    res = {'model': a.model, 'tokens': a.tokens, 'head_dim': head_dim, 'n_kv_heads': getattr(cfg, 'num_key_value_heads', None),
           'docs': {p: len(v) for p, v in data.items()}, 'bits_per_channel': {m: kv_bits(m, head_dim, cfg.num_hidden_layers) for m in a.methods}, 'results': {}}
    t0 = time.time()
    for pack, docs in data.items():
        for ids in docs:
            STATE['method'] = 'none'; base = score(model, ids, a.device)
            for m in a.methods:
                STATE['method'] = m; got = metrics(base, score(model, ids, a.device), ids)
                r = res['results'].setdefault(pack, {}).setdefault(m, {'kl': [], 'top1': [], 'dnll': []})
                for k in r:
                    r[k].append(got[k])
        print(f'{pack}: {len(docs)} docs, {time.time() - t0:.0f}s', flush=True)
    summary = {p: {m: {k: round(float(np.mean(v)), 5) for k, v in r.items()} for m, r in d.items()} for p, d in res['results'].items()}
    res['summary'] = summary
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({'bits_per_channel': res['bits_per_channel'], 'summary': summary}, indent=1))


if __name__ == '__main__':
    main()
