#!/usr/bin/env python3
"""The README's figures, drawn from the committed results (light and dark SVG of each):  python figures/make_figures.py

Every number comes from ../results/*.json or from vLLM 0.33.0's slot sizes (vllm_bits below), never typed in.
Motion is CSS inside each SVG: it plays once on load (one slow marker breathes), stops for prefers-reduced-motion, and
a renderer without CSS animation shows the finished figure. --static writes the finished state only (for previews).
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / 'results'

PAL = {
    'light': dict(bg='#FBFAF7', card='#FFFFFF', line='#E6E1D6', ink='#1C1B19', muted='#7A746A', soft='#F3F0E9',
                  grid='#EEEAE1', navy='#25324E', navy2='#56688F', navy3='#9AA6C0', brass='#A9844B', sage='#5C8B79',
                  clay='#B5705F'),
    'dark': dict(bg='#0F1115', card='#15181D', line='#272B33', ink='#ECE8E0', muted='#9A958A', soft='#1B1F25',
                 grid='#21252C', navy='#B3C1DD', navy2='#8396BC', navy3='#5A6785', brass='#D3AF73', sage='#88B9A5',
                 clay='#D9978A'),
}
SANS = "Inter, 'SF Pro Text', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif"
MONO = "'JetBrains Mono', 'SF Mono', ui-monospace, Menlo, Consolas, monospace"
W = 880


def vllm_bits(key_bits: int, value_bits: int, head_dim: int) -> float:
    """Bits per element on a compressed layer in vLLM 0.33.0: per head and token, keys take ceil(d*b/8) bytes plus an
    fp16 norm, values the same plus an fp16 scale and zero point."""
    k = math.ceil(head_dim * key_bits / 8) + 2
    v = math.ceil(head_dim * value_bits / 8) + 4
    return (k + v) * 8 / (2 * head_dim)


def tokens_vs_bf16(bits: float, layers: int) -> float:
    """KV tokens against bf16 when min(2, layers // 2) layers at each end stay bf16 (vLLM 0.33.0's boundary skip)."""
    skip = 2 * min(2, layers // 2)
    return 16 * layers / (bits * (layers - skip) + 16 * skip)


def doc_values(name: str, method: str, metric: str = 'kl') -> list[float]:
    res = json.loads((RES / name).read_text())['results']
    return [x for pack in res.values() if method in pack for x in pack[method][metric]]


def mean(xs):
    return sum(xs) / len(xs)


class Svg:
    def __init__(self, theme: str, h: int, static: bool):
        self.p, self.h, self.static, self.parts = PAL[theme], h, static, []

    def add(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, cls='', anchor='start', fill=None, delay=None, extra=''):
        f = f' fill="{fill}"' if fill else ''
        st = f' style="animation-delay:{delay:.2f}s"' if delay is not None and not self.static else ''
        c = f' class="{cls}"' if cls else ''
        self.add(f'<text x="{x:.1f}" y="{y:.1f}"{c} text-anchor="{anchor}"{f}{st}{extra}>{s}</text>')

    def render(self, title: str) -> str:
        p = self.p
        anim = '' if self.static else (
            '@keyframes grow{from{transform:scaleX(0)}to{transform:scaleX(1)}}'
            '@keyframes rise{from{transform:scaleY(0)}to{transform:scaleY(1)}}'
            '@keyframes fade{from{opacity:0;transform:translateY(5px)}to{opacity:1;transform:none}}'
            '@keyframes show{from{opacity:0}to{opacity:1}}'
            '@keyframes draw{from{stroke-dashoffset:1}to{stroke-dashoffset:0}}'
            '@keyframes breathe{0%,100%{opacity:.25;transform:scale(.85)}50%{opacity:.75;transform:scale(1.25)}}'
            '.g{transform-box:fill-box;transform-origin:left center;animation:grow 1.3s cubic-bezier(.22,.8,.24,1) both}'
            '.r{transform-box:fill-box;transform-origin:center bottom;animation:rise 1.1s cubic-bezier(.22,.8,.24,1) both}'
            '.f{animation:fade .9s cubic-bezier(.2,.7,.2,1) both}'
            '.s{animation:show .7s ease-out both}'
            '.d{stroke-dasharray:1;animation:draw 1.8s cubic-bezier(.45,.05,.25,1) both}'
            '.b{transform-box:fill-box;transform-origin:center;animation:breathe 3.6s ease-in-out infinite}'
            '@media (prefers-reduced-motion:reduce){*{animation:none!important}}')
        css = (f'text{{font-family:{SANS};fill:{p["ink"]}}}'
               f'.t{{font-size:21px;font-weight:600;letter-spacing:-.012em}}'
               f'.st{{font-size:12.5px;fill:{p["muted"]}}}'
               f'.cap{{font-size:10px;font-weight:600;letter-spacing:.14em;fill:{p["muted"]}}}'
               f'.lab{{font-size:12px;fill:{p["muted"]}}}'
               f'.lab2{{font-size:12.5px;font-weight:500}}'
               f'.n{{font-family:{MONO};font-variant-numeric:tabular-nums;font-size:12px}}'
               f'.big{{font-family:{MONO};font-variant-numeric:tabular-nums;font-size:34px;font-weight:500;letter-spacing:-.02em}}'
               f'.ax{{font-family:{MONO};font-size:10.5px;fill:{p["muted"]}}}' + anim)
        return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{self.h}" viewBox="0 0 {W} {self.h}" '
                f'role="img" aria-label="{title}"><title>{title}</title><style>{css}</style>'
                f'<rect x=".5" y=".5" width="{W - 1}" height="{self.h - 1}" rx="16" fill="{p["bg"]}" stroke="{p["line"]}"/>'
                + ''.join(self.parts) + '</svg>\n')


def header(s: Svg, kicker: str, title: str, sub: str):
    p = s.p
    s.text(36, 44, kicker, 'cap', fill=p['brass'])
    s.text(36, 72, title, 't')
    s.text(36, 94, sub, 'st')


# ---------------------------------------------------------------------------------------------------------------- hero
def hero(theme, static, v):
    s = Svg(theme, 286, static); p = s.p
    s.text(36, 46, 'TEAM 12 · SERVING · KV CACHE', 'cap', fill=p['brass'])
    s.text(36, 78, 'How far can the KV cache shrink?', 't', extra=' style="font-size:25px"')
    s.text(36, 102, f"fp8 and TurboQuant in vLLM {v['vllm']} · sarvam-1 as a proxy · one NVIDIA L4 per dtype", 'st')
    cls = '' if static else ' class="g"'
    s.add(f'<rect x="36" y="116" width="64" height="2" rx="1" fill="{p["brass"]}"{cls}/>')
    d = v['dtypes']; fp8, tq = d['fp8'], d['turboquant_4bit_nc']
    t2 = tokens_vs_bf16(vllm_bits(4, 4, 64), 8)
    tiles = [
        ('FP8 · THE SAFE FIRST STEP', f"{fp8['kv_ratio']:.2f}×", 'KV tokens against bf16',
         f"ΔNLL ≈ 0 · {fp8['speed_change']:+.0%} tok/s", p['sage']),
        ('TURBOQUANT 4BIT_NC · SARVAM-1', f"{tq['kv_ratio']:.2f}×", 'not the 3.8× of its docstring',
         f"ΔNLL {tq['drift']['dnll_mean']:+.3f} · {tq['speed_change']:+.0%} tok/s".replace('-', '−'), p['navy']),
        ('TURBOQUANT 4BIT_NC · OUR T2', f'{t2:.2f}×', 'below fp8’s 2.00× (computed)',
         '4 of 8 layers stay bf16', p['clay']),
    ]
    tw, gap, x0, y0 = 258, 15, 36, 140
    for i, (cap, big, l1, l2, col) in enumerate(tiles):
        x = x0 + i * (tw + gap)
        a = '' if static else f' class="f" style="animation-delay:{.25 + .18 * i:.2f}s"'
        s.add(f'<g{a}><rect x="{x}" y="{y0}" width="{tw}" height="118" rx="12" fill="{p["card"]}" stroke="{p["line"]}"/>'
              f'<rect x="{x}" y="{y0 + 18}" width="3" height="82" rx="1.5" fill="{col}"/>'
              f'<text x="{x + 20}" y="{y0 + 30}" class="cap">{cap}</text>'
              f'<text x="{x + 20}" y="{y0 + 72}" class="big" fill="{col}">{big}</text>'
              f'<text x="{x + 20}" y="{y0 + 92}" class="lab">{l1}</text>'
              f'<text x="{x + 20}" y="{y0 + 108}" class="n" fill="{p["muted"]}">{l2}</text></g>')
    return s.render('KV-cache quantization at a glance')


# ------------------------------------------------------------------------------------------------ capacity vs drift
def tradeoff(theme, static, v):
    s = Svg(theme, 446, static); p = s.p
    header(s, 'VLLM 0.33.0 · SARVAM-1 · 18 DOCUMENTS × 1,024 TOKENS', 'Capacity against drift',
           'KV tokens the cache holds (× bf16) and the change in next-token NLL against the bf16 cache; bars are ±1 s.e.')
    X0, X1, Y0, Y1 = 92, 840, 372, 128           # plot box
    xs = lambda x: X0 + (x - 0.8) / (3.4 - 0.8) * (X1 - X0)
    ys = lambda y: Y0 - (y + 0.004) / (0.06 + 0.004) * (Y0 - Y1)
    for gx in (1, 1.5, 2, 2.5, 3):
        s.add(f'<line x1="{xs(gx):.1f}" y1="{Y1}" x2="{xs(gx):.1f}" y2="{Y0}" stroke="{p["grid"]}"/>')
        s.text(xs(gx), Y0 + 18, f'{gx:g}×', 'ax', 'middle')
    for gy in (0, 0.02, 0.04, 0.06):
        s.add(f'<line x1="{X0}" y1="{ys(gy):.1f}" x2="{X1}" y2="{ys(gy):.1f}" stroke="{p["grid"]}"/>')
        s.text(X0 - 10, ys(gy) + 3.5, f'{gy:.2f}', 'ax', 'end')
    s.text(X0 - 10, Y1 - 12, 'ΔNLL, nats', 'ax', 'end')
    s.text(X1, Y0 + 36, 'KV tokens vs bf16 →', 'ax', 'end')
    d = v['dtypes']
    order = [('fp8', 'fp8', p['sage']), ('turboquant_4bit_nc', '4bit_nc', p['navy']),
             ('turboquant_k3v4_nc', 'k3v4_nc', p['navy2']), ('turboquant_3bit_nc', '3bit_nc', p['navy3'])]
    tq = [(xs(d[k]['kv_ratio']), ys(d[k]['drift']['dnll_mean'])) for k, _, _ in order[1:]]
    path = 'M' + ' L'.join(f'{x:.1f},{y:.1f}' for x, y in tq)
    dcls = '' if static else ' class="d" pathLength="1"'
    s.add(f'<path d="{path}" fill="none" stroke="{p["navy3"]}" stroke-width="1.2" stroke-dasharray="1"{dcls} opacity=".7"/>'
          if not static else f'<path d="{path}" fill="none" stroke="{p["navy3"]}" stroke-width="1.2" stroke-dasharray="3 4" opacity=".7"/>')
    bx, by = xs(1), ys(0)
    s.add(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="5.5" fill="{p["bg"]}" stroke="{p["muted"]}" stroke-width="1.4"/>')
    s.text(bx + 12, by + 4, 'bf16', 'lab')
    for i, (k, name, col) in enumerate(order):
        e = d[k]; dr = e['drift']
        x, y = xs(e['kv_ratio']), ys(dr['dnll_mean'])
        se = (ys(dr['dnll_mean'] - dr['dnll_se']) - ys(dr['dnll_mean'] + dr['dnll_se'])) / 2
        a = '' if static else f' class="f" style="animation-delay:{.4 + .22 * i:.2f}s"'
        lx, anchor = (x - 14, 'end') if k in ('turboquant_k3v4_nc', 'turboquant_3bit_nc') else (x + 14, 'start')
        ly = y + (4 if k != 'fp8' else -10)
        s.add(f'<g{a}><line x1="{x:.1f}" y1="{y - se:.1f}" x2="{x:.1f}" y2="{y + se:.1f}" stroke="{col}" stroke-width="1.4"/>'
              f'<circle cx="{x:.1f}" cy="{y:.1f}" r="6.5" fill="{col}"/>'
              f'<text x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}" class="lab2">{name}</text>'
              f'<text x="{lx:.1f}" y="{ly + 16:.1f}" text-anchor="{anchor}" class="n" fill="{p["muted"]}">'
              f'{e["kv_ratio"]:.2f}× · {dr["dnll_mean"]:+.4f} · {e["speed_change"]:+.0%} tok/s</text></g>'.replace('+-', '−').replace(' -', ' −'))
    note = '' if static else ' class="s" style="animation-delay:1.5s"'
    s.add(f'<g{note}><text x="{X0 + 18}" y="{Y1 + 14}" class="lab">fp8: no measurable drift, and the fastest</text>'
          f'<text x="{X0 + 18}" y="{Y1 + 30}" class="lab">TurboQuant: more tokens, steadily more drift</text></g>')
    return s.render('Capacity against drift in vLLM 0.33.0 on sarvam-1')


# ------------------------------------------------------------------------------------------------------ layer strips
def layers(theme, static, v):
    s = Svg(theme, 330, static); p = s.p
    header(s, 'WHY 2.7× AND NOT 3.8×', 'vLLM keeps two layers at each end in bf16',
           'min(2, layers // 2) at each end, for every TurboQuant preset, whatever the depth; a user skip list cannot remove them')
    b128, b64 = vllm_bits(4, 4, 128), vllm_bits(4, 4, 64)
    rows = [('sarvam-1 · 28 layers · head dim 128', 28, b128, False),
            ('T2 · dry run 2 · 8 layers · head dim 64', 8, b64, False),
            ('docstring · compressed layers only', 28, b128, True)]
    x0, x1, y = 36, 640, 132
    for r, (label, L, bits, ghost) in enumerate(rows):
        yy = y + r * 62
        s.text(x0, yy, label, 'lab2' if not ghost else 'lab')
        cw = (x1 - x0 - (L - 1) * 3) / L
        skip = 0 if ghost else min(2, L // 2)
        for i in range(L):
            kept = i < skip or i >= L - skip
            col = p['brass'] if kept else (p['navy'] if not ghost else 'none')
            stroke = f' stroke="{p["navy3"]}" stroke-dasharray="2 3"' if ghost else ''
            a = '' if static else f' class="s" style="animation-delay:{.15 + r * .35 + i * .018:.3f}s"'
            s.add(f'<rect x="{x0 + i * (cw + 3):.1f}" y="{yy + 10}" width="{cw:.1f}" height="22" rx="3" fill="{col}"{stroke}{a}/>')
        tot = bits if ghost else (bits * (L - 2 * skip) + 16 * 2 * skip) / L
        ratio = 16 / tot
        a = '' if static else f' class="f" style="animation-delay:{.6 + r * .35:.2f}s"'
        col = p['muted'] if ghost else p['ink']
        s.add(f'<g{a}><text x="{x1 + 26}" y="{yy + 18}" class="n" fill="{p["muted"]}">{tot:.2f} bits</text>'
              f'<text x="{x1 + 26}" y="{yy + 36}" class="big" style="font-size:20px" fill="{col}">{ratio:.2f}×</text></g>')
    ly = 312
    s.add(f'<rect x="{x0}" y="{ly - 9}" width="12" height="10" rx="2" fill="{p["brass"]}"/>')
    s.text(x0 + 18, ly, 'kept in bf16 (16 bits)', 'lab')
    s.add(f'<rect x="{x0 + 170}" y="{ly - 9}" width="12" height="10" rx="2" fill="{p["navy"]}"/>')
    s.text(x0 + 188, ly, f'turboquant_4bit_nc ({b128:g} bits at head dim 128, {b64:g} at 64)', 'lab')
    return s.render('Why vLLM TurboQuant gives 2.7x and not 3.8x')


# ------------------------------------------------------------------------------------------------- our shapes curve
def shapes(theme, static, v):
    s = Svg(theme, 426, static); p = s.p
    header(s, 'OUR SHAPES · COMPUTED FROM VLLM 0.33.0 SLOT SIZES', 'TurboQuant passes fp8 only from 13 layers',
           'KV tokens against bf16 for turboquant_4bit_nc at head dim 64 (2 KV heads), as the depth grows; fp8 stays at 2×')
    X0, X1, Y0, Y1 = 92, 760, 360, 128
    xs = lambda L: X0 + (L - 4) / (32 - 4) * (X1 - X0)
    ys = lambda r: Y0 - (r - 1) / (3.8 - 1) * (Y0 - Y1)
    for L in (4, 8, 12, 16, 20, 24, 28, 32):
        s.add(f'<line x1="{xs(L):.1f}" y1="{Y1}" x2="{xs(L):.1f}" y2="{Y0}" stroke="{p["grid"]}"/>')
        s.text(xs(L), Y0 + 18, f'{L}', 'ax', 'middle')
    for r in (1, 1.5, 2, 2.5, 3, 3.5):
        s.add(f'<line x1="{X0}" y1="{ys(r):.1f}" x2="{X1}" y2="{ys(r):.1f}" stroke="{p["grid"]}"/>')
        s.text(X0 - 10, ys(r) + 3.5, f'{r:g}×', 'ax', 'end')
    s.text(X1, Y0 + 36, 'layers →', 'ax', 'end')
    bits = vllm_bits(4, 4, 64)
    lim = 16 / bits
    s.add(f'<line x1="{X0}" y1="{ys(lim):.1f}" x2="{X1}" y2="{ys(lim):.1f}" stroke="{p["navy3"]}" stroke-dasharray="2 4"/>')
    s.text(X1 + 10, ys(lim) + 4, f'limit {lim:.2f}×', 'n', fill=p['muted'])
    s.add(f'<line x1="{X0}" y1="{ys(2):.1f}" x2="{X1}" y2="{ys(2):.1f}" stroke="{p["sage"]}" stroke-width="1.6"/>')
    s.text(X1 + 10, ys(2) + 4, 'fp8 2.00×', 'n', fill=p['sage'])
    pts = [(L, tokens_vs_bf16(bits, L)) for L in range(4, 33)]
    path = 'M' + ' L'.join(f'{xs(L):.1f},{ys(r):.1f}' for L, r in pts)
    dcls = '' if static else ' class="d" pathLength="1"'
    s.add(f'<path d="{path}" fill="none" stroke="{p["navy"]}" stroke-width="2.2" stroke-linejoin="round"{dcls}/>')
    cross = next(L for L, r in pts if r > 2)
    lo = cross - 1 + (2 - tokens_vs_bf16(bits, cross - 1)) / (tokens_vs_bf16(bits, cross) - tokens_vs_bf16(bits, cross - 1))
    cb = '' if static else ' class="b"'
    s.add(f'<circle cx="{xs(lo):.1f}" cy="{ys(2):.1f}" r="11" fill="{p["brass"]}" opacity=".35"{cb}/>'
          f'<circle cx="{xs(lo):.1f}" cy="{ys(2):.1f}" r="4" fill="{p["brass"]}"/>')
    a = '' if static else ' class="f" style="animation-delay:1.4s"'
    s.add(f'<g{a}><text x="{xs(lo) + 14:.1f}" y="{ys(2) + 22:.1f}" class="lab2" fill="{p["brass"]}">passes fp8 at {cross} layers</text></g>')
    for i, (name, L) in enumerate((('T1', 6), ('T2 · dry run 2', 8), ('T3', 10))):
        r = tokens_vs_bf16(bits, L)
        a = '' if static else f' class="f" style="animation-delay:{.9 + .2 * i:.2f}s"'
        s.add(f'<g{a}><circle cx="{xs(L):.1f}" cy="{ys(r):.1f}" r="5" fill="{p["clay"]}"/>'
              f'<text x="{xs(L) + 9:.1f}" y="{ys(r) + 17:.1f}" class="n" fill="{p["clay"]}">{name} {r:.2f}×</text></g>')
    m = v['dtypes']['turboquant_4bit_nc']['kv_ratio']
    a = '' if static else ' class="f" style="animation-delay:1.6s"'
    s.add(f'<g{a}><circle cx="{xs(28):.1f}" cy="{ys(m):.1f}" r="5.5" fill="{p["bg"]}" stroke="{p["navy"]}" stroke-width="1.6"/>'
          f'<text x="{xs(28) - 10:.1f}" y="{ys(m) - 12:.1f}" text-anchor="end" class="n" fill="{p["muted"]}">'
          f'sarvam-1, measured {m:.2f}× (head dim 128)</text></g>')
    return s.render('TurboQuant against fp8 capacity as the number of layers grows')


# ---------------------------------------------------------------------------------------------- chunked prefill idea
def prefill(theme, static, v):
    s = Svg(theme, 300, static); p = s.p
    chunk = v.get('chunk') or 32
    header(s, 'HOW THE DRIFT WAS MEASURED', 'TurboQuant is read only after the first prefill',
           'vLLM prefills a prompt on the exact K/V; the compressed cache is read by decode and by continuation chunks')
    x0, x1 = 36, 844
    s.text(x0, 136, 'One-piece prefill · 1,024 tokens', 'lab2')
    a = '' if static else ' class="g" style="animation-delay:.2s"'
    s.add(f'<rect x="{x0}" y="146" width="{x1 - x0}" height="24" rx="4" fill="{p["brass"]}" opacity=".85"{a}/>')
    s.text(x1, 136, 'exact K/V everywhere: every preset looks identical, nothing measured', 'lab', 'end')
    s.text(x0, 210, f'Chunked prefill · {1024 // chunk} pieces of {chunk} tokens', 'lab2')
    s.text(x1, 210, 'later pieces attend through the compressed cache, as decode does', 'lab', 'end')
    n = 1024 // chunk
    cw = (x1 - x0 - (n - 1) * 2) / n
    for i in range(n):
        col = p['brass'] if i == 0 else p['navy']
        a = '' if static else f' class="s" style="animation-delay:{.6 + i * .035:.3f}s"'
        s.add(f'<rect x="{x0 + i * (cw + 2):.1f}" y="220" width="{cw:.1f}" height="24" rx="3" fill="{col}"{a}/>')
    ly = 274
    s.add(f'<rect x="{x0}" y="{ly - 9}" width="12" height="10" rx="2" fill="{p["brass"]}"/>')
    s.text(x0 + 18, ly, 'exact K/V (flash-attn)', 'lab')
    s.add(f'<rect x="{x0 + 170}" y="{ly - 9}" width="12" height="10" rx="2" fill="{p["navy"]}"/>')
    s.text(x0 + 188, ly, 'TurboQuant decode kernel over the compressed cache (continuation q_len ≤ 128)', 'lab')
    return s.render('How the drift was measured')


# ------------------------------------------------------------------------------------------------- emulation findings
def findings(theme, static, v):
    s = Svg(theme, 372, static); p = s.p
    header(s, 'EMULATION · KVQ_STUDY.PY · SARVAM-1', 'Four things the emulation shows',
           'Mean KL(base ‖ quant) in nats over the documents; lower is better')
    P, S1, T = 'sarvam1_vllm_presets_T1024.json', 'sarvam1_step1_T1024.json', 'sarvam1_T1024.json'
    k4v3, k3v4 = mean(doc_values(P, 'tqn4/u3@skip2')), mean(doc_values(P, 'tqn3/u4@skip2'))
    tq4, tqp4 = mean(doc_values(S1, 'tq4')), mean(doc_values(S1, 'tqp4/tq4'))
    tqn4 = mean(doc_values(S1, 'tqn4'))
    int4, tq4b = mean(doc_values(T, 'int4')), mean(doc_values(T, 'tq4'))
    panels = [
        ('KEYS MATTER MORE · 5.45 BITS', [('4-bit keys, 3-bit values', k4v3, p['sage']), ('k3v4_nc (vLLM)', k3v4, p['navy2'])],
         f'{100 * (1 - k4v3 / k3v4):.0f}% less KL; not a vLLM preset'),
        ('QJL HURTS ATTENTION', [('TurboQuant-MSE, 4-bit', tq4, p['sage']), ('TurboQuant-prod (QJL)', tqp4, p['clay'])],
         f'{tqp4 / tq4:.1f}× the KL; vLLM leaves it out'),
        ('NORM CORRECTION IS FREE', [('tq4 + _nc', tqn4, p['sage']), ('tq4', tq4, p['navy2'])],
         f'{100 * (1 - tqn4 / tq4):.0f}% less KL at no extra bits'),
        ('ROTATION BEATS PLAIN INT4', [('TurboQuant 4-bit', tq4b, p['sage']), ('int4 absmax', int4, p['clay'])],
         f'{int4 / tq4b:.1f}× lower KL at the same 4 bits'),
    ]
    pw, ph, gx, gy, x0, y0 = 396, 118, 16, 14, 36, 116
    for i, (cap, bars, note) in enumerate(panels):
        x, y = x0 + (i % 2) * (pw + gx), y0 + (i // 2) * (ph + gy)
        top = max(b[1] for b in bars)
        a = '' if static else f' class="f" style="animation-delay:{.2 + .15 * i:.2f}s"'
        s.add(f'<g{a}><rect x="{x}" y="{y}" width="{pw}" height="{ph}" rx="12" fill="{p["card"]}" stroke="{p["line"]}"/>'
              f'<text x="{x + 18}" y="{y + 26}" class="cap">{cap}</text></g>')
        for j, (lab, val, col) in enumerate(bars):
            by = y + 44 + j * 26
            bw = (pw - 36 - 150 - 64) * val / top
            s.text(x + 18, by + 11, lab, 'lab')
            ga = '' if static else f' class="g" style="animation-delay:{.45 + .15 * i + .1 * j:.2f}s"'
            s.add(f'<rect x="{x + 168}" y="{by}" width="{bw:.1f}" height="14" rx="3" fill="{col}"{ga}/>')
            s.text(x + 168 + bw + 8, by + 11, f'{val:.4f}', 'n')
        s.text(x + 18, y + ph - 14, note, 'n', fill=p['muted'])
    return s.render('Four findings of the KV-cache emulation')


FIGS = dict(hero=hero, tradeoff=tradeoff, layers=layers, shapes=shapes, prefill=prefill, findings=findings)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--static', action='store_true', help='no animation (for previews)')
    ap.add_argument('--out', type=Path, default=HERE)
    a = ap.parse_args()
    v = json.loads((RES / 'vllm_summary.json').read_text())
    a.out.mkdir(parents=True, exist_ok=True)
    for name, fn in FIGS.items():
        for theme in PAL:
            (a.out / f'{name}-{theme}.svg').write_text(fn(theme, a.static, v), encoding='utf-8')
    print(f'wrote {2 * len(FIGS)} SVGs to {a.out}')


if __name__ == '__main__':
    main()
