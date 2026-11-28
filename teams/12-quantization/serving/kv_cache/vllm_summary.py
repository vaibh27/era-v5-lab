#!/usr/bin/env python3
"""Summarize modal_vllm_kv.py's two outputs against the bf16 KV cache (`auto`): capacity, speed and drift.

    python vllm_summary.py vllm_capacity_speed.json vllm_drift.json --out results/vllm_summary.json

Drift, per document: ΔNLL = mean over positions of (-log p_quant(actual next token)) - (-log p_bf16(actual next token)),
and same top-1 = the share of positions where the rank-1 token is the same. Positions are aligned by index; position 0
has no log-prob and is skipped for every dtype. The summary holds these per-document numbers and counts only (no token
ids), so it can be committed; the two inputs cannot (per-position values on training text).
"""
from __future__ import annotations

import argparse, json, math
from pathlib import Path

BASE = 'auto'


def drift_rows(base: dict, other: dict) -> list[dict]:
    rows = []
    for b, q in zip(base['docs'], other['docs'], strict=True):
        if len(b['actual']) != len(q['actual']) or len(b['top1']) != len(q['top1']):
            raise SystemExit(f"{other['kv_dtype']}: a document has a different number of positions than {BASE}")
        n = len(b['actual'])
        rows.append(dict(positions=n,
                         dnll=sum(x - y for x, y in zip(b['actual'], q['actual'])) / n,
                         same_top1=sum(x == y for x, y in zip(b['top1'], q['top1'])) / n))
    return rows


def summarize(capacity_speed: list[dict], drift: list[dict]) -> dict:
    cs = {r['kv_dtype']: r for r in capacity_speed}
    dr = {r['kv_dtype']: r for r in drift}
    for name, d in (('capacity/speed', cs), ('drift', dr)):
        bad = {k: v['error'] for k, v in d.items() if 'error' in v}
        if bad or BASE not in d:
            raise SystemExit(f'{name}: missing {BASE} or failed dtypes {bad}')
    b = cs[BASE]
    out = dict(vllm=b['vllm'], gpu=b['gpu'], chunk=dr[BASE].get('chunk'), base=BASE, dtypes={})
    for k, r in cs.items():
        e = dict(kv_tokens=r['kv_tokens'], kv_ratio=round(r['kv_tokens'] / b['kv_tokens'], 3),
                 tok_per_s=r['bench']['tok_per_s'], speed_change=round(r['bench']['tok_per_s'] / b['bench']['tok_per_s'] - 1, 3))
        if k != BASE and k in dr:
            rows = drift_rows(dr[BASE], dr[k])
            d = [x['dnll'] for x in rows]
            m = sum(d) / len(d)
            se = math.sqrt(sum((x - m) ** 2 for x in d) / (len(d) - 1) / len(d)) if len(d) > 1 else None   # across documents
            e['drift'] = dict(docs=len(rows), dnll_mean=round(m, 5), dnll_se=None if se is None else round(se, 5),
                              docs_worse=sum(x > 0 for x in d),
                              same_top1=round(sum(x['same_top1'] * x['positions'] for x in rows) / sum(x['positions'] for x in rows), 4),
                              per_doc=[dict(dnll=round(x['dnll'], 5), same_top1=round(x['same_top1'], 4)) for x in rows])
        out['dtypes'][k] = e
    return out


def table(s: dict) -> str:
    lines = ['| `--kv-cache-dtype` | KV tokens | vs bf16 | tok/s | ΔNLL (s.e.) | docs worse | same top-1 |', '|---|---:|---:|---:|---:|---:|---:|']
    for k, e in s['dtypes'].items():
        d = e.get('drift')
        lines.append(f"| {k} | {e['kv_tokens']:,} | {e['kv_ratio']:.2f}× | {e['tok_per_s']:,} ({e['speed_change']:+.0%}) | "
                     + (f"{d['dnll_mean']:+.4f} ({'n/a' if d['dnll_se'] is None else format(d['dnll_se'], '.4f')}) | {d['docs_worse']}/{d['docs']} | "
                        f"{100 * d['same_top1']:.1f}% |" if d else '| | |'))
    return '\n'.join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('capacity_speed', type=Path, help="modal_vllm_kv.py run()'s output")
    ap.add_argument('drift', type=Path, help="modal_vllm_kv.py drift()'s output")
    ap.add_argument('--out', type=Path)
    a = ap.parse_args()
    s = summarize(json.loads(a.capacity_speed.read_text()), json.loads(a.drift.read_text()))
    print(table(s))
    if a.out:
        a.out.write_text(json.dumps(s, indent=1, ensure_ascii=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
