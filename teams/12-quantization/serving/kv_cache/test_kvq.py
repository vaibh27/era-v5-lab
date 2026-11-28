#!/usr/bin/env python3
"""Tests for kvq_study.py and vllm_summary.py (CPU, a few seconds):  python test_kvq.py

The TurboQuant tests need a checkout of github.com/The-School-of-AI/LLM (TSAI's V4 code, lightninglm/tqp) in
LIGHTNINGLM, and are skipped without it. No model download and no data.
"""
import math, os, sys, unittest
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import kvq_study as K  # noqa: E402
import vllm_summary as V  # noqa: E402

LIGHTNINGLM = os.environ.get('LIGHTNINGLM')


class BitsTests(unittest.TestCase):
    """Bits per number, norm and scale included, averaged over layers (skipped layers stay bf16 = 16 bits)."""
    def test_presets_on_a_28_layer_head_dim_128_model(self):
        self.assertEqual(K.kv_bits('fp8', 128, 28), 8.0)
        self.assertEqual(K.kv_bits('tqn4/u4', 128, 28), 4.1875)          # keys 4 + 16/128, values 4 + 32/128
        self.assertEqual(K.kv_bits('tqn4/u4@skip2', 128, 28), 5.875)     # turboquant_4bit_nc as vLLM runs it
        self.assertEqual(K.kv_bits('tqn3/u4@skip2', 128, 28), 5.4464)    # turboquant_k3v4_nc
        self.assertEqual(K.kv_bits('tqn3/u3@skip2', 128, 28), 5.0179)    # turboquant_3bit_nc
        self.assertEqual(K.kv_bits('none', 128, 28), 16.0)

    def test_capacity_ratios_follow(self):
        for m, ratio in (('tqn4/u4@skip2', 2.72), ('tqn3/u4@skip2', 2.94), ('tqn3/u3@skip2', 3.19)):
            self.assertAlmostEqual(16 / K.kv_bits(m, 128, 28), ratio, places=2)


class QuantizerTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(4, 2, 16, 128, generator=torch.Generator().manual_seed(0))

    def test_none_is_identity_and_dtype_is_kept(self):
        self.assertIs(K.fake_quant(self.x, 'none'), self.x)
        self.assertEqual(K.fake_quant(self.x.half(), 'u4').dtype, torch.float16)

    def test_uniform_error_is_at_most_half_a_step(self):
        for b in (2, 3, 4):
            y = K.fake_quant(self.x, f'u{b}')
            step = (self.x.amax(-1, keepdim=True) - self.x.amin(-1, keepdim=True)) / (2 ** b - 1)
            self.assertTrue(((y - self.x).abs() <= step / 2 + 1e-2 * step).all())
            self.assertLessEqual(len(torch.unique(y[0, 0, 0])), 2 ** b)

    def test_fp8_is_e4m3(self):
        y = K.fake_quant(self.x, 'fp8')
        self.assertTrue(torch.equal(y, self.x.to(torch.float8_e4m3fn).float()))

    def test_int4_has_at_most_16_levels(self):
        self.assertLessEqual(len(torch.unique(K.fake_quant(self.x, 'int4')[0, 0, 0])), 16)


@unittest.skipUnless(LIGHTNINGLM, 'set LIGHTNINGLM to a checkout of github.com/The-School-of-AI/LLM')
class TurboQuantTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        K.use_lightninglm(LIGHTNINGLM)
        cls.x = torch.randn(4096, 128, generator=torch.Generator().manual_seed(1))

    def nmse(self, m):
        y = K.fake_quant(self.x, m)
        return ((y - self.x).pow(2).sum(-1) / self.x.pow(2).sum(-1)).mean().item()

    def test_normalized_mse_is_the_papers(self):
        # TurboQuant paper, MSE quantizer on the unit sphere: about 0.117 (2 bits), 0.034 (3), 0.009 (4)
        for b, want in ((2, 0.117), (3, 0.034), (4, 0.0093)):
            self.assertAlmostEqual(self.nmse(f'tq{b}') / want, 1.0, delta=0.08)
        self.assertLess(self.nmse('tq4'), self.nmse('int4'))

    def test_norm_correction_keeps_the_stored_norm(self):
        y = K.fake_quant(self.x, 'tqn4')
        self.assertTrue(torch.allclose(y.norm(dim=-1), self.x.norm(dim=-1).half().float(), rtol=1e-5))
        self.assertTrue(K.fake_quant(self.x, 'tq4').norm(dim=-1).mean() < self.x.norm(dim=-1).mean())   # tq shrinks

    def test_qjl_inner_products_are_unbiased(self):
        q = torch.randn(4096, 128, generator=torch.Generator().manual_seed(2))
        ip, ipq = (q * self.x).sum(-1), (q * K.fake_quant(self.x, 'tqp3')).sum(-1)
        slope = (ip * ipq).sum() / (ip * ip).sum()
        self.assertAlmostEqual(slope.item(), 1.0, delta=0.012)   # without QJL's sqrt(pi/2) scale the slope is about 0.98


class MetricTests(unittest.TestCase):
    """kvq_study.metrics on toy logits: KL direction, top-1 share and the next-token alignment of the NLL."""
    def test_identical_logits(self):
        x = torch.randn(5, 7, generator=torch.Generator().manual_seed(3))
        self.assertEqual(K.metrics(x, x.clone(), [0, 1, 2, 3, 4]), dict(kl=0.0, top1=1.0, dnll=0.0))

    def test_kl_is_base_against_quant_and_nll_uses_the_next_token(self):
        base = torch.log(torch.tensor([[0.5, 0.5], [0.9, 0.1]]))
        q = torch.log(torch.tensor([[0.9, 0.1], [0.9, 0.1]]))
        m = K.metrics(base, q, [1, 1])        # position 0 predicts token ids[1] = 1
        kl0 = 0.5 * math.log(0.5 / 0.9) + 0.5 * math.log(0.5 / 0.1)    # KL(base || q) at position 0; position 1 is 0
        self.assertAlmostEqual(m['kl'], kl0 / 2, places=6)
        self.assertEqual(m['top1'], 1.0)      # base's tie at position 0 resolves to index 0, as q's argmax does
        flipped = torch.log(torch.tensor([[0.9, 0.1], [0.1, 0.9]]))
        self.assertEqual(K.metrics(base, flipped, [1, 1])['top1'], 0.5)
        self.assertAlmostEqual(m['dnll'], -math.log(0.1) + math.log(0.5), places=5)
        self.assertAlmostEqual(K.metrics(base, q, [0, 0])['dnll'], -math.log(0.9) + math.log(0.5), places=5)


class SkipTests(unittest.TestCase):
    """@skipN leaves the first and last N layers unquantized; keys and values take their own method."""
    def test_skip_and_split(self):
        from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
        seen = []
        keep = ALL_ATTENTION_FUNCTIONS['sdpa']
        ALL_ATTENTION_FUNCTIONS['sdpa'] = lambda module, q, k, v, mask, **kw: seen.append((module.layer_idx, k, v))
        try:
            K.install()
            attn = ALL_ATTENTION_FUNCTIONS['kvq']
        finally:
            ALL_ATTENTION_FUNCTIONS['sdpa'] = keep
        k, v = torch.randn(1, 2, 8, 128), torch.randn(1, 2, 8, 128)
        K.STATE.update(method='fp8/u2@skip1', n_layers=4)
        try:
            for i in range(4):
                attn(type('M', (), {'layer_idx': i})(), None, k, v, None)
        finally:
            K.STATE.update(method='none')
        for i, kk, vv in seen:
            if i in (0, 3):
                self.assertIs(kk, k); self.assertIs(vv, v)
            else:
                self.assertTrue(torch.equal(kk, K.fake_quant(k, 'fp8'))); self.assertTrue(torch.equal(vv, K.fake_quant(v, 'u2')))


class SummaryTests(unittest.TestCase):
    def run_pair(self, actual_q, top_q):
        base = dict(kv_dtype='auto', vllm='0.33.0', gpu='L4', kv_tokens=100, bench=dict(tok_per_s=10.0), chunk=32,
                    docs=[dict(actual=[-1.0, -2.0], top1=[5, 6])])
        q = dict(kv_dtype='q', kv_tokens=250, bench=dict(tok_per_s=9.0), docs=[dict(actual=actual_q, top1=top_q)])
        return V.summarize([base, q], [base, q])

    def test_drift_capacity_and_speed(self):
        s = self.run_pair([-1.5, -2.0], [5, 7])['dtypes']['q']
        self.assertEqual((s['kv_ratio'], s['speed_change']), (2.5, -0.1))
        self.assertEqual((s['drift']['dnll_mean'], s['drift']['same_top1'], s['drift']['docs_worse']), (0.25, 0.5, 1))

    def test_two_documents_standard_error_and_a_tie(self):
        base = dict(kv_dtype='auto', vllm='0.33.0', gpu='L4', kv_tokens=100, bench=dict(tok_per_s=10.0), chunk=32,
                    docs=[dict(actual=[-1.0, -1.0], top1=[1, 2]), dict(actual=[-1.0, -1.0], top1=[1, 2])])
        q = dict(kv_dtype='q', kv_tokens=200, bench=dict(tok_per_s=10.0),
                 docs=[dict(actual=[-1.2, -1.2], top1=[1, 2]), dict(actual=[-1.0, -1.0], top1=[1, 3])])
        d = V.summarize([base, q], [base, q])['dtypes']['q']['drift']
        self.assertAlmostEqual(d['dnll_mean'], 0.1, places=6)
        self.assertAlmostEqual(d['dnll_se'], 0.1, places=6)    # sd 0.1414 across the 2 documents / sqrt(2)
        self.assertEqual((d['docs_worse'], d['same_top1']), (1, 0.75))   # the tied document is not counted as worse

    def test_unequal_positions_and_failed_runs_stop(self):
        with self.assertRaises(SystemExit):
            self.run_pair([-1.0], [5])
        base = dict(kv_dtype='auto', error='boom')
        with self.assertRaises(SystemExit):
            V.summarize([base], [base])


if __name__ == '__main__':
    unittest.main(verbosity=1)
