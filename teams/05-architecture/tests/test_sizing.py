"""The closed-form parameter count in harness/sizing.py matches the real model, and the grid stays in 30B-120B."""

from pathlib import Path
import importlib.util
import sys
import unittest

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM / "harness"))
import sizing


class SizingTests(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("torch"), "needs torch")
    def test_formula_equals_model(self):
        # T1/T2/T3 with the dense head, dry run 2 with the KAS head: exact, parameter for parameter.
        for arm, _, formula, real in sizing.check():
            with self.subTest(arm=arm):
                self.assertEqual(formula, real)

    def test_instructor_sizes(self):
        # The instructor totals use tied embeddings (one V x d table, no KE2 projection).
        for name, t in sizing.T_SIZES.items():
            s = sizing.spec(t["d_model"], t["n_layer"], t["n_head"], t["n_kv_head"], t["ffn_hidden"],
                            t["num_experts"], head="tied")
            with self.subTest(size=name):
                self.assertAlmostEqual(sizing.count(s)["total"] / 1e6, t["total"] / 1e6, delta=0.5)

    def test_active_uses_top_k_plus_shared(self):
        s = sizing.big(4096, 32)
        c = sizing.count(s)
        expert = 3 * s["d_model"] * s["ffn_hidden"]
        self.assertEqual(c["total"] - c["active"], s["n_layer"] * (16 - 2) * expert)

    def test_mtp_adds_one_dense_head(self):
        a, b = sizing.big(4096, 32), sizing.big(4096, 32, mtp=True)
        self.assertEqual(sizing.count(b)["total"] - sizing.count(a)["total"], sizing.V * 4096)

    def test_grid_in_range(self):
        rows = sizing.grid()
        self.assertTrue(rows)
        for _, _, c in rows:
            self.assertTrue(30e9 <= c["total"] <= 120e9)

    def test_bad_shapes_rejected(self):
        with self.assertRaises(ValueError):
            sizing.spec(1000, 8, 7, 1, 1536)
        with self.assertRaises(ValueError):
            sizing.spec(1024, 8, 8, 3, 1536)


if __name__ == "__main__":
    unittest.main()
