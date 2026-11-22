"""run.compile: blocks compiled in place, same parameter names, same value as eager."""

import copy
from pathlib import Path
import sys
import unittest
import torch
from torch.nn import functional as F

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM / "harness"))
sys.path.insert(0, str(TEAM / "tests"))
from test_week2_model import fixture
from week2.config import load, load_runtime, merge, validate
from week2.model import LanguageModel
from train_week2 import compile_blocks, model_hash


def loss_and_grads(model, ids, valid):
    model.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model(ids, valid).reshape(-1, 31), (ids + 1).reshape(-1) % 31)
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


class CompileConfigTests(unittest.TestCase):
    def test_arm_is_valid_and_only_changes_compile(self):
        small = load(TEAM / "configs/small/smoke_resolution.yaml")
        arm = merge(load_runtime(TEAM / "configs/arms/dryrun2_compile.yaml"), small)
        base = merge(load_runtime(TEAM / "configs/arms/dryrun2.yaml"), small)
        self.assertEqual(validate(arm), [])
        self.assertIs(arm["run"]["compile"], True)
        self.assertIs(base["run"]["compile"], False)
        self.assertEqual({**arm["run"], "compile": False}, base["run"])
        self.assertEqual(arm["model"], base["model"])

    def test_non_boolean_compile_is_refused(self):
        c, _, _ = fixture()
        c["run"]["compile"] = "inductor"
        self.assertIn("run.compile must be true or false", validate(c))


class CompileModelTests(unittest.TestCase):
    def setUp(self):
        torch._dynamo.reset()
        torch.manual_seed(1234)
        c, rows, lengths = fixture()
        self.eager = LanguageModel(c["model"], rows, lengths)
        self.compiled = compile_blocks(copy.deepcopy(self.eager))
        self.ids = torch.arange(24).reshape(3, 8) % 31
        self.valid = torch.ones_like(self.ids, dtype=torch.bool)

    def test_same_parameter_names_and_hash(self):
        self.assertEqual(list(self.eager.state_dict()), list(self.compiled.state_dict()))
        self.assertEqual(model_hash(self.eager), model_hash(self.compiled))

    def test_compiled_matches_eager_value_and_gradients(self):
        ref_loss, ref_grads = loss_and_grads(self.eager, self.ids, self.valid)
        loss, grads = loss_and_grads(self.compiled, self.ids, self.valid)
        torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
        self.assertEqual(grads.keys(), ref_grads.keys())
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-4, atol=1e-6, msg=name)


if __name__ == "__main__":
    unittest.main()
