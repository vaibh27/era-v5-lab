"""norm_impl: fused (torch F.rms_norm) and swiglu_impl: triton (copied Team 6 kernel) against the torch ops."""

import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch
import torch
from torch.nn import functional as F

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM / "harness"))
sys.path.insert(0, str(TEAM / "tests"))
from test_week2_model import fixture
from week2.config import load, load_runtime, merge, validate
from week2.model import LanguageModel, RMSNorm
from week2 import swiglu_triton


def model_pair(device="cpu", **impl):
    torch.manual_seed(1234)
    c, rows, lengths = fixture()
    ref = LanguageModel(c["model"], rows, lengths).to(device)
    c2 = copy.deepcopy(c)
    c2["model"].update(impl)
    new = LanguageModel(c2["model"], rows, lengths).to(device)
    new.load_state_dict(ref.state_dict())
    return ref, new


def loss_and_grads(model, ids, valid, amp=False):
    model.zero_grad(set_to_none=True)
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.autocast("cpu", enabled=False)
    with ctx:
        logits = model(ids, valid)
    loss = F.cross_entropy(logits.float().reshape(-1, 31), (ids + 1).reshape(-1) % 31)
    loss.backward()
    return loss.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


class FusedConfigTests(unittest.TestCase):
    def test_arm_is_valid_and_only_changes_kernels(self):
        small = load(TEAM / "configs/small/smoke_resolution.yaml")
        arm = merge(load_runtime(TEAM / "configs/arms/dryrun2_fused.yaml"), small)
        base = merge(load_runtime(TEAM / "configs/arms/dryrun2.yaml"), small)
        self.assertEqual(validate(arm), [])
        self.assertEqual((arm["model"]["norm_impl"], arm["model"]["swiglu_impl"]), ("fused", "triton"))
        self.assertEqual({**base["model"], "norm_impl": "fused", "swiglu_impl": "triton"}, arm["model"])

    def test_unknown_kernel_choices_are_refused(self):
        c, _, _ = fixture()
        c["model"].update(norm_impl="apex", swiglu_impl="cuda")
        errors = validate(c)
        self.assertIn("norm_impl must be torch or fused", errors)
        self.assertIn("swiglu_impl must be torch or triton", errors)

    def test_defaults_stay_torch(self):
        c, rows, lengths = fixture()
        model = LanguageModel(c["model"], rows, lengths)
        self.assertFalse(model.norm.fused)
        self.assertFalse(model.blocks[0].moe.triton)


class FusedCpuTests(unittest.TestCase):
    def test_fused_rms_norm_forward_is_bit_identical_on_cpu(self):
        for dtype in [torch.float32, torch.bfloat16]:
            x = torch.randn(4, 8, 64, dtype=dtype)
            ref, fused = RMSNorm(64, 1e-5), RMSNorm(64, 1e-5, fused=True)
            torch.nn.init.uniform_(ref.weight, 0.5, 1.5)
            fused.load_state_dict(ref.state_dict())
            with self.subTest(dtype=dtype):
                self.assertTrue(torch.equal(fused(x), ref(x)))

    def test_fused_model_matches_torch_ops(self):
        ref, new = model_pair(norm_impl="fused", swiglu_impl="triton")
        ids = torch.arange(24).reshape(3, 8) % 31
        valid = torch.ones_like(ids, dtype=torch.bool)
        ref_loss, ref_grads = loss_and_grads(ref, ids, valid)
        loss, grads = loss_and_grads(new, ids, valid)
        self.assertTrue(torch.equal(loss, ref_loss))  # CPU: F.rms_norm forward is exact, SwiGLU falls back to torch
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-5, atol=1e-7, msg=name)

    def test_swiglu_falls_back_to_torch_without_cuda(self):
        g, u = torch.randn(5, 7), torch.randn(5, 7)
        self.assertFalse(swiglu_triton.supported(g, u))
        self.assertTrue(torch.equal(swiglu_triton.swiglu(g, u), F.silu(g) * u))


class FusedCompileTests(unittest.TestCase):
    """Under torch.compile the SwiGLU wrapper uses the torch ops, so the kernel adds no graph breaks."""

    def tearDown(self):
        torch._dynamo.reset()

    def test_block_traces_as_one_graph_with_the_kernel_enabled(self):
        torch._dynamo.reset()
        _, new = model_pair(norm_impl="fused", swiglu_impl="triton")
        ids = torch.arange(24).reshape(3, 8) % 31
        valid = torch.ones_like(ids, dtype=torch.bool)
        # Pretend the kernel is available (CPU has no Triton path), so tracing would reach it without the guard.
        with patch.object(swiglu_triton, "supported", lambda g, u: True):
            explained = torch._dynamo.explain(new.blocks[0])(new.embedding(ids).detach(), valid)
        self.assertEqual((explained.graph_count, explained.graph_break_count), (1, 0), explained.break_reasons)


@unittest.skipUnless(torch.cuda.is_available() and swiglu_triton.triton is not None, "CUDA + Triton only")
class FusedCudaTests(unittest.TestCase):
    def test_swiglu_kernel_close_to_torch_on_3d_expert_activations(self):
        g = torch.randn(256, 16, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        u = torch.randn(256, 16, 768, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        dh = torch.randn_like(g)
        ref = F.silu(g.float()) * u.float()
        ref_dg, ref_du = torch.autograd.grad(ref, (g, u), dh.float())
        out = swiglu_triton.swiglu(g, u)
        dg, du = torch.autograd.grad(out, (g, u), dh)
        torch.testing.assert_close(out.float(), ref, rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(dg.float(), ref_dg.float(), rtol=1e-2, atol=1e-2)
        torch.testing.assert_close(du.float(), ref_du.float(), rtol=1e-2, atol=1e-2)

    def test_bf16_model_close_to_torch_ops(self):
        ref, new = model_pair("cuda", norm_impl="fused", swiglu_impl="triton")
        ids = (torch.arange(24).reshape(3, 8) % 31).cuda()
        valid = torch.ones_like(ids, dtype=torch.bool)
        ref_loss, _ = loss_and_grads(ref, ids, valid, amp=True)
        loss, _ = loss_and_grads(new, ids, valid, amp=True)
        torch.testing.assert_close(loss, ref_loss, rtol=1e-2, atol=1e-2)

    def test_compiled_block_has_no_graph_breaks_on_cuda(self):
        torch._dynamo.reset()
        try:
            _, new = model_pair("cuda", norm_impl="fused", swiglu_impl="triton")
            ids = (torch.arange(24).reshape(3, 8) % 31).cuda()
            valid = torch.ones_like(ids, dtype=torch.bool)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                explained = torch._dynamo.explain(new.blocks[0])(new.embedding(ids).detach(), valid)
        finally:
            torch._dynamo.reset()
        self.assertEqual(explained.graph_break_count, 0, explained.break_reasons)

    def test_bf16_model_repeats_bitwise_under_deterministic_mode(self):
        torch.use_deterministic_algorithms(True)
        try:
            _, new = model_pair("cuda", norm_impl="fused", swiglu_impl="triton")
            ids = (torch.arange(24).reshape(3, 8) % 31).cuda()
            valid = torch.ones_like(ids, dtype=torch.bool)
            first = loss_and_grads(new, ids, valid, amp=True)
            second = loss_and_grads(new, ids, valid, amp=True)
        finally:
            torch.use_deterministic_algorithms(False)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(all(torch.equal(first[1][k], second[1][k]) for k in first[1]))


if __name__ == "__main__":
    unittest.main()
