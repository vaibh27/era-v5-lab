"""grouped_mm MoE dispatch: the dense path's value and gradients from routed experts only."""

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
from week2.model import MoE, LanguageModel


def moe_pair(device="cpu", n_embd=16, ffn_hidden=24, experts=4, top_k=2):
    torch.manual_seed(1234)
    c, _, _ = fixture()
    m = c["model"]
    m["n_embd"] = n_embd
    m["moe"].update(num_experts=experts, top_k=top_k, ffn_hidden=ffn_hidden)
    dense = MoE(m).to(device)
    m = copy.deepcopy(m)
    m["moe"]["dispatch"] = "grouped_mm"
    grouped = MoE(m).to(device)
    grouped.load_state_dict(dense.state_dict())
    return dense, grouped


def forward_backward(moe, x, valid, amp=False):
    moe.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.autocast("cpu", enabled=False)
    with ctx:
        out = moe(x, valid)
    (out.float() * torch.linspace(-1, 1, out.shape[-1], device=x.device)).sum().backward()
    grads = {n: p.grad.clone() for n, p in moe.named_parameters() if p.grad is not None}
    return out.detach().float(), x.grad, grads


class GroupedMmTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.randn(3, 8, 16, generator=torch.Generator().manual_seed(5))
        self.valid = torch.ones(3, 8, dtype=torch.bool)
        self.valid[0, 5:] = False

    def assert_matches_dense(self, dense, grouped, x, valid, **tol):
        ref, ref_dx, ref_grads = forward_backward(dense, x, valid)
        out, dx, grads = forward_backward(grouped, x, valid)
        torch.testing.assert_close(out, ref, **tol)
        torch.testing.assert_close(dx, ref_dx, **tol)
        self.assertEqual(grads.keys(), ref_grads.keys())
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], msg=name, **tol)
        return grads

    def test_matches_dense_value_and_gradients(self):
        for top_k in [1, 2]:
            dense, grouped = moe_pair(top_k=top_k)
            with self.subTest(top_k=top_k):
                self.assert_matches_dense(dense, grouped, self.x, self.valid, rtol=1e-5, atol=1e-6)

    def test_idle_experts_get_exact_zero_gradients(self):
        dense, grouped = moe_pair()
        with torch.no_grad():
            for moe in (dense, grouped):
                moe.router.weight[3] = -1e4  # with positive inputs, expert 3 is never in the top 2
        grads = self.assert_matches_dense(dense, grouped, self.x.abs(), self.valid, rtol=1e-5, atol=1e-6)
        for name in ["gate", "up", "down"]:
            self.assertTrue(torch.equal(grads[f"experts.3.{name}.weight"], torch.zeros_like(grads[f"experts.3.{name}.weight"])))

    def test_padding_tokens_get_no_routed_output(self):
        dense, grouped = moe_pair()
        with torch.no_grad():
            for e in grouped.shared:
                for p in e.parameters():
                    p.zero_()
        out, _, _ = forward_backward(grouped, self.x, self.valid)
        self.assertTrue(torch.equal(out[0, 5:], torch.zeros_like(out[0, 5:])))
        self.assertGreater(out[0, :5].abs().sum().item(), 0)

    def test_whole_model_training_tracks_dense(self):
        c, rows, lengths = fixture()
        torch.manual_seed(1234)
        dense = LanguageModel(c["model"], rows, lengths)
        c2 = copy.deepcopy(c)
        c2["model"]["moe"]["dispatch"] = "grouped_mm"
        grouped = LanguageModel(c2["model"], rows, lengths)
        grouped.load_state_dict(dense.state_dict())
        ids = torch.arange(24).reshape(3, 8) % 31
        valid = torch.ones_like(ids, dtype=torch.bool)
        losses = []
        for model in (dense, grouped):
            opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
            trace = []
            for _ in range(5):
                opt.zero_grad()
                loss = F.cross_entropy(model(ids, valid).reshape(-1, 31), (ids + 1).reshape(-1) % 31)
                loss.backward()
                opt.step()
                trace.append(loss.item())
            losses.append(trace)
        torch.testing.assert_close(torch.tensor(losses[1]), torch.tensor(losses[0]), rtol=1e-4, atol=1e-5)

    def test_arm_is_valid_and_only_changes_dispatch(self):
        small = load(TEAM / "configs/small/smoke_resolution.yaml")
        arm = merge(load_runtime(TEAM / "configs/arms/dryrun2_grouped_mm.yaml"), small)
        base = merge(load_runtime(TEAM / "configs/arms/dryrun2.yaml"), small)
        self.assertEqual(validate(arm), [])
        self.assertEqual(arm["model"]["moe"]["dispatch"], "grouped_mm")
        base["model"]["moe"]["dispatch"] = "grouped_mm"
        self.assertEqual(arm["model"], base["model"])

    def test_unknown_dispatch_is_refused(self):
        c, _, _ = fixture()
        c["model"]["moe"]["dispatch"] = "sorted"
        self.assertTrue(any("grouped_mm" in e for e in validate(c)))


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0), "CUDA SM80+ only"
)
class GroupedMmCudaTests(unittest.TestCase):
    def setUp(self):
        self.dense, self.grouped = moe_pair("cuda", n_embd=512, ffn_hidden=768, experts=16)
        g = torch.Generator(device="cuda").manual_seed(5)
        self.x = torch.randn(2, 256, 512, device="cuda", generator=g)
        self.valid = torch.ones(2, 256, dtype=torch.bool, device="cuda")
        self.valid[1, 200:] = False

    def test_bf16_close_to_dense(self):
        ref, ref_dx, ref_grads = forward_backward(self.dense, self.x, self.valid, amp=True)
        out, dx, grads = forward_backward(self.grouped, self.x, self.valid, amp=True)
        torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(dx, ref_dx, rtol=2e-2, atol=2e-2)
        for name in ref_grads:
            torch.testing.assert_close(grads[name], ref_grads[name], rtol=5e-2, atol=5e-2, msg=name)

    # torch 2.14 has the fused grouped GEMM only on SM90+; below it falls back to a loop that syncs.
    @unittest.skipUnless(torch.cuda.is_available() and torch.cuda.get_device_capability() >= (9, 0), "SM90+ only")
    def test_no_host_synchronisation(self):
        forward_backward(self.grouped, self.x, self.valid, amp=True)  # warm up allocator and kernels
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            forward_backward(self.grouped, self.x, self.valid, amp=True)
        finally:
            torch.cuda.set_sync_debug_mode("default")

    def test_bf16_repeats_bitwise_under_deterministic_mode(self):
        torch.use_deterministic_algorithms(True)
        try:
            first = forward_backward(self.grouped, self.x, self.valid, amp=True)
            second = forward_backward(self.grouped, self.x, self.valid, amp=True)
        finally:
            torch.use_deterministic_algorithms(False)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(first[1], second[1]))
        self.assertTrue(all(torch.equal(first[2][k], second[2][k]) for k in first[2]))

    def test_fp32_on_cuda_is_refused(self):
        with self.assertRaisesRegex(ValueError, "bf16"):
            self.grouped(self.x, self.valid)


if __name__ == "__main__":
    unittest.main()
