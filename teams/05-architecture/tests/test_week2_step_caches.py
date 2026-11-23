"""RoPE angle cache and cached expert-weight stacks: bit-identical training to the per-call code."""

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
from week2.model import GQA, MoE, LanguageModel
from week2.optim import build_optimizer
from train_week2 import scored_ntp_loss


def legacy_rotary(self, x):
    """GQA.rotary before the angle cache."""
    inv = self.theta ** (-torch.arange(0, self.rotary_dim, 2, device=x.device).float() / self.rotary_dim)
    angle = torch.arange(x.shape[-2], device=x.device).float()[:, None] * inv[None, :]
    a, b = x.float()[..., : self.rotary_dim : 2], x.float()[..., 1 : self.rotary_dim : 2]
    rotated = torch.stack(
        (a * angle.cos() - b * angle.sin(), a * angle.sin() + b * angle.cos()), -1
    ).flatten(-2).to(x.dtype)
    return rotated if self.rotary_dim == self.d else torch.cat((rotated, x[..., self.rotary_dim :]), dim=-1)


def legacy_moe_forward(self, x, valid):
    """MoE.forward before the stack cache: stack and cast every expert weight in every call."""
    shape = x.shape
    flat = x.reshape(-1, shape[-1])
    active = valid.reshape(-1)
    with torch.autocast(device_type=x.device.type, enabled=False):
        scores = F.linear(flat.float(), self.router.weight.float()).softmax(-1)
    gates, choice = scores.topk(self.topk, dim=-1)
    gates = gates / gates.sum(-1, keepdim=True)
    gate_dense = torch.zeros_like(scores).scatter(-1, choice, gates) * active[:, None]
    Wg = torch.stack([e.gate.weight for e in self.experts])
    Wu = torch.stack([e.up.weight for e in self.experts])
    Wd = torch.stack([e.down.weight for e in self.experts])
    dt = flat.dtype
    if torch.is_autocast_enabled():
        dt = torch.get_autocast_dtype("cuda")
    xe = flat.to(dt)
    hg = torch.einsum("td,ehd->teh", xe, Wg.to(dt))
    hu = torch.einsum("td,ehd->teh", xe, Wu.to(dt))
    y = torch.einsum("teh,edh->ted", F.silu(hg) * hu, Wd.to(dt))
    output = torch.einsum("te,ted->td", gate_dense.float(), y.float()).to(flat.dtype)
    for expert in self.shared:
        output = output + expert(flat) * active[:, None]
    return output.reshape(shape)


def train(model, cfg, steps, device, micro=2, amp=False):
    """Trainer-shaped loop: microbatches normalised by the batch count, Muon + AdamW, clip, step."""
    optimizer = build_optimizer(model, cfg["optim"])
    g = torch.Generator().manual_seed(99)
    trace = []
    for _ in range(steps):
        ids = torch.randint(0, 31, (4, 8), generator=g).to(device)
        targets = torch.randint(0, 31, (4, 8), generator=g).to(device)
        targets[0, -2:] = -100
        count = int((targets != -100).sum())
        optimizer.zero_grad(set_to_none=True)
        for start in range(0, len(ids), micro):
            ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.autocast("cpu", enabled=False)
            with ctx:
                loss = scored_ntp_loss(model, ids[start : start + micro], targets[start : start + micro]) / count
            loss.backward()
            trace.append(loss.detach())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    return torch.stack(trace), {k: v.detach().clone() for k, v in model.state_dict().items()}


def pair(device="cpu"):
    torch.manual_seed(1234)
    c, rows, lengths = fixture()
    c["optim"].update(name="muon", muon_lr=0.01)
    new = LanguageModel(c["model"], rows, lengths).to(device)
    old = copy.deepcopy(new)
    return c, new, old


def run_legacy(model, *args, **kwargs):
    with patch.object(GQA, "rotary", legacy_rotary), patch.object(MoE, "forward", legacy_moe_forward):
        return train(model, *args, **kwargs)


class StepCacheTests(unittest.TestCase):
    def test_training_is_bit_identical_to_the_per_call_code(self):
        c, new, old = pair()
        ref_trace, ref_state = run_legacy(old, c, 4, "cpu")
        trace, state = train(new, c, 4, "cpu")
        self.assertTrue(torch.equal(trace, ref_trace))
        self.assertEqual(state.keys(), ref_state.keys())
        for k in ref_state:
            self.assertTrue(torch.equal(state[k], ref_state[k]), k)

    def test_stack_rebuilds_after_an_optimizer_step_and_a_reload(self):
        c, model, _ = pair()
        moe = model.blocks[0].moe
        fresh = lambda: torch.stack([e.up.weight for e in moe.experts]).detach()
        before = moe.stacked("up", torch.float32).detach().clone()
        self.assertTrue(torch.equal(before, fresh()))
        train(model, c, 1, "cpu")
        self.assertFalse(torch.equal(moe.stacked("up", torch.float32), before))
        self.assertTrue(torch.equal(moe.stacked("up", torch.float32), fresh()))
        state = copy.deepcopy(model.state_dict())
        for e in moe.experts:
            torch.nn.init.zeros_(e.up.weight)
        self.assertTrue(torch.equal(moe.stacked("up", torch.float32), torch.zeros_like(before)))
        model.load_state_dict(state)
        self.assertTrue(torch.equal(moe.stacked("up", torch.float32), fresh()))

    def test_inference_mode_then_training(self):
        c, model, _ = pair()
        ids = torch.arange(16).reshape(2, 8) % 31
        with torch.inference_mode():
            model(ids, torch.ones_like(ids, dtype=torch.bool))
        train(model, c, 1, "cpu")

    def test_stack_gradient_reaches_every_expert_weight(self):
        _, model, _ = pair()
        moe = model.blocks[0].moe
        (moe.stacked("down", torch.float32) * 2).sum().backward()
        for e in moe.experts:
            self.assertTrue(torch.equal(e.down.weight.grad, torch.full_like(e.down.weight, 2.0)))

    def test_rotary_cache_matches_per_call_angles_bitwise(self):
        c, _, _ = fixture()
        for fraction in [1.0, 0.5]:
            c["model"]["rotary_fraction"] = fraction
            attn = GQA(c["model"], 0)
            x = torch.randn(2, 4, 8, attn.d)
            with self.subTest(fraction=fraction):
                self.assertTrue(torch.equal(attn.rotary(x), legacy_rotary(attn, x)))
                self.assertTrue(torch.equal(attn.rotary(x[:, :, :5]), legacy_rotary(attn, x[:, :, :5])))

    def test_caches_add_no_state(self):
        _, model, _ = pair()
        before = set(model.state_dict())
        ids = torch.arange(16).reshape(2, 8) % 31
        model(ids, torch.ones_like(ids, dtype=torch.bool))
        self.assertEqual(set(model.state_dict()), before)


class StepCacheCompileTests(unittest.TestCase):
    """Under torch.compile both caches step aside, so they add no graph breaks or recompiles."""

    def tearDown(self):
        torch._dynamo.reset()

    def test_block_traces_as_one_graph(self):
        torch._dynamo.reset()
        _, model, _ = pair()
        ids = torch.arange(24).reshape(3, 8) % 31
        valid = torch.ones_like(ids, dtype=torch.bool)
        explained = torch._dynamo.explain(model.blocks[0])(model.embedding(ids).detach(), valid)
        self.assertEqual((explained.graph_count, explained.graph_break_count), (1, 0), explained.break_reasons)

    def test_optimizer_steps_cause_no_recompiles(self):
        torch._dynamo.reset()
        c, model, _ = pair()
        for block in model.blocks:
            block.compile(backend="eager", fullgraph=True)
        train(model, c, 1, "cpu")
        with torch._dynamo.config.patch(error_on_recompile=True):
            train(model, c, 3, "cpu")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA only")
class StepCacheCudaTests(unittest.TestCase):
    def test_bf16_training_is_bit_identical_to_the_per_call_code(self):
        torch.use_deterministic_algorithms(True)
        try:
            c, new, old = pair("cuda")
            ref_trace, ref_state = run_legacy(old, c, 3, "cuda", amp=True)
            trace, state = train(new, c, 3, "cuda", amp=True)
        finally:
            torch.use_deterministic_algorithms(False)
        self.assertTrue(torch.equal(trace, ref_trace))
        for k in ref_state:
            self.assertTrue(torch.equal(state[k], ref_state[k]), k)


if __name__ == "__main__":
    unittest.main()
