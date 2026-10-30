"""Optimizer parameter assignment, numerical updates, state restore, and scaler checks."""
import copy
import socket
import pytest
import torch

from models.config import ModelConfig
from models.gpt import GPT
from models.optim import Muon, MuonAdamW, Optimizers, build_optimizer

parametrize = pytest.mark.parametrize
skipif = pytest.mark.skipif


@parametrize("embedding", ["table", "ke1"])
def test_optimizer_ownership_and_norm_gains(embedding):
    # Small vocabulary here is a unit fixture, never a performance configuration.
    cfg = ModelConfig(vocab_size=16, block_size=4, n_layer=1, n_head=2, n_embd=8,
                      embedding=embedding, ffn="moe",
                      moe=dict(num_experts=2, top_k=1, num_shared=1))
    from models.ke1 import token_bytes_random
    model = GPT(cfg, token_bytes_random(16) if embedding == "ke1" else None)
    opt = build_optimizer(model, dict(name="muon", lr=0.001, muon_lr=0.02, weight_decay=0.01))
    optimizers = dict(opt.named)
    assert isinstance(optimizers["opt.muon"], Muon)
    assert isinstance(optimizers["opt.adamw"], torch.optim.AdamW)
    assert optimizers["opt.adamw"].defaults["betas"] == (0.9, 0.95)
    assert optimizers["opt.adamw"].defaults["fused"] == next(model.parameters()).is_cuda
    groups = {name: {id(p) for g in o.param_groups for p in g["params"]} for name, o in opt.named}
    muon, adamw = groups["opt.muon"], groups["opt.adamw"]
    assert not muon & adamw
    assert muon | adamw == {id(p) for p in model.parameters() if p.requires_grad}
    for name, p in model.named_parameters():
        expected = muon if name.startswith("blocks.") and p.ndim == 2 else adamw
        assert id(p) in expected, name
    assert id(model.head.weight) in adamw
    norms = [m for m in model.modules() if isinstance(m, torch.nn.LayerNorm)]
    assert norms and all(m.weight.requires_grad and id(m.weight) in adamw for m in norms)
    adamw_optimizer = optimizers["opt.adamw"]
    assert all(group["weight_decay"] == 0.0 for group in adamw_optimizer.param_groups
               if any(p.ndim < 2 for p in group["params"]))
    assert all(all(p.ndim < 2 for p in group["params"])
               for group in adamw_optimizer.param_groups if group["weight_decay"] == 0.0)
    model(torch.tensor([[1, 2, 3, 4]]), torch.tensor([[2, 3, 4, 5]]))[1].backward()
    assert all(m.weight.grad is not None for m in norms)
    opt.step()
    assert all(torch.isfinite(p).all() for p in model.parameters())


def test_pure_adamw_owns_every_parameter_and_uses_expected_settings():
    cfg = ModelConfig(vocab_size=16, block_size=4, n_layer=1, n_head=2, n_embd=8)
    model = GPT(cfg)
    opt = build_optimizer(model, dict(name="adamw", lr=0.001, weight_decay=0.1))
    assert len(opt.named) == 1
    name, adamw = opt.named[0]
    assert name == "opt.adamw"
    assert isinstance(adamw, torch.optim.AdamW)
    assert adamw.defaults["betas"] == (0.9, 0.95)
    assert adamw.defaults["fused"] == next(model.parameters()).is_cuda
    assigned = {id(p) for group in adamw.param_groups for p in group["params"]}
    assert assigned == {id(p) for p in model.parameters() if p.requires_grad}
    for group in adamw.param_groups:
        has_bias_or_gain = any(p.ndim < 2 for p in group["params"])
        assert group["weight_decay"] == (0.0 if has_bias_or_gain else 0.1)


def test_muon_update_and_missing_gradient_skip():
    p = torch.nn.Parameter(torch.tensor([[2.0]]))
    opt = Muon([p], lr=0.1, momentum=0.9, ns_steps=0, weight_decay=0.1)
    p.grad = torch.tensor([[3.0]])
    opt.step()
    # NS with zero iterations returns the normalized positive scalar update (+1).
    assert torch.allclose(p, torch.tensor([[1.88]]), atol=1e-6)
    old_param = p.detach().clone()
    old_momentum = opt.state[p]["buf"].clone()
    p.grad = None
    opt.step()
    assert torch.equal(p, old_param)
    assert torch.equal(opt.state[p]["buf"], old_momentum)

    adamw_p = torch.nn.Parameter(torch.tensor([[2.0]]))
    adamw = torch.optim.AdamW([adamw_p], lr=0.1, weight_decay=0.1)
    adamw_p.grad = torch.ones_like(adamw_p)
    adamw.step()
    after_used_step = adamw_p.detach().clone()
    adamw_p.grad = None
    adamw.step()
    assert torch.equal(adamw_p, after_used_step)


def _reference_muon_step(param, grad, momentum_buffer, lr, momentum, weight_decay, ns_steps=5):
    """Independent small-tensor Muon calculation, including the production bf16 NS path on CPU."""
    if momentum_buffer is None:
        momentum_buffer = torch.zeros_like(grad)
    momentum_buffer = momentum_buffer.clone().lerp(grad, 1 - momentum)
    nesterov_grad = grad.lerp(momentum_buffer, momentum)

    # Match only the documented compute precision; implement the NS iteration here rather than
    # calling zeropower_via_newtonschulz5 so this can detect mistakes in that implementation.
    x = nesterov_grad.to(torch.bfloat16)
    transposed = x.shape[0] > x.shape[1]
    if transposed:
        x = x.T
    x = x / (x.norm() + 1e-7)
    a, b, c = 3.4445, -4.7750, 2.0315
    for _ in range(ns_steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    if transposed:
        x = x.T

    scale = max(1.0, param.shape[0] / param.shape[1]) ** 0.5
    expected = param * (1 - lr * weight_decay)
    expected = expected - x.to(param.dtype) * (lr * scale)
    return expected, momentum_buffer


@parametrize("shape", [(2, 3), (3, 2)])
def test_muon_five_step_update_matches_independent_reference(shape):
    lr, momentum, weight_decay = 0.03, 0.8, 0.04
    initial = torch.arange(1, shape[0] * shape[1] + 1, dtype=torch.float32).reshape(shape) / 3
    param = torch.nn.Parameter(initial.clone())
    opt = Muon([param], lr=lr, momentum=momentum, ns_steps=5, weight_decay=weight_decay)
    expected = initial.clone()
    ref_buffer = None

    for step in range(3):
        grad = torch.arange(1, param.numel() + 1, dtype=param.dtype).reshape(shape)
        grad = grad * (0.07 * (step + 1)) - 0.2
        expected, ref_buffer = _reference_muon_step(
            expected, grad, ref_buffer, lr, momentum, weight_decay, ns_steps=5)
        param.grad = grad.clone()
        opt.step()
        torch.testing.assert_close(param, expected, rtol=0, atol=1e-6)
        torch.testing.assert_close(opt.state[param]["buf"], ref_buffer, rtol=0, atol=1e-7)


def test_pure_adamw_two_steps_match_independent_reference():
    class TinyParams(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.tensor([[1.5, -0.75], [0.25, 2.0]]))
            self.bias = torch.nn.Parameter(torch.tensor([0.5, -1.0]))

    model = TinyParams()
    opt = build_optimizer(model, dict(name="adamw", lr=0.02, weight_decay=0.1))
    adamw = dict(opt.named)["opt.adamw"]
    params = {"matrix": model.matrix, "bias": model.bias}
    expected = {name: p.detach().clone() for name, p in params.items()}
    moments = {name: (torch.zeros_like(p), torch.zeros_like(p)) for name, p in params.items()}
    decay = {"matrix": 0.1, "bias": 0.0}
    beta1, beta2, eps, lr = 0.9, 0.95, adamw.defaults["eps"], 0.02

    for step in range(1, 3):
        grads = {
            "matrix": torch.tensor([[0.2, -0.1], [0.3, 0.4]]) * step,
            "bias": torch.tensor([-0.25, 0.15]) * step,
        }
        for name, param in params.items():
            param.grad = grads[name].clone()
            m, v = moments[name]
            m.mul_(beta1).add_(grads[name], alpha=1 - beta1)
            v.mul_(beta2).addcmul_(grads[name], grads[name], value=1 - beta2)
            m_hat = m / (1 - beta1 ** step)
            v_hat = v / (1 - beta2 ** step)
            expected[name].mul_(1 - lr * decay[name])
            expected[name].addcdiv_(m_hat, v_hat.sqrt().add(eps), value=-lr)
        opt.step()
        for name, param in params.items():
            torch.testing.assert_close(param, expected[name], rtol=1e-6, atol=1e-7)


def _reference_polar_muon_step(param, grad, momentum_buffer, second_buffer, lr, weight_decay):
    """Small CPU reference for Polar Express Muon; does not call production helpers."""
    momentum = 0.95
    if momentum_buffer is None:
        momentum_buffer = torch.zeros_like(grad)
        rows, cols = param.shape
        second_buffer = torch.zeros((rows, 1) if rows >= cols else (1, cols))
    momentum_buffer = momentum_buffer.clone().lerp(grad, 1 - momentum)
    x = grad.lerp(momentum_buffer, momentum).float()
    rows, cols = x.shape
    target = x.norm() / rows ** 0.5
    x = x * (target / x.norm(dim=-1, keepdim=True).clamp_min(1e-6))
    x = x / (x.norm() * 1.01 + 1e-6)
    coeffs = (
        (8.156554524902461, -22.48329292557795, 15.878769915207462),
        (4.042929935166739, -2.808917465908714, 0.5000178451051316),
        (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
        (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
        (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
    )
    if rows > cols:
        for a, b, c in coeffs:
            gram = x.T @ x
            x = a * x + x @ (b * gram + c * (gram @ gram))
    else:
        for a, b, c in coeffs:
            gram = x @ x.T
            x = a * x + (b * gram + c * (gram @ gram)) @ x
    x = x * (min(rows, cols) ** 0.5 / x.norm().clamp_min(1e-6))
    red_dim = -1 if rows >= cols else -2
    variance = x.square().mean(dim=red_dim, keepdim=True)
    second_buffer = second_buffer.clone().lerp(variance, 1 - 0.95)
    step_size = second_buffer.clamp_min(1e-10).rsqrt()
    reduction_size = x.size(red_dim)
    norm_before = (variance.sum() * reduction_size).sqrt()
    norm_after = ((variance * reduction_size) * step_size.square()).sum().sqrt()
    x = x * step_size * (norm_before / norm_after.clamp_min(1e-10))
    lr *= max(1.0, rows / cols) ** 0.5
    expected = param - lr * x - lr * weight_decay * param * (x * param >= 0)
    return expected, momentum_buffer, second_buffer


def test_polar_muon_adamw_is_one_optimizer_and_matches_reference():
    class TinyBlock(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 2)

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = torch.nn.Embedding(4, 3)
            self.blocks = torch.nn.ModuleList([TinyBlock()])
            self.norm = torch.nn.LayerNorm(2)
            self.head = torch.nn.Linear(2, 4, bias=False)

    model = TinyModel()
    lr, muon_lr, weight_decay = 0.01, 0.02, 0.03
    opt = build_optimizer(model, dict(name="polar_muon_adamw", lr=lr, muon_lr=muon_lr,
                                      weight_decay=weight_decay))
    assert len(opt.named) == 1
    name, combined = opt.named[0]
    assert name == "opt.polar_muon_adamw"
    assert isinstance(combined, MuonAdamW)
    muon_group = next(group for group in combined.param_groups if group["kind"] == "muon")
    muon_ids = {id(p) for p in muon_group["params"]}
    expected_muon = {id(model.blocks[0].linear.weight)}
    assert muon_ids == expected_muon
    assert id(model.embedding.weight) not in muon_ids
    assert id(model.head.weight) not in muon_ids
    assert id(model.norm.weight) not in muon_ids

    torch.manual_seed(31)
    refs = {}
    ref_momentum = ref_second = None
    for i, param in enumerate(model.parameters()):
        grad = torch.linspace(-0.2, 0.3, param.numel()).reshape_as(param) + i * 0.01
        param.grad = grad.clone()
        if id(param) in muon_ids:
            refs[id(param)], ref_momentum, ref_second = _reference_polar_muon_step(
                param.detach().clone(), grad, ref_momentum, ref_second,
                muon_lr, weight_decay)
        else:
            decay = weight_decay if param.ndim >= 2 else 0.0
            ref_param = torch.nn.Parameter(param.detach().clone())
            ref_opt = torch.optim.AdamW([ref_param], lr=lr, betas=(0.9, 0.95),
                                        weight_decay=decay, foreach=False, fused=False)
            ref_param.grad = grad.clone()
            ref_opt.step()
            refs[id(param)] = ref_param.detach().clone()

    opt.step()
    for param in model.parameters():
        torch.testing.assert_close(param, refs[id(param)], rtol=2e-5, atol=2e-6)


@parametrize("shape", [(2, 3), (3, 2)])
def test_polar_muon_rectangular_scaling_matches_reference(shape):
    p = torch.nn.Parameter(torch.arange(1, shape[0] * shape[1] + 1, dtype=torch.float32).reshape(shape) / 5)
    grad = torch.linspace(-0.3, 0.4, p.numel()).reshape(shape)
    lr, weight_decay = 0.015, 0.02
    expected, _, _ = _reference_polar_muon_step(
        p.detach().clone(), grad, None, None, lr, weight_decay)
    opt = MuonAdamW([{
        "params": [p], "kind": "muon", "lr": lr, "momentum": 0.95,
        "beta2": 0.95, "ns_steps": 5, "weight_decay": weight_decay,
    }])
    p.grad = grad.clone()
    opt.step()
    torch.testing.assert_close(p, expected, rtol=2e-5, atol=2e-6)


def test_hybrid_optimizer_state_dict_resumes_identically():
    cfg = ModelConfig(vocab_size=16, block_size=4, n_layer=1, n_head=2, n_embd=8,
                      ffn="moe", moe=dict(num_experts=2, top_k=1, num_shared=1))
    model = GPT(cfg)
    optim = build_optimizer(model, dict(name="muon", lr=0.001, muon_lr=0.02, weight_decay=0.01))

    def set_grads(module, multiplier):
        for i, param in enumerate(module.parameters()):
            param.grad = torch.full_like(param, multiplier * (i + 1) / 100)

    set_grads(model, 1.0)
    optim.step()
    saved_optimizer_state = copy.deepcopy(optim.state_dict())
    resumed_model = GPT(cfg)
    resumed_model.load_state_dict(model.state_dict())
    resumed_optim = build_optimizer(
        resumed_model, dict(name="muon", lr=0.001, muon_lr=0.02, weight_decay=0.01))
    resumed_optim.load_state_dict(saved_optimizer_state)

    set_grads(model, 2.0)
    set_grads(resumed_model, 2.0)
    optim.step()
    resumed_optim.step()
    for live, resumed in zip(model.parameters(), resumed_model.parameters()):
        torch.testing.assert_close(resumed, live, rtol=0, atol=1e-7)


@skipif(not torch.cuda.is_available(), reason="GradScaler overflow behavior requires CUDA")
def test_grad_scaler_skips_both_hybrid_optimizers_on_overflow():
    muon_param = torch.nn.Parameter(torch.tensor([[1.0, 2.0], [3.0, 4.0]], device="cuda"))
    adamw_param = torch.nn.Parameter(torch.tensor([2.0, -1.0], device="cuda"))
    optim = Optimizers([
        ("opt.muon", Muon([muon_param], lr=0.1, weight_decay=0.1)),
        ("opt.adamw", torch.optim.AdamW([adamw_param], lr=0.1, weight_decay=0.1, fused=True)),
    ])
    scaler = torch.amp.GradScaler("cuda", init_scale=8.0)
    before = (muon_param.detach().clone(), adamw_param.detach().clone())
    scaler.scale(muon_param.sum() + adamw_param.sum()).backward()
    muon_param.grad[0, 0] = torch.inf
    adamw_param.grad[0] = torch.inf

    optim.unscale(scaler)
    optim.step(scaler)
    scaler.update()

    torch.testing.assert_close(muon_param, before[0], rtol=0, atol=0)
    torch.testing.assert_close(adamw_param, before[1], rtol=0, atol=0)
    assert scaler.get_scale() < 8.0


def test_cosine_learning_rate_schedule():
    p = torch.nn.Parameter(torch.ones(()))
    optimizers = Optimizers([("opt", torch.optim.SGD([p], lr=0.2))])
    optimizers.set_lr(0, 5, schedule="cosine", warmup_steps=2, min_lr_ratio=0.1)
    assert optimizers.named[0][1].param_groups[0]["lr"] == pytest.approx(0.1)
    optimizers.set_lr(1, 5, schedule="cosine", warmup_steps=2, min_lr_ratio=0.1)
    assert optimizers.named[0][1].param_groups[0]["lr"] == pytest.approx(0.2)
    optimizers.set_lr(4, 5, schedule="cosine", warmup_steps=2, min_lr_ratio=0.1)
    assert optimizers.named[0][1].param_groups[0]["lr"] == pytest.approx(0.02)


class _DDPToy(torch.nn.Module):
    """Small model with one Muon matrix and AdamW parameters, all used in forward."""
    def __init__(self):
        super().__init__()
        self.blocks = torch.nn.ModuleList([torch.nn.Linear(3, 3)])
        self.head = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.head(torch.tanh(self.blocks[0](x)))


def _ddp_optimizer_worker(rank, world_size, port, optimizer_name):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            init_method=f"tcp://127.0.0.1:{port}?use_libuv=0")
    try:
        torch.manual_seed(123)
        model = _DDPToy()
        ddp = DDP(model)
        optim_cfg = dict(name=optimizer_name, lr=0.001, muon_lr=0.02, weight_decay=0.01)
        opt = build_optimizer(ddp, optim_cfg)
        if optimizer_name == "muon":
            groups = {name: {id(p) for group in optimizer.param_groups for p in group["params"]}
                      for name, optimizer in opt.named}
            muon_ids, adamw_ids = groups["opt.muon"], groups["opt.adamw"]
        else:
            combined = dict(opt.named)["opt.polar_muon_adamw"]
            muon_ids = {id(p) for group in combined.param_groups if group["kind"] == "muon"
                        for p in group["params"]}
            adamw_ids = {id(p) for group in combined.param_groups if group["kind"] == "adamw"
                         for p in group["params"]}
        assert id(model.blocks[0].weight) in muon_ids
        assert id(model.blocks[0].bias) in adamw_ids
        assert id(model.head.weight) in adamw_ids

        reference = copy.deepcopy(model)
        xs = [torch.tensor([[0.2, -0.1, 0.4], [0.5, 0.3, -0.2]]),
              torch.tensor([[-0.3, 0.6, 0.1], [0.7, -0.4, 0.2]])]
        targets = [torch.tensor([[0.1, -0.2], [0.3, 0.4]]),
                   torch.tensor([[-0.5, 0.2], [0.6, -0.1]])]

        # DDP averages each rank's local mean gradient; this is the same as the
        # gradient of the concatenated batch when local batch sizes match.
        ref_opt = build_optimizer(reference, optim_cfg)
        torch.nn.functional.mse_loss(reference(torch.cat(xs)), torch.cat(targets)).backward()
        ref_opt.step()

        torch.nn.functional.mse_loss(ddp(xs[rank]), targets[rank]).backward()
        opt.step()
        for actual, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6)

        # All replicas must remain bitwise aligned after the optimizer update.
        for p in model.parameters():
            lo, hi = p.detach().clone(), p.detach().clone()
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
            torch.testing.assert_close(lo, hi, rtol=0, atol=0)

        from models.moe import SparseMoE
        moe = SparseMoE(2, num_experts=2, top_k=1, bias_gamma=0.1)
        moe._counts = torch.tensor([2.0, 0.0]) if rank == 0 else torch.tensor([1.0, 1.0])
        moe.update_bias()
        torch.testing.assert_close(moe.router_bias, torch.tensor([-0.1, 0.1]))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("optimizer_name", ["muon", "polar_muon_adamw"])
def test_ddp_gradients_and_muon_adamw_step_match_global_batch(optimizer_name):
    """Two Gloo ranks exercise DDP grouping, averaged gradients, and updates on CPU."""
    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_ddp_optimizer_worker, args=(2, port, optimizer_name), nprocs=2, join=True)


class _DDPMoEToy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from models.moe import SparseMoE
        self.blocks = torch.nn.ModuleList([
            SparseMoE(2, num_experts=2, top_k=1, num_shared=0, bias_gamma=0.1)
        ])
        self.head = torch.nn.Linear(2, 1)

    def forward(self, x):
        return self.head(self.blocks[0](x))


def _ddp_moe_worker(rank, world_size, port):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP

    dist.init_process_group("gloo", rank=rank, world_size=world_size,
                            init_method=f"tcp://127.0.0.1:{port}?use_libuv=0")
    try:
        torch.manual_seed(456)
        model = _DDPMoEToy()
        ddp = DDP(model, find_unused_parameters=True)
        opt = build_optimizer(ddp, dict(name="muon", lr=0.001, muon_lr=0.02, weight_decay=0.0))
        x = torch.tensor([[0.2, -0.3], [0.5, 0.1]]) + rank * 0.05
        # Each rank uses a different expert; this leaves one expert's local
        # gradient absent on each rank and exercises DDP's unused-parameter path.
        model.blocks[0]._forced = torch.full((x.shape[0], 1), rank, dtype=torch.long)
        ddp(x).square().mean().backward()
        model.blocks[0]._forced = None
        opt.step()
        model.blocks[0].update_bias()

        for p in model.parameters():
            lo, hi = p.detach().clone(), p.detach().clone()
            dist.all_reduce(lo, op=dist.ReduceOp.MIN)
            dist.all_reduce(hi, op=dist.ReduceOp.MAX)
            torch.testing.assert_close(lo, hi, rtol=0, atol=0)
        torch.testing.assert_close(model.blocks[0].router_bias, torch.zeros(2))
    finally:
        dist.destroy_process_group()


def test_ddp_moe_with_rank_local_unused_experts_stays_synchronized():
    """Per-rank routing may differ, but DDP parameters and global expert balance must agree."""
    import torch.multiprocessing as mp

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    mp.spawn(_ddp_moe_worker, args=(2, port), nprocs=2, join=True)
