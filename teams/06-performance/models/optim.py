"""Optimizers: AdamW, or Muon on every 2D weight + AdamW on the rest.  [implemented: MVP]

  * build_optimizer(model, ocfg) -> object with zero_grad(), step(scaler=None), unscale(scaler), flops().
    ocfg = cfg["optim"]: name adamw | muon | polar_muon_adamw, lr, muon_lr, weight_decay.
  * adamw: torch.optim.AdamW (fused=True on CUDA).
  * muon: Muon (Keller Jordan): momentum 0.95 nesterov, update = NS5(momentum grad) * max(1, rows/cols)**0.5,
    Newton-Schulz coefficients (3.4445, -4.7750, 2.0315), 5 steps, bf16 (fp32 on GPUs without bf16: T4/P100/V100).
    Muon handles 2D block weights (including routers). AdamW handles embedding parameters
    (token table / KE projection / position table), output head, norm gains and biases. Decoupled
    weight decay applies to AdamW matrices, not 1D biases or norm gains.
    A parameter with grad=None is skipped, matching AdamW's unused-expert behavior. With DDP, build_optimizer
    accepts either the base module or its DDP wrapper; DDP synchronizes gradients, and each rank keeps local,
    replicated optimizer state. This is not optimizer-state sharding.
  * polar_muon_adamw: one torch optimizer combines AdamW with Polar Express Muon, row equilibration,
    norm correction, factored variance reduction and cautious weight decay. With DDP, gradients are synchronized
    by DDP and optimizer state remains replicated; this implementation does not shard optimizer state.
  * Regions (level 2): "opt.muon", "opt.adamw"; (level 3): "opt.zero_grad".
  * flops(): Newton-Schulz FLOPs per step = sum over matrices of 5 * (4 m^2 n + 2 m^3), m <= n (for HFU).
Cost to watch: one small matmul chain per matrix (8 experts x 2 x layers + attention + ...): likely
launch-bound. A batched version (same-shaped matrices through bmm) is a week-2 target.
"""
from __future__ import annotations

import math

import torch

from perfkit.regions import region

NS_COEFFS = (3.4445, -4.7750, 2.0315)

# Polar Express Muon coefficients (five iterations).
POLAR_EXPRESS_COEFFS = (
    (8.156554524902461, -22.48329292557795, 15.878769915207462),
    (4.042929935166739, -2.808917465908714, 0.5000178451051316),
    (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
    (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
    (2.3465413258596377, -1.7097828382687081, 0.42323551169305323),
)


def _native_bf16():
    try:
        return torch.cuda.is_bf16_supported(including_emulation=False)
    except TypeError:  # torch < 2.4
        return torch.cuda.get_device_capability()[0] >= 8


def zeropower_via_newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximately orthogonalise G (the U V^T of its SVD), in bf16 where the GPU has it."""
    a, b, c = NS_COEFFS
    X = G.to(torch.bfloat16 if (not G.is_cuda or _native_bf16()) else torch.float32)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    X = X / (X.norm() + eps)
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    return X.T if transpose else X


class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, weight_decay=0.0):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov,
                                      ns_steps=ns_steps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state[p]
                # Match AdamW: an unused expert receives no update and its moments stay unchanged.
                if p.grad is None:
                    continue
                if "buf" not in st:
                    st["buf"] = torch.zeros_like(p.grad)
                buf = st["buf"]
                buf.lerp_(p.grad, 1 - group["momentum"])
                g = p.grad.lerp(buf, group["momentum"]) if group["nesterov"] else buf
                with region("opt.muon.ns", 3):
                    u = zeropower_via_newtonschulz5(g, group["ns_steps"])
                if group["weight_decay"]:
                    p.mul_(1 - group["lr"] * group["weight_decay"])
                p.add_(u.to(p.dtype), alpha=-group["lr"] * max(1.0, p.size(0) / p.size(1)) ** 0.5)


class MuonAdamW(torch.optim.Optimizer):
    """Single-process combined optimizer using Polar Express Muon + AdamW updates.

    Adapted from Andrej Karpathy's reference implementation:
    https://github.com/karpathy/nanochat/blob/master/nanochat/optim.py
    (MIT, Copyright 2025 Andrej Karpathy).
    This port keeps the Polar Express, row equilibration, Muon+ norm correction, factored
    variance reduction, and cautious decay update. Under DDP, DDP synchronizes gradients before
    this local update and each rank keeps replicated optimizer state. It does not implement
    optimizer-state sharding, and uses ordinary PyTorch ops rather than compiled fused kernels.
    """

    def __init__(self, param_groups):
        super().__init__(param_groups, defaults={})

    @staticmethod
    def _adamw_step(p, grad, state, group):
        if not state:
            state["step"] = 0
            state["exp_avg"] = torch.zeros_like(p, dtype=torch.float32)
            state["exp_avg_sq"] = torch.zeros_like(p, dtype=torch.float32)
        state["step"] += 1
        step = state["step"]
        p32, grad32 = p.float(), grad.float()
        exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
        beta1, beta2 = group["betas"]
        p32.mul_(1 - group["lr"] * group["weight_decay"])
        exp_avg.lerp_(grad32, 1 - beta1)
        exp_avg_sq.lerp_(grad32.square(), 1 - beta2)
        denom = (exp_avg_sq / (1 - beta2 ** step)).sqrt().add_(group["eps"])
        p32.addcdiv_(exp_avg / (1 - beta1 ** step), denom, value=-group["lr"])
        p.copy_(p32)

    @staticmethod
    def _muon_step(p, grad, state, group):
        if not state:
            state["momentum_buffer"] = torch.zeros_like(p)
            rows, cols = p.shape
            state_shape = (rows, 1) if rows >= cols else (1, cols)
            state["second_momentum_buffer"] = torch.zeros(state_shape, device=p.device, dtype=p.dtype)
        momentum = group["momentum"]
        momentum_buffer = state["momentum_buffer"]
        momentum_buffer.lerp_(grad, 1 - momentum)
        g = grad.lerp(momentum_buffer, momentum)

        x = g.to(torch.bfloat16 if p.is_cuda and _native_bf16() else torch.float32)
        rows, cols = x.shape
        target = x.float().norm() / (rows ** 0.5)
        row_norm = x.float().norm(dim=-1, keepdim=True).clamp_min(1e-6)
        x = x * (target / row_norm).to(x.dtype)
        x = x / (x.norm() * 1.01 + 1e-6)
        red_dim = -1 if rows >= cols else -2
        if rows > cols:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                gram = x.T @ x
                x = a * x + x @ (b * gram + c * (gram @ gram))
        else:
            for a, b, c in POLAR_EXPRESS_COEFFS[:group["ns_steps"]]:
                gram = x @ x.T
                x = a * x + (b * gram + c * (gram @ gram)) @ x

        g = x.to(p.dtype)
        target_norm = min(rows, cols) ** 0.5
        g = g * (target_norm / g.float().norm().clamp_min(1e-6)).to(g.dtype)
        v_mean = g.float().square().mean(dim=red_dim, keepdim=True)
        second = state["second_momentum_buffer"]
        second.lerp_(v_mean.to(second.dtype), 1 - group["beta2"])
        step_size = second.clamp_min(1e-10).rsqrt()
        reduction_size = g.size(red_dim)
        v_norm = (v_mean.sum() * reduction_size).sqrt()
        v_norm_new = ((v_mean * reduction_size) * step_size.float().square()).sum().sqrt()
        g = g * (step_size * (v_norm / v_norm_new.clamp_min(1e-10))).to(g.dtype)

        lr = group["lr"] * max(1.0, rows / cols) ** 0.5
        wd = group["weight_decay"]
        mask = (g * p) >= 0
        p.sub_(lr * g + lr * wd * p * mask)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            kind = group["kind"]
            region_name = "opt.muon" if kind == "muon" else "opt.adamw"
            with region(region_name, 2):
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if kind == "muon":
                        with region("opt.muon.ns", 3):
                            self._muon_step(p, p.grad, self.state[p], group)
                    elif kind == "adamw":
                        self._adamw_step(p, p.grad, self.state[p], group)
                    else:
                        raise ValueError(f"unknown MuonAdamW optimizer group kind {kind!r}")
        return loss


class Optimizers:
    """One or two torch optimizers behind one zero_grad() / step(), with timed regions."""

    def __init__(self, named: list):
        self.named = named          # [(region name, torch optimizer)]
        names = [name for name, _ in named]
        if len(names) != len(set(names)):
            raise ValueError("optimizer region names must be unique")
        for _, optimizer in named:
            for group in optimizer.param_groups:
                group.setdefault("initial_lr", group["lr"])

    def state_dict(self):
        """Return checkpoint state for every optimizer managed by this wrapper."""
        return {name: optimizer.state_dict() for name, optimizer in self.named}

    def load_state_dict(self, state_dict):
        """Restore optimizer states, requiring the same named optimizer set."""
        expected = {name for name, _ in self.named}
        received = set(state_dict)
        if received != expected:
            raise ValueError(f"optimizer state names differ: expected {sorted(expected)}, got {sorted(received)}")
        for name, optimizer in self.named:
            optimizer.load_state_dict(state_dict[name])

    def set_lr(self, step: int, total_steps: int, schedule="constant", warmup_steps=0,
               min_lr_ratio=0.1):
        """Set each optimizer group's LR from its own base LR.

        cosine warms linearly from zero, then decays to min_lr_ratio. constant
        keeps timing runs at fixed rates unless a warmup is explicitly requested.
        """
        if schedule not in ("constant", "cosine"):
            raise ValueError(f"optim.schedule={schedule!r}: constant | cosine")
        if total_steps <= 0 or warmup_steps < 0 or warmup_steps >= total_steps:
            raise ValueError("Require total_steps > warmup_steps >= 0")
        if not 0.0 <= min_lr_ratio <= 1.0:
            raise ValueError("min_lr_ratio must be in [0, 1]")
        if step < 0 or step >= total_steps:
            raise ValueError("step must be in [0, total_steps)")
        warmup = (step + 1) / warmup_steps if warmup_steps and step < warmup_steps else 1.0
        if schedule == "cosine" and step >= warmup_steps:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
            decay = min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(progress * math.pi))
        else:
            decay = 1.0
        multiplier = warmup * decay
        for _, optimizer in self.named:
            for group in optimizer.param_groups:
                group["lr"] = group["initial_lr"] * multiplier

    def zero_grad(self):
        with region("opt.zero_grad", 3):
            for _, o in self.named:
                o.zero_grad(set_to_none=True)

    def step(self, scaler=None):
        for name, o in self.named:
            with region(name, 2):
                scaler.step(o) if scaler is not None else o.step()

    def unscale(self, scaler):
        for _, o in self.named:
            scaler.unscale_(o)

    def flops(self):
        """Newton-Schulz FLOPs per step (Muon is not free; counted in HFU, not in MFU)."""
        total = 0
        for _, o in self.named:
            if isinstance(o, (Muon, MuonAdamW)):
                for g in o.param_groups:
                    if isinstance(o, MuonAdamW) and g["kind"] != "muon":
                        continue
                    for p in g["params"]:
                        m, n = sorted(p.shape[:2])
                        steps = g.get("ns_steps", 5)
                        total += steps * (4 * m * m * n + 2 * m ** 3)
        return total


def build_optimizer(model, ocfg: dict):
    # DDP adds a ``module.`` prefix to every parameter name. Classify using the
    # underlying model's names while keeping the same Parameter objects so the
    # returned optimizers work with either wrapped or unwrapped models.
    named_model = model
    while isinstance(getattr(named_model, "module", None), torch.nn.Module):
        named_model = named_model.module
    named_params = [(n, p) for n, p in named_model.named_parameters() if p.requires_grad]
    params = [p for _, p in named_params]
    fused = params[0].is_cuda

    def adamw(named):
        decay, no_decay = [], []
        for name, p in named:
            (decay if p.ndim >= 2 and not name.endswith(".bias") else no_decay).append(p)
        groups = []
        if decay:
            groups.append({"params": decay, "weight_decay": ocfg["weight_decay"]})
        if no_decay:
            groups.append({"params": no_decay, "weight_decay": 0.0})
        return torch.optim.AdamW(groups, lr=ocfg["lr"], betas=(0.9, 0.95), fused=fused)

    if ocfg["name"] == "adamw":
        return Optimizers([("opt.adamw", adamw(named_params))])
    if ocfg["name"] == "polar_muon_adamw":
        muon_params = [p for n, p in named_params
                       if p.dim() == 2 and n.startswith("blocks.") and n.endswith(".weight")]
        muon_ids = {id(p) for p in muon_params}
        adamw_params = [(n, p) for n, p in named_params if id(p) not in muon_ids]
        groups = []
        if muon_params:
            groups.append({
                "params": muon_params, "kind": "muon", "lr": ocfg["muon_lr"],
                "momentum": 0.95, "beta2": 0.95, "ns_steps": 5,
                "weight_decay": ocfg["weight_decay"],
            })
        decay, no_decay = [], []
        for name, p in adamw_params:
            (decay if p.ndim >= 2 and not name.endswith(".bias") else no_decay).append(p)
        if decay:
            groups.append({"params": decay, "kind": "adamw", "lr": ocfg["lr"],
                           "betas": (0.9, 0.95), "eps": 1e-8,
                           "weight_decay": ocfg["weight_decay"]})
        if no_decay:
            groups.append({"params": no_decay, "kind": "adamw", "lr": ocfg["lr"],
                           "betas": (0.9, 0.95), "eps": 1e-8, "weight_decay": 0.0})
        return Optimizers([("opt.polar_muon_adamw", MuonAdamW(groups))])
    if ocfg["name"] == "muon":
        if "muon_params" in ocfg and ocfg["muon_params"] != "hidden":
            raise ValueError("Remove optim.muon_params: embeddings and head must use AdamW")

        def use_muon(name, p):
            return p.dim() == 2 and name.startswith("blocks.") and name.endswith(".weight")
        two_d = [p for n, p in named_params if use_muon(n, p)]
        rest = [(n, p) for n, p in named_params if not use_muon(n, p)]
        named = []
        if two_d:
            named.append(("opt.muon", Muon(two_d, lr=ocfg["muon_lr"], weight_decay=ocfg["weight_decay"])))
        if rest:
            named.append(("opt.adamw", adamw(rest)))
        return Optimizers(named)
    raise ValueError(f"optim.name={ocfg['name']!r}: adamw | muon | polar_muon_adamw")
