"""Reference GQA/RoPE/SwiGLU MoE with G4-compatible projected byte codes.

Dense untied output head per TSAI review; Kronecker projection is input-only.
No fused kernels, reversible layers or expert parallelism are claimed.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F


def codec(rows, lengths):
    """Independent dense FP32 reference; same unbiased-std convention as G4."""
    p = rows.shape[1]
    n = len(rows)
    lengths = lengths.long()
    positions = torch.arange(p, device=rows.device).expand(n, -1)
    indices = rows.long() * p + positions
    source = (positions < lengths[:, None]).float() * lengths.clamp_min(
        1
    ).float().rsqrt()[:, None]
    out = torch.zeros(n, 256 * p, device=rows.device)
    # Each position has a distinct flattened column, including padded positions.
    out.scatter_(1, indices, source)
    centered = out - out.mean(-1, keepdim=True)
    return centered / (centered.std(-1, keepdim=True, correction=1) + 1e-6)


class KE2(nn.Module):
    def __init__(self, rows, lengths, width):
        super().__init__()
        self.register_buffer("rows", torch.as_tensor(rows, dtype=torch.uint8).clone())
        self.register_buffer(
            "lengths", torch.as_tensor(lengths, dtype=torch.long).clone()
        )
        self.p = rows.shape[1]
        self.vocab = len(rows)
        self.width = width
        self.projection = nn.Linear(256 * self.p, width, bias=False)
        nn.init.normal_(self.projection.weight, std=1 / math.sqrt(256 * self.p))

    def projected(self, ids):
        """Sparse closed-form application of the same normalized G4 code.

        Avoid allocating V x (256P) raw codes. Equivalent in real arithmetic;
        FP32 reduction ordering differs and is tested against the dense reference.
        """
        shape = ids.shape
        flat = ids.reshape(-1)
        lengths = self.lengths[flat]
        positions = torch.arange(self.p, device=ids.device)
        indices = self.rows[flat].long() * self.p + positions
        weight = self.projection.weight.float().t()
        selected = weight[indices]  # [tokens, P, d_model]
        valid = positions[None, :] < lengths[:, None]
        selected_sum = (selected * valid[:, :, None]).sum(1)
        dim = 256 * self.p
        l = lengths.float()
        scale = l.clamp_min(1).rsqrt()
        mean = l * scale / dim
        variance = (l * (scale - mean).square() + (dim - l) * mean.square()) / (dim - 1)
        denom = variance.sqrt() + 1e-6
        high = (scale - mean) / denom
        low = -mean / denom
        result = (high - low)[:, None] * selected_sum + low[:, None] * weight.sum(0)
        return result.reshape(*shape, self.width)

    def forward(self, ids):
        return self.projected(ids)


class RMSNorm(nn.Module):
    def __init__(self, width, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        return (
            x.float()
            * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
            * self.weight
        ).to(x.dtype)


class GQA(nn.Module):
    def __init__(self, m, layer_index=0):
        super().__init__()
        w = m["n_embd"]
        self.nh = m["n_head"]
        self.nkv = m["n_kv_head"]
        self.d = w // self.nh
        # Each query head has its own projection. Groups of queries share K and V.
        self.q = nn.Linear(w, w, bias=False)
        self.k = nn.Linear(w, self.nkv * self.d, bias=False)
        self.v = nn.Linear(w, self.nkv * self.d, bias=False)
        self.o = nn.Linear(w, w, bias=False)
        self.theta = m["rope_theta"]
        self.rotary_dim = 2 * math.floor(m.get("rotary_fraction", 1.0) * self.d / 2)
        self._rotary_cache = {}
        pattern = m.get("attn_pattern", ["G"] * m["n_layer"])
        self.local = pattern[layer_index] == "L"
        # SDPA covers causal global layers only; local layers keep the eager window and sink.
        self.sdpa = m.get("attn_impl", "eager") == "sdpa" and not self.local
        self.window = m.get("attn_window") if self.local else None
        self.sink = (
            nn.Parameter(torch.zeros(self.nh, dtype=torch.float32))
            if self.local and m.get("attn_sink", False)
            else None
        )
        self.gate = None
        if m.get("attn_gate", False):
            required = ["attn_gate_input", "attn_gate_bias", "attn_gate_init"]
            if any(m.get(k) is None for k in required):
                raise ValueError("Gate choices must be explicit")
            if m["attn_gate_input"] not in ("normalized", "residual"):
                raise ValueError("Invalid gate input")
            self.gate_input = m["attn_gate_input"]
            self.gate = nn.Linear(w, w, bias=m["attn_gate_bias"])
            if m["attn_gate_init"] == "zero":
                nn.init.zeros_(self.gate.weight)
                if self.gate.bias is not None:
                    nn.init.zeros_(self.gate.bias)
            elif m["attn_gate_init"] != "linear_default":
                raise ValueError("Invalid gate initialization")

    def rotary_angles(self, t, device):
        """cos and sin of the RoPE angles for positions 0..t-1, built once per length and device.

        Same ops as computing them per call, so the values are bit-identical. Inference-mode
        tensors cannot enter autograd, so they are cached separately.
        """
        key = (t, device, torch.is_inference_mode_enabled())
        cache = self._rotary_cache
        if key not in cache:
            inv = self.theta ** (
                -torch.arange(0, self.rotary_dim, 2, device=device).float()
                / self.rotary_dim
            )
            angle = torch.arange(t, device=device).float()[:, None] * inv[None, :]
            cache[key] = (angle.cos(), angle.sin())
        return cache[key]

    def rotary(self, x):
        # One window per document; position offset zero for each independent window.
        cos, sin = self.rotary_angles(x.shape[-2], x.device)
        a, b = (
            x.float()[..., : self.rotary_dim : 2],
            x.float()[..., 1 : self.rotary_dim : 2],
        )
        rotated = (
            torch.stack((a * cos - b * sin, a * sin + b * cos), -1)
            .flatten(-2)
            .to(x.dtype)
        )
        return (
            rotated
            if self.rotary_dim == self.d
            else torch.cat((rotated, x[..., self.rotary_dim :]), dim=-1)
        )

    def forward(self, x, residual=None):
        b, t, w = x.shape
        q = self.rotary(self.q(x).view(b, t, self.nh, self.d).transpose(1, 2))
        k = self.rotary(self.k(x).view(b, t, self.nkv, self.d).transpose(1, 2))
        v = self.v(x).view(b, t, self.nkv, self.d).transpose(1, 2)
        if self.sdpa:
            # Fused causal attention on the shared K/V heads, without a T x T score matrix.
            heads = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, enable_gqa=self.nh != self.nkv
            )
        else:
            heads = self.eager_attention(q, k, v)
        out = heads.transpose(1, 2).reshape(b, t, w)
        # Apply the channel gate before mixing heads with the output projection.
        if self.gate is not None:
            gate_input = x if self.gate_input == "normalized" else residual
            if gate_input is None:
                raise ValueError("Residual gate input requires the pre-norm residual")
            out = out * torch.sigmoid(self.gate(gate_input))
        return self.o(out)

    def eager_attention(self, q, k, v):
        b, _, t, _ = q.shape
        k = k.repeat_interleave(self.nh // self.nkv, dim=1)
        v = v.repeat_interleave(self.nh // self.nkv, dim=1)
        logits = q.float() @ k.float().transpose(-2, -1) / math.sqrt(self.d)
        mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
        # Intersect the causal mask with the local window on local layers.
        if self.local:
            positions = torch.arange(t, device=q.device)
            mask = mask & ((positions[:, None] - positions[None, :]) < self.window)
        logits = logits.masked_fill(~mask, float("-inf"))
        # A disabled (-inf) sink takes the exact legacy softmax path. This also
        # avoids changing reduction shape in the equivalence regression test.
        if self.sink is not None and not bool(torch.isneginf(self.sink).all()):
            sink = self.sink.float().view(1, self.nh, 1, 1).expand(b, -1, t, -1)
            a = torch.cat((logits, sink), dim=-1).softmax(-1)[..., :t].to(v.dtype)
        else:
            a = logits.softmax(-1).to(v.dtype)
        return a @ v


class SwiGLU(nn.Module):
    def __init__(self, w, h, bias=False):
        super().__init__()
        self.gate = nn.Linear(w, h, bias=bias)
        self.up = nn.Linear(w, h, bias=bias)
        self.down = nn.Linear(h, w, bias=bias)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class _StackedWeights(torch.autograd.Function):
    """Hand back a cached [E, ...] stack of expert weights; route its gradient to each weight.

    Backward is the one of torch.stack(weights).to(dtype): cast the gradient back to the
    weights' dtype and unbind it, so parameter gradients are bit-identical.
    """

    @staticmethod
    def forward(ctx, stacked, *weights):
        ctx.weight_dtype = weights[0].dtype
        return stacked

    @staticmethod
    def backward(ctx, grad):
        return (None, *grad.to(ctx.weight_dtype).unbind(0))


class MoE(nn.Module):
    def __init__(self, m):
        super().__init__()
        w = m["n_embd"]
        h = m["moe"]["ffn_hidden"]
        cfg = m["moe"]
        self.topk = cfg["top_k"]
        self.expert_load = None
        self.router = nn.Linear(w, cfg["num_experts"], bias=False)
        self.experts = nn.ModuleList(
            SwiGLU(w, h, cfg["bias"]) for _ in range(cfg["num_experts"])
        )
        self.shared = nn.ModuleList(
            SwiGLU(w, h, cfg["bias"]) for _ in range(cfg["num_shared"])
        )
        self._stack_cache = {}
        # "loop" is the config name of the dense path below; "grouped_mm" runs only the routed experts.
        self.dispatch = cfg.get("dispatch", "loop")

    def stacked(self, name, dtype):
        """[E, out, in] expert weights in dtype, stacked and cast once per weight update.

        The dense path used to stack and cast all expert weights in every call (every layer,
        every microbatch). The cache key holds each weight's version counter, which every
        in-place update (optimizer step, load_state_dict) bumps, so a changed weight rebuilds it.
        """
        weights = [getattr(e, name).weight for e in self.experts]
        key = (
            dtype,
            torch.is_inference_mode_enabled(),
            tuple((w._version, w.data_ptr()) for w in weights),
        )
        cached = self._stack_cache.get(name)
        if cached is None or cached[0] != key:
            with torch.no_grad():
                cached = (key, torch.stack(weights).to(dtype))
            self._stack_cache[name] = cached
        return _StackedWeights.apply(cached[1], *weights)

    def forward(self, x, valid):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        active = valid.reshape(-1)
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = F.linear(flat.float(), self.router.weight.float()).softmax(-1)
        gates, choice = scores.topk(self.topk, dim=-1)
        if self.expert_load is not None and self.training:
            selected = torch.zeros_like(scores, dtype=torch.bool)
            selected.scatter_(-1, choice, True)
            self.expert_load.add_((selected & active[:, None]).sum(dim=0))
        gates = gates / gates.sum(-1, keepdim=True)
        dt = flat.dtype
        if torch.is_autocast_enabled():
            dt = torch.get_autocast_dtype("cuda")
        Wg = self.stacked("gate", dt)   # [E, h, w]
        Wu = self.stacked("up", dt)
        Wd = self.stacked("down", dt)   # [E, w, h]
        xe = flat.to(dt)
        if self.dispatch == "grouped_mm":
            output = self.grouped_experts(xe, active, choice, gates, Wg, Wu, Wd)
            output = output.to(flat.dtype)
        else:
            # Execution-only fast path (TSAI): every expert runs on every token in one batched product, then each
            # token keeps only its selected experts with their normalised gates. Same value as the loop below,
            # with no data-dependent shapes and no host synchronisation. Padding rows get gate 0.
            gate_dense = torch.zeros_like(scores).scatter(-1, choice, gates) * active[:, None]
            hg = torch.einsum("td,ehd->teh", xe, Wg)
            hu = torch.einsum("td,ehd->teh", xe, Wu)
            y = torch.einsum("teh,edh->ted", F.silu(hg) * hu, Wd)
            output = torch.einsum("te,ted->td", gate_dense.float(), y.float()).to(flat.dtype)
        for expert in self.shared:
            output = output + expert(flat) * active[:, None]
        return output.reshape(shape)

    def grouped_experts(self, xe, active, choice, gates, Wg, Wu, Wd):
        """Run each token only through its top-k experts: one grouped GEMM per projection.

        All N x top_k routes are kept, padding tokens included with gate 0, so the row count is
        fixed: no capacity readback, no padding to the busiest expert and no host synchronisation.
        Idle experts get exact zero gradients, as on the dense path. Returns fp32 [N, d].
        """
        if xe.is_cuda and xe.dtype != torch.bfloat16:
            raise ValueError("grouped_mm dispatch needs bf16 autocast on CUDA (torch grouped GEMM)")
        n_experts = Wg.shape[0]
        routes = choice.flatten()                                    # [N * k], token-major
        order = routes.argsort(stable=True)                          # routes grouped by expert
        offsets = torch.searchsorted(
            routes[order], torch.arange(n_experts, device=routes.device), right=True
        ).to(torch.int32)                                            # end row of each expert's group
        packed = xe[order // self.topk]
        hg = F.grouped_mm(packed, Wg.transpose(-2, -1), offs=offsets)
        hu = F.grouped_mm(packed, Wu.transpose(-2, -1), offs=offsets)
        y = F.grouped_mm(F.silu(hg) * hu, Wd.transpose(-2, -1), offs=offsets)
        weight = (gates * active[:, None]).flatten()[order]
        # Back to (token, slot) order with a permutation copy, then sum the k slots in fp32.
        weighted = y.float() * weight[:, None]
        unsorted = torch.empty_like(weighted).index_copy(0, order, weighted)
        return unsorted.view(-1, self.topk, unsorted.shape[-1]).sum(1)


class Block(nn.Module):
    def __init__(self, m, layer_index=0):
        super().__init__()
        self.n1 = RMSNorm(m["n_embd"], m["norm_eps"])
        self.n2 = RMSNorm(m["n_embd"], m["norm_eps"])
        self.attn = GQA(m, layer_index)
        self.moe = MoE(m)

    def forward(self, x, valid):
        # Normalize before attention, then add its output to the residual stream.
        x = x + self.attn(self.n1(x), residual=x)
        # Feed the updated residual through the pre-norm MoE sublayer.
        return x + self.moe(self.n2(x), valid)


class LanguageModel(nn.Module):
    def __init__(self, m, rows, lengths):
        super().__init__()
        self.spec = m
        self.embedding = KE2(rows, lengths, m["n_embd"])
        self.blocks = nn.ModuleList(Block(m, i) for i in range(m["n_layer"]))
        self.norm = RMSNorm(m["n_embd"], m["norm_eps"])
        self.mtp_head = (
            nn.Linear(m["n_embd"], m["vocab_size"], bias=False) if m["mtp"] else None
        )
        self.head = nn.Linear(m["n_embd"], m["vocab_size"], bias=False)

    def hidden(self, ids, valid):
        x = self.embedding(ids)
        for block in self.blocks:
            x = block(x, valid)
        return self.norm(x)

    def forward(self, ids, valid, return_mtp=False):
        h = self.hidden(ids, valid)
        logits = self.head(h)
        return (
            (logits, self.mtp_head(h) if self.mtp_head is not None else None)
            if return_mtp
            else logits
        )

    def start_expert_load(self):
        for block in self.blocks:
            moe = block.moe
            moe.expert_load = moe.router.weight.new_zeros(len(moe.experts), dtype=torch.long)

    def finish_expert_load(self):
        counts = []
        for block in self.blocks:
            counts.append(block.moe.expert_load)
            block.moe.expert_load = None
        return torch.stack(counts).tolist()

    def parameter_count(self):
        return sum(p.numel() for p in self.parameters())
