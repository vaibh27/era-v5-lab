"""regulariser_backward_ (dt/mask.py): the gradient of -lambda * regulariser(model), layer by layer and without an
autograd graph, must equal autograd's bit for bit and keep no saved tensors (tiny random Llama, CPU, no download)."""
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g10_dt_mask_regbwd_test", HERE.parent / "mask.py")
dtmask = importlib.util.module_from_spec(spec)
sys.modules["g10_dt_mask_regbwd_test"] = dtmask
spec.loader.exec_module(dtmask)

_parametrize = pytest.mark.parametrize  # undotted alias, as in test_mask.py
IDS = torch.tensor([[3, 7, 9, 11, 13, 5, 6, 8], [4, 4, 10, 12, 1, 2, 3, 0]])


def tiny(seed=0, hidden=32, inter=64):
    """The tiny random Llama of test_mask.py (no bias)."""
    from transformers import LlamaConfig, LlamaForCausalLM  # noqa: PLC0415

    cfg = LlamaConfig(vocab_size=64, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                      tie_word_embeddings=True, mlp_bias=False, attention_bias=False)
    torch.manual_seed(seed)
    return LlamaForCausalLM(cfg).float()


# --- regulariser_backward_: the regulariser's gradient without an autograd graph (memory) ------------------------


def dt_pair(base_dtype, seed=4, total_steps=10):
    """Two identical tiny models with DT on gate/up/down, logits spread around 0, mid-schedule alpha (370 with 10 steps;
    485.71... with 7, not exact in FP32), and the frozen W stored in `base_dtype` (FP32 as in sft/bench, bf16 as in
    rl-loop)."""
    import copy  # noqa: PLC0415
    m = tiny(seed=seed)
    dtmask.convert(m, dtmask.TARGET_SETS["mlp"], alpha_start=100.0, alpha_end=1000.0, total_steps=total_steps)
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, layer in dtmask.masked_layers(m):
            layer.gate.copy_(torch.randn(layer.gate.shape, generator=gen) * 0.02)
            if base_dtype != torch.float32:
                layer._parameters["weight"].data = layer.base_weight.data.to(base_dtype)
    for _ in range(3):
        dtmask.step_all(m)  # mid-schedule: a band where sigmoid' is neither 0 nor 1/4
    return m, copy.deepcopy(m)


@_parametrize("base_dtype", [torch.float32, torch.bfloat16])
@_parametrize("prior_grad", [False, True])
@_parametrize("total_steps", [10, 7])
def test_regulariser_backward_equals_autograd_bit_for_bit(base_dtype, prior_grad, total_steps):
    a, b = dt_pair(base_dtype, total_steps=total_steps)
    la, lb = dtmask.masked_layers(a), dtmask.masked_layers(b)
    if prior_grad:  # the task gradient is already accumulated when the regulariser runs (once per optimizer step)
        gen = torch.Generator().manual_seed(9)
        for (_, x), (_, y) in zip(la, lb):
            # at the regulariser's own scale, so a last-bit difference in its gradient is not absorbed by the sum
            x.gate.grad = torch.randn(x.gate.shape, generator=gen) * 1e-9
            y.gate.grad = x.gate.grad.clone()
    lam = 1e-5 * (1.0 - 3 / 10)
    reg = dtmask.regulariser(a)
    (-lam * reg).backward()
    value = dtmask.regulariser_backward_(b, lam)
    assert value == float(reg.detach())
    for (n, x), (_, y) in zip(la, lb):
        assert torch.equal(x.gate.grad, y.gate.grad), n
    assert all(l.base_weight.dtype == base_dtype for _, l in lb)


@_parametrize("base_dtype", [torch.float32, torch.bfloat16])
def test_regulariser_backward_keeps_no_autograd_graph(base_dtype):
    """The memory claim, measured as bytes autograd saves for backward: the old path keeps sigmoid(alpha G), W * m and
    the W it multiplies for every masked layer at once (>= 8 B per masked weight); the new one keeps nothing."""
    a, b = dt_pair(base_dtype)
    n_weights = sum(l.gate.numel() for _, l in dtmask.masked_layers(a))
    saved = {"bytes": 0}

    def pack(t):
        saved["bytes"] += t.numel() * t.element_size()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        reg = dtmask.regulariser(a)
    old = saved["bytes"]
    (-1e-5 * reg).backward()
    saved["bytes"] = 0
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        dtmask.regulariser_backward_(b, 1e-5)
    assert old >= 8 * n_weights and saved["bytes"] == 0
    assert all(l.gate.grad is not None for _, l in dtmask.masked_layers(b))


def test_regulariser_backward_gives_the_same_training_run():
    """Three optimizer steps the way sft/bench/train.py runs them (task backward, regulariser once per step, clip,
    AdamW, schedule step): the logits after the old and the new regulariser are bit-identical."""
    a, b = dt_pair(torch.float32, seed=6)
    runs = []
    for m, new in ((a, False), (b, True)):
        params = [p for p in m.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-2, betas=(0.9, 0.95), weight_decay=0.0)
        m.train()
        values = []
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            m(input_ids=IDS, labels=IDS).loss.backward()
            lam = dtmask.lambda_at(step, 10, 1e-5)
            if new:
                values.append(dtmask.regulariser_backward_(m, lam))
            else:
                reg = dtmask.regulariser(m)
                (-lam * reg).backward()
                values.append(float(reg.detach()))
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            dtmask.step_all(m)
        runs.append((values, [l.gate.detach().clone() for _, l in dtmask.masked_layers(m)]))
    assert runs[0][0] == runs[1][0]
    for x, y in zip(runs[0][1], runs[1][1]):
        assert torch.equal(x, y)


def test_regulariser_backward_frozen_logits_and_strided_weights():
    """A frozen logit gets no gradient (as in autograd) but its term still counts; a W with transposed strides still
    gives a contiguous gradient in the logit's own layout, with the same values as autograd."""
    a, b = dt_pair(torch.float32, seed=8)
    for m in (a, b):
        layers = dtmask.masked_layers(m)
        layers[0][1].gate.requires_grad_(False)
        w = layers[1][1].base_weight.data
        layers[1][1]._parameters["weight"].data = w.t().contiguous().t()  # same values, transposed strides
    reg = dtmask.regulariser(a)
    (-1e-5 * reg).backward()
    value = dtmask.regulariser_backward_(b, 1e-5)
    assert value == float(reg.detach())
    la, lb = dtmask.masked_layers(a), dtmask.masked_layers(b)
    assert la[0][1].gate.grad is None and lb[0][1].gate.grad is None
    assert lb[1][1].gate.grad.is_contiguous()
    for (n, x), (_, y) in zip(la[1:], lb[1:]):
        assert torch.equal(x.gate.grad, y.gate.grad), n
