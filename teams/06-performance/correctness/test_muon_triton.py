"""Standalone Triton Muon kernels: Newton-Schulz and one Muon update step.

These tests intentionally do not import the standalone kernels into models.optim.
They compare the kernels with the existing Muon reference implementation.
"""
import pytest
import torch

from kernels.muon import muon_update_, newton_schulz5, supported
from models.optim import Muon, zeropower_via_newtonschulz5

CUDA_BF16 = pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="the Triton Muon path is exercised on CUDA with native bf16",
)


def _assert_update_close(actual, expected, label):
    max_abs = (actual.float() - expected.float()).abs().max().item()
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        rtol=0.05,
        atol=0.02,
        msg=f"{label} exceeded tolerance; max_abs_error={max_abs:.6g}",
    )


@CUDA_BF16
@pytest.mark.parametrize("shape", [(33, 47), (47, 33), (64, 64)])
@pytest.mark.parametrize("steps", [0, 1, 5])
def test_newton_schulz5_matches_reference(shape, steps):
    torch.manual_seed(17)
    grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    expected = zeropower_via_newtonschulz5(grad, steps=steps)
    actual = newton_schulz5(grad, steps=steps)

    assert actual.shape == grad.shape
    assert actual.dtype == torch.bfloat16
    assert torch.isfinite(actual).all()
    _assert_update_close(actual, expected, f"NS5 shape={shape} steps={steps}")


@CUDA_BF16
@pytest.mark.parametrize("shape", [(33, 47), (47, 33)])
@pytest.mark.parametrize("weight_decay", [0.0, 0.03])
@pytest.mark.parametrize("nesterov", [True, False])
def test_muon_update_matches_muon_optimizer(shape, weight_decay, nesterov):
    torch.manual_seed(23)
    lr, momentum = 0.02, 0.9
    grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    initial = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    param = torch.nn.Parameter(initial.clone())
    param.grad = grad.clone()
    reference = Muon([param], lr=lr, momentum=momentum, nesterov=nesterov,
                     ns_steps=5, weight_decay=weight_decay)
    reference.step()

    actual = initial.clone()
    actual_buf = torch.zeros_like(actual)
    muon_update_(actual, grad, actual_buf, lr=lr, momentum=momentum,
                 nesterov=nesterov, weight_decay=weight_decay)

    _assert_update_close(actual, param.detach(), "Muon update")
    torch.testing.assert_close(
        actual_buf.float(), reference.state[param]["buf"].float(),
        rtol=0, atol=1e-3, msg="momentum buffer differs from the Muon reference",
    )


@CUDA_BF16
@pytest.mark.parametrize("shape", [(33, 47), (47, 33)])
def test_muon_update_matches_over_three_steps(shape):
    torch.manual_seed(29)
    lr, momentum, weight_decay = 0.02, 0.9, 0.03
    grads = [torch.randn(shape, device="cuda", dtype=torch.bfloat16) * (step + 1)
             for step in range(3)]
    initial = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

    param = torch.nn.Parameter(initial.clone())
    reference = Muon([param], lr=lr, momentum=momentum, ns_steps=5,
                     weight_decay=weight_decay)
    actual = initial.clone()
    actual_buf = torch.zeros_like(actual)

    for grad in grads:
        param.grad = grad.clone()
        reference.step()
        muon_update_(actual, grad, actual_buf, lr=lr, momentum=momentum,
                     weight_decay=weight_decay)
        _assert_update_close(actual, param.detach(), "multi-step Muon update")
        torch.testing.assert_close(
            actual_buf.float(), reference.state[param]["buf"].float(),
            rtol=0, atol=0.01, msg="multi-step momentum buffer differs",
        )


@CUDA_BF16
@pytest.mark.parametrize("shape", [(33, 47), (47, 33)])
def test_muon_optimizer_triton_impl_matches_loop(shape):
    torch.manual_seed(37)
    lr, momentum, weight_decay = 0.02, 0.9, 0.03
    initial = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
    loop_param = torch.nn.Parameter(initial.clone())
    triton_param = torch.nn.Parameter(initial.clone())
    loop = Muon([loop_param], lr=lr, momentum=momentum, ns_steps=5,
                weight_decay=weight_decay, impl="loop")
    triton = Muon([triton_param], lr=lr, momentum=momentum, ns_steps=5,
                  weight_decay=weight_decay, impl="triton")

    for step in range(3):
        grad = torch.randn(shape, device="cuda", dtype=torch.bfloat16) * (step + 1)
        loop_param.grad = grad.clone()
        triton_param.grad = grad.clone()
        loop.step()
        triton.step()
        _assert_update_close(triton_param, loop_param.detach(), "optimizer-level Muon update")
        torch.testing.assert_close(
            triton.state[triton_param]["buf"].float(),
            loop.state[loop_param]["buf"].float(),
            rtol=0, atol=0.01, msg="optimizer-level momentum buffer differs",
        )


def _newton_schulz5_fp32(grad, steps=5, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    x = grad.float()
    transpose = x.size(0) > x.size(1)
    if transpose:
        x = x.T
    x = x / (x.norm() + eps)
    for _ in range(steps):
        gram = x @ x.T
        x = a * x + (b * gram + c * (gram @ gram)) @ x
    return x.T if transpose else x


def test_newton_schulz5_cpu_fallback_matches_float_reference():
    torch.manual_seed(31)
    grad = torch.randn(17, 23)
    actual = newton_schulz5(grad)
    assert actual.device.type == "cpu"
    assert actual.dtype == torch.float32
    reference = _newton_schulz5_fp32(grad)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)


def test_muon_kernel_input_validation():
    with pytest.raises(ValueError, match="2D tensor"):
        newton_schulz5(torch.randn(3))
    with pytest.raises(ValueError, match="nonnegative integer"):
        newton_schulz5(torch.randn(3, 4), steps=-1)
    with pytest.raises(ValueError, match="positive"):
        newton_schulz5(torch.randn(3, 4), eps=0.0)
    with pytest.raises(ValueError, match="2D parameter"):
        muon_update_(torch.randn(3), torch.randn(3), torch.randn(3), lr=0.1)
    with pytest.raises(ValueError, match="same shape"):
        muon_update_(torch.randn(3, 4), torch.randn(4, 3), torch.randn(3, 4), lr=0.1)
    with pytest.raises(ValueError, match="momentum"):
        muon_update_(torch.randn(3, 4), torch.randn(3, 4), torch.randn(3, 4),
                     lr=0.1, momentum=1.0)
    assert not supported(torch.randn(3, 4))
