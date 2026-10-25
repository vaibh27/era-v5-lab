"""Smoke tests for CorrectnessVerifier — identity PASS, *1.01 negative FAIL, gap upgrades."""

from __future__ import annotations

import io
from contextlib import redirect_stdout

import torch

from verifier import CorrectnessVerifier, make_perturbed_fn


def _linear_ref(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x @ w


def test_identity_forward_backward_pass() -> None:
    torch.manual_seed(0)
    v = CorrectnessVerifier()
    x = torch.randn(4, 8, dtype=torch.float32)
    w = torch.randn(8, 3, dtype=torch.float32)
    assert v.verify_all(_linear_ref, _linear_ref, (x, w), test_name="identity")


def test_negative_perturb_1_01_fails_forward() -> None:
    torch.manual_seed(1)
    v = CorrectnessVerifier()
    x = torch.randn(4, 8, dtype=torch.float32)
    w = torch.randn(8, 3, dtype=torch.float32)
    bad = make_perturbed_fn(_linear_ref, scale=1.01)
    ok = v.verify_forward(_linear_ref, bad, (x, w), test_name="neg_1_01")
    assert ok is False


def test_assert_forward_raises_on_perturb() -> None:
    torch.manual_seed(2)
    v = CorrectnessVerifier()
    x = torch.randn(2, 4, dtype=torch.float32)
    w = torch.randn(4, 2, dtype=torch.float32)
    bad = make_perturbed_fn(_linear_ref, scale=1.01)
    try:
        v.assert_forward(_linear_ref, bad, (x, w), test_name="assert_neg")
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_pass_log_includes_max_rel() -> None:
    torch.manual_seed(3)
    v = CorrectnessVerifier()
    x = torch.randn(3, 4, dtype=torch.float32)
    w = torch.randn(4, 2, dtype=torch.float32)
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok = v.verify_forward(_linear_ref, _linear_ref, (x, w), test_name="log_max_rel")
    assert ok is True
    text = buf.getvalue()
    assert "[PASS]" in text
    assert "max_rel=" in text


def test_gold_fp64_identity_passes() -> None:
    torch.manual_seed(4)
    v = CorrectnessVerifier()
    x = torch.randn(4, 8, dtype=torch.float32)
    w = torch.randn(8, 3, dtype=torch.float32)
    assert v.verify_all(
        _linear_ref,
        _linear_ref,
        (x, w),
        test_name="gold_fp64",
        gold_fp64=True,
    )


def test_fp8_tol_slots_exist() -> None:
    v = CorrectnessVerifier(atol_fp8=0.2, rtol_fp8=0.15)
    atol, rtol = v._tol_fp8()
    assert v.fp8_tol == (0.2, 0.15)
    assert atol == 0.2 and rtol == 0.15
    assert atol < 1.0 and rtol < 1.0
    defaults = CorrectnessVerifier()
    d_atol, d_rtol = defaults._tol_fp8()
    assert d_atol == 1e-1 and d_rtol == 1e-1
