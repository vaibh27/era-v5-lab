"""The framework must catch broken code.

Checks, for EVERY registered pair:

  * import every correctness/test_*.py so all pairs are registered
  * for each pair: check(pair, device, fast_override=scaled(1.01)) must raise AssertionError
  * a toy pair: correct passes; wrong gradient (x1.01 in backward only) fails; NaN fails; wrong shape fails
  * rel_err(x * 1.01, x) == 0.01
"""
from __future__ import annotations

import importlib
from pathlib import Path

import pytest
import torch

from framework import Case, PAIRS, Pair, check, pair, rel_err, scaled


def _import_all_test_modules() -> None:
    here = Path(__file__).resolve().parent
    for path in sorted(here.glob("test_*.py")):
        if path.name == Path(__file__).name:
            continue
        importlib.import_module(path.stem)


_import_all_test_modules()


@pair("_framework.toy", tol=1e-4)
def _toy_pair(device):
    def ref_fn(x):
        return x * 2

    def fast_fn(x):
        return x * 2

    x = torch.randn(3, 3, device=device, dtype=torch.float32, requires_grad=True)
    return Case(ref=ref_fn, fast=fast_fn, inputs=[x], grads=True)


class _ScaleBwd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, y, k):
        ctx.k = k
        return y

    @staticmethod
    def backward(ctx, g):
        return g * ctx.k, None


def test_rel_err_one_percent():
    x = torch.randn(8, 8, dtype=torch.float64)
    err = rel_err(x * 1.01, x)
    assert abs(err - 0.01) < 1e-9


def test_rel_err_shape_mismatch():
    with pytest.raises(AssertionError, match="shape"):
        rel_err(torch.zeros(2), torch.zeros(3))


def test_rel_err_nan():
    a = torch.tensor([1.0, float("nan")])
    b = torch.tensor([1.0, 1.0])
    with pytest.raises(AssertionError, match="NaN/Inf"):
        rel_err(a, b)


def test_toy_correct_passes(device):
    errors = check(PAIRS["_framework.toy"], device)
    assert "out[0]" in errors
    assert errors["out[0]"] < 1e-4


def test_toy_scaled_fails(device):
    with pytest.raises(AssertionError):
        check(PAIRS["_framework.toy"], device, fast_override=scaled(1.01))


def test_wrong_gradient_fails(device):
    def build(dev):
        def ref_fn(x):
            return x * 2

        def fast_fn(x):
            return _ScaleBwd.apply(x * 2, 1.01)

        x = torch.randn(5, device=dev, dtype=torch.float32, requires_grad=True)
        return Case(ref=ref_fn, fast=fast_fn, inputs=[x], grads=True)

    p = Pair("_framework.bad_grad", build, tol=1e-4)
    with pytest.raises(AssertionError, match="grad"):
        check(p, device)


def test_nan_output_fails(device):
    def build(dev):
        def ref_fn(x):
            return x * 2

        def fast_fn(x):
            return x * 2 + torch.nan

        x = torch.randn(2, 2, device=dev, dtype=torch.float32, requires_grad=True)
        return Case(ref=ref_fn, fast=fast_fn, inputs=[x], grads=False)

    p = Pair("_framework.nan", build, tol=1e-4)
    with pytest.raises(AssertionError, match="NaN/Inf"):
        check(p, device)


def test_wrong_shape_fails(device):
    def build(dev):
        def ref_fn(x):
            return x * 2

        def fast_fn(x):
            return (x * 2).flatten()

        x = torch.randn(2, 2, device=dev, dtype=torch.float32, requires_grad=True)
        return Case(ref=ref_fn, fast=fast_fn, inputs=[x], grads=False)

    p = Pair("_framework.shape", build, tol=1e-4)
    with pytest.raises(AssertionError, match="shape"):
        check(p, device)


def test_every_registered_pair_scaled_fails(device):
    assert PAIRS, "expected at least the toy pair"
    for name, p in list(PAIRS.items()):
        with pytest.raises(AssertionError):
            check(p, device, fast_override=scaled(1.01))
