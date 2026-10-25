"""Reference-vs-fast checks.  [CONTRACT: Case / Pair / pair registry + checks]

Registering a pair (anyone, in correctness/test_<topic>.py):

    @pair("moe.loop", tol=1e-4, grad_tol=1e-3)
    def _(device):
        ...
        return Case(ref=ref_fn, fast=fast_fn, inputs=[x], params_ref=[...], params_fast=[...])

Rules: the reference is written independently from the definition, in fp64, never by calling the
fast code. tolerances < 1e-2 always (so a 1.01 error fails).

MVP
  * rel_err(fast, ref) = ||fast - ref|| / ||ref|| in fp64; AssertionError on shape mismatch or NaN/Inf.
  * check(pair, device, fast_override=None): build the Case; clone the inputs (fp64 copy for ref);
    compare every output; backpropagate the SAME random cotangent through both; compare the grads of
    inputs that require grad and of params_fast vs params_ref. AssertionError naming the pair, the
    tensor, the error and the tolerance. Returns {name: error}.
  * scaled(k=1.01): returns a wrapper factory; check applies fast_override(case.fast). Outputs * k.
Do not use torch.allclose defaults: in bf16 (rtol 1.6e-2) they let a 1% error pass.

Precision: pairs compare the fast code in fp32 against the fp64 reference, so tol < 1e-2 is easy and the
1.01 drill bites. bf16 errors are often larger than 1e-2 / 3 (bf16 has 8 mantissa bits), so do NOT make a
bf16 @pair with a loosened tolerance. Check bf16 behaviour with plain property tests and a stated
threshold (e.g. reversible reconstruction_error < 1e-3), outside the 1.01 drill. Decision D2.
"""
from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

PAIRS: dict = {}
DEFAULT_TOL = 1e-4


@dataclass
class Case:
    ref: Callable
    fast: Callable
    inputs: list
    params_ref: list = field(default_factory=list)
    params_fast: list = field(default_factory=list)
    grads: bool = True


@dataclass
class Pair:
    name: str
    build: Callable          # device -> Case
    tol: float = DEFAULT_TOL
    grad_tol: float | None = None


def pair(name, tol=DEFAULT_TOL, grad_tol=None):
    assert tol < 1e-2 and (grad_tol is None or grad_tol < 1e-2), f"{name}: tolerance must be < 1e-2"

    def deco(fn):
        assert name not in PAIRS, f"duplicate pair {name}"
        PAIRS[name] = Pair(name, fn, tol, grad_tol)
        return fn
    return deco


def pairs_in(module_name):
    """Names of the pairs a test module registered (for pytest.mark.parametrize)."""
    return [n for n, p in PAIRS.items() if p.build.__module__ == module_name]


def to64(module):
    """fp64 deep copy of a module, for references."""
    return copy.deepcopy(module).double()


def _as_tensor_list(obj: Any) -> list[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        tensors = [t for t in obj if isinstance(t, torch.Tensor)]
        if not tensors:
            raise TypeError("output contained no tensors")
        return tensors
    raise TypeError(f"unsupported output type: {type(obj)}")


def _clone_inputs(inputs: list, *, dtype: torch.dtype, device) -> list:
    out: list = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            want = dtype if x.is_floating_point() else x.dtype
            y = x.detach().to(device=device, dtype=want).clone()
            if x.requires_grad and y.is_floating_point():
                y.requires_grad_(True)
            out.append(y)
        else:
            out.append(x)
    return out


def _zero_grads(tensors: list) -> None:
    for t in tensors:
        if isinstance(t, torch.Tensor) and t.grad is not None:
            t.grad = None


def rel_err(fast, ref) -> float:
    if not isinstance(fast, torch.Tensor) or not isinstance(ref, torch.Tensor):
        raise AssertionError(f"rel_err expects tensors, got {type(fast)} vs {type(ref)}")
    if fast.shape != ref.shape:
        raise AssertionError(
            f"shape mismatch: fast {tuple(fast.shape)} vs ref {tuple(ref.shape)}"
        )
    f = fast.detach().to(dtype=torch.float64)
    r = ref.detach().to(dtype=torch.float64)
    if not torch.isfinite(f).all() or not torch.isfinite(r).all():
        raise AssertionError("NaN/Inf in fast or ref")
    diff = f - r
    if not torch.isfinite(diff).all():
        raise AssertionError("NaN/Inf in difference")
    ref_norm = torch.linalg.vector_norm(r)
    diff_norm = torch.linalg.vector_norm(diff)
    if ref_norm.item() == 0.0:
        if diff_norm.item() == 0.0:
            return 0.0
        raise AssertionError("ref norm is zero but fast differs")
    err = (diff_norm / ref_norm).item()
    if not math.isfinite(err):
        raise AssertionError("NaN/Inf relative error")
    return float(err)


def scaled(k=1.01):
    """Return a factory: scaled(k)(fast_fn) multiplies every tensor output by k."""

    def decorate(fn):
        def wrapped(*args, **kwargs):
            out = fn(*args, **kwargs)
            if isinstance(out, torch.Tensor):
                return out * k
            if isinstance(out, tuple):
                return tuple(t * k if isinstance(t, torch.Tensor) else t for t in out)
            if isinstance(out, list):
                return [t * k if isinstance(t, torch.Tensor) else t for t in out]
            return out

        return wrapped

    return decorate


def check(p: Pair, device, fast_override=None) -> dict:
    case = p.build(device)
    fast_fn = case.fast
    if fast_override is not None:
        fast_fn = fast_override(case.fast)

    ref_inputs = _clone_inputs(case.inputs, dtype=torch.float64, device=device)
    fast_inputs = _clone_inputs(case.inputs, dtype=torch.float32, device=device)

    ref_out = case.ref(*ref_inputs)
    fast_out = fast_fn(*fast_inputs)
    ref_tensors = _as_tensor_list(ref_out)
    fast_tensors = _as_tensor_list(fast_out)
    if len(ref_tensors) != len(fast_tensors):
        raise AssertionError(
            f"{p.name}: output arity {len(fast_tensors)} vs {len(ref_tensors)}"
        )

    errors: dict[str, float] = {}
    tol = p.tol
    for i, (ft, rt) in enumerate(zip(fast_tensors, ref_tensors)):
        name = f"out[{i}]"
        try:
            err = rel_err(ft, rt)
        except AssertionError as exc:
            raise AssertionError(f"{p.name}: {name}: {exc}") from exc
        if err > tol:
            raise AssertionError(
                f"{p.name}: {name}: rel_err={err:.6e} > tol={tol}"
            )
        errors[name] = err

    if not case.grads:
        return errors

    grad_tol = p.grad_tol if p.grad_tol is not None else p.tol
    _zero_grads(ref_inputs + fast_inputs + list(case.params_ref) + list(case.params_fast))

    cotangents = [
        torch.randn(rt.shape, dtype=torch.float64, device=rt.device)
        for rt in ref_tensors
    ]
    n = len(ref_tensors)
    for i, (rt, ft, g64) in enumerate(zip(ref_tensors, fast_tensors, cotangents)):
        retain = i < n - 1
        rt.backward(g64.to(dtype=rt.dtype), retain_graph=retain)
        ft.backward(
            g64.to(dtype=ft.dtype, device=ft.device),
            retain_graph=retain,
        )

    for i, (ri, fi) in enumerate(zip(ref_inputs, fast_inputs)):
        if not isinstance(ri, torch.Tensor) or not ri.requires_grad:
            continue
        name = f"grad_in[{i}]"
        if ri.grad is None or fi.grad is None:
            raise AssertionError(f"{p.name}: {name}: missing grad (ref={ri.grad is not None}, fast={fi.grad is not None})")
        try:
            err = rel_err(fi.grad, ri.grad)
        except AssertionError as exc:
            raise AssertionError(f"{p.name}: {name}: {exc}") from exc
        if err > grad_tol:
            raise AssertionError(
                f"{p.name}: {name}: rel_err={err:.6e} > grad_tol={grad_tol}"
            )
        errors[name] = err

    if len(case.params_ref) != len(case.params_fast):
        raise AssertionError(
            f"{p.name}: params arity {len(case.params_fast)} vs {len(case.params_ref)}"
        )
    for i, (pr, pf) in enumerate(zip(case.params_ref, case.params_fast)):
        name = f"grad_param[{i}]"
        if pr.grad is None or pf.grad is None:
            raise AssertionError(f"{p.name}: {name}: missing param grad")
        try:
            err = rel_err(pf.grad, pr.grad)
        except AssertionError as exc:
            raise AssertionError(f"{p.name}: {name}: {exc}") from exc
        if err > grad_tol:
            raise AssertionError(
                f"{p.name}: {name}: rel_err={err:.6e} > grad_tol={grad_tol}"
            )
        errors[name] = err

    return errors
