"""Team 6 CorrectnessVerifier — ref vs fast, forward/backward, dtype tolerances.

Sibling to framework.py (Case/@pair checks). Optional class-style verifier with
gold_fp64, PASS max_rel logging, and fp8_tol placeholders.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch

TensorLike = Union[torch.Tensor, Any]
FnType = Callable[..., Any]


def _clone_inputs(
    inputs: Sequence[Any],
    *,
    force_grad: bool = False,
    dtype: Optional[torch.dtype] = None,
) -> List[Any]:
    out: List[Any] = []
    for x in inputs:
        if isinstance(x, torch.Tensor):
            want = dtype if (dtype is not None and x.is_floating_point()) else x.dtype
            y = x.detach().to(dtype=want).clone()
            if force_grad and y.is_floating_point():
                y.requires_grad_(True)
            elif x.requires_grad and y.is_floating_point():
                y.requires_grad_(True)
            out.append(y)
        else:
            out.append(x)
    return out


def _as_tensor_list(obj: Any) -> List[torch.Tensor]:
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (tuple, list)):
        tensors = [t for t in obj if isinstance(t, torch.Tensor)]
        if not tensors:
            raise TypeError("output contained no tensors")
        return tensors
    raise TypeError(f"unsupported output type: {type(obj)}")


def _metrics(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = torch.abs(a.detach() - b.detach())
    denom = torch.abs(b.detach()) + 1e-8
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "max_rel": float((diff / denom).max().item()),
    }


def make_perturbed_fn(fn: FnType, scale: float = 1.01) -> FnType:
    """Wrap fn so every tensor output is multiplied by scale (Gate 1 negative test)."""

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        out = fn(*args, **kwargs)
        if isinstance(out, torch.Tensor):
            return out * scale
        if isinstance(out, tuple):
            return tuple(t * scale if isinstance(t, torch.Tensor) else t for t in out)
        if isinstance(out, list):
            return [t * scale if isinstance(t, torch.Tensor) else t for t in out]
        return out

    return wrapped


class CorrectnessVerifier:
    """Compare reference vs fast implementations within dtype-specific tolerances."""

    def __init__(
        self,
        atol_fp32: float = 1e-5,
        rtol_fp32: float = 1e-5,
        atol_bf16: float = 1e-2,
        rtol_bf16: float = 1e-2,
        atol_fp8: float = 1e-1,
        rtol_fp8: float = 1e-1,
        gold_fp64: bool = False,
    ) -> None:
        self.gold_fp64 = gold_fp64
        # Placeholders until a real fp8 kernel is calibrated (see tech doc).
        self.fp8_tol: Tuple[float, float] = (atol_fp8, rtol_fp8)
        self.tolerances: Dict[torch.dtype, Tuple[float, float]] = {
            torch.float32: (atol_fp32, rtol_fp32),
            torch.float64: (atol_fp32, rtol_fp32),
            torch.bfloat16: (atol_bf16, rtol_bf16),
            torch.float16: (atol_bf16, rtol_bf16),
        }

    def _tol(self, dtype: torch.dtype) -> Tuple[float, float]:
        return self.tolerances.get(dtype, (1e-3, 1e-3))

    def _tol_fp8(self) -> Tuple[float, float]:
        return self.fp8_tol

    def _resolve_gold(self, gold_fp64: Optional[bool]) -> bool:
        return self.gold_fp64 if gold_fp64 is None else gold_fp64

    def _promote_pair(
        self,
        ref: torch.Tensor,
        fast: torch.Tensor,
        *,
        gold_fp64: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if gold_fp64:
            return (
                ref.detach().to(dtype=torch.float64),
                fast.detach().to(dtype=torch.float64),
            )
        return ref, fast

    def _compare_pair(
        self,
        ref: torch.Tensor,
        fast: torch.Tensor,
        *,
        test_name: str,
        tag: str,
        idx: int,
        gold_fp64: bool = False,
    ) -> Tuple[bool, Dict[str, Any]]:
        ref_c, fast_c = self._promote_pair(ref, fast, gold_fp64=gold_fp64)

        if ref_c.shape != fast_c.shape:
            detail = {
                "tag": tag,
                "idx": idx,
                "pass": False,
                "error": f"shape {tuple(ref_c.shape)} vs {tuple(fast_c.shape)}",
            }
            print(f"[FAIL] {test_name} ({tag} #{idx}) shape mismatch: {detail['error']}")
            return False, detail
        if ref_c.dtype != fast_c.dtype:
            detail = {
                "tag": tag,
                "idx": idx,
                "pass": False,
                "error": f"dtype {ref_c.dtype} vs {fast_c.dtype}",
            }
            print(f"[FAIL] {test_name} ({tag} #{idx}) dtype mismatch: {detail['error']}")
            return False, detail

        atol, rtol = self._tol(ref_c.dtype)
        m = _metrics(ref_c, fast_c)
        ok = bool(torch.allclose(ref_c, fast_c, atol=atol, rtol=rtol))
        detail = {
            "tag": tag,
            "idx": idx,
            "pass": ok,
            "dtype": str(ref_c.dtype),
            "atol": atol,
            "rtol": rtol,
            **m,
        }
        if ok:
            print(
                f"[PASS] {test_name} ({tag} #{idx}) | dtype={ref_c.dtype} | "
                f"max_abs={m['max_abs']:.6e} max_rel={m['max_rel']:.6e}"
            )
        else:
            print(
                f"[FAIL] {test_name} ({tag} #{idx}) | dtype={ref_c.dtype} | "
                f"max_abs={m['max_abs']:.6e} mean_abs={m['mean_abs']:.6e} "
                f"max_rel={m['max_rel']:.6e} | atol={atol} rtol={rtol}"
            )
        return ok, detail

    def verify_forward(
        self,
        ref_fn: FnType,
        fast_fn: FnType,
        inputs: Sequence[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        test_name: str = "Kernel Test",
        gold_fp64: Optional[bool] = None,
    ) -> bool:
        if kwargs is None:
            kwargs = {}
        use_gold = self._resolve_gold(gold_fp64)

        ref_dtype = torch.float64 if use_gold else None
        ref_inputs = _clone_inputs(inputs, force_grad=False, dtype=ref_dtype)
        fast_inputs = _clone_inputs(inputs, force_grad=False)

        ref_out = ref_fn(*ref_inputs, **kwargs)
        fast_out = fast_fn(*fast_inputs, **kwargs)

        ref_tensors = _as_tensor_list(ref_out)
        fast_tensors = _as_tensor_list(fast_out)
        if len(ref_tensors) != len(fast_tensors):
            print(
                f"[FAIL] {test_name} output arity "
                f"{len(ref_tensors)} vs {len(fast_tensors)}"
            )
            return False

        passed = True
        for idx, (r, f) in enumerate(zip(ref_tensors, fast_tensors)):
            ok, _ = self._compare_pair(
                r, f, test_name=test_name, tag="fwd", idx=idx, gold_fp64=use_gold
            )
            passed = passed and ok
        return passed

    def verify_backward(
        self,
        ref_fn: FnType,
        fast_fn: FnType,
        inputs: Sequence[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        test_name: str = "Gradient Test",
        grad_out: Optional[torch.Tensor] = None,
        gold_fp64: Optional[bool] = None,
    ) -> bool:
        if kwargs is None:
            kwargs = {}
        use_gold = self._resolve_gold(gold_fp64)

        ref_dtype = torch.float64 if use_gold else None
        ref_inputs = _clone_inputs(inputs, force_grad=True, dtype=ref_dtype)
        fast_inputs = _clone_inputs(inputs, force_grad=True)

        ref_out = ref_fn(*ref_inputs, **kwargs)
        fast_out = fast_fn(*fast_inputs, **kwargs)

        ref_tensors = _as_tensor_list(ref_out)
        fast_tensors = _as_tensor_list(fast_out)
        if len(ref_tensors) != len(fast_tensors):
            print(f"[FAIL] {test_name} output arity mismatch on backward path")
            return False

        if len(ref_tensors) == 1:
            rt0 = ref_tensors[0]
            ft0 = fast_tensors[0]
            if grad_out is None:
                g64 = torch.ones(rt0.shape, dtype=torch.float64, device=rt0.device)
            else:
                g64 = grad_out.detach().to(dtype=torch.float64, device=rt0.device)
            rt0.backward(g64.to(dtype=rt0.dtype))
            ft0.backward(g64.to(dtype=ft0.dtype, device=ft0.device))
        else:
            ref_loss = sum(t.sum() for t in ref_tensors)
            fast_loss = sum(t.sum() for t in fast_tensors)
            ref_loss.backward()
            fast_loss.backward()

        passed = True
        for idx, (r_in, f_in) in enumerate(zip(ref_inputs, fast_inputs)):
            if not isinstance(r_in, torch.Tensor) or r_in.grad is None:
                continue
            if f_in.grad is None:
                print(f"[FAIL] {test_name} (grad #{idx}) fast grad is None")
                passed = False
                continue
            ok, _ = self._compare_pair(
                r_in.grad,
                f_in.grad,
                test_name=test_name,
                tag="grad",
                idx=idx,
                gold_fp64=use_gold,
            )
            passed = passed and ok
        return passed

    def verify_all(
        self,
        ref_fn: FnType,
        fast_fn: FnType,
        inputs: Sequence[Any],
        kwargs: Optional[Dict[str, Any]] = None,
        test_name: str = "Kernel+Grad Test",
        gold_fp64: Optional[bool] = None,
    ) -> bool:
        fwd = self.verify_forward(
            ref_fn,
            fast_fn,
            inputs,
            kwargs,
            test_name=f"{test_name}/fwd",
            gold_fp64=gold_fp64,
        )
        bwd = self.verify_backward(
            ref_fn,
            fast_fn,
            inputs,
            kwargs,
            test_name=f"{test_name}/bwd",
            gold_fp64=gold_fp64,
        )
        return fwd and bwd

    def assert_forward(self, *args: Any, **kwargs: Any) -> None:
        if not self.verify_forward(*args, **kwargs):
            raise AssertionError(
                f"forward correctness failed: {kwargs.get('test_name', args[-1] if args else '')}"
            )

    def assert_backward(self, *args: Any, **kwargs: Any) -> None:
        if not self.verify_backward(*args, **kwargs):
            raise AssertionError(f"backward correctness failed: {kwargs.get('test_name', '')}")

    def assert_all(self, *args: Any, **kwargs: Any) -> None:
        if not self.verify_all(*args, **kwargs):
            raise AssertionError(f"verify_all failed: {kwargs.get('test_name', '')}")
