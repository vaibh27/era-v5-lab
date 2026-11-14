"""Device selection, backend capability report, and separate smoke / measured modes for the Team 5 path.

    python runtime.py report --device auto
    python runtime.py run --mode smoke    --adapter-config CFG.yaml --device cpu --out DIR
    python runtime.py run --mode measured --adapter-config CFG.yaml --device cuda --out DIR

A dispatcher, not proof of support. What each backend has actually been shown to do is in the report:
    cpu    correctness fixtures (test_team5_adapter.py)
    mps    detected only; G5 forward and native sampling on MPS are untested here, so measured mode refuses it
    cuda   needs its own canary with the real G5 artifact; ROCm builds of torch report hip and stay unverified
    vllm / hf_batch samplers and Triton kernels: not available for G5's LanguageModel (no HF/vLLM model class,
           eager attention only); asking for them is an error, never a silent fallback to the native path.

Smoke mode accepts a fixture bundle and labels its output SMOKE_NOT_A_RESULT. Measured mode needs a READY real
G5 bundle, an explicit device (no auto), and refuses CPU/MPS unless --accept-non-gpu is given, in which case the
output says gpu_experiment=false. It times forward/backward and sampling; it does not train.

Exit codes: 0 ok, 2 bad request or backend unavailable, 3 WAITING_FOR_G5_ARTIFACT, 4 refused (fixture or
non-real bundle in measured mode).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

EXIT_OK, EXIT_BAD, EXIT_WAITING, EXIT_REFUSED = 0, 2, 3, 4
SAMPLERS = ("native", "hf_batch", "vllm")
KERNELS = ("eager", "triton")


class BackendError(RuntimeError):
    pass


def _has(pkg: str) -> bool:
    return importlib.util.find_spec(pkg) is not None  # does not import the package


def capability_report() -> dict:
    cuda = torch.cuda.is_available()
    hip = getattr(torch.version, "hip", None)
    rep = {
        "platform": platform.platform(), "machine": platform.machine(), "python": sys.version.split()[0],
        "torch": torch.__version__, "torch_cuda_build": torch.version.cuda, "torch_hip_build": hip,
        "cuda": {"available": cuda, "device_count": torch.cuda.device_count() if cuda else 0,
                 "name": torch.cuda.get_device_name(0) if cuda else None,
                 "capability": list(torch.cuda.get_device_capability(0)) if cuda else None,
                 "bf16": bool(cuda and torch.cuda.is_bf16_supported()),
                 "status": ("ROCm build: unverified until run on AMD hardware" if hip else
                            "needs a real-artifact canary; nothing measured by T0")},
        "mps": {"available": bool(torch.backends.mps.is_available()),
                "status": "detected only; G5 forward and native sampling untested on MPS"},
        "cpu": {"available": True, "status": "correctness fixtures only (test_team5_adapter.py)"},
        "packages": {p: _has(p) for p in ("vllm", "triton", "transformers", "safetensors", "tokenizers")},
        "g5_samplers": {"native": "supported (re-forward, no KV cache)",
                        "hf_batch": "unsupported: G5 LanguageModel has no generate()",
                        "vllm": "unsupported: no vLLM model class for G5 LanguageModel, no parity proof"},
        "g5_kernels": {"eager": "supported (G5 validate requires attn_impl eager)",
                       "triton": "unsupported: G5 harness has no Triton path"},
    }
    return rep


def resolve_device(requested: str) -> tuple[str, dict]:
    """Explicit device or auto (cuda > mps > cpu). Unavailable explicit devices are an error, never a fallback."""
    info = {"requested": requested}
    if requested == "cpu":
        dev = "cpu"
    elif requested == "mps":
        if not torch.backends.mps.is_available():
            raise BackendError("device mps requested but torch.backends.mps.is_available() is False")
        dev = "mps"
    elif requested == "cuda":
        if not torch.cuda.is_available():
            raise BackendError("device cuda requested but torch.cuda.is_available() is False")
        dev = "cuda"
    elif requested == "auto":
        dev = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        info["resolved_from_auto"] = True
    else:
        raise BackendError(f"unknown device {requested!r}; choose auto, cpu, mps or cuda")
    info["device"] = dev
    info["rocm_unverified"] = bool(dev == "cuda" and getattr(torch.version, "hip", None))
    return dev, info


def check_options(device: str, sampler: str, kernels: str) -> None:
    if sampler not in SAMPLERS:
        raise BackendError(f"unknown sampler {sampler!r}")
    if kernels not in KERNELS:
        raise BackendError(f"unknown kernels {kernels!r}")
    if sampler in ("vllm", "hf_batch"):
        need = "" if device == "cuda" or sampler == "hf_batch" else " (and vLLM needs CUDA)"
        raise BackendError(f"sampler {sampler} is not available for the G5 LanguageModel{need}; "
                           "use --sampler native. No silent fallback.")
    if kernels == "triton":
        why = "CUDA" if device != "cuda" else ("the triton package" if not _has("triton") else "a G5 Triton path")
        raise BackendError(f"kernels triton need {why}; G5 harness is eager-only. Use --kernels eager.")


def _make_generator(device: str, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(seed)  # on the sampling device


def _sync(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def _exercise(bundle, device: str, seed: int, repeats: int) -> dict:
    """One small forward/backward on trainable weights and one sample; returns timings and checks."""
    torch.manual_seed(seed)
    v = bundle.spec["vocab_size"]
    T = min(16, bundle.spec["block_size"])
    ids = torch.randint(0, v, (2, T))
    valid = torch.ones_like(ids, dtype=torch.bool)
    valid[1, T // 2:] = False
    fwd, bwd = [], []
    for _ in range(repeats):
        bundle.model.zero_grad(set_to_none=True)
        _sync(device); t0 = time.perf_counter()
        s, n = bundle.token_nll(ids, valid)
        _sync(device); t1 = time.perf_counter()
        (s / n).backward()
        _sync(device); t2 = time.perf_counter()
        fwd.append(t1 - t0); bwd.append(t2 - t1)
    grads = {name: p.grad for name, p in bundle.model.named_parameters() if p.requires_grad}
    finite = all(g is not None and bool(torch.isfinite(g).all()) for g in grads.values())
    router_grad = any(p.grad is not None for name, p in bundle.model.named_parameters()
                      if name in bundle.selection["router_names"])
    bundle.model.zero_grad(set_to_none=True)
    prompt = ids[0, :4].tolist()
    _sync(device); t0 = time.perf_counter()
    seq, cmask, reason = bundle.sample(prompt, 2, 4, 1.0, _make_generator(device, seed))
    _sync(device); ts = time.perf_counter() - t0
    return {"loss": float(s.detach() / n), "targets": n, "trainable_grads_finite": finite, "router_grad_present": router_grad,
            "forward_s": fwd, "backward_s": bwd, "sample_s": ts, "sample_tokens": int(cmask.sum()),
            "stop_reasons": reason}


def run(args) -> int:
    import team5_adapter as ta  # noqa: PLC0415
    try:
        if args.mode == "measured" and args.device == "auto":
            raise BackendError("measured mode needs an explicit --device (auto could silently pick another backend)")
        device, dinfo = resolve_device(args.device)
        check_options(device, args.sampler, args.kernels)
        if args.mode == "measured" and device in ("cpu", "mps") and not args.accept_non_gpu:
            raise BackendError(f"measured mode on {device} would not be the GPU experiment; pass --accept-non-gpu "
                               "to record it explicitly as gpu_experiment=false")
    except BackendError as e:
        print(json.dumps({"status": "BACKEND_ERROR", "error": str(e)}))
        return EXIT_BAD
    result = ta.build(args.adapter_config, device=device)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    head = {"mode": args.mode, "adapter_status": result.status, "adapter_mode": result.mode, "device": dinfo,
            "sampler": args.sampler, "kernels": args.kernels, "capabilities": capability_report()}
    if result.status == ta.WAITING:
        rec = head | {"status": ta.WAITING, "reasons": result.reasons}
        (out / f"{args.mode}.json").write_text(json.dumps(rec, indent=2, default=str))
        print(json.dumps({"status": ta.WAITING, "reasons": result.reasons}))
        return EXIT_WAITING
    if args.mode == "measured":
        try:
            bundle = ta.require_learning_ready(result)
        except RuntimeError as e:
            print(json.dumps({"status": "REFUSED", "error": str(e)}))
            return EXIT_REFUSED
        status = "MEASURED"
    else:
        bundle = result.require_bundle()
        status = "SMOKE_NOT_A_RESULT"
    checks = _exercise(bundle, device, args.seed, args.repeats)
    rec = head | {"status": status, "gpu_experiment": device == "cuda" and args.mode == "measured",
                  "fixture": result.mode == "fixture", "checks": checks,
                  "manifest_status": result.manifest["status"], "model_hash_g5": result.manifest["model_hash_g5"]}
    (out / f"{args.mode}.json").write_text(json.dumps(rec, indent=2, default=str))
    print(json.dumps({"status": status, "device": device, "fixture": rec["fixture"],
                      "grads_finite": checks["trainable_grads_finite"], "router_grad": checks["router_grad_present"]}))
    return EXIT_OK if checks["trainable_grads_finite"] and not checks["router_grad_present"] else EXIT_BAD


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    rp = sub.add_parser("report", help="print the backend capability report")
    rp.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    rr = sub.add_parser("run", help="smoke or measured run on the adapter bundle")
    rr.add_argument("--mode", required=True, choices=["smoke", "measured"])
    rr.add_argument("--adapter-config", required=True)
    rr.add_argument("--device", required=True, choices=["auto", "cpu", "mps", "cuda"])
    rr.add_argument("--sampler", default="native", choices=list(SAMPLERS))
    rr.add_argument("--kernels", default="eager", choices=list(KERNELS))
    rr.add_argument("--out", required=True)
    rr.add_argument("--seed", type=int, default=1234)
    rr.add_argument("--repeats", type=int, default=1)
    rr.add_argument("--accept-non-gpu", action="store_true")
    args = ap.parse_args(argv)
    if args.cmd == "report":
        rep = capability_report()
        try:
            _, rep["selected"] = resolve_device(args.device)
        except BackendError as e:
            rep["selected"] = {"requested": args.device, "error": str(e)}
            print(json.dumps(rep, indent=2))
            return EXIT_BAD
        print(json.dumps(rep, indent=2))
        return EXIT_OK
    return run(args)


if __name__ == "__main__":
    sys.exit(main())
