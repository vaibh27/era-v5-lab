"""Tests for FLOP, hardware-utilization and unavailable-reason metrics."""
import pytest

from models.config import ModelConfig
from models.gpt import GPT
from perfkit.metrics import flops_per_token, mfu, peak_tflops, peak_unavailable_reason
from perfkit.schema import example_run, validate_run


def _tiny_model(**overrides):
    cfg = ModelConfig(vocab_size=32, block_size=8, n_layer=2, n_head=2, n_embd=8, **overrides)
    return GPT(cfg)


def test_flops_per_token_matches_documented_model_and_hardware_formulas():
    model = _tiny_model()
    seq_len = 5
    _, active = model.param_counts()
    head = model.head.weight.numel()
    model_fpt = 6 * active + 12 * model.cfg.n_layer * model.cfg.n_embd * seq_len
    stack_fwd = 2 * (active - head) + 4 * model.cfg.n_layer * model.cfg.n_embd * seq_len

    result = flops_per_token(model, seq_len, opt_flops_per_step=600, tokens_per_step=10)

    assert result["model_flops_per_token"] == model_fpt
    assert result["hw_flops_per_token"] == pytest.approx(model_fpt + 60)
    assert result["params_total"] == sum(p.numel() for p in model.parameters())
    assert result["params_active"] == active
    assert stack_fwd > 0  # documents the reversible/chunked extra-work basis below


@pytest.mark.parametrize(
    ("overrides", "extra"),
    [({"reversible": True}, "reversible"), ({"ce_impl": "chunked"}, "recompute"),
     ({"ce_impl": "fused"}, "cached"), ({"ce_impl": "fused_cce"}, "recompute"),
     ({"ce_impl": "fused_liger"}, "cached")],
)
def test_flops_per_token_counts_extra_reversible_or_chunked_work(overrides, extra):
    model = _tiny_model(**overrides)
    seq_len = 5
    _, active = model.param_counts()
    head = model.head.weight.numel()
    ke = model.tok.proj.weight.numel() if model.cfg.embedding == "ke1" else 0
    model_fpt = 6 * active + 12 * model.cfg.n_layer * model.cfg.n_embd * seq_len
    expected = model_fpt
    if extra == "reversible":
        stack_fwd = 2 * (active - head - ke) + 4 * model.cfg.n_layer * model.cfg.n_embd * seq_len
        expected += 2 * stack_fwd
    elif extra == "recompute":
        expected += 2 * head
    result = flops_per_token(model, seq_len)
    assert result["model_flops_per_token"] == model_fpt
    assert result["hw_flops_per_token"] == expected


def test_mfu_and_hfu_use_model_and_hardware_flops_respectively():
    result = mfu({"model_flops_per_token": 2e9, "hw_flops_per_token": 3e9}, 1000, 100)
    assert result["model_tflops_per_s"] == pytest.approx(2.0)
    assert result["mfu"] == pytest.approx(0.02)
    assert result["hfu"] == pytest.approx(0.03)
    assert result["mfu_unavailable_reason"] is None
    assert result["hfu_unavailable_reason"] is None


def test_mfu_without_peak_leaves_utilization_unknown_but_reports_throughput():
    result = mfu({"model_flops_per_token": 2e9}, 1000, None)
    assert result == {
        "model_tflops_per_s": pytest.approx(2.0),
        "mfu": None,
        "hfu": None,
        "mfu_unavailable_reason": "peak_tflops_unavailable",
        "hfu_unavailable_reason": "peak_tflops_unavailable",
    }


def test_mfu_hfu_are_values_with_null_reasons_when_peak_is_known():
    out = mfu({"model_flops_per_token": 2e12, "hw_flops_per_token": 3e12}, 100, 100)

    assert out["mfu"] == 2.0
    assert out["hfu"] == 3.0
    assert out["mfu_unavailable_reason"] is None
    assert out["hfu_unavailable_reason"] is None


def test_missing_peak_sets_separate_reason_fields_for_both_metrics():
    out = mfu({"model_flops_per_token": 2e12, "hw_flops_per_token": 3e12}, 100, None,
              peak_reason="unsupported_dtype")

    assert out["mfu"] is None and out["hfu"] is None
    assert out["mfu_unavailable_reason"] == "unsupported_dtype"
    assert out["hfu_unavailable_reason"] == "unsupported_dtype"


def test_hfu_can_be_unavailable_while_mfu_is_available():
    out = mfu({"model_flops_per_token": 2e12}, 100, 100)

    assert out["mfu"] == 2.0
    assert out["mfu_unavailable_reason"] is None
    assert out["hfu"] is None
    assert out["hfu_unavailable_reason"] == "hw_flops_unavailable"


def test_peak_unavailable_reason_distinguishes_gpu_dtype_and_bad_override():
    assert peak_unavailable_reason("unlisted device", "bf16") == "unknown_gpu"
    assert peak_unavailable_reason("NVIDIA A100", "fp8") == "unsupported_dtype"
    assert peak_unavailable_reason("NVIDIA A100", "bf16", -1) == "invalid_peak_tflops"
    assert peak_unavailable_reason("NVIDIA A100", "bf16", 312) is None


def test_schema_accepts_null_metric_with_reason_and_rejects_unknown_reason():
    run = example_run()
    run["metrics"]["mfu"] = None
    run["metrics"]["mfu_unavailable_reason"] = "unknown_gpu"
    assert validate_run(run) == []

    run["metrics"]["hfu_unavailable_reason"] = "vendor-magic"
    assert any("metrics.hfu_unavailable_reason" in error for error in validate_run(run))


@pytest.mark.parametrize(
    ("gpu", "spec", "peaks"),
    [
        # Dense TFLOPS by dtype. Expected values are independent of perfkit.metrics.PEAKS.
        ("NVIDIA B200", "B200", {"bf16": 2250, "fp16": 2250}),
        ("NVIDIA B300", None, {}),
        ("NVIDIA H200 SXM", "H200", {"bf16": 989, "fp16": 989, "fp32": 67}),
        ("NVIDIA H200 NVL", "H200 NVL", {"bf16": 835.5, "fp16": 835.5, "fp32": 60}),
        ("NVIDIA H100 PCIe", "H100 PCIe", {"bf16": 756, "fp16": 756, "fp32": 51}),
        ("NVIDIA H100 SXM5 80GB", "H100", {"bf16": 989, "fp16": 989, "fp32": 67}),
        ("NVIDIA H100 NVL", "H100 NVL", {}),
        ("NVIDIA A100-SXM4-40GB", "A100", {"bf16": 312, "fp16": 312, "fp32": 19.5}),
        ("NVIDIA A100-SXM4-80GB", "A100", {"bf16": 312, "fp16": 312, "fp32": 19.5}),
        ("NVIDIA A100-PCIE-40GB", "A100", {"bf16": 312, "fp16": 312, "fp32": 19.5}),
        ("NVIDIA GeForce RTX 4090", "RTX 4090", {"bf16": 165, "fp16": 165, "fp32": 82.6}),
        ("NVIDIA GeForce RTX 3060", "RTX 3060", {"bf16": 50.5, "fp16": 50.5, "fp32": 13}),
        ("NVIDIA GeForce RTX 5070 Ti", None, {}),
        ("NVIDIA GeForce RTX 3090", "RTX 3090", {"bf16": 71, "fp16": 71, "fp32": 35.6}),
        ("NVIDIA L4", "L4", {"bf16": 121, "fp16": 121, "fp32": 30.3}),
        ("Tesla T4", "T4", {"fp16": 65, "fp32": 8.1}),
        ("NVIDIA V100", "V100", {"fp16": 125, "fp32": 15.7}),
        ("NVIDIA P100", "P100", {"fp16": 18.7, "fp32": 9.3}),
        ("NVIDIA A10G", "A10G", {}),
        ("NVIDIA A10", "A10", {"bf16": 125, "fp16": 125, "fp32": 31.2}),
        ("unlisted accelerator", None, {}),
    ],
)
def test_peak_tflops_for_gpu_name_across_supported_dtypes(gpu, spec, peaks):
    for dtype in ("bf16", "fp16", "fp32", "fp8", "fp4"):
        assert peak_tflops(gpu, dtype) == (peaks.get(dtype), spec), f"{gpu}, dtype={dtype}"
