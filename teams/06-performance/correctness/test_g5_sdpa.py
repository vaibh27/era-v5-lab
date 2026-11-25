"""g5_switches attn_impl=sdpa on the G5 trainer, with no edit to teams/05-architecture.

  * the patch is pinned to the G5 attention code it was checked against (fails here first if G5 changes it)
  * eager (the default) patches nothing; global layers under sdpa match G5's eager value and gradients
  * local (windowed) layers keep G5's eager path; no parameter or state change
  * train.py: --g6 parsing, G5's trainer runs with the switch, the switch is in the manifest and run identity,
    and G5's modules are restored afterwards
  * CUDA: bf16 close to eager, backward repeats bitwise in deterministic mode, Flash serves the call
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch.nn import functional as F

PERF = Path(__file__).resolve().parents[1]
G5 = PERF.parent / "05-architecture"
sys.path.insert(0, str(G5 / "harness"))
sys.path.insert(0, str(G5 / "tests"))
sys.path.insert(0, str(PERF))

import g5_switches  # noqa: E402
from g5_switches import train as g6_train  # noqa: E402
from test_week2_model import fixture  # noqa: E402
from week2 import model as g5_model  # noqa: E402
from week2.model import GQA, LanguageModel  # noqa: E402

SDPA = {"attn_impl": "sdpa"}


@pytest.fixture
def sdpa_installed():
    g5_switches.install(SDPA)
    yield
    g5_switches.uninstall()


def spec(n_kv_head=2, **extra):
    c, rows, lengths = fixture()
    c["model"].update(n_kv_head=n_kv_head, **extra)
    return c, rows, lengths


def loss_and_grads(model, ids, valid):
    model.zero_grad(set_to_none=True)
    logits = model(ids, valid)
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), (ids + 1).reshape(-1) % 31)
    loss.backward()
    return logits.detach(), {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


# ---------------------------------------------------------------- switches and pin

def test_patch_is_pinned_to_the_current_g5_attention():
    assert g5_switches.g5_forward_sha256(g5_model) == g5_switches.G5_GQA_FORWARD_SHA256, (
        "G5 changed GQA.forward: re-check g5_switches._sdpa_forward against it, then update G5_GQA_FORWARD_SHA256")


def test_changed_g5_attention_stops_the_sdpa_patch():
    with patch.object(g5_switches, "G5_GQA_FORWARD_SHA256", "0" * 64):
        with pytest.raises(RuntimeError, match="G5 changed GQA.forward"):
            g5_switches.install(SDPA)
    assert not hasattr(GQA, "_g5_forward")


def test_parse_defaults_and_refusals():
    assert g5_switches.parse([]) == {"attn_impl": "eager"}
    assert g5_switches.parse(["attn_impl=sdpa"]) == SDPA
    for bad in ["attn_impl=flash", "attn=sdpa", "attn_impl"]:
        with pytest.raises(ValueError):
            g5_switches.parse([bad])


def test_eager_patches_nothing():
    forward = GQA.forward
    g5_switches.install({"attn_impl": "eager"})
    assert GQA.forward is forward and not hasattr(GQA, "_g5_forward")


def test_install_is_idempotent_and_uninstall_restores(sdpa_installed):
    original = GQA._g5_forward
    g5_switches.install(SDPA)
    assert GQA._g5_forward is original
    g5_switches.uninstall()
    assert GQA.forward is original
    g5_switches.install(SDPA)   # the fixture uninstalls again


# ---------------------------------------------------------------- parity (CPU)

@pytest.mark.parametrize("n_kv", [1, 2, 4])
def test_sdpa_matches_eager_value_and_gradients(n_kv):
    torch.manual_seed(1234)
    c, rows, lengths = spec(n_kv_head=n_kv)
    model = LanguageModel(c["model"], rows, lengths)
    ids = torch.arange(16).reshape(2, 8) % 31
    valid = torch.ones_like(ids, dtype=torch.bool)
    ref_logits, ref_grads = loss_and_grads(model, ids, valid)
    g5_switches.install(SDPA)
    try:
        logits, grads = loss_and_grads(model, ids, valid)
    finally:
        g5_switches.uninstall()
    torch.testing.assert_close(logits, ref_logits, rtol=1e-5, atol=1e-5)
    assert grads.keys() == ref_grads.keys()
    for name in ref_grads:
        torch.testing.assert_close(grads[name], ref_grads[name], rtol=1e-4, atol=1e-5, msg=name)


def test_local_layers_keep_the_g5_eager_path_bitwise():
    c, _, _ = spec(attn_pattern=["L", "G"], attn_window=3, attn_sink=True)
    torch.manual_seed(1234)
    local = GQA(c["model"], 0)
    torch.nn.init.normal_(local.sink)
    x = torch.randn(2, 8, 16)
    ref = local(x)
    g5_switches.install(SDPA)
    try:
        out = local(x)
    finally:
        g5_switches.uninstall()
    assert local.local and torch.equal(out, ref)


def test_sdpa_adds_no_parameters_or_state(sdpa_installed):
    c, rows, lengths = spec()
    with_patch = LanguageModel(c["model"], rows, lengths)
    g5_switches.uninstall()
    plain = LanguageModel(c["model"], rows, lengths)
    g5_switches.install(SDPA)
    assert {k: v.shape for k, v in with_patch.state_dict().items()} == {
        k: v.shape for k, v in plain.state_dict().items()}


# ---------------------------------------------------------------- launcher

def test_split_args():
    pairs, rest = g6_train.split_args(["--g6", "attn_impl=sdpa", "--config", "a.yaml", "--g6=attn_impl=eager", "--smoke"])
    assert pairs == ["attn_impl=sdpa", "attn_impl=eager"]
    assert rest == ["--config", "a.yaml", "--smoke"]


def smoke_run(root, name, g6_args):
    """One G5 smoke run through g5_switches/train.py on the G5 test fixture; returns (manifest, forwards seen)."""
    import train_week2

    cfg, byte_rows, _ = fixture()
    cfg["model"]["ke_max_pos"] = byte_rows.shape[1]
    cfg["run"].update(batch_size=2, micro_batch_size=1)
    artifacts = SimpleNamespace(
        bytes=byte_rows, lengths=np.ones(len(byte_rows), dtype=np.int16), pos_dim=byte_rows.shape[1],
        bos_id=0, pad_id=1, encode=lambda text: [ord(ch) - ord("a") + 2 for ch in text],
        identity=lambda: {}, normalize_input=False,
    )
    train, val = root / "train.jsonl", root / "val.jsonl"
    train.write_text('{"text":"aaaa"}\n{"text":"aaa"}\n', encoding="utf8")
    val.write_text('{"text":"bbbb"}\n', encoding="utf8")
    seen = []
    g5_build = train_week2.LanguageModel

    def build(*args, **kwargs):
        seen.append(GQA.forward)
        return g5_build(*args, **kwargs)

    argv = [*g6_args, "--config", "fixture.yaml", "--tokenizer", "fixture.json", "--byte-table", "fixture.npz",
            "--train-data", str(train), "--heldout-data", str(val), "--smoke", "--smoke-steps", "2",
            "--no-checkpoint", "--out", str(root / name)]
    with (
        patch.object(train_week2, "load_runtime", return_value=cfg),
        patch.object(train_week2, "Artifacts", return_value=artifacts),
        patch.object(train_week2, "LanguageModel", build),
        patch.object(sys, "argv", ["train.py"]),
        patch.object(sys, "path", [str(G5.parent / "07-infrastructure"), *sys.path]),
    ):
        g6_train.main(argv)
    return json.loads((root / name / "manifest.json").read_text()), seen


def test_launcher_runs_g5_with_the_switch_and_records_it():
    import train_week2

    g5_forward, g5_validate = GQA.forward, train_week2.validate
    with TemporaryDirectory() as temp:
        root = Path(temp)
        plain, seen_plain = smoke_run(root, "plain", [])
        sdpa, seen_sdpa = smoke_run(root, "sdpa", ["--g6", "attn_impl=sdpa"])
    assert seen_plain == [g5_forward] and seen_sdpa[0] is not g5_forward
    assert "g6_switches" not in plain["config"]["provenance"]
    record = sdpa["config"]["provenance"]["g6_switches"]
    assert record["switches"] == SDPA and set(record["sources"]) == {"__init__.py", "train.py"}
    assert plain["identity_sha256"] != sdpa["identity_sha256"]
    assert GQA.forward is g5_forward and train_week2.validate is g5_validate and not hasattr(GQA, "_g5_forward")


# ---------------------------------------------------------------- CUDA

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@pytest.fixture
def deterministic():
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(False)


def attention(seed=1234):
    torch.manual_seed(seed)
    c, _, _ = spec()
    c["model"].update(n_embd=256, n_head=4, n_kv_head=2, block_size=256)
    attn = GQA(c["model"], 0).cuda()
    x = torch.randn(2, 256, 256, device="cuda")
    return attn, x


def run_bf16(attn, x):
    attn.zero_grad(set_to_none=True)
    x = x.clone().requires_grad_()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = attn(x)
    out.float().square().mean().backward()
    return out.float(), x.grad, [p.grad.clone() for p in attn.parameters()]


@cuda
def test_bf16_sdpa_close_to_eager(deterministic):
    attn, x = attention()
    ref, ref_dx, _ = run_bf16(attn, x)
    g5_switches.install(SDPA)
    try:
        out, dx, _ = run_bf16(attn, x)
    finally:
        g5_switches.uninstall()
    torch.testing.assert_close(out, ref, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(dx, ref_dx, rtol=5e-2, atol=5e-3)


@cuda
def test_bf16_sdpa_backward_repeats_bitwise(deterministic, sdpa_installed):
    attn, x = attention()
    first, second = run_bf16(attn, x), run_bf16(attn, x)
    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])
    assert all(torch.equal(a, b) for a, b in zip(first[2], second[2]))


@cuda
def test_flash_alone_serves_the_trainer_call(deterministic, sdpa_installed):
    """Flash alone can serve the call: GQA, causal, bf16, deterministic mode, head dim 64."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    attn, x = attention()
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION]):
        out, dx, grads = run_bf16(attn, x)
    assert torch.isfinite(out).all() and torch.isfinite(dx).all() and all(torch.isfinite(g).all() for g in grads)


PROFILE_DEFAULT_BACKEND = """
import sys
import torch
from torch.profiler import ProfilerActivity, profile
sys.path.insert(0, sys.argv[1])
import test_g5_sdpa as t
torch.use_deterministic_algorithms(True)
t.g5_switches.install(t.SDPA)
attn, x = t.attention()
t.run_bf16(attn, x)
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    t.run_bf16(attn, x)
    torch.cuda.synchronize()
print("\\n".join(sorted({e.key for e in prof.key_averages()})))
"""


@cuda
def test_default_backend_is_flash():
    """With no backend forced (as in the trainer), the attention kernels are Flash, not the math fallback.
    Profiled in a fresh process: CUDA activity capture is unreliable once other tests in this process have
    run the profiler (it can record no kernels at all)."""
    import os
    import subprocess

    env = {**os.environ, "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG", ":4096:8")}
    r = subprocess.run([sys.executable, "-c", PROFILE_DEFAULT_BACKEND, str(Path(__file__).parent)],
                       capture_output=True, text=True, env=env, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    kernels = r.stdout.split("\n")
    assert len(kernels) > 3, "profiler recorded no CUDA kernels: " + r.stdout
    assert any("flash" in k.lower() for k in kernels), "; ".join(kernels)
