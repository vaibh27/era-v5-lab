"""Marker rows as their own tensor (--train-marker-rows) and bf16 storage of frozen Linear weights (--frozen-bf16).

Both change only where trained or frozen values are stored, not the computation. The reference for the marker rows is
the earlier implementation (the whole embedding / LM head trainable, with a gradient hook that zeroes every other row),
kept here as `old_marker_rows`.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

_parametrize = pytest.mark.parametrize  # undotted alias, as in the other Team 10 test files

BENCH = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BENCH))


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


train = _load("g10_bench_train_mr", BENCH / "train.py")
IDS = torch.tensor([[3, 7, 9, 11, 13, 5, 6, 8], [4, 4, 10, 12, 1, 2, 3, 0]])
MARKERS = [5, 9, 12]


def tiny(tied=True, seed=0, bf16_exact=False):
    from transformers import LlamaConfig, LlamaForCausalLM  # noqa: PLC0415

    cfg = LlamaConfig(vocab_size=64, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, tie_word_embeddings=tied)
    torch.manual_seed(seed)
    m = LlamaForCausalLM(cfg).float()
    if bf16_exact:  # what a bf16 checkpoint loaded in FP32 looks like
        with torch.no_grad():
            for p in m.parameters():
                p.copy_(p.to(torch.bfloat16).float())
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def old_marker_rows(model, ids, scope):
    """The earlier enable_marker_rows: whole matrix trainable, other rows' gradient zeroed by a hook."""
    emb, head = model.get_input_embeddings().weight, model.lm_head.weight
    chosen = {"both": (emb, head), "input": (emb,), "output": (head,)}[scope]
    out = []
    for w in {id(t): t for t in chosen}.values():
        rowmask = torch.zeros(w.shape[0], 1, dtype=w.dtype)
        rowmask[ids] = 1
        w.requires_grad_(True)
        w.register_hook(lambda g, m=rowmask: g * m)
        out.append(w)
    return out


def effective(model):
    """(embedding, head) matrices as the forward sees them."""
    return model.get_input_embeddings().weight.detach().clone(), model.lm_head.weight.detach().clone()


@_parametrize("clip", [None, 1e-3], ids=["no-clip", "clip-active"])
@_parametrize("tied,scope", [(True, "both"), (False, "both"), (False, "input"), (False, "output")])
def test_marker_rows_give_the_same_losses_and_rows_as_the_whole_matrix_version(tied, scope, clip):
    ref, new = tiny(tied), tiny(tied)
    ref_params = old_marker_rows(ref, MARKERS, scope)
    new_params = train.enable_marker_rows(new, MARKERS, scope)
    runs = []
    for m, params in ((ref, ref_params), (new, new_params)):
        opt = torch.optim.AdamW(params, lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.0)
        losses = []
        for _ in range(4):
            opt.zero_grad(set_to_none=True)
            loss = m(input_ids=IDS, labels=IDS).loss
            loss.backward()
            if clip is not None:  # the trainer clips every step; 1e-3 makes the clip act on every step here
                torch.nn.utils.clip_grad_norm_(params, clip)
            opt.step()
            losses.append(loss.item())
        runs.append((losses, effective(m), opt))
    assert runs[0][0] == runs[1][0]
    for a, b in zip(runs[0][1], runs[1][1]):
        assert torch.equal(a, b)
    base = effective(tiny(tied))
    moved = [not torch.equal(a[MARKERS], b[MARKERS]) for a, b in zip(runs[1][1], base)]
    assert any(moved)
    for a, b in zip(runs[1][1], base):  # every other row is bit-identical to the pretrained one
        keep = torch.ones(a.shape[0], dtype=torch.bool)
        keep[MARKERS] = False
        assert torch.equal(a[keep], b[keep])
    # the memory point: AdamW state for len(ids) rows, not for the whole matrix
    new_state = sum(s["exp_avg"].numel() + s["exp_avg_sq"].numel() for s in runs[1][2].state.values())
    ref_state = sum(s["exp_avg"].numel() + s["exp_avg_sq"].numel() for s in runs[0][2].state.values())
    assert new_state == 2 * len(MARKERS) * 32 * len(new_params) and ref_state == 2 * 64 * 32 * len(ref_params)


@_parametrize("tied", [True, False])
def test_merge_marker_rows_restores_plain_keys_values_and_tie(tied):
    m = tiny(tied)
    keys = set(m.state_dict())
    params = train.enable_marker_rows(m, MARKERS, "both")
    with torch.no_grad():
        for p in params:
            p.add_(1.0)
    want = effective(m)
    train.merge_marker_rows(m)
    assert set(m.state_dict()) == keys
    for a, b in zip(effective(m), want):
        assert torch.equal(a, b)
    assert (m.lm_head.weight is m.get_input_embeddings().weight) == tied
    assert not any(p.requires_grad for p in m.parameters())


def test_frozen_sha256_ignores_the_marker_rows_and_sees_the_frozen_matrix():
    m = tiny(True)
    params = train.enable_marker_rows(m, MARKERS, "both")
    before = train.frozen_sha256(m, MARKERS)
    with torch.no_grad():
        params[0].add_(1.0)  # training the rows leaves every frozen tensor alone
    assert train.frozen_sha256(m, MARKERS) == before


def test_frozen_sha256_still_covers_the_frozen_embedding_matrix_in_marker_runs():
    m = tiny(True)
    train.enable_marker_rows(m, MARKERS, "both")
    before = train.frozen_sha256(m, MARKERS)
    with torch.no_grad():
        m.get_input_embeddings().parametrizations.weight.original[0, 0] += 1.0  # a non-marker row of the frozen matrix
    assert train.frozen_sha256(m, MARKERS) != before


def test_deletion_stats_do_not_count_the_marker_rows_twice():
    m = tiny(False)
    train.dtmask.convert(m, ["gate_proj"], 100.0, 1000.0, 10)
    want = train.dtmask.deletion_stats(m)["model_weights"]
    train.enable_marker_rows(m, MARKERS, "both")
    assert train.dtmask.deletion_stats(m)["model_weights"] == want


def dt_model(tied=False):
    m = tiny(tied, bf16_exact=True)
    train.dtmask.convert(m, ["gate_proj"], 100.0, 1000.0, 10)
    gen = torch.Generator().manual_seed(1)
    with torch.no_grad():
        for _, layer in train.dtmask.masked_layers(m):
            layer.gate.copy_(torch.randn(layer.gate.shape, generator=gen) * 0.02)
    return m


def lora_model(tied=False):
    m = tiny(tied, bf16_exact=True)
    train.apply_method(m, "lora", SimpleNamespace(lora_r=4, lora_alpha=8.0, seed=3))
    return m


def test_frozen_bf16_converts_only_frozen_plain_linears_and_round_trips():
    m = dt_model(tied=False)
    sd = {k: v.clone() for k, v in m.state_dict().items()}
    info = train.frozen_linears_to_bf16(m)
    dtypes = {n: p.dtype for n, p in m.named_parameters()}
    for n, dt in dtypes.items():
        plain = any(n.endswith(s) for s in ("q_proj.weight", "k_proj.weight", "v_proj.weight", "o_proj.weight",
                                            "up_proj.weight", "down_proj.weight", "lm_head.weight"))
        assert dt == (torch.bfloat16 if plain else torch.float32), n  # gate_proj (MaskedLinear), logits, norms, emb
    assert info["weights"] == sum(p.numel() for n, p in m.named_parameters() if p.dtype == torch.bfloat16)
    assert info["bytes_saved"] == 2 * info["weights"]
    train.frozen_linears_to_fp32(m)
    after = m.state_dict()
    assert set(after) == set(sd) and all(torch.equal(after[k], sd[k]) and after[k].dtype == sd[k].dtype for k in sd)


def test_frozen_bf16_refuses_weights_that_are_not_exact_in_bf16():
    m = tiny(False)  # FP32 values: not representable in bf16
    with pytest.raises(SystemExit, match="not exactly representable"):
        train.frozen_linears_to_bf16(m)
    assert all(p.dtype == torch.float32 for p in m.parameters())  # nothing converted


def test_frozen_bf16_keeps_a_tied_head_and_marker_matrices_in_fp32():
    m = tiny(True, bf16_exact=True)
    train.enable_marker_rows(m, MARKERS, "both")
    train.frozen_linears_to_bf16(m)
    assert m.get_input_embeddings().parametrizations.weight.original.dtype == torch.float32
    assert m.lm_head.parametrizations.weight.original.dtype == torch.float32
    assert m.model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16


def test_frozen_bf16_leaves_a_tied_head_and_the_embedding_in_fp32():
    m = tiny(True, bf16_exact=True)
    info = train.frozen_linears_to_bf16(m)
    assert m.lm_head.weight is m.get_input_embeddings().weight and m.lm_head.weight.dtype == torch.float32
    assert info["linears"] == 2 * 7  # q k v o gate up down per layer; the tied head is the embedding
    assert m.model.layers[1].mlp.down_proj.weight.dtype == torch.bfloat16


@_parametrize("tied", [True, False], ids=["tied", "untied"])
@_parametrize("build", [dt_model, lora_model], ids=["dt", "lora"])
def test_frozen_bf16_gives_identical_loss_and_gradients_under_bf16_autocast(build, tied):
    a = build(tied)
    b = copy.deepcopy(a)
    train.frozen_linears_to_bf16(b)
    out = []
    for m in (a, b):
        m.train()
        with torch.autocast("cpu", dtype=torch.bfloat16):
            loss = m(input_ids=IDS, labels=IDS).loss
        loss.backward()
        out.append((loss.item(), {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}))
    assert out[0][0] == out[1][0]
    assert out[0][1].keys() == out[1][1].keys() and out[0][1]
    for n in out[0][1]:
        assert torch.equal(out[0][1][n], out[1][1][n]), n


def test_frozen_bf16_flag_needs_bf16_autocast_and_frozen_weights():
    base = ["--train", "t", "--model", "m", "--revision", "none", "--device", "cpu", "--out", "o", "--frozen-bf16"]
    with pytest.raises(SystemExit, match="--frozen-bf16 needs"):
        train.train(train.parse_args([*base, "--method", "dt"]))
    with pytest.raises(SystemExit, match="--frozen-bf16 needs"):
        train.train(train.parse_args([*base, "--method", "full", "--autocast", "bf16"]))


# --- the flag through train.train (the same code path as on CUDA; CPU autocast through the AUTOCAST device seam) ----


@pytest.fixture(scope="module")
def bench_dir(tmp_path_factory):
    import json  # noqa: PLC0415
    import random  # noqa: PLC0415

    d = tmp_path_factory.mktemp("frozen_bf16")
    m = tiny(True, bf16_exact=True)
    m.save_pretrained(d / "base")
    rng = random.Random(0)
    marker_ids = sorted(i for _, i in train.retok_markers() if i < 64)
    with open(d / "train.tok.jsonl", "w", encoding="utf-8") as f:
        for i in range(24):
            ids = [rng.randrange(3, 64) for _ in range(20)]
            ids[3], ids[9] = marker_ids[i % len(marker_ids)], marker_ids[(i + 1) % len(marker_ids)]
            row = {"id": f"r{i}", "split": "train", "input_ids": ids, "loss_mask": [0] * 4 + [1] * 16}
            f.write(json.dumps(row) + "\n")
    (d / "tokdir").mkdir()
    return d


def run_cpu_autocast(d, out, method, extra, monkeypatch):
    monkeypatch.setitem(train.AUTOCAST, "device", "cpu")
    monkeypatch.setitem(train.AUTOCAST, "dtype", None)
    args = train.parse_args(["--train", str(d / "train.tok.jsonl"), "--model", str(d / "base"), "--revision", "none",
                             "--tokenizer-dir", str(d / "tokdir"), "--method", method, "--steps", "5", "--warmup", "2",
                             "--lr", "1e-2", "--seed", "3", "--device", "cpu", "--micro-tokens", "64",
                             "--autocast", "bf16", "--out", str(d / out), "--log-every", "100", *extra])
    return train.train(args)


@_parametrize("method,extra", [("lora", ["--grad-checkpointing"]), ("dt", []),
                                ("markers", ["--train-marker-rows", "--marker-lr", "1e-2", "--wd", "0"]),
                                ("lora", ["--untie-head", "--train-marker-rows", "--marker-lr", "1e-2", "--wd", "0",
                                          "--marker-scope", "output"])],
              ids=["lora-ckpt", "dt", "markers", "lora-untied-output-markers"])
def test_frozen_bf16_through_train_gives_the_same_run(bench_dir, method, extra, monkeypatch):
    tag = method + "-" + "-".join(e.strip("-") for e in extra if e.startswith("--"))
    a = run_cpu_autocast(bench_dir, tag + "-fp32", method, extra, monkeypatch)
    b = run_cpu_autocast(bench_dir, tag + "-bf16", method, [*extra, "--frozen-bf16"], monkeypatch)
    assert [r["loss"] for r in a["log"]] == [r["loss"] for r in b["log"]]
    assert [r["grad_norm"] for r in a["log"]] == [r["grad_norm"] for r in b["log"]]
    assert a["saved_state_sha256"] == b["saved_state_sha256"]
    assert b["method"]["frozen_bf16"]["linears"] > 0 and "frozen_bf16" not in a["method"]
    assert b["checks"]["frozen_weights_unchanged"]["pass"] and b["checks"]["reload_parity"]["pass"]
    assert "stored in bf16" in b["dtype"]


def test_resume_key_is_unchanged_when_the_new_flag_is_off():
    args = train.parse_args(["--train", "t", "--model", "m", "--out", "o"])
    key = train.resume_key(args)
    assert "frozen_bf16" not in key
    args.frozen_bf16 = True
    assert "frozen_bf16" in train.resume_key(args)
