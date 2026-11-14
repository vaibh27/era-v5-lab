"""Tests for sft/train_probe.py. CPU only; synthetic integer fixtures only (no OCR-2 rows, no tokenizer)."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
from pathlib import Path

import pytest

module_fixture = pytest.fixture(scope="module")
parametrize = pytest.mark.parametrize
import torch

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("g10_train_probe", HERE / "train_probe.py")
tp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tp)

CONFIG = HERE / "config" / "probe.yaml"
IGN = tp.IGNORE


@module_fixture
def cfg() -> dict:
    return tp.load_config(CONFIG)


@module_fixture
def tiny_cfg(cfg) -> dict:
    return dict(cfg, d_model=16, n_heads=2, n_layers=1, max_len=32)


def row(ids, mask):
    return {"input_ids": list(ids), "loss_mask": list(mask)}


# ---------------------------------------------------------------- config


def test_config_loads_and_respects_update_cap(cfg):
    assert cfg["max_steps"] <= tp.MAX_UPDATES == 30
    assert cfg["pad_id"] != cfg["eos_id"]


@parametrize("change", [{"max_steps": 31}, {"max_steps": 0}, {"pad_id": 999}, {"d_model": 30}])
def test_config_rejects_bad_values(cfg, change):
    with pytest.raises(ValueError):
        tp.check_config(dict(cfg, **change))


def test_config_rejects_unknown_and_missing_keys(cfg):
    with pytest.raises(ValueError, match="unknown"):
        tp.check_config(dict(cfg, lr_typo=1.0))
    broken = dict(cfg)
    del broken["lr"]
    with pytest.raises(ValueError, match="missing"):
        tp.check_config(broken)


# ---------------------------------------------------------------- row validation and labels


def test_validate_accepts_good_row():
    r = row([5, 6, 7, 3], [0, 0, 1, 1])
    assert tp.validate_example(r) is r


@parametrize(
    "bad",
    [row([5, 6, 7], [0, 1]), row([5, 6, 7], [0, 2, 1]), {"input_ids": [1, 2]}],
)
def test_validate_rejects_malformed(bad):
    with pytest.raises(ValueError):
        tp.validate_example(bad)


def test_all_masked_row_rejected():
    with pytest.raises(tp.ZeroTargetError):
        tp.validate_example(row([5, 6, 7, 8], [0, 0, 0, 0]))


def test_mask_only_on_position_zero_has_no_shifted_target():
    # Position 0 has no preceding context, so it can never be a target after the causal shift.
    with pytest.raises(tp.ZeroTargetError):
        tp.validate_example(row([5, 6, 7], [1, 0, 0]))


def test_single_token_row_rejected():
    with pytest.raises(tp.ZeroTargetError):
        tp.validate_example(row([5], [1]))


def test_out_of_vocab_id_rejected():
    with pytest.raises(ValueError, match="vocabulary"):
        tp.validate_example(row([5, 70], [0, 1]), vocab_size=64)


def test_shift_labels_targets_are_next_tokens_where_masked():
    ids = [10, 11, 12, 13, 14]
    mask = [0, 0, 1, 1, 0]
    assert tp.shift_labels(ids, mask) == [IGN, 12, 13, IGN]


def test_shift_label_count_equals_mask_sum_after_position_zero():
    ids = list(range(10, 20))
    mask = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]
    labels = tp.shift_labels(ids, mask)
    assert len(labels) == len(ids) - 1
    assert sum(y != IGN for y in labels) == sum(mask[1:])


def test_truncation_removing_assistant_is_rejected():
    r = row([5, 6, 7, 8, 9, 10], [0, 0, 0, 0, 1, 1])
    with pytest.raises(tp.ZeroTargetError):
        tp.truncate(r, 4)


def test_truncation_keeping_some_targets_is_kept():
    r = row([5, 6, 7, 8, 9, 10], [0, 0, 1, 1, 1, 1])
    out = tp.truncate(r, 4)
    assert out["input_ids"] == [5, 6, 7, 8] and sum(out["loss_mask"][1:]) == 2


def test_tool_call_then_masked_tool_result():
    # assistant call (1) -> tool result (0) -> assistant reply (1)
    ids = [2, 6, 8, 20, 21, 3, 4, 2, 7, 30, 31, 3, 4, 2, 6, 40, 41, 3]
    mask = [0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
    labels = tp.shift_labels(ids, mask)
    targets = [y for y in labels if y != IGN]
    assert targets == [8, 20, 21, 3, 40, 41, 3]
    assert 30 not in targets and 31 not in targets  # tool result tokens never become targets


def test_multiple_assistant_turns_all_counted():
    ids = [5, 9, 6, 20, 21, 3, 5, 9, 6, 30, 3]
    mask = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1]
    b = tp.collate([row(ids, mask)], pad_id=0, eos_id=1)
    assert b.n_targets == 5


# ---------------------------------------------------------------- collate, EOS vs pad


def test_collate_unequal_lengths_pads_right_and_ignores_pads():
    rows = [row([5, 6, 7, 3], [0, 0, 1, 1]), row([5, 6, 3], [0, 1, 1]), row([5, 6, 7, 8, 9, 3], [0, 0, 0, 1, 1, 1])]
    b = tp.collate(rows, pad_id=0, eos_id=1)
    assert b.input_ids.shape == (3, 6) and b.labels.shape == (3, 5)
    assert b.attention_mask.sum(1).tolist() == [4, 3, 6]
    assert (b.labels[1, 2:] == IGN).all() and (b.kinds[1, 2:] == tp.KIND_PAD).all()
    assert b.n_targets == 2 + 2 + 3


def test_padding_comes_from_length_not_token_value():
    # A real token that happens to equal pad_id must still be a target when marked.
    b = tp.collate([row([5, 0, 3], [0, 1, 1]), row([5, 6], [0, 1])], pad_id=0, eos_id=1)
    assert b.labels[0].tolist() == [0, 3]
    assert b.kinds[0].tolist() == [tp.KIND_TARGET, tp.KIND_TARGET]
    assert b.kinds[1].tolist() == [tp.KIND_TARGET, tp.KIND_PAD]


def test_pad_equal_to_eos_keeps_supervised_eos():
    # OLMoE-style native template: EOS is a trained turn ending; pad shares its id.
    eos = pad = 1
    rows = [row([5, 6, 7, eos], [0, 0, 1, 1]), row([5, 6, 7, 8, 9, eos], [0, 0, 1, 1, 1, 1])]
    b = tp.collate(rows, pad_id=pad, eos_id=eos)
    assert b.labels[0].tolist() == [IGN, 7, eos, IGN, IGN]
    assert b.kinds[0].tolist() == [tp.KIND_CONTEXT, tp.KIND_TARGET, tp.KIND_EOS_TARGET, tp.KIND_PAD, tp.KIND_PAD]
    assert int((b.kinds == tp.KIND_EOS_TARGET).sum()) == 2
    # the padded slots hold the same id as EOS but are not targets
    assert b.input_ids[0, 4:].tolist() == [pad, pad] and b.attention_mask[0, 4:].tolist() == [0, 0]


def test_unsupervised_final_eos_is_context_not_target():
    # G10 chat-format v0: <|im_end|> (3) trained, newline (4) and final <|endoftext|> (eos=1) not trained.
    b = tp.collate([row([2, 6, 20, 3, 4, 1], [0, 0, 1, 1, 0, 0])], pad_id=0, eos_id=1)
    assert b.kinds[0].tolist() == [tp.KIND_CONTEXT, tp.KIND_TARGET, tp.KIND_TARGET, tp.KIND_CONTEXT, tp.KIND_CONTEXT]
    assert int((b.kinds == tp.KIND_EOS_TARGET).sum()) == 0


def test_collate_empty_batch_rejected():
    with pytest.raises(ValueError):
        tp.collate([], 0, 1)


# ---------------------------------------------------------------- loss


def test_masked_nll_sum_matches_manual_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(1, 3, 5)
    labels = torch.tensor([[2, IGN, 4]])
    total, count = tp.masked_nll_sum(logits, labels)
    lp = torch.log_softmax(logits, -1)
    assert count == 2
    assert torch.isclose(total, -(lp[0, 0, 2] + lp[0, 2, 4]))


def test_loss_is_global_token_mean_not_mean_of_row_means():
    torch.manual_seed(1)
    logits = torch.randn(2, 4, 7)
    labels = torch.tensor([[1, IGN, IGN, IGN], [2, 3, 4, 5]])
    total, count = tp.masked_nll_sum(logits, labels)
    per_tok = torch.nn.functional.cross_entropy(logits.reshape(-1, 7), labels.reshape(-1), ignore_index=IGN, reduction="none").reshape(2, 4)
    row_means = torch.stack([per_tok[0, :1].mean(), per_tok[1].mean()]).mean()
    assert count == 5
    assert torch.isclose(total / count, per_tok.sum() / 5)
    assert not torch.isclose(total / count, row_means)  # the two reductions really differ here


def test_masked_positions_do_not_affect_loss():
    torch.manual_seed(2)
    logits = torch.randn(1, 4, 6)
    labels = torch.tensor([[IGN, 3, IGN, 5]])
    base, _ = tp.masked_nll_sum(logits, labels)
    noisy = logits.clone()
    noisy[0, 0] += 100.0
    noisy[0, 2] -= 50.0
    assert torch.equal(tp.masked_nll_sum(noisy, labels)[0], base)


def test_gradient_is_zero_at_masked_positions():
    logits = torch.randn(1, 4, 6, requires_grad=True)
    labels = torch.tensor([[IGN, 3, IGN, 5]])
    total, _ = tp.masked_nll_sum(logits, labels)
    total.backward()
    g = logits.grad[0]
    assert g[0].abs().sum() == 0 and g[2].abs().sum() == 0
    assert g[1].abs().sum() > 0 and g[3].abs().sum() > 0


def test_zeroing_one_mask_marker_removes_exactly_that_target(tiny_cfg):
    # Zeroing the mask on a token removes it as a target: the loss no longer counts it.
    model = tp.build_model(tiny_cfg)
    ids = [5, 6, 20, 21, 3]
    full = tp.collate([row(ids, [0, 0, 1, 1, 1])], 0, 1)
    zeroed = tp.collate([row(ids, [0, 0, 1, 0, 1])], 0, 1)
    with torch.no_grad():
        lf = tp.shifted_logits(model, full)
        tf, cf = tp.masked_nll_sum(lf, full.labels)
        tz, cz = tp.masked_nll_sum(tp.shifted_logits(model, zeroed), zeroed.labels)
        nll_21 = torch.nn.functional.cross_entropy(lf[0, 2:3], torch.tensor([21]), reduction="sum")
    assert (cf, cz) == (3, 2)
    assert torch.isclose(tf - tz, nll_21, atol=1e-6)


# ---------------------------------------------------------------- tiny model


def test_model_is_causal(tiny_cfg):
    model = tp.build_model(tiny_cfg).eval()
    a = torch.tensor([[5, 6, 7, 8, 9]])
    b = torch.tensor([[5, 6, 7, 40, 41]])
    with torch.no_grad():
        la, lb = model(a), model(b)
    assert torch.allclose(la[0, :3], lb[0, :3], atol=1e-6)
    assert not torch.allclose(la[0, 3:], lb[0, 3:])


def test_right_padding_does_not_change_real_logits(tiny_cfg):
    model = tp.build_model(tiny_cfg).eval()
    short = row([5, 6, 7, 3], [0, 0, 1, 1])
    long = row([5, 6, 7, 8, 9, 10, 3], [0, 0, 1, 1, 1, 1, 1])
    with torch.no_grad():
        alone = model(tp.collate([short], 0, 1).input_ids)
        batched = model(tp.collate([short, long], 0, 1).input_ids)
    assert torch.allclose(alone[0], batched[0, :4], atol=1e-5)


def test_no_dropout_train_mode_is_deterministic(tiny_cfg):
    model = tp.build_model(tiny_cfg).train()
    x = torch.tensor([[5, 6, 7, 8]])
    assert torch.equal(model(x), model(x))


def test_forward_backward_finite_and_every_param_gets_a_gradient(cfg):
    model = tp.build_model(cfg)
    batch = tp.collate(tp.synthetic_rows(cfg)[:3], cfg["pad_id"], cfg["eos_id"])
    loss = tp.batch_loss(model, batch)
    loss.backward()
    assert math.isfinite(loss.item())
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    # an untrained model over vocab V starts near ln(V)
    assert abs(loss.item() - math.log(cfg["vocab_size"])) < 1.5


def test_uneven_accumulation_matches_one_shot_gradient(tiny_cfg):
    rows = [
        row([5, 6, 7, 3], [0, 0, 1, 1]),
        row([5, 6, 9, 10, 11, 12, 3], [0, 0, 1, 1, 1, 1, 1]),
        row([5, 13, 3], [0, 1, 1]),
        row([5, 6, 7, 8, 14, 3], [0, 0, 0, 0, 1, 1]),
        row([5, 6, 15, 16, 17, 3], [0, 1, 1, 0, 1, 1]),
    ]
    model = tp.build_model(tiny_cfg)
    one = tp.collate(rows, 0, 1)
    model.zero_grad()
    tp.batch_loss(model, one).backward()
    ref = [p.grad.clone() for p in model.parameters()]

    model.zero_grad()
    mbs = [tp.collate(rows[:1], 0, 1), tp.collate(rows[1:4], 0, 1), tp.collate(rows[4:], 0, 1)]
    loss, count = tp.accumulate_gradients(model, mbs)
    assert count == one.n_targets
    for g_ref, p in zip(ref, model.parameters()):
        assert torch.allclose(g_ref, p.grad, atol=1e-6, rtol=1e-5)
    with torch.no_grad():
        assert math.isclose(loss, tp.batch_loss(model, one).item(), rel_tol=1e-5)


def test_naive_per_microbatch_mean_would_differ(tiny_cfg):
    # Guards the reason for global-count division: averaging microbatch means weights tokens unequally.
    rows = [row([5, 6, 3], [0, 1, 1]), row([5, 6, 9, 10, 11, 12, 3], [0, 0, 1, 1, 1, 1, 1])]
    model = tp.build_model(tiny_cfg)
    mbs = [tp.collate(rows[:1], 0, 1), tp.collate(rows[1:], 0, 1)]
    with torch.no_grad():
        naive = sum(tp.batch_loss(model, mb).item() for mb in mbs) / 2
        right = tp.batch_loss(model, tp.collate(rows, 0, 1)).item()
    assert not math.isclose(naive, right, rel_tol=1e-4)


# ---------------------------------------------------------------- synthetic fixture and loader


def test_synthetic_fixture_shape(cfg):
    rows = tp.synthetic_rows(cfg)
    assert len(rows) == cfg["fixture_rows"]
    assert len({len(r["input_ids"]) for r in rows}) > 1  # unequal lengths
    for r in rows:
        tp.validate_example(r, cfg["vocab_size"])
        assert r["input_ids"][-1] == cfg["eos_id"] and r["loss_mask"][-1] == 0
    assert any(8 in r["input_ids"] and 7 in r["input_ids"] for r in rows)  # tool call + tool result
    assert tp.synthetic_rows(cfg) == rows  # deterministic


def test_load_jsonl_counts_rejects(tmp_path, cfg):
    p = tmp_path / "rows.jsonl"
    lines = [
        row([5, 6, 7, 3], [0, 0, 1, 1]),
        row([5, 6, 7, 3], [0, 0, 0, 0]),  # all masked
        row([5, 6, 7], [0, 1]),  # length mismatch
        dict(row([5, 6, 7, 3], [0, 1, 1, 1]), text="ignored", id="synthetic/3"),
    ]
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n", encoding="utf-8")
    rows, rejected = tp.load_jsonl(p, cfg)
    assert len(rows) == 2 and rejected == {"zero_targets": 1, "malformed": 1}
    assert set(rows[1]) == {"input_ids", "loss_mask"}


# ---------------------------------------------------------------- the probe itself


@module_fixture
def report():
    cfg = tp.load_config(CONFIG)
    return cfg, tp.run_probe(cfg, tp.synthetic_rows(cfg))


def test_probe_overfits_fixed_batch(report):
    cfg, r = report
    assert r["all_finite"] and r["overfit_pass"]
    assert r["final_eval_loss"] < cfg["overfit_ratio"] * r["initial_loss"]
    assert r["losses"][-1] < r["losses"][0]


def test_probe_loss_falls_after_warmup(report):
    _, r = report
    after = r["losses"][5:]
    assert all(b < a for a, b in zip(after, after[1:]))


def test_probe_stores_fp32_and_weights_change(report):
    _, r = report
    assert r["param_dtype"] == "torch.float32"
    assert r["weight_change_l2"] > 0


def test_probe_logs_targets_grad_norms_and_lrs(report):
    cfg, r = report
    rows = tp.synthetic_rows(cfg)
    expected = sum(sum(x["loss_mask"][1:]) for x in rows)
    assert r["steps"] == cfg["max_steps"] and r["microbatches"] == 3
    assert r["targets_per_update"] == [expected] * cfg["max_steps"]
    assert len(r["grad_norms"]) == cfg["max_steps"] and all(g > 0 and math.isfinite(g) for g in r["grad_norms"])
    w = cfg["warmup_steps"]
    assert r["lrs"][:w] == pytest.approx([cfg["lr"] * (i + 1) / w for i in range(w)])
    assert r["lrs"][w:] == pytest.approx([cfg["lr"]] * (cfg["max_steps"] - w))
    assert r["pad_positions"] > 0 and r["eos_targets"] == 0  # G10 contract: final EOS untrained


def test_probe_is_reproducible(cfg):
    small = dict(cfg, max_steps=3)
    a = tp.run_probe(small, tp.synthetic_rows(small))
    b = tp.run_probe(small, tp.synthetic_rows(small))
    assert a["losses"] == b["losses"]


def test_probe_with_pad_equal_eos_and_supervised_eos(cfg):
    small = dict(cfg, pad_id=1, eos_id=1, max_steps=5)
    rows = [dict(r, loss_mask=r["loss_mask"][:-1] + [1]) for r in tp.synthetic_rows(small)]
    r = tp.run_probe(small, rows)
    assert r["eos_targets"] == len(rows) and r["all_finite"]


def test_probe_rejects_zero_target_row(cfg):
    with pytest.raises(tp.ZeroTargetError):
        tp.run_probe(dict(cfg, max_steps=1), [row([5, 6, 7], [0, 0, 0])])


def test_cli_main_returns_zero(capsys):
    assert tp.main(["--config", str(CONFIG)]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["overfit_pass"] is True and out["rejected"] == {}


# ---------------------------------------------------------------- scope guard


def test_probe_imports_no_frozen_tokenizer_or_ocr2_code():
    tree = ast.parse((HERE / "train_probe.py").read_text(encoding="utf-8"))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
    assert names <= {"__future__", "argparse", "json", "math", "random", "sys", "dataclasses", "pathlib", "torch", "torch.nn", "torch.nn.functional", "yaml"}
    src = (HERE / "train_probe.py").read_text(encoding="utf-8")
    for banned in ("verify_frozen_tokenizer", "pack_admission", "prepare_ocr2", "run_ocr2", "ocr2_fixtures", "importlib", "tokenizers"):
        assert banned not in src, banned
