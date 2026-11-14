"""Tests for the GPU-run pieces: fp32 master weights, batched sampling, TIS, routing stability, bootstrap CI.

Everything here runs on a CPU with a tiny random OLMoE. The vLLM sampler needs a GPU and is exercised by the
Modal smoke run instead (rl-loop/modal/), not here. No decorators with a dot (pr-guard reads "+@a.b" as an email).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grpo  # noqa: E402
import heldout  # noqa: E402
import rollout  # noqa: E402
import routing  # noqa: E402
import train  # noqa: E402
from policy import hf_policy, use_fp32_master  # noqa: E402


def tiny_olmoe(eos=None, vocab=16):
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(0)
    return OlmoeForCausalLM(OlmoeConfig(vocab_size=vocab, hidden_size=32, intermediate_size=32, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=4, num_experts=4,
                                        num_experts_per_tok=2, max_position_embeddings=64, eos_token_id=eos,
                                        bos_token_id=None, pad_token_id=None))


# --- fp32 master weights ------------------------------------------------------------------------

def _changed_fraction_after_one_step(fp32_master: bool) -> float:
    model = tiny_olmoe().to(torch.bfloat16)
    for n, p in model.named_parameters():
        p.requires_grad_(".self_attn." in n)
    pol = hf_policy(model)
    if fp32_master:
        assert use_fp32_master(pol) > 0 and pol.amp_dtype == torch.bfloat16
    params = [p for p in model.parameters() if p.requires_grad]
    before = [p.detach().float().clone() for p in params]
    opt = torch.optim.AdamW(params, lr=1e-6, weight_decay=0.0)
    ids = torch.randint(0, 16, (4, 6))
    pol.logits(ids).float().pow(2).mean().backward()
    opt.step()
    changed = sum(int((p.detach().float() != b).sum()) for p, b in zip(params, before))
    return changed / sum(b.numel() for b in before)


def test_bf16_weights_lose_small_updates_and_fp32_masters_keep_them():
    assert _changed_fraction_after_one_step(fp32_master=True) > 0.9
    assert _changed_fraction_after_one_step(fp32_master=False) < 0.3


def test_fp32_master_is_a_no_op_for_an_fp32_model():
    pol = hf_policy(tiny_olmoe())
    assert use_fp32_master(pol) == 0 and pol.amp_dtype is None


def test_frozen_weights_stay_bf16_under_fp32_master():
    model = tiny_olmoe().to(torch.bfloat16)
    for n, p in model.named_parameters():
        p.requires_grad_(".self_attn." in n)
    use_fp32_master(hf_policy(model))
    for n, p in model.named_parameters():
        assert p.dtype == (torch.float32 if ".self_attn." in n else torch.bfloat16), n


# --- batched sampling ---------------------------------------------------------------------------

def test_left_padded_batch_matches_one_prompt_at_a_time():
    pol = hf_policy(tiny_olmoe())
    prompts = [[1, 2, 3, 4, 5], [6, 7], [8, 9, 10]]
    batched = rollout.HFBatchSampler(pol).sample(prompts, 2, 4, temperature=0.0)
    single = rollout.PolicySampler(pol).sample(prompts, 2, 4, temperature=0.0)
    for p, b, s in zip(prompts, batched, single):
        assert b.seq.shape == (2, len(p) + 4)
        assert torch.equal(b.seq, s.seq), (p, b.seq, s.seq)  # greedy: padding must not change the answer
        assert b.cmask.all() and b.rollout_lp is None


def test_completion_mask_keeps_eos_and_drops_what_follows():
    comp = torch.tensor([[5, 3, 9, 9], [3, 5, 5, 5], [5, 5, 5, 5]])
    m = rollout.completion_mask(comp, eos_id=3)
    assert m.tolist() == [[True, True, False, False], [True, False, False, False], [True] * 4]
    assert rollout.completion_mask(comp, None).all()


def test_unknown_sampler_is_an_error():
    with pytest.raises(ValueError):
        rollout.make_sampler("nope", hf_policy(tiny_olmoe()))
    with pytest.raises(ValueError):
        rollout.make_sampler("vllm", hf_policy(tiny_olmoe()))


# --- truncated importance sampling --------------------------------------------------------------

def test_tis_weights_are_one_when_sampler_equals_trainer():
    lp = torch.randn(3, 5)
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1], [1, 0, 0, 0, 0]], dtype=torch.bool)
    w = grpo.tis_weights(lp, lp, mask, "grpo", 2.0)
    assert torch.allclose(w, mask.float())
    assert torch.allclose(grpo.tis_weights(lp, lp, mask, "gspo", 2.0), torch.ones(3))
    m = grpo.mismatch(lp, lp, mask)
    assert m["k3"] == pytest.approx(0.0, abs=1e-7) and m["abs_logratio"] == pytest.approx(0.0, abs=1e-7)


def test_tis_weights_are_truncated():
    old = torch.zeros(1, 2)
    roll = torch.tensor([[-5.0, 0.5]])  # trainer far more likely on token 0, less likely on token 1
    w = grpo.tis_weights(old, roll, torch.ones(1, 2, dtype=torch.bool), "grpo", 2.0)
    assert w[0, 0] == pytest.approx(2.0) and w[0, 1] == pytest.approx(float(torch.exp(torch.tensor(-0.5))))


def test_tis_weight_scales_the_loss():
    lp = torch.zeros(2, 3)
    adv, mask = torch.tensor([1.0, -1.0]), torch.ones(2, 3, dtype=torch.bool)
    base, _ = grpo.policy_loss(lp, lp, adv, mask, "grpo")
    w = torch.tensor([[2.0] * 3, [1.0] * 3])
    scaled, _ = grpo.policy_loss(lp, lp, adv, mask, "grpo", is_weight=w)
    assert float(base) == pytest.approx(0.0) and float(scaled) == pytest.approx(-0.5)


# --- routing stability --------------------------------------------------------------------------

def test_routing_reads_top_k_per_layer_and_load_sums_to_one():
    pol = hf_policy(tiny_olmoe())
    seq = torch.randint(0, 16, (3, 7))
    w = routing.routing(pol, seq)
    assert w.shape == (2, 21, 4)  # 2 layers, 3 x 7 tokens, 4 experts (dense weights)
    assert ((w > 0).sum(-1) == 2).all()  # top-2: exactly two experts carry weight per token and layer
    load = routing.expert_load(w)
    assert torch.allclose(load.sum(-1), torch.ones(2))
    assert (routing.load_ratio(load) >= 1.0).all()


def test_routing_respects_the_token_mask():
    pol = hf_policy(tiny_olmoe())
    seq = torch.randint(0, 16, (2, 5))
    mask = routing.completion_positions(3, torch.ones(2, 2, dtype=torch.bool))
    assert routing.routing(pol, seq, mask).shape == (2, 4, 4)


def test_flip_rate_is_zero_for_the_same_model_and_positive_after_changing_attention():
    model = tiny_olmoe()
    pol = hf_policy(model)
    seq = torch.randint(0, 16, (4, 8))
    a = routing.routing(pol, seq)
    assert routing.flip_rate(a, routing.routing(pol, seq))[0] == 0.0
    with torch.no_grad():
        for n, p in model.named_parameters():
            if ".self_attn." in n:
                p.add_(torch.randn_like(p))
    b = routing.routing(pol, seq)
    any_layer, per_layer = routing.flip_rate(a, b)
    assert any_layer > 0 and len(per_layer) == 2
    assert 0 < routing.mass_shift(a, b) <= 1 and routing.mass_shift(a, a) == 0.0
    assert routing.top1_flip_rate(a, a) == 0.0


def _dense(rows):  # rows: per token, {expert: weight}; one layer, 4 experts
    t = torch.zeros(1, len(rows), 4)
    for i, r in enumerate(rows):
        for e, v in r.items():
            t[0, i, e] = v
    return t


def test_even_and_uneven_load_ratios():
    even = _dense([{0: .5, 1: .5}, {2: .5, 3: .5}])  # each expert picked once
    assert float(routing.load_ratio(routing.expert_load(even))[0]) == pytest.approx(1.0)
    skew = _dense([{0: .5, 1: .5}, {0: .5, 2: .5}])  # expert 0 gets 2 of 4 picks: share 0.5 vs mean 0.25
    assert float(routing.load_ratio(routing.expert_load(skew))[0]) == pytest.approx(2.0)


def test_a_near_tie_swap_flips_the_set_but_moves_little_mass():
    before = _dense([{0: .90, 1: .05}])
    after = _dense([{0: .90, 2: .05}])  # the small second expert swapped: the set changed, top-1 did not
    assert routing.flip_rate(before, after)[0] == 1.0
    assert routing.top1_flip_rate(before, after) == 0.0
    assert routing.mass_shift(before, after) == pytest.approx(0.05 / 0.95)
    swapped_top = _dense([{1: .90, 0: .05}])  # same set, top-1 changed, most mass moved
    assert routing.flip_rate(before, swapped_top)[0] == 0.0 and routing.top1_flip_rate(before, swapped_top) == 1.0
    assert routing.mass_shift(before, swapped_top) == pytest.approx(0.85 / 0.95)


def test_a_model_without_router_modules_has_no_routing_log():
    from torch import nn

    class NoRouter(nn.Module):
        def forward(self, ids, mask):
            return torch.zeros(*ids.shape, 4)

    from policy import module_policy
    assert routing.routing(module_policy(NoRouter()), torch.zeros(1, 3, dtype=torch.long)) is None


# --- bootstrap ----------------------------------------------------------------------------------

def test_bootstrap_of_identical_scores_is_zero_and_does_not_exclude_zero():
    s = [0.0, 0.25, 0.5, 1.0] * 50
    r = heldout.paired_bootstrap(s, s)
    assert r["diff"] == 0.0 and r["ci_low"] == 0.0 and r["ci_high"] == 0.0 and not r["ci_excludes_zero"]


def test_bootstrap_of_a_constant_shift_is_that_shift():
    s = [0.0, 0.25, 0.5, 0.75] * 50
    r = heldout.paired_bootstrap(s, [x + 0.1 for x in s])
    assert r["diff"] == pytest.approx(0.1) and r["ci_low"] == pytest.approx(0.1) and r["ci_excludes_zero"]


def test_bootstrap_of_noise_has_an_interval_around_zero():
    g = torch.Generator().manual_seed(3)
    s = torch.rand(200, generator=g).tolist()
    e = [x + float(n) for x, n in zip(s, torch.randn(200, generator=g) * 0.05)]
    r = heldout.paired_bootstrap(s, e)
    assert r["ci_low"] < r["diff"] < r["ci_high"] and r["ci_high"] - r["ci_low"] < 0.05


def test_bootstrap_needs_paired_scores():
    with pytest.raises(ValueError):
        heldout.paired_bootstrap([0.1], [0.1, 0.2])


# --- end to end with the new logs ---------------------------------------------------------------

def test_smoke_with_routing_log_kl_to_start_and_bf16(tmp_path):
    cfg = yaml.safe_load((HERE / "config" / "smoke_tiny_olmoe.yaml").read_text())
    tiny_olmoe().to(torch.bfloat16).save_pretrained(tmp_path / "ckpt")
    cfg.update(out_dir=str(tmp_path / "out"), verbose=False,
               model={"kind": "hf", "path": str(tmp_path / "ckpt"), "dtype": "bfloat16"})
    cfg["train"].update(steps=20, eval_every=10, routing_log=True, routing_probe=4, kl_to_start=True,
                        trainable=[r"\.self_attn\."])
    s = train.train(cfg)
    assert s["router_unchanged"] and s["fp32_master_params"] > 0 and s["sampler"] == "hf_batch"
    curve = s["routing_curve"]
    assert [c["step"] for c in curve] == [0, 10, 20]
    assert curve[0]["flip_vs_step0"] == 0.0 and curve[0]["kl_to_start"] == pytest.approx(0.0, abs=1e-6)
    assert curve[0]["mass_shift_vs_step0"] == 0.0 and curve[0]["flip_noise_floor"] == 0.0
    assert curve[-1]["kl_to_start"] > 0  # attention moved, so the policy moved away from the start
    assert all(c["load_ratio_max"] >= 1.0 for c in curve)
    hc = s["heldout_change"]
    assert hc["problems"] == 64 and hc["ci_low"] <= hc["diff"] <= hc["ci_high"]
    wc = s["trainable_weight_change"]
    assert wc["share_changed"] > 0.9 and wc["relative_l2"] > 0  # fp32 masters: (almost) every weight moves
    assert s["groups_used_total"] > 0 and set(s["mean_times_s_after_step1"]) >= {"rollout", "train", "sync"}


def test_math_task_renders_the_instruction_and_scores_with_math_verify(tmp_path):
    pytest.importorskip("math_verify", reason="math_verify not installed: math_task scoring is UNTESTED here")
    if not (HERE.parent / "rl-data" / "math_check.py").is_file():
        pytest.skip("rl-data/math_check.py not in this checkout: math_task scoring is UNTESTED here")
    rows = [{"id": "m1", "problem": "1+1?", "answer": "2"}, {"id": "m2", "problem": "2+2?", "answer": "4"}]
    (tmp_path / "t.jsonl").write_text("\n".join(__import__("json").dumps(r) for r in rows) + "\n")
    (tmp_path / "tr.txt").write_text("m1\n")
    (tmp_path / "ho.txt").write_text("m2\n")
    (tmp_path / "instr.txt").write_text("Q: {problem}")
    seen = []

    class Tok:
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
            seen.append(msgs[0]["content"])
            return [len(seen), 9]

        def decode(self, ids, skip_special_tokens=True):
            return r"so \boxed{2}" if ids == [1] else r"\boxed{5}"

    task = train.math_task({"tasks": str(tmp_path / "t.jsonl"), "train_ids": str(tmp_path / "tr.txt"),
                            "heldout_ids": str(tmp_path / "ho.txt"),
                            "instruction_file": str(tmp_path / "instr.txt")}, Tok())
    assert seen == ["Q: 1+1?", "Q: 2+2?"]
    assert task.reward(task.train[0], [1]) == 1.0 and task.reward(task.heldout[0], [3]) == 0.0
    (tmp_path / "bad.txt").write_text("no placeholder")
    with pytest.raises(ValueError):
        train.math_task({"tasks": str(tmp_path / "t.jsonl"), "train_ids": str(tmp_path / "tr.txt"),
                         "heldout_ids": str(tmp_path / "ho.txt"), "instruction_file": str(tmp_path / "bad.txt")},
                        Tok())


def test_micro_batched_backward_gives_the_same_gradient_for_every_algo():
    torch.manual_seed(1)
    model = tiny_olmoe()
    pol = hf_policy(model)
    seq = torch.randint(0, 16, (8, 9))
    plen = 3
    cmask = torch.ones(8, 6, dtype=torch.bool)
    cmask[2, 4:] = False
    cmask[5, 2:] = False
    adv = torch.randn(8)
    with torch.no_grad():
        old = train.completion_logprobs(pol, seq, plen) + 0.05 * torch.randn(8, 6)
    params = [p for p in model.parameters() if p.requires_grad]
    for algo in grpo.ALGOS:
        grads = []
        for micro in (None, 4, 3):
            model.zero_grad()
            isw = grpo.tis_weights(old, old - 0.1, cmask, algo, 2.0)
            loss, st = train.backward_group(pol, (seq, cmask, adv, old, None, plen, isw), algo, 0.2, 0.0, 2, micro)
            grads.append([p.grad.clone() for p in params])
        for g in grads[1:]:
            for a, b in zip(grads[0], g):
                assert torch.allclose(a, b, atol=1e-6, rtol=1e-4), algo
    assert torch.allclose(train.chunked_logprobs(pol, seq, plen, 3), train.completion_logprobs(pol, seq, plen),
                          atol=1e-6)


def test_router_trainable_control_changes_the_checksum_and_does_not_raise(tmp_path):
    cfg = yaml.safe_load((HERE / "config" / "smoke_tiny_olmoe.yaml").read_text())
    cfg.update(out_dir=str(tmp_path / "out"), verbose=False)
    cfg["train"].update(steps=5, eval_every=5, router_trainable=True)
    s = train.train(cfg)
    assert s["router_trainable"] and not s["router_unchanged"] and s["router_params_frozen"] == []


def test_extra_budget_eval_length_stats_and_saved_weights(tmp_path):
    cfg = yaml.safe_load((HERE / "config" / "smoke_tiny_olmoe.yaml").read_text())
    cfg.update(out_dir=str(tmp_path / "out"), verbose=False)
    cfg["train"].update(steps=4, eval_every=4, extra_eval_max_new=3, save_trainable=str(tmp_path / "w.pt"))
    s = train.train(cfg)
    e = s["extra_eval"]
    assert e["max_new"] == 3 and 0 <= e["start"]["pass"] <= 1 and e["change"]["problems"] == 64
    assert e["start"]["truncated"] == 1.0  # no EOS in the tiny model: every completion runs to the cap
    assert s["routing_curve"][0]["truncated"] == 1.0 and s["routing_curve"][0]["mean_tokens"] == 2.0
    w = torch.load(tmp_path / "w.pt")
    assert s["saved_trainable_weights"]["bytes"] > 0 and all(v.dtype == torch.bfloat16 for v in w.values())
    assert len(w) == sum(1 for p in tiny_olmoe().parameters())  - 2  # every trainable tensor (router excluded)


def test_length_stats_counts_cut_answers_only():
    from rollout import Rollout
    seq = torch.tensor([[1, 5, 5, 5], [1, 5, 9, 0], [1, 5, 5, 9]])  # prompt [1], 3 new tokens, eos = 9
    cmask = torch.tensor([[True, True, True], [True, True, False], [True, True, True]])
    st = train.length_stats([Rollout(seq, cmask)], 3, 9)
    assert st["truncated"] == round(1 / 3, 4)  # row 0 hit the cap without EOS; row 2 ended with EOS at the cap
    assert st["mean_tokens"] == round(8 / 3, 1)
