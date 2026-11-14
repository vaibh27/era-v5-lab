"""Tests for the RL loop: advantages and losses on hand-worked numbers, the frozen router, the model adapters, and
end-to-end smoke runs where the held-out reward must rise with the router checksum unchanged. CPU only, no network.

Run:  python -m pytest teams/10-post-training/rl-loop/test_rl_loop.py -q -p no:cacheprovider
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grpo  # noqa: E402
import router  # noqa: E402
import tasks  # noqa: E402
import train  # noqa: E402
from policy import completion_logprobs  # noqa: E402


# --- advantages (worked by hand) ---------------------------------------------------------------

def test_advantages_on_hand_worked_numbers():
    r = torch.tensor([1.0, 0.0, 0.0, 1.0])  # mean 0.5, population std 0.5
    g = grpo.advantages(r, "grpo")
    assert torch.allclose(g, torch.tensor([1.0, -1.0, -1.0, 1.0]) * 0.5 / (0.5 + 1e-4))
    assert torch.allclose(grpo.advantages(r, "dr_grpo"), torch.tensor([0.5, -0.5, -0.5, 0.5]))
    # rloo: r_i minus the mean of the other three. For r_1: 1 - (0+0+1)/3 = 2/3
    assert torch.allclose(grpo.advantages(r, "rloo"), torch.tensor([2 / 3, -2 / 3, -2 / 3, 2 / 3]))
    assert torch.allclose(grpo.advantages(r, "gspo"), g)


def test_a_group_with_equal_rewards_has_no_signal_and_is_dropped():
    for algo in grpo.ALGOS:
        assert grpo.advantages(torch.tensor([1.0, 1.0, 1.0]), algo) is None
        assert grpo.advantages(torch.tensor([0.0, 0.0]), algo) is None


def test_unknown_algo_is_an_error():
    with pytest.raises(ValueError):
        grpo.advantages(torch.tensor([1.0, 0.0]), "ppo2")


# --- losses (worked by hand) -------------------------------------------------------------------

def test_token_ratio_is_clipped_for_a_positive_advantage():
    # one sequence, one token, new/old probability ratio 2, advantage 1, clip 0.2: min(2*1, 1.2*1) = 1.2
    new, old = torch.tensor([[math.log(0.4)]]), torch.tensor([[math.log(0.2)]])
    loss, st = grpo.policy_loss(new, old, torch.tensor([1.0]), torch.ones(1, 1, dtype=torch.bool), "grpo", 0.2)
    assert loss.item() == pytest.approx(-1.2) and st["clip_frac"] == 1.0


def test_gspo_uses_one_sequence_ratio():
    # token log-ratios log 2 and 0 -> sequence ratio exp(log2 / 2) = 1.414, clipped to 1.2 for A = 1
    new = torch.tensor([[math.log(2.0), 0.0]])
    loss, _ = grpo.policy_loss(new, torch.zeros(1, 2), torch.tensor([1.0]), torch.ones(1, 2, dtype=torch.bool), "gspo", 0.2)
    assert loss.item() == pytest.approx(-1.2)
    loss, _ = grpo.policy_loss(new, torch.zeros(1, 2), torch.tensor([1.0]), torch.ones(1, 2, dtype=torch.bool), "gspo", 0.5)
    assert loss.item() == pytest.approx(-math.sqrt(2.0), rel=1e-5)


def test_masked_tokens_do_not_count():
    new = torch.tensor([[0.0, 5.0]])  # the second token is after EOS: its huge log-ratio must not matter
    m = torch.tensor([[True, False]])
    for algo in ("grpo", "gspo"):
        loss, _ = grpo.policy_loss(new, torch.zeros(1, 2), torch.tensor([1.0]), m, algo, 0.2)
        assert loss.item() == pytest.approx(-1.0)


def test_dr_grpo_has_no_length_normalisation():
    # same per-token surrogate, but a longer completion contributes more (sum / constant), unlike grpo
    new, old = torch.zeros(2, 4), torch.zeros(2, 4)
    adv = torch.tensor([1.0, 1.0])
    m = torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.bool)
    loss_dr, _ = grpo.policy_loss(new, old, adv, m, "dr_grpo")
    loss_g, _ = grpo.policy_loss(new, old, adv, m, "grpo")
    assert loss_dr.item() == pytest.approx(-(4 + 1) / 8) and loss_g.item() == pytest.approx(-1.0)


def test_kl_is_zero_when_the_policy_equals_the_reference():
    lp = torch.tensor([[-1.0, -2.0]])
    _, st = grpo.policy_loss(lp, lp, torch.tensor([1.0]), torch.ones(1, 2, dtype=torch.bool), "grpo", 0.2, lp, 0.1)
    assert st["kl"] == pytest.approx(0.0)


# --- frozen router ----------------------------------------------------------------------------

def tiny_olmoe():
    from transformers import OlmoeConfig, OlmoeForCausalLM
    torch.manual_seed(0)
    return OlmoeForCausalLM(OlmoeConfig(vocab_size=16, hidden_size=32, intermediate_size=32, num_hidden_layers=2,
                                        num_attention_heads=4, num_key_value_heads=4, num_experts=4,
                                        num_experts_per_tok=2, max_position_embeddings=64, eos_token_id=None,
                                        bos_token_id=None, pad_token_id=None))


def _train_steps(model, steps=3):
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.5)
    ids = torch.randint(0, 16, (4, 6))
    for _ in range(steps):
        opt.zero_grad()
        model(input_ids=ids).logits.pow(2).mean().backward()
        opt.step()


def test_frozen_router_keeps_its_checksum_and_the_rest_trains():
    m = tiny_olmoe()
    names = router.freeze_router(m)
    assert names == ["model.layers.0.mlp.gate.weight", "model.layers.1.mlp.gate.weight"]
    assert not any("experts" in n for n in names)  # experts.gate_up_proj must not match
    before = router.router_checksum(m)
    other_before = m.model.layers[0].self_attn.q_proj.weight.detach().clone()
    _train_steps(m)
    assert router.router_checksum(m) == before
    assert not torch.equal(m.model.layers[0].self_attn.q_proj.weight, other_before)


def test_a_trainable_router_is_caught():
    m = tiny_olmoe()
    before = router.router_checksum(m)
    with pytest.raises(AssertionError):
        router.assert_router_frozen(m)
    _train_steps(m)
    assert router.router_checksum(m) != before  # the check can see a change


def test_a_model_without_a_router_is_refused():
    with pytest.raises(ValueError):
        router.freeze_router(nn.Sequential(nn.Linear(4, 4)))


def test_attention_only_training_keeps_router_and_experts_frozen():
    m = tiny_olmoe()
    n = train.set_trainable(m, [r"\.self_attn\."])
    router.freeze_router(m)
    trainable = [k for k, p in m.named_parameters() if p.requires_grad]
    assert n > 0 and trainable and all(".self_attn." in k for k in trainable)


def test_checksum_works_for_bf16():
    m = tiny_olmoe().to(torch.bfloat16)
    assert len(router.router_checksum(m)) == 64


# --- adapters ---------------------------------------------------------------------------------

def test_completion_logprobs_match_a_direct_computation():
    policy, _ = train.build_model({"kind": "tiny_olmoe", "config": {"vocab_size": 16, "hidden_size": 32,
                                   "intermediate_size": 32, "num_hidden_layers": 1, "num_attention_heads": 4,
                                   "num_key_value_heads": 4, "num_experts": 4, "num_experts_per_tok": 2,
                                   "max_position_embeddings": 64}}, "cpu")
    seq, cmask = policy.sample([1, 2, 3], k=3, max_new=4, temperature=1.0, generator=torch.Generator().manual_seed(0))
    assert seq.shape == (3, 7) and cmask.shape == (3, 4) and cmask.all()
    lp = completion_logprobs(policy, seq, 3)
    logits = policy.model(input_ids=seq).logits
    want = torch.log_softmax(logits[:, 2:-1].float(), -1).gather(-1, seq[:, 3:, None]).squeeze(-1)
    assert torch.allclose(lp, want, atol=1e-5)


def test_toy_rules():
    t = tasks.ToyTask(vocab=16, offset=3, n_train=4, n_heldout=2)
    p = [1, 2, 5]
    assert t.reward(p, [8]) == 1.0 and t.reward(p, [9]) == 0.0 and t.reward(p, []) == 0.0
    t = tasks.ToyTask(vocab=50304, rule="parity", n_train=4, n_heldout=2)
    assert t.reward(p, [7]) == 1.0 and t.reward(p, [8]) == 0.0


def test_reward_env_task_maps_each_prompt_to_its_env():
    class Tok:
        def apply_chat_template(self, msgs, add_generation_prompt, tokenize):
            return [ord(c) % 50 for c in msgs[0]["content"]]

        def decode(self, ids, skip_special_tokens):
            return "".join(chr(i + 60) for i in ids)

    seen = []
    t = tasks.RewardEnvTask(Tok(), [("code/a", "first"), ("code/b", "second")], [("code/c", "third")],
                            reward_fn=lambda env_id, text: (seen.append((env_id, text)) or (1.0, {})))
    assert t.reward(t.train[1], [1, 2]) == 1.0 and seen == [("code/b", "=>")]


# --- end to end ------------------------------------------------------------------------------

def _smoke(config: str, tmp_path, **train_overrides):
    cfg = yaml.safe_load((HERE / "config" / config).read_text())
    cfg.update(out_dir=str(tmp_path / "out"), verbose=False)
    cfg["train"].update(train_overrides)
    return train.train(cfg)


def test_smoke_tiny_olmoe_heldout_reward_rises_with_the_router_frozen(tmp_path):
    for algo in grpo.ALGOS:  # a loop, not parametrize: this repo's pr-guard flags "+@pytest..." lines as emails
        s = _smoke("smoke_tiny_olmoe.yaml", tmp_path / algo, algo=algo)
        assert s["router_unchanged"], algo
        # starts near 1/16. With the batched sampler (5 Nov) the gains over seeds 1234 and 7 were +0.39 to +0.53,
        # except dr_grpo on seed 7 at +0.107, so +0.1 is the bar but that case is close to it
        assert s["heldout_reward_end"] >= s["heldout_reward_start"] + 0.1, (algo, s["heldout_curve"])
        assert (tmp_path / algo / "out" / "metrics.jsonl").exists() and (tmp_path / algo / "out" / "summary.json").exists()


def test_smoke_team5_model_heldout_reward_rises_with_the_router_frozen(tmp_path):
    if not (train.TEAM5_HARNESS / "week2" / "model.py").exists():
        pytest.skip("Team 5 harness not in this checkout")
    s = _smoke("smoke_team5_tiny.yaml", tmp_path)
    assert s["router_params_frozen"] == ["blocks.0.moe.router.weight", "blocks.1.moe.router.weight"]
    assert s["router_unchanged"] and s["heldout_reward_end"] >= s["heldout_reward_start"] + 0.2


def test_hf_checkpoint_path_loads_and_trains(tmp_path):
    tiny_olmoe().save_pretrained(tmp_path / "ckpt")
    cfg = yaml.safe_load((HERE / "config" / "smoke_tiny_olmoe.yaml").read_text())
    cfg.update(out_dir=str(tmp_path / "out"), verbose=False, model={"kind": "hf", "path": str(tmp_path / "ckpt")})
    s = train.train(cfg)
    assert s["router_unchanged"] and s["heldout_reward_end"] > s["heldout_reward_start"]


# --- T3 numerical cases on the actual trainer path (G10_TRAINING.md section 5) -------------------------------
#
# Cases 1-5 run train.train itself. Three test seams, none in the objective:
#   ScriptedTask    rewards for each training prompt come from a fixed script, one entry per sampled row, so a
#                   group's rewards are known exactly (all-pass, all-fail, [0,1,0,1], ...).
#   _PromptSeeded   wraps rollout.make_sampler's sampler: same Policy.sample, but each prompt gets its own
#                   generator seeded from its token ids, so a prompt's rollout does not depend on which other
#                   prompts share the batch (needed to compare a batch with and without a dropped group).
#   _RecAdamW       torch.optim.AdamW subclass that counts step() calls and keeps the starting weights.
# train.py has no learning-rate scheduler, so "no scheduler update" is checked as an unchanged lr.

from policy import hf_policy  # noqa: E402

_TINY = {"vocab_size": 16, "hidden_size": 32, "intermediate_size": 32, "num_hidden_layers": 2,
         "num_attention_heads": 4, "num_key_value_heads": 4, "num_experts": 4, "num_experts_per_tok": 2,
         "max_position_embeddings": 64}
ALL_PASS, ALL_FAIL, MIXED, MIXED_B = (1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 0)
HELDOUT = [[2, 4, 6, 8]]


class ScriptedTask:
    def __init__(self, scripts: dict):
        self.scripts, self.pos = scripts, {}
        self.train, self.heldout = [list(p) for p in scripts], [list(p) for p in HELDOUT]

    def reward(self, prompt, completion):
        key = tuple(prompt)
        if key not in self.scripts:
            return 0.0  # held-out prompts
        i = self.pos.get(key, 0)
        self.pos[key] = i + 1
        s = self.scripts[key]
        return float(s[i % len(s)])


class _PromptSeeded:
    def __init__(self, inner):
        self.inner = inner

    def sample(self, prompts, k, max_new, temperature, generator=None):
        out = []
        for p in prompts:
            g = torch.Generator().manual_seed(1000 + sum(int(t) * 31 ** i for i, t in enumerate(p)))
            out += self.inner.sample([p], k, max_new, temperature, g)
        return out


class _RecAdamW(torch.optim.AdamW):
    made: list = []

    def __init__(self, params, **kw):
        params = list(params)
        super().__init__(params, **kw)
        self.n_steps = 0
        self.start = [p.detach().clone() for g in self.param_groups for p in g["params"]]
        self.lr0 = [g["lr"] for g in self.param_groups]
        _RecAdamW.made.append(self)

    def step(self, *a, **kw):
        self.n_steps += 1
        return super().step(*a, **kw)

    def params(self):
        return [p for g in self.param_groups for p in g["params"]]


def _run(monkeypatch, tmp_path, name, scripts, k=4, kl=0.1, steps=1, ppo_epochs=2, algo="grpo"):
    """One real train.train run on the scripted task. Returns (summary, optimizer, records)."""
    rec = {"logprob_prompts": [], "logprob_policies": set(), "groups": []}
    real_lp, real_bg, real_make = train.chunked_logprobs, train.backward_group, train.rollout.make_sampler

    def lp(pol, seq, plen, micro):
        rec["logprob_prompts"].append(tuple(seq[0, :plen].tolist()))
        rec["logprob_policies"].add(id(pol))
        return real_lp(pol, seq, plen, micro)

    def bg(pol, group, *a, **kw):
        rec["groups"].append((tuple(group[0][0, :group[5]].tolist()), group[2].detach().clone()))
        return real_bg(pol, group, *a, **kw)

    monkeypatch.setattr(train, "chunked_logprobs", lp)
    monkeypatch.setattr(train, "backward_group", bg)
    monkeypatch.setattr(train.rollout, "make_sampler", lambda *a, **kw: _PromptSeeded(real_make(*a, **kw)))
    monkeypatch.setattr(train, "build_task", lambda tcfg, tok, seed: ScriptedTask(scripts))
    monkeypatch.setattr(torch.optim, "AdamW", _RecAdamW)
    _RecAdamW.made = []
    cfg = {"seed": 1234, "device": "cpu", "out_dir": str(tmp_path / name), "verbose": False,
           "model": {"kind": "tiny_olmoe", "config": dict(_TINY)}, "task": {"kind": "scripted"},
           "train": {"algo": algo, "steps": steps, "prompts_per_step": len(scripts), "k": k, "max_new": 3,
                     "temperature": 1.0, "lr": 0.01, "clip_eps": 0.2, "ppo_epochs": ppo_epochs, "kl_coef": kl,
                     "eval_every": 100, "eval_k": 2, "sampler": "policy", "fp32_master": False}}
    summary = train.train(cfg)
    recs = [json.loads(x) for x in (tmp_path / name / "metrics.jsonl").read_text().splitlines()]
    return summary, _RecAdamW.made[0], recs, rec


import json  # noqa: E402


# case 1 -------------------------------------------------------------------------------------------------
def test_t3_constant_groups_are_dropped_before_reference_kl(monkeypatch, tmp_path):
    for algo in grpo.ALGOS:  # the contract
        assert grpo.advantages(torch.ones(4), algo) is None and grpo.advantages(torch.zeros(4), algo) is None
    s, opt, recs, rec = _run(monkeypatch, tmp_path, "c1",
                             {ALL_PASS: [1, 1, 1, 1], ALL_FAIL: [0, 0, 0, 0], MIXED: [0, 1, 0, 1]}, kl=0.1)
    assert recs[0]["groups_dropped"] == 2 and recs[0]["groups_used"] == 1
    # old and reference log-probs were computed for the mixed group only: two policies (current and frozen
    # reference copy), one call each, never for the all-pass or all-fail prompt
    assert rec["logprob_prompts"] == [MIXED, MIXED] and len(rec["logprob_policies"]) == 2
    assert [p for p, _ in rec["groups"]] == [MIXED, MIXED]  # backward on the mixed group, once per ppo epoch
    assert opt.n_steps == 2 and math.isfinite(recs[0]["kl"])


# case 2 -------------------------------------------------------------------------------------------------
def test_t3_singleton_groups_are_dropped_without_nan(monkeypatch, tmp_path):
    for algo in grpo.ALGOS:
        assert grpo.advantages(torch.tensor([1.0]), algo) is None
        assert grpo.advantages(torch.tensor([0.0]), algo) is None
    s, opt, recs, rec = _run(monkeypatch, tmp_path, "c2", {MIXED: [0, 1, 0, 1], MIXED_B: [1, 0, 1, 0]}, k=1)
    assert recs[0]["groups_dropped"] == 2 and recs[0]["groups_used"] == 0
    assert rec["logprob_prompts"] == [] and rec["groups"] == [] and opt.n_steps == 0
    assert all(math.isfinite(v) for v in (recs[0]["loss"], recs[0]["kl"], recs[0]["train_reward"]))


# case 3 -------------------------------------------------------------------------------------------------
def test_t3_mixed_group_advantages_signs_and_sum(monkeypatch, tmp_path):
    r = torch.tensor([0.0, 1.0, 0.0, 1.0])
    expected = {"grpo": 0.5 / (0.5 + 1e-4), "gspo": 0.5 / (0.5 + 1e-4), "dr_grpo": 0.5, "rloo": 2 / 3}
    for algo, mag in expected.items():
        a = grpo.advantages(r, algo)
        assert torch.isfinite(a).all() and torch.equal(torch.sign(a), torch.tensor([-1.0, 1.0, -1.0, 1.0]))
        assert torch.allclose(a.abs(), torch.full((4,), mag)) and abs(float(a.sum())) < 1e-6, algo
    # and the advantages the trainer actually hands to backward_group
    s, opt, recs, rec = _run(monkeypatch, tmp_path, "c3", {MIXED: [0, 1, 0, 1]}, kl=0.0, ppo_epochs=1)
    (prompt, adv), = rec["groups"]
    assert prompt == MIXED and torch.allclose(adv, grpo.advantages(r, "grpo"))
    assert math.isfinite(recs[0]["loss"]) and recs[0]["grad_norm"] > 0 and opt.n_steps == 1


# case 4 -------------------------------------------------------------------------------------------------
def test_t3_dropped_group_does_not_change_the_update(monkeypatch, tmp_path):
    mixed = {MIXED: [0, 1, 0, 1], MIXED_B: [1, 1, 0, 0]}
    s0, opt0, recs0, rec0 = _run(monkeypatch, tmp_path, "c4a", dict(mixed), kl=0.1)
    s1, opt1, recs1, rec1 = _run(monkeypatch, tmp_path, "c4b", {**mixed, ALL_PASS: [1, 1, 1, 1]}, kl=0.1)
    assert recs1[0]["groups_dropped"] == 1 and recs0[0]["groups_dropped"] == 0
    assert recs0[0]["groups_used"] == recs1[0]["groups_used"] == 2
    assert ALL_PASS not in rec1["logprob_prompts"]
    # identical rollouts and advantages for the mixed groups in both runs
    adv0 = {p: a for p, a in rec0["groups"]}
    assert all(torch.equal(adv0[p], a) for p, a in rec1["groups"])
    # same loss and gradient norm; same weights and AdamW moments after the update (group order may differ
    # between the runs because the batch is shuffled, so allow float summation-order differences)
    assert recs1[0]["loss"] == pytest.approx(recs0[0]["loss"], abs=1e-7)
    assert recs1[0]["grad_norm"] == pytest.approx(recs0[0]["grad_norm"], rel=1e-5)
    assert opt0.n_steps == opt1.n_steps == 2
    for p0, p1 in zip(opt0.params(), opt1.params()):
        assert torch.allclose(p0, p1, atol=1e-7, rtol=0)
        for key in ("exp_avg", "exp_avg_sq"):
            assert torch.allclose(opt0.state[p0][key], opt1.state[p1][key], atol=1e-9, rtol=1e-5)


# case 5 -------------------------------------------------------------------------------------------------
def test_t3_all_groups_dropped_means_no_optimizer_step(monkeypatch, tmp_path):
    s, opt, recs, rec = _run(monkeypatch, tmp_path, "c5", {ALL_PASS: [1, 1, 1, 1], ALL_FAIL: [0, 0, 0, 0]},
                             kl=0.1, steps=2)
    assert [r["groups_used"] for r in recs] == [0, 0] and [r["groups_dropped"] for r in recs] == [2, 2]
    assert opt.n_steps == 0 and len(opt.state) == 0  # AdamW creates moments on its first step: none exist
    assert [g["lr"] for g in opt.param_groups] == opt.lr0  # no scheduler in train.py; lr untouched
    assert all(torch.equal(p, p0) for p, p0 in zip(opt.params(), opt.start))
    assert all(p.grad is None for p in opt.params())
    assert rec["logprob_prompts"] == [] and s["trainable_weight_change"]["share_changed"] == 0.0
    assert s["router_unchanged"] and s["groups_used_total"] == 0


# cases 6 and 7: train.backward_group, the trainer's own chunked loss/backward -----------------------------

def _uneven_group(lengths=(1, 6, 3, 5, 2), T=6, seed=3, kl_noise=0.3):
    torch.manual_seed(seed)
    model = tiny_olmoe()
    pol = hf_policy(model)
    k, plen = len(lengths), 3
    seq = torch.randint(0, 16, (k, plen + T))
    cmask = torch.zeros(k, T, dtype=torch.bool)
    for i, n in enumerate(lengths):
        cmask[i, :n] = True
    adv = grpo.advantages(torch.tensor([0.0, 1.0, 1.0, 0.0, 1.0][:k]), "grpo")
    with torch.no_grad():
        cur = completion_logprobs(pol, seq, plen)
    old = cur + 0.3 * torch.randn(k, T)  # far enough from current that some ratios clip
    ref = cur - kl_noise * torch.randn(k, T).abs()
    return model, pol, seq, cmask, adv, old, ref, plen


def _backward(model, pol, group, algo, kl, micro):
    model.zero_grad(set_to_none=True)
    total, st = train.backward_group(pol, group, algo, 0.2, kl, 1, micro)
    return total, st, {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def _max_grad_diff(a, b):
    assert a.keys() == b.keys()
    return max(float((a[n] - b[n]).abs().max()) for n in a), max(float(a[n].abs().max()) for n in a)


def test_t3_uneven_lengths_and_micro_rows_match_unchunked_without_kl():
    model, pol, seq, cmask, adv, old, ref, plen = _uneven_group()
    for algo in grpo.ALGOS:
        group = (seq, cmask, adv, old, None, plen, None)
        t0, _, g0 = _backward(model, pol, group, algo, 0.0, None)
        for micro in (2, 3, 4):  # k = 5: every chunking leaves an uneven last chunk
            t, _, g = _backward(model, pol, group, algo, 0.0, micro)
            diff, scale = _max_grad_diff(g0, g)
            assert t == pytest.approx(t0, abs=1e-6), (algo, micro)
            assert diff <= 1e-6 + 1e-4 * scale, (algo, micro, diff)


def _global_k3(cur_lp, ref_lp, cmask):
    m = cmask.float()
    d = (ref_lp - cur_lp) * m
    return float(((torch.exp(d) - d - 1) * m).sum() / m.sum())


def test_t3_kl_with_equal_lengths_chunked_matches_global_normalisation():
    """Control: with every row the same length, row-share weighting of per-chunk token means is exact."""
    model, pol, seq, cmask, adv, old, ref, plen = _uneven_group(lengths=(6, 6, 6, 6, 6))
    with torch.no_grad():
        cur = completion_logprobs(pol, seq, plen)
    group = (seq, cmask, adv, old, ref, plen, None)
    t0, st0, g0 = _backward(model, pol, group, "grpo", 0.1, None)
    assert st0["kl"] == pytest.approx(_global_k3(cur, ref, cmask), rel=1e-6)  # unchunked = group-global mean
    for micro in (1, 2, 3, 4):
        t, st, g = _backward(model, pol, group, "grpo", 0.1, micro)
        diff, scale = _max_grad_diff(g0, g)
        assert t == pytest.approx(t0, abs=1e-6) and st["kl"] == pytest.approx(st0["kl"], rel=1e-6)
        assert diff <= 1e-6 + 1e-4 * scale


def kl_chunking_discrepancy(lengths=(1, 6, 3, 5, 2), algo="grpo", kl=0.1):
    """Numbers for the report: chunked vs unchunked backward_group with nonzero reference KL."""
    model, pol, seq, cmask, adv, old, ref, plen = _uneven_group(lengths=lengths)
    with torch.no_grad():
        cur = completion_logprobs(pol, seq, plen)
    group = (seq, cmask, adv, old, ref, plen, None)
    t0, st0, g0 = _backward(model, pol, group, algo, kl, None)
    p0, _, gp0 = _backward(model, pol, (seq, cmask, adv, old, None, plen, None), algo, 0.0, None)
    rows = {"global_k3": _global_k3(cur, ref, cmask), "unchunked": {"loss": t0, "kl": st0["kl"]}, "chunked": {}}
    for micro in (1, 2, 3, 4):
        t, st, g = _backward(model, pol, group, algo, kl, micro)
        p, _, gp = _backward(model, pol, (seq, cmask, adv, old, None, plen, None), algo, 0.0, micro)
        diff, scale = _max_grad_diff(g0, g)
        rows["chunked"][micro] = {"loss": t, "kl_reported": st["kl"], "loss_diff": t - t0,
                                  "loss_diff_without_kl": p - p0, "kl_term_diff": t - t0 - (p - p0),
                                  "implied_kl": st0["kl"] + (t - t0 - (p - p0)) / kl,
                                  "max_abs_grad_diff": diff, "max_abs_grad": scale}
    return rows


def test_t3_kl_uneven_lengths_chunked_matches_global_normalisation():
    """Exposed the per-chunk KL normalisation (fixed 11 Nov); kept with the two uneven patterns that showed it."""
    for lengths in ((1, 6, 3, 5, 2), (1, 1, 1, 1, 6)):
        rows = kl_chunking_discrepancy(lengths=lengths)
        assert rows["unchunked"]["kl"] == pytest.approx(rows["global_k3"], rel=1e-6)  # the reference itself is right
        for micro, r in rows["chunked"].items():
            assert abs(r["loss_diff_without_kl"]) < 1e-6  # the policy part is exact (case 6)
            assert r["kl_reported"] == pytest.approx(rows["global_k3"], rel=1e-6), (lengths, micro, r)
            assert abs(r["loss_diff"]) < 1e-6 and r["max_abs_grad_diff"] <= 1e-6 + 1e-4 * r["max_abs_grad"], (
                lengths, micro, r)
