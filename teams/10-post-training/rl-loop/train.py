"""RL loop v0: group-relative policy optimisation with a frozen MoE router.

    python teams/10-post-training/rl-loop/train.py --config teams/10-post-training/rl-loop/config/smoke_tiny_olmoe.yaml

One step: take `prompts_per_step` training prompts, sample `k` completions for each, score them, drop groups whose
rewards are all equal, and take `ppo_epochs` optimiser steps on the clipped surrogate of the chosen algorithm (grpo,
dr_grpo, rloo or gspo; see grpo.py). Every `eval_every` steps the held-out prompts are scored (that is the number
Gate 4 should be judged on, not the training reward). The router is frozen before training and its checksum must
be identical at the end; the run fails otherwise.

Writes to `out_dir` (default ./out, the TSAI job rule): metrics.jsonl (one line per step), summary.json.

Model kinds (config `model.kind`):
    tiny_olmoe   a random tiny OLMoE from a config (no download); for smoke tests on a laptop
    hf           AutoModelForCausalLM.from_pretrained(model.path), e.g. a local copy of allenai/OLMoE-1B-7B-0125-Instruct
    team5        Team 5's harness LanguageModel (teams/05-architecture/harness/week2/model.py), built with their own
                 config loader from model.arm + model.resolution, plus model.overrides; the KE2 code table is a
                 seeded stand-in until Team 5 ships the real model on 18 Nov (smoke only)
    team5_adapter  Team 5's real checkpoint through team5_adapter.py (model.adapter_config): pinned G5 code,
                 strict checkpoint/KE2/tokenizer checks, G10 chat codec, router and trainable sets from the adapter.
                 Exits 3 with out_dir/status.json = WAITING_FOR_G5_ARTIFACT until the checkpoint exists. A fixture
                 adapter config needs model.allow_fixture_smoke: true and writes smoke_summary.json, never summary.json.
Task kinds (config `task.kind`): toy (see tasks.py), code_tasks (rl/tasks via rl/reward.py; needs an hf tokenizer).

GPU-run options (config `train.*`, all optional):
    sampler        policy | hf_batch | vllm (rollout.py). Default hf_batch for HF models, policy otherwise.
    vllm           kwargs for the colocated engine, e.g. {gpu_memory_utilization: 0.3, max_model_len: 2048}
    fp32_master    default true: trainable weights in fp32 under bf16 autocast (policy.use_fp32_master)
    tis_clip       truncated importance sampling for the vLLM/trainer gap (grpo.tis_weights), e.g. 2.0
    routing_log    expert load and top-k flip rates (routing.py); routing_probe = held-out prompts in the probe
    kl_to_start    K3 KL(current || step 0) on held-out completions at every eval
metrics.jsonl adds per-step timings (rollout, reward, old_logprobs, train, sync), sampled tokens and tokens/s,
GPU memory, and the sampler/trainer mismatch. summary.json adds the paired-bootstrap CI of the held-out change,
the trainable-weight change, and the library versions.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
import yaml
from torch import no_grad  # decorator without a dot: pr-guard reads '+@a.b' as an email

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import grpo  # noqa: E402
import heldout  # noqa: E402
import rollout  # noqa: E402
import router  # noqa: E402
import routing  # noqa: E402
import tasks  # noqa: E402
from policy import HFPolicy, completion_logprobs, hf_policy, module_policy, use_fp32_master  # noqa: E402

TEAM5_HARNESS = HERE.parents[1] / "05-architecture" / "harness"


# --- building the model and the task --------------------------------------------------------------

def build_model(mcfg: dict, device: str):
    kind = mcfg["kind"]
    if kind == "tiny_olmoe":
        from transformers import OlmoeConfig, OlmoeForCausalLM  # noqa: PLC0415
        cfg = OlmoeConfig(eos_token_id=None, bos_token_id=None, pad_token_id=None, **mcfg["config"])
        return hf_policy(OlmoeForCausalLM(cfg), None, device), None
    if kind == "hf":
        from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
        dtype = getattr(torch, mcfg.get("dtype", "float32"))
        model = AutoModelForCausalLM.from_pretrained(mcfg["path"], dtype=dtype, local_files_only=True)
        try:
            tok = AutoTokenizer.from_pretrained(mcfg["path"], local_files_only=True)
        except (OSError, ValueError):
            tok = None  # the toy task needs no tokenizer
        if mcfg.get("gradient_checkpointing"):
            model.gradient_checkpointing_enable()
        eos = tok.eos_token_id if tok is not None and mcfg.get("stop_at_eos", True) else None
        return hf_policy(model, eos, device), tok
    if kind == "team5":
        import numpy as np  # noqa: PLC0415
        sys.path.insert(0, str(TEAM5_HARNESS))
        from week2.config import load, load_runtime, merge  # noqa: PLC0415
        from week2.model import LanguageModel  # noqa: PLC0415
        team5 = TEAM5_HARNESS.parent
        spec = merge(load_runtime(team5 / mcfg["arm"]), load(team5 / mcfg["resolution"]))["model"]
        moe = mcfg.get("overrides", {}).pop("moe", {}) if mcfg.get("overrides") else {}
        spec.update(mcfg.get("overrides", {}))
        spec["moe"].update(moe)
        rng = np.random.default_rng(mcfg.get("ke2_seed", 7))  # stand-in KE2 code table until Team 5's model lands
        rows = rng.integers(0, 256, (spec["vocab_size"], 4), dtype=np.uint8)
        lengths = rng.integers(1, 5, spec["vocab_size"], dtype=np.int16)
        return module_policy(LanguageModel(spec, rows, lengths), None, device), None
    if kind == "team5_adapter":
        import team5_adapter  # noqa: PLC0415
        return team5_adapter.build_policy_and_codec(mcfg["adapter_config"], device, mcfg.get("effort", "4k"),
                                                    bool(mcfg.get("allow_fixture_smoke", False)))
    raise ValueError(f"unknown model.kind {kind!r}")


def build_task(tcfg: dict, tokenizer, seed: int):
    if tcfg["kind"] == "toy":
        return tasks.ToyTask(**{k: v for k, v in tcfg.items() if k != "kind"}, seed=seed)
    if tcfg["kind"] == "code_tasks":
        if tokenizer is None:
            raise ValueError("code_tasks needs model.kind hf (a tokenizer)")
        return tasks.RewardEnvTask(tokenizer, tasks.code_task_prompts(tcfg["train"]),
                                   tasks.code_task_prompts(tcfg["heldout"]))
    if tcfg["kind"] == "math":
        return math_task(tcfg, tokenizer)
    raise ValueError(f"unknown task.kind {tcfg['kind']!r}")


def math_task(tcfg: dict, tokenizer):
    """MATH-style tasks (rl-data/build_math.py records) scored by rl-data/math_check.py.

    task: {kind: math, tasks: <jsonl>, train_ids: <file>, heldout_ids: <file>, instruction_file: <file or null>}
    The instruction file is written by a team member (TSAI 3 Nov rule) and holds the user message with {problem};
    without it the raw problem text is the prompt.
    """
    import importlib.util  # noqa: PLC0415
    if tokenizer is None:
        raise ValueError("math tasks need model.kind hf (a tokenizer)")
    spec = importlib.util.spec_from_file_location("g10_math_check", HERE.parent / "rl-data" / "math_check.py")
    mc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mc)
    by_id = {}
    for line in Path(tcfg["tasks"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            t = json.loads(line)
            by_id[t["id"]] = t
    instr = Path(tcfg["instruction_file"]).read_text(encoding="utf-8") if tcfg.get("instruction_file") else None
    if instr is not None and "{problem}" not in instr:
        raise ValueError("instruction_file needs a {problem} placeholder")

    def items(ids_file):
        ids = [x.strip() for x in Path(ids_file).read_text(encoding="utf-8").splitlines() if x.strip()]
        missing = [i for i in ids if i not in by_id]
        if missing:
            raise ValueError(f"{len(missing)} ids not in {tcfg['tasks']}, e.g. {missing[:3]}")
        return [(i, instr.replace("{problem}", by_id[i]["problem"]) if instr else by_id[i]["problem"]) for i in ids]

    def reward_fn(env_id: str, text: str):
        r = mc.score(by_id[env_id], text)
        return float(r["reward"]), r

    return tasks.RewardEnvTask(tokenizer, items(tcfg["train_ids"]), items(tcfg["heldout_ids"]), reward_fn)


def set_trainable(model, patterns: list[str] | None) -> int:
    """If patterns are given, train only matching parameters (e.g. attention); the router stays frozen regardless."""
    if patterns:
        rx = [re.compile(p) for p in patterns]
        for n, p in model.named_parameters():
            p.requires_grad_(any(r.search(n) for r in rx))
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# --- evaluation and training ----------------------------------------------------------------------

def evaluate(sampler, task, prompts, k: int, max_new: int, temperature: float, gen):
    """Per-problem pass@1 (mean reward over k samples) and the rollouts (reused as the routing probe)."""
    rolls = sampler.sample(prompts, k, max_new, temperature, gen)
    per_problem = []
    for p, r in zip(prompts, rolls):
        rs = [task.reward(p, row[m].tolist()) for row, m in zip(r.seq[:, len(p):], r.cmask)]
        per_problem.append(sum(rs) / len(rs))
    return per_problem, rolls


def length_stats(rolls, max_new: int, eos_id) -> dict:
    """Share of completions that hit the token cap without ending (a cut answer scores 0), and mean length."""
    n = cut = toks = 0
    for r in rolls:
        lens = r.cmask.sum(1)
        last = r.seq[torch.arange(r.seq.shape[0]), r.seq.shape[1] - r.cmask.shape[1] + lens.clamp_min(1) - 1]
        hit = lens == max_new
        if eos_id is not None:
            hit = hit & (last != eos_id)
        cut += int(hit.sum())
        toks += int(lens.sum())
        n += lens.numel()
    return {"truncated": round(cut / max(n, 1), 4), "mean_tokens": round(toks / max(n, 1), 1)}


def gpu_memory() -> dict:
    if not torch.cuda.is_available():
        return {}
    free, total = torch.cuda.mem_get_info()
    return {"peak_torch_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "gpu_used_gb": round((total - free) / 2**30, 2), "gpu_total_gb": round(total / 2**30, 2)}


def versions() -> dict:
    out = {"torch": torch.__version__}
    for mod in ("transformers", "vllm"):
        try:
            out[mod] = __import__(mod).__version__
        except ImportError:
            out[mod] = None
    return out


class StartWeights:
    """The trainable weights at step 0, kept on the CPU, for the KL-to-start metric (cheaper than a second model)."""

    def __init__(self, model):
        self.params = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.saved = [p.detach().to("cpu", copy=True) for _, p in self.params]

    @torch.no_grad()
    def change(self) -> dict:
        """How far the trainable weights moved: share of entries that changed, and ||w - w0|| / ||w0||."""
        changed = total = 0
        diff_sq = base_sq = 0.0
        for (_, p), s in zip(self.params, self.saved):
            w = p.detach().to("cpu", torch.float32)
            s32 = s.to(torch.float32)
            changed += int((w != s32).sum())
            total += w.numel()
            diff_sq += float((w - s32).pow(2).sum())
            base_sq += float(s32.pow(2).sum())
        return {"share_changed": changed / max(total, 1), "relative_l2": (diff_sq ** 0.5) / max(base_sq ** 0.5, 1e-30)}

    @torch.no_grad()
    def swap(self):
        for (_, p), s in zip(self.params, self.saved):
            tmp = p.detach().to("cpu", copy=True)
            p.copy_(s.to(p.device))
            s.copy_(tmp)


@no_grad()
def kl_to_start(policy, start: StartWeights, rolls, prompts) -> float:
    """K3 estimate of KL(current || start) per completion token, on completions sampled from the current policy."""
    cur = [completion_logprobs(policy, r.seq, len(p)) for p, r in zip(prompts, rolls)]
    start.swap()
    try:
        ref = [completion_logprobs(policy, r.seq, len(p)) for p, r in zip(prompts, rolls)]
    finally:
        start.swap()
    num = den = 0.0
    for c, f, r in zip(cur, ref, rolls):
        m = r.cmask.float()
        d = (f - c) * m
        num += float(((torch.exp(d) - d - 1) * m).sum())
        den += float(m.sum())
    return num / max(den, 1.0)


def chunked_logprobs(policy, seq: torch.Tensor, plen: int, micro_rows: int | None) -> torch.Tensor:
    """completion_logprobs over row chunks: the [rows, T, vocab] fp32 logits are the memory peak (C1/C2 ran out of
    memory on an 80 GB H100 at step 84 with 8 rows x ~1,500 tokens x 50,304 vocab)."""
    if not micro_rows or micro_rows >= seq.shape[0]:
        return completion_logprobs(policy, seq, plen)
    return torch.cat([completion_logprobs(policy, seq[i:i + micro_rows], plen)
                      for i in range(0, seq.shape[0], micro_rows)], 0)


def backward_group(policy, group, algo: str, clip_eps: float, kl_coef: float, n_groups: int,
                   micro_rows: int | None) -> tuple[float, dict]:
    """Loss and backward for one group, optionally in row chunks. Every loss in grpo.py is a mean over the group's
    rows (grpo, gspo) or a sum divided by rows x T (dr_grpo, rloo), so a chunk's loss times rows_in_chunk / k is
    its exact share, and the summed gradients equal the one-shot gradient.

    The reference KL (kl_coef > 0) is the group-global token mean: the k3 term of grpo.policy_loss summed over each
    chunk's valid tokens and divided by the valid-token count of the whole group, so every chunking gives the same
    value as the unchunked group. (Before 11 Nov it was normalised per chunk and weighted by row share, which is
    wrong for uneven completion lengths.) With kl_coef == 0 this function computes exactly what it did before."""
    seq, cmask, adv, old_lp, ref_lp, plen, isw = group
    k = seq.shape[0]
    step = micro_rows if micro_rows and micro_rows < k else k
    use_kl = ref_lp is not None and kl_coef > 0
    n_valid = float(cmask.float().sum().clamp_min(1))  # group-global KL denominator
    total, agg = 0.0, {"clip_frac": 0.0, "ratio_mean": 0.0, "kl": 0.0}
    for i in range(0, k, step):
        sl = slice(i, min(i + step, k))
        rows = sl.stop - sl.start
        new_lp = completion_logprobs(policy, seq[sl], plen)
        loss, st = grpo.policy_loss(new_lp, old_lp[sl], adv[sl], cmask[sl], algo, clip_eps, None, 0.0,
                                    None if isw is None else isw[sl])
        share = rows / k
        obj = loss * share
        if use_kl:
            m = cmask[sl].float()
            d = (ref_lp[sl] - new_lp) * m
            kl_part = ((torch.exp(d) - d - 1) * m).sum() / n_valid  # same k3 as grpo.policy_loss
            obj = obj + kl_coef * kl_part
            agg["kl"] += float(kl_part.detach())
        (obj / n_groups).backward()
        total += float(loss.detach()) * share / n_groups
        if use_kl:
            total += kl_coef * float(kl_part.detach()) / n_groups
        for key in ("clip_frac", "ratio_mean"):
            agg[key] += st[key] * share
    return total, agg


@no_grad()
def check_vllm_sync(policy, vllm, prompts, trainable_named, max_new: int = 16) -> dict:
    """Prove that sync() really changes what vLLM samples (lessons file: a silent sync failure looks like
    "RL does not learn"). Zero every q_proj weight in the HF model, and measure the sampler/trainer gap (K3 on
    vLLM's samples) three ways: before the change, after the change without a sync, and after a sync. Then put
    the weights back and sync again. A working sync gives: unsynced gap >> synced gap ~ baseline gap."""
    def gap():
        rolls = vllm.sample(prompts, 2, max_new, 1.0, None)
        ks = [grpo.mismatch(completion_logprobs(policy, r.seq, len(p)), r.rollout_lp, r.cmask)["k3"]
              for p, r in zip(prompts, rolls)]
        return sum(ks) / len(ks)

    targets = [(n, p) for n, p in trainable_named if n.endswith("q_proj.weight")]
    if not targets:
        return {"skipped": "no trainable q_proj weights"}
    base = gap()
    saved = [p.detach().clone() for _, p in targets]
    for _, p in targets:
        p.zero_()
    unsynced = gap()
    vllm.sync(trainable_named)
    synced = gap()
    for (_, p), s_ in zip(targets, saved):
        p.copy_(s_)
    vllm.sync(trainable_named)
    restored = gap()
    ok = unsynced > 10 * max(synced, 1e-4) and synced < 10 * base + 0.05 and restored < 10 * base + 0.05
    return {"baseline_k3": base, "zeroed_unsynced_k3": unsynced, "zeroed_synced_k3": synced,
            "restored_k3": restored, "ok": bool(ok), "sync_path": vllm.sync_path}


def train(cfg: dict) -> dict:
    seed = int(cfg.get("seed", 1234))
    random.seed(seed)
    torch.manual_seed(seed)
    device = cfg.get("device", "cpu")
    out = Path(cfg.get("out_dir", "out"))
    out.mkdir(parents=True, exist_ok=True)
    t = cfg["train"]
    algo, k, max_new = t["algo"], int(t["k"]), int(t["max_new"])
    temperature = float(t["temperature"])
    sampler_kind = t.get("sampler") or ("hf_batch" if cfg["model"]["kind"] in ("hf", "tiny_olmoe") else "policy")
    if cfg["model"]["kind"] == "team5_adapter" and sampler_kind != "policy":
        raise ValueError(f"sampler {sampler_kind} is not available for the G5 LanguageModel; use sampler: policy")
    vllm = None
    if sampler_kind == "vllm":  # vLLM reserves its share of the GPU first, then the training model loads
        vllm = rollout.VLLMSampler(cfg["model"]["path"], seed=seed, **t.get("vllm", {}))
    policy, tok = build_model(cfg["model"], device)
    task = build_task(cfg["task"], tok, seed)

    model = policy.model
    adapter = getattr(tok, "adapter_result", None)  # team5_adapter: one router/trainable set for everything below
    if adapter is not None:
        if t.get("trainable") or cfg.get("router_patterns") or t.get("router_trainable"):
            raise ValueError("model.kind team5_adapter takes trainable and router sets from the adapter config only")
        patterns = tuple(adapter.bundle.selection["router_patterns"])
    else:
        set_trainable(model, t.get("trainable"))
        patterns = tuple(cfg.get("router_patterns", router.ROUTER_PATTERNS))
    router_trainable = bool(t.get("router_trainable", False))  # control runs only; Gate 4 keeps the router frozen
    if router_trainable:
        frozen = []
        for _, p_ in router.router_params(model, patterns):
            p_.requires_grad_(True)
    else:
        frozen = router.freeze_router(model, patterns)
        router.assert_router_frozen(model, patterns)
    if adapter is not None and frozen != adapter.bundle.selection["router_names"]:
        raise RuntimeError(f"frozen routers {frozen} differ from the adapter's {adapter.bundle.selection['router_names']}")
    checksum_before = router.router_checksum(model, patterns)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_fp32 = use_fp32_master(policy) if t.get("fp32_master", True) else 0
    trainable_named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    pad_id = tok.pad_token_id if tok is not None and tok.pad_token_id is not None else None
    sampler = rollout.make_sampler(sampler_kind, policy, vllm, pad_id)
    ref = None
    if float(t.get("kl_coef", 0)) > 0:
        ref = copy.deepcopy(policy)
        ref.model.requires_grad_(False)
        ref.model.eval()
    start_w = StartWeights(model)  # CPU copy of the trainable weights: weight-change check, and KL to start
    opt = torch.optim.AdamW([p for _, p in trainable_named], lr=float(t["lr"]),
                            weight_decay=float(t.get("weight_decay", 0.0)))
    tis_clip = float(t["tis_clip"]) if t.get("tis_clip") else None
    micro_rows = int(t["micro_rows"]) if t.get("micro_rows") else None
    log_routing = bool(t.get("routing_log", False)) and bool(routing.router_modules(model))
    n_experts = getattr(getattr(model, "config", None), "num_experts", None)
    # Native sampling draws on the model's device, so its generator lives there; HF/vLLM keep host-side generators.
    gen_device = device if sampler_kind == "policy" and not isinstance(policy, HFPolicy) else "cpu"
    gen = torch.Generator(device=gen_device).manual_seed(seed)
    order = list(range(len(task.train)))
    fixture_smoke = adapter is not None and adapter.status != "READY"
    metrics_path = out / ("smoke_metrics.jsonl" if fixture_smoke else "metrics.jsonl")
    if fixture_smoke:
        (out / "FIXTURE_SMOKE_NOT_A_RESULT").write_text("Built from a fixture adapter config; not a G5 result.\n")
    metrics_f = open(metrics_path, "w")
    start = time.time()
    heldout_curve, heldout_scores, routing_curve = [], {}, []
    probe: dict = {}
    eval_k = int(t.get("eval_k", 4))
    extra_max_new = int(t["extra_eval_max_new"]) if t.get("extra_eval_max_new") else None
    gen_extra = torch.Generator(device=gen_device).manual_seed(seed + 1)  # own stream: training stays as without it
    extra_evals: dict = {}

    def extra_eval(step):
        scores, rolls = evaluate(sampler, task, task.heldout, eval_k, extra_max_new, temperature, gen_extra)
        extra_evals[step] = {"pass": sum(scores) / len(scores), "scores": scores,
                             **length_stats(rolls, extra_max_new, policy.eos_id)}
        print(json.dumps({"extra_eval_step": step, "max_new": extra_max_new,
                          **{k_: v for k_, v in extra_evals[step].items() if k_ != "scores"}}), flush=True)

    def log_heldout(step):
        scores, rolls = evaluate(sampler, task, task.heldout, eval_k, max_new, temperature, gen)
        r = sum(scores) / len(scores)
        heldout_curve.append((step, r))
        heldout_scores[step] = scores
        extra = dict(length_stats(rolls, max_new, policy.eos_id))
        if t.get("kl_to_start"):
            extra["kl_to_start"] = float(f"{kl_to_start(policy, start_w, rolls, task.heldout):.4g}")
        if log_routing:
            n = int(t.get("routing_probe", 8))
            if not probe:  # fixed probe batch: the first n held-out prompts with their step-0 completions
                probe["items"] = [(p, r_.seq, routing.completion_positions(len(p), r_.cmask))
                                  for p, r_ in zip(task.heldout[:n], rolls[:n])]
            idx = torch.cat([routing.routing(policy, s_, m_) for _, s_, m_ in probe["items"]], 1)
            if "idx0" not in probe:  # noise floor: two identical forwards (GPU kernels need not be bitwise stable)
                again = torch.cat([routing.routing(policy, s_, m_) for _, s_, m_ in probe["items"]], 1)
                extra["flip_noise_floor"] = round(routing.flip_rate(idx, again)[0], 4)
            probe.setdefault("idx0", idx)
            ratio = routing.load_ratio(routing.expert_load(idx, n_experts))
            flip_any, flip_layers = routing.flip_rate(probe["idx0"], idx)
            extra.update({"load_ratio_max": round(float(ratio.max()), 4),
                          "load_ratio_per_layer": [round(float(x), 4) for x in ratio],
                          "flip_vs_step0": round(flip_any, 4), "flip_vs_step0_per_layer": flip_layers,
                          "top1_flip_vs_step0": round(routing.top1_flip_rate(probe["idx0"], idx), 4),
                          "mass_shift_vs_step0": float(f"{routing.mass_shift(probe['idx0'], idx):.4g}")})
        routing_curve.append({"step": step, **extra})
        return r, extra

    sync_check = None
    if vllm is not None and t.get("verify_sync", True):
        sync_check = check_vllm_sync(policy, vllm, task.train[:2], trainable_named)
        print(json.dumps({"vllm_sync_check": sync_check}), flush=True)
        (out / "vllm_sync_check.json").write_text(json.dumps(sync_check, indent=1))
        if sync_check.get("ok") is False:
            raise RuntimeError(f"vLLM weight sync does not work: {sync_check}")
    h0, extra0 = log_heldout(0)
    if cfg.get("verbose", True):
        print(json.dumps({"step": 0, "heldout_reward": h0, **extra0}), flush=True)
    if extra_max_new:
        extra_eval(0)
    for step in range(1, int(t["steps"]) + 1):
        times = {}
        random.shuffle(order)
        batch = [task.train[i] for i in order[: int(t["prompts_per_step"])]]
        c0 = time.time()
        rolls = sampler.sample(batch, k, max_new, temperature, gen)
        times["rollout"] = time.time() - c0
        sampled_tokens = int(sum(int(r.cmask.sum()) for r in rolls))
        c0 = time.time()
        groups, rewards_all, dropped, mism = [], [], 0, []
        scored = []
        for p, r in zip(batch, rolls):
            rewards = torch.tensor([task.reward(p, row[m].tolist()) for row, m in zip(r.seq[:, len(p):], r.cmask)])
            rewards_all += rewards.tolist()
            scored.append((p, r, grpo.advantages(rewards, algo)))
        times["reward"] = time.time() - c0
        c0 = time.time()
        for p, r, adv in scored:
            if adv is None:
                dropped += 1
                continue
            with torch.no_grad():
                old_lp = chunked_logprobs(policy, r.seq, len(p), micro_rows)
                ref_lp = chunked_logprobs(ref, r.seq, len(p), micro_rows) if ref is not None else None
            isw = None
            if r.rollout_lp is not None:
                mism.append(grpo.mismatch(old_lp, r.rollout_lp, r.cmask))
                if tis_clip:
                    isw = grpo.tis_weights(old_lp, r.rollout_lp, r.cmask, algo, tis_clip)
            groups.append((r.seq, r.cmask, adv.to(device), old_lp, ref_lp, len(p), isw))
        times["old_logprobs"] = time.time() - c0
        stats = {"clip_frac": 0.0, "ratio_mean": 1.0, "kl": 0.0}
        loss_val = 0.0
        step_flip = step_top1 = step_mass = None
        grad_norm = None
        c0 = time.time()
        if groups:
            if log_routing:
                g = groups[0]
                fmask = routing.completion_positions(g[5], g[1])
                idx_before = routing.routing(policy, g[0], fmask)
            if vllm is not None:
                vllm.sleep()
            model.train()
            for _ in range(int(t.get("ppo_epochs", 1))):
                opt.zero_grad()
                for g_ in groups:
                    lv, stats = backward_group(policy, g_, algo, float(t.get("clip_eps", 0.2)),
                                               float(t.get("kl_coef", 0)), len(groups), micro_rows)
                    loss_val += lv
                grad_norm = float(torch.nn.utils.clip_grad_norm_([p for _, p in trainable_named],
                                                                 float(t.get("grad_clip", 1.0))))
                opt.step()
            if log_routing:
                idx_after = routing.routing(policy, g[0], fmask)
                step_flip = routing.flip_rate(idx_before, idx_after)[0]
                step_top1 = routing.top1_flip_rate(idx_before, idx_after)
                step_mass = routing.mass_shift(idx_before, idx_after)
        times["train"] = time.time() - c0
        c0 = time.time()
        sync_bytes = 0
        if vllm is not None:
            vllm.wake_up()
            if groups:
                sync_bytes = vllm.sync(trainable_named)
        times["sync"] = time.time() - c0
        rec = {"step": step, "train_reward": sum(rewards_all) / len(rewards_all), "groups_used": len(groups),
               "groups_dropped": dropped, "loss": loss_val, **stats,
               "sampled_tokens": sampled_tokens,
               "rollout_tokens_per_s": round(sampled_tokens / max(times["rollout"], 1e-9), 1),
               "times_s": {k_: round(v, 3) for k_, v in times.items()}, "step_s": round(sum(times.values()), 3),
               "sync_mb": round(sync_bytes / 2**20, 1), **gpu_memory(),
               "seconds": round(time.time() - start, 1)}
        if mism:
            rec["mismatch_k3"] = round(sum(x["k3"] for x in mism) / len(mism), 6)
            rec["mismatch_abs_logratio"] = round(sum(x["abs_logratio"] for x in mism) / len(mism), 6)
        if step_flip is not None:
            rec["step_flip"] = round(step_flip, 4)
            rec["step_top1_flip"] = round(step_top1, 4)
            rec["step_mass_shift"] = float(f"{step_mass:.4g}")
        if grad_norm is not None:
            rec["grad_norm"] = float(f"{grad_norm:.4g}")
        if step % int(t.get("eval_every", 10)) == 0 or step == int(t["steps"]):
            rec["heldout_reward"], extra = log_heldout(step)
            rec.update({k_: v for k_, v in extra.items() if not k_.endswith("per_layer")})
        metrics_f.write(json.dumps(rec) + "\n")
        metrics_f.flush()
        if cfg.get("verbose", True):
            print(json.dumps(rec), flush=True)
    metrics_f.close()
    if extra_max_new:
        extra_eval(int(t["steps"]))
    saved_weights = None
    if t.get("save_trainable"):
        import hashlib  # noqa: PLC0415
        path = Path(t["save_trainable"])
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({n: p_.detach().to(torch.bfloat16).cpu() for n, p_ in trainable_named}, path)
        saved_weights = {"path": str(path), "bytes": path.stat().st_size,
                         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    checksum_after = router.router_checksum(model, patterns)
    recs = [json.loads(line) for line in metrics_path.read_text().splitlines()]
    later = recs[1:] or recs  # step 1 includes warm-up (compilation, caches); report it separately
    last = max(heldout_scores)
    summary = {"algo": algo, "steps": int(t["steps"]), "k": k, "prompts_per_step": int(t["prompts_per_step"]),
               "max_new": max_new, "sampler": sampler_kind, "trainable_params": n_trainable,
               "fp32_master_params": n_fp32, "tis_clip": tis_clip,
               "router_params_frozen": frozen, "router_checksum_before": checksum_before,
               "router_checksum_after": checksum_after, "router_unchanged": checksum_before == checksum_after,
               "heldout_reward_start": h0, "heldout_reward_end": heldout_curve[-1][1], "heldout_curve": heldout_curve,
               "heldout_change": heldout.paired_bootstrap(heldout_scores[0], heldout_scores[last], seed=seed),
               "groups_used_total": sum(r["groups_used"] for r in recs),
               "trainable_weight_change": {k_: float(f"{v:.4g}") for k_, v in start_w.change().items()},
               "step1_s": recs[0]["step_s"] if recs else None,
               "mean_step_s_after_step1": round(sum(r["step_s"] for r in later) / len(later), 3),
               "mean_times_s_after_step1": {k_: round(sum(r["times_s"][k_] for r in later) / len(later), 3)
                                            for k_ in later[0]["times_s"]},
               "mean_rollout_tokens_per_s": round(sum(r["rollout_tokens_per_s"] for r in later) / len(later), 1),
               "routing_curve": routing_curve, "vllm_sync_path": getattr(vllm, "sync_path", None),
               "vllm_sync_check": sync_check,
               "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "seconds": round(time.time() - start, 1),
               "seconds_per_step": round((time.time() - start) / max(int(t["steps"]), 1), 2),
               "peak_gpu_memory_gb": (round(torch.cuda.max_memory_allocated() / 2**30, 2)
                                      if device.startswith("cuda") else None),
               **{f"end_{k_}": v for k_, v in gpu_memory().items()}, "versions": versions()}
    if any("mismatch_k3" in r for r in recs):
        ks = [r["mismatch_k3"] for r in recs if "mismatch_k3" in r]
        summary["mean_mismatch_k3"] = round(sum(ks) / len(ks), 6)
    summary["router_trainable"] = router_trainable
    summary["saved_trainable_weights"] = saved_weights
    if extra_max_new:
        e0, e1 = extra_evals[0], extra_evals[int(t["steps"])]
        summary["extra_eval"] = {"max_new": extra_max_new,
                                 "start": {k_: v for k_, v in e0.items() if k_ != "scores"},
                                 "end": {k_: v for k_, v in e1.items() if k_ != "scores"},
                                 "change": heldout.paired_bootstrap(e0["scores"], e1["scores"], seed=seed)}
    if adapter is not None:
        m = adapter.manifest
        summary["team5_adapter"] = {k_: m.get(k_) for k_ in ("status", "mode", "model_hash_g5", "model_spec_sha256",
                                                            "learning_report_allowed")}
        summary["team5_adapter"].update(g5_commit=m["g5"]["commit"], checkpoint_sha256=m["checkpoint"]["sha256"],
                                        tokenizer_sha256=m["tokenizer"]["sha256"], ke2=m["ke2"],
                                        router_names=m["selection"]["router_names"],
                                        trainable_names=m["selection"]["trainable_names"])
    if fixture_smoke:  # fixture: a smoke record, never a learning report
        summary["status"] = "FIXTURE_SMOKE_NOT_A_RESULT"
        (out / "smoke_summary.json").write_text(json.dumps(summary, indent=1))
    else:
        (out / "summary.json").write_text(json.dumps(summary, indent=1))
    if not summary["router_unchanged"] and not router_trainable:
        raise RuntimeError("router checksum changed during RL")
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path)
    ap.add_argument("--steps", type=int, help="override train.steps")
    ap.add_argument("--model-path", help="override model.path (a local checkpoint folder)")
    ap.add_argument("--device", help="override device (cpu, cuda, mps)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text())
    if args.out_dir:
        cfg["out_dir"] = str(args.out_dir)
    if args.steps:
        cfg["train"]["steps"] = args.steps
    if args.model_path:
        cfg["model"]["path"] = args.model_path
    if args.device:
        cfg["device"] = args.device
    try:
        s = train(cfg)
    except Exception as e:
        if type(e).__name__ != "WaitingForArtifact":
            raise
        out = Path(cfg.get("out_dir", "out"))
        out.mkdir(parents=True, exist_ok=True)
        rec = {"status": e.result.status, "reasons": e.result.reasons, "manifest": e.result.manifest}
        (out / "status.json").write_text(json.dumps(rec, indent=1, default=str))
        print(json.dumps({"status": e.result.status, "reasons": e.result.reasons}))
        return 3
    print(json.dumps({k: v for k, v in s.items() if k != "heldout_curve"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
