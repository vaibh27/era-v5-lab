"""Group-relative policy-gradient losses. The algorithm is a config choice, because TSAI picks it at Gate 4.

For one prompt with k sampled completions and rewards r_1..r_k:

    algo       advantage A_i                         ratio                         reduction over tokens
    grpo       (r_i - mean) / (std + eps)            per token                     mean per sequence, then mean
    dr_grpo    r_i - mean                            per token                     sum / max_new  (no length bias)
    rloo       r_i - mean of the other k-1 rewards   per token                     sum / max_new
    gspo       (r_i - mean) / (std + eps)            per sequence: exp(mean token  one term per sequence
                                                     log-ratio), proposed for MoE

std is the population std (divide by k). A group whose rewards are all equal has no signal: advantages() returns
None and the trainer drops it (DAPO's dynamic sampling, arXiv 2503.14476). The surrogate is PPO-clipped:
min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A). The optional KL to a reference policy uses the k3 estimator
exp(ref - new) - (ref - new) - 1 per token.

Sources: GRPO (DeepSeekMath, arXiv 2402.03300), Dr. GRPO (arXiv 2503.20783), RLOO (arXiv 2402.14740),
GSPO (arXiv 2507.18071).
"""
from __future__ import annotations

import torch

ALGOS = ("grpo", "dr_grpo", "rloo", "gspo")


def advantages(rewards: torch.Tensor, algo: str, eps: float = 1e-4) -> torch.Tensor | None:
    if algo not in ALGOS:
        raise ValueError(f"unknown algo {algo!r}; choose from {ALGOS}")
    r = rewards.float()
    if r.numel() < 2 or torch.all(r == r[0]):
        return None
    mean = r.mean()
    if algo in ("grpo", "gspo"):
        return (r - mean) / (r.std(correction=0) + eps)
    if algo == "dr_grpo":
        return r - mean
    k = r.numel()
    return (r - mean) * k / (k - 1)  # r_i - mean of the others


def tis_weights(old_lp: torch.Tensor, rollout_lp: torch.Tensor, mask: torch.Tensor, algo: str,
                clip_max: float) -> torch.Tensor:
    """Truncated importance weights for the sampler/trainer gap (TIS; Yao et al. 2025, research/g10_gpu/q3).

    The completions were sampled by vLLM (rollout_lp) but the PPO ratio uses the trainer's own logprobs (old_lp),
    so each term is re-weighted by pi_train_old / pi_rollout, truncated at clip_max so one token can't dominate.
    Token level for grpo, dr_grpo and rloo ([k, T]); sequence level for gspo ([k], geometric mean over tokens),
    to match its sequence ratio. Detached: the weights scale the gradient, they don't receive one.
    """
    m = mask.float()
    d = (old_lp - rollout_lp).detach() * m
    if algo == "gspo":
        return torch.exp(d.sum(-1) / m.sum(-1).clamp_min(1)).clamp(max=clip_max)
    return torch.exp(d).clamp(max=clip_max) * m


def mismatch(old_lp: torch.Tensor, rollout_lp: torch.Tensor, mask: torch.Tensor) -> dict:
    """How far the sampler is from the trainer on the sampled tokens: mean |log ratio| and the K3 estimate of
    KL(sampler || trainer), exp(r) - 1 - r with r = log pi_train - log pi_rollout."""
    m = mask.float()
    r = (old_lp - rollout_lp).detach() * m
    n = m.sum().clamp_min(1)
    return {"abs_logratio": float((r.abs()).sum() / n), "k3": float(((torch.exp(r) - 1 - r) * m).sum() / n)}


def policy_loss(new_lp: torch.Tensor, old_lp: torch.Tensor, adv: torch.Tensor, mask: torch.Tensor, algo: str,
                clip_eps: float = 0.2, ref_lp: torch.Tensor | None = None, kl_coef: float = 0.0,
                is_weight: torch.Tensor | None = None) -> tuple[torch.Tensor, dict]:
    """new_lp, old_lp, ref_lp, mask: [k, T]; adv: [k]; is_weight: tis_weights() or None. Returns (loss, stats)."""
    m = mask.float()
    n_tok = m.sum(-1).clamp_min(1)
    log_ratio = (new_lp - old_lp) * m
    if algo == "gspo":
        seq_ratio = torch.exp(log_ratio.sum(-1) / n_tok)
        surr = torch.minimum(seq_ratio * adv, seq_ratio.clamp(1 - clip_eps, 1 + clip_eps) * adv)
        if is_weight is not None:
            surr = surr * is_weight
        loss = -surr.mean()
        clipped = ((seq_ratio - 1).abs() > clip_eps).float().mean()
        ratio_mean = seq_ratio.mean()
    else:
        ratio = torch.exp(log_ratio)
        a = adv[:, None]
        surr = torch.minimum(ratio * a, ratio.clamp(1 - clip_eps, 1 + clip_eps) * a) * m
        if is_weight is not None:
            surr = surr * is_weight
        if algo == "grpo":
            loss = -(surr.sum(-1) / n_tok).mean()
        else:  # dr_grpo, rloo: constant normaliser, no per-sequence length division
            loss = -surr.sum() / (m.shape[0] * m.shape[1])
        clipped = ((((ratio - 1).abs() > clip_eps).float() * m).sum() / m.sum().clamp_min(1))
        ratio_mean = (ratio * m).sum() / m.sum().clamp_min(1)
    kl = torch.zeros((), device=new_lp.device)
    if ref_lp is not None and kl_coef > 0:
        d = (ref_lp - new_lp) * m
        kl = ((torch.exp(d) - d - 1) * m).sum() / m.sum().clamp_min(1)
        loss = loss + kl_coef * kl
    return loss, {"clip_frac": float(clipped.detach()), "ratio_mean": float(ratio_mean.detach()), "kl": float(kl.detach())}
