"""CPU smoke probe for the G10 SFT objective (Package T1, code and tests only).

What it checks, on a tiny randomly initialised causal LM and code-generated integer fixtures:

* rows carry `input_ids` and a 0/1 `loss_mask` per token, the layout `convert.py --tokenizer` writes;
  a 1 means "this token is a target";
* labels come from `loss_mask` with the causal shift: the logit at position t predicts token t+1, and
  it is supervised only if `loss_mask[t + 1] == 1`. Position 0 can never be a target;
* rows with no valid shifted target (all masked, or truncated past the assistant) are rejected;
* padding is found from each row's length, never from the pad token id, so a pad id equal to the EOS id
  cannot hide a supervised EOS target;
* the loss is the sum of token NLL over valid shifted targets divided by their global count over the
  whole update (all accumulation microbatches), not a mean of per-row or per-microbatch means;
* a few AdamW updates on one fixed batch reduce that loss, and the FP32 weights actually change.

It does not load a tokenizer, a checkpoint or any admitted data, and it imports nothing from the
frozen-tokenizer or OCR-2 admission code. It says nothing about OLMoE, G5 or the data mixture.

Usage:
    python sft/train_probe.py --config sft/config/probe.yaml            # synthetic fixture
    python sft/train_probe.py --config sft/config/probe.yaml --data rows.jsonl   # input_ids + loss_mask rows
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

IGNORE = -100

# Kind of each shifted target position (what the logit at position t is asked to predict).
KIND_PAD = 0  # t + 1 lies past the row's real length
KIND_CONTEXT = 1  # real token, loss_mask 0 (system, user, tool result, role header, final EOS in G10)
KIND_TARGET = 2  # real token, loss_mask 1, not EOS
KIND_EOS_TARGET = 3  # real token, loss_mask 1, equal to eos_id

REQUIRED_KEYS = {
    "seed", "vocab_size", "d_model", "n_heads", "n_layers", "max_len", "pad_id", "eos_id",
    "lr", "weight_decay", "betas", "warmup_steps", "max_steps", "grad_clip", "microbatch_size",
    "fixture_rows", "overfit_ratio",
}
MAX_UPDATES = 30  # G10_TRAINING.md section 3, protocol step 2: "at most 30 updates"


class ZeroTargetError(ValueError):
    """The row has no valid shifted supervised target."""


# ---------------------------------------------------------------- config


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return check_config(cfg)


def check_config(cfg: dict) -> dict:
    if not isinstance(cfg, dict):
        raise ValueError("config must be a mapping")
    missing = REQUIRED_KEYS - cfg.keys()
    unknown = cfg.keys() - REQUIRED_KEYS
    if missing:
        raise ValueError(f"config missing keys: {sorted(missing)}")
    if unknown:
        raise ValueError(f"config has unknown keys: {sorted(unknown)}")
    if not 1 <= cfg["max_steps"] <= MAX_UPDATES:
        raise ValueError(f"max_steps must be 1..{MAX_UPDATES}")
    for key in ("pad_id", "eos_id"):
        if not 0 <= cfg[key] < cfg["vocab_size"]:
            raise ValueError(f"{key} outside the vocabulary")
    if cfg["d_model"] % cfg["n_heads"]:
        raise ValueError("d_model must be divisible by n_heads")
    if cfg["microbatch_size"] < 1 or cfg["max_len"] < 2:
        raise ValueError("microbatch_size >= 1 and max_len >= 2 are needed")
    return cfg


# ---------------------------------------------------------------- rows and labels


def validate_example(row: dict, vocab_size: int | None = None) -> dict:
    """Check one row; return it unchanged. Raises ValueError (or ZeroTargetError) if it cannot train."""
    ids, mask = row.get("input_ids"), row.get("loss_mask")
    if not isinstance(ids, list) or not isinstance(mask, list):
        raise ValueError("row needs list fields input_ids and loss_mask")
    if len(ids) != len(mask):
        raise ValueError(f"input_ids ({len(ids)}) and loss_mask ({len(mask)}) differ in length")
    if len(ids) < 2:
        raise ZeroTargetError("row shorter than 2 tokens has no shifted target")
    if any(m not in (0, 1) for m in mask):
        raise ValueError("loss_mask must be 0/1")
    if vocab_size is not None and any(not 0 <= t < vocab_size for t in ids):
        raise ValueError("token id outside the vocabulary")
    if sum(mask[1:]) == 0:
        raise ZeroTargetError("no valid shifted target (all masked, or only position 0 marked)")
    return row


def truncate(row: dict, max_len: int) -> dict:
    """Keep the first `max_len` tokens. Raises ZeroTargetError if that removes every target."""
    out = dict(row, input_ids=row["input_ids"][:max_len], loss_mask=row["loss_mask"][:max_len])
    return validate_example(out)


def shift_labels(input_ids: list[int], loss_mask: list[int]) -> list[int]:
    """labels[t] = input_ids[t + 1] if loss_mask[t + 1] else IGNORE; length len(input_ids) - 1."""
    return [input_ids[t + 1] if loss_mask[t + 1] else IGNORE for t in range(len(input_ids) - 1)]


@dataclass
class Batch:
    input_ids: torch.Tensor  # (B, T) right-padded with pad_id
    attention_mask: torch.Tensor  # (B, T) 1 on real tokens, from row lengths
    labels: torch.Tensor  # (B, T - 1) shifted targets, IGNORE where not supervised
    kinds: torch.Tensor  # (B, T - 1) KIND_* per shifted position

    @property
    def n_targets(self) -> int:
        return int((self.labels != IGNORE).sum())


def collate(rows: list[dict], pad_id: int, eos_id: int) -> Batch:
    """Right-pad a list of validated rows. Padding comes from lengths, never from token values."""
    if not rows:
        raise ValueError("empty batch")
    T = max(len(r["input_ids"]) for r in rows)
    ids = torch.full((len(rows), T), pad_id, dtype=torch.long)
    attn = torch.zeros((len(rows), T), dtype=torch.long)
    labels = torch.full((len(rows), T - 1), IGNORE, dtype=torch.long)
    kinds = torch.full((len(rows), T - 1), KIND_PAD, dtype=torch.long)
    for b, r in enumerate(rows):
        n = len(r["input_ids"])
        ids[b, :n] = torch.tensor(r["input_ids"])
        attn[b, :n] = 1
        lab = shift_labels(r["input_ids"], r["loss_mask"])
        labels[b, : n - 1] = torch.tensor(lab)
        for t, y in enumerate(lab):
            if y == IGNORE:
                kinds[b, t] = KIND_CONTEXT
            else:
                kinds[b, t] = KIND_EOS_TARGET if y == eos_id else KIND_TARGET
    return Batch(ids, attn, labels, kinds)


# ---------------------------------------------------------------- loss


def masked_nll_sum(logits: torch.Tensor, labels: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Sum of token NLL over positions with a label, and how many there were.

    `logits` is (B, T - 1, V) (already shifted: logit t predicts token t + 1), `labels` is (B, T - 1).
    """
    total = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1), ignore_index=IGNORE, reduction="sum")
    return total, int((labels != IGNORE).sum())


def shifted_logits(model: nn.Module, batch: Batch) -> torch.Tensor:
    return model(batch.input_ids, batch.attention_mask)[:, :-1, :]


def batch_loss(model: nn.Module, batch: Batch) -> torch.Tensor:
    """Valid-token mean loss for one batch on its own (sum / its own count)."""
    total, count = masked_nll_sum(shifted_logits(model, batch), batch.labels)
    return total / count


def accumulate_gradients(model: nn.Module, microbatches: list[Batch]) -> tuple[float, int]:
    """Backward over all microbatches so the gradient is that of sum(NLL) / global target count.

    Returns (loss value for the update, global target count). Gradients add into `.grad`; the caller zeroes them.
    """
    global_count = sum(mb.n_targets for mb in microbatches)
    if global_count == 0:
        raise ZeroTargetError("update has no valid targets")
    loss_value = 0.0
    for mb in microbatches:
        total, _ = masked_nll_sum(shifted_logits(model, mb), mb.labels)
        part = total / global_count
        part.backward()
        loss_value += part.item()
    return loss_value, global_count


# ---------------------------------------------------------------- tiny model


class TinyCausalLM(nn.Module):
    """Embedding + causal transformer encoder + untied LM head. Dropout is 0 (diagnostic run).

    Rows are right-padded, so under the causal mask a real position never attends to a pad position;
    pad positions themselves only see real tokens before them, so no attention row is fully masked.
    """

    def __init__(self, vocab_size: int, d_model: int, n_heads: int, n_layers: int, max_len: int):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, 4 * d_model, dropout=0.0, batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        T = input_ids.size(1)
        pos = torch.arange(T, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(pos)[None]
        causal = nn.Transformer.generate_square_subsequent_mask(T, device=input_ids.device)
        x = self.blocks(x, mask=causal, is_causal=True)
        return self.head(self.norm(x))


def build_model(cfg: dict) -> TinyCausalLM:
    torch.manual_seed(cfg["seed"])
    return TinyCausalLM(cfg["vocab_size"], cfg["d_model"], cfg["n_heads"], cfg["n_layers"], cfg["max_len"])


# ---------------------------------------------------------------- synthetic fixture


def synthetic_rows(cfg: dict) -> list[dict]:
    """Code-generated integer rows shaped like G10 chat-format v0 (test data only, never training text).

    Reserved ids: 0 pad, 1 eos (config), 2 <|im_start|>, 3 <|im_end|>, 4 newline, 5 role-user,
    6 role-assistant, 7 role-tool, 8 tool-call marker. Content ids are drawn from 9..vocab-1.
    Masking follows convert.py: assistant body and its <|im_end|> are 1; system/user/tool turns, the
    `<|im_start|>assistant` header, the newline after <|im_end|> and the final EOS are 0. Row lengths differ,
    every third row has a tool call followed by a masked tool result and a second assistant turn.
    """
    rng = random.Random(cfg["seed"])
    eos = cfg["eos_id"]
    START, END, NL, USER, ASSIST, TOOL, CALL = 2, 3, 4, 5, 6, 7, 8
    lo = 9
    hi = cfg["vocab_size"] - 1

    def content(n: int) -> list[int]:
        return [rng.randint(lo, hi) for _ in range(n)]

    rows = []
    for i in range(cfg["fixture_rows"]):
        ids: list[int] = []
        mask: list[int] = []

        def add(tokens: list[int], m: int) -> None:
            ids.extend(tokens)
            mask.extend([m] * len(tokens))

        add([START, USER] + content(rng.randint(2, 5)) + [END, NL], 0)
        add([START, ASSIST], 0)
        if i % 3 == 0:
            add([CALL] + content(3) + [END], 1)  # assistant tool call
            add([NL, START, TOOL] + content(rng.randint(2, 4)) + [END, NL], 0)  # tool result, masked
            add([START, ASSIST], 0)
        add(content(rng.randint(3, 7)) + [END], 1)  # assistant reply and its turn end
        add([NL, eos], 0)
        rows.append(validate_example({"input_ids": ids, "loss_mask": mask}, cfg["vocab_size"]))
    return rows


def load_jsonl(path: str | Path, cfg: dict) -> tuple[list[dict], dict]:
    """Read rows with input_ids + loss_mask; truncate to max_len; reject (and count) rows that cannot train."""
    rows, rejected = [], {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        raw = json.loads(line)
        try:
            row = truncate(validate_example({"input_ids": raw.get("input_ids"), "loss_mask": raw.get("loss_mask")}, cfg["vocab_size"]), cfg["max_len"])
        except ZeroTargetError:
            rejected["zero_targets"] = rejected.get("zero_targets", 0) + 1
        except ValueError:
            rejected["malformed"] = rejected.get("malformed", 0) + 1
        else:
            rows.append(row)
    return rows, rejected


# ---------------------------------------------------------------- probe


def warmup_lambda(warmup_steps: int):
    return lambda step: min(1.0, (step + 1) / warmup_steps) if warmup_steps > 0 else 1.0


def microbatches(rows: list[dict], size: int, pad_id: int, eos_id: int) -> list[Batch]:
    return [collate(rows[i : i + size], pad_id, eos_id) for i in range(0, len(rows), size)]


def param_vector(model: nn.Module) -> torch.Tensor:
    return torch.cat([p.detach().reshape(-1).clone() for p in model.parameters() if p.requires_grad])


def run_probe(cfg: dict, rows: list[dict]) -> dict:
    """Overfit one fixed batch for cfg['max_steps'] AdamW updates. Returns a JSON-able report."""
    cfg = check_config(cfg)
    rows = [truncate(validate_example(r, cfg["vocab_size"]), cfg["max_len"]) for r in rows]
    model = build_model(cfg)
    model.train()
    mbs = microbatches(rows, cfg["microbatch_size"], cfg["pad_id"], cfg["eos_id"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, warmup_lambda(cfg["warmup_steps"]))
    start = param_vector(model)
    losses, grad_norms, lrs, tokens = [], [], [], []
    for _ in range(cfg["max_steps"]):
        opt.zero_grad(set_to_none=True)
        loss, count = accumulate_gradients(model, mbs)
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
        lrs.append(opt.param_groups[0]["lr"])
        opt.step()
        sched.step()
        losses.append(loss)
        grad_norms.append(float(gn))
        tokens.append(count)
    model.eval()
    with torch.no_grad():
        total = sum(masked_nll_sum(shifted_logits(model, mb), mb.labels)[0].item() for mb in mbs)
        final_eval = total / sum(mb.n_targets for mb in mbs)
    end = param_vector(model)
    kinds = torch.cat([mb.kinds.reshape(-1) for mb in mbs])
    report = {
        "rows": len(rows),
        "microbatches": len(mbs),
        "steps": len(losses),
        "losses": losses,
        "grad_norms": grad_norms,
        "lrs": lrs,
        "targets_per_update": tokens,
        "eos_targets": int((kinds == KIND_EOS_TARGET).sum()),
        "pad_positions": int((kinds == KIND_PAD).sum()),
        "initial_loss": losses[0],
        "final_train_loss": losses[-1],
        "final_eval_loss": final_eval,
        "weight_change_l2": float((end - start).norm()),
        "param_dtype": str(next(model.parameters()).dtype),
        "all_finite": all(math.isfinite(v) for v in losses + grad_norms + [final_eval]),
    }
    report["overfit_pass"] = bool(report["all_finite"] and final_eval < cfg["overfit_ratio"] * losses[0] and report["weight_change_l2"] > 0)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(Path(__file__).with_name("config") / "probe.yaml"))
    ap.add_argument("--data", help="JSONL rows with input_ids and loss_mask (default: synthetic fixture)")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    torch.manual_seed(cfg["seed"])
    if args.data:
        rows, rejected = load_jsonl(args.data, cfg)
        rows = rows[: cfg["fixture_rows"]]
    else:
        rows, rejected = synthetic_rows(cfg), {}
    report = run_probe(cfg, rows)
    report["rejected"] = rejected
    json.dump(report, sys.stdout, indent=2)
    print()
    return 0 if report["overfit_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
