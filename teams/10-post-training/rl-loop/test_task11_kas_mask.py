"""DT mask and batched RL integration on Team 5's KAS classes with scripted fixture rewards."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "capstone"), str(HERE.parent / "dt" / "tests")]

import train  # noqa: E402
import team5_adapter as ta  # noqa: E402
from test_mask import dtmask  # noqa: E402
from test_rl_loop import MIXED, MIXED_B, ScriptedTask  # noqa: E402
from test_router_bias_kas import _kas_model, _routing_bias  # noqa: E402


def test_baked_dt_mask_survives_g5_batch_with_kas_head(tmp_path, monkeypatch):
    model = _routing_bias(_kas_model())
    dtmask.convert(model, dtmask.TARGET_SETS["g5_gate"], total_steps=1, seed=3)
    masks = dtmask.bake(model)
    assert sum(int((~keep).sum()) for keep in masks.values()) > 0
    # The RL phase trains the surviving weights. The router is frozen by the real trainer.
    model.requires_grad_(True)
    bundle = ta.Bundle(
        mode="fixture", model=model, spec=model.spec if hasattr(model, "spec") else {
            "block_size": 32, "vocab_size": 131072,
        }, artifacts=None,
        special={"stop_ids": [130816, 130822], "pad_id": 130726,
                 "tokens": {"end_of_turn": {"id": 130816}}},
        selection={}, manifest={}, template_path=Path("unused"), render_mod=None, kv_cache=True,
    )
    policy = bundle.policy()
    monkeypatch.setattr(train, "build_model", lambda cfg, device: (policy, None))
    monkeypatch.setattr(
        train, "build_task",
        lambda cfg, codec, seed: ScriptedTask({MIXED: [0, 1, 0, 1], MIXED_B: [1, 0, 1, 0]}),
    )
    blob, metadata = dtmask.pack_masks(masks)
    mask_path = tmp_path / "dt_keep_masks.bin"
    mask_path.write_bytes(blob)
    Path(str(mask_path) + ".json").write_text(json.dumps(metadata))
    cfg = {
        "seed": 7, "device": "cpu", "out_dir": str(tmp_path / "rl"), "verbose": False,
        "model": {"kind": "team5_adapter"},
        "task": {"kind": "scripted"},
        "train": {
            "algo": "objective_v1", "sampler": "g5_batch", "sampler_max_rows": 8,
            "steps": 3, "prompts_per_step": 2, "k": 4, "max_new": 2, "temperature": 1.0,
            "lr": 1e-4, "ppo_epochs": 1, "kl_coef": 0.0, "eval_every": 3, "eval_k": 2,
            "micro_rows": 1, "logprob_gate": True, "update_mode": "auto",
            "keep_zero": {"path": str(mask_path), "enforce": False},
        },
    }
    result = train.train(cfg)
    assert result["sampler"] == "g5_batch"
    assert result["update_mode"] == "eval" and result["logprob_gate"]["ok"]
    assert result["objective"]["steps_with_optimizer_update"] == 3
    assert result["router_unchanged"] and result["router_bias_unchanged"]
    assert result["keep_zero"]["deleted_at_start"] > 0
    assert result["keep_zero"]["max_deleted_nonzero"] == result["keep_zero"]["deleted_nonzero_end"] == 0
    assert result["trainable_weight_change"]["share_changed"] > 0
