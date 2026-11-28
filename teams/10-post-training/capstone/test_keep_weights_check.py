"""`keep_weights.py check`: the reader's model config is read where Team 9's scorer reads --model-config.

    python -m pytest capstone/test_keep_weights_check.py -q

A file of its own: test_keep_weights.py is skipped as a whole below the torch version Team 9's reader needs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import keep_weights as kw  # noqa: E402

SPEC = {"head_mode": "kas_v2", "vocab_size": 256, "ke_max_pos": 32, "n_layer": 2, "n_embd": 64,
        "moe": {"num_experts": 4, "bias_gamma": 0.001}}
WRAPS = {"run manifest.json": lambda m: {"config": {"model": m}, "status": "x"}, "model key": lambda m: {"model": m},
         "model map": lambda m: m}


def _folder(tmp_path):
    record = {**{k: SPEC[k] for k in kw.ROUTING_KEYS}, "bias_gamma": 0.001, "moe": SPEC["moe"]}
    (tmp_path / kw.MODEL_CONFIG_FILE).write_text(json.dumps(record))
    return str(tmp_path)


@pytest.mark.parametrize("shape", sorted(WRAPS))
def test_check_accepts_the_right_config_and_refuses_another_in_every_shape(tmp_path, capsys, shape):
    run, given = _folder(tmp_path), tmp_path / "given.json"
    given.write_text(json.dumps(WRAPS[shape](SPEC)))
    assert kw.main(["check", "--run-dir", run, "--model-spec", str(given)]) == 0
    assert '"ok": true' in capsys.readouterr().out
    given.write_text(json.dumps(WRAPS[shape]({**SPEC, "n_layer": 3})))
    assert kw.main(["check", "--run-dir", run, "--model-spec", str(given)]) == 1
    assert "differs" in capsys.readouterr().out


@pytest.mark.parametrize("text", ["{", "[1]", '{"config": {}}', '{"model": 3}'])
def test_check_refuses_an_unreadable_model_spec_with_a_message(tmp_path, capsys, text):
    given = tmp_path / "given.json"
    given.write_text(text)
    assert kw.main(["check", "--run-dir", _folder(tmp_path), "--model-spec", str(given)]) == 1
    assert "REFUSED" in capsys.readouterr().out
