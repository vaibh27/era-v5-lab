"""Tests for validation_bpb in metrics_writer.py.

    PYTHONPATH=teams/08-monitoring pytest teams/08-monitoring/ -q
    cd teams/08-monitoring/metrics_logger && python -m pytest test_validation_bpb.py -q
"""
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from metrics_writer import MetricsWriter, validation_bpb

G5_HARNESS = Path(__file__).resolve().parents[2] / "05-architecture" / "harness"


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("nats, nbytes, bpb", [
    (math.log(2) * 1000, 1000, 1.0),                 # ln(2) nats per byte = 1 bit per byte
    (2000.0, 1000, 2.0 / math.log(2)),
    (torch.tensor(50.0, dtype=torch.float64), torch.tensor(25), 2.0 / math.log(2)),
    (np.float64(30.0), np.int64(10), 3.0 / math.log(2)),
    (0.0, 10, 0.0),
])
def test_bits_per_byte(nats, nbytes, bpb):
    assert validation_bpb(nats, nbytes) == pytest.approx(bpb)


@pytest.mark.parametrize("nats, nbytes", [
    (None, 10), (10.0, None),                                # missing
    (10.0, 0), (10.0, -3), (10.0, 2.5), (10.0, True),        # not a positive whole byte count
    (math.nan, 10), (math.inf, 10), (-1.0, 10), ("x", 10), (True, 10),
])
def test_cannot_be_told(nats, nbytes):
    assert validation_bpb(nats, nbytes) is None


def test_writer_only_rows_with_validation_get_it(tmp_path):
    path = tmp_path / "metrics.jsonl"
    nll, nbytes = 1160.0, 1000
    with MetricsWriter(str(path), flush_every=1) as w:
        w.write(99, loss_ntp=3.1)
        w.write(100, loss_ntp=3.0, **{"val/nll_nats": nll, "val/utf8_bytes": nbytes, "val/scored_tokens": 400,
                                      "val/bpb": 9.9})       # worked out from nll / bytes, not G5's val/bpb
        w.write(101, loss_ntp=2.9, validation_bpb=1.25)                    # passed by the trainer: kept
        w.write(102, validation_bpb=1.2, **{"val/nll_nats": 1.0, "val/utf8_bytes": 1})
    r = rows(path)
    assert "validation_bpb" not in r[0]
    assert r[1]["validation_bpb"] == pytest.approx(nll / (math.log(2) * nbytes))
    assert r[1]["validation_loss"] == pytest.approx(nll / 400)          # validation_loss still worked out
    assert [x.get("validation_bpb") for x in r[2:]] == [1.25, 1.2]
    assert all(not k.startswith("val/") for x in r for k in x)          # the inputs are used up, not stored


def test_falls_back_to_g5_val_bpb(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricsWriter(str(path), flush_every=1) as w:
        w.write(1, **{"val/bpb": 1.4})                                  # no nll / bytes: G5's own value
        w.write(2, **{"val/nll_nats": 10.0, "val/utf8_bytes": 0, "val/bpb": 1.3})
        w.write(3, **{"val/bpb": float("nan")})                         # not a number to plot: no value
        w.write(4, **{"val/bpb": True})
    assert [x.get("validation_bpb") for x in rows(path)] == [1.4, 1.3, None, None]


def test_inputs_are_not_reported_as_unknown(tmp_path, capsys):
    with MetricsWriter(str(tmp_path / "m.jsonl")) as w:
        w.write(1, **{"val/nll_nats": 10.0, "val/utf8_bytes": 5, "val/bpb": 2.9})
    err = capsys.readouterr().err
    assert "val/utf8_bytes" not in err and "val/bpb" not in err


class _Windows:
    """The part of G5's DocumentWindows that evaluate() uses: windows of tokens, -100 where nothing is scored."""
    def __init__(self, seqs, bytes_per_token):
        self.windows = seqs
        self.window_slices = [None] * len(seqs)
        self.bytes_per_token = bytes_per_token
        self.raw_bytes = sum(len(s) for s in seqs) * bytes_per_token

    def batch(self, idx, device):
        idx = list(idx)
        x = torch.tensor([self.windows[i] for i in idx], device=device)
        y = torch.roll(x, -1, dims=1)
        y[:, -1] = -100
        lengths = torch.tensor([[self.bytes_per_token] * len(self.windows[i]) for i in idx])
        return x, y, lengths


class _Model(torch.nn.Module):
    def __init__(self, vocab=11):
        super().__init__()
        torch.manual_seed(0)
        self.emb = torch.nn.Embedding(vocab, vocab)

    def forward(self, x, mask):
        return self.emb(x)


def test_matches_g5_evaluate():
    """validation_bpb from the real G5 evaluate() result equals its val/bpb."""
    if not (G5_HARNESS / "week2" / "evaluation.py").is_file():
        pytest.skip("G5 week2 harness not in this checkout")
    sys.path.insert(0, str(G5_HARNESS))
    try:
        from week2.evaluation import evaluate
    finally:
        sys.path.remove(str(G5_HARNESS))
    val = _Windows([[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 1, 2], [3, 3, 3, 3, 3, 3]], bytes_per_token=3)
    result = evaluate(_Model(), val, 2, "cpu", nullcontext)
    assert result["val/utf8_bytes"] == 54
    assert validation_bpb(result["val/nll_nats"], result["val/utf8_bytes"]) == pytest.approx(result["val/bpb"], rel=1e-12)
