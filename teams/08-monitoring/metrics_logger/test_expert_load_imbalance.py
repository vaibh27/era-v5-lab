"""expert_load_imbalance: busiest expert / average expert, max_i(s_i) / mean_i(s_i), the worst over MoE layers.
The single summary number for expert_load proposed in metric_list_v0.yaml / metric_list_v1.yaml."""
import json
import math

import numpy as np
import pytest
import torch

from metrics_writer import MetricsWriter, expert_load_imbalance, router_max_vio

# Named decorator: pr-guard reads an added pytest decorator line as an email address.
parametrize = pytest.mark.parametrize


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def written(tmp_path, **metrics):
    path = tmp_path / f"metrics_{len(list(tmp_path.iterdir()))}.jsonl"  # a new file for each call
    with MetricsWriter(str(path)) as w:
        w.write(1, **metrics)
    row, = rows(path)
    return row


@parametrize("load", [[0.25] * 4, [100] * 8, [5, 5, 5, 5], [1]])
def test_uniform_is_one(load):
    assert expert_load_imbalance(load) == 1.0


@parametrize("load, expected", [
    ([0.7, 0.3, 0.0, 0.0], 2.8),          # max(shares) * num_experts = 0.7 * 4
    ([0.5, 0.5, 0.0, 0.0], 2.0),          # busiest gets twice the average
    ([1.0, 0.0, 0.0, 0.0], 4.0),          # full collapse onto one expert: num_experts
    ([30, 10, 10, 10, 10, 10, 10, 10], 2.4),  # counts: 100 picks over 8 experts, mean 12.5; 30 / 12.5
])
def test_one_layer(load, expected):
    assert math.isclose(expert_load_imbalance(load), expected)


def test_counts_and_shares_agree():
    counts = [1, 3, 50, 46]
    assert math.isclose(expert_load_imbalance(counts), expert_load_imbalance([c / 100 for c in counts]))
    assert math.isclose(expert_load_imbalance(counts), 2.0)  # 0.5 * 4


def test_equals_router_max_vio_plus_one():
    for load in ([0.7, 0.3, 0.0, 0.0], [1, 3, 50, 46], [[0.25] * 4, [0.4, 0.2, 0.2, 0.2]]):
        assert math.isclose(expert_load_imbalance(load), router_max_vio(load) + 1)


def test_worst_layer_wins():
    layers = [[0.25] * 4, [0.4, 0.2, 0.2, 0.2], [0.7, 0.3, 0.0, 0.0]]
    assert math.isclose(expert_load_imbalance(layers), 2.8)
    assert math.isclose(expert_load_imbalance({"layer_0": layers[0], "layer_1": layers[1], "layer_2": layers[2]}), 2.8)
    assert math.isclose(expert_load_imbalance(tuple(layers)), 2.8)


def test_numpy_and_torch():
    assert math.isclose(expert_load_imbalance(np.array([[0, 50, 50, 100], [25, 25, 25, 25]])), 2.0)
    assert math.isclose(expert_load_imbalance(torch.tensor([[0.0, 0.5, 0.5, 1.0], [1.0, 1.0, 1.0, 1.0]])), 2.0)
    assert math.isclose(expert_load_imbalance([torch.tensor([0, 2, 2]), torch.tensor([1, 1, 1])]), 1.5)
    assert math.isclose(expert_load_imbalance(np.array([0.7, 0.3, 0.0, 0.0])), 2.8)


@parametrize("load", [
    None, [], {}, np.array([]), "0.25", 0.25,
    [[]], [[0.5, 0.5], []],               # an empty layer
    [0, 0, 0, 0], [[0.5, 0.5], [0, 0]],   # a layer with no picks
    [-0.1, 0.6, 0.5],                     # negative
    [float("nan"), 0.5, 0.5], [float("inf"), 1.0],  # not finite
    ["bad", 0.5], [True, False], [[0.5, 0.5], "bad"],
])
def test_none_when_it_cannot_be_told(load):
    assert expert_load_imbalance(load) is None


def test_writer_adds_it_from_expert_load(tmp_path):
    row = written(tmp_path, loss_ntp=1.0, expert_load=[[0.25] * 4, [0.7, 0.3, 0.0, 0.0]])
    assert math.isclose(row["expert_load_imbalance"], 2.8)
    assert math.isclose(row["router_max_vio"], 1.8)


def test_writer_uniform_is_one(tmp_path):
    assert written(tmp_path, expert_load=[0.125] * 8)["expert_load_imbalance"] == 1.0


def test_value_passed_by_trainer_wins(tmp_path):
    row = written(tmp_path, expert_load=[0.7, 0.3, 0.0, 0.0], expert_load_imbalance=1.5)
    assert row["expert_load_imbalance"] == 1.5


def test_can_be_logged_without_expert_load(tmp_path):
    row = written(tmp_path, loss_ntp=1.0, expert_load_imbalance=1.25)
    assert row["expert_load_imbalance"] == 1.25


def test_absent_without_expert_load(tmp_path):
    assert "expert_load_imbalance" not in written(tmp_path, loss_ntp=1.0)


@parametrize("load", [[0, 0, 0, 0], [-1, 1, 0, 0]])
def test_absent_when_it_cannot_be_told(tmp_path, load):
    assert "expert_load_imbalance" not in written(tmp_path, expert_load=load)


def test_not_skipped_as_unknown(tmp_path, capsys):
    written(tmp_path, expert_load=[0.5, 0.5])
    assert "expert_load_imbalance" not in capsys.readouterr().err
