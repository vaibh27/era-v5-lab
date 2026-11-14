"""Issue #17: mem_used_mib (MiB) is stored as gpu_memory_used in bytes (x 1,048,576)."""
import json
from pathlib import Path

import pytest

from metrics_writer import BYTES_PER_MIB, METRICS, MetricsWriter, gpu_memory_used

METRIC_LISTS = Path(__file__).resolve().parent.parent / "metrics"


def get_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_one_mib_is_1048576_bytes():
    assert BYTES_PER_MIB == 1024 * 1024 == 1048576


@pytest.mark.parametrize("mib, expected", [
    (0, 0.0),
    (1, 1048576.0),
    (2, 2097152.0),
    (0.5, 524288.0),
    (31009, 32515293184.0),          # 30.28 GiB
])
def test_mib_to_bytes(mib, expected):
    assert gpu_memory_used(mib) == expected


@pytest.mark.parametrize("bad", [None, True, False, "abc", [31009], -1, float("nan"), float("inf"), float("-inf")])
def test_cannot_tell_gives_none(bad):
    assert gpu_memory_used(bad) is None


def test_numpy_scalars():
    np = pytest.importorskip("numpy")
    assert gpu_memory_used(np.int64(31009)) == 32515293184.0
    assert gpu_memory_used(np.float32(2.0)) == 2097152.0
    assert gpu_memory_used(np.array([1, 2])) is None       # not one number


def test_torch_scalars():
    torch = pytest.importorskip("torch")
    assert gpu_memory_used(torch.tensor(31009)) == 32515293184.0


def test_writer_stores_bytes_not_mib(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricsWriter(str(path)) as w:
        w.write(1, loss_ntp=2.0, mem_used_mib=31009)
    row, = get_rows(path)
    assert row["gpu_memory_used"] == 32515293184.0
    assert "mem_used_mib" not in row


def test_bytes_passed_directly_win(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricsWriter(str(path)) as w:
        w.write(1, mem_used_mib=31009, gpu_memory_used=123.0)
    assert get_rows(path)[0]["gpu_memory_used"] == 123.0


@pytest.mark.parametrize("reading", [None, float("nan"), -5, "abc"])
def test_no_usable_reading_no_field(tmp_path, reading):
    path = tmp_path / "metrics.jsonl"
    with MetricsWriter(str(path)) as w:
        w.write(1, loss_ntp=2.0, mem_used_mib=reading)
    row, = get_rows(path)
    assert "gpu_memory_used" not in row and "mem_used_mib" not in row


def test_missing_reading_no_field(tmp_path):
    path = tmp_path / "metrics.jsonl"
    with MetricsWriter(str(path)) as w:
        w.write(1, loss_ntp=2.0)
    assert "gpu_memory_used" not in get_rows(path)[0]


def test_mem_used_mib_is_not_an_unknown_metric(tmp_path, capsys):
    with MetricsWriter(str(tmp_path / "metrics.jsonl")) as w:
        w.write(1, mem_used_mib=31009)
    assert "skipping unknown metric" not in capsys.readouterr().err


def test_gpu_memory_used_is_in_the_metric_list():
    assert "gpu_memory_used" in METRICS


@pytest.mark.parametrize("name", ["metric_list_v0.yaml", "metric_list_v1.yaml"])
def test_metric_lists_say_bytes(name):
    text = (METRIC_LISTS / name).read_text(encoding="utf-8")
    entry = text.split("- name: gpu_memory_used", 1)[1].split("\n- name:", 1)[0]
    assert "unit: bytes" in entry and "1,048,576" in entry
