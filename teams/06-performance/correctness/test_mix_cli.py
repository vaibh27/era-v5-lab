"""The mix loader command line: data and weights resolve into one config; show, summary and same run."""
from __future__ import annotations

import json

import yaml

from dataio import mix_cli
from test_mix_loader import LANES, Corpus


def _weights(c, tmp_path):
    path = tmp_path / "w.yaml"
    path.write_text(yaml.safe_dump({"lanes": LANES, "indic": {"share": 0.08, "split_csv": str(c.csv)}}))
    return path


def _base(tmp_path):
    path = tmp_path / "base.yaml"
    path.write_text(yaml.safe_dump({"seed": 7, "stages": [{"from_batch": 0, "seq_len": 64, "rows": 8}],
                                    "shards": {"verify": "size"}, "caps": {"max_source_passes": 3}}))
    return path


def test_actions_run_on_a_shard_folder(tmp_path, capsys):
    c = Corpus(tmp_path / "c")
    args = ["--data", str(c.root), "--weights", str(_weights(c, tmp_path)), "--config", str(_base(tmp_path)),
            "--batches", "6", "--work", str(tmp_path / "work")]
    assert mix_cli.main(["show"] + args) == 0
    out = capsys.readouterr().out
    assert "batch 0: 8 rows x 64 tokens" in out and "tokens per lane" in out
    cfg = yaml.safe_load((tmp_path / "work" / "config.yaml").read_text())
    assert cfg["shards"]["roots"] == [str(c.root.resolve())] and cfg["shards"]["missing"] == "renormalise"
    assert mix_cli.main(["summary"] + args) == 0
    assert '"manifest"' in capsys.readouterr().out
    assert mix_cli.main(["same"] + args) == 0
    out = capsys.readouterr().out
    assert out.count("PASS") >= 4 and "FAIL" not in out


def test_fix_lanes_takes_the_lane_from_domain(tmp_path):
    c = Corpus(tmp_path / "c")
    m = next(p for p in sorted(c.root.glob("*.json")) if "en-web" in p.name)
    m.write_text(json.dumps(json.loads(m.read_text()) | {"capability_lane": "general_language", "domain": "english"}))
    assert mix_cli.fix_lanes(c.root, tmp_path / "fixed") == 1
    assert json.loads((tmp_path / "fixed" / m.name).read_text())["capability_lane"] == "english"
    assert (tmp_path / "fixed" / m.name.replace(".json", ".bin")).is_symlink()
