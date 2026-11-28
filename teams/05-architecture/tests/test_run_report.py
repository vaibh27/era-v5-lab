"""run_report.py: the run report reads metrics.jsonl and gets the numbers right."""

from pathlib import Path
import json
import sys
from tempfile import TemporaryDirectory
import unittest

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM / "harness"))
import run_report as rr

DRYRUN1 = TEAM / "jobs/evidence/2026-11-15-dryrun1-full/metrics.jsonl"


def row(step, loss=5.0, norm=1.0, lr=1e-3, scored=100, positions=200, ms=1000.0, **extra):
    return {"step": step, "loss": loss, "gradient_norm": norm, "lr": lr, "train/scored_tokens": scored,
            "train/input_positions": positions, "perf/step_ms": ms, **extra}


class DryRun1Tests(unittest.TestCase):
    """The committed dry run 1 record: the report must reproduce its known numbers."""

    @classmethod
    def setUpClass(cls):
        cls.text = rr.report(rr.read_rows(DRYRUN1), "dryrun1")

    def test_health_and_steps(self):
        self.assertIn("1221 training rows, last step 1221", self.text)
        self.assertIn("Non-finite loss or gradient norm: none", self.text)

    def test_fill_matches_the_log(self):
        # 7,206,344 scored tokens in 20,004,864 input positions (36.0%), as #825 quotes.
        self.assertIn("7,206,344 scored tokens in 20,004,864 input positions (36.0%)", self.text)

    def test_heldout_bpb_at_each_eval(self):
        for step, bpb in [(0, "2.5052"), (611, "1.6217"), (1221, "1.5159")]:
            self.assertIn(f"| {step} | {bpb} |", self.text)

    def test_no_expert_load_recorded(self):
        self.assertIn("None recorded", self.text)


class ExpertLoadTests(unittest.TestCase):
    def test_layer_statistics(self):
        counts = [100] * 16
        counts[3] = 1000
        st = rr.layer_load(counts)
        self.assertAlmostEqual(st["busiest"], 1000 / 2500)
        self.assertAlmostEqual(st["max_over_mean"], 1000 / (2500 / 16))
        self.assertEqual((st["dead"], st["experts"]), (0, 16))
        self.assertEqual(rr.layer_load([5, 0, 5, 5])["dead"], 1)
        self.assertIsNone(rr.layer_load([0, 0]))

    def test_uniform_layer_has_fair_share(self):
        st = rr.layer_load([10] * 16)
        self.assertAlmostEqual(st["busiest"], 1 / 16)
        self.assertAlmostEqual(st["cv"], 0.0)

    def test_report_tracks_the_worst_layer(self):
        skew = [100] * 16
        skew[3] = 1000
        even = [100] * 16
        text = rr.report([row(1), row(50, expert_load=[even, skew]), row(100, expert_load=[even, even])], "x")
        self.assertIn("| 50 | 40.0% | 0 |", text)
        self.assertIn("| 100 | 6.2% | 0 |", text)
        self.assertIn("fair share 6.25% with 16 experts", text)


class InputTests(unittest.TestCase):
    def test_non_finite_rows_are_reported(self):
        text = rr.report([row(1), row(2, loss=float("nan")), row(3, norm=float("inf"))], "x")
        self.assertIn("2 rows, first at step 2", text)

    def test_real_token_throughput_ignores_padding(self):
        text = rr.report([row(1, scored=500, positions=2000, ms=1000.0)], "x")
        self.assertIn("Real (scored) tokens per second: 500", text)
        self.assertIn("(25.0%)", text)

    def test_bad_json_line_stops(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "metrics.jsonl"
            p.write_text(json.dumps(row(1)) + "\nnot json\n", encoding="utf-8")
            with self.assertRaises(SystemExit):
                rr.read_rows(p)


if __name__ == "__main__":
    unittest.main()
