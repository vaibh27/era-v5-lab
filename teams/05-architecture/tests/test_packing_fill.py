"""packing_fill.py: positions per pass under today's loader, length-bucketing and packing."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np

TEAM = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TEAM / "harness"))
import packing_fill as pf


class CountTests(unittest.TestCase):
    def test_current_matches_the_loader_windows(self):
        # 10 tokens at L=4: windows 4, 4, 2 -> 3 rows, 12 positions; 3 tokens -> 1 row.
        lengths = np.array([10, 3])
        self.assertEqual(pf.current(lengths, 4), (16, 4))
        self.assertEqual(sorted(pf.windows(lengths, 4).tolist()), [2, 3, 4, 4])

    def test_packed_fills_rows_with_eos_between_documents(self):
        # 10 + 3 tokens plus 2 EOS = 15 -> 4 rows of 4.
        self.assertEqual(pf.packed(np.array([10, 3]), 4), (16, 4))
        self.assertEqual(pf.packed(np.array([3, 3, 3, 3]), 8), (16, 2))

    def test_bucketed_pads_to_the_longest_in_each_batch(self):
        # Equal windows bucket perfectly: no padding at all.
        lengths = np.full(64, 5)
        self.assertEqual(pf.bucketed(lengths, 8, rows_per_batch=4)[0], 64 * 5)
        # Never more positions than today, never fewer than the content.
        rng = np.random.default_rng(0)
        lengths = rng.integers(1, 5000, size=500)
        b, _ = pf.bucketed(lengths, 2048, rows_per_batch=8)
        self.assertLessEqual(b, pf.current(lengths, 2048)[0])
        self.assertGreaterEqual(b, int(lengths.sum()))

    def test_bucketed_short_last_batch_is_not_padded_to_a_full_batch(self):
        # 5 equal windows, batches of 4: one full batch and one batch of 1 -> 5 x 7 positions, not 8 x 7.
        self.assertEqual(pf.bucketed(np.full(5, 7), 8, rows_per_batch=4)[0], 35)

    def test_packed_counts_eos_and_pads_only_the_last_row(self):
        # 106,665 documents and 113,426,354 content tokens (the note's two shards) -> 55,437 rows of 2,048.
        lengths = np.full(106_665, 113_426_354 // 106_665)
        lengths[: 113_426_354 - int(lengths.sum())] += 1
        self.assertEqual(int(lengths.sum()), 113_426_354)
        self.assertEqual(pf.packed(lengths, 2048), (113_534_976, 55_437))
        self.assertEqual(pf.packed(lengths, 4096), (113_537_024, 27_719))

    def test_index_reading_removes_the_eos(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "s.idx.npy"
            np.save(p, np.array([0, 11, 15], dtype=np.int64))
            self.assertEqual(pf.document_lengths([p]).tolist(), [10, 3])

    def test_bad_index_stops(self):
        with TemporaryDirectory() as d:
            p = Path(d) / "s.idx.npy"
            np.save(p, np.array([0, 5, 3], dtype=np.int64))
            with self.assertRaises(SystemExit):
                pf.document_lengths([p])

    def test_report_lists_each_option(self):
        text = pf.report(np.array([10, 3, 700, 2500]), [2048], 16384)
        for name in ("current", "bucketed", "packed"):
            self.assertIn(f"| 2,048 | {name} |", text)


if __name__ == "__main__":
    unittest.main()
