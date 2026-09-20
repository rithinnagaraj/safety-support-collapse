from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SECTION = Path(__file__).resolve().parents[1]
for path in (ROOT, SECTION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pilot_analysis import CheckpointMetrics, select_matched_pair


class PairSelectionTests(unittest.TestCase):
    def test_largest_gap_then_index_tiebreak(self) -> None:
        common = dict(current_s=0.97, benign_mean=0.51, future_u=0.60, future_s=0.96, future_j=0.58)
        checkpoints = [
            CheckpointMetrics("initial", 0, future_c16=0.40, **common),
            CheckpointMetrics("math", 1, future_c16=0.55, **common),
            CheckpointMetrics("logic", 2, future_c16=0.70, **common),
        ]
        result = select_matched_pair(checkpoints)
        self.assertEqual(result["low"], "initial")
        self.assertEqual(result["high"], "logic")

    def test_no_matched_pair(self) -> None:
        checkpoints = [
            CheckpointMetrics("initial", 0, 0.90, 0.50, 0.50, 0.90, 0.45, 0.40),
            CheckpointMetrics("math", 1, 0.80, 0.60, 0.60, 0.80, 0.50, 0.70),
        ]
        self.assertEqual(select_matched_pair(checkpoints)["status"], "no matched comparison")


if __name__ == "__main__":
    unittest.main()

