from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SECTION = Path(__file__).resolve().parents[1]
for path in (ROOT, SECTION):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from pilot_analysis import recovery_crossing


class RecoveryCrossingTests(unittest.TestCase):
    def test_uses_second_consecutive_point(self) -> None:
        points = [
            {"trajectories": 0, "future_joint": 0.5, "future_safety": 0.96, "current_safety_decrease": 0},
            {"trajectories": 2048, "future_joint": 0.61, "future_safety": 0.96, "current_safety_decrease": 0.01},
            {"trajectories": 4096, "future_joint": 0.62, "future_safety": 0.97, "current_safety_decrease": 0.02},
        ]
        self.assertEqual(recovery_crossing(points)["trajectories"], 4096)

    def test_nonconsecutive_does_not_cross(self) -> None:
        points = [
            {"trajectories": 0, "future_joint": 0.61, "future_safety": 0.96, "current_safety_decrease": 0},
            {"trajectories": 2048, "future_joint": 0.59, "future_safety": 0.96, "current_safety_decrease": 0},
            {"trajectories": 4096, "future_joint": 0.61, "future_safety": 0.96, "current_safety_decrease": 0},
        ]
        self.assertEqual(recovery_crossing(points)["status"], "not_recovered")


if __name__ == "__main__":
    unittest.main()

