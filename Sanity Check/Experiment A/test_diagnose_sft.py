from __future__ import annotations

from pathlib import Path
import sys
import unittest

EXPERIMENT_DIR = Path(__file__).resolve().parent
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from diagnose_sft import analyze


def row(instance_id: str, rollout: int, *, termination: str, joint: int, turns: list[str]) -> dict:
    return {
        "instance_id": instance_id,
        "rollout": rollout,
        "family": "file_access",
        "horizon": "current",
        "template_id": "calibration-current-file_access-bank0",
        "termination": termination,
        "joint": joint,
        "utility": joint,
        "safety": 1,
        "generated_tokens": 10,
        "assistant_turns": turns,
    }


class DiagnoseSftTests(unittest.TestCase):
    def test_detects_new_malformed_and_preserves_raw_turn(self) -> None:
        valid = '{"tool":"finish","arguments":{"answer":"4"}}'
        initial = [row("a", 0, termination="finish", joint=1, turns=[valid])]
        trained = [row("a", 0, termination="malformed", joint=0, turns=["not json"])]
        report = analyze(initial, trained, [{"epoch": 1, "loss": 2.0}], max_examples=1)
        self.assertEqual(report["changes"]["joint"], -1.0)
        self.assertEqual(report["changes"]["malformed"], 1.0)
        self.assertEqual(report["after_sft_malformed_parser_categories"], {"invalid_json": 1})
        self.assertEqual(report["newly_malformed_examples"][0]["after_sft_turns"], ["not json"])
        self.assertEqual(report["termination_transitions"], {"finish -> malformed": 1})

    def test_rejects_missing_pair(self) -> None:
        valid = '{"tool":"finish","arguments":{"answer":"4"}}'
        initial = [row("a", 0, termination="finish", joint=1, turns=[valid])]
        trained = [row("a", 1, termination="finish", joint=1, turns=[valid])]
        with self.assertRaises(ValueError):
            analyze(initial, trained, [])


if __name__ == "__main__":
    unittest.main()
