from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
SECTION = Path(__file__).resolve().parents[1]
EXPERIMENT = Path(__file__).resolve().parent
for path in (ROOT, SECTION, EXPERIMENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from learning_check import select_development_checkpoint, validate_training_diversity
from pilot_analysis import experiment_a_decision
from safety_tasks import generate_safety_split


def _summary(joint: float, invalid: float) -> dict:
    return {"metrics": {"joint": joint, "invalid_rate": invalid}}


class LearningCheckTests(unittest.TestCase):
    def test_training_requires_64_balanced_latent_tasks(self) -> None:
        rows = generate_safety_split(
            split="training", prompt_count=128, horizon="mixed", seed=80, template_namespace="calibration"
        )
        report = validate_training_diversity(rows, expected_prompts=128, expected_latents=64)
        self.assertEqual(report["unique_latent_tasks"], 64)
        self.assertEqual(set(report["latent_tasks_by_family_and_horizon"].values()), {16})

    def test_development_selection_enforces_malformed_guard(self) -> None:
        report = select_development_checkpoint(
            {
                0: _summary(0.30, 0.20),
                1: _summary(0.34, 0.21),
                2: _summary(0.40, 0.27),
                3: _summary(0.36, 0.22),
            },
            maximum_malformed_rate_increase=0.05,
            minimum_joint_improvement=0.0,
        )
        self.assertEqual(report["selected_epoch"], 3)
        self.assertEqual([row["epoch"] for row in report["rejected_candidates"]], [2])

    def test_no_development_improvement_selects_nothing(self) -> None:
        report = select_development_checkpoint(
            {0: _summary(0.30, 0.20), 1: _summary(0.30, 0.19), 2: _summary(0.29, 0.18)},
            maximum_malformed_rate_increase=0.05,
            minimum_joint_improvement=0.0,
        )
        self.assertIsNone(report["selected_epoch"])

    def test_final_gate_requires_joint_gain_and_malformed_noninferiority(self) -> None:
        settings = {
            "initial_utility_range": [0.30, 0.80],
            "initial_joint_range": [0.20, 0.70],
            "initial_c32_range": [0.40, 0.95],
            "repeat_c32_interval_max_width": 0.10,
            "sft_min_joint_improvement": 0.05,
            "maximum_malformed_rate_increase": 0.05,
        }
        common = dict(utility=0.4, joint=0.3, c32=0.6)
        decision = experiment_a_decision(
            oracle_pass=True,
            unit_pass=True,
            manual_review_pass=True,
            calibration_smoke_pass=True,
            first_metrics=common,
            second_metrics=common,
            repeated_interval={"lower": -0.01, "upper": 0.01, "width": 0.02},
            sft_interval={"estimate": 0.08, "lower": 0.04, "upper": 0.12},
            malformed_interval={"estimate": 0.01, "lower": -0.01, "upper": 0.04},
            development_checkpoint_selected=True,
            gate_settings=settings,
        )
        self.assertTrue(decision["pass"])
        decision["checks"]["held_out_malformed_noninferiority"] = False
        failing = experiment_a_decision(
            oracle_pass=True,
            unit_pass=True,
            manual_review_pass=True,
            calibration_smoke_pass=True,
            first_metrics=common,
            second_metrics=common,
            repeated_interval={"lower": -0.01, "upper": 0.01, "width": 0.02},
            sft_interval={"estimate": 0.08, "lower": 0.04, "upper": 0.12},
            malformed_interval={"estimate": 0.03, "lower": 0.0, "upper": 0.06},
            development_checkpoint_selected=True,
            gate_settings=settings,
        )
        self.assertFalse(failing["checks"]["learning_check"])


if __name__ == "__main__":
    unittest.main()
