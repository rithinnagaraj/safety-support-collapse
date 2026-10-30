"""End-to-end with the mock backend: sample -> grade -> coverage -> audit -> recovery cost."""

import csv
import json
import tempfile
import unittest
from pathlib import Path

from ssc.audit.export import export
from ssc.audit.score import score
from ssc.envs import make_tasks
from ssc.eval import coverage as coverage_cli
from ssc.eval.backends import POLAR_FRAC_HI, POLAR_HI, POLAR_LO, MockBackend, su_rate_profile
from ssc.eval.coverage import evaluate
from ssc.eval.retest import reliability
from ssc.train.ledger import CostLedger
from ssc.train.target import load_points, time_to_target


class Coverage(unittest.TestCase):
    tasks = make_tasks("fb_injection", "heldout", 400)

    def test_deterministic_cells(self):
        res = evaluate(self.tasks[:50], MockBackend(lambda t: {"U_notS": 1.0}), n=8)
        s = res["summary"]
        self.assertEqual(s["pass1"], {"su": 0.0, "useful": 1.0, "safe": 0.0})
        self.assertEqual(s["C8"]["useful"], 1.0)
        self.assertEqual(s["group_signal"]["all_fail"], 1.0)

    def test_matches_analytic_values(self):
        res = evaluate(self.tasks, MockBackend(su_rate_profile("polarized"), seed=3), n=32)
        s = res["summary"]
        expected_c32 = POLAR_FRAC_HI * (1 - (1 - POLAR_HI) ** 32) + (1 - POLAR_FRAC_HI) * (1 - (1 - POLAR_LO) ** 32)
        expected_fail = POLAR_FRAC_HI * (1 - POLAR_HI) ** 8 + (1 - POLAR_FRAC_HI) * (1 - POLAR_LO) ** 8
        self.assertAlmostEqual(s["pass1"]["su"], 0.30, delta=0.02)
        self.assertAlmostEqual(s["C32"]["su"], expected_c32, delta=0.04)
        self.assertAlmostEqual(s["group_signal"]["all_fail"], expected_fail, delta=0.03)

    def test_uniform_vs_polarized(self):
        """Same pass@1, different coverage: what H1 measures."""
        u = evaluate(self.tasks, MockBackend(su_rate_profile("uniform"), seed=1), n=32)["summary"]
        p = evaluate(self.tasks, MockBackend(su_rate_profile("polarized"), seed=1), n=32)["summary"]
        self.assertAlmostEqual(u["pass1"]["su"], p["pass1"]["su"], delta=0.03)
        self.assertGreater(u["C32"]["su"] - p["C32"]["su"], 0.1)

    def test_retest_between_independent_draws(self):
        a = evaluate(self.tasks, MockBackend(su_rate_profile("polarized"), seed=1), n=32)
        b = evaluate(self.tasks, MockBackend(su_rate_profile("polarized"), seed=2), n=32)
        r = reliability(a, b)
        self.assertEqual(r["num_tasks"], 400)
        self.assertGreater(r["p_correlation"], 0.8)


class Audit(unittest.TestCase):
    def test_export_blind_and_score(self):
        tasks = make_tasks("fa_secret", "heldout", 60)
        probs = lambda t: {"SU": 0.4, "U_notS": 0.3, "notU_S": 0.2, "notU_notS": 0.1}
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            evaluate(tasks, MockBackend(probs, seed=0), n=4, samples_path=d / "samples.jsonl")
            info = export(d / "samples.jsonl", d / "audit", per_cell=20)
            self.assertEqual(info["exported"], 80)

            with open(d / "audit.csv", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertNotIn("cell", rows[0])  # blind: grader verdict not shown
            key = json.loads((d / "audit.key.json").read_text())

            # A labeler who agrees with the grader everywhere -> pass.
            for r in rows:
                k = key["items"][r["audit_id"]]
                r["human_useful"], r["human_safe"] = ("y" if k["useful"] else "n"), ("y" if k["safe"] else "n")
            self._write(d / "audit.csv", rows)
            res = score(str(d / "audit.csv"), str(d / "audit.key.json"))
            self.assertTrue(res["pass"])
            self.assertEqual(res["unsafe_detection"]["precision"], 1.0)

            # Flip safety on 10 grader-unsafe rows -> precision drops, run fails, disagreements listed.
            flipped = 0
            for r in rows:
                if not key["items"][r["audit_id"]]["safe"] and flipped < 10:
                    r["human_safe"] = "y"
                    flipped += 1
            self._write(d / "audit.csv", rows)
            res = score(str(d / "audit.csv"), str(d / "audit.key.json"))
            self.assertFalse(res["pass"])
            self.assertEqual(len(res["disagreements"]), 10)

    @staticmethod
    def _write(path, rows):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


class CLIAndRecoveryCost(unittest.TestCase):
    def test_coverage_cli_then_time_to_target(self):
        """A fake recovery run: the ledger counts cost, coverage evals score checkpoints, target finds the crossing."""
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            led = CostLedger(group_size=8, path=d / "ledger.jsonl")
            for step in range(30):
                led.current_step = step
                led.record(["t"] * 8, [0] * 8, [100] * 8)
            # Mock "checkpoints" whose held-out SU rate rises with training step.
            for step, profile in [(10, "polarized"), (20, "uniform"), (30, "uniform")]:
                coverage_cli.main(["--backend", "mock", "--mock-profile", profile, "--family", "fb_injection",
                                   "--num-tasks", "200", "--n", "8", "--meta", f"step={step}",
                                   "--out", str(d / "evals" / f"step{step}.json")])
            pts = load_points(str(d / "ledger.jsonl"), str(d / "evals" / "*.json"))
            self.assertEqual([p.trajectories for p in pts], [80, 160, 240])
            out = time_to_target(pts, tau=0.25)
            self.assertTrue(out.reached)
            self.assertEqual(out.step, 10)
            self.assertEqual(out.tokens, 8000)


if __name__ == "__main__":
    unittest.main()
