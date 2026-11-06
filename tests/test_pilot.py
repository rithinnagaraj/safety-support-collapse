"""Pilot driver (dry run) and pilot analysis (on synthetic pilot outputs)."""

import json
import os
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from ssc.eval.backends import MockBackend, su_rate_profile
from ssc.eval.suite import run_suite
from ssc.pilot.analyze import Analysis, to_markdown
from ssc.pilot.run import Pilot
from ssc.train.ledger import CostLedger

ANALYSIS = {"tau_margin": 0.05, "min_tau_gap": 0.15, "consecutive": 2, "max_benign_drop": 0.03,
            "equivalence_margin": 0.02, "min_coverage_drop": 0.05, "plateau_last_k": 3,
            "budget_multiple_of_median": 2.0, "power": 0.8, "alpha": 0.05}


def small_config(root: Path) -> dict:
    return {
        "model": "tiny-model", "runs_dir": str(root / "runs"), "results_dir": str(root / "results"),
        "group_size": 8, "keep_checkpoints": False, "grpo_defaults": {"learning_rate": 1e-6},
        "stage0": {"family": "fa_secret", "data": "data/fa_secret/train.jsonl", "seed": 0, "budget_trajectories": 1000},
        "benign_seeds": [1, 2, 3],
        "benign_stages": [
            {"name": "stage1", "family": "logic_kk", "data": "data/logic_kk/train.jsonl", "budget_trajectories": 1000},
            {"name": "stage2", "family": "logic_kk", "data": "data/logic_kk/train.jsonl", "budget_trajectories": 1000},
        ],
        "coverage_evals": [
            {"name": "fb", "family": "fb_injection", "data": None, "num_tasks": 200, "n": 8, "draws": 2},
            {"name": "fa", "family": "fa_secret", "data": None, "num_tasks": 100, "n": 8, "draws": 2},
            {"name": "logic", "family": "logic_kk", "data": None, "num_tasks": 50, "n": 8, "draws": 1},
        ],
        "benign_evals": ["logic"],
        "recovery": {"family": "fb_injection", "data": "data/fb_injection/train.jsonl", "budget_trajectories": 2560,
                     "seeds_from_stage0": [1, 2], "seeds_from_stage4": [1, 2], "grpo": {"save_steps": 5}},
        "ceiling": {"budget_multiplier": 3, "seed": 101},
        "recovery_eval": {"target": {"name": "fb", "family": "fb_injection", "num_tasks": 50, "n": 2},
                          "benign": [{"name": "logic", "family": "logic_kk", "num_tasks": 20, "n": 2}]},
        "analysis": ANALYSIS,
    }


class InTempDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self._cwd = os.getcwd()
        os.chdir(self.root)  # the driver checks data/ relative to the working directory

    def tearDown(self):
        os.chdir(self._cwd)
        self._tmp.cleanup()


class DryRun(InTempDir):
    def test_plans_every_run_once(self):
        cfg = small_config(self.root)
        pilot = Pilot(cfg, Path("pilot.json"), dry_run=True)
        for step in ("data", "stage0", "benign", "coverage", "recovery", "ceiling", "eval_runs", "analyze"):
            getattr(pilot, f"step_{step}")()
        modules = Counter(cmd[2] for cmd in pilot.commands)
        self.assertEqual(modules["ssc.envs.generate"], 4)
        self.assertEqual(modules["ssc.envs.prepare_benign"], 1)
        # stage0 + 3 seeds x 2 stages + recovery (2 from stage0 + 3 lineages x 2) + 1 ceiling (other needs coverage)
        self.assertEqual(modules["ssc.train.grpo"], 1 + 6 + 8 + 1)
        self.assertEqual(modules["ssc.eval.suite"], 7)  # one per checkpoint
        self.assertEqual(modules["ssc.pilot.analyze"], 1)
        self.assertFalse((self.root / "runs").exists(), "a dry run must not write anything")

    def test_stages_chain_from_the_previous_checkpoint(self):
        pilot = Pilot(small_config(self.root), Path("pilot.json"), dry_run=True)
        pilot.step_benign()
        self.assertEqual(len(pilot.commands), 6)

    def test_lineage_seeds_differ_per_stage(self):
        pilot = Pilot(small_config(self.root), Path("pilot.json"), dry_run=False)
        written = []
        pilot.sh = lambda args, why: written.append(json.loads(Path(args[2]).read_text()))
        pilot.step_benign()
        self.assertEqual(len({c["seed"] for c in written}), 6)
        s1_stage2 = next(c for c in written if c["seed"] == 101)
        self.assertTrue(s1_stage2["model"].replace("\\", "/").endswith("benign/s1/stage1/final"))


class InlineRecoveryEval(InTempDir):
    """recovery.eval_inline: each run is scored and pruned right after it trains."""

    def test_scores_and_prunes_each_run_before_the_next(self):
        cfg = small_config(self.root)
        cfg["recovery"]["eval_inline"] = True
        cfg["ceiling"]["grpo"] = {"save_steps": 15}
        pilot = Pilot(cfg, Path("pilot.json"), dry_run=False)
        order = []

        def fake_sh(args, why):
            if args[0] == "ssc.train.grpo":
                run_cfg = json.loads(Path(args[2]).read_text())
                out = Path(args[4])
                order.append(("train", out.name, run_cfg["grpo"]["save_steps"]))
                for step in (5, 10):
                    (out / f"checkpoint-{step}").mkdir(parents=True)
                (out / "final").mkdir()
                (out / "final" / "config.json").write_text("{}")
                (out / "final" / "model.safetensors").write_text("w")
                (out / "run_meta.json").write_text(json.dumps({"global_steps": 10}))
            else:
                spec = json.loads(Path(args[args.index("--spec") + 1]).read_text())
                order.append(("eval", Path(args[args.index("--model") + 1]).parent.name))
                for item in spec:
                    Path(item["out"]).parent.mkdir(parents=True, exist_ok=True)
                    Path(item["out"]).write_text("{}")

        pilot.sh = fake_sh
        pilot.step_recovery()
        pilot.step_ceiling()
        first = pilot.runs / "recovery" / "stage0_r1"
        self.assertEqual(order[:3], [("train", "stage0_r1", 5), ("eval", "stage0_r1"), ("eval", "stage0_r1")])
        self.assertIn(("train", "ceiling_stage0", 15), order)
        self.assertFalse(list(first.glob("checkpoint-*")))
        self.assertFalse((first / "final" / "model.safetensors").exists())
        self.assertTrue(pilot.trained(first))
        self.assertTrue((first / "evals" / "step10.json").exists())


class SyntheticPilot(InTempDir):
    """Fabricate a full pilot's outputs, with a planted effect, and check the analysis finds it.

    Planted: benign training polarizes per-prompt success (same mean, lower coverage),
    and recovery from stage-4 checkpoints is slower than from stage 0.
    """

    def build(self) -> dict:
        cfg = small_config(self.root)
        pilot = Pilot(cfg, Path("pilot.json"), dry_run=False)
        for ck in pilot.checkpoints():
            profile = "uniform" if ck == "stage0" else "polarized"
            run_suite(pilot.coverage_spec(ck), MockBackend(su_rate_profile(profile)), workers=1)

        def fake_run(name, start, slope, steps, plateau):
            run = pilot.runs / "recovery" / name
            led = CostLedger(group_size=8, path=run / "ledger.jsonl")
            for step in range(steps):
                led.current_step = step
                led.record([f"t{i // 8}" for i in range(64)], [0.0] * 60 + [1.0] * 4, [50] * 64)
            pilot._write_step0(start, run / "evals" / "step0.json")
            for step in range(5, steps + 1, 5):
                score = min(plateau, 0.3 + slope * step)
                (run / "evals" / f"step{step}.json").write_text(json.dumps(
                    {"summary": {"pass1": {"su": score}}, "meta": {"step": step, "benign": 0.65}, "wall_seconds": 30}))
            (run / "run_meta.json").write_text(json.dumps({"wall_seconds": 1800}))

        for name, (start, seed) in pilot.recovery_runs().items():
            slope = (0.03 if start == "stage0" else 0.015) + 0.002 * seed
            fake_run(name, start, slope, 40, 0.9)
        for name, start in pilot.ceiling_runs().items():
            fake_run(name, start, 0.03 if start == "stage0" else 0.015, 120, 0.9 if start == "stage0" else 0.8)
        for path in [pilot.stage0_path()] + [pilot.lineage_path(s, st["name"]) for s in cfg["benign_seeds"]
                                             for st in cfg["benign_stages"]]:
            path.mkdir(parents=True, exist_ok=True)
            (path / "run_meta.json").write_text(json.dumps({"wall_seconds": 3600}))
        return cfg

    def test_analysis_recovers_the_planted_effects(self):
        cfg = self.build()
        r = Analysis(cfg, Path("pilot.json")).run()

        h1 = r["h1_preview"]
        self.assertTrue(h1["coverage_dropped"])
        self.assertLess(abs(h1["mean_deltas"]["fb_su_pass1"]), 0.06)
        self.assertGreater(r["stage_table"][-1]["fb_all_fail"]["mean"], r["stage_table"][0]["fb_all_fail"]["mean"])

        self.assertEqual(len(r["tau"]["plateaus"]), 2)
        self.assertAlmostEqual(r["tau"]["tau"], 0.75)
        self.assertEqual(r["tau"]["excluded_checkpoints"], [])

        rec = r["recovery"]
        self.assertEqual(len(rec["runs"]), 8)
        self.assertTrue(all(x["reached"] for x in rec["runs"]))
        self.assertGreater(rec["cost_ratio_stage4_over_stage0"], 1.5)

        self.assertIn("log_cost_seed_sd", r["noise_and_power"])
        self.assertIn("recovery_runs_per_group", r["noise_and_power"])
        self.assertIn("fb_C_measurement_sd", r["noise_and_power"])
        bt = r["budget_and_time"]
        self.assertIsNotNone(bt["recommended_budget_trajectories"])
        self.assertGreater(bt["projected_phase1_gpu_hours"]["total"], 0)
        self.assertEqual(r["prereg_draft"]["tau"], 0.75)
        self.assertIn("H1 preview", to_markdown(r))

    def test_partial_results_do_not_crash(self):
        cfg = small_config(self.root)
        r = Analysis(cfg, Path("pilot.json")).run()
        self.assertIsNone(r["stage_table"])
        self.assertTrue(r["notes"])
        self.assertIn("Not yet available", to_markdown(r))


if __name__ == "__main__":
    unittest.main()
