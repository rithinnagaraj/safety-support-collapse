"""M1 pilot driver: runs every pilot step in order, resumably.

    python -m ssc.pilot.run --config configs/pilot.json              # everything
    python -m ssc.pilot.run --config configs/pilot.json --dry-run    # print the plan, run nothing
    python -m ssc.pilot.run --config configs/pilot.json --steps coverage,analyze

Steps (each skips work whose outputs already exist):
  data       generate the procedural task families; download + convert math/code
  stage0     GRPO safety training on fa_secret -> the shared starting checkpoint
  benign     for each seed: stage1 math -> stage2 code -> stage3 logic -> stage4 hard math
  coverage   32-sample coverage of every checkpoint on every eval set (fb, fa: 2 draws)
  recovery   GRPO on fb_injection from stage0 and from every stage-4 checkpoint
  ceiling    3x-budget recovery from stage0 and from the most-collapsed stage-4
             checkpoint; their plateaus set the recovery target tau
  eval_runs  score every saved recovery/ceiling checkpoint on held-out fb (+ benign
             guardrail); deletes each checkpoint afterwards unless keep_checkpoints
  analyze    write results/pilot/report.md, report.json, prereg_draft.json

Every training/eval step is a separate process, so GPU memory is released between them.
"""

from __future__ import annotations

import argparse
import json
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

STEPS = ("data", "stage0", "benign", "coverage", "recovery", "ceiling", "eval_runs", "analyze")
PROCEDURAL = ("fa_secret", "fa_secret_v2", "fb_injection", "logic_kk")


class Pilot:
    def __init__(self, cfg: dict, config_path: Path, backend: str = "vllm", dry_run: bool = False,
                 python: str = sys.executable):
        self.cfg = cfg
        self.config_path = config_path
        self.backend = backend
        self.dry = dry_run
        self.py = python
        self.runs = Path(cfg["runs_dir"])
        self.res = Path(cfg["results_dir"])
        self.commands: list[list[str]] = []  # everything that ran (or would run, in a dry run)

    # ------------------------------------------------------------ helpers

    def sh(self, args: list[str], why: str) -> None:
        cmd = [self.py, "-m", *args]
        self.commands.append(cmd)
        print(f"[{why}] $ {shlex.join(cmd)}", flush=True)
        if not self.dry:
            subprocess.run(cmd, check=True)

    def write_json(self, path: Path, obj) -> None:
        if self.dry:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(obj, indent=2))

    @staticmethod
    def trained(out: Path) -> bool:
        return (out / "final" / "config.json").exists()

    def grpo(self, out: Path, model: str, family: str, data: str, seed: int, budget: int, grpo_overrides: dict,
             why: str) -> None:
        if self.trained(out):
            print(f"[{why}] done: {out}")
            return
        if (out / "ledger.jsonl").exists():
            raise RuntimeError(f"{out} holds an unfinished run; delete it to restart that run")
        run_cfg = {
            "model": model, "family": family, "data": data, "seed": seed,
            "group_size": self.cfg["group_size"], "budget_trajectories": budget,
            "grpo": {**self.cfg["grpo_defaults"], **grpo_overrides},
        }
        self.write_json(out / "run_config.json", run_cfg)
        self.sh(["ssc.train.grpo", "--config", str(out / "run_config.json"), "--out", str(out)], why)

    def suite(self, model: str, spec: list[dict], spec_path: Path, why: str) -> None:
        if all(Path(it["out"]).exists() for it in spec):
            print(f"[{why}] done: {spec_path.parent}")
            return
        self.write_json(spec_path, spec)
        self.sh(["ssc.eval.suite", "--backend", self.backend, "--model", model, "--spec", str(spec_path)], why)

    # ------------------------------------------------------------ checkpoints

    def stage0_path(self) -> Path:
        return self.runs / "stage0"

    def lineage_path(self, seed: int, stage: str) -> Path:
        return self.runs / "benign" / f"s{seed}" / stage

    def checkpoints(self) -> dict[str, Path]:
        """Every model the pilot evaluates: stage0 plus each seed's four benign stages."""
        out = {"stage0": self.stage0_path() / "final"}
        for seed in self.cfg["benign_seeds"]:
            for st in self.cfg["benign_stages"]:
                out[f"s{seed}_{st['name']}"] = self.lineage_path(seed, st["name"]) / "final"
        return out

    def last_stage(self) -> str:
        return self.cfg["benign_stages"][-1]["name"]

    def recovery_runs(self) -> dict[str, tuple[str, int]]:
        """run name -> (start checkpoint name, recovery seed)."""
        rec = self.cfg["recovery"]
        runs = {f"stage0_r{s}": ("stage0", s) for s in rec["seeds_from_stage0"]}
        for seed in self.cfg["benign_seeds"]:
            start = f"s{seed}_{self.last_stage()}"
            runs.update({f"{start}_r{s}": (start, s) for s in rec["seeds_from_stage4"]})
        return runs

    def coverage_out(self, ckpt: str, name: str, draw: int) -> Path:
        return self.res / "coverage" / ckpt / f"{name}_d{draw}.json"

    def most_collapsed(self) -> str | None:
        """Stage-4 checkpoint with the lowest fb SU coverage (mean over draws), once coverage exists."""
        fb = next(e for e in self.cfg["coverage_evals"] if e["name"] == "fb")
        scores = {}
        for seed in self.cfg["benign_seeds"]:
            ck = f"s{seed}_{self.last_stage()}"
            files = [self.coverage_out(ck, "fb", d) for d in range(1, fb["draws"] + 1)]
            if not all(f.exists() for f in files):
                return None
            vals = [json.loads(f.read_text())["summary"][f"C{fb['n']}"]["su"] for f in files]
            scores[ck] = sum(vals) / len(vals)
        return min(scores, key=scores.get)

    def ceiling_runs(self) -> dict[str, str]:
        runs = {"ceiling_stage0": "stage0"}
        mc = self.most_collapsed()
        if mc:
            runs[f"ceiling_{mc}"] = mc
        return runs

    # ------------------------------------------------------------ steps

    def step_data(self) -> None:
        for fam in PROCEDURAL:
            if Path("data", fam, "manifest.json").exists():
                print(f"[data] done: data/{fam}")
                continue
            self.sh(["ssc.envs.generate", "--family", fam, "--out", f"data/{fam}"], "data")
        if all(Path("data", d, "manifest.json").exists() for d in ("math_easy", "math_hard", "code")):
            print("[data] done: data/math_easy, data/math_hard, data/code")
        else:
            self.sh(["ssc.envs.prepare_benign", "--out", "data"], "data")

    def step_stage0(self) -> None:
        s0 = self.cfg["stage0"]
        self.grpo(self.stage0_path(), self.cfg["model"], s0["family"], s0["data"], s0["seed"],
                  s0["budget_trajectories"], s0.get("grpo", {}), "stage0")

    def step_benign(self) -> None:
        for seed in self.cfg["benign_seeds"]:
            prev = self.stage0_path() / "final"
            for i, st in enumerate(self.cfg["benign_stages"]):
                out = self.lineage_path(seed, st["name"])
                self.grpo(out, str(prev), st["family"], st["data"], 100 * seed + i, st["budget_trajectories"],
                          st.get("grpo", {}), f"benign s{seed} {st['name']}")
                prev = out / "final"

    def coverage_spec(self, ck: str) -> list[dict]:
        spec = []
        for ev in self.cfg["coverage_evals"]:
            for draw in range(1, ev["draws"] + 1):
                spec.append({"name": f"{ev['name']}_d{draw}", "family": ev["family"], "data": ev.get("data"),
                             "num_tasks": ev["num_tasks"], "n": ev["n"], "seed": draw,
                             "max_tokens": ev.get("max_tokens"), "out": str(self.coverage_out(ck, ev["name"], draw)),
                             "meta": {"checkpoint": ck, "draw": draw}})
        return spec

    def step_coverage(self) -> None:
        for ck, path in self.checkpoints().items():
            self.suite(str(path), self.coverage_spec(ck), self.res / "coverage" / ck / "spec.json", f"coverage {ck}")

    def _recovery(self, name: str, start: str, seed: int, budget: int, why: str) -> None:
        rec = self.cfg["recovery"]
        self.grpo(self.runs / "recovery" / name, str(self.checkpoints()[start]), rec["family"], rec["data"],
                  seed, budget, rec.get("grpo", {}), why)

    def step_recovery(self) -> None:
        for name, (start, seed) in self.recovery_runs().items():
            self._recovery(name, start, seed, self.cfg["recovery"]["budget_trajectories"], f"recovery {name}")

    def step_ceiling(self) -> None:
        runs = self.ceiling_runs()
        if len(runs) < 2:
            print("[ceiling] most-collapsed stage-4 checkpoint unknown until the coverage step has run")
        budget = self.cfg["recovery"]["budget_trajectories"] * self.cfg["ceiling"]["budget_multiplier"]
        for name, start in runs.items():
            self._recovery(name, start, self.cfg["ceiling"]["seed"], budget, f"ceiling {name}")

    def run_starts(self) -> dict[str, str]:
        starts = {name: start for name, (start, _) in self.recovery_runs().items()}
        starts.update(self.ceiling_runs())
        return starts

    def recovery_eval_spec(self, run_name: str, step: int) -> list[dict]:
        re_cfg = self.cfg["recovery_eval"]
        evals = self.runs / "recovery" / run_name / "evals"
        spec = [{**b, "seed": 1, "out": str(evals / "benign" / f"step{step}_{b['name']}.json"), "meta": {"step": step}}
                for b in re_cfg["benign"]]
        spec.append({**re_cfg["target"], "seed": 1, "out": str(evals / f"step{step}.json"), "meta": {"step": step},
                     "attach_benign": [b["name"] for b in re_cfg["benign"]]})
        return spec

    def step_eval_runs(self) -> None:
        for name, start in self.run_starts().items():
            run = self.runs / "recovery" / name
            evals = run / "evals"
            self._write_step0(start, evals / "step0.json")
            ckpt_dirs = sorted(run.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
            if not ckpt_dirs and not self.dry:
                print(f"[eval {name}] no checkpoints found (has the run finished?)")
            for ck in ckpt_dirs:
                step = int(ck.name.split("-")[1])
                self.suite(str(ck), self.recovery_eval_spec(name, step), evals / "specs" / f"step{step}.json",
                           f"eval {name} step {step}")
                if not self.cfg.get("keep_checkpoints", True) and not self.dry and (evals / f"step{step}.json").exists():
                    shutil.rmtree(ck)

    def _write_step0(self, start: str, out: Path) -> None:
        """The start checkpoint's own scores, from the coverage step, as the run's step-0 point."""
        if out.exists() or self.dry:
            return
        fb = self.coverage_out(start, "fb", 1)
        benign = [self.coverage_out(start, b, 1) for b in self.cfg["benign_evals"]]
        if not fb.exists() or not all(b.exists() for b in benign):
            print(f"[eval] coverage for {start} missing; step-0 point not written")
            return
        b_scores = [json.loads(b.read_text())["summary"]["pass1"]["su"] for b in benign]
        self.write_json(out, {"summary": {"pass1": json.loads(fb.read_text())["summary"]["pass1"]},
                              "meta": {"step": 0, "benign": sum(b_scores) / len(b_scores), "source": str(fb)}})

    def step_analyze(self) -> None:
        self.sh(["ssc.pilot.analyze", "--config", str(self.config_path)], "analyze")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", type=Path, default=Path("configs/pilot.json"))
    p.add_argument("--steps", default="all", help=f"comma-separated subset of {','.join(STEPS)}")
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf", "mock"])
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    steps = STEPS if args.steps == "all" else tuple(s.strip() for s in args.steps.split(","))
    unknown = set(steps) - set(STEPS)
    if unknown:
        p.error(f"unknown steps: {sorted(unknown)}")
    pilot = Pilot(json.loads(args.config.read_text()), args.config, args.backend, args.dry_run)
    for step in STEPS:  # always in pipeline order
        if step in steps:
            print(f"\n=== {step} ===", flush=True)
            getattr(pilot, f"step_{step}")()


if __name__ == "__main__":
    main()
