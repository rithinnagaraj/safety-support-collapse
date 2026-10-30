"""GRPO training with the SU reward, cost ledger, and a trajectory budget.

    py -m ssc.train.grpo --config configs/recovery_fb.json --model <checkpoint> --seed 1 --out runs/rec/K0-s1-stage4/seed1

Used for every RL run: stage-0 safety training (fa_secret), the benign stages
(math, code, logic_kk; reward = correct), and recovery runs (fb_injection,
fa_secret_v2). The reward is binary SU from the task family's own grader.

Writes to --out:
  ledger.jsonl      one row per generation batch: trajectories, tokens, group types
  checkpoint-<N>/   saved every `save_steps` optimizer steps, for held-out evaluation
  final/            the model at the end of the run (input to the next stage)
  run_meta.json     resolved config, ledger totals, wall-clock time, steps

Training stops once the ledger has counted `budget_trajectories` rollouts.
Requires the [gpu] extras (torch, transformers, trl, datasets; vllm for use_vllm).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from ssc.envs import make_tasks
from ssc.train.ledger import CostLedger
from ssc.train.rewards import SUReward


def build_dataset(cfg: dict):
    from datasets import Dataset

    if cfg.get("data"):
        from ssc.envs.generate import load_tasks

        tasks = load_tasks(cfg["data"])
    else:
        tasks = make_tasks(cfg["family"], "train", cfg.get("num_tasks", 4000), cfg.get("data_seed", 0))
    return Dataset.from_list([{"prompt": t.messages(), "task_json": t.to_json()} for t in tasks])


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--model", help="overrides config 'model'")
    p.add_argument("--seed", type=int, help="overrides config 'seed'")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args(argv)

    cfg = json.loads(args.config.read_text())
    if args.model:
        cfg["model"] = args.model
    if args.seed is not None:
        cfg["seed"] = args.seed
    group_size = cfg["group_size"]
    budget = cfg["budget_trajectories"]

    from transformers import AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    args.out.mkdir(parents=True, exist_ok=True)
    ledger = CostLedger(group_size=group_size, path=args.out / "ledger.jsonl")
    if ledger.path.exists():
        raise FileExistsError(f"{ledger.path} already exists; use a fresh --out directory")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    reward = SUReward(ledger, level="train", tokenizer=tokenizer)

    class BudgetCallback(TrainerCallback):
        def on_step_begin(self, _args, state, control, **_):
            ledger.current_step = state.global_step

        def on_step_end(self, _args, state, control, **_):
            if ledger.trajectories >= budget:
                control.should_training_stop = True
                control.should_save = True

    trainer_args = GRPOConfig(
        output_dir=str(args.out),
        seed=cfg["seed"],
        num_generations=group_size,
        **cfg["grpo"],
    )
    trainer = GRPOTrainer(
        model=cfg["model"],
        reward_funcs=[reward],
        args=trainer_args,
        train_dataset=build_dataset(cfg),
        processing_class=tokenizer,
        callbacks=[BudgetCallback()],
    )
    t0 = time.time()
    trainer.train()
    wall = time.time() - t0
    trainer.save_model(str(args.out / "final"))
    tokenizer.save_pretrained(str(args.out / "final"))

    totals = ledger.totals()
    (args.out / "run_meta.json").write_text(json.dumps({
        "config": cfg,
        "ledger_totals": totals,
        "reward_cells": dict(reward.cells),
        "hit_budget": ledger.trajectories >= budget,
        "global_steps": trainer.state.global_step,
        "wall_seconds": wall,
    }, indent=2))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
