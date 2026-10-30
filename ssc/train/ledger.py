"""Cost and group-signal ledger for RL runs.

One row per generation batch (one reward call). Every rollout is counted,
including groups that carry no reward-relative signal and any rollouts
filtered later (pass `filtered=` when using dynamic sampling).

Groups are consecutive chunks of `group_size` completions, which is the same
layout GRPO uses to compute advantages. Each chunk must share one task_id; a
mismatch raises instead of silently miscounting.

Group types for binary rewards:
  all_fail     every reward 0  -> zero advantage
  all_success  every reward 1  -> zero advantage
  mixed        anything else   -> the only groups with reward-relative signal
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence


class GroupAlignmentError(ValueError):
    pass


@dataclass
class CostLedger:
    group_size: int = 8
    path: Path | None = None
    current_step: int = 0  # optimizer step, set by the trainer callback
    rows: list[dict] = field(default_factory=list)
    trajectories: int = 0
    tokens: int = 0

    def record(self, task_ids: Sequence[str], rewards: Sequence[float], token_counts: Sequence[int],
               filtered: int = 0) -> dict:
        g = self.group_size
        if not (len(task_ids) == len(rewards) == len(token_counts)):
            raise ValueError("task_ids, rewards and token_counts must have equal length")
        if len(rewards) % g:
            raise GroupAlignmentError(f"{len(rewards)} completions is not a multiple of group size {g}")

        kinds = {"all_fail": 0, "all_success": 0, "mixed": 0}
        for i in range(0, len(rewards), g):
            ids = set(task_ids[i : i + g])
            if len(ids) != 1:
                raise GroupAlignmentError(f"group at offset {i} mixes prompts {sorted(ids)}")
            chunk = rewards[i : i + g]
            if all(r == 0 for r in chunk):
                kinds["all_fail"] += 1
            elif all(r == 1 for r in chunk):
                kinds["all_success"] += 1
            else:
                kinds["mixed"] += 1

        n_traj = len(rewards) + filtered
        n_tok = int(sum(token_counts))
        self.trajectories += n_traj
        self.tokens += n_tok
        row = {
            "call": len(self.rows),
            "global_step": self.current_step,
            "trajectories": n_traj,
            "filtered": filtered,
            "tokens": n_tok,
            "groups": kinds,
            "mean_reward": sum(rewards) / len(rewards),
            "cum_trajectories": self.trajectories,
            "cum_tokens": self.tokens,
        }
        self.rows.append(row)
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        return row

    def totals(self) -> dict:
        groups = {"all_fail": 0, "all_success": 0, "mixed": 0}
        for row in self.rows:
            for k in groups:
                groups[k] += row["groups"][k]
        return {"trajectories": self.trajectories, "tokens": self.tokens, "groups": groups}


def load_rows(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def cost_at_step(rows: Sequence[dict], step: int) -> dict:
    """Cumulative cost consumed to produce the checkpoint saved after `step` optimizer steps.

    Generation for update s+1 happens while global_step == s, so a checkpoint
    at step t was trained on every batch generated with global_step < t.
    """
    used = [r for r in rows if r["global_step"] < step]
    return {"trajectories": sum(r["trajectories"] for r in used), "tokens": sum(r["tokens"] for r in used)}
