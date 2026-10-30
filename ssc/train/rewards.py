"""Binary safe-and-useful reward in the call signature TRL's GRPOTrainer expects.

TRL calls `reward(prompts=..., completions=..., <dataset columns>...)` once per
generation batch and returns one float per completion. Every call is also
recorded in the CostLedger (trajectories, tokens, group types).

The reward uses the train-level grader. Held-out evaluation uses the eval
level, whose hidden checks the policy never receives reward from.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Sequence

from ssc.envs import grade_many
from ssc.train.ledger import CostLedger
from ssc.types import Task


def completion_text(completion: Any) -> str:
    """TRL passes plain strings for standard prompts and message lists for chat prompts."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        return "".join(m.get("content", "") for m in completion if m.get("role", "assistant") == "assistant")
    raise TypeError(f"unrecognized completion type {type(completion).__name__}")


class SUReward:
    def __init__(self, ledger: CostLedger, level: str = "train", tokenizer: Any = None, workers: int = 16):
        self.__name__ = "su_reward"  # TRL names the logged metric after this
        self.ledger = ledger
        self.level = level
        self.workers = workers
        self.tokenizer = tokenizer
        self.cells: Counter[str] = Counter()
        self._task_cache: dict[str, Task] = {}

    def _task(self, task_json: str) -> Task:
        if task_json not in self._task_cache:
            self._task_cache[task_json] = Task.from_json(task_json)
        return self._task_cache[task_json]

    def _token_counts(self, texts: Sequence[str], completion_ids: Sequence[Sequence[int]] | None) -> list[int]:
        if completion_ids is not None:
            return [len(ids) for ids in completion_ids]
        if self.tokenizer is not None:
            return [len(self.tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts]
        raise ValueError("need completion_ids from the trainer or a tokenizer to count tokens")

    def __call__(self, prompts: Sequence[Any], completions: Sequence[Any], task_json: Sequence[str],
                 completion_ids: Sequence[Sequence[int]] | None = None, **_: Any) -> list[float]:
        texts = [completion_text(c) for c in completions]
        tasks = [self._task(s) for s in task_json]
        grades = grade_many(tasks, texts, self.level, self.workers)
        rewards = [1.0 if g.su else 0.0 for g in grades]
        self.cells.update(g.cell for g in grades)
        self.ledger.record([t.task_id for t in tasks], rewards, self._token_counts(texts, completion_ids))
        return rewards
