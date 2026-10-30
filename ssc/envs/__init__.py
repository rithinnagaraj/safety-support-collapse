"""Task-family registry.

Safety families (procedural; useful and safe graded separately):
  fa_secret     FA, v1 policy: stage-0 safety training family
  fa_secret_v2  FA, v2 policy update: secondary recovery family
  fb_injection  FB: primary recovery family (held out of stage 0)

Benign families (no safety constraint, so SU == correct):
  math          GSM8K + MATH, dataset-backed (S1 easy, S4 hard)
  code          MBPP, dataset-backed (S2); grading executes code
  logic_kk      knights and knaves, procedural (S3)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Callable, Sequence

from ssc.envs import code_tasks, fa_secret, fb_injection, logic_kk, math_tasks
from ssc.types import Grade, Task

SPLITS = ("train", "heldout", "test")


@dataclass(frozen=True)
class Family:
    name: str
    generate: Callable[..., Task]  # (split, index, seed) -> Task
    grade: Callable[[Task, str, str], Grade]  # (task, response, level) -> Grade
    synthetic_responses: Callable[[Task], dict[str, str]]  # reference reply per reachable cell
    procedural: bool = True  # False: tasks must be loaded from a prepared JSONL file
    safety: bool = True  # False: benign family, safe is always True


FAMILIES = {
    "fa_secret": Family("fa_secret", partial(fa_secret.generate, policy="v1"), fa_secret.grade, fa_secret.synthetic_responses),
    "fa_secret_v2": Family("fa_secret_v2", partial(fa_secret.generate, policy="v2"), fa_secret.grade, fa_secret.synthetic_responses),
    "fb_injection": Family("fb_injection", fb_injection.generate, fb_injection.grade, fb_injection.synthetic_responses),
    "math": Family("math", math_tasks.generate, math_tasks.grade, math_tasks.synthetic_responses, procedural=False, safety=False),
    "code": Family("code", code_tasks.generate, code_tasks.grade, code_tasks.synthetic_responses, procedural=False, safety=False),
    "logic_kk": Family("logic_kk", logic_kk.generate, logic_kk.grade, logic_kk.synthetic_responses, safety=False),
}


def get_family(name: str) -> Family:
    try:
        return FAMILIES[name]
    except KeyError:
        raise KeyError(f"unknown family {name!r}; choose from {sorted(FAMILIES)}") from None


def grade(task: Task, response: str, level: str = "eval") -> Grade:
    return get_family(task.family).grade(task, response, level)


def grade_many(tasks: Sequence[Task], responses: Sequence[str], level: str = "eval", workers: int = 16) -> list[Grade]:
    """Grade in parallel. Code grading spends its time in subprocesses, so threads help."""
    if workers <= 1 or len(responses) < 2:
        return [grade(t, r, level) for t, r in zip(tasks, responses)]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda tr: grade(tr[0], tr[1], level), zip(tasks, responses)))


def make_tasks(family: str, split: str, n: int, seed: int = 0) -> list[Task]:
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    fam = get_family(family)
    if not fam.procedural:
        raise ValueError(f"{family} is dataset-backed; prepare it with ssc.envs.prepare_benign and pass --data")
    return [fam.generate(split, i, seed) for i in range(n)]
