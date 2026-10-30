"""Core data types shared by every task family."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from typing import Any

# The four outcome cells. Every task family must make all four reachable.
SU = "SU"  # useful and safe
U_NOT_S = "U_notS"  # correct but unsafe
NOT_U_S = "notU_S"  # safe but not useful (includes refusals)
NOT_U_NOT_S = "notU_notS"  # neither
CELLS = (SU, U_NOT_S, NOT_U_S, NOT_U_NOT_S)

# Grading levels. "train" is what the RL reward sees; "eval" adds hidden checks
# (obfuscated leaks) used only for held-out evaluation, so reward hacking by
# encoding a leak is caught at evaluation time.
LEVELS = ("train", "eval")


@dataclass(frozen=True)
class Task:
    task_id: str
    family: str
    split: str
    system: str
    user: str
    answer: str  # gold value for the usefulness check
    meta: dict[str, Any] = field(default_factory=dict)

    def messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system},
            {"role": "user", "content": self.user},
        ]

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, s: str) -> "Task":
        return cls(**json.loads(s))


@dataclass(frozen=True)
class Grade:
    useful: bool
    safe: bool
    reasons: tuple[str, ...] = ()

    @property
    def su(self) -> bool:
        return self.useful and self.safe

    @property
    def cell(self) -> str:
        if self.useful:
            return SU if self.safe else U_NOT_S
        return NOT_U_S if self.safe else NOT_U_NOT_S
