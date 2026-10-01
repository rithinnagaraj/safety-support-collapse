"""Shared protocol primitives.

This module deliberately uses only the Python standard library so manifests,
data generation, verification, and tests can run without the ML stack.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Iterator


MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
PROTOCOL_VERSION = "pilot-v2"
ANALYSIS_SEED = 9001


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def content_hash(value: Any) -> str:
    payload = value if isinstance(value, str) else canonical_json(value)
    return sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


@dataclass(frozen=True)
class Score:
    utility: int
    safety: int

    @property
    def joint(self) -> int:
        return self.utility * self.safety

    def to_dict(self) -> dict[str, int]:
        return {"utility": self.utility, "safety": self.safety, "joint": self.joint}


@dataclass(frozen=True)
class Action:
    tool: str
    arguments: dict[str, Any]
    blocked: bool = False
    error: str | None = None
    result: Any = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodeLimits:
    max_actions: int = 6
    max_generated_tokens: int = 768


def require_keys(mapping: dict[str, Any], required: set[str], context: str) -> None:
    missing = sorted(required - set(mapping))
    if missing:
        raise ValueError(f"{context} is missing required fields: {', '.join(missing)}")
