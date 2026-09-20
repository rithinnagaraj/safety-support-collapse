"""Small resource ledger helper shared by training and evaluation jobs."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass
class ResourceMeter:
    policy: Any
    started: float | None = None
    elapsed: float = 0.0

    def _sync(self) -> None:
        torch = getattr(self.policy, "torch", None)
        if torch is not None and torch.cuda.is_available():
            torch.cuda.synchronize()

    def start(self) -> None:
        self.elapsed = 0.0
        self.resume()

    def resume(self) -> None:
        if self.started is not None:
            raise RuntimeError("Resource meter is already running")
        self._sync()
        self.started = perf_counter()

    def pause(self) -> None:
        if self.started is None:
            return
        self._sync()
        self.elapsed += perf_counter() - self.started
        self.started = None

    def stop(self, *, input_tokens: int, generated_tokens: int, category: str) -> dict[str, float | int | str]:
        self.pause()
        elapsed = self.elapsed
        device = str(getattr(self.policy, "device", "cpu"))
        accelerator_count = 1 if device.startswith("cuda") else 0
        return {
            "category": category,
            "input_tokens": input_tokens,
            "generated_tokens": generated_tokens,
            "wall_seconds": elapsed,
            "gpu_seconds": elapsed * accelerator_count,
        }
