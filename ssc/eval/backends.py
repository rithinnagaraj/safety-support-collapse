"""Sampling backends: `generate(tasks, n) -> n completions per task`.

  vllm  production backend (Linux + GPU)
  hf    transformers fallback for CPU smoke tests with a tiny model
  mock  no model: samples the family's hand-built replies with known
        per-prompt probabilities, so the whole pipeline can be checked
        against analytic values
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

from ssc.envs import get_family
from ssc.types import CELLS, Task


@dataclass(frozen=True)
class Completion:
    text: str
    n_tokens: int
    finish_reason: str  # "stop" or "length" (truncated)


class Backend(Protocol):
    def generate(self, tasks: Sequence[Task], n: int, seed: int | None = None,
                 max_tokens: int | None = None) -> list[list[Completion]]: ...
    # seed / max_tokens override the constructor values for this call, so one
    # loaded model can serve several evaluations (e.g. two independent draws).


class VLLMBackend:
    def __init__(self, model: str, temperature: float = 1.0, top_p: float = 1.0, max_tokens: int = 1024,
                 seed: int = 0, gpu_memory_utilization: float = 0.85):
        from vllm import LLM, SamplingParams

        self.llm = LLM(model=model, seed=seed, dtype="bfloat16", gpu_memory_utilization=gpu_memory_utilization)
        self.params = dict(temperature=temperature, top_p=top_p, max_tokens=max_tokens, seed=seed)
        self._sp = SamplingParams

    def generate(self, tasks: Sequence[Task], n: int, seed: int | None = None,
                 max_tokens: int | None = None) -> list[list[Completion]]:
        params = {**self.params, **({"seed": seed} if seed is not None else {}),
                  **({"max_tokens": max_tokens} if max_tokens is not None else {})}
        outs = self.llm.chat([t.messages() for t in tasks], self._sp(n=n, **params), use_tqdm=True)
        return [[Completion(c.text, len(c.token_ids), c.finish_reason or "stop") for c in o.outputs] for o in outs]


class HFBackend:
    def __init__(self, model: str, temperature: float = 1.0, top_p: float = 1.0, max_tokens: int = 512, seed: int = 0):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        torch.manual_seed(seed)
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(model)
        self.kw = dict(do_sample=True, temperature=temperature, top_p=top_p, max_new_tokens=max_tokens)

    def generate(self, tasks: Sequence[Task], n: int, seed: int | None = None,
                 max_tokens: int | None = None) -> list[list[Completion]]:
        if seed is not None:
            self.torch.manual_seed(seed)
        kw = {**self.kw, **({"max_new_tokens": max_tokens} if max_tokens is not None else {})}
        results = []
        for t in tasks:
            ids = self.tok.apply_chat_template(t.messages(), add_generation_prompt=True, return_tensors="pt")
            with self.torch.no_grad():
                out = self.model.generate(ids, num_return_sequences=n, pad_token_id=self.tok.eos_token_id, **kw)
            row = []
            for seq in out[:, ids.shape[1]:]:
                toks = [int(x) for x in seq if int(x) != self.tok.pad_token_id]
                stopped = self.tok.eos_token_id in toks
                if stopped:
                    toks = toks[: toks.index(self.tok.eos_token_id)]
                row.append(Completion(self.tok.decode(toks, skip_special_tokens=True), len(toks),
                                      "stop" if stopped else "length"))
            results.append(row)
        return results


class MockBackend:
    """Draws each reply from one of the four hand-built outcome-cell replies.

    `cell_probs(task)` returns a probability per cell. With probability 1 on SU
    or U_notS etc. this is a fully controlled ground truth for the estimators.
    Benign families have no unsafe replies, so mass on an unsafe cell moves to
    its safe counterpart (U_notS -> SU, notU_notS -> notU_S).
    """

    FOLD = {"U_notS": "SU", "notU_notS": "notU_S"}

    def __init__(self, cell_probs: Callable[[Task], dict[str, float]], seed: int = 0):
        self.cell_probs = cell_probs
        self.rng = random.Random(seed)

    def generate(self, tasks: Sequence[Task], n: int, seed: int | None = None,
                 max_tokens: int | None = None) -> list[list[Completion]]:
        if seed is not None:
            self.rng = random.Random(seed)
        results = []
        for t in tasks:
            replies = get_family(t.family).synthetic_responses(t)
            probs = dict.fromkeys(CELLS, 0.0)
            for c, p in self.cell_probs(t).items():
                probs[c if c in replies else self.FOLD[c]] += p
            cells = self.rng.choices(CELLS, weights=[probs[c] for c in CELLS], k=n)
            results.append([Completion(replies[c], len(replies[c].split()), "stop") for c in cells])
        return results


def su_rate_profile(name: str) -> Callable[[Task], dict[str, float]]:
    """Mock profiles. "uniform" and "polarized" have the same mean SU rate (0.30)
    but very different coverage, which is the effect H1 looks for."""

    def split_rest(p_su: float) -> dict[str, float]:
        rest = 1.0 - p_su
        return {"SU": p_su, "U_notS": rest / 2, "notU_S": rest / 4, "notU_notS": rest / 4}

    if name == "uniform":
        return lambda t: split_rest(MOCK_MEAN)
    if name == "polarized":
        return lambda t: split_rest(POLAR_HI if int(t.task_id[-6:]) % 10 < 10 * POLAR_FRAC_HI else POLAR_LO)
    raise ValueError(f"unknown mock profile {name!r}")


MOCK_MEAN = 0.30
POLAR_HI, POLAR_FRAC_HI = 0.90, 0.3
POLAR_LO = (MOCK_MEAN - POLAR_FRAC_HI * POLAR_HI) / (1 - POLAR_FRAC_HI)  # = 0.043, so the mean stays 0.30


def make_backend(kind: str, model: str | None = None, mock_profile: str = "uniform", **kw) -> Backend:
    if kind == "mock":
        return MockBackend(su_rate_profile(mock_profile), seed=kw.get("seed", 0))
    if model is None:
        raise ValueError(f"--model is required for backend {kind!r}")
    if kind == "vllm":
        return VLLMBackend(model, **kw)
    if kind == "hf":
        return HFBackend(model, **{k: v for k, v in kw.items() if k != "gpu_memory_utilization"})
    raise ValueError(f"unknown backend {kind!r}")
