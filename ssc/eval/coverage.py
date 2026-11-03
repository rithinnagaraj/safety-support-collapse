"""Coverage evaluation: sample n replies per prompt, grade each, summarize.

    # primary measurement (A100 box)
    py -m ssc.eval.coverage --backend vllm --model runs/zoo/K0-s1/stage2 \
        --family fb_injection --data data/fb_injection/heldout.jsonl --n 32 --seed 1 \
        --out results/K0-s1-stage2/fb_draw1.json
    # second independent draw for test-retest reliability: same command with --seed 2

    # pipeline check without a model
    py -m ssc.eval.coverage --backend mock --mock-profile polarized --family fb_injection \
        --num-tasks 500 --n 32 --out results/mock_polarized.json

Output JSON: "summary" (pass@1 and C_n for SU / useful / safe, pass@k curve,
predicted group signal, p histogram, truncation, bootstrap CIs) and
"per_task" (counts per prompt). --save-samples also writes every graded reply
as JSONL, which is the input to the verifier audit.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Sequence

from ssc.envs import FAMILIES, grade_many, make_tasks
from ssc.eval.backends import Backend, make_backend
from ssc.eval.passk import bootstrap_ci, coverage, group_signal, p_histogram, passk_curve, powers_of_two_upto
from ssc.types import CELLS, Task


def evaluate(tasks: Sequence[Task], backend: Backend, n: int, level: str = "eval", group_size: int = 8,
             samples_path: Path | None = None, batch_size: int = 256, seed: int | None = None,
             max_tokens: int | None = None, workers: int = 16) -> dict:
    per_task = []
    if samples_path:
        Path(samples_path).parent.mkdir(parents=True, exist_ok=True)
    sample_f = open(samples_path, "w", encoding="utf-8") if samples_path else None
    try:
        for start in range(0, len(tasks), batch_size):
            batch = tasks[start : start + batch_size]
            # Offset the seed per batch so batches are not sampled with identical streams.
            batch_seed = None if seed is None else seed * 1_000_003 + start
            outs = backend.generate(batch, n, seed=batch_seed, max_tokens=max_tokens)
            for task, comps in zip(batch, outs):
                if len(comps) != n:
                    raise RuntimeError(f"backend returned {len(comps)} completions for {task.task_id}, expected {n}")
            flat_tasks = [t for t, comps in zip(batch, outs) for _ in comps]
            flat_grades = grade_many(flat_tasks, [c.text for comps in outs for c in comps], level, workers)
            for i, (task, comps) in enumerate(zip(batch, outs)):
                cells = Counter()
                for j, comp in enumerate(comps):
                    g = flat_grades[i * n + j]
                    cells[g.cell] += 1
                    if sample_f:
                        sample_f.write(json.dumps({
                            "task_id": task.task_id, "sample": j, "response": comp.text,
                            "n_tokens": comp.n_tokens, "finish_reason": comp.finish_reason,
                            "useful": g.useful, "safe": g.safe, "cell": g.cell, "reasons": list(g.reasons),
                            "task": task.to_json(),
                        }) + "\n")
                per_task.append({
                    "task_id": task.task_id,
                    "n": n,
                    "su": cells["SU"],
                    "useful": cells["SU"] + cells["U_notS"],
                    "safe": cells["SU"] + cells["notU_S"],
                    "cells": {c: cells[c] for c in CELLS},
                    "mean_tokens": fmean(c.n_tokens for c in comps),
                    "truncated": sum(c.finish_reason == "length" for c in comps),
                })
    finally:
        if sample_f:
            sample_f.close()
    return {"summary": summarize(per_task, n, group_size), "per_task": per_task}


def summarize(per_task: list[dict], n: int, group_size: int = 8) -> dict:
    def counts(key: str) -> list[tuple[int, int]]:
        return [(r["n"], r[key]) for r in per_task]

    su = counts("su")
    total = sum(r["n"] for r in per_task)
    return {
        "num_tasks": len(per_task),
        "n": n,
        "pass1": {k: coverage(counts(k), 1) for k in ("su", "useful", "safe")},
        f"C{n}": {k: coverage(counts(k), n) for k in ("su", "useful", "safe")},
        "ci95": {
            "pass1_su": bootstrap_ci(su, lambda xs: coverage(xs, 1)),
            f"C{n}_su": bootstrap_ci(su, lambda xs: coverage(xs, n)),
        },
        "passk_su": passk_curve(su, powers_of_two_upto(n)),
        "group_signal": group_signal(su, group_size) if n >= group_size else None,
        "group_size": group_size,
        "p_histogram_su": p_histogram(su),
        "cell_rates": {c: sum(r["cells"][c] for r in per_task) / total for c in CELLS},
        "mean_tokens": fmean(r["mean_tokens"] for r in per_task),
        "truncation_rate": sum(r["truncated"] for r in per_task) / total,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--backend", choices=["vllm", "hf", "mock"], required=True)
    p.add_argument("--model")
    p.add_argument("--mock-profile", default="uniform", choices=["uniform", "polarized"])
    p.add_argument("--family", required=True, choices=sorted(FAMILIES))
    p.add_argument("--data", type=Path, help="tasks JSONL; otherwise generated on the fly")
    p.add_argument("--split", default="heldout")
    p.add_argument("--num-tasks", type=int, default=500)
    p.add_argument("--data-seed", type=int, default=0)
    p.add_argument("--n", type=int, default=32)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--seed", type=int, default=1, help="sampling seed; use 1 and 2 for the two draws")
    p.add_argument("--level", default="eval", choices=["train", "eval"])
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--meta", action="append", default=[], help="key=value tags stored in the output, e.g. step=50")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--save-samples", type=Path)
    args = p.parse_args(argv)

    if args.data:
        from ssc.envs.generate import load_tasks

        tasks = load_tasks(args.data)[: args.num_tasks]
        if any(t.family != args.family for t in tasks):
            p.error(f"{args.data} contains tasks that are not from family {args.family}")
    else:
        tasks = make_tasks(args.family, args.split, args.num_tasks, args.data_seed)

    # Create output folders before loading the model, so a bad path fails in seconds, not after model load.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.save_samples:
        args.save_samples.parent.mkdir(parents=True, exist_ok=True)

    kw = dict(temperature=args.temperature, top_p=args.top_p, max_tokens=args.max_tokens, seed=args.seed)
    backend = make_backend(args.backend, args.model, args.mock_profile, **kw)
    result = evaluate(tasks, backend, args.n, args.level, args.group_size, args.save_samples)
    result["config"] = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    result["meta"] = dict(kv.split("=", 1) for kv in args.meta)

    args.out.write_text(json.dumps(result, indent=2))
    s = result["summary"]
    print(json.dumps({"pass1": s["pass1"], f"C{args.n}": s[f"C{args.n}"], "group_signal": s["group_signal"],
                      "truncation_rate": s["truncation_rate"]}, indent=2))


if __name__ == "__main__":
    main()
