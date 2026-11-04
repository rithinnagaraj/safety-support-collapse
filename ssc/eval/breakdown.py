"""Break down saved samples (coverage --save-samples) by task attributes.

    python -m ssc.eval.breakdown results/base/fa_samples.jsonl --by mode,pressure
    python -m ssc.eval.breakdown results/base/fb_samples.jsonl --by pressure,injection --examples 3

Prints, overall and per value of each --by meta field: sample count, the four
outcome cells (SU, useful-but-unsafe, safe-but-useless, neither), mean reply
length, the most common usefulness verdicts and leak kinds. --examples prints
that many random replies from each cell.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from ssc.types import CELLS


def load(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    for r in rows:
        if isinstance(r["task"], str):
            r["task"] = json.loads(r["task"])
    return rows


def describe(rows: list[dict]) -> str:
    n = len(rows)
    cells = Counter(r["cell"] for r in rows)
    verdicts = Counter(r["reasons"][0] for r in rows).most_common(3)
    leaks = Counter(":".join(x.split(":")[1:]) for r in rows for x in r["reasons"][1:]).most_common(3)
    rates = "  ".join(f"{c} {cells[c] / n:5.1%}" for c in CELLS)
    tokens = sum(r["n_tokens"] for r in rows) / n
    return f"n={n:6d}  {rates}  tokens {tokens:5.0f}\n      verdicts {verdicts}\n      leaks {leaks}"


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("samples")
    p.add_argument("--by", default="", help="comma-separated task meta fields")
    p.add_argument("--examples", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    rows = load(args.samples)
    print(f"ALL  {describe(rows)}")
    for field in filter(None, args.by.split(",")):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r["task"]["meta"].get(field))].append(r)
        print(f"\nby {field}:")
        for value in sorted(groups):
            print(f"  {value:<10} {describe(groups[value])}")

    rng = random.Random(args.seed)
    for cell in CELLS if args.examples else ():
        pool = [r for r in rows if r["cell"] == cell]
        for r in rng.sample(pool, min(args.examples, len(pool))):
            print(f"\n----- {cell} | {r['task']['task_id']} | {r['reasons']}\n{r['response'][:800]}")


if __name__ == "__main__":
    main()
