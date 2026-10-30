"""Export a blind, stratified sample of graded replies for hand labeling.

    py -m ssc.audit.export --samples results/pilot/fb_samples.jsonl --per-cell 50 --out audit/fb_injection

Writes:
  <out>.csv       what the labeler sees: prompt, reply, and empty human_useful /
                  human_safe columns. The grader's verdict is NOT shown, so the
                  labels are not anchored on it.
  <out>.key.json  the grader's verdicts, plus each cell's population size and
                  sample size, which are used to reweight the stratified sample.

Sampling is stratified by the grader's outcome cell (default 50 per cell, so
200 in total), which keeps rare cells such as correct-but-unsafe in the audit.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

from ssc.types import CELLS, Task


def export(samples_path: Path, out: Path, per_cell: int = 50, seed: int = 0) -> dict:
    by_cell: dict[str, list[dict]] = defaultdict(list)
    with open(samples_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                by_cell[row["cell"]].append(row)

    rng = random.Random(seed)
    picked = []
    for cell in CELLS:
        pool = by_cell.get(cell, [])
        picked += rng.sample(pool, min(per_cell, len(pool)))
    rng.shuffle(picked)  # labelers must not be able to infer the cell from the order

    out.parent.mkdir(parents=True, exist_ok=True)
    key = {
        "population": {c: len(by_cell.get(c, [])) for c in CELLS},
        "sampled": {c: sum(r["cell"] == c for r in picked) for c in CELLS},
        "items": {},
    }
    with open(out.with_suffix(".csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "task_id", "system", "user", "gold_answer", "response",
                    "human_useful", "human_safe", "notes"])
        for i, row in enumerate(picked):
            aid = f"a{i:04d}"
            task = Task.from_json(row["task"])
            w.writerow([aid, task.task_id, task.system, task.user, task.answer, row["response"], "", "", ""])
            key["items"][aid] = {"useful": row["useful"], "safe": row["safe"], "cell": row["cell"],
                                 "reasons": row["reasons"]}
    out.with_suffix(".key.json").write_text(json.dumps(key, indent=2))
    return {"exported": len(picked), "population": key["population"], "sampled": key["sampled"]}


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", required=True, type=Path, help="JSONL written by coverage.py --save-samples")
    p.add_argument("--out", required=True, type=Path, help="output path stem (no extension)")
    p.add_argument("--per-cell", type=int, default=50)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)
    print(json.dumps(export(args.samples, args.out, args.per_cell, args.seed), indent=2))


if __name__ == "__main__":
    main()
