"""Recovery cost: when did a run reach the target, and at what cost?

    py -m ssc.train.target --ledger runs/rec/ledger.jsonl --evals "runs/rec/evals/*.json" --tau 0.60

Each eval file is a coverage.py output on the held-out split of the recovery
family, tagged with `--meta step=<optimizer step>` (and optionally
`--meta benign=<accuracy>`). The score is held-out SU pass@1.

Rules (pre-registered):
  * reached at the FIRST of `consecutive` (default 2) evaluations in a row
    with score >= tau, so one noisy evaluation cannot count
  * an evaluation only counts if benign accuracy has not dropped more than
    `max_benign_drop` below the starting value (guardrail)
  * runs that never reach the target are right-censored at the cost of
    the last evaluation, not dropped
"""

from __future__ import annotations

import argparse
import glob
import json
from dataclasses import asdict, dataclass
from typing import Sequence

from ssc.train.ledger import cost_at_step, load_rows


@dataclass(frozen=True)
class EvalPoint:
    step: int
    trajectories: int
    tokens: int
    score: float
    benign: float | None = None


@dataclass(frozen=True)
class RecoveryOutcome:
    reached: bool  # False -> right-censored
    step: int
    trajectories: int
    tokens: int
    auc: float  # mean held-out score over the run, trajectory-weighted (threshold-free)


def time_to_target(points: Sequence[EvalPoint], tau: float, consecutive: int = 2,
                   benign_start: float | None = None, max_benign_drop: float = 0.03) -> RecoveryOutcome:
    if not points:
        raise ValueError("no evaluation points")
    pts = sorted(points, key=lambda p: p.step)

    def ok(p: EvalPoint) -> bool:
        if p.score < tau:
            return False
        if benign_start is not None and p.benign is not None:
            return p.benign >= benign_start - max_benign_drop
        return True

    hit = None
    for i in range(len(pts) - consecutive + 1):
        if all(ok(p) for p in pts[i : i + consecutive]):
            hit = pts[i]
            break

    span = pts[-1].trajectories - pts[0].trajectories
    if span > 0:
        area = sum((b.trajectories - a.trajectories) * (a.score + b.score) / 2 for a, b in zip(pts, pts[1:]))
        auc = area / span
    else:
        auc = pts[0].score
    end = hit or pts[-1]
    return RecoveryOutcome(hit is not None, end.step, end.trajectories, end.tokens, auc)


def check_tau(tau: float, start_scores: Sequence[float], min_gap: float = 0.15) -> list[int]:
    """Indices of checkpoints for which tau is NOT at least `min_gap` above the starting score.

    The plan shrinks the comparison set rather than moving tau, so these are
    the checkpoints to exclude.
    """
    return [i for i, s in enumerate(start_scores) if tau < s + min_gap]


def load_points(ledger_path: str, eval_glob: str) -> list[EvalPoint]:
    rows = load_rows(ledger_path)
    points = []
    for path in sorted(glob.glob(eval_glob)):
        with open(path) as f:
            res = json.load(f)
        meta = res.get("meta", {})
        if "step" not in meta:
            raise ValueError(f"{path} has no step tag; run coverage.py with --meta step=<N>")
        step = int(meta["step"])
        cost = cost_at_step(rows, step)
        benign = float(meta["benign"]) if "benign" in meta else None
        points.append(EvalPoint(step, cost["trajectories"], cost["tokens"], res["summary"]["pass1"]["su"], benign))
    return points


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ledger", required=True)
    p.add_argument("--evals", required=True, help="glob of coverage.py outputs")
    p.add_argument("--tau", type=float, required=True)
    p.add_argument("--consecutive", type=int, default=2)
    p.add_argument("--benign-start", type=float)
    p.add_argument("--max-benign-drop", type=float, default=0.03)
    args = p.parse_args(argv)
    pts = load_points(args.ledger, args.evals)
    out = time_to_target(pts, args.tau, args.consecutive, args.benign_start, args.max_benign_drop)
    print(json.dumps({"outcome": asdict(out), "num_evals": len(pts)}, indent=2))


if __name__ == "__main__":
    main()
