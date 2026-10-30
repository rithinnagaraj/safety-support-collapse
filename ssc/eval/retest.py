"""Test-retest reliability between two independent coverage draws of one checkpoint.

    py -m ssc.eval.retest results/ckpt/fb_draw1.json results/ckpt/fb_draw2.json

Reports how often a prompt's "covered" status (>= 1 SU reply in n) agrees
across draws, Cohen's kappa for that status, and the correlation of per-prompt
SU rates. The correlation is the reliability used for the errors-in-variables
correction in H2.
"""

from __future__ import annotations

import argparse
import json
from statistics import correlation, fmean


def reliability(a: dict, b: dict) -> dict:
    ra = {r["task_id"]: r for r in a["per_task"]}
    rb = {r["task_id"]: r for r in b["per_task"]}
    ids = sorted(ra.keys() & rb.keys())
    if len(ids) < 2:
        raise ValueError("the two draws share fewer than two prompts")
    ca = [ra[i]["su"] > 0 for i in ids]
    cb = [rb[i]["su"] > 0 for i in ids]
    pa = [ra[i]["su"] / ra[i]["n"] for i in ids]
    pb = [rb[i]["su"] / rb[i]["n"] for i in ids]

    agree = fmean(x == y for x, y in zip(ca, cb))
    qa, qb = fmean(ca), fmean(cb)
    chance = qa * qb + (1 - qa) * (1 - qb)
    kappa = (agree - chance) / (1 - chance) if chance < 1 else 1.0
    try:
        r = correlation(pa, pb)
    except Exception:  # constant input (e.g. every prompt at p=0)
        r = float("nan")
    return {
        "num_tasks": len(ids),
        "coverage_draw_a": qa,
        "coverage_draw_b": qb,
        "covered_agreement": agree,
        "covered_kappa": kappa,
        "p_correlation": r,
    }


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("draw_a")
    p.add_argument("draw_b")
    args = p.parse_args(argv)
    with open(args.draw_a) as fa, open(args.draw_b) as fb:
        print(json.dumps(reliability(json.load(fa), json.load(fb)), indent=2))


if __name__ == "__main__":
    main()
