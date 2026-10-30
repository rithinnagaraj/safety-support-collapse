"""Coverage metrics from per-prompt success counts.

For prompt i with c_i successes out of n_i samples:

  pass@k_i      = 1 - C(n-c, k) / C(n, k)       unbiased for 1 - (1-p_i)^k  (Chen et al. 2021)
  C_k           = mean_i pass@k_i                with k = n = 32 this is exactly
                                                 "fraction of prompts with >= 1 success in 32 draws"
  all-fail_i    = C(n-c, G) / C(n, G)            unbiased for (1-p_i)^G
  all-success_i = C(c, G) / C(n, G)              unbiased for p_i^G

The last two are the predicted fractions of GRPO groups (size G) that carry no
reward-relative signal. Plug-in estimates (c/n)^G are biased, so we use the
hypergeometric forms, which are unbiased for any n >= G.
"""

from __future__ import annotations

import random
from math import comb
from statistics import fmean
from typing import Callable, Sequence


def pass_at_k(n: int, c: int, k: int) -> float:
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n (n={n}, c={c}, k={k})")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def all_fail_prob(n: int, c: int, g: int) -> float:
    return comb(n - c, g) / comb(n, g)


def all_success_prob(n: int, c: int, g: int) -> float:
    return comb(c, g) / comb(n, g)


def coverage(counts: Sequence[tuple[int, int]], k: int) -> float:
    return fmean(pass_at_k(n, c, k) for n, c in counts)


def passk_curve(counts: Sequence[tuple[int, int]], ks: Sequence[int]) -> dict[int, float]:
    return {k: coverage(counts, k) for k in ks}


def group_signal(counts: Sequence[tuple[int, int]], g: int = 8) -> dict[str, float]:
    """Predicted fraction of GRPO groups of size g that are all-fail / all-success / mixed."""
    f = fmean(all_fail_prob(n, c, g) for n, c in counts)
    s = fmean(all_success_prob(n, c, g) for n, c in counts)
    return {"all_fail": f, "all_success": s, "mixed": 1.0 - f - s}


def p_histogram(counts: Sequence[tuple[int, int]], bins: int = 10) -> dict[str, int]:
    """Histogram of per-prompt success rates, with exact 0 and exact 1 counted separately."""
    labels = [f"[{i / bins:.1f},{(i + 1) / bins:.1f})" for i in range(bins)]
    hist = {"p=0": 0, **dict.fromkeys(labels, 0), "p=1": 0}
    for n, c in counts:
        if c == 0:
            hist["p=0"] += 1
        elif c == n:
            hist["p=1"] += 1
        else:
            hist[labels[min(int(c / n * bins), bins - 1)]] += 1
    return hist


def bootstrap_ci(
    items: Sequence, stat: Callable[[Sequence], float], b: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap over prompts."""
    rng = random.Random(seed)
    m = len(items)
    vals = sorted(stat([items[rng.randrange(m)] for _ in range(m)]) for _ in range(b))
    return vals[int(alpha / 2 * b)], vals[int((1 - alpha / 2) * b) - 1]


def powers_of_two_upto(n: int) -> list[int]:
    ks, k = [], 1
    while k <= n:
        ks.append(k)
        k *= 2
    if ks[-1] != n:
        ks.append(n)
    return ks
