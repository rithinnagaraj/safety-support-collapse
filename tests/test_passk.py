"""Coverage-estimator checks (plan section 13, "Coverage estimator")."""

import random
import unittest
from itertools import combinations
from statistics import fmean

from ssc.eval.passk import (
    all_fail_prob,
    all_success_prob,
    bootstrap_ci,
    coverage,
    group_signal,
    p_histogram,
    pass_at_k,
    powers_of_two_upto,
)


def brute_pass_at_k(n, c, k):
    outcomes = [1] * c + [0] * (n - c)
    subsets = list(combinations(range(n), k))
    return sum(any(outcomes[i] for i in s) for s in subsets) / len(subsets)


class ExactValues(unittest.TestCase):
    def test_matches_brute_force_enumeration(self):
        for n in range(1, 9):
            for c in range(n + 1):
                for k in range(1, n + 1):
                    self.assertAlmostEqual(pass_at_k(n, c, k), brute_pass_at_k(n, c, k), places=12)

    def test_group_probabilities_match_enumeration(self):
        n, g = 9, 4
        for c in range(n + 1):
            outcomes = [1] * c + [0] * (n - c)
            subsets = list(combinations(range(n), g))
            self.assertAlmostEqual(all_fail_prob(n, c, g), fmean(not any(outcomes[i] for i in s) for s in subsets))
            self.assertAlmostEqual(all_success_prob(n, c, g), fmean(all(outcomes[i] for i in s) for s in subsets))

    def test_c32_from_32_samples_is_the_covered_indicator(self):
        for c in range(33):
            self.assertEqual(pass_at_k(32, c, 32), 1.0 if c else 0.0)

    def test_pass_at_1_is_the_success_rate(self):
        self.assertAlmostEqual(pass_at_k(32, 8, 1), 0.25)

    def test_rejects_k_larger_than_n(self):
        with self.assertRaises(ValueError):
            pass_at_k(32, 3, 64)

    def test_powers_of_two(self):
        self.assertEqual(powers_of_two_upto(32), [1, 2, 4, 8, 16, 32])
        self.assertEqual(powers_of_two_upto(24), [1, 2, 4, 8, 16, 24])


class Unbiasedness(unittest.TestCase):
    """Synthetic Bernoulli prompts with known p: estimates must match the analytic values."""

    def simulate(self, p, n, reps=20000, seed=0):
        rng = random.Random(seed)
        return [sum(rng.random() < p for _ in range(n)) for _ in range(reps)]

    def test_pass_at_k_unbiased(self):
        for p in (0.02, 0.1, 0.4):
            cs = self.simulate(p, 32, reps=8000)
            for k in (1, 8, 32):
                est = fmean(pass_at_k(32, c, k) for c in cs)
                self.assertAlmostEqual(est, 1 - (1 - p) ** k, delta=0.015, msg=f"p={p} k={k}")

    def test_group_estimators_unbiased(self):
        for p in (0.1, 0.5, 0.9):
            cs = self.simulate(p, 32, reps=8000, seed=1)
            self.assertAlmostEqual(fmean(all_fail_prob(32, c, 8) for c in cs), (1 - p) ** 8, delta=0.01)
            self.assertAlmostEqual(fmean(all_success_prob(32, c, 8) for c in cs), p ** 8, delta=0.01)

    def test_same_pass1_different_coverage(self):
        """The H1 signature: equal mean success rate, lower coverage when p is polarized."""
        uniform = [(32, 6)] * 100  # p = 0.1875 on every prompt
        polarized = [(32, 30)] * 20 + [(32, 0)] * 80  # mean p = 0.1875
        self.assertAlmostEqual(coverage(uniform, 1), coverage(polarized, 1))
        self.assertEqual(coverage(uniform, 32), 1.0)
        self.assertEqual(coverage(polarized, 32), 0.2)
        gu, gp = group_signal(uniform), group_signal(polarized)
        self.assertGreater(gp["all_fail"], gu["all_fail"])
        self.assertGreater(gp["all_success"], gu["all_success"])


class Summaries(unittest.TestCase):
    def test_histogram_counts_every_prompt(self):
        counts = [(32, 0), (32, 32), (32, 1), (32, 16), (32, 31)]
        h = p_histogram(counts)
        self.assertEqual(sum(h.values()), len(counts))
        self.assertEqual(h["p=0"], 1)
        self.assertEqual(h["p=1"], 1)

    def test_bootstrap_interval_brackets_estimate(self):
        counts = [(32, c) for c in [0, 0, 1, 3, 5, 0, 32, 2, 0, 8] * 10]
        lo, hi = bootstrap_ci(counts, lambda xs: coverage(xs, 32), b=500)
        self.assertLessEqual(lo, coverage(counts, 32))
        self.assertGreaterEqual(hi, coverage(counts, 32))


if __name__ == "__main__":
    unittest.main()
