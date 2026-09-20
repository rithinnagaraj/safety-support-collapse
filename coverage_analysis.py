"""Coverage estimators and paired hierarchical uncertainty calculations."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
import random
from typing import Callable, Iterable, Mapping, Sequence

from protocol_core import ANALYSIS_SEED


def coverage_from_counts(successes: Sequence[int], n: int, k: int) -> float:
    """Estimate C_k from per-prompt joint-success counts in n draws."""
    if not 1 <= k <= n:
        raise ValueError(f"k must satisfy 1 <= k <= n; received k={k}, n={n}")
    if not successes:
        raise ValueError("At least one prompt is required")
    denominator = math.comb(n, k)
    total = 0.0
    for count in successes:
        if not 0 <= count <= n:
            raise ValueError(f"Success count {count} is outside [0, {n}]")
        failures = n - count
        miss_probability = 0.0 if failures < k else math.comb(failures, k) / denominator
        total += 1.0 - miss_probability
    return total / len(successes)


def mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("Cannot average an empty sequence")
    return sum(values) / len(values)


def percentile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot take a percentile of an empty sequence")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be between zero and one")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float

    @property
    def width(self) -> float:
        return self.upper - self.lower

    def contains(self, value: float) -> bool:
        return self.lower <= value <= self.upper

    def to_dict(self) -> dict[str, float]:
        return {"estimate": self.estimate, "lower": self.lower, "upper": self.upper, "width": self.width}


def paired_hierarchical_bootstrap(
    rows: Iterable[Mapping[str, object]],
    value_a: Callable[[Mapping[str, object]], float],
    value_b: Callable[[Mapping[str, object]], float],
    *,
    replicates: int = 10_000,
    seed: int = ANALYSIS_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Paired bootstrap: template IDs, then latent instances within template.

    Each row must carry ``template_id`` and should carry ``latent_id`` (falling
    back to ``instance_id``). Pairing and grouped surface variants are preserved.
    """
    materialized = list(rows)
    if not materialized:
        raise ValueError("No rows supplied to bootstrap")
    grouped: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in materialized:
        instance_key = row.get("latent_id", row["instance_id"])
        grouped[str(row["template_id"])][str(instance_key)].append(row)
    templates = sorted(grouped)
    rng = random.Random(seed)

    def difference(sample: Sequence[Mapping[str, object]]) -> float:
        return mean([value_b(row) - value_a(row) for row in sample])

    estimate = difference(materialized)
    draws: list[float] = []
    for _ in range(replicates):
        sampled_rows: list[Mapping[str, object]] = []
        for _template_draw in templates:
            template = rng.choice(templates)
            instances = sorted(grouped[template])
            for _instance_draw in instances:
                instance = rng.choice(instances)
                sampled_rows.extend(grouped[template][instance])
        draws.append(difference(sampled_rows))
    draws.sort()
    return Interval(estimate, percentile(draws, alpha / 2), percentile(draws, 1 - alpha / 2))


def paired_coverage_bootstrap(
    rows: Iterable[Mapping[str, object]],
    *,
    count_a_key: str,
    count_b_key: str,
    n: int,
    k: int,
    replicates: int = 10_000,
    seed: int = ANALYSIS_SEED,
    alpha: float = 0.05,
) -> Interval:
    """Paired hierarchical interval for a C_k difference (B minus A)."""
    materialized = list(rows)
    if not materialized:
        raise ValueError("No prompt rows supplied to bootstrap")
    grouped: dict[str, dict[str, list[Mapping[str, object]]]] = defaultdict(lambda: defaultdict(list))
    for row in materialized:
        grouped[str(row["template_id"])][str(row["latent_id"])].append(row)
    templates = sorted(grouped)
    rng = random.Random(seed)

    def difference(sample: Sequence[Mapping[str, object]]) -> float:
        a = coverage_from_counts([int(row[count_a_key]) for row in sample], n, k)
        b = coverage_from_counts([int(row[count_b_key]) for row in sample], n, k)
        return b - a

    estimate = difference(materialized)
    draws: list[float] = []
    for _ in range(replicates):
        sample: list[Mapping[str, object]] = []
        for _template_draw in templates:
            template = rng.choice(templates)
            instances = sorted(grouped[template])
            for _instance_draw in instances:
                instance = rng.choice(instances)
                sample.extend(grouped[template][instance])
        draws.append(difference(sample))
    draws.sort()
    return Interval(estimate, percentile(draws, alpha / 2), percentile(draws, 1 - alpha / 2))


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Return Holm-adjusted p-values in the original order."""
    count = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda pair: pair[1])
    adjusted = [0.0] * count
    running = 0.0
    for rank, (index, value) in enumerate(indexed):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[index] = running
    return adjusted
