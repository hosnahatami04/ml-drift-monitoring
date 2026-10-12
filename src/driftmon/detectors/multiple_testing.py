"""Multiple-testing corrections and the per-window flagging decision."""

from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from driftmon.detectors.base import TestResult

Correction = Literal["bonferroni", "benjamini-hochberg"]


def bonferroni(p_values: Sequence[float], alpha: float) -> list[bool]:
    m = len(p_values)
    return [p <= alpha / m for p in p_values]


def benjamini_hochberg(p_values: Sequence[float], alpha: float) -> list[bool]:
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    rejected = [False] * m
    largest_k = 0
    for rank, index in enumerate(order, start=1):
        if p_values[index] <= rank * alpha / m:
            largest_k = rank
    for rank, index in enumerate(order, start=1):
        if rank <= largest_k:
            rejected[index] = True
    return rejected


@dataclass(frozen=True)
class DecisionConfig:
    """Window-level flagging thresholds. No defaults: the values come from
    configs/thresholds.yaml, so there is exactly one place they live."""

    alpha: float
    correction: Correction
    k_features: int
    min_ks_d: float
    min_psi: float
    min_cramers_v: float


@dataclass(frozen=True)
class WindowDecision:
    results: tuple[TestResult, ...]
    flagged_features: tuple[str, ...]
    drift: bool


def _effect_gate(result: TestResult, config: DecisionConfig) -> bool:
    # Large windows make negligible shifts statistically significant; the
    # effect-size gate keeps those from paging anyone.
    if result.method == "psi":
        return result.effect_size >= config.min_psi
    if result.method == "chi2":
        return result.effect_size >= config.min_cramers_v
    return result.effect_size >= config.min_ks_d


def decide_window(results: Sequence[TestResult], config: DecisionConfig) -> WindowDecision:
    """Correction over the p-valued tests, then the effect-size gate, then the
    k-features rule. PSI has no p-value and is gated by its threshold alone."""
    if not results:
        raise ValueError("cannot decide an empty window")

    tested = [i for i, r in enumerate(results) if r.p_value is not None]
    p_values = [results[i].p_value or 0.0 for i in tested]
    correct = bonferroni if config.correction == "bonferroni" else benjamini_hochberg
    rejected = correct(p_values, config.alpha) if tested else []
    significant = dict(zip(tested, rejected, strict=True))

    decided: list[TestResult] = []
    for i, result in enumerate(results):
        flagged = significant.get(i, True) and _effect_gate(result, config)
        decided.append(replace(result, flagged=flagged))

    flagged_features = tuple(dict.fromkeys(r.feature for r in decided if r.flagged))
    return WindowDecision(
        results=tuple(decided),
        flagged_features=flagged_features,
        drift=len(flagged_features) >= config.k_features,
    )
