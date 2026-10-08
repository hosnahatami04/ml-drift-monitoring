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
    """Window-level flagging thresholds. Values here are starting points;
    the calibrated ones live in configs/thresholds.yaml."""

    alpha: float = 0.01
    correction: Correction = "bonferroni"
    k_features: int = 1
    min_ks_d: float = 0.05
    min_psi: float = 0.1
    min_cramers_v: float = 0.05


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

    tested = [r for r in results if r.p_value is not None]
    significant: dict[int, bool] = {}
    if tested:
        p_values = [r.p_value for r in tested if r.p_value is not None]
        if config.correction == "bonferroni":
            rejected = bonferroni(p_values, config.alpha)
        else:
            rejected = benjamini_hochberg(p_values, config.alpha)
        significant = dict(zip([id(r) for r in tested], rejected, strict=True))

    decided: list[TestResult] = []
    for result in results:
        passes_significance = significant[id(result)] if result.p_value is not None else True
        flagged = passes_significance and _effect_gate(result, config)
        decided.append(replace(result, flagged=flagged))

    flagged_features = tuple(dict.fromkeys(r.feature for r in decided if r.flagged))
    return WindowDecision(
        results=tuple(decided),
        flagged_features=flagged_features,
        drift=len(flagged_features) >= config.k_features,
    )
