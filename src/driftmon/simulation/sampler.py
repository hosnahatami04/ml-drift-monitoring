"""Seeded sampling from the stream pool, including importance resampling.

Every (stream seed, window index, purpose) triple gets its own generator, so
windows can be produced in any order - or in parallel - without changing the
result, and scenarios that must share row draws (control and concept) do so
by construction.
"""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from driftmon.detectors.base import FloatArray

_SUBSTREAMS = {"rows": 0, "labels": 1}

IntArray = np.ndarray[tuple[int], np.dtype[np.int64]]


def window_rng(seed: int, window_index: int, substream: str) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence((seed, window_index, _SUBSTREAMS[substream]))
    )


def sample_uniform(rng: np.random.Generator, pool_size: int, n: int) -> IntArray:
    """Sampling is with replacement: the pool (~14.6k rows) is reused across
    60 windows of 1000 rows, so windows cannot be disjoint anyway."""
    return rng.integers(0, pool_size, size=n, dtype=np.int64)


def importance_probabilities(z_scores: FloatArray, beta: float) -> FloatArray:
    """Resampling weights exp(beta * z). Reweighting real rows shifts the
    marginal while keeping the joint feature structure intact, which is what
    makes the scenario realistic compared to adding synthetic noise."""
    weights = np.exp(beta * z_scores)
    result: FloatArray = weights / weights.sum()
    return result


def sample_weighted(rng: np.random.Generator, probabilities: FloatArray, n: int) -> IntArray:
    return rng.choice(probabilities.size, size=n, replace=True, p=probabilities).astype(np.int64)


def standardize(values: FloatArray, mean: float, std: float) -> FloatArray:
    if std <= 0:
        raise ValueError("cannot standardize with non-positive std")
    result: FloatArray = (values - mean) / std
    return result


def reference_feature_stats(
    reference: pd.DataFrame, features: Sequence[str]
) -> dict[str, tuple[float, float]]:
    """Mean and std of each feature on the reference split; the simulator's z
    scores use reference statistics, mirroring what the detectors see."""
    stats: dict[str, tuple[float, float]] = {}
    for feature in features:
        values = reference[feature].to_numpy(dtype=float)
        stats[feature] = (float(values.mean()), float(values.std(ddof=1)))
    return stats
