"""Population stability index for numeric and categorical features.

PSI has no sampling distribution, so results carry p_value=None and the flag
is driven purely by the calibrated effect threshold.
"""

from collections.abc import Mapping

import numpy as np

from driftmon.detectors.base import FloatArray, TestResult, require_valid_sample

N_BINS = 10

EPSILON = 1e-4


def quantile_bin_edges(values: FloatArray, n_bins: int = N_BINS) -> list[float]:
    """Bin edges at reference quantiles; duplicate edges (heavy point masses
    such as capital-gain == 0) are merged."""
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.unique(np.quantile(values, quantiles))
    return [float(e) for e in edges]


def _psi_from_fractions(reference: FloatArray, current: FloatArray) -> float:
    ref = np.clip(reference, EPSILON, None)
    cur = np.clip(current, EPSILON, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def _bin_fractions(values: FloatArray, edges: list[float]) -> FloatArray:
    # Interior edges only: values beyond the reference range fall into the
    # first or last bin instead of being dropped.
    interior = np.asarray(edges[1:-1], dtype=float)
    indices = np.searchsorted(interior, values, side="right")
    counts = np.bincount(indices, minlength=len(edges) - 1).astype(float)
    return counts / values.size


def psi_numeric(
    reference: FloatArray,
    current: FloatArray,
    feature: str,
    bin_edges: list[float] | None = None,
) -> TestResult:
    require_valid_sample(reference, "reference")
    require_valid_sample(current, "current")
    edges = bin_edges if bin_edges is not None else quantile_bin_edges(reference)
    if len(edges) < 3:
        # A constant reference collapses to a single bin; distribution shape
        # carries no usable signal, so report zero drift rather than failing.
        value = 0.0
    else:
        value = _psi_from_fractions(
            _bin_fractions(reference, edges), _bin_fractions(current, edges)
        )
    return TestResult(
        feature=feature, method="psi", statistic=value, p_value=None, effect_size=value
    )


def psi_categorical(
    reference_frequencies: Mapping[str, float],
    current: "np.ndarray[tuple[int], np.dtype[np.object_]]",
    feature: str,
) -> TestResult:
    if current.size == 0:
        raise ValueError("current sample is empty")
    categories = sorted(set(reference_frequencies) | set(current.tolist()))
    counts = {category: 0 for category in categories}
    values, value_counts = np.unique(current, return_counts=True)
    for value, count in zip(values.tolist(), value_counts.tolist(), strict=True):
        counts[str(value)] = int(count)
    ref = np.array([reference_frequencies.get(c, 0.0) for c in categories])
    cur = np.array([counts[c] / current.size for c in categories])
    value = _psi_from_fractions(ref, cur)
    return TestResult(
        feature=feature, method="psi", statistic=value, p_value=None, effect_size=value
    )
