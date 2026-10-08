"""Two-sample Kolmogorov-Smirnov test implemented on numpy.

The D statistic is computed by hand from the merged sorted ECDFs. For the
p-value only the distribution function comes from scipy (`kstwo`, the
two-sided KS distribution at the effective sample size), which is exactly what
scipy.stats.ks_2samp(method="asymp") evaluates; modern scipy no longer uses
the classical Kolmogorov asymptotic series there. See ADR 0005.
"""

import numpy as np
from scipy.stats import kstwo

from driftmon.detectors.base import FloatArray, TestResult, require_valid_sample

PREDICTION_FEATURE = "__prediction__"


def ks_statistic(reference: FloatArray, current: FloatArray) -> float:
    ref_sorted = np.sort(reference)
    cur_sorted = np.sort(current)
    pooled = np.concatenate([ref_sorted, cur_sorted])
    cdf_ref = np.searchsorted(ref_sorted, pooled, side="right") / ref_sorted.size
    cdf_cur = np.searchsorted(cur_sorted, pooled, side="right") / cur_sorted.size
    return float(np.max(np.abs(cdf_ref - cdf_cur)))


def ks_p_value(d: float, n_reference: int, n_current: int) -> float:
    effective_n = n_reference * n_current / (n_reference + n_current)
    p_value = float(kstwo.sf(d, int(np.round(effective_n))))
    return min(1.0, max(0.0, p_value))


def ks_test(reference: FloatArray, current: FloatArray, feature: str) -> TestResult:
    require_valid_sample(reference, "reference")
    require_valid_sample(current, "current")
    d = ks_statistic(reference, current)
    p_value = ks_p_value(d, reference.size, current.size)
    return TestResult(feature=feature, method="ks", statistic=d, p_value=p_value, effect_size=d)


def prediction_drift_test(reference_probs: FloatArray, current_probs: FloatArray) -> TestResult:
    """Label-free drift signal: KS on the model's predicted probabilities."""
    result = ks_test(reference_probs, current_probs, PREDICTION_FEATURE)
    return TestResult(
        feature=result.feature,
        method="prediction_ks",
        statistic=result.statistic,
        p_value=result.p_value,
        effect_size=result.effect_size,
    )
