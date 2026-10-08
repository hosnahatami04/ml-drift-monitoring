"""Hand-written detectors must agree with scipy to 1e-6 on arbitrary inputs."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from scipy import stats as scipy_stats

from driftmon.detectors.chi2 import pearson_chi2
from driftmon.detectors.ks import ks_test

TOLERANCE = 1e-6

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False, width=64
)


def float_sample(min_size: int = 2, max_size: int = 300) -> st.SearchStrategy[np.ndarray]:
    return arrays(
        dtype=np.float64,
        shape=st.integers(min_value=min_size, max_value=max_size),
        elements=finite_floats,
    )


@settings(max_examples=200, deadline=None)
@given(reference=float_sample(), current=float_sample())
def test_ks_matches_scipy_asymptotic(reference: np.ndarray, current: np.ndarray) -> None:
    ours = ks_test(reference, current, "feature")
    scipy_result = scipy_stats.ks_2samp(current, reference, method="asymp")

    assert abs(ours.statistic - float(scipy_result.statistic)) <= TOLERANCE
    assert abs((ours.p_value or 0.0) - float(scipy_result.pvalue)) <= TOLERANCE


@settings(max_examples=200, deadline=None)
@given(
    reference=float_sample(min_size=30, max_size=300),
    shift=st.floats(min_value=-100.0, max_value=100.0, allow_nan=False),
)
def test_ks_matches_scipy_on_shifted_discrete_data(reference: np.ndarray, shift: float) -> None:
    # Heavy ties mimic the integer-valued, zero-inflated Adult columns.
    discrete_ref = np.round(reference / 10.0) * 10.0
    discrete_cur = discrete_ref[: max(2, discrete_ref.size // 2)] + round(shift)

    ours = ks_test(discrete_ref, discrete_cur, "feature")
    scipy_result = scipy_stats.ks_2samp(discrete_cur, discrete_ref, method="asymp")

    assert abs(ours.statistic - float(scipy_result.statistic)) <= TOLERANCE
    assert abs((ours.p_value or 0.0) - float(scipy_result.pvalue)) <= TOLERANCE


@settings(max_examples=200, deadline=None)
@given(
    counts=st.lists(
        st.tuples(
            st.integers(min_value=1, max_value=5000),
            st.integers(min_value=1, max_value=5000),
        ),
        min_size=2,
        max_size=20,
    )
)
def test_pearson_chi2_matches_scipy(counts: list[tuple[int, int]]) -> None:
    table = np.asarray(counts, dtype=float)

    statistic, p_value, dof = pearson_chi2(table)
    scipy_result = scipy_stats.chi2_contingency(table, correction=False)

    assert abs(statistic - float(scipy_result.statistic)) <= TOLERANCE
    assert abs(p_value - float(scipy_result.pvalue)) <= TOLERANCE
    assert dof == int(scipy_result.dof)
