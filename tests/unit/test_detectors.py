import itertools

import numpy as np
import pytest

from driftmon.detectors.base import TestResult as DetectorResult
from driftmon.detectors.chi2 import (
    OTHER_CATEGORY,
    _merged_counts,
    chi2_from_counts,
    chi2_test,
)
from driftmon.detectors.ks import ks_p_value, ks_test, prediction_drift_test
from driftmon.detectors.psi import psi_categorical, psi_numeric, quantile_bin_edges

RNG = np.random.default_rng(7)


class TestKs:
    def test_identical_sample_has_zero_statistic(self) -> None:
        sample = RNG.normal(0, 1, 500)

        result = ks_test(sample, sample.copy(), "x")

        assert result.statistic == 0.0
        assert result.p_value == 1.0

    def test_large_shift_is_detected(self) -> None:
        reference = RNG.normal(0, 1, 1000)
        current = RNG.normal(3, 1, 1000)

        result = ks_test(reference, current, "x")

        assert result.statistic > 0.8
        assert result.p_value is not None and result.p_value < 1e-10

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            ks_test(np.array([]), np.array([1.0]), "x")

    def test_nan_raises(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            ks_test(np.array([1.0, np.nan]), np.array([1.0, 2.0]), "x")

    def test_constant_feature_is_quiet(self) -> None:
        constant = np.full(500, 3.0)

        result = ks_test(constant, constant.copy(), "x")

        assert result.statistic == 0.0

    def test_p_value_decreases_with_statistic(self) -> None:
        ds = np.linspace(0.01, 0.9, 30)
        values = [ks_p_value(float(d), 1000, 1000) for d in ds]
        assert all(0.0 <= v <= 1.0 for v in values)
        assert all(a >= b for a, b in itertools.pairwise(values))

    def test_prediction_drift_uses_own_method_name(self) -> None:
        result = prediction_drift_test(RNG.uniform(0, 1, 300), RNG.uniform(0, 1, 300))

        assert result.method == "prediction_ks"
        assert result.feature == "__prediction__"


class TestPsi:
    def test_zero_for_identical_distribution(self) -> None:
        sample = RNG.normal(0, 1, 2000)

        result = psi_numeric(sample, sample.copy(), "x")

        assert result.statistic == pytest.approx(0.0, abs=1e-12)
        assert result.p_value is None

    def test_non_negative(self) -> None:
        for _ in range(10):
            reference = RNG.normal(0, 1, 500)
            current = RNG.normal(RNG.uniform(-2, 2), 1, 500)
            assert psi_numeric(reference, current, "x").statistic >= 0.0

    def test_symmetric(self) -> None:
        reference = RNG.normal(0, 1, 2000)
        current = RNG.normal(0.5, 1.2, 2000)
        edges = quantile_bin_edges(reference)

        forward = psi_numeric(reference, current, "x", bin_edges=edges)
        backward = psi_numeric(current, reference, "x", bin_edges=edges)

        assert forward.statistic == pytest.approx(backward.statistic, rel=1e-9)

    def test_large_shift_gives_large_psi(self) -> None:
        reference = RNG.normal(0, 1, 1000)
        current = RNG.normal(2, 1, 1000)

        assert psi_numeric(reference, current, "x").statistic > 0.5

    def test_constant_reference_collapses_to_zero(self) -> None:
        reference = np.full(500, 1.0)
        current = RNG.normal(5, 1, 500)

        result = psi_numeric(reference, current, "x")

        assert result.statistic == 0.0

    def test_values_outside_reference_range_fall_in_edge_bins(self) -> None:
        reference = RNG.uniform(0, 1, 1000)
        current = np.concatenate([RNG.uniform(0, 1, 500), np.full(500, 100.0)])

        result = psi_numeric(reference, current, "x")

        assert np.isfinite(result.statistic)
        assert result.statistic > 0.5

    def test_categorical_psi_handles_unseen_category(self) -> None:
        frequencies = {"a": 0.5, "b": 0.5}
        current = np.array(["a", "b", "c", "c"], dtype=object)

        result = psi_categorical(frequencies, current, "x")

        assert np.isfinite(result.statistic)
        assert result.statistic > 0.0

    def test_categorical_psi_zero_for_matching_frequencies(self) -> None:
        frequencies = {"a": 0.5, "b": 0.5}
        current = np.array(["a", "b"] * 100, dtype=object)

        result = psi_categorical(frequencies, current, "x")

        assert result.statistic == pytest.approx(0.0, abs=1e-12)

    def test_quantile_edges_are_unique_and_sorted(self) -> None:
        values = np.array([0.0] * 90 + list(range(10)), dtype=float)

        edges = quantile_bin_edges(values)

        assert edges == sorted(set(edges))


class TestChi2:
    def test_identical_distribution_not_significant(self) -> None:
        categories = np.array(["a"] * 400 + ["b"] * 300 + ["c"] * 300, dtype=object)

        result = chi2_test(categories, categories.copy(), "x")

        assert result.statistic == pytest.approx(0.0, abs=1e-9)
        assert result.p_value == pytest.approx(1.0)

    def test_strong_shift_is_significant(self) -> None:
        reference = np.array(["a"] * 500 + ["b"] * 500, dtype=object)
        current = np.array(["a"] * 900 + ["b"] * 100, dtype=object)

        result = chi2_test(reference, current, "x")

        assert result.p_value is not None and result.p_value < 1e-10
        assert result.effect_size > 0.3

    def test_rare_categories_merge_into_other(self) -> None:
        reference = {"a": 500.0, "b": 500.0, "rare": 2.0}
        current = {"a": 500.0, "b": 500.0}

        names, rows = _merged_counts(reference, current)

        assert OTHER_CATEGORY in names
        assert "rare" not in names
        assert len(rows) == len(names)

    def test_unseen_current_category_goes_to_other(self) -> None:
        reference = {"a": 500.0, "b": 500.0}
        current = {"a": 400.0, "b": 400.0, "new": 200.0}

        names, _ = _merged_counts(reference, current)

        assert OTHER_CATEGORY in names
        assert "new" not in names

    def test_counts_and_array_paths_agree(self) -> None:
        reference = np.array(["a"] * 300 + ["b"] * 200, dtype=object)
        current = np.array(["a"] * 150 + ["b"] * 350, dtype=object)

        from_arrays = chi2_test(reference, current, "x")
        from_counts = chi2_from_counts({"a": 300.0, "b": 200.0}, {"a": 150.0, "b": 350.0}, "x")

        assert from_arrays.statistic == pytest.approx(from_counts.statistic)
        assert from_arrays.p_value == pytest.approx(from_counts.p_value or 0.0)

    def test_single_category_returns_no_evidence(self) -> None:
        constant = np.array(["only"] * 100, dtype=object)

        result = chi2_test(constant, constant.copy(), "x")

        assert result.statistic == 0.0
        assert result.p_value == 1.0
        assert not result.flagged

    def test_empty_sample_raises(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            chi2_test(np.array([], dtype=object), np.array(["a"], dtype=object), "x")


def test_result_is_immutable() -> None:
    result = DetectorResult(feature="x", method="ks", statistic=0.1, p_value=0.5, effect_size=0.1)
    with pytest.raises(AttributeError):
        result.flagged = True  # type: ignore[misc]
