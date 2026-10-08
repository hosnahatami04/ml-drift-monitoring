import numpy as np
import pytest

from driftmon.config import load_thresholds
from driftmon.detectors.base import TestResult as DetectorResult
from driftmon.detectors.ks import ks_test
from driftmon.detectors.multiple_testing import (
    DecisionConfig,
    benjamini_hochberg,
    bonferroni,
    decide_window,
)


def _result(
    feature: str,
    method: str = "ks",
    p_value: float | None = 0.001,
    effect_size: float = 0.2,
) -> DetectorResult:
    return DetectorResult(
        feature=feature,
        method=method,
        statistic=effect_size,
        p_value=p_value,
        effect_size=effect_size,
    )


class TestCorrections:
    # Hand-computed: m=4, alpha=0.05 -> Bonferroni cutoff 0.0125.
    def test_bonferroni_hand_example(self) -> None:
        p_values = [0.001, 0.012, 0.013, 0.9]

        assert bonferroni(p_values, alpha=0.05) == [True, True, False, False]

    # Hand-computed BH, m=5, alpha=0.05: sorted p (0.005, 0.009, 0.02, 0.04, 0.9)
    # against (0.01, 0.02, 0.03, 0.04, 0.05): largest k where p<=k*alpha/m is 4.
    def test_benjamini_hochberg_hand_example(self) -> None:
        p_values = [0.9, 0.009, 0.04, 0.005, 0.02]

        assert benjamini_hochberg(p_values, alpha=0.05) == [
            False,
            True,
            True,
            True,
            True,
        ]

    def test_bh_rejects_at_least_as_much_as_bonferroni(self) -> None:
        rng = np.random.default_rng(0)
        p_values = list(rng.uniform(0, 1, 50))

        bonf = bonferroni(p_values, alpha=0.05)
        bh = benjamini_hochberg(p_values, alpha=0.05)

        assert all(not b or h for b, h in zip(bonf, bh, strict=True))


class TestWindowDecision:
    def test_empty_window_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            decide_window([], DecisionConfig())

    def test_significant_and_large_effect_is_flagged(self) -> None:
        decision = decide_window([_result("age")], DecisionConfig())

        assert decision.flagged_features == ("age",)
        assert decision.drift

    def test_significant_but_tiny_effect_is_gated(self) -> None:
        # Large n makes negligible shifts significant; the gate must hold.
        rng = np.random.default_rng(1)
        reference = rng.normal(0, 1, 200_000)
        current = rng.normal(0.02, 1, 200_000)
        result = ks_test(reference, current, "age")
        assert result.p_value is not None and result.p_value < 0.01

        decision = decide_window([result], DecisionConfig(min_ks_d=0.05))

        assert not decision.drift
        assert decision.flagged_features == ()

    def test_large_effect_but_not_significant_is_not_flagged(self) -> None:
        decision = decide_window([_result("age", p_value=0.5, effect_size=0.5)], DecisionConfig())

        assert not decision.drift

    def test_psi_flagged_by_threshold_alone(self) -> None:
        results = [
            _result("age", method="psi", p_value=None, effect_size=0.3),
            _result("hours", method="psi", p_value=None, effect_size=0.01),
        ]

        decision = decide_window(results, DecisionConfig(min_psi=0.1))

        assert decision.flagged_features == ("age",)

    def test_k_features_rule(self) -> None:
        results = [_result("age"), _result("hours")]

        strict = decide_window(results, DecisionConfig(k_features=3))
        loose = decide_window(results, DecisionConfig(k_features=2))

        assert not strict.drift
        assert loose.drift

    def test_feature_counted_once_across_methods(self) -> None:
        results = [
            _result("age", method="ks"),
            _result("age", method="psi", p_value=None, effect_size=0.5),
        ]

        decision = decide_window(results, DecisionConfig(k_features=2))

        assert decision.flagged_features == ("age",)
        assert not decision.drift

    def test_bonferroni_is_stricter_within_window(self) -> None:
        results = [_result(f"f{i}", p_value=0.004) for i in range(5)]

        bonf = decide_window(results, DecisionConfig(alpha=0.01, correction="bonferroni"))
        bh = decide_window(results, DecisionConfig(alpha=0.01, correction="benjamini-hochberg"))

        assert not bonf.drift
        assert bh.drift

    def test_chi2_gated_by_cramers_v(self) -> None:
        results = [_result("workclass", method="chi2", p_value=0.0001, effect_size=0.01)]

        decision = decide_window(results, DecisionConfig(min_cramers_v=0.05))

        assert not decision.drift


def test_thresholds_file_loads_into_valid_config() -> None:
    thresholds = load_thresholds()

    assert thresholds.correction == "bonferroni"
    assert thresholds.page_hinkley.lambda_ > 0
    config = DecisionConfig(
        alpha=thresholds.alpha,
        correction=thresholds.correction,
        k_features=thresholds.k_features,
        min_ks_d=thresholds.min_effect.ks_d,
        min_psi=thresholds.min_effect.psi,
        min_cramers_v=thresholds.min_effect.cramers_v,
    )
    assert config.alpha == thresholds.alpha
