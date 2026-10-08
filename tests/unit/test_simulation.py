import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from driftmon.config import ConfigError
from driftmon.data import schema
from driftmon.detectors.ks import ks_test
from driftmon.simulation import sampler
from driftmon.simulation.ground_truth import build_ground_truth
from driftmon.simulation.scenarios import (
    ScenarioConfig,
    StreamParams,
    SubgroupRule,
    Window,
    generate_stream,
    load_scenario,
    load_scenarios,
    ramp_fraction,
    seasonal_beta,
    subgroup_mask,
)

from .conftest import make_adult_frame

PARAMS = StreamParams(n_windows=10, window_size=400, drift_start=4, label_delay=3)

CONTROL = ScenarioConfig(name="control", kind="control")

SUDDEN = ScenarioConfig(
    name="sudden_covariate",
    kind="sudden_covariate",
    feature="age",
    severity={"medium": {"beta": 0.8}},
)

GRADUAL = ScenarioConfig(
    name="gradual_covariate",
    kind="gradual_covariate",
    feature="hours-per-week",
    ramp_windows=4,
    severity={"medium": {"beta": 0.8}},
)

CONCEPT = ScenarioConfig(
    name="concept",
    kind="concept",
    subgroup=[
        SubgroupRule(feature="hours-per-week", min=45),
        SubgroupRule(feature="education-num", min=13),
    ],
    severity={"medium": {"p": 0.3}},
)

SEASONAL = ScenarioConfig(
    name="seasonal",
    kind="seasonal",
    feature="age",
    period=4,
    severity={"medium": {"beta": 0.8}},
)


@pytest.fixture(scope="module")
def pool() -> pd.DataFrame:
    return make_adult_frame(4000, seed=20)


@pytest.fixture(scope="module")
def stats(pool: pd.DataFrame) -> dict[str, tuple[float, float]]:
    reference = make_adult_frame(1500, seed=21)
    return sampler.reference_feature_stats(reference, schema.NUMERIC_FEATURES)


def _windows(
    pool: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    scenario: ScenarioConfig,
    severity: str | None,
    seed: int = 5,
) -> list[Window]:
    return list(generate_stream(pool, stats, scenario, severity, seed, PARAMS))


def test_same_seed_gives_identical_stream(
    pool: pd.DataFrame, stats: dict[str, tuple[float, float]]
) -> None:
    first = _windows(pool, stats, SUDDEN, "medium")
    second = _windows(pool, stats, SUDDEN, "medium")

    for a, b in zip(first, second, strict=True):
        pd.testing.assert_frame_equal(a.X, b.X)
        np.testing.assert_array_equal(a.y_true, b.y_true)


def test_control_matches_pool_distribution(
    pool: pd.DataFrame, stats: dict[str, tuple[float, float]]
) -> None:
    windows = _windows(pool, stats, CONTROL, None)
    combined = np.concatenate([w.X["age"].to_numpy(dtype=float) for w in windows])

    result = ks_test(pool["age"].to_numpy(dtype=float), combined, "age")

    assert result.p_value is not None and result.p_value > 0.01


def test_covariate_windows_before_t0_match_control(
    pool: pd.DataFrame, stats: dict[str, tuple[float, float]]
) -> None:
    control = _windows(pool, stats, CONTROL, None)
    sudden = _windows(pool, stats, SUDDEN, "medium")
    gradual = _windows(pool, stats, GRADUAL, "medium")

    for i in range(PARAMS.drift_start):
        pd.testing.assert_frame_equal(sudden[i].X, control[i].X)
        pd.testing.assert_frame_equal(gradual[i].X, control[i].X)


def test_sudden_shifts_target_feature_with_expected_magnitude(
    pool: pd.DataFrame, stats: dict[str, tuple[float, float]]
) -> None:
    windows = _windows(pool, stats, SUDDEN, "medium")
    mean, std = stats["age"]
    z = sampler.standardize(pool["age"].to_numpy(dtype=float), mean, std)
    probabilities = sampler.importance_probabilities(z, 0.8)
    expected_mean = float(np.sum(probabilities * pool["age"].to_numpy(dtype=float)))
    pool_mean = float(pool["age"].mean())

    drifted_mean = float(
        np.mean([w.X["age"].mean() for w in windows if w.index >= PARAMS.drift_start])
    )

    assert expected_mean > pool_mean
    assert drifted_mean > pool_mean
    assert abs(drifted_mean - expected_mean) < 0.15 * (expected_mean - pool_mean) + 1.0


def test_ramp_fraction_is_exactly_linear() -> None:
    assert ramp_fraction(3, drift_start=4, ramp_windows=4) == 0.0
    assert ramp_fraction(4, drift_start=4, ramp_windows=4) == 0.25
    assert ramp_fraction(5, drift_start=4, ramp_windows=4) == 0.5
    assert ramp_fraction(7, drift_start=4, ramp_windows=4) == 1.0
    assert ramp_fraction(9, drift_start=4, ramp_windows=4) == 1.0


def test_concept_shares_control_feature_matrix_bitwise(
    pool: pd.DataFrame, stats: dict[str, tuple[float, float]]
) -> None:
    control = _windows(pool, stats, CONTROL, None)
    concept = _windows(pool, stats, CONCEPT, "medium")

    for c, k in zip(control, concept, strict=True):
        pd.testing.assert_frame_equal(c.X, k.X)


def test_concept_flips_only_subgroup_labels_at_about_p(
    pool: pd.DataFrame, stats: dict[str, tuple[float, float]]
) -> None:
    control = _windows(pool, stats, CONTROL, None)
    concept = _windows(pool, stats, CONCEPT, "medium")

    assert CONCEPT.subgroup is not None
    changed_outside = 0
    flips = 0
    subgroup_rows = 0
    for c, k in zip(control, concept, strict=True):
        diff = c.y_true != k.y_true
        mask = subgroup_mask(k.X, CONCEPT.subgroup)
        if k.index < PARAMS.drift_start:
            assert not diff.any()
            continue
        changed_outside += int(np.sum(diff & ~mask))
        flips += int(np.sum(diff & mask))
        subgroup_rows += int(mask.sum())

    assert changed_outside == 0
    assert subgroup_rows > 100
    rate = flips / subgroup_rows
    assert abs(rate - 0.3) < 0.08


def test_seasonal_beta_repeats_with_period() -> None:
    for i in range(4, 30):
        assert seasonal_beta(i, 4, 4, 0.8) == pytest.approx(
            seasonal_beta(i + 4, 4, 4, 0.8), abs=1e-12
        )
    assert seasonal_beta(3, 4, 4, 0.8) == 0.0
    assert seasonal_beta(4, 4, 4, 0.8) == pytest.approx(0.0, abs=1e-12)
    assert seasonal_beta(5, 4, 4, 0.8) == pytest.approx(0.8, abs=1e-12)


def test_labels_available_exactly_after_delay(
    pool: pd.DataFrame, stats: dict[str, tuple[float, float]]
) -> None:
    for window in _windows(pool, stats, CONTROL, None):
        assert window.label_available_at == window.index + PARAMS.label_delay


def test_missing_severity_raises(pool: pd.DataFrame, stats: dict[str, tuple[float, float]]) -> None:
    with pytest.raises(ValueError, match="severity"):
        next(generate_stream(pool, stats, SUDDEN, "extreme", 0, PARAMS))


class TestGroundTruth:
    def test_control_has_no_drift(self) -> None:
        truth = build_ground_truth(CONTROL, None, PARAMS)

        assert truth.t0 is None
        assert truth.active_windows == ()
        assert not truth.affects_label

    def test_sudden_active_from_t0(self) -> None:
        truth = build_ground_truth(SUDDEN, "medium", PARAMS)

        assert truth.t0 == PARAMS.drift_start
        assert truth.active_windows == tuple(range(4, 10))
        assert truth.affected_features == ("age",)

    def test_concept_affects_label_only(self) -> None:
        truth = build_ground_truth(CONCEPT, "medium", PARAMS)

        assert truth.affects_label
        assert truth.affected_features == ()
        assert truth.params["p"] == 0.3

    def test_seasonal_active_windows_follow_amplitude(self) -> None:
        truth = build_ground_truth(SEASONAL, "medium", PARAMS)

        for i in range(PARAMS.drift_start, PARAMS.n_windows):
            expected_active = abs(math.sin(2 * math.pi * (i - PARAMS.drift_start) / 4)) >= 0.5
            assert (i in truth.active_windows) == expected_active

    def test_round_trip_through_dict(self) -> None:
        truth = build_ground_truth(SEASONAL, "medium", PARAMS)

        from driftmon.simulation.ground_truth import GroundTruth

        assert GroundTruth.from_dict(truth.to_dict()) == truth


class TestScenarioFiles:
    def test_all_shipped_scenarios_load(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        scenarios = load_scenarios(repo_root / "configs" / "scenarios")

        assert set(scenarios) == {
            "control",
            "sudden_covariate",
            "gradual_covariate",
            "concept",
            "seasonal",
        }
        assert all(
            set(s.severity) == {"low", "medium", "high"}
            for s in scenarios.values()
            if s.kind != "control"
        )

    def test_invalid_scenario_is_rejected(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad.yaml"
        bad.write_text("name: x\nkind: sudden_covariate\n", encoding="utf-8")

        with pytest.raises(ConfigError, match="feature"):
            load_scenario(bad)
