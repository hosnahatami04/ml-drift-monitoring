from pathlib import Path

import numpy as np
import pytest

from driftmon.config import CalibrationGrid, SeedsConfig, ThresholdsConfig, load_thresholds
from driftmon.data import schema
from driftmon.evaluation import calibration, harness
from driftmon.evaluation.calibration import (
    CalibrationError,
    CandidateScore,
    InputCandidate,
    PerformanceCandidate,
)
from driftmon.evaluation.harness import HarnessInputs, StreamRecord
from driftmon.model.artifacts import ModelArtifact
from driftmon.simulation import sampler
from driftmon.simulation.scenarios import ScenarioConfig, StreamParams

from .conftest import make_adult_frame

PARAMS = StreamParams(n_windows=8, window_size=200, drift_start=3, label_delay=1)

SCENARIOS = {
    "control": ScenarioConfig(name="control", kind="control"),
    "sudden_covariate": ScenarioConfig(
        name="sudden_covariate",
        kind="sudden_covariate",
        feature="age",
        severity={"medium": {"beta": 1.0}},
    ),
}

TINY_GRID = CalibrationGrid(
    alpha=[0.001, 0.05],
    ks_d=[0.05],
    psi=[0.1],
    cramers_v=[0.1],
    k_features=[1],
    ph_delta=[0.005],
    ph_lambda=[0.05, 0.5],
)


def score(
    alarms: list[bool], first: list[float], alerts: list[int] | None = None
) -> CandidateScore:
    return CandidateScore(
        control_alarm=np.asarray(alarms, dtype=bool),
        control_alerts=np.asarray(alerts or [int(a) for a in alarms], dtype=np.int64),
        first_detection=np.asarray(first, dtype=float),
    )


def calibration_records(
    tiny_artifact: ModelArtifact, seeds: list[int]
) -> tuple[list[StreamRecord], HarnessInputs]:
    inputs = HarnessInputs(
        pool=make_adult_frame(3000, seed=80),
        simulation_stats=sampler.reference_feature_stats(
            make_adult_frame(1500, seed=81), schema.NUMERIC_FEATURES
        ),
        model=tiny_artifact.model,
        reference=tiny_artifact.reference,
    )
    tasks = harness.build_tasks(SCENARIOS, ["medium"], seeds, PARAMS, horizon=4)
    return harness.run_streams(tasks, inputs, n_workers=1), inputs


def test_evaluation_seeds_are_refused() -> None:
    config = SeedsConfig()

    calibration.ensure_calibration_seeds(config.calibration, config)
    with pytest.raises(CalibrationError, match="evaluation seeds"):
        calibration.ensure_calibration_seeds([0, 1, 100], config)


def test_calibrate_rejects_streams_from_evaluation_seeds(
    tiny_artifact: ModelArtifact,
) -> None:
    records, inputs = calibration_records(tiny_artifact, [100])

    with pytest.raises(CalibrationError, match="evaluation seeds"):
        calibration.calibrate(
            records, inputs.reference, ThresholdsConfig(), TINY_GRID, SeedsConfig(), n_workers=1
        )


def test_select_prefers_detection_under_the_false_alarm_ceiling() -> None:
    # Two control streams, two drifted streams (one scenario), t0 = 10.
    inputs = [
        score([True, True], [10, 10]),  # detects everything but always alarms
        score([False, False], [12, np.inf]),
        score([False, False], [11, 12]),
    ]
    performance = [score([False, False], [np.inf, np.inf])]

    i, j, feasible, detection, fa = calibration.select(
        inputs, performance, ["s", "s"], t0=10, horizon=5, max_false_alarm_rate=0.1
    )

    assert (i, j, feasible) == (2, 0, True)
    assert detection == {"s": 1.0}
    assert fa == 0.0


def test_select_breaks_ties_on_speed_then_false_alarms() -> None:
    inputs = [
        score([False, False], [14, 14]),
        score([False, False], [11, 11]),
    ]
    performance = [
        score([True, False], [np.inf, np.inf]),
        score([False, False], [np.inf, np.inf]),
    ]

    i, j, _, _, fa = calibration.select(
        inputs, performance, ["s", "s"], t0=10, horizon=5, max_false_alarm_rate=0.5
    )

    assert (i, j) == (1, 1)
    assert fa == 0.0


def test_select_combines_alert_families_as_a_union() -> None:
    # Inputs miss stream 2, Page-Hinkley catches it: together they detect both.
    inputs = [score([False], [10, np.inf])]
    performance = [score([False], [np.inf, 13])]

    _, _, _, detection, _ = calibration.select(
        inputs, performance, ["s", "s"], t0=10, horizon=5, max_false_alarm_rate=0.1
    )

    assert detection == {"s": 1.0}


def test_select_falls_back_to_lowest_false_alarm_when_nothing_is_feasible() -> None:
    inputs = [score([True, True], [10, 10]), score([True, False], [10, 10])]
    performance = [score([False, False], [np.inf, np.inf])]

    i, _, feasible, _, fa = calibration.select(
        inputs, performance, ["s", "s"], t0=10, horizon=5, max_false_alarm_rate=0.1
    )

    assert not feasible
    assert i == 1
    assert fa == 0.5


def test_apply_candidates_and_thresholds_file_round_trip(tmp_path: Path) -> None:
    base = ThresholdsConfig()
    tuned = calibration.apply_candidates(
        base,
        InputCandidate(alpha=0.005, ks_d=0.07, psi=0.2, cramers_v=0.08, k_features=2),
        PerformanceCandidate(delta=0.01, lambda_=0.3),
    )
    path = tmp_path / "thresholds.yaml"

    calibration.write_thresholds(path, tuned, [0, 1, 2])
    loaded = load_thresholds(path)

    assert loaded == tuned
    assert loaded.alerts == base.alerts
    assert loaded.page_hinkley.lambda_ == 0.3
    assert path.read_text(encoding="utf-8").startswith("# Frozen by `driftmon calibrate`")


def test_calibrate_picks_a_grid_point_on_calibration_seeds(
    tiny_artifact: ModelArtifact,
) -> None:
    records, inputs = calibration_records(tiny_artifact, [0, 1, 2])

    result = calibration.calibrate(
        records, inputs.reference, ThresholdsConfig(), TINY_GRID, SeedsConfig(), n_workers=1
    )

    assert result.thresholds.alpha in TINY_GRID.alpha
    assert result.thresholds.page_hinkley.lambda_ in TINY_GRID.ph_lambda
    assert result.n_input_candidates == 2
    assert result.n_performance_candidates == 2
    assert set(result.detection_rate) == {"sudden_covariate"}
    assert result.summary()["thresholds"]["page_hinkley"]["lambda"] in TINY_GRID.ph_lambda


def test_parallel_scoring_matches_serial(tiny_artifact: ModelArtifact) -> None:
    records, inputs = calibration_records(tiny_artifact, [0, 1])

    serial = calibration.calibrate(
        records, inputs.reference, ThresholdsConfig(), TINY_GRID, SeedsConfig(), n_workers=1
    )
    parallel = calibration.calibrate(
        records, inputs.reference, ThresholdsConfig(), TINY_GRID, SeedsConfig(), n_workers=2
    )

    assert serial == parallel


def test_calibrate_needs_control_streams(tiny_artifact: ModelArtifact) -> None:
    records, inputs = calibration_records(tiny_artifact, [0])
    drifted_only = [r for r in records if r.truth.t0 is not None]

    with pytest.raises(CalibrationError, match="control"):
        calibration.calibrate(
            drifted_only, inputs.reference, ThresholdsConfig(), TINY_GRID, SeedsConfig(), 1
        )
