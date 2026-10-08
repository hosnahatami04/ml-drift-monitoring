import numpy as np
import pandas as pd
import pytest

from driftmon.config import ThresholdsConfig
from driftmon.data import schema
from driftmon.model.artifacts import ModelArtifact
from driftmon.monitoring.monitor import Monitor, WindowReport

from .conftest import make_adult_frame


@pytest.fixture
def monitor(tiny_artifact: ModelArtifact, thresholds: ThresholdsConfig) -> Monitor:
    return Monitor(tiny_artifact.reference, thresholds)


def _window_from(frame: pd.DataFrame, artifact: ModelArtifact) -> tuple[pd.DataFrame, np.ndarray]:
    features = frame[list(schema.MODEL_FEATURES)]
    probabilities = artifact.model.predict_proba(features)[:, 1]
    return features, probabilities


def test_clean_window_raises_no_drift(monitor: Monitor, tiny_artifact: ModelArtifact) -> None:
    clean = make_adult_frame(600, seed=33)
    features, probabilities = _window_from(clean, tiny_artifact)

    report = monitor.process_window(0, features, probabilities)

    assert not report.feature_drift
    assert not report.prediction_drift
    assert report.flagged_features == ()


def test_shifted_window_flags_the_shifted_feature(
    monitor: Monitor, tiny_artifact: ModelArtifact
) -> None:
    shifted = make_adult_frame(600, seed=34)
    shifted["age"] = np.clip(shifted["age"] + 15, 17, 90)
    features, probabilities = _window_from(shifted, tiny_artifact)

    report = monitor.process_window(1, features, probabilities)

    assert report.feature_drift
    assert "age" in report.flagged_features


def test_prediction_shift_flags_prediction_drift(
    monitor: Monitor, tiny_artifact: ModelArtifact
) -> None:
    reference_probs = np.asarray(tiny_artifact.reference.prediction_probs)
    clean = make_adult_frame(len(reference_probs), seed=35)
    features, _ = _window_from(clean, tiny_artifact)
    shifted_probs = np.clip(reference_probs + 0.25, 0.0, 1.0)

    report = monitor.process_window(2, features, shifted_probs)

    assert report.prediction_drift


def test_same_window_gives_identical_report(
    monitor: Monitor, tiny_artifact: ModelArtifact, thresholds: ThresholdsConfig
) -> None:
    clean = make_adult_frame(600, seed=36)
    features, probabilities = _window_from(clean, tiny_artifact)

    first = monitor.process_window(3, features, probabilities)
    fresh = Monitor(tiny_artifact.reference, thresholds)
    second = fresh.process_window(3, features, probabilities)

    assert first.to_dict() == second.to_dict()


def test_window_report_counts_every_detector(
    monitor: Monitor, tiny_artifact: ModelArtifact
) -> None:
    clean = make_adult_frame(600, seed=37)
    features, probabilities = _window_from(clean, tiny_artifact)

    report = monitor.process_window(4, features, probabilities)

    # 5 numeric features x (KS + PSI) + 4 categorical x (PSI + chi2) + prediction KS.
    assert len(report.results) == 19


def test_empty_window_raises(monitor: Monitor) -> None:
    with pytest.raises(ValueError, match="empty"):
        monitor.process_window(0, pd.DataFrame(columns=list(schema.MODEL_FEATURES)), np.array([]))


def test_length_mismatch_raises(monitor: Monitor, tiny_artifact: ModelArtifact) -> None:
    clean = make_adult_frame(100, seed=38)
    features, _ = _window_from(clean, tiny_artifact)

    with pytest.raises(ValueError, match="same length"):
        monitor.process_window(0, features, np.array([0.5]))


def test_missing_categoricals_are_tolerated(monitor: Monitor, tiny_artifact: ModelArtifact) -> None:
    frame = make_adult_frame(600, seed=39)
    frame.loc[frame.index[:50], "workclass"] = None
    features, probabilities = _window_from(frame, tiny_artifact)

    report = monitor.process_window(5, features, probabilities)

    assert isinstance(report, WindowReport)


def test_error_rate_and_page_hinkley_flow(monitor: Monitor) -> None:
    probabilities = np.array([0.9, 0.8, 0.2, 0.1])
    labels = np.array([1, 0, 0, 0])

    assert Monitor.error_rate(probabilities, labels) == 0.25

    # A stable error stream stays quiet; a sustained jump eventually alarms.
    assert not any(monitor.observe_error_rate(0.1) for _ in range(30))
    alarms = [monitor.observe_error_rate(0.6) for _ in range(20)]
    assert any(alarms)

    monitor.reset_performance()
    assert monitor.page_hinkley.score == 0.0
