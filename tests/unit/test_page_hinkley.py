import numpy as np
import pytest

from driftmon.detectors.page_hinkley import PageHinkley


def test_rejects_invalid_parameters() -> None:
    with pytest.raises(ValueError):
        PageHinkley(delta=-0.1, lambda_=1.0)
    with pytest.raises(ValueError):
        PageHinkley(delta=0.1, lambda_=0.0)


def test_quiet_on_stationary_noise() -> None:
    rng = np.random.default_rng(42)
    detector = PageHinkley(delta=0.02, lambda_=0.5)

    alarms = [detector.update(float(v)) for v in rng.normal(0.1, 0.05, 1000)]

    assert not any(alarms)


def test_detects_step_change_with_bounded_delay() -> None:
    rng = np.random.default_rng(42)
    detector = PageHinkley(delta=0.02, lambda_=0.5)

    for value in rng.normal(0.1, 0.05, 200):
        assert not detector.update(float(value))

    delay = None
    for step, value in enumerate(rng.normal(0.5, 0.05, 50), start=1):
        if detector.update(float(value)):
            delay = step
            break

    assert delay is not None and delay <= 10


def test_reset_clears_state() -> None:
    detector = PageHinkley(delta=0.0, lambda_=0.5)
    for value in (1.0, 2.0, 3.0, 4.0):
        detector.update(value)
    assert detector.score > 0

    detector.reset()

    assert detector.score == 0.0
    assert not detector.update(0.0)


def test_score_grows_monotonically_under_sustained_shift() -> None:
    detector = PageHinkley(delta=0.01, lambda_=10.0)
    for _ in range(50):
        detector.update(0.1)
    baseline = detector.score

    for _ in range(20):
        detector.update(0.9)

    assert detector.score > baseline
