import math

import numpy as np
import pytest

from driftmon.evaluation import kpis
from driftmon.evaluation.kpis import AlertEvent, StreamOutcome


def outcome(
    alerts: list[tuple[int, str]],
    *,
    kind: str = "sudden_covariate",
    t0: int | None = 20,
    seed: int = 0,
    horizon: int = 20,
    degradation: int | None = None,
) -> StreamOutcome:
    return StreamOutcome(
        scenario=kind,
        kind=kind,
        severity=None if t0 is None else "medium",
        seed=seed,
        t0=t0,
        horizon=horizon,
        n_windows=60,
        alerts=tuple(AlertEvent(window=w, type=t) for w, t in alerts),
        degradation_window=degradation,
    )


def test_first_correct_alert_defines_detection_and_ttd() -> None:
    o = outcome([(5, "feature_drift"), (23, "prediction_drift"), (25, "feature_drift")])

    assert kpis.first_detection(o) == 23
    assert kpis.time_to_detect(o) == 3


def test_alert_in_drift_start_window_has_zero_ttd() -> None:
    assert kpis.time_to_detect(outcome([(20, "feature_drift")])) == 0


def test_alert_past_horizon_is_not_a_detection() -> None:
    o = outcome([(40, "feature_drift")])

    assert kpis.first_detection(o) is None
    assert kpis.time_to_detect(o) is None


def test_last_window_inside_horizon_still_counts() -> None:
    assert kpis.time_to_detect(outcome([(39, "feature_drift")])) == 19


def test_concept_counts_only_performance_alerts() -> None:
    o = outcome([(21, "feature_drift"), (26, "performance_drift")], kind="concept")

    assert kpis.time_to_detect(o) == 6
    # The input alert inside the horizon is wrong on a concept stream.
    assert kpis.alert_counts(o) == (1, 1)


def test_precision_counts_early_alerts_wrong_and_ignores_late_ones() -> None:
    o = outcome(
        [
            (3, "feature_drift"),
            (19, "performance_drift"),
            (20, "feature_drift"),
            (24, "prediction_drift"),
            (45, "feature_drift"),
        ]
    )

    assert kpis.alert_counts(o) == (2, 2)


def test_precision_is_undefined_for_control() -> None:
    with pytest.raises(ValueError, match="drifted"):
        kpis.alert_counts(outcome([], t0=None))


def test_aggregate_rates_match_hand_computed_values() -> None:
    drifted = [
        outcome([(20, "feature_drift")]),
        outcome([(22, "feature_drift"), (5, "feature_drift")]),
        outcome([]),
        outcome([(50, "feature_drift")]),
    ]

    rate = kpis.detection_rate(drifted, samples=200, seed=1)
    ttd = kpis.median_ttd(drifted, samples=200, seed=1)
    precision = kpis.alert_precision(drifted, samples=200, seed=1)

    assert rate.value == 0.5
    assert ttd.value == 1.0
    # Correct: 20 and 22. Wrong: 5. Window 50 is past the horizon.
    assert precision.value == pytest.approx(2 / 3)
    assert rate.n == 4


def test_false_alarm_rates_on_control() -> None:
    control = [
        outcome([(3, "feature_drift"), (9, "performance_drift")], t0=None),
        outcome([], t0=None),
        outcome([], t0=None),
        outcome([(40, "prediction_drift")], t0=None),
    ]

    stream = kpis.stream_false_alarm_rate(control, samples=100, seed=0)
    window = kpis.per_window_false_alarm_rate(control, samples=100, seed=0)

    assert stream.value == 0.5
    assert window.value == pytest.approx(3 / 240)


def test_degradation_needs_two_consecutive_windows_below_the_line() -> None:
    auc = [math.nan, 0.92, 0.88, 0.91, 0.88, 0.87, 0.86]

    assert kpis.degradation_window(auc, reference_auc=0.92, margin=0.03, start=1) == 4
    assert kpis.degradation_window(auc, reference_auc=0.92, margin=0.03, start=5) == 5
    assert kpis.degradation_window(auc[:5], reference_auc=0.92, margin=0.03, start=1) is None


def test_lead_time_positive_when_alert_precedes_degradation() -> None:
    o = outcome([(10, "feature_drift"), (22, "feature_drift")], degradation=30)

    assert kpis.lead_time(o) == 8


def test_lead_time_negative_when_alert_follows_degradation() -> None:
    o = outcome([(33, "performance_drift")], kind="concept", degradation=30)

    assert kpis.lead_time(o) == -3


def test_no_degradation_means_no_lead_time_and_no_harm() -> None:
    harmless = outcome([(21, "feature_drift")])
    silent_harm = outcome([], degradation=30)

    assert kpis.lead_time(harmless) is None
    assert kpis.lead_time(silent_harm) is None
    share = kpis.harmful_share([harmless, silent_harm], samples=100, seed=0)
    assert share.value == 0.5
    lead = kpis.median_lead_time([harmless, silent_harm], samples=100, seed=0)
    # One harmful stream, never alerted: counted in n, no value.
    assert lead.n == 1
    assert lead.value is None


def test_paired_difference_uses_matching_seeds() -> None:
    treated = [outcome([(30, "feature_drift")], kind="concept", seed=s) for s in (1, 2)]
    control = [
        outcome([(30, "feature_drift")], t0=None, seed=1),
        outcome([], t0=None, seed=2),
    ]

    diff = kpis.paired_rate_difference(treated, control, kpis.INPUT_ALERT_TYPES, 100, 0)

    assert diff.value == 0.5


def test_bootstrap_is_deterministic_and_brackets_the_estimate() -> None:
    data = np.array([1.0] * 27 + [0.0] * 3)

    first = kpis.bootstrap(data, lambda v: float(np.mean(v)), samples=500, seed=7)
    second = kpis.bootstrap(data, lambda v: float(np.mean(v)), samples=500, seed=7)

    assert first == second
    assert first.ci_low is not None and first.ci_high is not None
    assert first.ci_low <= 0.9 <= first.ci_high


def test_bootstrap_of_empty_set_has_no_value() -> None:
    estimate = kpis.bootstrap(np.array([]), lambda v: 0.0, samples=10, seed=0)

    assert estimate.value is None
    assert estimate.n == 0
