from dataclasses import replace

import numpy as np
import pytest

from driftmon.config import Settings, ThresholdsConfig
from driftmon.data import schema
from driftmon.evaluation import harness
from driftmon.evaluation.harness import HarnessInputs, StreamRecord
from driftmon.evaluation.kpis import AlertEvent
from driftmon.model.artifacts import ModelArtifact
from driftmon.monitoring.alerts import AlertManager
from driftmon.monitoring.monitor import Monitor
from driftmon.simulation import sampler
from driftmon.simulation.scenarios import (
    ScenarioConfig,
    StreamParams,
    SubgroupRule,
    generate_stream,
)

from .conftest import make_adult_frame

PARAMS = StreamParams(n_windows=10, window_size=200, drift_start=4, label_delay=2)

SCENARIOS = {
    "control": ScenarioConfig(name="control", kind="control"),
    "sudden_covariate": ScenarioConfig(
        name="sudden_covariate",
        kind="sudden_covariate",
        feature="age",
        severity={"medium": {"beta": 1.0}},
    ),
    "concept": ScenarioConfig(
        name="concept",
        kind="concept",
        subgroup=[SubgroupRule(feature="hours-per-week", min=45)],
        severity={"medium": {"p": 0.5}},
    ),
}


@pytest.fixture(scope="module")
def inputs(tiny_artifact: ModelArtifact) -> HarnessInputs:
    pool = make_adult_frame(3000, seed=70)
    return HarnessInputs(
        pool=pool,
        simulation_stats=sampler.reference_feature_stats(
            make_adult_frame(1500, seed=71), schema.NUMERIC_FEATURES
        ),
        model=tiny_artifact.model,
        reference=tiny_artifact.reference,
    )


@pytest.fixture(scope="module")
def records(inputs: HarnessInputs) -> list[StreamRecord]:
    tasks = harness.build_tasks(SCENARIOS, ["medium"], [100, 101], PARAMS, horizon=4)
    return harness.run_streams(tasks, inputs, n_workers=1)


def test_build_tasks_runs_control_once_per_seed() -> None:
    tasks = harness.build_tasks(SCENARIOS, ["low", "medium"], [1, 2, 3], PARAMS, horizon=4)

    control = [t for t in tasks if t.scenario.kind == "control"]
    assert len(control) == 3
    assert all(t.severity is None for t in control)
    assert len(tasks) == 3 + 2 * 2 * 3


def test_record_shapes_follow_the_stream(records: list[StreamRecord]) -> None:
    record = records[0]

    assert record.n_windows == PARAMS.n_windows
    assert record.statistic.shape == (PARAMS.n_windows, len(record.tests))
    assert np.isnan(record.rolling_auc[0])
    # PSI rows carry no p-value; every other test does.
    psi = [j for j, (_, method) in enumerate(record.tests) if method == "psi"]
    assert np.isnan(record.p_value[:, psi]).all()
    assert len(record.window_results(0, frozenset({"psi"}))) == len(psi)


def test_serial_and_parallel_runs_are_identical(
    inputs: HarnessInputs, records: list[StreamRecord]
) -> None:
    tasks = [r.task for r in records]

    parallel = harness.run_streams(tasks, inputs, n_workers=2)

    for a, b in zip(records, parallel, strict=True):
        assert a.task == b.task
        np.testing.assert_array_equal(a.statistic, b.statistic)
        np.testing.assert_array_equal(a.p_value, b.p_value)
        np.testing.assert_array_equal(a.error_rate, b.error_rate)
        np.testing.assert_array_equal(a.rolling_auc, b.rolling_auc)


def test_rerun_gives_identical_outcomes(
    inputs: HarnessInputs, records: list[StreamRecord], thresholds: ThresholdsConfig
) -> None:
    again = harness.run_streams([r.task for r in records], inputs, n_workers=1)

    first = harness.outcomes_for(records, inputs.reference, thresholds, 0.03)
    second = harness.outcomes_for(again, inputs.reference, thresholds, 0.03)

    assert first == second


def test_replay_matches_the_live_monitor_path(
    inputs: HarnessInputs, records: list[StreamRecord], thresholds: ThresholdsConfig
) -> None:
    """The harness timeline must equal running Monitor.process_window and the
    alert manager window by window, the way the service runner does."""
    record = next(r for r in records if r.task.scenario.kind == "sudden_covariate")
    task = record.task

    monitor = Monitor(inputs.reference, thresholds)
    manager = AlertManager(thresholds.alerts)
    errors: list[float] = []
    live: list[AlertEvent] = []
    for window in generate_stream(
        inputs.pool, inputs.simulation_stats, task.scenario, task.severity, task.seed, task.params
    ):
        labelled = window.index - task.params.label_delay
        if labelled >= 0:
            alarm = monitor.observe_error_rate(errors[labelled])
            alert = manager.evaluate_performance(labelled, errors[labelled], alarm)
            if alert is not None:
                live.append(AlertEvent(window.index, alert.type))
        features = window.X[list(schema.MODEL_FEATURES)]
        probabilities = inputs.model.predict_proba(features)[:, 1]
        report = monitor.process_window(window.index, features, probabilities)
        live.extend(AlertEvent(window.index, a.type) for a in manager.evaluate_window(report))
        errors.append(Monitor.error_rate(probabilities, window.y_true))

    replayed = harness.alert_timeline(
        record, Monitor(inputs.reference, thresholds), thresholds.alerts
    )

    assert list(replayed) == live
    assert live, "the high-beta stream should alert at least once"


def test_input_and_performance_replays_compose(
    inputs: HarnessInputs, records: list[StreamRecord], thresholds: ThresholdsConfig
) -> None:
    monitor = Monitor(inputs.reference, thresholds)
    for record in records:
        both = harness.alert_timeline(record, monitor, thresholds.alerts)
        only_inputs = harness.alert_timeline(record, monitor, thresholds.alerts, performance=False)
        only_perf = harness.alert_timeline(record, monitor, thresholds.alerts, inputs=False)

        assert sorted(both, key=repr) == sorted(only_inputs + only_perf, key=repr)


def test_performance_alerts_are_raised_when_labels_arrive(
    inputs: HarnessInputs, records: list[StreamRecord]
) -> None:
    record = next(r for r in records if r.task.scenario.kind == "control")
    hair_trigger = ThresholdsConfig.model_validate({"page_hinkley": {"delta": 0.0, "lambda": 1e-9}})
    # A constant error stream keeps Page-Hinkley at zero; a jump at window 5
    # alarms immediately, and that alarm surfaces when window 5's labels land.
    errors = np.full(record.n_windows, 0.1)
    errors[5:] = 0.5
    stepped = replace(record, error_rate=errors)

    events = harness.alert_timeline(
        stepped, Monitor(inputs.reference, hair_trigger), hair_trigger.alerts, inputs=False
    )

    assert events[0] == AlertEvent(5 + PARAMS.label_delay, "performance_drift")


def test_method_filter_restricts_the_decision(
    inputs: HarnessInputs, records: list[StreamRecord], thresholds: ThresholdsConfig
) -> None:
    record = records[0]
    monitor = Monitor(inputs.reference, thresholds)

    report = monitor.decide(0, PARAMS.window_size, record.window_results(0, frozenset({"ks"})))

    assert {r.method for r in report.results} == {"ks"}


def test_scaled_params_keep_traffic_volume() -> None:
    base = StreamParams(n_windows=60, window_size=1000, drift_start=20, label_delay=3)

    small, small_horizon, factor = harness.scaled_task_params(base, 20, 250)
    large, large_horizon, _ = harness.scaled_task_params(base, 20, 2000)

    assert factor == 4
    assert (small.n_windows, small.drift_start, small.label_delay, small_horizon) == (
        240,
        80,
        12,
        80,
    )
    assert small.n_windows * small.window_size == 60_000
    # 1.5 windows of delay rounds up: labels never arrive earlier than in the base.
    assert (large.n_windows, large.drift_start, large.label_delay, large_horizon) == (30, 10, 2, 10)


def test_scaled_scenario_rescales_timing_only() -> None:
    gradual = ScenarioConfig(
        name="gradual_covariate",
        kind="gradual_covariate",
        feature="hours-per-week",
        ramp_windows=15,
        severity={"medium": {"beta": 0.5}},
    )

    scaled = harness.scaled_scenario(gradual, 0.5)

    assert scaled.ramp_windows == 8
    assert scaled.severity == gradual.severity


def test_decision_ablations_cover_every_arm(
    inputs: HarnessInputs, records: list[StreamRecord], thresholds: ThresholdsConfig
) -> None:
    arms = harness.decision_ablations(records, inputs.reference, thresholds, 0.03, 200)

    groups = {(a.group, a.arm) for a in arms}
    assert ("correction", "benjamini-hochberg") in groups
    assert ("effect_gate", "without KS/chi2 gate") in groups
    assert ("method", "PSI only") in groups
    for arm in arms:
        assert all(a.type == "feature_drift" for o in arm.outcomes for a in o.alerts)
        assert all(o.kind != "concept" for o in arm.outcomes)


def test_run_evaluation_smoke_uses_first_evaluation_seeds(
    inputs: HarnessInputs, thresholds: ThresholdsConfig
) -> None:
    settings = Settings(
        window_size=PARAMS.window_size,
        label_delay=PARAMS.label_delay,
        stream={"n_windows": PARAMS.n_windows, "drift_start": 4, "detection_horizon": 4},
        evaluation={"smoke_seed_count": 2},
    )

    run = harness.run_evaluation(settings, inputs, SCENARIOS, thresholds, smoke=True, n_workers=1)

    assert run.seeds == [100, 101]
    assert run.ablations == []
    assert {o.severity for o in run.outcomes} == {None, "medium"}


def test_detector_correctness_is_within_tolerance() -> None:
    worst = harness.detector_correctness(seed=3, n_cases=30)

    assert set(worst) == {"ks_statistic", "ks_p_value", "chi2_statistic", "chi2_p_value"}
    assert max(worst.values()) <= 1e-6
