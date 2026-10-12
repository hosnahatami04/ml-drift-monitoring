"""Deterministic, parallel evaluation harness.

Workers run the expensive, threshold-independent half of the monitor once per
stream; alerting is replayed from those statistics, so a threshold sweep or an
ablation costs a replay, not a new detector pass. Each (seed, window) has its
own generator, so worker count and scheduling never change a result.
"""

import copy
import logging
import math
import multiprocessing
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from driftmon.config import AlertSettings, Settings, ThresholdsConfig
from driftmon.data import schema, split
from driftmon.detectors.base import FloatArray, TestResult
from driftmon.evaluation.kpis import AlertEvent, StreamOutcome, degradation_window
from driftmon.model import artifacts
from driftmon.model.artifacts import ModelArtifact
from driftmon.model.reference import ReferenceStats
from driftmon.monitoring.alerts import AlertManager
from driftmon.monitoring.monitor import Monitor
from driftmon.simulation import sampler
from driftmon.simulation.ground_truth import GroundTruth, build_ground_truth
from driftmon.simulation.scenarios import (
    SEVERITIES,
    ScenarioConfig,
    StreamParams,
    generate_stream,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamTask:
    scenario: ScenarioConfig
    severity: str | None
    seed: int
    params: StreamParams
    horizon: int


@dataclass(frozen=True)
class HarnessInputs:
    pool: pd.DataFrame
    simulation_stats: dict[str, tuple[float, float]]
    model: Pipeline
    reference: ReferenceStats


@dataclass(frozen=True)
class StreamRecord:
    """Everything about one stream that does not depend on thresholds."""

    task: StreamTask
    truth: GroundTruth
    tests: tuple[tuple[str, str], ...]
    statistic: FloatArray
    p_value: FloatArray
    effect: FloatArray
    error_rate: FloatArray
    rolling_auc: FloatArray

    @property
    def n_windows(self) -> int:
        return int(self.error_rate.size)

    def window_results(self, index: int, methods: frozenset[str] | None = None) -> list[TestResult]:
        results: list[TestResult] = []
        for j, (feature, method) in enumerate(self.tests):
            if methods is not None and method not in methods:
                continue
            p = float(self.p_value[index, j])
            results.append(
                TestResult(
                    feature=feature,
                    method=method,
                    statistic=float(self.statistic[index, j]),
                    p_value=None if math.isnan(p) else p,
                    effect_size=float(self.effect[index, j]),
                )
            )
        return results


def prepare_inputs(
    settings: Settings, artifact_dir: Path | None = None
) -> tuple[HarnessInputs, ModelArtifact]:
    processed = settings.processed_data_dir
    pool = split.load_split(processed, "stream_pool")
    reference_df = split.load_split(processed, "reference")
    artifact = artifacts.load_artifact(
        artifact_dir or artifacts.latest_artifact_dir(settings.artifacts_dir)
    )
    model = copy.deepcopy(artifact.model)
    # Parallelism comes from the stream workers; a multi-threaded booster in
    # each of them would only oversubscribe the cores. Per-row predictions do
    # not depend on the thread count.
    model.set_params(model__n_jobs=1)
    inputs = HarnessInputs(
        pool=pool,
        simulation_stats=sampler.reference_feature_stats(reference_df, schema.NUMERIC_FEATURES),
        model=model,
        reference=artifact.reference,
    )
    return inputs, artifact


def build_tasks(
    scenarios: dict[str, ScenarioConfig],
    severities: Sequence[str],
    seeds: Sequence[int],
    params: StreamParams,
    horizon: int,
) -> list[StreamTask]:
    tasks: list[StreamTask] = []
    for name in sorted(scenarios):
        scenario = scenarios[name]
        levels: list[str | None] = [None] if scenario.kind == "control" else [s for s in severities]
        for level in levels:
            for seed in seeds:
                tasks.append(StreamTask(scenario, level, seed, params, horizon))
    return tasks


def _scale(windows: int, factor: float, minimum: int = 1) -> int:
    # Half-up, not Python's half-to-even: 1.5 windows becomes 2 like 2.5
    # becomes 3, so rounding never depends on parity.
    return max(minimum, math.floor(windows * factor + 0.5))


def scaled_task_params(
    base: StreamParams, horizon: int, window_size: int
) -> tuple[StreamParams, int, float]:
    """Stream timing for another window size at constant traffic volume.
    Quantities that do not divide evenly are rounded and reported as such."""
    factor = base.window_size / window_size
    params = StreamParams(
        n_windows=_scale(base.n_windows, factor),
        window_size=window_size,
        drift_start=_scale(base.drift_start, factor),
        label_delay=_scale(base.label_delay, factor),
    )
    return params, _scale(horizon, factor), factor


def scaled_scenario(scenario: ScenarioConfig, factor: float) -> ScenarioConfig:
    update: dict[str, int] = {}
    if scenario.ramp_windows is not None:
        update["ramp_windows"] = _scale(scenario.ramp_windows, factor)
    if scenario.period is not None:
        update["period"] = _scale(scenario.period, factor, minimum=2)
    return scenario.model_copy(update=update)


def scaled_thresholds(thresholds: ThresholdsConfig, factor: float) -> ThresholdsConfig:
    """The alert cooldown is a span of traffic too, so it scales with the
    window size; detector thresholds stay frozen."""
    cooldown = _scale(thresholds.alerts.cooldown_windows, factor, minimum=0)
    alerts = thresholds.alerts.model_copy(update={"cooldown_windows": cooldown})
    return thresholds.model_copy(update={"alerts": alerts})


def stream_timing(
    params: StreamParams,
    horizon: int,
    thresholds: ThresholdsConfig,
    scenarios: dict[str, ScenarioConfig],
) -> dict[str, int]:
    timing = {
        "window_size": params.window_size,
        "n_windows": params.n_windows,
        "drift_start": params.drift_start,
        "label_delay": params.label_delay,
        "horizon": horizon,
        "cooldown_windows": thresholds.alerts.cooldown_windows,
    }
    for scenario in scenarios.values():
        if scenario.ramp_windows is not None:
            timing["ramp_windows"] = scenario.ramp_windows
        if scenario.period is not None:
            timing["seasonal_period"] = scenario.period
    return timing


def _rolling_auc(labels: np.ndarray, probabilities: FloatArray) -> float:
    if labels.min() == labels.max():
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def run_stream(task: StreamTask, inputs: HarnessInputs, monitor: Monitor) -> StreamRecord:
    statistic: list[list[float]] = []
    p_value: list[list[float]] = []
    effect: list[list[float]] = []
    errors: list[float] = []
    aucs: list[float] = []
    tests: tuple[tuple[str, str], ...] = ()
    previous: tuple[np.ndarray, FloatArray] | None = None

    for window in generate_stream(
        inputs.pool,
        inputs.simulation_stats,
        task.scenario,
        task.severity,
        task.seed,
        task.params,
    ):
        features = window.X[list(schema.MODEL_FEATURES)]
        probabilities = np.asarray(inputs.model.predict_proba(features)[:, 1], dtype=float)
        results = monitor.run_tests(features, probabilities)
        if not tests:
            tests = tuple((r.feature, r.method) for r in results)
        statistic.append([r.statistic for r in results])
        p_value.append([math.nan if r.p_value is None else r.p_value for r in results])
        effect.append([r.effect_size for r in results])
        errors.append(Monitor.error_rate(probabilities, window.y_true))

        if previous is None:
            aucs.append(float("nan"))
        else:
            aucs.append(
                _rolling_auc(
                    np.concatenate([previous[0], window.y_true]),
                    np.concatenate([previous[1], probabilities]),
                )
            )
        previous = (window.y_true, probabilities)

    return StreamRecord(
        task=task,
        truth=build_ground_truth(task.scenario, task.severity, task.params),
        tests=tests,
        statistic=np.asarray(statistic, dtype=float),
        p_value=np.asarray(p_value, dtype=float),
        effect=np.asarray(effect, dtype=float),
        error_rate=np.asarray(errors, dtype=float),
        rolling_auc=np.asarray(aucs, dtype=float),
    )


_WORKER_STATE: tuple[HarnessInputs, Monitor] | None = None


def _init_worker(inputs: HarnessInputs) -> None:
    global _WORKER_STATE
    _WORKER_STATE = (inputs, Monitor(inputs.reference, ThresholdsConfig()))


def _run_task(task: StreamTask) -> StreamRecord:
    assert _WORKER_STATE is not None, "worker not initialised"
    inputs, monitor = _WORKER_STATE
    return run_stream(task, inputs, monitor)


def run_streams(
    tasks: Sequence[StreamTask], inputs: HarnessInputs, n_workers: int
) -> list[StreamRecord]:
    """Records come back in task order regardless of worker count."""
    if n_workers <= 1 or len(tasks) <= 1:
        monitor = Monitor(inputs.reference, ThresholdsConfig())
        return [run_stream(task, inputs, monitor) for task in tasks]
    # spawn, not fork: forking a process that already initialised OpenMP
    # threads (XGBoost) can deadlock the children.
    context = multiprocessing.get_context("spawn")
    with context.Pool(n_workers, initializer=_init_worker, initargs=(inputs,)) as pool:
        return pool.map(_run_task, tasks, chunksize=1)


def alert_timeline(
    record: StreamRecord,
    monitor: Monitor,
    alert_settings: AlertSettings,
    *,
    inputs: bool = True,
    performance: bool = True,
    methods: frozenset[str] | None = None,
) -> tuple[AlertEvent, ...]:
    """Replay a stream through Monitor.decide, Page-Hinkley and the alert
    manager, in the live runner's order: labels that arrive at window i are
    consumed before window i's inputs are judged. Cooldown is per alert type,
    so input and performance alerts can be replayed separately."""
    monitor.reset_performance()
    manager = AlertManager(alert_settings)
    delay = record.task.params.label_delay
    window_size = record.task.params.window_size
    events: list[AlertEvent] = []
    for i in range(record.n_windows):
        labelled = i - delay
        if performance and labelled >= 0:
            error = float(record.error_rate[labelled])
            alarm = monitor.observe_error_rate(error)
            alert = manager.evaluate_performance(labelled, error, alarm)
            if alert is not None:
                events.append(AlertEvent(window=i, type=alert.type))
        if inputs:
            report = monitor.decide(i, window_size, record.window_results(i, methods))
            events.extend(
                AlertEvent(window=i, type=a.type) for a in manager.evaluate_window(report)
            )
    return tuple(events)


def to_outcome(
    record: StreamRecord,
    alerts: Iterable[AlertEvent],
    reference_auc: float,
    degradation_margin: float,
) -> StreamOutcome:
    task = record.task
    return StreamOutcome(
        scenario=task.scenario.name,
        kind=task.scenario.kind,
        severity=task.severity,
        seed=task.seed,
        t0=record.truth.t0,
        horizon=task.horizon,
        n_windows=record.n_windows,
        alerts=tuple(alerts),
        # Control streams are scanned over the same windows as drifted ones,
        # which makes their degradation rate a noise floor for harmful share.
        degradation_window=degradation_window(
            record.rolling_auc.tolist(),
            reference_auc,
            degradation_margin,
            start=task.params.drift_start,
        ),
    )


def outcomes_for(
    records: Sequence[StreamRecord],
    reference: ReferenceStats,
    thresholds: ThresholdsConfig,
    degradation_margin: float,
    *,
    inputs: bool = True,
    performance: bool = True,
    methods: frozenset[str] | None = None,
) -> list[StreamOutcome]:
    monitor = Monitor(reference, thresholds)
    return [
        to_outcome(
            record,
            alert_timeline(
                record,
                monitor,
                thresholds.alerts,
                inputs=inputs,
                performance=performance,
                methods=methods,
            ),
            reference.auc,
            degradation_margin,
        )
        for record in records
    ]


FEATURE_VIEW = frozenset({"feature_drift"})

METHOD_ARMS: dict[str, frozenset[str] | None] = {
    "all tests (frozen)": None,
    "KS only": frozenset({"ks", "prediction_ks"}),
    "PSI only": frozenset({"psi", "prediction_ks"}),
}


@dataclass(frozen=True)
class AblationArm:
    group: str
    arm: str
    view: str
    window_size: int
    outcomes: list[StreamOutcome]
    timing: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRun:
    outcomes: list[StreamOutcome]
    ablations: list[AblationArm]
    thresholds: ThresholdsConfig
    seeds: list[int]
    severities: list[str]
    params: StreamParams
    horizon: int


def _feature_view(outcomes: Sequence[StreamOutcome]) -> list[StreamOutcome]:
    return [
        replace(o, alerts=tuple(a for a in o.alerts if a.type in FEATURE_VIEW)) for o in outcomes
    ]


def decision_ablations(
    records: Sequence[StreamRecord],
    reference: ReferenceStats,
    thresholds: ThresholdsConfig,
    degradation_margin: float,
    window_size: int,
) -> list[AblationArm]:
    """Arms that only change the input-side decision. They are scored on
    feature_drift alerts alone: prediction and performance alerts are
    untouched by these arms and would mask the difference. Thresholds stay
    frozen; no arm is re-tuned."""
    covariate = [r for r in records if r.task.scenario.kind != "concept"]
    arms: list[AblationArm] = []

    def add(group: str, arm: str, config: ThresholdsConfig, methods: frozenset[str] | None) -> None:
        outcomes = outcomes_for(
            covariate,
            reference,
            config,
            degradation_margin,
            performance=False,
            methods=methods,
        )
        arms.append(AblationArm(group, arm, "feature_drift", window_size, _feature_view(outcomes)))

    for name, methods in METHOD_ARMS.items():
        add("method", name, thresholds, methods)
    # PSI has no p-value, so its threshold is its only decision rule and is
    # kept; the gate removed here is the one on KS D and Cramer's V.
    ungated = thresholds.model_copy(
        update={
            "min_effect": thresholds.min_effect.model_copy(update={"ks_d": 0.0, "cramers_v": 0.0})
        }
    )
    add("effect_gate", "with gate (frozen)", thresholds, None)
    add("effect_gate", "without KS/chi2 gate", ungated, None)
    add(
        "correction",
        "bonferroni (frozen)",
        thresholds.model_copy(update={"correction": "bonferroni"}),
        None,
    )
    add(
        "correction",
        "benjamini-hochberg",
        thresholds.model_copy(update={"correction": "benjamini-hochberg"}),
        None,
    )
    return arms


def run_evaluation(
    settings: Settings,
    inputs: HarnessInputs,
    scenarios: dict[str, ScenarioConfig],
    thresholds: ThresholdsConfig,
    *,
    smoke: bool,
    n_workers: int,
) -> EvaluationRun:
    evaluation = settings.evaluation
    seeds = settings.seeds.evaluation
    severities = list(SEVERITIES)
    if smoke:
        seeds = seeds[: evaluation.smoke_seed_count]
        severities = list(evaluation.smoke_severities)
    params = StreamParams.from_settings(settings)
    horizon = settings.stream.detection_horizon
    margin = evaluation.degradation_margin

    tasks = build_tasks(scenarios, severities, seeds, params, horizon)
    logger.info("evaluating %d streams on seeds %d-%d", len(tasks), seeds[0], seeds[-1])
    records = run_streams(tasks, inputs, n_workers)
    outcomes = outcomes_for(records, inputs.reference, thresholds, margin)

    ablations: list[AblationArm] = []
    if not smoke:
        level = evaluation.ablation_severity
        in_scope = [r for r in records if r.task.severity in (None, level)]
        ablations.extend(
            decision_ablations(in_scope, inputs.reference, thresholds, margin, params.window_size)
        )
        for window_size in evaluation.ablation_window_sizes:
            arm_params, arm_horizon, factor = scaled_task_params(params, horizon, window_size)
            arm_scenarios = {n: scaled_scenario(s, factor) for n, s in scenarios.items()}
            arm_thresholds = scaled_thresholds(thresholds, factor)
            if window_size == params.window_size:
                arm_records = in_scope
            else:
                arm_tasks = build_tasks(arm_scenarios, [level], seeds, arm_params, arm_horizon)
                logger.info("window-size ablation n=%d: %d streams", window_size, len(arm_tasks))
                arm_records = run_streams(arm_tasks, inputs, n_workers)
            ablations.append(
                AblationArm(
                    "window_size",
                    f"n={window_size}",
                    "all alerts",
                    window_size,
                    outcomes_for(arm_records, inputs.reference, arm_thresholds, margin),
                    stream_timing(arm_params, arm_horizon, arm_thresholds, arm_scenarios),
                )
            )

    return EvaluationRun(
        outcomes=outcomes,
        ablations=ablations,
        thresholds=thresholds,
        seeds=list(seeds),
        severities=severities,
        params=params,
        horizon=horizon,
    )


def detector_correctness(seed: int, n_cases: int) -> dict[str, float]:
    """Largest absolute deviation of the hand-written KS and chi-square from
    scipy over random inputs of varied size and shape."""
    from scipy.stats import chi2_contingency, ks_2samp

    from driftmon.detectors.chi2 import pearson_chi2
    from driftmon.detectors.ks import ks_test

    rng = np.random.default_rng(seed)
    worst = {"ks_statistic": 0.0, "ks_p_value": 0.0, "chi2_statistic": 0.0, "chi2_p_value": 0.0}
    for case in range(n_cases):
        n_ref, n_cur = (int(v) for v in rng.integers(20, 3000, size=2))
        shape = case % 3
        if shape == 0:
            ref, cur = rng.normal(0, 1, n_ref), rng.normal(rng.normal(0, 0.3), 1, n_cur)
        elif shape == 1:
            ref, cur = rng.exponential(1, n_ref), rng.exponential(rng.uniform(0.7, 1.3), n_cur)
        else:
            ref = rng.integers(0, 12, n_ref).astype(float)
            cur = rng.integers(0, 12, n_cur).astype(float)
        mine = ks_test(ref, cur, "x")
        theirs = ks_2samp(ref, cur, method="asymp")
        if mine.p_value is None:
            raise RuntimeError("KS returned no p-value")
        worst["ks_statistic"] = max(worst["ks_statistic"], abs(mine.statistic - theirs.statistic))
        worst["ks_p_value"] = max(worst["ks_p_value"], abs(mine.p_value - theirs.pvalue))

        k = int(rng.integers(2, 12))
        table = rng.integers(5, 2000, size=(k, 2)).astype(float)
        statistic, p_value, _ = pearson_chi2(table)
        reference = chi2_contingency(table, correction=False)
        worst["chi2_statistic"] = max(
            worst["chi2_statistic"], abs(statistic - float(reference.statistic))
        )
        worst["chi2_p_value"] = max(worst["chi2_p_value"], abs(p_value - float(reference.pvalue)))
    return worst
