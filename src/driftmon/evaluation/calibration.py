"""Threshold calibration on calibration seeds only.

Objective: the highest mean detection rate over the drift scenarios at the
calibration severity, subject to a stream-level false alarm rate on control
streams at or below the configured ceiling. Ties go to the faster detector,
then to fewer false alarms.

Input-side thresholds and Page-Hinkley parameters are scored separately and
combined afterwards: the alert manager's cooldown is per alert type, so the
two alert families never interact and a stream's outcome under a combined
configuration is exactly the union of its two timelines.
"""

import itertools
import logging
import multiprocessing
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from driftmon.config import (
    CalibrationGrid,
    EffectGates,
    PageHinkleySettings,
    SeedsConfig,
    ThresholdsConfig,
)
from driftmon.evaluation import kpis
from driftmon.evaluation.harness import StreamRecord, alert_timeline, to_outcome
from driftmon.model.reference import ReferenceStats
from driftmon.monitoring.monitor import Monitor

logger = logging.getLogger(__name__)


class CalibrationError(RuntimeError):
    pass


@dataclass(frozen=True)
class InputCandidate:
    alpha: float
    ks_d: float
    psi: float
    cramers_v: float
    k_features: int


@dataclass(frozen=True)
class PerformanceCandidate:
    delta: float
    lambda_: float


@dataclass(frozen=True)
class CandidateScore:
    """Per-stream outcomes of one candidate: whether each control stream
    alerted and how many alerts it raised, and the first correct detection
    window of each drifted stream (inf when missed)."""

    control_alarm: np.ndarray
    control_alerts: np.ndarray
    first_detection: np.ndarray


@dataclass(frozen=True)
class CalibrationResult:
    thresholds: ThresholdsConfig
    feasible: bool
    detection_rate: dict[str, float]
    stream_false_alarm_rate: float
    n_input_candidates: int
    n_performance_candidates: int

    def summary(self) -> dict[str, Any]:
        return {
            "feasible": self.feasible,
            "detection_rate": self.detection_rate,
            "stream_false_alarm_rate": self.stream_false_alarm_rate,
            "n_input_candidates": self.n_input_candidates,
            "n_performance_candidates": self.n_performance_candidates,
            "thresholds": thresholds_document(self.thresholds),
        }


def ensure_calibration_seeds(seeds: Sequence[int], config: SeedsConfig) -> None:
    leaked = sorted(set(seeds) & set(config.evaluation))
    if leaked:
        raise CalibrationError(f"calibration must never touch evaluation seeds, got {leaked}")


def input_candidates(grid: CalibrationGrid) -> list[InputCandidate]:
    return [
        InputCandidate(alpha=a, ks_d=d, psi=p, cramers_v=v, k_features=k)
        for a, d, p, v, k in itertools.product(
            grid.alpha, grid.ks_d, grid.psi, grid.cramers_v, grid.k_features
        )
    ]


def performance_candidates(grid: CalibrationGrid) -> list[PerformanceCandidate]:
    return [
        PerformanceCandidate(delta=d, lambda_=lam)
        for d, lam in itertools.product(grid.ph_delta, grid.ph_lambda)
    ]


def apply_candidates(
    base: ThresholdsConfig,
    inputs: InputCandidate | None = None,
    performance: PerformanceCandidate | None = None,
) -> ThresholdsConfig:
    update: dict[str, Any] = {}
    if inputs is not None:
        update["alpha"] = inputs.alpha
        update["k_features"] = inputs.k_features
        update["min_effect"] = EffectGates(
            ks_d=inputs.ks_d, psi=inputs.psi, cramers_v=inputs.cramers_v
        )
    if performance is not None:
        update["page_hinkley"] = PageHinkleySettings.model_validate(
            {"delta": performance.delta, "lambda": performance.lambda_}
        )
    return base.model_copy(update=update)


def score_candidate(
    records: Sequence[StreamRecord],
    reference: ReferenceStats,
    thresholds: ThresholdsConfig,
    *,
    inputs: bool,
    performance: bool,
) -> CandidateScore:
    monitor = Monitor(reference, thresholds)
    control_alarm: list[bool] = []
    control_alerts: list[int] = []
    first: list[float] = []
    for record in records:
        events = alert_timeline(
            record, monitor, thresholds.alerts, inputs=inputs, performance=performance
        )
        # Degradation is irrelevant to the objective; skip computing it.
        outcome = to_outcome(record, events, reference_auc=0.0, degradation_margin=1.0)
        if outcome.t0 is None:
            control_alarm.append(kpis.has_alert(outcome))
            control_alerts.append(len(outcome.alerts))
        else:
            hit = kpis.first_detection(outcome)
            first.append(np.inf if hit is None else float(hit))
    return CandidateScore(
        control_alarm=np.asarray(control_alarm, dtype=bool),
        control_alerts=np.asarray(control_alerts, dtype=np.int64),
        first_detection=np.asarray(first, dtype=float),
    )


_WORKER_STATE: tuple[list[StreamRecord], ReferenceStats, ThresholdsConfig] | None = None


def _init_worker(
    records: list[StreamRecord], reference: ReferenceStats, base: ThresholdsConfig
) -> None:
    global _WORKER_STATE
    _WORKER_STATE = (records, reference, base)


def _score_input(candidate: InputCandidate) -> CandidateScore:
    assert _WORKER_STATE is not None
    records, reference, base = _WORKER_STATE
    thresholds = apply_candidates(base, inputs=candidate)
    return score_candidate(records, reference, thresholds, inputs=True, performance=False)


def _score_performance(candidate: PerformanceCandidate) -> CandidateScore:
    assert _WORKER_STATE is not None
    records, reference, base = _WORKER_STATE
    thresholds = apply_candidates(base, performance=candidate)
    return score_candidate(records, reference, thresholds, inputs=False, performance=True)


def _score_all(
    records: list[StreamRecord],
    reference: ReferenceStats,
    base: ThresholdsConfig,
    inputs: list[InputCandidate],
    performance: list[PerformanceCandidate],
    n_workers: int,
) -> tuple[list[CandidateScore], list[CandidateScore]]:
    if n_workers <= 1:
        _init_worker(records, reference, base)
        return [_score_input(c) for c in inputs], [_score_performance(c) for c in performance]
    context = multiprocessing.get_context("spawn")
    with context.Pool(
        n_workers, initializer=_init_worker, initargs=(records, reference, base)
    ) as pool:
        input_scores = pool.map(_score_input, inputs, chunksize=4)
        performance_scores = pool.map(_score_performance, performance, chunksize=1)
    return input_scores, performance_scores


def select(
    input_scores: Sequence[CandidateScore],
    performance_scores: Sequence[CandidateScore],
    drift_scenarios: Sequence[str],
    t0: int,
    horizon: int,
    max_false_alarm_rate: float,
) -> tuple[int, int, bool, dict[str, float], float]:
    """Pick the best (input, performance) pair. `drift_scenarios` labels each
    drifted stream, aligned with first_detection."""
    fa_in = np.stack([s.control_alarm for s in input_scores])
    fa_ph = np.stack([s.control_alarm for s in performance_scores])
    n_in = np.stack([s.control_alerts for s in input_scores])
    n_ph = np.stack([s.control_alerts for s in performance_scores])
    first_in = np.stack([s.first_detection for s in input_scores])
    first_ph = np.stack([s.first_detection for s in performance_scores])

    false_alarm = (fa_in[:, None, :] | fa_ph[None, :, :]).mean(axis=-1)
    alert_count = n_in.sum(axis=-1)[:, None] + n_ph.sum(axis=-1)[None, :]
    first = np.minimum(first_in[:, None, :], first_ph[None, :, :])
    detected = np.isfinite(first)
    ttd = np.where(detected, first - t0, np.nan)

    labels = np.asarray(drift_scenarios)
    names = sorted(set(drift_scenarios))
    rates = np.stack([detected[..., labels == name].mean(axis=-1) for name in names])
    with warnings.catch_warnings():
        # All-NaN slices (scenario never detected) are expected here.
        warnings.simplefilter("ignore", RuntimeWarning)
        medians = np.stack([np.nanmedian(ttd[..., labels == name], axis=-1) for name in names])
    mean_rate = rates.mean(axis=0)
    # A scenario never detected contributes the worst possible delay.
    mean_ttd = np.where(np.isnan(medians), float(horizon), medians).mean(axis=0)

    feasible = false_alarm <= max_false_alarm_rate + 1e-12
    any_feasible = bool(feasible.any())
    if not any_feasible:
        logger.warning(
            "no candidate meets the false alarm ceiling %.3f; taking the lowest",
            max_false_alarm_rate,
        )
        feasible = false_alarm == false_alarm.min()

    flat = np.flatnonzero(feasible.ravel())
    keys = (
        -mean_rate.ravel()[flat],
        mean_ttd.ravel()[flat],
        false_alarm.ravel()[flat],
        alert_count.ravel()[flat],
        flat,
    )
    best = flat[np.lexsort(keys[::-1])[0]]
    i, j = np.unravel_index(best, false_alarm.shape)
    detection = {name: float(rates[k, i, j]) for k, name in enumerate(names)}
    return int(i), int(j), any_feasible, detection, float(false_alarm[i, j])


def calibrate(
    records: list[StreamRecord],
    reference: ReferenceStats,
    base: ThresholdsConfig,
    grid: CalibrationGrid,
    seeds_config: SeedsConfig,
    n_workers: int,
) -> CalibrationResult:
    ensure_calibration_seeds([r.task.seed for r in records], seeds_config)
    if not any(r.truth.t0 is None for r in records):
        raise CalibrationError("calibration needs control streams")
    timing = {(r.truth.t0, r.task.horizon) for r in records if r.truth.t0 is not None}
    if len(timing) != 1:
        raise CalibrationError("calibration streams must share one drift start and horizon")
    ((t0, horizon),) = timing
    assert t0 is not None

    inputs = input_candidates(grid)
    performance = performance_candidates(grid)
    logger.info(
        "scoring %d input and %d Page-Hinkley candidates on %d streams",
        len(inputs),
        len(performance),
        len(records),
    )
    input_scores, performance_scores = _score_all(
        records, reference, base, inputs, performance, n_workers
    )
    drift_scenarios = [r.task.scenario.name for r in records if r.truth.t0 is not None]
    i, j, feasible, detection, false_alarm = select(
        input_scores,
        performance_scores,
        drift_scenarios,
        t0,
        horizon,
        grid.max_stream_false_alarm_rate,
    )
    thresholds = apply_candidates(base, inputs[i], performance[j])
    return CalibrationResult(
        thresholds=thresholds,
        feasible=feasible,
        detection_rate=detection,
        stream_false_alarm_rate=false_alarm,
        n_input_candidates=len(inputs),
        n_performance_candidates=len(performance),
    )


def thresholds_document(thresholds: ThresholdsConfig) -> dict[str, Any]:
    return thresholds.model_dump(by_alias=True, mode="json")


def write_thresholds(path: Path, thresholds: ThresholdsConfig, seeds: Sequence[int]) -> None:
    header = (
        f"# Frozen by `driftmon calibrate` on calibration seeds {seeds[0]}-{seeds[-1]}.\n"
        "# Do not edit by hand: re-run calibration instead (ADR 0008).\n"
    )
    body = yaml.safe_dump(thresholds_document(thresholds), sort_keys=False)
    path.write_text(header + body, encoding="utf-8")
