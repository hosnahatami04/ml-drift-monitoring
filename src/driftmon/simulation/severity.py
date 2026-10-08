"""Measure what each severity level actually does: feature shift in reference
standard deviations, and the model's AUC on drifted windows.

Runs on calibration seeds only; evaluation seeds stay untouched until the KPI
report.
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from driftmon.data import schema
from driftmon.simulation.ground_truth import build_ground_truth
from driftmon.simulation.scenarios import (
    SEVERITIES,
    ScenarioConfig,
    StreamParams,
    generate_stream,
    subgroup_mask,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SeverityMeasurement:
    scenario: str
    severity: str
    mean_abs_shift_sd: float | None
    realized_flip_rate: float | None
    auc_drifted: float
    auc_delta: float


def _collect_active_windows(
    pool: pd.DataFrame,
    reference_stats: dict[str, tuple[float, float]],
    scenario: ScenarioConfig,
    severity: str,
    seed: int,
    params: StreamParams,
) -> tuple[pd.DataFrame, "np.ndarray[tuple[int], np.dtype[np.int64]]", list[pd.DataFrame]]:
    truth = build_ground_truth(scenario, severity, params)
    active = set(truth.active_windows)
    frames: list[pd.DataFrame] = []
    labels: list[np.ndarray] = []
    per_window: list[pd.DataFrame] = []
    for window in generate_stream(pool, reference_stats, scenario, severity, seed, params):
        if window.index in active:
            frames.append(window.X)
            labels.append(window.y_true)
            per_window.append(window.X)
    X = pd.concat(frames, ignore_index=True)
    y = np.concatenate(labels)
    return X, y, per_window


def measure_severity(
    pool: pd.DataFrame,
    reference_stats: dict[str, tuple[float, float]],
    scenario: ScenarioConfig,
    severity: str,
    seeds: list[int],
    params: StreamParams,
    model: Pipeline,
    reference_auc: float,
) -> SeverityMeasurement:
    shifts: list[float] = []
    flip_rates: list[float] = []
    aucs: list[float] = []

    for seed in seeds:
        X, y, per_window = _collect_active_windows(
            pool, reference_stats, scenario, severity, seed, params
        )

        if scenario.feature is not None:
            mean, std = reference_stats[scenario.feature]
            window_shifts = [
                abs(float(w[scenario.feature].mean()) - mean) / std for w in per_window
            ]
            shifts.append(float(np.mean(window_shifts)))

        if scenario.kind == "concept":
            # Same seed and row substream: the control twin shares row draws,
            # so any label disagreement is exactly the injected flips. The
            # rate is reported within the subgroup, where p applies.
            assert scenario.subgroup is not None
            control = ScenarioConfig(name="control", kind="control")
            twin_labels = np.concatenate(
                [
                    w.y_true
                    for w in generate_stream(pool, reference_stats, control, None, seed, params)
                    if w.index >= params.drift_start
                ]
            )
            in_subgroup = subgroup_mask(X, scenario.subgroup)
            flip_rates.append(float(np.mean(y[in_subgroup] != twin_labels[in_subgroup])))

        probs = model.predict_proba(X[list(schema.MODEL_FEATURES)])[:, 1]
        aucs.append(float(roc_auc_score(y, probs)))

    auc_drifted = float(np.mean(aucs))
    return SeverityMeasurement(
        scenario=scenario.name,
        severity=severity,
        mean_abs_shift_sd=float(np.mean(shifts)) if shifts else None,
        realized_flip_rate=float(np.mean(flip_rates)) if flip_rates else None,
        auc_drifted=auc_drifted,
        auc_delta=auc_drifted - reference_auc,
    )


def measure_all(
    pool: pd.DataFrame,
    reference_stats: dict[str, tuple[float, float]],
    scenarios: dict[str, ScenarioConfig],
    seeds: list[int],
    params: StreamParams,
    model: Pipeline,
    reference_auc: float,
) -> list[SeverityMeasurement]:
    measurements: list[SeverityMeasurement] = []
    for name in sorted(scenarios):
        scenario = scenarios[name]
        if scenario.kind == "control":
            continue
        for severity in SEVERITIES:
            logger.info("measuring %s / %s", name, severity)
            measurements.append(
                measure_severity(
                    pool, reference_stats, scenario, severity, seeds, params, model, reference_auc
                )
            )
    return measurements


def write_severity_report(
    path: Path,
    measurements: list[SeverityMeasurement],
    reference_auc: float,
    seeds: list[int],
) -> None:
    lines = [
        "# Severity calibration",
        "",
        f"Measured on calibration seeds {seeds[0]}-{seeds[-1]} "
        f"({len(seeds)} streams per cell), active drifted windows only.",
        f"Reference AUC: {reference_auc:.4f}.",
        "",
        "Shift is the mean absolute displacement of the affected feature's",
        "window mean, in reference standard deviations. Flip rate is the",
        "realized label-flip share among drifted-window rows (concept only).",
        "",
        "| Scenario | Severity | Shift (sd) | Flip rate | AUC (drifted) | AUC delta |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for m in measurements:
        shift = f"{m.mean_abs_shift_sd:.3f}" if m.mean_abs_shift_sd is not None else "-"
        flip = f"{m.realized_flip_rate:.3f}" if m.realized_flip_rate is not None else "-"
        lines.append(
            f"| {m.scenario} | {m.severity} | {shift} | {flip} "
            f"| {m.auc_drifted:.4f} | {m.auc_delta:+.4f} |"
        )
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote severity report to %s", path)
