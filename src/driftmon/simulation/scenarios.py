"""Scenario definitions (loaded from YAML) and the windowed stream generator."""

import math
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, ValidationError, model_validator

from driftmon.config import ConfigError, Settings
from driftmon.data import schema
from driftmon.simulation import sampler

ScenarioKind = Literal["control", "sudden_covariate", "gradual_covariate", "concept", "seasonal"]

SEVERITIES = ("low", "medium", "high")


class SeverityParams(BaseModel):
    beta: float | None = None
    p: float | None = None


class SubgroupRule(BaseModel):
    feature: str
    min: float


class ScenarioConfig(BaseModel):
    name: str
    kind: ScenarioKind
    feature: str | None = None
    ramp_windows: int | None = None
    period: int | None = None
    subgroup: list[SubgroupRule] | None = None
    severity: dict[str, SeverityParams] = {}

    @model_validator(mode="after")
    def _kind_specific_fields(self) -> "ScenarioConfig":
        if self.kind in ("sudden_covariate", "gradual_covariate", "seasonal"):
            if self.feature is None:
                raise ValueError(f"scenario kind {self.kind!r} requires a feature")
            for level, params in self.severity.items():
                if params.beta is None:
                    raise ValueError(f"severity {level!r} requires beta for {self.kind!r}")
        if self.kind == "gradual_covariate" and self.ramp_windows is None:
            raise ValueError("gradual_covariate requires ramp_windows")
        if self.kind == "seasonal" and self.period is None:
            raise ValueError("seasonal requires period")
        if self.kind == "concept":
            if not self.subgroup:
                raise ValueError("concept requires a subgroup definition")
            for level, params in self.severity.items():
                if params.p is None:
                    raise ValueError(f"severity {level!r} requires p for concept")
        if self.kind != "control" and not self.severity:
            raise ValueError(f"scenario {self.name!r} requires severity levels")
        return self


def load_scenario(path: Path) -> ScenarioConfig:
    if not path.is_file():
        raise ConfigError(f"scenario file not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"scenario file is not valid YAML: {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"scenario file must contain a mapping: {path}")
    try:
        return ScenarioConfig(**raw)
    except ValidationError as exc:
        raise ConfigError(f"invalid scenario in {path}: {exc}") from exc


def load_scenarios(scenarios_dir: Path) -> dict[str, ScenarioConfig]:
    paths = sorted(scenarios_dir.glob("*.yaml"))
    if not paths:
        raise ConfigError(f"no scenario files found in {scenarios_dir}")
    scenarios = {s.name: s for s in (load_scenario(p) for p in paths)}
    return scenarios


@dataclass(frozen=True)
class StreamParams:
    n_windows: int
    window_size: int
    drift_start: int
    label_delay: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "StreamParams":
        return cls(
            n_windows=settings.stream.n_windows,
            window_size=settings.window_size,
            drift_start=settings.stream.drift_start,
            label_delay=settings.label_delay,
        )


@dataclass(frozen=True)
class Window:
    index: int
    X: pd.DataFrame
    y_true: "np.ndarray[tuple[int], np.dtype[np.int64]]"
    label_available_at: int


def ramp_fraction(window_index: int, drift_start: int, ramp_windows: int) -> float:
    """Shifted share of a gradual window: 1/K at drift start, 1 from
    drift_start + K - 1 on."""
    if window_index < drift_start:
        return 0.0
    return min(1.0, (window_index - drift_start + 1) / ramp_windows)


def seasonal_beta(window_index: int, drift_start: int, period: int, amplitude: float) -> float:
    if window_index < drift_start:
        return 0.0
    return amplitude * math.sin(2.0 * math.pi * (window_index - drift_start) / period)


def subgroup_mask(
    X: pd.DataFrame, rules: list[SubgroupRule]
) -> "np.ndarray[tuple[int], np.dtype[np.bool_]]":
    mask = np.ones(len(X), dtype=bool)
    for rule in rules:
        mask &= X[rule.feature].to_numpy(dtype=float) >= rule.min
    return mask


def generate_stream(
    pool: pd.DataFrame,
    reference_stats: dict[str, tuple[float, float]],
    scenario: ScenarioConfig,
    severity: str | None,
    seed: int,
    params: StreamParams,
) -> Iterator[Window]:
    """Yield the windows of one simulated stream.

    Row selection depends only on (seed, window index) and the sampling
    branch taken, so: all scenarios are identical to control before
    drift_start, and concept streams share control's feature matrix
    bit-for-bit for the whole stream.
    """
    if scenario.kind != "control" and (severity is None or severity not in scenario.severity):
        raise ValueError(
            f"scenario {scenario.name!r} needs a severity out of "
            f"{sorted(scenario.severity)}, got {severity!r}"
        )
    pool_labels = (pool[schema.LABEL] == schema.POSITIVE_LABEL).to_numpy(dtype=np.int64)
    features = pool.drop(columns=[schema.LABEL]).reset_index(drop=True)
    pool_size = len(pool)

    z_scores = None
    if scenario.feature is not None:
        mean, std = reference_stats[scenario.feature]
        z_scores = sampler.standardize(pool[scenario.feature].to_numpy(dtype=float), mean, std)

    for index in range(params.n_windows):
        rows_rng = sampler.window_rng(seed, index, "rows")
        drifted = index >= params.drift_start and scenario.kind != "control"

        if not drifted or scenario.kind == "concept":
            idx = sampler.sample_uniform(rows_rng, pool_size, params.window_size)
        elif scenario.kind == "sudden_covariate":
            assert z_scores is not None and severity is not None
            beta = scenario.severity[severity].beta or 0.0
            probabilities = sampler.importance_probabilities(z_scores, beta)
            idx = sampler.sample_weighted(rows_rng, probabilities, params.window_size)
        elif scenario.kind == "gradual_covariate":
            assert z_scores is not None and severity is not None
            assert scenario.ramp_windows is not None
            beta = scenario.severity[severity].beta or 0.0
            fraction = ramp_fraction(index, params.drift_start, scenario.ramp_windows)
            n_shifted = round(fraction * params.window_size)
            probabilities = sampler.importance_probabilities(z_scores, beta)
            shifted = sampler.sample_weighted(rows_rng, probabilities, n_shifted)
            original = sampler.sample_uniform(rows_rng, pool_size, params.window_size - n_shifted)
            idx = np.concatenate([shifted, original])
        else:  # seasonal
            assert z_scores is not None and severity is not None
            assert scenario.period is not None
            amplitude = scenario.severity[severity].beta or 0.0
            beta = seasonal_beta(index, params.drift_start, scenario.period, amplitude)
            probabilities = sampler.importance_probabilities(z_scores, beta)
            idx = sampler.sample_weighted(rows_rng, probabilities, params.window_size)

        X = features.iloc[idx].reset_index(drop=True)
        y = pool_labels[idx].copy()

        if scenario.kind == "concept" and drifted:
            assert scenario.subgroup is not None and severity is not None
            p_flip = scenario.severity[severity].p or 0.0
            labels_rng = sampler.window_rng(seed, index, "labels")
            flips = subgroup_mask(X, scenario.subgroup) & (
                labels_rng.random(params.window_size) < p_flip
            )
            y[flips] = 1 - y[flips]

        yield Window(
            index=index,
            X=X,
            y_true=y,
            label_available_at=index + params.label_delay,
        )
