"""Detection KPIs as pure functions over alert timelines and ground truth.

Windows are indexed by when an alert is raised. A performance alert about
window w is raised at w + L, when that window's labels arrive, so concept
detection can never beat the label delay.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

INPUT_ALERT_TYPES = frozenset({"feature_drift", "prediction_drift"})
ALL_ALERT_TYPES = INPUT_ALERT_TYPES | {"performance_drift"}


@dataclass(frozen=True)
class AlertEvent:
    window: int
    type: str


@dataclass(frozen=True)
class StreamOutcome:
    scenario: str
    kind: str
    severity: str | None
    seed: int
    t0: int | None
    horizon: int
    n_windows: int
    alerts: tuple[AlertEvent, ...]
    degradation_window: int | None = None


@dataclass(frozen=True)
class Estimate:
    value: float | None
    ci_low: float | None
    ci_high: float | None
    n: int

    def to_dict(self) -> dict[str, Any]:
        return {"value": self.value, "ci_low": self.ci_low, "ci_high": self.ci_high, "n": self.n}


def relevant_types(kind: str) -> frozenset[str]:
    """Concept drift leaves P(x) untouched, so only the label-based detector
    can legitimately catch it; an input alert there is a false alarm."""
    if kind == "concept":
        return frozenset({"performance_drift"})
    return ALL_ALERT_TYPES


def correct_alerts(outcome: StreamOutcome) -> list[AlertEvent]:
    if outcome.t0 is None:
        return []
    end = outcome.t0 + outcome.horizon
    types = relevant_types(outcome.kind)
    return [a for a in outcome.alerts if outcome.t0 <= a.window < end and a.type in types]


def first_detection(outcome: StreamOutcome) -> int | None:
    hits = correct_alerts(outcome)
    return min(a.window for a in hits) if hits else None


def time_to_detect(outcome: StreamOutcome) -> int | None:
    first = first_detection(outcome)
    if first is None or outcome.t0 is None:
        return None
    return first - outcome.t0


def alert_counts(outcome: StreamOutcome) -> tuple[int, int]:
    """(correct, wrong) alerts of a drifted stream. Alerts before t0 are
    wrong; alerts past the horizon are ignored."""
    if outcome.t0 is None:
        raise ValueError("alert precision is only defined for drifted streams")
    end = outcome.t0 + outcome.horizon
    types = relevant_types(outcome.kind)
    correct = wrong = 0
    for alert in outcome.alerts:
        if alert.window >= end:
            continue
        if alert.window >= outcome.t0 and alert.type in types:
            correct += 1
        else:
            wrong += 1
    return correct, wrong


def has_alert(outcome: StreamOutcome, types: frozenset[str] = ALL_ALERT_TYPES) -> bool:
    return any(a.type in types for a in outcome.alerts)


def degradation_window(
    rolling_auc: Sequence[float], reference_auc: float, margin: float, start: int
) -> int | None:
    """First window from `start` where the rolling AUC sits below
    reference - margin for two consecutive windows. NaN never counts."""
    limit = reference_auc - margin
    values = list(rolling_auc)
    for i in range(start, len(values) - 1):
        if values[i] < limit and values[i + 1] < limit:
            return i
    return None


def lead_time(outcome: StreamOutcome) -> int | None:
    """Degradation window minus the first relevant alert at or after t0.
    Positive means the monitor warned before the model measurably broke."""
    if outcome.degradation_window is None or outcome.t0 is None:
        return None
    types = relevant_types(outcome.kind)
    after = [a.window for a in outcome.alerts if a.window >= outcome.t0 and a.type in types]
    if not after:
        return None
    return outcome.degradation_window - min(after)


def bootstrap(
    data: "np.ndarray[Any, Any]",
    statistic: Callable[["np.ndarray[Any, Any]"], float],
    samples: int,
    seed: int,
) -> Estimate:
    """Point estimate plus a 95% percentile interval from resampling rows
    (streams). Resamples where the statistic is undefined (NaN) are dropped."""
    n = len(data)
    if n == 0:
        return Estimate(value=None, ci_low=None, ci_high=None, n=0)
    point = statistic(data)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, n, size=(samples, n))
    stats = np.array([statistic(data[idx]) for idx in draws], dtype=float)
    stats = stats[~np.isnan(stats)]
    value = None if np.isnan(point) else float(point)
    if stats.size == 0:
        return Estimate(value=value, ci_low=None, ci_high=None, n=n)
    low, high = np.percentile(stats, [2.5, 97.5])
    return Estimate(value=value, ci_low=float(low), ci_high=float(high), n=n)


def _mean(values: "np.ndarray[Any, Any]") -> float:
    return float(np.mean(values))


def _nanmedian(values: "np.ndarray[Any, Any]") -> float:
    finite = values[~np.isnan(values)]
    return float(np.median(finite)) if finite.size else float("nan")


def _ratio(pairs: "np.ndarray[Any, Any]") -> float:
    totals = pairs.sum(axis=0)
    return float(totals[0] / totals[1]) if totals[1] > 0 else float("nan")


def detection_rate(outcomes: Sequence[StreamOutcome], samples: int, seed: int) -> Estimate:
    data = np.array([first_detection(o) is not None for o in outcomes], dtype=float)
    return bootstrap(data, _mean, samples, seed)


def median_ttd(outcomes: Sequence[StreamOutcome], samples: int, seed: int) -> Estimate:
    ttds = [time_to_detect(o) for o in outcomes]
    data = np.array([np.nan if t is None else float(t) for t in ttds], dtype=float)
    return bootstrap(data, _nanmedian, samples, seed)


def alert_precision(outcomes: Sequence[StreamOutcome], samples: int, seed: int) -> Estimate:
    counts = [alert_counts(o) for o in outcomes]
    data = np.array([(c, c + w) for c, w in counts], dtype=float)
    return bootstrap(data, _ratio, samples, seed)


def stream_false_alarm_rate(
    outcomes: Sequence[StreamOutcome],
    samples: int,
    seed: int,
    types: frozenset[str] = ALL_ALERT_TYPES,
) -> Estimate:
    data = np.array([has_alert(o, types) for o in outcomes], dtype=float)
    return bootstrap(data, _mean, samples, seed)


def per_window_false_alarm_rate(
    outcomes: Sequence[StreamOutcome],
    samples: int,
    seed: int,
    types: frozenset[str] = ALL_ALERT_TYPES,
) -> Estimate:
    data = np.array(
        [(sum(a.type in types for a in o.alerts), o.n_windows) for o in outcomes], dtype=float
    )
    return bootstrap(data, _ratio, samples, seed)


def harmful_share(outcomes: Sequence[StreamOutcome], samples: int, seed: int) -> Estimate:
    data = np.array([o.degradation_window is not None for o in outcomes], dtype=float)
    return bootstrap(data, _mean, samples, seed)


def median_lead_time(outcomes: Sequence[StreamOutcome], samples: int, seed: int) -> Estimate:
    """Over harmful streams only; a harmful stream the monitor never alerted
    on counts in n but has no lead time."""
    harmful = [o for o in outcomes if o.degradation_window is not None]
    leads = [lead_time(o) for o in harmful]
    data = np.array([np.nan if t is None else float(t) for t in leads], dtype=float)
    return bootstrap(data, _nanmedian, samples, seed)


def paired_rate_difference(
    treated: Sequence[StreamOutcome],
    control: Sequence[StreamOutcome],
    types: frozenset[str],
    samples: int,
    seed: int,
) -> Estimate:
    """Alert-rate difference between two stream sets sharing seeds,
    bootstrapped over seed pairs."""
    by_seed = {o.seed: o for o in control}
    pairs = [(t, by_seed[t.seed]) for t in treated if t.seed in by_seed]
    data = np.array([float(has_alert(t, types)) - float(has_alert(c, types)) for t, c in pairs])
    return bootstrap(data, _mean, samples, seed)
