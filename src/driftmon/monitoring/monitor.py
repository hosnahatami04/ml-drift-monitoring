"""The monitoring core shared by the live runner and the evaluation harness.

It does no IO, so the harness can score the same detectors and decision rule
the service runs.
"""

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from driftmon.config import ThresholdsConfig
from driftmon.detectors.base import FloatArray, TestResult
from driftmon.detectors.chi2 import chi2_from_counts
from driftmon.detectors.ks import ks_test, prediction_drift_test
from driftmon.detectors.multiple_testing import DecisionConfig, decide_window
from driftmon.detectors.page_hinkley import PageHinkley
from driftmon.detectors.psi import psi_categorical, psi_numeric
from driftmon.model.reference import MISSING_CATEGORY, ReferenceStats


def decision_config_from(thresholds: ThresholdsConfig) -> DecisionConfig:
    return DecisionConfig(
        alpha=thresholds.alpha,
        correction=thresholds.correction,
        k_features=thresholds.k_features,
        min_ks_d=thresholds.min_effect.ks_d,
        min_psi=thresholds.min_effect.psi,
        min_cramers_v=thresholds.min_effect.cramers_v,
    )


@dataclass(frozen=True)
class WindowReport:
    window_id: int
    n_rows: int
    results: tuple[TestResult, ...]
    flagged_features: tuple[str, ...]
    feature_drift: bool
    prediction_drift: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "n_rows": self.n_rows,
            "results": [
                {
                    "feature": r.feature,
                    "method": r.method,
                    "statistic": r.statistic,
                    "p_value": r.p_value,
                    "effect_size": r.effect_size,
                    "flagged": r.flagged,
                }
                for r in self.results
            ],
            "flagged_features": list(self.flagged_features),
            "feature_drift": self.feature_drift,
            "prediction_drift": self.prediction_drift,
        }


class Monitor:
    def __init__(self, reference: ReferenceStats, thresholds: ThresholdsConfig) -> None:
        self.reference = reference
        self.thresholds = thresholds
        self.decision_config = decision_config_from(thresholds)
        self.page_hinkley = PageHinkley(
            delta=thresholds.page_hinkley.delta,
            lambda_=thresholds.page_hinkley.lambda_,
        )
        self._reference_numeric: dict[str, FloatArray] = {
            name: np.asarray(stats.sample, dtype=float) for name, stats in reference.numeric.items()
        }
        self._reference_probs: FloatArray = np.asarray(reference.prediction_probs, dtype=float)

    def process_window(
        self,
        window_id: int,
        features: pd.DataFrame,
        probabilities: FloatArray,
    ) -> WindowReport:
        results = self.run_tests(features, probabilities)
        return self.decide(window_id, len(features), results)

    def run_tests(self, features: pd.DataFrame, probabilities: FloatArray) -> list[TestResult]:
        """The expensive, threshold-independent half of process_window. Split
        out so calibration can score many threshold sets against one pass of
        the detectors."""
        if len(features) == 0:
            raise ValueError("cannot monitor an empty window")
        if len(features) != probabilities.size:
            raise ValueError("features and probabilities must have the same length")

        results: list[TestResult] = []
        for name, stats in self.reference.numeric.items():
            current = features[name].to_numpy(dtype=float)
            results.append(ks_test(self._reference_numeric[name], current, name))
            results.append(
                psi_numeric(
                    self._reference_numeric[name],
                    current,
                    name,
                    bin_edges=stats.bin_edges,
                )
            )
        for name, stats_cat in self.reference.categorical.items():
            current_values = (
                features[name].fillna(MISSING_CATEGORY).astype(str).to_numpy(dtype=object)
            )
            results.append(psi_categorical(stats_cat.frequencies, current_values, name))
            ref_counts = {
                category: frequency * self.reference.n_rows
                for category, frequency in stats_cat.frequencies.items()
            }
            cur_counts = Counter(current_values.tolist())
            results.append(chi2_from_counts(ref_counts, dict(cur_counts), name))
        results.append(prediction_drift_test(self._reference_probs, probabilities))
        return results

    def decide(self, window_id: int, n_rows: int, results: Sequence[TestResult]) -> WindowReport:
        decision = decide_window(results, self.decision_config)
        prediction_flagged = any(
            r.flagged and r.method == "prediction_ks" for r in decision.results
        )
        input_features = tuple(f for f in decision.flagged_features if f != "__prediction__")
        return WindowReport(
            window_id=window_id,
            n_rows=n_rows,
            results=decision.results,
            flagged_features=input_features,
            feature_drift=len(input_features) >= self.decision_config.k_features,
            prediction_drift=prediction_flagged,
        )

    def observe_error_rate(self, error_rate: float) -> bool:
        """Feed one fully labelled window's error rate to Page-Hinkley, in
        window order. The detector restarts after an alarm: otherwise its
        statistic stays above the threshold and one change keeps firing
        after every cooldown (ADR 0010)."""
        alarm = self.page_hinkley.update(error_rate)
        if alarm:
            self.page_hinkley.reset()
        return alarm

    def reset_performance(self) -> None:
        self.page_hinkley.reset()

    @staticmethod
    def error_rate(
        probabilities: FloatArray,
        labels: "np.ndarray[tuple[int], np.dtype[np.int64]]",
        threshold: float = 0.5,
    ) -> float:
        predictions = (probabilities >= threshold).astype(np.int64)
        return float(np.mean(predictions != labels))
