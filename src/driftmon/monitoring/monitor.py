"""The shared monitoring core.

One Monitor instance watches one model: it holds the reference statistics and
frozen thresholds, runs every detector on a window of features plus predicted
probabilities, and tracks the error-rate stream once labels arrive. No IO
here - the live runner and the evaluation harness both import this class, so
evaluation measures exactly the code that runs in production.
"""

from collections import Counter
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from driftmon.config import ThresholdsConfig
from driftmon.data import schema
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
        """Run every input-side detector on one complete window."""
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

        decision = decide_window(results, self.decision_config)
        prediction_flagged = any(
            r.flagged and r.method == "prediction_ks" for r in decision.results
        )
        input_features = tuple(f for f in decision.flagged_features if f != "__prediction__")
        return WindowReport(
            window_id=window_id,
            n_rows=len(features),
            results=decision.results,
            flagged_features=input_features,
            feature_drift=len(input_features) >= self.decision_config.k_features,
            prediction_drift=prediction_flagged,
        )

    def observe_error_rate(self, error_rate: float) -> bool:
        """Feed one window's realized error rate to Page-Hinkley. Called only
        once that window's labels have fully arrived, in window order."""
        return self.page_hinkley.update(error_rate)

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

    def expected_features(self) -> list[str]:
        return list(schema.MODEL_FEATURES)
