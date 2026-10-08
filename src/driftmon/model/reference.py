"""Reference statistics consumed by the drift detectors."""

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.pipeline import Pipeline

from driftmon.data import schema
from driftmon.detectors.psi import quantile_bin_edges

MISSING_CATEGORY = "missing"


@dataclass(frozen=True)
class NumericReference:
    mean: float
    std: float
    bin_edges: list[float]
    sample: list[float]


@dataclass(frozen=True)
class CategoricalReference:
    frequencies: dict[str, float]


@dataclass(frozen=True)
class ReferenceStats:
    n_rows: int
    numeric: dict[str, NumericReference]
    categorical: dict[str, CategoricalReference]
    prediction_probs: list[float]
    auc: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_rows": self.n_rows,
            "numeric": {
                name: {
                    "mean": ref.mean,
                    "std": ref.std,
                    "bin_edges": ref.bin_edges,
                    "sample": ref.sample,
                }
                for name, ref in self.numeric.items()
            },
            "categorical": {
                name: {"frequencies": ref.frequencies} for name, ref in self.categorical.items()
            },
            "prediction_probs": self.prediction_probs,
            "auc": self.auc,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ReferenceStats":
        return cls(
            n_rows=payload["n_rows"],
            numeric={
                name: NumericReference(
                    mean=item["mean"],
                    std=item["std"],
                    bin_edges=item["bin_edges"],
                    sample=item["sample"],
                )
                for name, item in payload["numeric"].items()
            },
            categorical={
                name: CategoricalReference(frequencies=item["frequencies"])
                for name, item in payload["categorical"].items()
            },
            prediction_probs=payload["prediction_probs"],
            auc=payload["auc"],
        )


def compute_reference_stats(
    reference_df: pd.DataFrame, model: Pipeline, reference_auc: float
) -> ReferenceStats:
    numeric: dict[str, NumericReference] = {}
    for name in schema.NUMERIC_FEATURES:
        values = reference_df[name].to_numpy(dtype=float)
        numeric[name] = NumericReference(
            mean=float(values.mean()),
            std=float(values.std(ddof=1)),
            bin_edges=quantile_bin_edges(values),
            sample=[float(v) for v in values],
        )

    categorical: dict[str, CategoricalReference] = {}
    for name in schema.CATEGORICAL_FEATURES:
        series = reference_df[name].fillna(MISSING_CATEGORY)
        freqs = series.value_counts(normalize=True)
        categorical[name] = CategoricalReference(
            frequencies={str(k): float(v) for k, v in freqs.items()}
        )

    probs = model.predict_proba(reference_df[list(schema.MODEL_FEATURES)])[:, 1]

    return ReferenceStats(
        n_rows=len(reference_df),
        numeric=numeric,
        categorical=categorical,
        prediction_probs=[float(p) for p in probs],
        auc=reference_auc,
    )
