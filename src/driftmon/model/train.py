"""Preprocessing pipelines and baseline model training."""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from driftmon.data import schema

logger = logging.getLogger(__name__)

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 600,
    "learning_rate": 0.1,
    "max_depth": 6,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "tree_method": "hist",
    "eval_metric": "auc",
    "early_stopping_rounds": 30,
    # Fixed thread count: histogram aggregation order depends on it, and the
    # artifact must be bit-for-bit reproducible across runs.
    "n_jobs": 4,
}


@dataclass(frozen=True)
class EvalMetrics:
    auc: float
    accuracy: float
    f1: float
    brier: float

    def as_dict(self) -> dict[str, float]:
        return {
            "auc": self.auc,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "brier": self.brier,
        }


@dataclass(frozen=True)
class TrainResult:
    model: Pipeline
    baseline_model: Pipeline
    metrics: dict[str, dict[str, dict[str, float]]]
    xgb_params: dict[str, Any]


def to_labels(df: pd.DataFrame) -> "np.ndarray[Any, np.dtype[np.int_]]":
    return np.asarray((df[schema.LABEL] == schema.POSITIVE_LABEL).to_numpy(), dtype=np.int_)


def build_preprocessor(*, scale_numeric: bool) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale_numeric:
        numeric_steps.append(("scale", StandardScaler()))
    categorical = Pipeline(
        [
            # Missing workclass/occupation become their own category: the fact
            # that a value is unknown carries signal here.
            ("impute", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", Pipeline(numeric_steps), list(schema.NUMERIC_FEATURES)),
            ("categorical", categorical, list(schema.CATEGORICAL_FEATURES)),
        ],
        remainder="drop",
    )


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df[list(schema.MODEL_FEATURES)]


def evaluate(model: Pipeline, df: pd.DataFrame) -> EvalMetrics:
    y_true = to_labels(df)
    probs = model.predict_proba(features(df))[:, 1]
    preds = (probs >= 0.5).astype(int)
    return EvalMetrics(
        auc=float(roc_auc_score(y_true, probs)),
        accuracy=float(accuracy_score(y_true, preds)),
        f1=float(f1_score(y_true, preds)),
        brier=float(brier_score_loss(y_true, probs)),
    )


def train_models(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed: int,
    xgb_overrides: dict[str, Any] | None = None,
) -> TrainResult:
    """Train the logistic regression baseline and the XGBoost model.

    XGBoost uses early stopping against the validation split, so the
    preprocessor is fitted first and the booster is fitted on transformed
    arrays; the returned pipeline bundles both fitted steps.
    """
    X_train = features(train_df)
    y_train = to_labels(train_df)
    y_validation = to_labels(validation_df)

    lr = Pipeline(
        [
            ("preprocess", build_preprocessor(scale_numeric=True)),
            ("model", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    lr.fit(X_train, y_train)

    preprocessor = build_preprocessor(scale_numeric=False)
    X_train_t = preprocessor.fit_transform(X_train)
    X_validation_t = preprocessor.transform(features(validation_df))

    params = {**XGB_PARAMS, **(xgb_overrides or {}), "random_state": seed}
    booster = XGBClassifier(**params)
    booster.fit(X_train_t, y_train, eval_set=[(X_validation_t, y_validation)], verbose=False)
    logger.info("xgboost stopped at %s trees", booster.best_iteration + 1)

    xgb_pipeline = Pipeline([("preprocess", preprocessor), ("model", booster)])

    metrics: dict[str, dict[str, dict[str, float]]] = {
        "logistic_regression": {"validation": evaluate(lr, validation_df).as_dict()},
        "xgboost": {"validation": evaluate(xgb_pipeline, validation_df).as_dict()},
    }
    return TrainResult(model=xgb_pipeline, baseline_model=lr, metrics=metrics, xgb_params=params)


def write_baseline_report(
    path: Path,
    metrics: dict[str, dict[str, dict[str, float]]],
    metadata: dict[str, Any],
) -> None:
    lines = [
        "# Baseline models",
        "",
        "Logistic regression baseline and XGBoost on the UCI Adult income task",
        "(binary classification, target: income > 50K).",
        "",
        "## Data",
        "",
        f"- Raw rows after deduplication and split: "
        f"{sum(metadata['split_sizes'].values())} across committed splits",
        f"- Duplicate rows removed: {metadata['n_duplicates_removed']}",
        f"- Seed: {metadata['seed']}",
        "",
        "| Split | Rows |",
        "| --- | --- |",
    ]
    for name, size in metadata["split_sizes"].items():
        lines.append(f"| {name} | {size} |")
    lines += [
        "",
        "## Metrics",
        "",
        "| Model | Split | AUC | Accuracy | F1 | Brier |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for model_name, splits in metrics.items():
        for split_name, values in splits.items():
            lines.append(
                f"| {model_name} | {split_name} | {values['auc']:.4f} "
                f"| {values['accuracy']:.4f} | {values['f1']:.4f} "
                f"| {values['brier']:.4f} |"
            )
    lines += [
        "",
        "## Environment",
        "",
        "| Library | Version |",
        "| --- | --- |",
    ]
    for lib, lib_version in metadata["library_versions"].items():
        lines.append(f"| {lib} | {lib_version} |")
    lines.append("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("wrote baseline report to %s", path)
