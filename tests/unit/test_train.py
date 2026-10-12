from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from driftmon.data import schema
from driftmon.detectors.psi import quantile_bin_edges
from driftmon.model import artifacts
from driftmon.model.reference import compute_reference_stats
from driftmon.model.train import evaluate, train_models, write_baseline_report

from .conftest import make_adult_frame

FAST_XGB: dict[str, Any] = {"n_estimators": 40, "early_stopping_rounds": 10}


def _train_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    return make_adult_frame(1200, seed=1), make_adult_frame(300, seed=2)


def test_same_seed_gives_identical_predictions() -> None:
    train_df, validation_df = _train_frames()

    first = train_models(train_df, validation_df, seed=42, xgb_overrides=FAST_XGB)
    second = train_models(train_df, validation_df, seed=42, xgb_overrides=FAST_XGB)

    X = validation_df[list(schema.MODEL_FEATURES)]
    np.testing.assert_array_equal(first.model.predict_proba(X), second.model.predict_proba(X))
    np.testing.assert_array_equal(
        first.baseline_model.predict_proba(X), second.baseline_model.predict_proba(X)
    )


def test_model_learns_signal() -> None:
    train_df, validation_df = _train_frames()

    result = train_models(train_df, validation_df, seed=42, xgb_overrides=FAST_XGB)

    assert result.metrics["xgboost"]["validation"]["auc"] > 0.7


def test_artifact_roundtrip_preserves_predictions(tmp_path: Path) -> None:
    train_df, validation_df = _train_frames()
    result = train_models(train_df, validation_df, seed=42, xgb_overrides=FAST_XGB)
    reference_stats = compute_reference_stats(validation_df, result.model, reference_auc=0.9)
    metadata = artifacts.build_metadata(
        data_hashes={"adult.data": "x", "adult.test": "y"},
        seed=42,
        metrics=result.metrics,
        n_duplicates_removed=0,
        split_sizes={"train": len(train_df), "validation": len(validation_df)},
        split_fractions={"train": 0.5},
        model_params=result.xgb_params,
    )
    artifact = artifacts.ModelArtifact(
        model=result.model,
        baseline_model=result.baseline_model,
        reference=reference_stats,
        metadata=metadata,
    )

    out_dir = artifacts.save_artifact(artifact, tmp_path)
    loaded = artifacts.load_artifact(out_dir)

    X = validation_df[list(schema.MODEL_FEATURES)]
    np.testing.assert_array_equal(artifact.model.predict_proba(X), loaded.model.predict_proba(X))
    assert loaded.reference.to_dict() == reference_stats.to_dict()
    assert loaded.metadata["seed"] == 42
    assert artifacts.latest_artifact_dir(tmp_path) == out_dir


def test_quantile_bins_merge_point_masses() -> None:
    values = np.array([0.0] * 95 + [10.0, 20.0, 30.0, 40.0, 50.0])

    edges = quantile_bin_edges(values)

    assert edges == sorted(set(edges))
    assert edges[0] == 0.0


def test_reference_stats_cover_all_model_features() -> None:
    train_df, validation_df = _train_frames()
    result = train_models(train_df, validation_df, seed=42, xgb_overrides=FAST_XGB)

    stats = compute_reference_stats(validation_df, result.model, reference_auc=0.9)

    assert set(stats.numeric) == set(schema.NUMERIC_FEATURES)
    assert set(stats.categorical) == set(schema.CATEGORICAL_FEATURES)
    assert stats.n_rows == len(validation_df)
    assert len(stats.prediction_probs) == len(validation_df)
    for ref in stats.categorical.values():
        assert abs(sum(ref.frequencies.values()) - 1.0) < 1e-9


def test_baseline_report_contains_metrics(tmp_path: Path) -> None:
    train_df, validation_df = _train_frames()
    result = train_models(train_df, validation_df, seed=42, xgb_overrides=FAST_XGB)
    metadata = artifacts.build_metadata(
        data_hashes={"adult.data": "x", "adult.test": "y"},
        seed=42,
        metrics=result.metrics,
        n_duplicates_removed=7,
        split_sizes={"train": len(train_df), "validation": len(validation_df)},
        split_fractions={"train": 0.5},
        model_params=result.xgb_params,
    )
    report = tmp_path / "baseline.md"

    write_baseline_report(report, result.metrics, metadata)

    text = report.read_text(encoding="utf-8")
    auc = result.metrics["xgboost"]["validation"]["auc"]
    assert f"{auc:.4f}" in text
    assert "| xgboost | validation |" in text
    assert "Duplicate rows removed: 7" in text


def test_evaluate_returns_probability_based_metrics() -> None:
    train_df, validation_df = _train_frames()
    result = train_models(train_df, validation_df, seed=42, xgb_overrides=FAST_XGB)

    metrics = evaluate(result.model, validation_df)

    assert 0.0 <= metrics.auc <= 1.0
    assert 0.0 <= metrics.brier <= 1.0


def test_fingerprint_tracks_everything_that_shapes_the_model() -> None:
    base = artifacts.build_metadata(
        data_hashes={"adult.data": "x", "adult.test": "y"},
        seed=42,
        metrics={},
        n_duplicates_removed=0,
        split_sizes={},
        split_fractions={"train": 0.5},
        model_params={"max_depth": 6},
    )
    retrained = {**base, "created_at": "later", "metrics": {"xgboost": {}}}
    deeper = {**base, "model_params": {"max_depth": 8}}
    resplit = {**base, "split_fractions": {"train": 0.6}}

    assert artifacts.fingerprint(retrained) == artifacts.fingerprint(base)
    assert artifacts.fingerprint(deeper) != artifacts.fingerprint(base)
    assert artifacts.fingerprint(resplit) != artifacts.fingerprint(base)
