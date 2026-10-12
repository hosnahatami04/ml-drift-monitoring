from pathlib import Path

import pytest

from driftmon.config import ThresholdsConfig, load_thresholds
from driftmon.model.artifacts import ModelArtifact, build_metadata
from driftmon.model.reference import ReferenceStats, compute_reference_stats
from driftmon.model.train import evaluate, train_models

from .unit.conftest import make_adult_frame

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def tiny_artifact() -> ModelArtifact:
    """A real trained artifact on synthetic data, small enough for tests."""
    train_df = make_adult_frame(1500, seed=31)
    validation_df = make_adult_frame(1500, seed=32)
    result = train_models(
        train_df,
        validation_df,
        seed=7,
        xgb_overrides={"n_estimators": 40, "early_stopping_rounds": 10},
    )
    reference_auc = evaluate(result.model, validation_df).auc
    reference: ReferenceStats = compute_reference_stats(validation_df, result.model, reference_auc)
    metadata = build_metadata(
        data_hashes={"adult.data": "test", "adult.test": "test"},
        seed=7,
        metrics=result.metrics,
        n_duplicates_removed=0,
        split_sizes={"train": len(train_df), "validation": len(validation_df)},
        split_fractions={"train": 0.5},
        model_params=result.xgb_params,
    )
    return ModelArtifact(
        model=result.model,
        baseline_model=result.baseline_model,
        reference=reference,
        metadata=metadata,
    )


@pytest.fixture(scope="session")
def thresholds() -> ThresholdsConfig:
    return load_thresholds(REPO_ROOT / "configs" / "thresholds.yaml")
