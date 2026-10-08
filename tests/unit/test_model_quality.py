"""Quality gates that need the real dataset and a trained artifact.

These run after `make data train` on a developer machine and are skipped in CI
where the dataset is not downloaded.
"""

from pathlib import Path

import pytest

from driftmon.config import load_settings
from driftmon.data import split
from driftmon.model import artifacts
from driftmon.model.train import evaluate

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG = REPO_ROOT / "configs" / "default.yaml"

pytestmark = pytest.mark.data


def _settings_or_skip() -> "tuple[object, Path]":
    settings = load_settings(CONFIG)
    processed = REPO_ROOT / settings.processed_data_dir
    if not (processed / "validation.parquet").is_file():
        pytest.skip("processed dataset not available; run 'make data' first")
    try:
        artifact_dir = artifacts.latest_artifact_dir(REPO_ROOT / settings.artifacts_dir)
    except FileNotFoundError:
        pytest.skip("model artifact not available; run 'make train' first")
    return settings, artifact_dir


def test_xgboost_beats_target_and_baseline() -> None:
    settings, artifact_dir = _settings_or_skip()
    artifact = artifacts.load_artifact(artifact_dir)
    validation = split.load_split(
        REPO_ROOT / settings.processed_data_dir,  # type: ignore[attr-defined]
        "validation",
    )

    xgb = evaluate(artifact.model, validation)
    lr = evaluate(artifact.baseline_model, validation)

    assert xgb.auc >= 0.90
    assert xgb.auc > lr.auc
