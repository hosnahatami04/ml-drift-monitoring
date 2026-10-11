"""End-to-end: simulated streams through the live API, alerts out the other
side, at the window size the frozen thresholds were calibrated for."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from driftmon.config import Settings
from driftmon.data import schema
from driftmon.model.artifacts import ModelArtifact
from driftmon.service.api import ServiceState, build_state, create_app
from driftmon.simulation import sampler
from driftmon.simulation.scenarios import (
    ScenarioConfig,
    StreamParams,
    generate_stream,
)

from ..unit.conftest import make_adult_frame

REPO_ROOT = Path(__file__).resolve().parents[2]

# The frozen thresholds are calibrated for 1000-row windows. At smaller
# windows the PSI gate false-alarms (window-size ablation in
# reports/results.md), and that alarm's cooldown can hide the real drift.
PARAMS = StreamParams(n_windows=6, window_size=1000, drift_start=3, label_delay=2)

CONTROL = ScenarioConfig(name="control", kind="control")

SUDDEN_HIGH = ScenarioConfig(
    name="sudden_covariate",
    kind="sudden_covariate",
    feature="age",
    severity={"high": {"beta": 1.0}},
)


@pytest.fixture(scope="module")
def pool() -> pd.DataFrame:
    return make_adult_frame(5000, seed=60)


@pytest.fixture(scope="module")
def stats(pool: pd.DataFrame) -> dict[str, tuple[float, float]]:
    return sampler.reference_feature_stats(make_adult_frame(1500, seed=61), schema.NUMERIC_FEATURES)


def _replay(
    tiny_artifact: ModelArtifact,
    tmp_path: Path,
    pool: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
    scenario: ScenarioConfig,
    severity: str | None,
) -> list[dict[str, object]]:
    settings = Settings(
        window_size=PARAMS.window_size,
        thresholds_path=REPO_ROOT / "configs" / "thresholds.yaml",
        db_path=tmp_path / "replay.db",
    )
    state: ServiceState = build_state(settings, artifact=tiny_artifact)
    client = TestClient(create_app(state))

    pending: dict[int, list[tuple[str, int]]] = {}
    for window in generate_stream(pool, stats, scenario, severity, 100, PARAMS):
        for arrived in [w for w in pending if w <= window.index]:
            for request_id, label in pending.pop(arrived):
                client.post("/labels", json={"request_id": request_id, "label": label})
        ids: list[tuple[str, int]] = []
        rows = window.X[list(schema.MODEL_FEATURES)].to_dict(orient="records")
        for row, label in zip(rows, window.y_true.tolist(), strict=True):
            payload = {
                k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if pd.notna(v)
            }
            response = client.post("/predict", json=payload)
            assert response.status_code == 200, response.text
            ids.append((response.json()["request_id"], int(label)))
        pending[window.label_available_at] = ids

    alerts: list[dict[str, object]] = client.get("/alerts", params={"limit": 1000}).json()
    return alerts


def test_control_replay_stays_quiet(
    tiny_artifact: ModelArtifact,
    tmp_path: Path,
    pool: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
) -> None:
    alerts = _replay(tiny_artifact, tmp_path, pool, stats, CONTROL, None)

    assert len(alerts) <= 1


def test_sudden_high_replay_alerts_fast(
    tiny_artifact: ModelArtifact,
    tmp_path: Path,
    pool: pd.DataFrame,
    stats: dict[str, tuple[float, float]],
) -> None:
    alerts = _replay(tiny_artifact, tmp_path, pool, stats, SUDDEN_HIGH, "high")

    drift_alerts = [a for a in alerts if a["type"] in ("feature_drift", "prediction_drift")]
    assert drift_alerts, "expected at least one drift alert"
    early = [
        a
        for a in drift_alerts
        if PARAMS.drift_start <= int(a["window_id"]) <= PARAMS.drift_start + 1
    ]
    assert early, f"expected an alert within two windows of drift start, got {drift_alerts}"
