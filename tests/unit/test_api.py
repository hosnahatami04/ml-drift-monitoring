from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from driftmon.config import Settings, ThresholdsConfig
from driftmon.data import schema
from driftmon.model.artifacts import ModelArtifact
from driftmon.monitoring.monitor import Monitor
from driftmon.service.api import ServiceState, build_state, create_app

from .conftest import make_adult_frame

REPO_ROOT = Path(__file__).resolve().parents[2]

WINDOW_SIZE = 40


@pytest.fixture
def state(tiny_artifact: ModelArtifact, tmp_path: Path) -> ServiceState:
    settings = Settings(
        window_size=WINDOW_SIZE,
        thresholds_path=REPO_ROOT / "configs" / "thresholds.yaml",
        db_path=tmp_path / "service.db",
    )
    return build_state(settings, artifact=tiny_artifact)


@pytest.fixture
def client(state: ServiceState) -> TestClient:
    return TestClient(create_app(state))


def _payload(seed: int = 50) -> dict[str, object]:
    row = make_adult_frame(1, seed=seed)[list(schema.MODEL_FEATURES)].iloc[0]
    return {key: (value.item() if hasattr(value, "item") else value) for key, value in row.items()}


def test_predict_returns_probability_and_request_id(client: TestClient) -> None:
    response = client.post("/predict", json=_payload())

    assert response.status_code == 200
    body = response.json()
    assert 0.0 <= body["probability"] <= 1.0
    assert len(body["request_id"]) == 32


def test_invalid_value_is_422(client: TestClient) -> None:
    payload = _payload()
    payload["age"] = 300

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "age" in response.text


def test_missing_feature_is_422_naming_the_field(client: TestClient) -> None:
    payload = _payload()
    del payload["relationship"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "relationship" in response.text


def test_unknown_category_is_422(client: TestClient) -> None:
    payload = _payload()
    payload["occupation"] = "Astronaut"

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
    assert "Astronaut" in response.text


def test_optional_missing_categories_are_accepted(client: TestClient) -> None:
    payload = _payload()
    del payload["workclass"]
    del payload["occupation"]

    response = client.post("/predict", json=payload)

    assert response.status_code == 200


def test_label_for_unknown_request_is_404(client: TestClient) -> None:
    response = client.post("/labels", json={"request_id": "nope", "label": 1})

    assert response.status_code == 404


def test_label_attaches_to_prediction(client: TestClient) -> None:
    request_id = client.post("/predict", json=_payload()).json()["request_id"]

    response = client.post("/labels", json={"request_id": request_id, "label": 1})

    assert response.status_code == 200
    assert response.json()["attached"] is True


def test_repeated_label_is_ignored(client: TestClient) -> None:
    request_id = client.post("/predict", json=_payload()).json()["request_id"]
    client.post("/labels", json={"request_id": request_id, "label": 1})

    repeat = client.post("/labels", json={"request_id": request_id, "label": 0})

    assert repeat.status_code == 200
    assert repeat.json()["attached"] is False
    metrics = client.get("/metrics").text
    assert "driftmon_labels_total 1.0" in metrics


@pytest.mark.parametrize("limit", [0, -1, 10_001])
def test_alert_listing_limit_is_bounded(client: TestClient, limit: int) -> None:
    assert client.get("/alerts", params={"limit": limit}).status_code == 422


def test_health_reports_counts(client: TestClient) -> None:
    client.post("/predict", json=_payload())

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["predictions"] == 1
    assert body["window_size"] == WINDOW_SIZE
    assert body["processed_windows"] == 0
    assert body["labelled_windows"] == 0
    assert body["performance_checked_windows"] == 0
    assert len(body["model_fingerprint"]) == 12


def test_completed_window_is_processed_and_alerts_listed(
    client: TestClient, state: ServiceState
) -> None:
    rows = make_adult_frame(WINDOW_SIZE, seed=51)[list(schema.MODEL_FEATURES)]
    for _, row in rows.iterrows():
        payload = {k: v for k, v in row.items() if pd.notna(v)}
        payload = {k: (v.item() if hasattr(v, "item") else v) for k, v in payload.items()}
        assert client.post("/predict", json=payload).status_code == 200

    assert state.store.processed_window_count() == 1
    assert client.get("/alerts").status_code == 200


def test_api_window_report_matches_in_process_monitor(
    client: TestClient,
    state: ServiceState,
    tiny_artifact: ModelArtifact,
    thresholds: ThresholdsConfig,
) -> None:
    frame = make_adult_frame(WINDOW_SIZE, seed=52)[list(schema.MODEL_FEATURES)]
    for _, row in frame.iterrows():
        payload = {
            k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if pd.notna(v)
        }
        client.post("/predict", json=payload)

    stored = state.store.window_predictions(0)
    # Rebuild the exact window the service processed and run the monitor
    # directly on it: both paths must agree bit for bit.
    features = pd.DataFrame([p.features for p in stored])[list(schema.MODEL_FEATURES)]
    probabilities = np.asarray([p.probability for p in stored], dtype=float)
    direct = Monitor(tiny_artifact.reference, thresholds).process_window(0, features, probabilities)

    import json
    import sqlite3

    with sqlite3.connect(state.store.path) as connection:
        row = connection.execute("SELECT report FROM windows WHERE window_id = 0").fetchone()
    assert row is not None
    assert json.loads(row[0]) == direct.to_dict()
