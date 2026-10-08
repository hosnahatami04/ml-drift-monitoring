"""FastAPI prediction service wired to the shared monitoring core."""

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException

from driftmon.config import Settings, load_thresholds
from driftmon.data import schema as data_schema
from driftmon.model.artifacts import (
    ModelArtifact,
    fingerprint,
    latest_artifact_dir,
    load_artifact,
)
from driftmon.monitoring.alerts import AlertManager, AlertSink, JsonlSink, WebhookSink
from driftmon.monitoring.monitor import Monitor
from driftmon.monitoring.runner import MonitorRunner
from driftmon.service.schemas import (
    AlertRecord,
    HealthResponse,
    LabelRequest,
    LabelResponse,
    PredictRequest,
    PredictResponse,
)
from driftmon.service.store import SqliteAlertSink, Store

logger = logging.getLogger(__name__)


@dataclass
class ServiceState:
    artifact: ModelArtifact
    store: Store
    runner: MonitorRunner


def build_state(
    settings: Settings,
    artifact: ModelArtifact | None = None,
    db_path: Path | None = None,
    webhook_url: str | None = None,
) -> ServiceState:
    if artifact is None:
        artifact = load_artifact(latest_artifact_dir(settings.artifacts_dir))
    thresholds = load_thresholds(settings.thresholds_path)
    store = Store(db_path or settings.db_path, window_size=settings.window_size)
    monitor = Monitor(artifact.reference, thresholds)
    sinks: list[AlertSink] = [
        SqliteAlertSink(store),
        JsonlSink((db_path or settings.db_path).with_suffix(".alerts.jsonl")),
    ]
    if webhook_url:
        sinks.append(WebhookSink(webhook_url))
    manager = AlertManager(thresholds.alerts, sinks)
    runner = MonitorRunner(store, monitor, manager)
    return ServiceState(artifact=artifact, store=store, runner=runner)


def create_app(state: ServiceState) -> FastAPI:
    app = FastAPI(title="driftmon", version="0.1.0")
    model = state.artifact.model

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        features = request.to_feature_dict()
        frame = pd.DataFrame([features])[list(data_schema.MODEL_FEATURES)]
        probability = float(model.predict_proba(frame)[0, 1])
        request_id = uuid.uuid4().hex
        _, row_count = state.store.insert_prediction(request_id, features, probability)
        if row_count >= state.store.window_size:
            state.runner.process_complete_windows()
        return PredictResponse(request_id=request_id, probability=probability)

    @app.post("/labels", response_model=LabelResponse)
    def labels(request: LabelRequest) -> LabelResponse:
        attached = state.store.attach_label(request.request_id, request.label)
        if not attached:
            raise HTTPException(
                status_code=404,
                detail=f"no prediction with request_id {request.request_id!r}",
            )
        state.runner.process_labelled_windows()
        return LabelResponse(request_id=request.request_id, attached=True)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_fingerprint=fingerprint(state.artifact.metadata),
            predictions=state.store.prediction_count(),
            processed_windows=state.store.processed_window_count(),
        )

    @app.get("/alerts", response_model=list[AlertRecord])
    def alerts(limit: int = 100) -> list[AlertRecord]:
        return [AlertRecord(**record) for record in state.store.list_alerts(limit)]

    return app
