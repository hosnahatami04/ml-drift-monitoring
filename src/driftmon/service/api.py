"""FastAPI prediction service wired to the shared monitoring core."""

import logging
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Response
from opentelemetry import trace

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
from driftmon.observability.metrics import Metrics, MetricsAlertSink
from driftmon.observability.tracing import get_tracer
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

MAX_ALERTS = 10_000


@dataclass
class ServiceState:
    artifact: ModelArtifact
    store: Store
    runner: MonitorRunner
    metrics: Metrics
    inline_monitoring: bool = True


def build_state(
    settings: Settings,
    artifact: ModelArtifact | None = None,
    db_path: Path | None = None,
    webhook_url: str | None = None,
    role: Literal["service", "monitor"] = "service",
) -> ServiceState:
    """Wire the store, monitor and alerting. The monitor worker builds the
    same state with role="monitor" and never serves requests."""
    if artifact is None:
        artifact = load_artifact(latest_artifact_dir(settings.artifacts_dir))
    thresholds = load_thresholds(settings.thresholds_path)
    store = Store(db_path or settings.db_path, window_size=settings.window_size)
    monitor = Monitor(artifact.reference, thresholds)
    sinks: list[AlertSink] = [
        SqliteAlertSink(store),
        JsonlSink((db_path or settings.db_path).with_suffix(".alerts.jsonl")),
    ]
    webhook_url = webhook_url or settings.service.webhook_url
    if webhook_url:
        sinks.append(WebhookSink(webhook_url))
    inline = settings.service.inline_monitoring
    metrics = Metrics(service=role == "service", monitor=role == "monitor" or inline)
    sinks.append(MetricsAlertSink(metrics))
    manager = AlertManager(thresholds.alerts, sinks)
    runner = MonitorRunner(store, monitor, manager, metrics=metrics)
    return ServiceState(
        artifact=artifact,
        store=store,
        runner=runner,
        metrics=metrics,
        inline_monitoring=inline,
    )


def create_app(state: ServiceState, tracer: trace.Tracer | None = None) -> FastAPI:
    app = FastAPI(title="driftmon", version="0.1.0")
    model = state.artifact.model
    if tracer is None:
        tracer = get_tracer()
    else:
        # Inline monitoring spans should land in the same provider as the
        # request spans that trigger them.
        state.runner.tracer = tracer

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest) -> PredictResponse:
        start = time.perf_counter()
        with tracer.start_as_current_span("predict") as span:
            features = request.to_feature_dict()
            frame = pd.DataFrame([features])[list(data_schema.MODEL_FEATURES)]
            probability = float(model.predict_proba(frame)[0, 1])
            request_id = uuid.uuid4().hex
            window_id, row_count = state.store.insert_prediction(request_id, features, probability)
            span.set_attribute("driftmon.window_id", window_id)
            if state.inline_monitoring and row_count >= state.store.window_size:
                state.runner.process_complete_windows()
        state.metrics.observe_prediction(time.perf_counter() - start)
        return PredictResponse(request_id=request_id, probability=probability)

    @app.post("/labels", response_model=LabelResponse)
    def labels(request: LabelRequest) -> LabelResponse:
        attached = state.store.attach_label(request.request_id, request.label)
        if attached is None:
            raise HTTPException(
                status_code=404,
                detail=f"no prediction with request_id {request.request_id!r}",
            )
        if attached:
            state.metrics.observe_label()
            if state.inline_monitoring:
                state.runner.process_labelled_windows()
        return LabelResponse(request_id=request.request_id, attached=attached)

    @app.get("/metrics")
    def metrics() -> Response:
        body, content_type = state.metrics.render()
        return Response(content=body, media_type=content_type)

    @app.get("/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            status="ok",
            model_fingerprint=fingerprint(state.artifact.metadata),
            window_size=state.store.window_size,
            predictions=state.store.prediction_count(),
            processed_windows=state.store.processed_window_count(),
            labelled_windows=state.store.fully_labelled_count(),
            performance_checked_windows=state.store.performance_checked_count(),
        )

    @app.get("/alerts", response_model=list[AlertRecord])
    def alerts(limit: Annotated[int, Query(ge=1, le=MAX_ALERTS)] = 100) -> list[AlertRecord]:
        return [AlertRecord(**record) for record in state.store.list_alerts(limit)]

    return app
