from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from prometheus_client.parser import text_string_to_metric_families

from driftmon.config import Settings, TelemetryConfig
from driftmon.data import schema
from driftmon.model.artifacts import ModelArtifact
from driftmon.observability import tracing
from driftmon.observability.metrics import Metrics
from driftmon.service.api import ServiceState, build_state, create_app

from .conftest import make_adult_frame

REPO_ROOT = Path(__file__).resolve().parents[2]

WINDOW = 200


def settings_for(tmp_path: Path, *, inline: bool = True) -> Settings:
    return Settings(
        window_size=WINDOW,
        thresholds_path=REPO_ROOT / "configs" / "thresholds.yaml",
        db_path=tmp_path / "store.db",
        service={"inline_monitoring": inline},
    )


def payloads(n: int, seed: int = 50) -> list[dict[str, object]]:
    frame = make_adult_frame(n, seed=seed)[list(schema.MODEL_FEATURES)]
    return [
        {k: (v.item() if hasattr(v, "item") else v) for k, v in row.items() if v == v}
        for row in frame.to_dict(orient="records")
    ]


def scrape(client: TestClient) -> dict[str, float]:
    response = client.get("/metrics")
    assert response.status_code == 200
    samples: dict[str, float] = {}
    for family in text_string_to_metric_families(response.text):
        for sample in family.samples:
            key = sample.name
            if sample.labels:
                key += "{" + ",".join(f"{k}={v}" for k, v in sorted(sample.labels.items())) + "}"
            samples[key] = sample.value
    return samples


@pytest.fixture
def state(tiny_artifact: ModelArtifact, tmp_path: Path) -> ServiceState:
    return build_state(settings_for(tmp_path), artifact=tiny_artifact)


def test_metrics_endpoint_counts_predictions_and_latency(state: ServiceState) -> None:
    client = TestClient(create_app(state))
    before = scrape(client)

    for payload in payloads(3):
        assert client.post("/predict", json=payload).status_code == 200
    after = scrape(client)

    assert before["driftmon_predictions_total"] == 0
    assert after["driftmon_predictions_total"] == 3
    assert after["driftmon_predict_latency_seconds_count"] == 3
    assert after["driftmon_predict_latency_seconds_sum"] > 0


def test_completed_window_updates_drift_gauges(state: ServiceState) -> None:
    client = TestClient(create_app(state))

    for payload in payloads(WINDOW):
        client.post("/predict", json=payload)
    samples = scrape(client)

    assert samples["driftmon_windows_processed_total"] == 1
    assert samples["driftmon_last_processed_window"] == 0
    assert "driftmon_drift_statistic{feature=age,method=ks}" in samples
    assert "driftmon_drift_flagged{feature=__prediction__,method=prediction_ks}" in samples


def test_labels_are_counted(state: ServiceState) -> None:
    client = TestClient(create_app(state))
    request_id = client.post("/predict", json=payloads(1)[0]).json()["request_id"]

    client.post("/labels", json={"request_id": request_id, "label": 1})

    assert scrape(client)["driftmon_labels_total"] == 1


def test_api_without_inline_monitoring_exports_no_monitor_metrics(
    tiny_artifact: ModelArtifact, tmp_path: Path
) -> None:
    api = build_state(settings_for(tmp_path, inline=False), artifact=tiny_artifact)
    worker = build_state(
        settings_for(tmp_path, inline=False), artifact=tiny_artifact, role="monitor"
    )

    api_names = {f.name for f in text_string_to_metric_families(api.metrics.render()[0].decode())}
    worker_names = {
        f.name for f in text_string_to_metric_families(worker.metrics.render()[0].decode())
    }

    assert "driftmon_predictions" in api_names
    assert "driftmon_last_processed_window" not in api_names
    assert "driftmon_last_processed_window" in worker_names
    assert "driftmon_predictions" not in worker_names


def test_metric_helpers_are_noops_for_absent_roles() -> None:
    metrics = Metrics(service=False, monitor=False)

    metrics.observe_prediction(0.01)
    metrics.observe_label()
    metrics.record_error_rate(0.2)

    assert metrics.render()[0] == b""


def test_predict_and_monitor_spans_are_recorded(state: ServiceState) -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = TestClient(create_app(state, tracer=provider.get_tracer("test")))

    for payload in payloads(WINDOW):
        client.post("/predict", json=payload)

    spans = exporter.get_finished_spans()
    predict = [s for s in spans if s.name == "predict"]
    windows = [s for s in spans if s.name == "monitor.process_window"]
    assert len(predict) == WINDOW
    assert predict[0].attributes is not None
    assert predict[0].attributes["driftmon.window_id"] == 0
    assert len(windows) == 1
    # The window is processed inside the request that completed it.
    assert windows[0].parent is not None
    assert windows[0].parent.span_id == predict[-1].context.span_id


def test_unreachable_collector_never_fails_a_request(
    state: ServiceState, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Port 9 (discard) has nothing listening: every export attempt fails.
    monkeypatch.setenv(tracing.OTLP_ENDPOINT_ENV, "http://127.0.0.1:9")
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_TIMEOUT", "1")
    provider = tracing.build_tracer_provider(TelemetryConfig(), "driftmon-test")
    client = TestClient(create_app(state, tracer=provider.get_tracer("test")))

    responses = [client.post("/predict", json=p) for p in payloads(5)]

    assert all(r.status_code == 200 for r in responses)
    provider.shutdown()


def test_exporter_selection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(tracing.OTLP_ENDPOINT_ENV, raising=False)

    console = tracing.build_tracer_provider(TelemetryConfig(), "a")
    silent = tracing.build_tracer_provider(TelemetryConfig(trace_exporter="none"), "b")

    assert console._active_span_processor._span_processors  # type: ignore[attr-defined]
    assert not silent._active_span_processor._span_processors  # type: ignore[attr-defined]
    assert console.resource.attributes["service.name"] == "a"


def test_sample_ratio_zero_records_nothing(state: ServiceState) -> None:
    exporter = InMemorySpanExporter()
    provider = tracing.build_tracer_provider(
        TelemetryConfig(trace_exporter="none", trace_sample_ratio=0.0), "c"
    )
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = TestClient(create_app(state, tracer=provider.get_tracer("test")))

    client.post("/predict", json=payloads(1)[0])

    assert exporter.get_finished_spans() == ()
