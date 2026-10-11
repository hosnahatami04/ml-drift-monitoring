"""Prometheus metrics for the prediction service and the monitor.

Each process owns its registry instead of using the global default one, so
the API, the monitor worker and the test suite never collide on metric
names. A process registers only the metrics of the roles it plays: an
unregistered metric is absent from /metrics, while a registered but unused
gauge would report a misleading 0.
"""

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)

from driftmon.monitoring.alerts import Alert
from driftmon.monitoring.monitor import WindowReport

# Single-row predictions take a few milliseconds; the buckets resolve that
# range and still catch a slow tail.
LATENCY_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)


class ServiceMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.predictions = Counter("driftmon_predictions", "Predictions served.", registry=registry)
        self.labels = Counter(
            "driftmon_labels", "Delayed labels attached to predictions.", registry=registry
        )
        self.predict_latency = Histogram(
            "driftmon_predict_latency_seconds",
            "Wall time of POST /predict.",
            buckets=LATENCY_BUCKETS,
            registry=registry,
        )


class MonitorMetrics:
    def __init__(self, registry: CollectorRegistry) -> None:
        self.drift_statistic = Gauge(
            "driftmon_drift_statistic",
            "Latest test statistic per monitored feature and method.",
            ["feature", "method"],
            registry=registry,
        )
        self.drift_flagged = Gauge(
            "driftmon_drift_flagged",
            "1 if the feature's test flagged drift in the latest window.",
            ["feature", "method"],
            registry=registry,
        )
        self.windows_processed = Counter(
            "driftmon_windows_processed", "Monitoring windows processed.", registry=registry
        )
        self.last_window = Gauge(
            "driftmon_last_processed_window",
            "Id of the most recent processed window.",
            registry=registry,
        )
        self.error_rate = Gauge(
            "driftmon_window_error_rate",
            "Error rate of the most recent fully labelled window.",
            registry=registry,
        )
        self.alerts = Counter(
            "driftmon_alerts",
            "Alerts raised, by type and severity.",
            ["type", "severity"],
            registry=registry,
        )


class Metrics:
    def __init__(self, *, service: bool = True, monitor: bool = True) -> None:
        self.registry = CollectorRegistry()
        self.service = ServiceMetrics(self.registry) if service else None
        self.monitor = MonitorMetrics(self.registry) if monitor else None

    def observe_prediction(self, seconds: float) -> None:
        if self.service is not None:
            self.service.predictions.inc()
            self.service.predict_latency.observe(seconds)

    def observe_label(self) -> None:
        if self.service is not None:
            self.service.labels.inc()

    def record_window(self, report: WindowReport) -> None:
        if self.monitor is None:
            return
        for result in report.results:
            labels = {"feature": result.feature, "method": result.method}
            self.monitor.drift_statistic.labels(**labels).set(result.statistic)
            self.monitor.drift_flagged.labels(**labels).set(1.0 if result.flagged else 0.0)
        self.monitor.windows_processed.inc()
        self.monitor.last_window.set(report.window_id)

    def record_error_rate(self, error_rate: float) -> None:
        if self.monitor is not None:
            self.monitor.error_rate.set(error_rate)

    def record_alert(self, alert: Alert) -> None:
        if self.monitor is not None:
            self.monitor.alerts.labels(type=alert.type, severity=alert.severity).inc()

    def render(self) -> tuple[bytes, str]:
        return generate_latest(self.registry), CONTENT_TYPE_LATEST


class MetricsAlertSink:
    """Counts alerts as they are delivered, alongside the other sinks."""

    def __init__(self, metrics: Metrics) -> None:
        self.metrics = metrics

    def emit(self, alert: Alert) -> None:
        self.metrics.record_alert(alert)
