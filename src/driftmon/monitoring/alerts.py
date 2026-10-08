"""Alert construction, cooldown and delivery sinks."""

import json
import logging
import urllib.request
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Protocol

from driftmon.config import AlertSettings
from driftmon.monitoring.monitor import WindowReport

logger = logging.getLogger(__name__)

AlertType = Literal["feature_drift", "prediction_drift", "performance_drift"]

AlertSeverity = Literal["warning", "critical"]


@dataclass(frozen=True)
class Alert:
    window_id: int
    type: AlertType
    severity: AlertSeverity
    features: tuple[str, ...]
    details: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "window_id": self.window_id,
            "type": self.type,
            "severity": self.severity,
            "features": list(self.features),
            "details": self.details,
            "created_at": self.created_at,
        }


class AlertSink(Protocol):
    def emit(self, alert: Alert) -> None: ...


class JsonlSink:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(self, alert: Alert) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(alert.to_dict()) + "\n")


class WebhookSink:
    """Optional generic HTTP sink; disabled unless a URL is configured.
    Delivery failures are logged, never raised: alerting must not take the
    service down."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self.url = url
        self.timeout = timeout

    def emit(self, alert: Alert) -> None:
        payload = json.dumps(alert.to_dict()).encode("utf-8")
        request = urllib.request.Request(
            self.url, data=payload, headers={"Content-Type": "application/json"}
        )
        try:
            urllib.request.urlopen(request, timeout=self.timeout)
        except OSError as exc:
            logger.warning("webhook delivery failed: %s", exc)


class AlertManager:
    """Turns window reports into alerts, applying a per-type cooldown so one
    sustained drift does not page on every window."""

    def __init__(self, settings: AlertSettings, sinks: list[AlertSink] | None = None) -> None:
        self.settings = settings
        self.sinks = sinks or []
        self._last_fired: dict[str, int] = {}

    def _in_cooldown(self, alert_type: str, window_id: int) -> bool:
        last = self._last_fired.get(alert_type)
        return last is not None and window_id - last <= self.settings.cooldown_windows

    def _fire(self, alert: Alert) -> Alert:
        self._last_fired[alert.type] = alert.window_id
        for sink in self.sinks:
            sink.emit(alert)
        return alert

    def evaluate_window(self, report: WindowReport) -> list[Alert]:
        alerts: list[Alert] = []
        created_at = datetime.now(UTC).isoformat(timespec="seconds")

        if report.feature_drift and not self._in_cooldown("feature_drift", report.window_id):
            severity: AlertSeverity = (
                "critical"
                if len(report.flagged_features) >= self.settings.critical_feature_count
                else "warning"
            )
            statistics = {
                r.feature: {"method": r.method, "statistic": r.statistic, "p_value": r.p_value}
                for r in report.results
                if r.flagged and r.feature in report.flagged_features
            }
            alerts.append(
                self._fire(
                    Alert(
                        window_id=report.window_id,
                        type="feature_drift",
                        severity=severity,
                        features=report.flagged_features,
                        details={"statistics": statistics},
                        created_at=created_at,
                    )
                )
            )

        if report.prediction_drift and not self._in_cooldown("prediction_drift", report.window_id):
            prediction_result = next(r for r in report.results if r.method == "prediction_ks")
            alerts.append(
                self._fire(
                    Alert(
                        window_id=report.window_id,
                        type="prediction_drift",
                        severity="warning",
                        features=(),
                        details={
                            "statistic": prediction_result.statistic,
                            "p_value": prediction_result.p_value,
                        },
                        created_at=created_at,
                    )
                )
            )
        return alerts

    def evaluate_performance(self, window_id: int, error_rate: float, alarm: bool) -> Alert | None:
        if not alarm or self._in_cooldown("performance_drift", window_id):
            return None
        created_at = datetime.now(UTC).isoformat(timespec="seconds")
        # A confirmed performance regression is always critical: the model is
        # measurably wrong on live labels, not merely seeing new inputs.
        return self._fire(
            Alert(
                window_id=window_id,
                type="performance_drift",
                severity="critical",
                features=(),
                details={"error_rate": error_rate},
                created_at=created_at,
            )
        )
