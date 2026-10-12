import json
import logging
import urllib.error
from pathlib import Path

import pytest

from driftmon.config import AlertSettings
from driftmon.detectors.base import TestResult as DetectorResult
from driftmon.monitoring.alerts import AlertManager, JsonlSink, WebhookSink
from driftmon.monitoring.monitor import WindowReport


def _report(
    window_id: int,
    flagged: tuple[str, ...] = ("age",),
    feature_drift: bool = True,
    prediction_drift: bool = False,
) -> WindowReport:
    results = tuple(
        DetectorResult(
            feature=f, method="ks", statistic=0.4, p_value=1e-6, effect_size=0.4, flagged=True
        )
        for f in flagged
    )
    if prediction_drift:
        results += (
            DetectorResult(
                feature="__prediction__",
                method="prediction_ks",
                statistic=0.3,
                p_value=1e-5,
                effect_size=0.3,
                flagged=True,
            ),
        )
    return WindowReport(
        window_id=window_id,
        n_rows=1000,
        results=results,
        flagged_features=flagged if feature_drift else (),
        feature_drift=feature_drift,
        prediction_drift=prediction_drift,
    )


def test_feature_alert_carries_statistics() -> None:
    manager = AlertManager(AlertSettings())

    alerts = manager.evaluate_window(_report(0))

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.type == "feature_drift"
    assert alert.severity == "warning"
    assert alert.features == ("age",)
    assert "age" in alert.details["statistics"]


def test_many_features_escalate_to_critical() -> None:
    manager = AlertManager(AlertSettings(critical_feature_count=3))

    alerts = manager.evaluate_window(_report(0, flagged=("age", "hours-per-week", "workclass")))

    assert alerts[0].severity == "critical"


def test_cooldown_suppresses_repeats() -> None:
    manager = AlertManager(AlertSettings(cooldown_windows=3))

    first = manager.evaluate_window(_report(10))
    suppressed = [manager.evaluate_window(_report(w)) for w in (11, 12, 13)]
    after = manager.evaluate_window(_report(14))

    assert len(first) == 1
    assert all(batch == [] for batch in suppressed)
    assert len(after) == 1


def test_cooldown_is_per_alert_type() -> None:
    manager = AlertManager(AlertSettings(cooldown_windows=3))

    manager.evaluate_window(_report(10))
    mixed = manager.evaluate_window(_report(11, feature_drift=True, prediction_drift=True))

    assert [a.type for a in mixed] == ["prediction_drift"]


def test_performance_alert_is_critical_and_respects_cooldown() -> None:
    manager = AlertManager(AlertSettings(cooldown_windows=2))

    first = manager.evaluate_performance(5, error_rate=0.4, alarm=True)
    suppressed = manager.evaluate_performance(6, error_rate=0.5, alarm=True)
    quiet = manager.evaluate_performance(9, error_rate=0.1, alarm=False)
    after = manager.evaluate_performance(8, error_rate=0.5, alarm=True)

    assert first is not None and first.severity == "critical"
    assert suppressed is None
    assert quiet is None
    assert after is not None


def test_jsonl_sink_appends_alerts(tmp_path: Path) -> None:
    sink = JsonlSink(tmp_path / "alerts.jsonl")
    manager = AlertManager(AlertSettings(), sinks=[sink])

    manager.evaluate_window(_report(0))
    manager.evaluate_performance(10, error_rate=0.4, alarm=True)

    lines = (tmp_path / "alerts.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert payload["type"] == "feature_drift"


def test_webhook_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def failing_urlopen(request: object, timeout: float = 0) -> None:
        raise urllib.error.URLError("connection refused")

    monkeypatch.setattr("urllib.request.urlopen", failing_urlopen)
    sink = WebhookSink("http://alerts.invalid/hook")
    manager = AlertManager(AlertSettings(), sinks=[sink])

    with caplog.at_level(logging.WARNING):
        alerts = manager.evaluate_window(_report(0))

    assert len(alerts) == 1
    assert any("webhook" in record.message for record in caplog.records)


class _BrokenSink:
    def emit(self, alert: object) -> None:
        raise RuntimeError("sink is down")


class _RecordingSink:
    def __init__(self) -> None:
        self.received: list[object] = []

    def emit(self, alert: object) -> None:
        self.received.append(alert)


def test_a_failing_sink_does_not_stop_the_others(caplog: pytest.LogCaptureFixture) -> None:
    recorder = _RecordingSink()
    manager = AlertManager(AlertSettings(), sinks=[_BrokenSink(), recorder])

    with caplog.at_level(logging.ERROR):
        alerts = manager.evaluate_window(_report(0))

    assert len(alerts) == 1
    assert recorder.received == alerts
    assert "sink is down" in caplog.text


def test_webhook_with_a_malformed_url_is_logged_not_raised(
    caplog: pytest.LogCaptureFixture,
) -> None:
    manager = AlertManager(AlertSettings(), sinks=[WebhookSink("not a url")])

    with caplog.at_level(logging.WARNING):
        manager.evaluate_window(_report(0))

    assert "webhook delivery failed" in caplog.text
