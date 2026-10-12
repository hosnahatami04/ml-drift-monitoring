"""Glue between the store, the shared Monitor and the alert manager.

The runner processes any window that has become complete and feeds
fully-labelled windows to the performance detector, in window order. It runs
either inline after API writes or in the standalone `driftmon monitor`
worker, which polls the shared store.
"""

import logging
import threading

import numpy as np
import pandas as pd
from opentelemetry import trace

from driftmon.data import schema
from driftmon.monitoring.alerts import Alert, AlertManager
from driftmon.monitoring.monitor import Monitor, WindowReport
from driftmon.observability.metrics import Metrics
from driftmon.observability.tracing import get_tracer
from driftmon.service.store import Store

logger = logging.getLogger(__name__)


class MonitorRunner:
    def __init__(
        self,
        store: Store,
        monitor: Monitor,
        alert_manager: AlertManager,
        metrics: Metrics | None = None,
        tracer: trace.Tracer | None = None,
    ) -> None:
        self.store = store
        self.monitor = monitor
        self.alert_manager = alert_manager
        self.metrics = metrics
        self.tracer = tracer or get_tracer()
        # Inline mode calls the runner from concurrent request threads; the
        # monitor and alert state must see windows strictly in order.
        self._lock = threading.Lock()

    def restore_state(self) -> None:
        """Rebuild Page-Hinkley and alert cooldowns from what the store
        already holds, so a restarted worker carries on where it stopped."""
        with self._lock:
            self.monitor.reset_performance()
            rates = self.store.checked_error_rates()
            for rate in rates:
                self.monitor.observe_error_rate(rate)
            self.alert_manager.restore(self.store.last_alert_windows())
        logger.info("restored monitor state from %d labelled windows", len(rates))

    def run_once(self) -> tuple[list[WindowReport], list[Alert]]:
        return self.process_complete_windows(), self.process_labelled_windows()

    def process_complete_windows(self) -> list[WindowReport]:
        reports: list[WindowReport] = []
        with self._lock:
            for window_id in self.store.unprocessed_complete_windows():
                try:
                    with self.tracer.start_as_current_span("monitor.process_window") as span:
                        span.set_attribute("driftmon.window_id", window_id)
                        report = self._process_window(window_id)
                except Exception:
                    # Later windows wait rather than overtake this one; it is
                    # retried on the next call and the error is logged each time.
                    logger.exception("processing window %d failed", window_id)
                    break
                if report is not None:
                    reports.append(report)
        return reports

    def _process_window(self, window_id: int) -> WindowReport | None:
        predictions = self.store.window_predictions(window_id)
        features = pd.DataFrame([p.features for p in predictions])[list(schema.MODEL_FEATURES)]
        probabilities = np.asarray([p.probability for p in predictions], dtype=float)
        report = self.monitor.process_window(window_id, features, probabilities)
        if not self.store.save_window_report(report.to_dict()):
            logger.info("window %d was already processed by another process", window_id)
            return None
        if self.metrics is not None:
            self.metrics.record_window(report)
        for alert in self.alert_manager.evaluate_window(report):
            logger.info(
                "alert %s (%s) on window %d: %s",
                alert.type,
                alert.severity,
                window_id,
                list(alert.features),
            )
        return report

    def process_labelled_windows(self) -> list[Alert]:
        alerts: list[Alert] = []
        with self._lock:
            for window_id in self.store.windows_awaiting_performance():
                try:
                    alert = self._check_performance(window_id)
                except Exception:
                    logger.exception("performance check of window %d failed", window_id)
                    break
                if alert is not None:
                    alerts.append(alert)
        return alerts

    def _check_performance(self, window_id: int) -> Alert | None:
        with self.tracer.start_as_current_span("monitor.process_labels") as span:
            span.set_attribute("driftmon.window_id", window_id)
            predictions = self.store.window_predictions(window_id)
            probabilities = np.asarray([p.probability for p in predictions], dtype=float)
            labels = np.asarray([p.label for p in predictions], dtype=np.int64)
            error_rate = Monitor.error_rate(probabilities, labels)
            # Claim first: if another process already checked this window, its
            # error rate must not reach this process's Page-Hinkley a second time.
            if not self.store.mark_performance_checked(window_id, error_rate):
                return None
            alarm = self.monitor.observe_error_rate(error_rate)
            span.set_attribute("driftmon.error_rate", error_rate)
            span.set_attribute("driftmon.performance_alarm", alarm)
        if self.metrics is not None:
            self.metrics.record_error_rate(error_rate)
        alert = self.alert_manager.evaluate_performance(window_id, error_rate, alarm)
        if alert is not None:
            logger.info(
                "alert performance_drift on window %d (error rate %.4f)", window_id, error_rate
            )
        return alert


def poll(runner: MonitorRunner, interval_seconds: float, stop: threading.Event) -> None:
    """The worker loop: process whatever is ready, then sleep until the next
    tick or until asked to stop."""
    while not stop.is_set():
        try:
            runner.run_once()
        except Exception:
            # A failed tick (the store briefly locked, a disk error) must not
            # take alerting down with it; the next tick tries again.
            logger.exception("monitor tick failed")
        stop.wait(interval_seconds)
