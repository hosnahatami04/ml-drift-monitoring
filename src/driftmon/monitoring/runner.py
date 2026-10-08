"""Glue between the store, the shared Monitor and the alert manager.

The runner is invoked after writes (predictions or labels): it processes any
window that just became complete and feeds fully-labelled windows to the
performance detector, in window order.
"""

import logging

import numpy as np
import pandas as pd

from driftmon.data import schema
from driftmon.monitoring.alerts import Alert, AlertManager
from driftmon.monitoring.monitor import Monitor, WindowReport
from driftmon.service.store import Store

logger = logging.getLogger(__name__)


class MonitorRunner:
    def __init__(self, store: Store, monitor: Monitor, alert_manager: AlertManager) -> None:
        self.store = store
        self.monitor = monitor
        self.alert_manager = alert_manager

    def process_complete_windows(self) -> list[WindowReport]:
        reports: list[WindowReport] = []
        for window_id in self.store.unprocessed_complete_windows():
            predictions = self.store.window_predictions(window_id)
            features = pd.DataFrame([p.features for p in predictions])[list(schema.MODEL_FEATURES)]
            probabilities = np.asarray([p.probability for p in predictions], dtype=float)
            report = self.monitor.process_window(window_id, features, probabilities)
            self.store.save_window_report(report.to_dict())
            alerts = self.alert_manager.evaluate_window(report)
            for alert in alerts:
                logger.info(
                    "alert %s (%s) on window %d: %s",
                    alert.type,
                    alert.severity,
                    window_id,
                    list(alert.features),
                )
            reports.append(report)
        return reports

    def process_labelled_windows(self) -> list[Alert]:
        alerts: list[Alert] = []
        for window_id in self.store.windows_awaiting_performance():
            predictions = self.store.window_predictions(window_id)
            probabilities = np.asarray([p.probability for p in predictions], dtype=float)
            labels = np.asarray([p.label for p in predictions], dtype=np.int64)
            error_rate = Monitor.error_rate(probabilities, labels)
            alarm = self.monitor.observe_error_rate(error_rate)
            self.store.mark_performance_checked(window_id, error_rate)
            alert = self.alert_manager.evaluate_performance(window_id, error_rate, alarm)
            if alert is not None:
                logger.info(
                    "alert performance_drift on window %d (error rate %.4f)",
                    window_id,
                    error_rate,
                )
                alerts.append(alert)
        return alerts
