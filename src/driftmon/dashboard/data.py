"""Data loading for the dashboard, kept free of any UI code.

Every query opens the store read-only: the dashboard must never be able to
write to the database the service and monitor share.
"""

import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import roc_auc_score

from driftmon.evaluation.report import ResultsDocument, validate_results

STATISTIC_COLUMNS = [
    "window_id",
    "feature",
    "method",
    "statistic",
    "p_value",
    "effect_size",
    "flagged",
]
ALERT_COLUMNS = ["window_id", "type", "severity", "features", "created_at"]
PERFORMANCE_COLUMNS = ["window_id", "error_rate", "auc"]


@dataclass(frozen=True)
class Status:
    predictions: int
    processed_windows: int
    last_window: int | None
    last_processed_at: str | None
    alerts: int
    latest_alert: dict[str, Any] | None


def _connect(db_path: Path) -> sqlite3.Connection | None:
    if not db_path.is_file():
        return None
    return sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=10.0)


def _query(db_path: Path, sql: str, params: tuple[Any, ...] = ()) -> list[tuple[Any, ...]]:
    connection = _connect(db_path)
    if connection is None:
        return []
    with closing(connection):
        try:
            return connection.execute(sql, params).fetchall()
        except sqlite3.OperationalError as exc:
            # A store the service has not migrated yet has no tables.
            if "no such table" in str(exc):
                return []
            raise


def load_status(db_path: Path) -> Status:
    counts = _query(
        db_path,
        "SELECT (SELECT COUNT(*) FROM predictions), (SELECT COUNT(*) FROM windows), "
        "(SELECT MAX(window_id) FROM windows), (SELECT MAX(processed_at) FROM windows), "
        "(SELECT COUNT(*) FROM alerts)",
    )
    latest = load_alerts(db_path, limit=1)
    if not counts:
        return Status(0, 0, None, None, 0, None)
    predictions, windows, last_window, last_at, alerts = counts[0]
    return Status(
        predictions=int(predictions),
        processed_windows=int(windows),
        last_window=None if last_window is None else int(last_window),
        last_processed_at=last_at,
        alerts=int(alerts),
        latest_alert=None if latest.empty else latest.iloc[0].to_dict(),
    )


def load_feature_statistics(db_path: Path) -> pd.DataFrame:
    rows = _query(db_path, "SELECT report FROM windows ORDER BY window_id")
    records = []
    for (report_json,) in rows:
        report = json.loads(report_json)
        for result in report["results"]:
            records.append({"window_id": report["window_id"], **result})
    return pd.DataFrame(records, columns=STATISTIC_COLUMNS)


def load_alerts(db_path: Path, limit: int = 1000) -> pd.DataFrame:
    rows = _query(
        db_path,
        "SELECT window_id, type, severity, features, created_at FROM alerts "
        "ORDER BY id DESC LIMIT ?",
        (limit,),
    )
    records = [
        {
            "window_id": int(window_id),
            "type": alert_type,
            "severity": severity,
            "features": ", ".join(json.loads(features)),
            "created_at": created_at,
        }
        for window_id, alert_type, severity, features, created_at in rows
    ]
    return pd.DataFrame(records, columns=ALERT_COLUMNS)


def load_performance(db_path: Path) -> pd.DataFrame:
    """Error rate and AUC of every window whose labels have all arrived."""
    rows = _query(
        db_path,
        "SELECT w.window_id, w.error_rate, p.probability, l.label FROM windows w "
        "JOIN predictions p ON p.window_id = w.window_id "
        "JOIN labels l ON l.request_id = p.request_id "
        "WHERE w.performance_checked = 1 ORDER BY w.window_id",
    )
    if not rows:
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)
    frame = pd.DataFrame(rows, columns=["window_id", "error_rate", "probability", "label"])
    records = []
    for window_id, group in frame.groupby("window_id", sort=True):
        labels = group["label"].to_numpy()
        auc = (
            float(roc_auc_score(labels, group["probability"].to_numpy()))
            if labels.min() != labels.max()
            else float("nan")
        )
        records.append(
            {
                "window_id": int(window_id),
                "error_rate": float(group["error_rate"].iloc[0]),
                "auc": auc,
            }
        )
    return pd.DataFrame(records, columns=PERFORMANCE_COLUMNS)


def load_evaluation(results_path: Path) -> ResultsDocument | None:
    if not results_path.is_file():
        return None
    return validate_results(json.loads(results_path.read_text(encoding="utf-8")))


def kpi_table(document: ResultsDocument) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "KPI": row.name,
                "scope": row.scope,
                "value": row.estimate.value,
                "ci_low": row.estimate.ci_low,
                "ci_high": row.estimate.ci_high,
                "target": row.target,
                "comparator": row.comparator,
                "met": row.met,
            }
            for row in document.kpis
        ]
    )


def cell_table(document: ResultsDocument) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": cell.scenario,
                "severity": cell.severity,
                "detection_rate": cell.detection_rate.value,
                "median_ttd": cell.median_ttd.value,
                "alert_precision": cell.alert_precision.value,
                "harmful_share": cell.harmful_share.value,
            }
            for cell in document.cells
        ]
    )
