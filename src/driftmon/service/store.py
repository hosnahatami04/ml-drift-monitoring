"""SQLite-backed prediction, label, window and alert store.

WAL mode, one connection per process guarded by a lock, and a tiny versioned
migration: enough for this scale, no server to run. The API and the monitor
worker may share a store, so every step that hands out work is a claim that
only one process can win.
"""

import json
import logging
import sqlite3
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from driftmon.config import ConfigError
from driftmon.monitoring.alerts import Alert
from driftmon.monitoring.windowing import window_id_for

logger = logging.getLogger(__name__)

CONNECT_TIMEOUT = 30.0

MIGRATIONS: tuple[str, ...] = (
    """
    CREATE TABLE IF NOT EXISTS predictions (
        seq INTEGER PRIMARY KEY AUTOINCREMENT,
        request_id TEXT NOT NULL UNIQUE,
        created_at TEXT NOT NULL,
        features TEXT NOT NULL,
        probability REAL NOT NULL,
        window_id INTEGER NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_predictions_window ON predictions(window_id);
    CREATE INDEX IF NOT EXISTS idx_predictions_created ON predictions(created_at);

    CREATE TABLE IF NOT EXISTS labels (
        request_id TEXT PRIMARY KEY REFERENCES predictions(request_id),
        label INTEGER NOT NULL,
        created_at TEXT NOT NULL
    );

    CREATE TABLE IF NOT EXISTS windows (
        window_id INTEGER PRIMARY KEY,
        n_rows INTEGER NOT NULL,
        processed_at TEXT NOT NULL,
        report TEXT NOT NULL,
        feature_drift INTEGER NOT NULL,
        prediction_drift INTEGER NOT NULL,
        error_rate REAL,
        performance_checked INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_windows_processed ON windows(processed_at);

    CREATE TABLE IF NOT EXISTS alerts (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        window_id INTEGER NOT NULL,
        type TEXT NOT NULL,
        severity TEXT NOT NULL,
        features TEXT NOT NULL,
        details TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_alerts_created ON alerts(created_at);
    CREATE INDEX IF NOT EXISTS idx_alerts_window ON alerts(window_id);
    """,
    """
    CREATE TABLE IF NOT EXISTS meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    ALTER TABLE windows ADD COLUMN performance_order INTEGER;
    """,
)


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


@dataclass(frozen=True)
class StoredPrediction:
    seq: int
    request_id: str
    features: dict[str, Any]
    probability: float
    window_id: int
    label: int | None


class Store:
    """One persistent WAL connection guarded by a lock: opening a connection
    per operation costs more than the queries themselves at this scale."""

    def __init__(self, path: Path, window_size: int) -> None:
        self.path = path
        self.window_size = window_size
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(path, timeout=CONNECT_TIMEOUT, check_same_thread=False)
        self._enable_wal()
        self._connection.execute("PRAGMA foreign_keys=ON")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._migrate()
        self._check_window_size()

    def _enable_wal(self) -> None:
        # Switching a new database to WAL needs an exclusive lock, and SQLite
        # fails that at once instead of waiting on the busy timeout, so two
        # processes creating the store together retry here.
        deadline = time.monotonic() + CONNECT_TIMEOUT
        while True:
            try:
                self._connection.execute("PRAGMA journal_mode=WAL")
                return
            except sqlite3.OperationalError as exc:
                if "locked" not in str(exc) or time.monotonic() > deadline:
                    raise
                time.sleep(0.05)

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    @contextmanager
    def _session(self) -> Iterator[sqlite3.Connection]:
        with self._lock, self._connection:
            yield self._connection

    def _migrate(self) -> None:
        # The write lock is taken before the version is read, so when the API
        # and the worker start together the second one sees the first one's
        # migration instead of repeating it.
        with self._lock:
            connection = self._connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                version = connection.execute("PRAGMA user_version").fetchone()[0]
                for target, script in enumerate(MIGRATIONS, start=1):
                    if version < target:
                        for statement in filter(str.strip, script.split(";")):
                            connection.execute(statement)
                        connection.execute(f"PRAGMA user_version = {target}")
                        logger.info("migrated store %s to version %d", self.path, target)
                connection.execute("COMMIT")
            except BaseException:
                connection.execute("ROLLBACK")
                raise

    def _check_window_size(self) -> None:
        """Window ids are assigned at write time, so every process using the
        store must agree on the window size it was created with."""
        with self._session() as connection:
            connection.execute(
                "INSERT OR IGNORE INTO meta (key, value) VALUES ('window_size', ?)",
                (str(self.window_size),),
            )
            stored = int(
                connection.execute("SELECT value FROM meta WHERE key = 'window_size'").fetchone()[0]
            )
        if stored != self.window_size:
            raise ConfigError(
                f"store {self.path} was created with window_size {stored}, but the "
                f"configured window_size is {self.window_size}"
            )

    def insert_prediction(
        self, request_id: str, features: dict[str, Any], probability: float
    ) -> tuple[int, int]:
        """Returns the window id the prediction landed in and that window's
        row count so far."""
        with self._session() as connection:
            cursor = connection.execute(
                "INSERT INTO predictions "
                "(request_id, created_at, features, probability, window_id) "
                "VALUES (?, ?, ?, ?, ?)",
                (request_id, _now(), json.dumps(features), probability, -1),
            )
            seq = cursor.lastrowid
            assert seq is not None
            window_id = window_id_for(seq, self.window_size)
            connection.execute(
                "UPDATE predictions SET window_id = ? WHERE seq = ?", (window_id, seq)
            )
            count = connection.execute(
                "SELECT COUNT(*) FROM predictions WHERE window_id = ?", (window_id,)
            ).fetchone()[0]
            return window_id, int(count)

    def attach_label(self, request_id: str, label: int) -> bool | None:
        """True for a new label, False for a repeat, None for an unknown
        prediction. The first label wins: once a window's error rate has
        been fed to the performance detector it must not change."""
        with self._session() as connection:
            exists = connection.execute(
                "SELECT 1 FROM predictions WHERE request_id = ?", (request_id,)
            ).fetchone()
            if exists is None:
                return None
            cursor = connection.execute(
                "INSERT OR IGNORE INTO labels (request_id, label, created_at) VALUES (?, ?, ?)",
                (request_id, label, _now()),
            )
            return cursor.rowcount == 1

    def window_predictions(self, window_id: int) -> list[StoredPrediction]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT p.seq, p.request_id, p.features, p.probability, p.window_id, l.label "
                "FROM predictions p LEFT JOIN labels l ON l.request_id = p.request_id "
                "WHERE p.window_id = ? ORDER BY p.seq",
                (window_id,),
            ).fetchall()
        return [
            StoredPrediction(
                seq=row[0],
                request_id=row[1],
                features=json.loads(row[2]),
                probability=row[3],
                window_id=row[4],
                label=row[5],
            )
            for row in rows
        ]

    def unprocessed_complete_windows(self) -> list[int]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT window_id FROM predictions "
                "WHERE window_id NOT IN (SELECT window_id FROM windows) "
                "GROUP BY window_id HAVING COUNT(*) >= ? ORDER BY window_id",
                (self.window_size,),
            ).fetchall()
        return [int(row[0]) for row in rows]

    def save_window_report(self, report_payload: dict[str, Any]) -> bool:
        """Claims the window; False when another process already saved it."""
        with self._session() as connection:
            cursor = connection.execute(
                "INSERT INTO windows "
                "(window_id, n_rows, processed_at, report, feature_drift, prediction_drift) "
                "VALUES (?, ?, ?, ?, ?, ?) ON CONFLICT(window_id) DO NOTHING",
                (
                    report_payload["window_id"],
                    report_payload["n_rows"],
                    _now(),
                    json.dumps(report_payload),
                    int(report_payload["feature_drift"]),
                    int(report_payload["prediction_drift"]),
                ),
            )
            return cursor.rowcount == 1

    _FULLY_LABELLED = (
        "w.n_rows = (SELECT COUNT(*) FROM predictions p "
        "JOIN labels l ON l.request_id = p.request_id WHERE p.window_id = w.window_id)"
    )

    def windows_awaiting_performance(self) -> list[int]:
        """Fully labelled windows not yet fed to the performance detector, in
        window order. A window whose labels never all arrive is skipped
        rather than blocking the ones after it."""
        with self._session() as connection:
            rows = connection.execute(
                "SELECT w.window_id FROM windows w "
                f"WHERE w.performance_checked = 0 AND {self._FULLY_LABELLED} "
                "ORDER BY w.window_id",
            ).fetchall()
        return [int(row[0]) for row in rows]

    def mark_performance_checked(self, window_id: int, error_rate: float) -> bool:
        """Claims the window for the performance detector and records the
        order it was consumed in; False when another process already did."""
        with self._session() as connection:
            cursor = connection.execute(
                "UPDATE windows SET performance_checked = 1, error_rate = ?, "
                "performance_order = "
                "(SELECT COALESCE(MAX(performance_order), 0) + 1 FROM windows) "
                "WHERE window_id = ? AND performance_checked = 0",
                (error_rate, window_id),
            )
            return cursor.rowcount == 1

    def performance_checked_count(self) -> int:
        with self._session() as connection:
            row = connection.execute(
                "SELECT COUNT(*) FROM windows WHERE performance_checked = 1"
            ).fetchone()
        return int(row[0])

    def fully_labelled_count(self) -> int:
        with self._session() as connection:
            row = connection.execute(
                f"SELECT COUNT(*) FROM windows w WHERE {self._FULLY_LABELLED}"
            ).fetchone()
        return int(row[0])

    def checked_error_rates(self) -> list[float]:
        """Error rates already fed to the performance detector, in the order
        it consumed them: replaying them rebuilds its state after a restart.
        Labels can complete out of window order, so window id is not enough."""
        with self._session() as connection:
            rows = connection.execute(
                "SELECT error_rate FROM windows WHERE performance_checked = 1 "
                "ORDER BY performance_order, window_id"
            ).fetchall()
        return [float(row[0]) for row in rows]

    def last_alert_windows(self) -> dict[str, int]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT type, MAX(window_id) FROM alerts GROUP BY type"
            ).fetchall()
        return {str(row[0]): int(row[1]) for row in rows}

    def insert_alert(self, alert: Alert) -> None:
        with self._session() as connection:
            connection.execute(
                "INSERT INTO alerts (window_id, type, severity, features, details, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    alert.window_id,
                    alert.type,
                    alert.severity,
                    json.dumps(list(alert.features)),
                    json.dumps(alert.details),
                    alert.created_at,
                ),
            )

    def list_alerts(self, limit: int) -> list[dict[str, Any]]:
        with self._session() as connection:
            rows = connection.execute(
                "SELECT window_id, type, severity, features, details, created_at "
                "FROM alerts ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {
                "window_id": row[0],
                "type": row[1],
                "severity": row[2],
                "features": json.loads(row[3]),
                "details": json.loads(row[4]),
                "created_at": row[5],
            }
            for row in rows
        ]

    def prediction_count(self) -> int:
        with self._session() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])

    def processed_window_count(self) -> int:
        with self._session() as connection:
            return int(connection.execute("SELECT COUNT(*) FROM windows").fetchone()[0])


class SqliteAlertSink:
    def __init__(self, store: Store) -> None:
        self.store = store

    def emit(self, alert: Alert) -> None:
        self.store.insert_alert(alert)
