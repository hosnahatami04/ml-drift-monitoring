import math
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from driftmon.dashboard import data
from driftmon.model.artifacts import ModelArtifact
from driftmon.service.api import build_state, create_app
from driftmon.service.client import InProcessClient, replay_windows

from .test_monitor_worker import PARAMS, stream
from .test_observability import REPO_ROOT, settings_for


@pytest.fixture(scope="module")
def db(tiny_artifact: ModelArtifact, tmp_path_factory: pytest.TempPathFactory) -> Path:
    tmp = tmp_path_factory.mktemp("dashboard")
    state = build_state(settings_for(tmp, inline=True), artifact=tiny_artifact)
    replay_windows(InProcessClient(TestClient(create_app(state))), stream())
    return state.store.path


def test_missing_store_gives_empty_views(tmp_path: Path) -> None:
    missing = tmp_path / "absent.db"

    status = data.load_status(missing)

    assert status == data.Status(0, 0, None, None, 0, None)
    assert data.load_feature_statistics(missing).empty
    assert data.load_alerts(missing).empty
    assert data.load_performance(missing).empty
    assert not missing.exists()


def test_unmigrated_store_gives_empty_views(tmp_path: Path) -> None:
    empty = tmp_path / "empty.db"
    sqlite3.connect(empty).close()

    assert data.load_status(empty).predictions == 0
    assert data.load_alerts(empty).empty


def test_status_reflects_the_store(db: Path) -> None:
    status = data.load_status(db)

    assert status.predictions == PARAMS.n_windows * PARAMS.window_size
    assert status.processed_windows == PARAMS.n_windows
    assert status.last_window == PARAMS.n_windows - 1
    assert status.alerts > 0
    assert status.latest_alert is not None
    assert status.latest_alert["type"] in {"feature_drift", "prediction_drift", "performance_drift"}


def test_feature_statistics_have_one_row_per_test_and_window(db: Path) -> None:
    statistics = data.load_feature_statistics(db)

    assert list(statistics.columns) == data.STATISTIC_COLUMNS
    # 19 tests per window (see test_monitor.py).
    assert len(statistics) == 19 * PARAMS.n_windows
    assert set(statistics["window_id"]) == set(range(PARAMS.n_windows))
    assert statistics["flagged"].dtype == bool


def test_alerts_are_newest_first_with_joined_features(db: Path) -> None:
    alerts = data.load_alerts(db)

    assert list(alerts.columns) == data.ALERT_COLUMNS
    with sqlite3.connect(db) as connection:
        newest = connection.execute(
            "SELECT window_id, type FROM alerts ORDER BY id DESC LIMIT 1"
        ).fetchone()
    assert (alerts.iloc[0]["window_id"], alerts.iloc[0]["type"]) == newest
    feature_alerts = alerts[alerts["type"] == "feature_drift"]
    assert not feature_alerts.empty
    assert any("age" in features for features in feature_alerts["features"])
    assert len(data.load_alerts(db, limit=1)) == 1


def test_performance_covers_fully_labelled_windows(db: Path) -> None:
    performance = data.load_performance(db)

    # Labels for the last L windows never arrive within the replay.
    assert list(performance["window_id"]) == list(range(PARAMS.n_windows - PARAMS.label_delay))
    assert performance["error_rate"].between(0, 1).all()
    assert all(0.0 <= a <= 1.0 for a in performance["auc"] if not math.isnan(a))


def test_dashboard_never_writes_to_the_store(db: Path) -> None:
    connection = data._connect(db)
    assert connection is not None

    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        connection.execute("DELETE FROM alerts")
    connection.close()


def test_evaluation_tables_come_from_results_json(tmp_path: Path) -> None:
    document = data.load_evaluation(REPO_ROOT / "reports" / "results.json")
    assert document is not None

    kpis = data.kpi_table(document)
    cells = data.cell_table(document)

    assert len(kpis) == len(document.kpis)
    assert {"KPI", "value", "ci_low", "ci_high", "target", "met"} <= set(kpis.columns)
    assert len(cells) == len(document.cells)
    assert data.load_evaluation(tmp_path / "missing.json") is None
