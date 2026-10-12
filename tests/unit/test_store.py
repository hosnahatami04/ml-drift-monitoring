import sqlite3
import threading
from pathlib import Path

import pytest

from driftmon.config import ConfigError
from driftmon.monitoring.alerts import Alert
from driftmon.monitoring.windowing import is_window_complete, window_id_for
from driftmon.service.store import Store


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return Store(tmp_path / "test.db", window_size=5)


class TestWindowing:
    def test_window_ids_follow_arrival_order(self) -> None:
        assert window_id_for(1, 5) == 0
        assert window_id_for(5, 5) == 0
        assert window_id_for(6, 5) == 1
        assert window_id_for(11, 5) == 2

    def test_invalid_inputs_raise(self) -> None:
        with pytest.raises(ValueError):
            window_id_for(0, 5)
        with pytest.raises(ValueError):
            window_id_for(1, 0)

    def test_completion_rule(self) -> None:
        assert not is_window_complete(4, 5)
        assert is_window_complete(5, 5)


def test_prediction_roundtrip(store: Store) -> None:
    window_id, row_count = store.insert_prediction("req-1", {"age": 40}, 0.7)

    assert window_id == 0
    assert row_count == 1
    rows = store.window_predictions(0)
    assert len(rows) == 1
    assert rows[0].request_id == "req-1"
    assert rows[0].features == {"age": 40}
    assert rows[0].probability == 0.7
    assert rows[0].label is None


def test_label_attachment(store: Store) -> None:
    store.insert_prediction("req-1", {"age": 40}, 0.7)

    assert store.attach_label("req-1", 1) is True
    assert store.attach_label("missing", 0) is None
    # A repeat never overwrites: the first label may already be in an error rate.
    assert store.attach_label("req-1", 0) is False
    assert store.window_predictions(0)[0].label == 1


def test_complete_windows_are_reported_once(store: Store) -> None:
    for i in range(5):
        store.insert_prediction(f"req-{i}", {"age": 40}, 0.5)

    assert store.unprocessed_complete_windows() == [0]
    store.save_window_report(
        {
            "window_id": 0,
            "n_rows": 5,
            "results": [],
            "flagged_features": [],
            "feature_drift": False,
            "prediction_drift": False,
        }
    )
    assert store.unprocessed_complete_windows() == []
    assert store.processed_window_count() == 1


def test_performance_queue_requires_all_labels(store: Store) -> None:
    for i in range(5):
        store.insert_prediction(f"req-{i}", {"age": 40}, 0.5)
    store.save_window_report(
        {
            "window_id": 0,
            "n_rows": 5,
            "results": [],
            "flagged_features": [],
            "feature_drift": False,
            "prediction_drift": False,
        }
    )
    for i in range(4):
        store.attach_label(f"req-{i}", 1)
    assert store.windows_awaiting_performance() == []

    store.attach_label("req-4", 0)
    assert store.windows_awaiting_performance() == [0]

    store.mark_performance_checked(0, error_rate=0.2)
    assert store.windows_awaiting_performance() == []


def test_concurrent_writes_in_wal_mode(tmp_path: Path) -> None:
    store = Store(tmp_path / "concurrent.db", window_size=1000)
    errors: list[Exception] = []

    def writer(offset: int) -> None:
        try:
            for i in range(50):
                store.insert_prediction(f"req-{offset}-{i}", {"age": 40}, 0.5)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    assert store.prediction_count() == 200


def test_alert_roundtrip(store: Store) -> None:
    alert = Alert(
        window_id=3,
        type="feature_drift",
        severity="warning",
        features=("age",),
        details={"statistics": {}},
        created_at="2026-10-08T12:00:00+00:00",
    )

    store.insert_alert(alert)

    records = store.list_alerts(limit=100)
    assert len(records) == 1
    assert records[0]["window_id"] == 3
    assert records[0]["features"] == ["age"]


def test_store_reopens_without_remigrating(tmp_path: Path) -> None:
    first = Store(tmp_path / "db.sqlite", window_size=5)
    first.insert_prediction("req-1", {"age": 40}, 0.5)

    second = Store(tmp_path / "db.sqlite", window_size=5)

    assert second.prediction_count() == 1


def report(window_id: int, n_rows: int = 5) -> dict[str, object]:
    return {
        "window_id": window_id,
        "n_rows": n_rows,
        "results": [],
        "flagged_features": [],
        "feature_drift": False,
        "prediction_drift": False,
    }


def fill_windows(store: Store, n_windows: int) -> None:
    for i in range(n_windows * 5):
        store.insert_prediction(f"req-{i}", {"age": 40}, 0.5)
    for window_id in range(n_windows):
        store.save_window_report(report(window_id))


def test_a_window_can_be_claimed_only_once(store: Store) -> None:
    fill_windows(store, 1)

    assert store.save_window_report(report(0)) is False
    assert store.mark_performance_checked(0, 0.1) is True
    assert store.mark_performance_checked(0, 0.9) is False
    # The losing claim must not reset the winner's performance check.
    store.save_window_report(report(0))
    assert store.checked_error_rates() == [0.1]


def test_restore_order_follows_the_order_windows_were_checked(store: Store) -> None:
    fill_windows(store, 2)
    for i in range(5, 10):
        store.attach_label(f"req-{i}", 1)

    # Window 1 is fully labelled first, so it reaches the detector first.
    assert store.windows_awaiting_performance() == [1]
    store.mark_performance_checked(1, 0.3)
    for i in range(5):
        store.attach_label(f"req-{i}", 0)
    assert store.windows_awaiting_performance() == [0]
    store.mark_performance_checked(0, 0.1)

    assert store.checked_error_rates() == [0.3, 0.1]
    assert store.fully_labelled_count() == 2
    assert store.performance_checked_count() == 2


def test_an_incomplete_window_does_not_block_later_ones(store: Store) -> None:
    fill_windows(store, 2)
    for i in range(4):
        store.attach_label(f"req-{i}", 1)
    for i in range(5, 10):
        store.attach_label(f"req-{i}", 1)

    assert store.windows_awaiting_performance() == [1]


def test_store_refuses_a_different_window_size(tmp_path: Path) -> None:
    Store(tmp_path / "db.sqlite", window_size=5)

    with pytest.raises(ConfigError, match="window_size 5"):
        Store(tmp_path / "db.sqlite", window_size=10)


def test_concurrent_first_start_migrates_once(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    errors: list[BaseException] = []

    def open_store() -> None:
        try:
            Store(path, window_size=5)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=open_store) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert errors == []
    with sqlite3.connect(path) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 2
